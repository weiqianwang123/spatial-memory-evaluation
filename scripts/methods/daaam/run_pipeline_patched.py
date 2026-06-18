"""Adapter wrapper for DAAAM's native run_pipeline.py.

This wrapper does not modify the external DAAAM repo. It patches DAAAM at
runtime to save the 3D position observations that the pipeline already keeps in
memory, and to feed Hydra's saved DSG back into DAAAM's native
SceneGraphService before shutdown. That lets DAAAM's own correction,
background-object, and feature-metadata logic produce the native graph used by
the eval adapter.
"""

from __future__ import annotations

import json
import queue
import sys
import types
import importlib.util
from pathlib import Path
from typing import Any


def _ensure_daaam_paths() -> Path:
    daaam_root = Path.cwd()
    for path in (daaam_root, daaam_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return daaam_root


def _install_spawn_import_shim() -> None:
    """Avoid DAAAM's eager pipeline import cycle in spawned worker processes.

    DAAAM's assignment/grounding workers import ``daaam.pipeline.models``. Under
    multiprocessing spawn, that normally executes ``daaam.pipeline.__init__``,
    which eagerly imports the orchestrator and can recurse back into partially
    initialized assignment modules. The workers only need the dataclasses from
    models.py, so load that submodule directly inside child interpreters.
    """

    daaam_root = _ensure_daaam_paths()
    if "daaam.pipeline.models" in sys.modules:
        return

    import daaam

    pipeline_dir = daaam_root / "src" / "daaam" / "pipeline"
    models_path = pipeline_dir / "models.py"
    if not models_path.exists():
        return

    pipeline_pkg = types.ModuleType("daaam.pipeline")
    pipeline_pkg.__file__ = str(pipeline_dir / "__init__.py")
    pipeline_pkg.__path__ = [str(pipeline_dir)]
    pipeline_pkg.__package__ = "daaam.pipeline"
    sys.modules["daaam.pipeline"] = pipeline_pkg
    setattr(daaam, "pipeline", pipeline_pkg)

    spec = importlib.util.spec_from_file_location("daaam.pipeline.models", models_path)
    if spec is None or spec.loader is None:
        return
    models_module = importlib.util.module_from_spec(spec)
    sys.modules["daaam.pipeline.models"] = models_module
    spec.loader.exec_module(models_module)
    setattr(pipeline_pkg, "models", models_module)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _patch_scene_graph_service() -> None:
    from daaam.scene_graph.services import SceneGraphService

    original_save_data = SceneGraphService.save_data

    def save_data_with_positions(self: Any, output_save_dir: Path) -> None:
        if getattr(self, "scene_graph_is_set", False):
            try:
                # Re-apply before every native save so late DAM corrections drained
                # during shutdown are reflected in dsg.json/background_objects.yaml.
                self.apply_corrections()
            except Exception as exc:
                self.logger.error(f"Failed to apply DAAAM corrections before save: {exc}")
        original_save_data(self, output_save_dir)
        output_dir = Path(output_save_dir)
        try:
            payload: dict[str, list[dict[str, Any]]] = {}
            with self.position_lock:
                positions_copy = {
                    int(semantic_id): list(observations)
                    for semantic_id, observations in self.object_3d_positions.items()
                }
            for semantic_id, observations in positions_copy.items():
                payload[str(semantic_id)] = [
                    {
                        "position_world": _jsonable(obs.position_world),
                        "position_camera": _jsonable(obs.position_camera),
                        "centroid_pixel": _jsonable(obs.centroid_pixel),
                        "median_depth": _jsonable(obs.median_depth),
                        "frame_id": _jsonable(obs.frame_id),
                        "timestamp": _jsonable(obs.timestamp),
                    }
                    for obs in observations
                ]
            (output_dir / "object_positions.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            (output_dir / "correction_stats.json").write_text(
                json.dumps(_jsonable(self.get_correction_stats()), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.error(f"Failed to save adapter object positions: {exc}")

    SceneGraphService.save_data = save_data_with_positions


def _empty_tracks() -> Any:
    import numpy as np

    return np.empty((0, 8), dtype=np.float32)


def _normalize_tracks(value: Any) -> Any:
    import numpy as np

    tracks = np.asarray(value)
    if tracks.size == 0:
        return _empty_tracks()
    if tracks.ndim == 1:
        return tracks.reshape(1, -1)
    return tracks


def _patch_tracking_empty_outputs() -> None:
    from daaam.tracking.services import BotSortAdapter, TrackingService

    if getattr(BotSortAdapter, "_spatial_memory_eval_patch", False):
        return

    original_adapter_update = BotSortAdapter.update
    original_service_update = TrackingService.update

    def adapter_update_with_normalized_shape(self: Any, detections: Any, frame: Any) -> Any:
        return _normalize_tracks(original_adapter_update(self, detections, frame))

    def service_update_with_empty_guard(self: Any, detections: Any, frame: Any) -> Any:
        import numpy as np

        detections_array = np.asarray(detections)
        if detections_array.size == 0:
            return _empty_tracks()
        if detections_array.ndim == 1:
            if detections_array.size < 6:
                return _empty_tracks()
            detections_array = detections_array.reshape(1, -1)
        try:
            return _normalize_tracks(original_service_update(self, detections_array, frame))
        except IndexError as exc:
            if "too many indices for array" not in str(exc):
                raise
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.warning(
                    "Adapter patch converted malformed empty BotSort output to an empty track set"
                )
            return _empty_tracks()

    BotSortAdapter.update = adapter_update_with_normalized_shape
    BotSortAdapter._spatial_memory_eval_patch = True
    TrackingService.update = service_update_with_empty_guard
    TrackingService._spatial_memory_eval_patch = True


def _snapshot_orchestrator_positions(orchestrator: Any) -> dict[int, list[dict[str, Any]]]:
    positions = getattr(orchestrator, "object_3d_positions", None)
    if not positions:
        return {}

    lock = getattr(orchestrator, "_state_lock", None)
    if lock is None:
        items = list(positions.items())
    else:
        with lock:
            items = list(positions.items())

    snapshot: dict[int, list[dict[str, Any]]] = {}
    for semantic_id, observations in items:
        rows: list[dict[str, Any]] = []
        for obs in observations:
            if isinstance(obs, dict):
                rows.append({str(key): _jsonable(value) for key, value in obs.items()})
            else:
                rows.append(
                    {
                        "position_world": _jsonable(getattr(obs, "position_world", None)),
                        "position_camera": _jsonable(getattr(obs, "position_camera", None)),
                        "centroid_pixel": _jsonable(getattr(obs, "centroid_pixel", None)),
                        "median_depth": _jsonable(getattr(obs, "median_depth", None)),
                        "frame_id": _jsonable(getattr(obs, "frame_id", None)),
                        "timestamp": _jsonable(getattr(obs, "timestamp", None)),
                    }
                )
        snapshot[int(semantic_id)] = rows
    return snapshot


def _drain_orchestrator_corrections(orchestrator: Any) -> int:
    correction_queue = getattr(orchestrator, "correction_queue", None)
    scene_graph_service = getattr(orchestrator, "scene_graph_service", None)
    if correction_queue is None or scene_graph_service is None:
        return 0

    drained = 0
    while True:
        try:
            correction = correction_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break

        try:
            enrich = getattr(orchestrator, "_enrich_correction_with_temporal_data", None)
            if enrich is not None:
                correction = enrich(correction)
            scene_graph_service.store_correction(correction)
            drained += 1
        except Exception as exc:
            logger = getattr(orchestrator, "logger", None)
            if logger is not None:
                logger.error(f"Failed to store drained correction in adapter patch: {exc}")
            break
    return drained


def _graph_counts(scene_graph: Any) -> dict[str, int | None]:
    try:
        return {
            "nodes": int(scene_graph.num_nodes()),
            "edges": int(scene_graph.num_edges()),
        }
    except Exception:
        return {"nodes": None, "edges": None}


def _save_corrected_native_dsg(runner: Any) -> None:
    import spark_dsg as sdsg

    output_dir = Path(runner.output_dir)
    hydra_dsg_path = output_dir / "hydra_output" / "backend" / "dsg.json"
    logger = getattr(runner, "logger", None)

    if not hydra_dsg_path.exists():
        if logger is not None:
            logger.warning(f"Adapter patch could not find Hydra DSG: {hydra_dsg_path}")
        return

    orchestrator = getattr(runner, "orchestrator", None)
    if orchestrator is None or not hasattr(orchestrator, "scene_graph_service"):
        if logger is not None:
            logger.warning("Adapter patch could not access DAAAM SceneGraphService")
        return

    service = orchestrator.scene_graph_service
    dsg = sdsg.DynamicSceneGraph.load(str(hydra_dsg_path))
    service.set_scene_graph(dsg)

    positions = _snapshot_orchestrator_positions(orchestrator)
    if positions:
        service.update_object_positions(positions)

    drained = _drain_orchestrator_corrections(orchestrator)
    service.apply_corrections()
    service.save_data(orchestrator.output_dir)

    corrected_dsg_path = Path(orchestrator.output_dir) / "dsg.json"
    status = {
        "status": "ok",
        "hydra_dsg_path": str(hydra_dsg_path),
        "corrected_dsg_path": str(corrected_dsg_path),
        "positions_semantic_ids": len(positions),
        "drained_corrections": drained,
        "scene_graph_set": bool(getattr(service, "scene_graph_is_set", False)),
        "correction_stats": _jsonable(service.get_correction_stats()),
        "graph_counts": _graph_counts(service.get_scene_graph()),
    }
    (Path(orchestrator.output_dir) / "adapter_corrected_dsg_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    if logger is not None:
        logger.info(f"Adapter patch saved corrected native DAAAM DSG to {corrected_dsg_path}")


def _patch_hydra_runner() -> None:
    from daaam.hydra.runner import HydraPipelineRunner

    if getattr(HydraPipelineRunner, "_spatial_memory_eval_patch", False):
        return

    original_save_results = HydraPipelineRunner._save_results

    def save_results_with_corrected_dsg(self: Any) -> None:
        original_save_results(self)
        try:
            _save_corrected_native_dsg(self)
        except Exception as exc:
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.error(f"Adapter patch failed to save corrected native DAAAM DSG: {exc}")
            else:
                raise

    HydraPipelineRunner._save_results = save_results_with_corrected_dsg
    HydraPipelineRunner._spatial_memory_eval_patch = True


def main() -> None:
    _ensure_daaam_paths()
    from scripts.run_pipeline import main as daaam_main

    # Importing DAAAM run_pipeline first lets DAAAM resolve its own internal
    # pipeline/scene_graph circular imports in the same order as the native CLI.
    _patch_scene_graph_service()
    _patch_tracking_empty_outputs()
    _patch_hydra_runner()
    daaam_main()


if __name__ == "__mp_main__":
    _install_spawn_import_shim()


if __name__ == "__main__":
    main()
