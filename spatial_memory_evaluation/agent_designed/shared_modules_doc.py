"""Render the shared perception stack into a doc the designer agent reads.

Resolves the live ``SharedModuleRegistry`` (formal profile) so the designer builds
its memory on the SAME detector / segmenter / CLIP / class-list as the hand-built
methods, plus the LOCAL qwen ollama stack for describing/embedding. Pure
string-rendering; no heavy imports.
"""

from __future__ import annotations

from typing import Any

# The runtime describer/captioner/embedder stack every method shares (local-only).
OLLAMA_ENDPOINT = "http://localhost:11434"
OLLAMA_VLM_MODEL = "qwen3.5:4b"
OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"


def _resolve_formal_modules() -> list[dict[str, Any]] | None:
    try:
        from spatial_memory_evaluation.shared_modules.registry import get_shared_module_registry

        reg = get_shared_module_registry()
        # DAAAM's formal profile resolves the full benchmark stack (SAM vit_h,
        # OpenCLIP ViT-H-14, FastSAM-x TRT, the shared scannet200 class list).
        settings = reg.method_settings("daaam", profile="formal")
        meta = settings.get("metadata", {})
        modules = meta.get("modules")
        return modules if isinstance(modules, list) else None
    except Exception:
        return None


def render_shared_modules_md() -> str:
    lines = [
        "# Shared Modules (the perception stack you must use)",
        "",
        "Build your memory on the SAME modules as the hand-built methods so the",
        "comparison is fair. You design the memory representation + query interface,",
        "NOT a stronger detector. All describing/captioning/embedding at build or",
        "query time must use the LOCAL qwen stack below (no Claude/Bedrock inside the",
        "memory; the coding agent that writes your code is development-time only).",
        "",
        "## Perception modules (formal profile, from the live registry)",
        "",
    ]
    modules = _resolve_formal_modules()
    if modules:
        for m in modules:
            name = m.get("name") or m.get("key")
            ckpt = m.get("checkpoint") or m.get("class_list") or m.get("model_name") or "(resolved by repo)"
            role = m.get("role") or m.get("kind") or ""
            lines.append(f"- **{name}** ({m.get('key')}): `{ckpt}`")
            if role:
                lines.append(f"  - {role}")
    else:
        lines += [
            "- (registry unavailable in this environment; resolve via",
            "  `spatial_memory_evaluation.shared_modules.registry.get_shared_module_registry()`",
            "  `.method_settings('daaam', profile='formal')`)",
            "- Shared OV class list: `spatial_memory_evaluation/assets/class_lists/scannet200.txt`",
            "- SAM vit_h, OpenCLIP ViT-H-14, YOLO-World-L, FastSAM-x TRT under",
            "  `/data/mondo-training-dataset/semantic_mapping/modules/`",
        ]
    lines += [
        "",
        "## Local describer / captioner / embedder (ollama)",
        "",
        f"- Endpoint: `{OLLAMA_ENDPOINT}`",
        f"- Vision/caption model: `{OLLAMA_VLM_MODEL}` via `POST /api/chat` "
        "(messages with base64 JPEG `images`).",
        f"- Text embedding model: `{OLLAMA_EMBED_MODEL}` via `POST /api/embed` "
        "(`{{\"model\": ..., \"input\": text}}`, 1024-d).",
        "",
        "Example (caption):",
        "```python",
        "import base64, requests",
        "img_b64 = base64.b64encode(open('rgb/000001.jpg','rb').read()).decode()",
        "r = requests.post('http://localhost:11434/api/chat', json={",
        f"    'model': '{OLLAMA_VLM_MODEL}', 'stream': False,",
        "    'messages': [{'role': 'user', 'content': 'Describe the objects.',",
        "                  'images': [img_b64]}]})",
        "caption = r.json()['message']['content']",
        "```",
        "",
        "Example (embed):",
        "```python",
        "r = requests.post('http://localhost:11434/api/embed', json={",
        f"    'model': '{OLLAMA_EMBED_MODEL}', 'input': caption}})",
        "vec = r.json()['embeddings'][0]",
        "```",
        "",
        "## Layout each DEV scene provides (under `dev_scenes/<scene>/`)",
        "",
        "- `rgb/<idx>.jpg` color, `depth/<idx>.png` uint16 mm (depth scale 1000),",
        "  `pose/<idx>.txt` 4x4 camera->world, `camera_info.json` (intrinsics 3x3).",
        "- Frames are sampled from the scene `.sens` at stride 5 (~6 fps effective).",
    ]
    return "\n".join(lines) + "\n"
