from __future__ import annotations

from typing import Sequence

from spatial_memory_evaluation import ObjectPrediction, RGBDSequence


class DummySpatialMemory:
    """Minimal adapter showing the required method interface."""

    def __init__(self, sequence: RGBDSequence):
        self.sequence = sequence

    def get_memory_text(self, question: str) -> str:
        return (
            f"I received {len(self.sequence.frames)} RGB-D frames for "
            f"{self.sequence.episode_history}, but this dummy method cannot answer: "
            f"{question}"
        )

    def get_object(self, query: str) -> Sequence[ObjectPrediction]:
        return [
            ObjectPrediction(
                label=query,
                score=0.0,
                attributes={"source": "dummy", "episode": self.sequence.episode_history},
            )
        ]


def create_method(sequence: RGBDSequence) -> DummySpatialMemory:
    return DummySpatialMemory(sequence)
