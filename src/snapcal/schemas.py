"""Serializable inference response schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class NutritionFacts:
    serving_size_g: Optional[float]
    serving_unit: str
    calories_kcal: Optional[float]
    protein_g: Optional[float]
    carbs_g: Optional[float]
    fat_g: Optional[float]

    def scaled(
        self,
        multiplier: float,
        serving_size_g: Optional[float] = None,
        serving_unit: Optional[str] = None,
    ) -> "NutritionFacts":
        def scale(value: Optional[float]) -> Optional[float]:
            return None if value is None else round(value * multiplier, 2)

        return NutritionFacts(
            serving_size_g=serving_size_g if serving_size_g is not None else scale(self.serving_size_g),
            serving_unit=serving_unit or self.serving_unit,
            calories_kcal=scale(self.calories_kcal),
            protein_g=scale(self.protein_g),
            carbs_g=scale(self.carbs_g),
            fat_g=scale(self.fat_g),
        )


@dataclass(frozen=True)
class ClassPrediction:
    class_name: str
    confidence: float
    nutrition_per_serving: NutritionFacts
    nutrition_adjusted: NutritionFacts


@dataclass(frozen=True)
class RequestedPortion:
    unit: str
    value: Optional[int]
    label: str
    grams: Optional[float]
    approximate: bool


@dataclass(frozen=True)
class PredictionResponse:
    selected_class: str
    top_predictions: List[ClassPrediction]
    requested_portion: RequestedPortion
    model_version: str
    segmentation_requested: bool = False
    segmentation_applied: bool = False
    segmentation_preview_url: Optional[str] = None
    latency_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
