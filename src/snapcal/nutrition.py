"""USDA mapping and calorie estimation helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import FOOD101_CLASSES, USDA_MAPPING_COLUMNS
from .schemas import ClassPrediction, NutritionFacts, PredictionResponse


def _parse_optional_float(raw_value: str) -> Optional[float]:
    raw_value = raw_value.strip()
    if raw_value == "":
        return None
    return float(raw_value)


@dataclass(frozen=True)
class NutritionMappingEntry:
    food101_class: str
    usda_food_name: str
    fdc_id: str
    serving_size_g: Optional[float]
    serving_unit: str
    calories_kcal: Optional[float]
    protein_g: Optional[float]
    carbs_g: Optional[float]
    fat_g: Optional[float]
    mapping_confidence: str
    notes: str

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "NutritionMappingEntry":
        return cls(
            food101_class=row["food101_class"],
            usda_food_name=row["usda_food_name"],
            fdc_id=row["fdc_id"],
            serving_size_g=_parse_optional_float(row["serving_size_g"]),
            serving_unit=row["serving_unit"],
            calories_kcal=_parse_optional_float(row["calories_kcal"]),
            protein_g=_parse_optional_float(row["protein_g"]),
            carbs_g=_parse_optional_float(row["carbs_g"]),
            fat_g=_parse_optional_float(row["fat_g"]),
            mapping_confidence=row["mapping_confidence"],
            notes=row["notes"],
        )

    def nutrition_facts(self) -> NutritionFacts:
        return NutritionFacts(
            serving_size_g=self.serving_size_g,
            serving_unit=self.serving_unit or "serving",
            calories_kcal=self.calories_kcal,
            protein_g=self.protein_g,
            carbs_g=self.carbs_g,
            fat_g=self.fat_g,
        )

    def is_complete(self) -> bool:
        return all(
            value is not None
            for value in (self.serving_size_g, self.calories_kcal, self.protein_g, self.carbs_g, self.fat_g)
        )


class NutritionLookup:
    def __init__(self, entries: Dict[str, NutritionMappingEntry]):
        self._entries = entries

    @classmethod
    def from_csv(cls, path: Path) -> "NutritionLookup":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_columns = [column for column in USDA_MAPPING_COLUMNS if column not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"USDA mapping missing columns: {missing_columns}")
            entries = {row["food101_class"]: NutritionMappingEntry.from_row(row) for row in reader}
        return cls(entries)

    def validate(self, expected_classes: Sequence[str] = FOOD101_CLASSES) -> List[str]:
        issues: List[str] = []
        for class_name in expected_classes:
            if class_name not in self._entries:
                issues.append(f"Missing USDA mapping for class '{class_name}'")
                continue
            if not self._entries[class_name].serving_unit:
                issues.append(f"Class '{class_name}' is missing serving_unit")
        return issues

    def get(self, class_name: str) -> NutritionMappingEntry:
        if class_name not in self._entries:
            raise KeyError(f"No USDA mapping for class '{class_name}'")
        return self._entries[class_name]

    def build_response(
        self,
        ranked_predictions: Iterable[Tuple[str, float]],
        portion_multiplier: float,
        model_version: str,
        segmentation_preview_url: Optional[str] = None,
        latency_ms: Optional[Dict[str, float]] = None,
        warnings: Optional[List[str]] = None,
    ) -> PredictionResponse:
        predictions: List[ClassPrediction] = []
        ranked_list = list(ranked_predictions)
        for class_name, confidence in ranked_list:
            entry = self.get(class_name)
            per_serving = entry.nutrition_facts()
            predictions.append(
                ClassPrediction(
                    class_name=class_name,
                    confidence=round(confidence, 6),
                    nutrition_per_serving=per_serving,
                    nutrition_adjusted=per_serving.scaled(portion_multiplier),
                )
            )
        if not predictions:
            raise ValueError("At least one prediction is required")
        return PredictionResponse(
            selected_class=predictions[0].class_name,
            top_predictions=predictions,
            portion_multiplier=portion_multiplier,
            model_version=model_version,
            segmentation_preview_url=segmentation_preview_url,
            latency_ms=latency_ms or {},
            warnings=warnings or [],
        )


def calorie_absolute_error(expected_kcal: float, predicted_kcal: Optional[float]) -> Optional[float]:
    if predicted_kcal is None:
        return None
    return abs(expected_kcal - predicted_kcal)
