"""Evaluation and reporting utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from .constants import FOOD101_CLASSES


@dataclass(frozen=True)
class PerClassMetrics:
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationReport:
    top1_accuracy: float
    top5_accuracy: float
    macro_f1: float
    per_class_f1: List[PerClassMetrics]
    confusion_matrix: List[List[int]]
    sample_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def build_confusion_matrix(labels: Sequence[int], predictions: Sequence[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, prediction in zip(labels, predictions):
        matrix[label][prediction] += 1
    return matrix


def compute_per_class_f1(confusion_matrix: Sequence[Sequence[int]], class_names: Sequence[str]) -> List[PerClassMetrics]:
    per_class: List[PerClassMetrics] = []
    for class_index, class_name in enumerate(class_names):
        tp = confusion_matrix[class_index][class_index]
        fp = sum(confusion_matrix[row][class_index] for row in range(len(class_names)) if row != class_index)
        fn = sum(confusion_matrix[class_index][column] for column in range(len(class_names)) if column != class_index)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        support = sum(confusion_matrix[class_index])
        per_class.append(
            PerClassMetrics(
                class_name=class_name,
                precision=round(precision, 6),
                recall=round(recall, 6),
                f1=round(f1, 6),
                support=support,
            )
        )
    return per_class


def summarize_predictions(
    labels: Sequence[int],
    ranked_predictions: Sequence[Sequence[int]],
    class_names: Sequence[str] = FOOD101_CLASSES,
) -> EvaluationReport:
    top1_predictions = [prediction[0] for prediction in ranked_predictions]
    top1_accuracy = mean(1.0 if label == predicted else 0.0 for label, predicted in zip(labels, top1_predictions))
    top5_accuracy = mean(
        1.0 if label in predicted[:5] else 0.0
        for label, predicted in zip(labels, ranked_predictions)
    )
    confusion = build_confusion_matrix(labels, top1_predictions, len(class_names))
    per_class = compute_per_class_f1(confusion, class_names)
    macro_f1 = mean(metrics.f1 for metrics in per_class)
    return EvaluationReport(
        top1_accuracy=round(top1_accuracy, 6),
        top5_accuracy=round(top5_accuracy, 6),
        macro_f1=round(macro_f1, 6),
        per_class_f1=per_class,
        confusion_matrix=confusion,
        sample_count=len(labels),
    )


def save_report(report: EvaluationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
