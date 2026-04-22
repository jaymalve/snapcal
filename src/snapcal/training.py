"""Config-driven model training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import TrainingConfig, ensure_directories
from .constants import FOOD101_CLASSES
from .datasets import Food101ManifestDataset
from .evaluation import save_report, summarize_predictions
from .models import build_image_transforms, build_model, extract_logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - depends on optional dependency
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainArtifacts:
    checkpoint_path: Path
    report_path: Path
    predictions_path: Path


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def _device(self):
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_loaders(self):
        from torch.utils.data import DataLoader

        train_dataset = Food101ManifestDataset(
            manifest_path=self.config.train_manifest,
            split="train",
            dataset_variant=self.config.dataset_variant,
            transform=build_image_transforms(self.config.image_size, train=True),
        )
        val_dataset = Food101ManifestDataset(
            manifest_path=self.config.val_manifest,
            split="val",
            dataset_variant=self.config.dataset_variant,
            transform=build_image_transforms(self.config.image_size, train=False),
        )
        test_dataset = Food101ManifestDataset(
            manifest_path=self.config.test_manifest,
            split="test",
            dataset_variant=self.config.dataset_variant,
            transform=build_image_transforms(self.config.image_size, train=False),
        )
        loader_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": True,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        return train_loader, val_loader, test_loader

    def _prepare_training(self):
        import torch

        seed_everything(self.config.seed)
        ensure_directories((self.config.output_dir, self.config.report_dir))
        model = build_model(self.config.model_name, self.config.num_classes)
        device = self._device()
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        start_epoch = 0
        checkpoint_path = self.config.output_dir / "last.pt"
        if checkpoint_path.exists():
            # These are trusted checkpoints produced by this project and include
            # non-tensor metadata such as pathlib paths in the saved config.
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
        return model, optimizer, scheduler, criterion, device, start_epoch

    def _forward(self, model, images):
        outputs = model(pixel_values=images) if self.config.model_name == "vit_b16" else model(images)
        return extract_logits(outputs)

    def _run_epoch(self, model, loader, optimizer, criterion, device):
        import torch

        model.train()
        running_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = self._forward(model, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item()) * int(labels.size(0))
        return running_loss / max(1, len(loader.dataset))

    def evaluate_loader(self, model, loader, device):
        import torch

        model.eval()
        labels: List[int] = []
        ranked_predictions: List[List[int]] = []
        scores_payload: List[Dict[str, object]] = []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                batch_labels = batch["label"].to(device)
                logits = self._forward(model, images)
                probabilities = torch.softmax(logits, dim=1)
                topk_scores, topk_indices = torch.topk(probabilities, k=min(5, probabilities.size(1)), dim=1)
                labels.extend(batch_labels.cpu().tolist())
                ranked_predictions.extend(topk_indices.cpu().tolist())
                for image_id, label, scores, indices in zip(
                    batch["image_id"],
                    batch_labels.cpu().tolist(),
                    topk_scores.cpu().tolist(),
                    topk_indices.cpu().tolist(),
                ):
                    scores_payload.append(
                        {
                            "image_id": image_id,
                            "label_index": label,
                            "label_name": FOOD101_CLASSES[label],
                            "topk_indices": indices,
                            "topk_classes": [FOOD101_CLASSES[index] for index in indices],
                            "topk_scores": [round(float(score), 6) for score in scores],
                        }
                    )
        report = summarize_predictions(labels, ranked_predictions, FOOD101_CLASSES)
        return report, scores_payload

    def fit(self) -> TrainArtifacts:
        import torch

        train_loader, val_loader, test_loader = self._build_loaders()
        model, optimizer, scheduler, criterion, device, start_epoch = self._prepare_training()
        best_val = -1.0
        best_checkpoint_path = self.config.output_dir / "best.pt"
        last_checkpoint_path = self.config.output_dir / "last.pt"
        for epoch in range(start_epoch, self.config.epochs):
            train_loss = self._run_epoch(model, train_loader, optimizer, criterion, device)
            val_report, _ = self.evaluate_loader(model, val_loader, device)
            scheduler.step()
            checkpoint = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_top1_accuracy": val_report.top1_accuracy,
                "config": self.config.__dict__,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(checkpoint, last_checkpoint_path)
            if val_report.top1_accuracy >= best_val:
                best_val = val_report.top1_accuracy
                torch.save(checkpoint, best_checkpoint_path)
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state"])
        test_report, predictions_payload = self.evaluate_loader(model, test_loader, device)
        report_path = self.config.report_dir / "test_report.json"
        predictions_path = self.config.report_dir / "test_predictions.json"
        save_report(test_report, report_path)
        with predictions_path.open("w", encoding="utf-8") as handle:
            json.dump(predictions_payload, handle, indent=2)
        return TrainArtifacts(
            checkpoint_path=best_checkpoint_path,
            report_path=report_path,
            predictions_path=predictions_path,
        )
