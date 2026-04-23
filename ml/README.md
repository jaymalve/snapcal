# ML Workflow

The `ml/scripts` entrypoints keep the research workflow reproducible and artifact-driven:

1. Build a canonical manifest from Food-101 metadata.
2. Fetch the MobileSAM checkpoint used by offline segmentation.
3. Run offline MobileSAM segmentation and update the manifest with generated paths and diagnostics.
4. Train raw and segmented baselines from JSON configs.
5. Evaluate saved predictions and export the best checkpoint as an inference bundle for the API.

## Minimal Local ResNet Ablation

Use the local ResNet configs when you want the fastest fair raw-versus-segmented comparison before expanding to the full 6-run matrix:

```bash
python ml/scripts/fetch_mobilesam_checkpoint.py
python ml/scripts/run_segmentation.py --manifest data/reference/manifests/food101_manifest.csv --config configs/segmentation/mobilesam_default.json
python ml/scripts/train.py --config configs/training/resnet50_raw_local.json
python ml/scripts/train.py --config configs/training/resnet50_segmented_local.json
```

Segmented training now fails fast if `data/processed/segmented/...` files are missing, so do not skip the segmentation step.

## Quick 1-2 Hour Pilot Ablation

If you need a same-machine pilot run rather than a full-dataset result, create a smaller manifest and use the faster local segmentation config:

```bash
python ml/scripts/sample_manifest.py \
  --manifest data/reference/manifests/food101_manifest.csv \
  --output artifacts/manifests/food101_quick_ablation_manifest.csv \
  --train-per-class 5 \
  --val-per-class 1 \
  --test-per-class 2
python ml/scripts/fetch_mobilesam_checkpoint.py
python ml/scripts/run_segmentation.py --manifest artifacts/manifests/food101_quick_ablation_manifest.csv --config configs/segmentation/mobilesam_fast_local.json
python ml/scripts/train.py --config configs/training/resnet50_raw_quick_ablation.json
python ml/scripts/train.py --config configs/training/resnet50_segmented_quick_ablation.json
```

This keeps all 101 classes but reduces the pilot to 808 total images. It is suitable for a preliminary local ablation, not a final headline result.

## Suggested Order

```bash
python ml/scripts/build_manifest.py --dataset-root data/raw/food-101 --output data/reference/manifests/food101_manifest.csv
python ml/scripts/fetch_mobilesam_checkpoint.py
python ml/scripts/run_segmentation.py --manifest data/reference/manifests/food101_manifest.csv --config configs/segmentation/mobilesam_default.json
python ml/scripts/train.py --config configs/training/resnet50_raw.json
python ml/scripts/train.py --config configs/training/efficientnet_b0_raw.json
python ml/scripts/train.py --config configs/training/vit_b16_raw.json
python ml/scripts/train.py --config configs/training/resnet50_segmented.json
python ml/scripts/train.py --config configs/training/efficientnet_b0_segmented.json
python ml/scripts/train.py --config configs/training/vit_b16_segmented.json
python ml/scripts/export_inference_bundle.py --checkpoint artifacts/models/vit_b16_segmented/best.pt --config configs/training/vit_b16_segmented.json --output-dir artifacts/models/production_bundle
```
