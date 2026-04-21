# ML Workflow

The `ml/scripts` entrypoints keep the research workflow reproducible and artifact-driven:

1. Build a canonical manifest from Food-101 metadata.
2. Run offline MobileSAM segmentation and update the manifest with generated paths and diagnostics.
3. Train raw and segmented baselines from JSON configs.
4. Evaluate saved predictions and export the best checkpoint as an inference bundle for the API.

## Suggested Order

```bash
python ml/scripts/build_manifest.py --dataset-root data/raw/food-101 --output data/reference/manifests/food101_manifest.csv
python ml/scripts/run_segmentation.py --manifest data/reference/manifests/food101_manifest.csv --config configs/segmentation/mobilesam_default.json
python ml/scripts/train.py --config configs/training/resnet50_raw.json
python ml/scripts/train.py --config configs/training/efficientnet_b0_raw.json
python ml/scripts/train.py --config configs/training/vit_b16_raw.json
python ml/scripts/train.py --config configs/training/resnet50_segmented.json
python ml/scripts/train.py --config configs/training/efficientnet_b0_segmented.json
python ml/scripts/train.py --config configs/training/vit_b16_segmented.json
python ml/scripts/export_inference_bundle.py --checkpoint artifacts/models/vit_b16_segmented/best.pt --config configs/training/vit_b16_segmented.json --output-dir artifacts/models/production_bundle
```
