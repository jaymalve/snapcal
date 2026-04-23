# SnapCal

SnapCal is a monorepo for SAM-guided food recognition and calorie estimation from meal photos. It combines an offline ML research pipeline, a curated nutrition mapping layer, a Django REST inference service, and a Next.js frontend for image upload and camera capture.

## Monorepo Layout

- `apps/web`: Next.js frontend for upload, camera capture, and results display
- `apps/api`: Django REST API for health checks and inference
- `ml`: scripts for manifest generation, segmentation, training, evaluation, and export
- `src/snapcal`: shared Python package used by both the ML pipeline and the API
- `configs`: JSON experiment and segmentation configs
- `data/reference`: versioned reference assets such as the USDA mapping template and manifest outputs

## What Is Implemented

- Reproducible config-driven training and segmentation scaffolding
- Food-101 manifest generation with a deterministic train/validation split
- MobileSAM mask ranking, crop generation, and segmentation diagnostics utilities
- Model registry and training loop stubs for `resnet50`, `efficientnet_b0`, and `vit_b16`
- Nutrition lookup and preset portion-selection utilities
- Django REST endpoints for `/api/v1/health` and `/api/v1/predict`
- Next.js upload UI wired to the API contract

## What Still Requires Local Setup

- Install Python and Node dependencies
- Add the Food-101 dataset under `data/raw/food-101`
- Replace the seeded USDA mapping template values with curated FoodData Central rows
- Train and export a production model bundle before enabling live inference

## Colab Workflow

- `notebooks/colab_train.ipynb` is a ready-to-run Google Colab notebook for mounting Drive, authenticating Kaggle, downloading Food-101, running the SnapCal ML pipeline, and exporting a Render-ready `production_bundle`.
- The notebook assumes you first push this repo to GitHub and replace the placeholder `REPO_URL` with your repository URL.
- Artifacts should be saved back to Drive after each major phase so Colab runtime resets do not wipe checkpoints or reports.

## Deployment Workflow

- `render.yaml` and `build.sh` provide a Render deployment path for the Django API.
- `scripts/fetch_model_bundle.py` optionally downloads and unpacks a zipped or tarred `production_bundle` when `SNAPCAL_MODEL_BUNDLE_URL` is set.
- The Next.js app already reads `NEXT_PUBLIC_API_BASE_URL` from [apps/web/.env.example](/Users/jaymalave/Desktop/SnapCal/apps/web/.env.example:1), so Vercel only needs that environment variable pointed at your deployed Django API URL.

## Quick Start

### Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,train,dev]"
```

### Web

```bash
npm install
npm run dev:web
```

### API

```bash
source .venv/bin/activate
cd apps/api
python manage.py runserver
```

### Render Build

```bash
./build.sh
```

### ML Scripts

```bash
source .venv/bin/activate
python ml/scripts/build_manifest.py --dataset-root data/raw/food-101 --output data/reference/manifests/food101_manifest.csv
python ml/scripts/run_segmentation.py --manifest data/reference/manifests/food101_manifest.csv --config configs/segmentation/mobilesam_default.json
python ml/scripts/train.py --config configs/training/vit_b16_segmented.json
```

## Notes

The `data/reference/food101_usda_mapping.csv` file is seeded as a repo-tracked template with one row per Food-101 class so the pipeline and schema are stable from the start. Replace placeholder values and `fdc_id`s with manually curated USDA FoodData Central entries before reporting calorie metrics.
