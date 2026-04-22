# Local Run

This guide is for running SnapCal entirely on your machine with a locally exported model bundle from Colab.

## 1. Download the bundle from Colab

After the Colab export step, download this folder from Drive or Colab:

- `production_bundle`

Place it in your local repo here:

- `artifacts/models/production_bundle`

The bundle should contain at least:

- `metadata.json`
- `model.pt`
- `nutrition_mapping.csv`

## 2. Expected local layout

From the repo root:

```text
SnapCal/
  artifacts/
    models/
      production_bundle/
        metadata.json
        model.pt
        nutrition_mapping.csv
```

## 3. Create and activate a Python environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api]"
```

## 4. Run the Django API locally

This project currently uses a raw model bundle, so segmentation must be disabled at inference time.

From the repo root:

```bash
export SNAPCAL_MODEL_BUNDLE=artifacts/models/production_bundle
export SNAPCAL_ENABLE_SEGMENTATION=false
export DJANGO_DEBUG=true
export DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
export CSRF_TRUSTED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001
python apps/api/manage.py migrate
python apps/api/manage.py runserver
```

The API will be available at:

- `http://127.0.0.1:8000`

## 5. Verify the API

Health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Expected result should include `"ready": true`.

Prediction test:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "portion_multiplier=1.0"
```

## 6. Run the frontend locally

Create `apps/web/.env.local` with:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Then from the repo root:

```bash
npm install
npm run dev:web
```

The frontend will usually be available at:

- `http://localhost:3000`

## 7. Notes

- Do not commit `artifacts/models/production_bundle`; it is already gitignored.
- If you later export a segmented model bundle, set `SNAPCAL_ENABLE_SEGMENTATION=true` only if the bundle was trained/exported for segmented inference.
- If the API reports not ready, check that `artifacts/models/production_bundle/metadata.json` exists and that `SNAPCAL_MODEL_BUNDLE` points to the correct folder.
