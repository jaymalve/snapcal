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

`SNAPCAL_ENABLE_SEGMENTATION` is now a backend capability flag. If you set it to `true`, the frontend checkbox can choose between the raw and segmented paths per request. If you set it to `false`, the checkbox stays disabled and every request uses the raw classifier path.

From the repo root:

```bash
export SNAPCAL_MODEL_BUNDLE=artifacts/models/production_bundle
export SNAPCAL_ENABLE_SEGMENTATION=true
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
  -F "enable_segmentation=false" \
  -F "portion_unit=serving"
```

Solid food example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=false" \
  -F "portion_unit=oz" \
  -F "portion_value=8"
```

Liquid example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=false" \
  -F "portion_unit=fl_oz" \
  -F "portion_value=8"
```

Segmented comparison example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=true" \
  -F "portion_unit=serving"
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
- Set `SNAPCAL_ENABLE_SEGMENTATION=true` only on backends that have MobileSAM dependencies, a valid segmentation config, and the checkpoint file available.
- If you want raw-only fastest mode, set `SNAPCAL_ENABLE_SEGMENTATION=false` and the frontend comparison checkbox will stay disabled.
- If the API reports not ready, check that `artifacts/models/production_bundle/metadata.json` exists and that `SNAPCAL_MODEL_BUNDLE` points to the correct folder.
