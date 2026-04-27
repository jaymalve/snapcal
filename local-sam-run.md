# Local SAM Run

Use this guide to run the full SnapCal app locally with:

- backend on `127.0.0.1:8000`
- frontend on `localhost:3000`
- all 3 local bundles available in the model picker
- optional SAM checkbox support

## 1. Backend Terminal

From the repo root:

```bash
cd /Users/jaymalave/Desktop/SnapCal
source .venv/bin/activate
pip install -e ".[api]"
```

If you want the SAM checkbox enabled locally, make sure the MobileSAM checkpoint exists:

```bash
python ml/scripts/fetch_mobilesam_checkpoint.py
```

Clear production-only env vars from this shell:

```bash
unset SNAPCAL_MODEL_REGISTRY
unset MODAL_AUTH_TOKEN
unset DATABASE_URL
unset DJANGO_SECURE_SSL_REDIRECT
```

Point the backend at the 3 local bundles:

```bash
export SNAPCAL_MODEL_BUNDLES='{"resnet50":"artifacts/models/production_bundle","efficientnet_b0":"artifacts/models/production_bundle_efficientnet_b0","vit_b16":"artifacts/models/production_bundle_vit_b16"}'
export SNAPCAL_DEFAULT_MODEL_ID='resnet50'
```

Enable local SAM support:

```bash
export SNAPCAL_ENABLE_SEGMENTATION='true'
```

If you want raw-only local mode instead:

```bash
export SNAPCAL_ENABLE_SEGMENTATION='false'
```

Set local Django env vars:

```bash
export DJANGO_DEBUG='true'
export DJANGO_ALLOWED_HOSTS='127.0.0.1,localhost'
export CSRF_TRUSTED_ORIGINS='http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001'
```

Run migrations and start the backend:

```bash
python apps/api/manage.py migrate
python apps/api/manage.py runserver 127.0.0.1:8000
```

## 2. Verify Backend

In another terminal:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

You should see:

- all 3 models in `models`
- if SAM is enabled and healthy, `segmentation_available: true`

Test a raw request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/Users/jaymalave/Desktop/SnapCal/burger.jpg" \
  -F "model_id=resnet50" \
  -F "enable_segmentation=false" \
  -F "portion_unit=serving"
```

If SAM is enabled, test a segmented request too:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/Users/jaymalave/Desktop/SnapCal/burger.jpg" \
  -F "model_id=resnet50" \
  -F "enable_segmentation=true" \
  -F "portion_unit=serving"
```

## 3. Frontend Terminal

From the repo root:

```bash
cd /Users/jaymalave/Desktop/SnapCal
npm install
```

Make sure `apps/web/.env.local` contains:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Then start the frontend:

```bash
npm run dev:web
```

## 4. Open the App

Open:

```text
http://localhost:3000
```

You should be able to:

- pick `ResNet-50`, `EfficientNet-B0`, or `ViT-B/16`
- upload `burger.jpg`
- use the SAM checkbox if local segmentation is enabled and healthy

## 5. If the SAM Checkbox Is Disabled

Run:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Check the `segmentation_reason` field.

Common causes:

- `artifacts/cache/mobile_sam.pt` is missing
- `mobile_sam` dependencies are not installed in the current Python environment
- the backend was started from the wrong directory or with the wrong env vars
