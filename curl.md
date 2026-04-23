# Curl Commands

Quick manual checks for the local Django API.

If you want segmented requests to work, start the backend with `SNAPCAL_ENABLE_SEGMENTATION=true`. The request flag only enables segmentation when the backend reports that it is available.

## Health

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Check that the response includes `"segmentation_available": true` before using segmented requests.

## Predict With Standard Serving

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=false" \
  -F "portion_unit=serving"
```

## Predict With Solid Portion

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=false" \
  -F "portion_unit=oz" \
  -F "portion_value=8"
```

## Predict With Liquid Portion

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=false" \
  -F "portion_unit=fl_oz" \
  -F "portion_value=8"
```

## Predict With Segmentation Enabled

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=true" \
  -F "portion_unit=serving"
```

## Compare Raw vs Segmented

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=false" \
  -F "portion_unit=serving"

curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=true" \
  -F "portion_unit=serving"
```

The response now includes `segmentation_requested` and `segmentation_applied` so you can confirm which path actually ran.

## Quick Latency Check

```bash
curl -s -o /dev/null -w "total=%{time_total}s\n" -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=false" \
  -F "portion_unit=serving"
```

Segmented latency check:

```bash
curl -s -o /dev/null -w "total=%{time_total}s\n" -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "enable_segmentation=true" \
  -F "portion_unit=serving"
```
