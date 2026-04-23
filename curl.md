# Curl Commands

Quick manual checks for the local Django API.

## Health

```bash
curl http://127.0.0.1:8000/api/v1/health
```

## Predict With Standard Serving

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "portion_unit=serving"
```

## Predict With Solid Portion

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "portion_unit=oz" \
  -F "portion_value=8"
```

## Predict With Liquid Portion

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "portion_unit=fl_oz" \
  -F "portion_value=8"
```

## Quick Latency Check

```bash
curl -s -o /dev/null -w "total=%{time_total}s\n" -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "portion_unit=serving"
```
