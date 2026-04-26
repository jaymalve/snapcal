# Deployment Assets

This folder contains deployment entrypoints and templates for the hosted SnapCal stack.

## Modal

- `modal_inference.py`: deploys one predict endpoint and one health endpoint per model.
- Expected Modal resources:
  - a Volume named `snapcal-models` mounted at `/models`
  - a Secret named `snapcal-modal-auth` exposing `AUTH_TOKEN`

## Lightsail

- `lightsail/snapcal.service.example`: `systemd` unit for Gunicorn
- `lightsail/nginx.snapcal.conf.example`: Nginx reverse proxy config
- `lightsail/.env.production.remote.example`: remote-inference environment template

## Production Split

- `Vercel`: frontend in `apps/web`
- `Lightsail`: Django API in `apps/api`
- `Modal`: classifier inference endpoints

For the first hosted deployment, keep `SNAPCAL_ENABLE_SEGMENTATION=false` so the backend serves the raw-only path.
