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
- `lightsail/mobilesam.runtime.json.example`: production MobileSAM config with an absolute checkpoint path for backend-side segmentation

## Production Split

- `Vercel`: frontend in `apps/web`
- `Lightsail`: Django API in `apps/api`
- `Modal`: classifier inference endpoints

For the first hosted deployment, keep `SNAPCAL_ENABLE_SEGMENTATION=false` so the backend serves the raw-only path.

To enable the production SAM checkbox later:

1. Copy `artifacts/cache/mobile_sam.pt` to the Lightsail server at `/opt/snapcal/artifacts/cache/mobile_sam.pt`
2. Copy `lightsail/mobilesam.runtime.json.example` to `/opt/snapcal/deploy/lightsail/mobilesam.runtime.json`
3. Set `SNAPCAL_ENABLE_SEGMENTATION=true`
4. Set `SNAPCAL_SEGMENTATION_CONFIG=/opt/snapcal/deploy/lightsail/mobilesam.runtime.json`
5. Optionally set `SNAPCAL_SEGMENTATION_MAX_SIDE=1024` to downscale large uploads before SAM runs
6. Reinstall backend dependencies with `pip install -e ".[api_remote]"`
7. Restart the `snapcal` systemd service
