#!/usr/bin/env bash
set -o errexit

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$REPO_ROOT"
python -m pip install --upgrade pip
python -m pip install -e ".[api]"
python scripts/fetch_model_bundle.py

cd "$REPO_ROOT/apps/api"
python manage.py collectstatic --noinput
python manage.py migrate --noinput



