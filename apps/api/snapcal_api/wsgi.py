"""WSGI config for the SnapCal API."""

from __future__ import annotations

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "snapcal_api.settings")

application = get_wsgi_application()
