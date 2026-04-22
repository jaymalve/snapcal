"""Django settings for SnapCal API."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str = "") -> List[str]:
    raw_value = os.getenv(name, default)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "snapcal-dev-secret-key")
DEBUG = _env_flag("DJANGO_DEBUG", True)

render_external_hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
render_external_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()

allowed_hosts = set(_env_list("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost"))
if render_external_hostname:
    allowed_hosts.add(render_external_hostname)
if DEBUG:
    allowed_hosts.add("0.0.0.0")
ALLOWED_HOSTS = sorted(allowed_hosts)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "prediction",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "snapcal_api.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "snapcal_api.wsgi.application"
ASGI_APPLICATION = "snapcal_api.asgi.application"

database_url = os.getenv("DATABASE_URL", "").strip()
if database_url:
    import dj_database_url

    DATABASES = {
        "default": dj_database_url.parse(
            database_url,
            conn_max_age=600,
            ssl_require=_env_flag("DJANGO_DB_SSL_REQUIRE", False),
        )
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

_cors_default = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
# Common Next.js dev ports; merged in DEBUG so a .env that only lists :3000 still allows :3001, etc.
_local_dev_origins = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
}

cors_allowed_origins = set(_env_list("CORS_ALLOWED_ORIGINS", _cors_default))
if DEBUG:
    cors_allowed_origins |= _local_dev_origins

csrf_trusted_origins = set(
    _env_list("CSRF_TRUSTED_ORIGINS", _cors_default)
)
if DEBUG:
    csrf_trusted_origins |= _local_dev_origins
if render_external_url:
    csrf_trusted_origins.add(render_external_url)

CORS_ALLOWED_ORIGINS = sorted(cors_allowed_origins)
CSRF_TRUSTED_ORIGINS = sorted(csrf_trusted_origins)
CORS_ALLOW_CREDENTIALS = False
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_SSL_REDIRECT = _env_flag("DJANGO_SECURE_SSL_REDIRECT", False)

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ]
}
