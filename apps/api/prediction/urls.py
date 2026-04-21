"""Prediction app routes."""

from __future__ import annotations

from django.urls import path

from .views import HealthView, PredictView

urlpatterns = [
    path("health", HealthView.as_view(), name="health"),
    path("predict", PredictView.as_view(), name="predict"),
]
