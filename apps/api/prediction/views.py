"""Inference endpoint views."""

from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from snapcal.inference import InferenceNotReadyError

from .serializers import HealthResponseSerializer, PredictRequestSerializer, PredictionResponseSerializer
from .services import UnknownModelError, get_prediction_health, get_prediction_runtime, resolve_model_id


class HealthView(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        payload = get_prediction_health()
        serializer = HealthResponseSerializer(payload)
        return Response(serializer.data)


class PredictView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image_file = serializer.validated_data["image"]
        model_id = serializer.validated_data.get("model_id")
        enable_segmentation = bool(serializer.validated_data["enable_segmentation"])
        portion_unit = str(serializer.validated_data["portion_unit"])
        portion_value = serializer.validated_data.get("portion_value")
        try:
            resolved_model_id = resolve_model_id(model_id)
            response = get_prediction_runtime(resolved_model_id).predict(
                image_bytes=image_file.read(),
                portion_unit=portion_unit,
                portion_value=portion_value,
                enable_segmentation=enable_segmentation,
                model_id=resolved_model_id,
            )
        except UnknownModelError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except InferenceNotReadyError as exc:
            return Response(
                {
                    "detail": str(exc),
                    "ready": False,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except RuntimeError as exc:
            return Response(
                {
                    "detail": str(exc),
                    "ready": False,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        response_serializer = PredictionResponseSerializer(response.to_dict())
        return Response(response_serializer.data, status=status.HTTP_200_OK)
