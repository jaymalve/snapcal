"""Request and response serializers for inference endpoints."""

from __future__ import annotations

from rest_framework import serializers


class PredictRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()
    portion_multiplier = serializers.FloatField(default=1.0, min_value=0.01)


class NutritionFactsSerializer(serializers.Serializer):
    serving_size_g = serializers.FloatField(allow_null=True)
    serving_unit = serializers.CharField()
    calories_kcal = serializers.FloatField(allow_null=True)
    protein_g = serializers.FloatField(allow_null=True)
    carbs_g = serializers.FloatField(allow_null=True)
    fat_g = serializers.FloatField(allow_null=True)


class ClassPredictionSerializer(serializers.Serializer):
    class_name = serializers.CharField()
    confidence = serializers.FloatField()
    nutrition_per_serving = NutritionFactsSerializer()
    nutrition_adjusted = NutritionFactsSerializer()


class PredictionResponseSerializer(serializers.Serializer):
    selected_class = serializers.CharField()
    top_predictions = ClassPredictionSerializer(many=True)
    portion_multiplier = serializers.FloatField()
    model_version = serializers.CharField()
    segmentation_preview_url = serializers.CharField(allow_null=True, required=False)
    latency_ms = serializers.DictField(child=serializers.FloatField(), required=False)
    warnings = serializers.ListField(child=serializers.CharField(), required=False)


class HealthResponseSerializer(serializers.Serializer):
    ready = serializers.BooleanField()
    model_version = serializers.CharField(required=False)
    model_name = serializers.CharField(required=False)
    bundle_dir = serializers.CharField(required=False)
    error = serializers.CharField(required=False)
