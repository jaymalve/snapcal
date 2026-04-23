"""Request and response serializers for inference endpoints."""

from __future__ import annotations

from rest_framework import serializers

from snapcal.constants import (
    LIQUID_PORTION_FLUID_OUNCE_VALUES,
    PORTION_UNIT_FL_OZ,
    PORTION_UNIT_OZ,
    PORTION_UNIT_SERVING,
    SOLID_PORTION_OUNCE_VALUES,
)


class PredictRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()
    enable_segmentation = serializers.BooleanField(default=False)
    portion_unit = serializers.ChoiceField(
        choices=(
            (PORTION_UNIT_SERVING, PORTION_UNIT_SERVING),
            (PORTION_UNIT_OZ, PORTION_UNIT_OZ),
            (PORTION_UNIT_FL_OZ, PORTION_UNIT_FL_OZ),
        ),
        default=PORTION_UNIT_SERVING,
    )
    portion_value = serializers.IntegerField(required=False, allow_null=True, min_value=1)

    def validate(self, attrs):
        unit = attrs.get("portion_unit", PORTION_UNIT_SERVING)
        value = attrs.get("portion_value")
        if unit == PORTION_UNIT_SERVING:
            attrs["portion_value"] = None
            return attrs
        if value is None:
            raise serializers.ValidationError(
                {"portion_value": "Choose one of the preset portion sizes for the selected unit."}
            )
        allowed_values = (
            SOLID_PORTION_OUNCE_VALUES if unit == PORTION_UNIT_OZ else LIQUID_PORTION_FLUID_OUNCE_VALUES
        )
        if value not in allowed_values:
            allowed_text = ", ".join(str(option) for option in allowed_values)
            raise serializers.ValidationError(
                {"portion_value": f"Allowed values for {unit} are: {allowed_text}."}
            )
        return attrs


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


class RequestedPortionSerializer(serializers.Serializer):
    unit = serializers.CharField()
    value = serializers.IntegerField(allow_null=True)
    label = serializers.CharField()
    grams = serializers.FloatField(allow_null=True)
    approximate = serializers.BooleanField()


class PredictionResponseSerializer(serializers.Serializer):
    selected_class = serializers.CharField()
    top_predictions = ClassPredictionSerializer(many=True)
    requested_portion = RequestedPortionSerializer()
    model_version = serializers.CharField()
    segmentation_requested = serializers.BooleanField()
    segmentation_applied = serializers.BooleanField()
    segmentation_preview_url = serializers.CharField(allow_null=True, required=False)
    latency_ms = serializers.DictField(child=serializers.FloatField(), required=False)
    warnings = serializers.ListField(child=serializers.CharField(), required=False)


class HealthResponseSerializer(serializers.Serializer):
    ready = serializers.BooleanField()
    model_version = serializers.CharField(required=False)
    model_name = serializers.CharField(required=False)
    bundle_dir = serializers.CharField(required=False)
    segmentation_available = serializers.BooleanField(required=False)
    segmentation_reason = serializers.CharField(required=False, allow_null=True)
    error = serializers.CharField(required=False)
