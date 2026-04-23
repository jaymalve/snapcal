"use client";

import Image from "next/image";
import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";

type PortionUnit = "serving" | "oz" | "fl_oz";

type NutritionFacts = {
  serving_size_g: number | null;
  serving_unit: string;
  calories_kcal: number | null;
  protein_g: number | null;
  carbs_g: number | null;
  fat_g: number | null;
};

type ClassPrediction = {
  class_name: string;
  confidence: number;
  nutrition_per_serving: NutritionFacts;
  nutrition_adjusted: NutritionFacts;
};

type RequestedPortion = {
  unit: PortionUnit;
  value: number | null;
  label: string;
  grams: number | null;
  approximate: boolean;
};

type PredictionResponse = {
  selected_class: string;
  top_predictions: ClassPrediction[];
  requested_portion: RequestedPortion;
  latency_ms?: {
    total?: number;
  };
};

type PortionOption = {
  value: number;
  label: string;
};

const SOLID_PORTION_OPTIONS: PortionOption[] = [
  { value: 4, label: "4 oz" },
  { value: 6, label: "6 oz" },
  { value: 8, label: "8 oz" },
  { value: 12, label: "12 oz" },
  { value: 16, label: "16 oz (1 lb)" },
  { value: 24, label: "24 oz (1 lb 8 oz)" },
  { value: 32, label: "32 oz (2 lb)" }
];

const LIQUID_PORTION_OPTIONS: PortionOption[] = [
  { value: 4, label: "4 fl oz" },
  { value: 8, label: "8 fl oz" },
  { value: 12, label: "12 fl oz" },
  { value: 16, label: "16 fl oz" }
];

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function defaultPortionValue(unit: Exclude<PortionUnit, "serving">) {
  return unit === "oz"
    ? String(SOLID_PORTION_OPTIONS[2].value)
    : String(LIQUID_PORTION_OPTIONS[1].value);
}

function formatLabel(value: string) {
  return value.replaceAll("_", " ");
}

function capitalizeFirstLetter(value: string) {
  if (!value) {
    return value;
  }
  return value[0].toUpperCase() + value.slice(1);
}

function formatCalories(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return "N/A";
  }
  return `${Math.round(value)} kcal`;
}

export function SnapCalConsole() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [portionUnit, setPortionUnit] = useState<PortionUnit>("serving");
  const [portionValue, setPortionValue] = useState<string>(defaultPortionValue("oz"));
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setResult(null);
    setError(null);
  }

  function onPortionUnitChange(event: ChangeEvent<HTMLSelectElement>) {
    const nextUnit = event.target.value as PortionUnit;
    setPortionUnit(nextUnit);
    setPortionValue(
      nextUnit === "serving" ? defaultPortionValue("oz") : defaultPortionValue(nextUnit)
    );
    setResult(null);
    setError(null);
  }

  function onPortionValueChange(event: ChangeEvent<HTMLSelectElement>) {
    setPortionValue(event.target.value);
    setResult(null);
    setError(null);
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("Choose an image before running inference.");
      return;
    }
    setIsSubmitting(true);
    setResult(null);
    setError(null);
    const formData = new FormData();
    formData.append("image", file);
    formData.append("portion_unit", portionUnit);
    if (portionUnit !== "serving") {
      formData.append("portion_value", portionValue);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: "POST",
        body: formData
      });
      const payload = (await response.json()) as PredictionResponse & {
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(
          payload.detail ??
            "Inference service is not ready yet. Export a model bundle first."
        );
      }
      setResult(payload);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : "Something went wrong while calling the API.";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }

  const primaryPrediction = result?.top_predictions?.[0] ?? null;
  const detectedMeal = primaryPrediction
    ? capitalizeFirstLetter(formatLabel(primaryPrediction.class_name))
    : result
      ? capitalizeFirstLetter(formatLabel(result.selected_class))
      : null;
  const estimatedCalories = primaryPrediction
    ? formatCalories(primaryPrediction.nutrition_adjusted.calories_kcal)
    : "N/A";
  const requestedPortion = result?.requested_portion ?? null;
  const requestedPortionText = requestedPortion?.label ?? "Standard serving";
  const requestedPortionNote =
    requestedPortion?.approximate && requestedPortion.grams !== null
      ? `Approx. ${Math.round(requestedPortion.grams)} g water-equivalent.`
      : null;
  const latencyText = result?.latency_ms?.total
    ? `Fetched in ${result.latency_ms.total.toFixed(0)}ms.`
    : null;
  const visiblePortionOptions =
    portionUnit === "oz"
      ? SOLID_PORTION_OPTIONS
      : portionUnit === "fl_oz"
        ? LIQUID_PORTION_OPTIONS
        : [];
  const portionHint =
    portionUnit === "serving"
      ? "Using the USDA standard serving unless you choose a specific size."
      : portionUnit === "fl_oz"
        ? "Liquid sizes use an approximate water-equivalent conversion."
        : "Solid sizes are converted from ounces to grams before scaling nutrition.";

  function openFilePicker() {
    inputRef.current?.click();
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <h2 className="headline">
          Ultra-fast
          <br /> meal calories estimation.
        </h2>
      </section>

      <section className="grid">
        <form className="panel panel-strong upload-panel" onSubmit={onSubmit}>
          <input
            ref={inputRef}
            id="image-upload"
            className="file-input-hidden"
            type="file"
            accept="image/*"
            capture="environment"
            onChange={onFileChange}
          />

          <button
            className={`dropzone dropzone-button${previewUrl ? " dropzone-filled" : ""}`}
            type="button"
            onClick={openFilePicker}
            aria-label={
              previewUrl ? "Replace selected image" : "Choose an image to analyze"
            }
          >
            {previewUrl ? (
              <Image
                className="dropzone-image"
                src={previewUrl}
                alt="Selected meal preview"
                fill
                unoptimized
                sizes="(max-width: 768px) 100vw, 760px"
              />
            ) : (
              <div className="dropzone-empty">
                <strong>Tap or click to upload an image.</strong>
              </div>
            )}

            {previewUrl && result ? (
              <>
                <div className="dropzone-scrim" aria-hidden="true" />
                <div className="result-drawer">
                  <p className="result-line">
                    Detected meal:{" "}
                    <span className="result-emphasis">{detectedMeal}</span>
                  </p>
                  <p className="result-line">
                    Estimated Calories:{" "}
                    <span className="result-emphasis">{estimatedCalories}</span>
                  </p>
                  <p className="result-line">
                    Portion used:{" "}
                    <span className="result-emphasis">{requestedPortionText}</span>
                  </p>
                  {requestedPortionNote ? (
                    <p className="result-muted">{requestedPortionNote}</p>
                  ) : null}
                  {latencyText ? (
                    <p className="result-muted">{latencyText}</p>
                  ) : null}
                </div>
              </>
            ) : null}
          </button>

          <div className="portion-controls">
            <label className="field" htmlFor="portion-unit">
              <span className="field-label">Portion</span>
              <select
                id="portion-unit"
                className="input"
                value={portionUnit}
                onChange={onPortionUnitChange}
              >
                <option value="serving">Standard serving</option>
                <option value="oz">Solid food</option>
                <option value="fl_oz">Liquid</option>
              </select>
            </label>

            {portionUnit !== "serving" ? (
              <label className="field" htmlFor="portion-value">
                <span className="field-label">Size</span>
                <select
                  id="portion-value"
                  className="input"
                  value={portionValue}
                  onChange={onPortionValueChange}
                >
                  {visiblePortionOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}

            <p className="field-hint">{portionHint}</p>
          </div>

          {error ? <p className="warning-line">{error}</p> : null}

          <button className="button" type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Analyzing..." : "Run SnapCal"}
          </button>
        </form>
      </section>
    </main>
  );
}
