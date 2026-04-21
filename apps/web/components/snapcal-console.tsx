"use client";

import { ChangeEvent, FormEvent, useEffect, useState } from "react";

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

type PredictionResponse = {
  selected_class: string;
  top_predictions: ClassPrediction[];
  portion_multiplier: number;
  model_version: string;
  segmentation_preview_url?: string | null;
  latency_ms?: Record<string, number>;
  warnings?: string[];
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function formatLabel(value: string) {
  return value.replaceAll("_", " ");
}

function formatNumber(value: number | null, suffix = "") {
  if (value === null || Number.isNaN(value)) {
    return "N/A";
  }
  return `${value.toFixed(1)}${suffix}`;
}

export function SnapCalConsole() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [portionMultiplier, setPortionMultiplier] = useState("1.0");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

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

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("Choose a meal photo before running inference.");
      return;
    }
    setIsSubmitting(true);
    setError(null);
    const formData = new FormData();
    formData.append("image", file);
    formData.append("portion_multiplier", portionMultiplier);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as PredictionResponse & {
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(
          payload.detail ??
            "Inference service is not ready yet. Export a model bundle first.",
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

  return (
    <main className="page-shell">
      <section className="hero">
        <span className="eyebrow">SAM-guided meal intelligence</span>
        <h1 className="headline">Segment the plate. Classify the meal. Estimate the calories.</h1>
        <p className="lede">
          SnapCal pairs MobileSAM food isolation with transformer-based
          classification and a USDA-backed nutrition layer. Upload a photo,
          adjust portion size, and inspect the top-3 predictions with calorie
          and macro estimates.
        </p>
      </section>

      <section className="grid">
        <form className="panel panel-strong upload-panel stack" onSubmit={onSubmit}>
          <div>
            <label className="label" htmlFor="image-upload">
              Meal photo
            </label>
            <div className="dropzone">
              {previewUrl ? (
                <img src={previewUrl} alt="Selected meal preview" />
              ) : (
                <div className="dropzone-empty">
                  <strong>Upload a photo or open the camera.</strong>
                  <p>
                    The backend expects the full SnapCal inference bundle before
                    live predictions will succeed.
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="controls">
            <input
              id="image-upload"
              className="file-input"
              type="file"
              accept="image/*"
              capture="environment"
              onChange={onFileChange}
            />

            <div className="field-row">
              <input
                className="input"
                type="number"
                min="0.01"
                step="0.05"
                value={portionMultiplier}
                onChange={(event) => setPortionMultiplier(event.target.value)}
                placeholder="Portion multiplier"
              />
              <button className="button" type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Analyzing..." : "Run SnapCal"}
              </button>
            </div>
          </div>
        </form>

        <aside className="panel results-panel">
          <div className="metric-strip">
            <article className="metric-card">
              <p className="metric-label">Selected class</p>
              <p className="metric-value">
                {result ? formatLabel(result.selected_class) : "Waiting"}
              </p>
            </article>
            <article className="metric-card">
              <p className="metric-label">Calories</p>
              <p className="metric-value">
                {primaryPrediction
                  ? formatNumber(
                      primaryPrediction.nutrition_adjusted.calories_kcal,
                      " kcal",
                    )
                  : "N/A"}
              </p>
            </article>
            <article className="metric-card">
              <p className="metric-label">Latency</p>
              <p className="metric-value">
                {result?.latency_ms?.total
                  ? `${result.latency_ms.total.toFixed(0)} ms`
                  : "N/A"}
              </p>
            </article>
          </div>

          {error ? <p className="warning-line">{error}</p> : null}

          <div className="prediction-list">
            {(result?.top_predictions ?? []).map((prediction) => (
              <article className="prediction-card" key={prediction.class_name}>
                <div className="prediction-header">
                  <h2 className="prediction-name">
                    {formatLabel(prediction.class_name)}
                  </h2>
                  <span className="prediction-confidence">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="nutrition-line">
                  Per serving:{" "}
                  {formatNumber(prediction.nutrition_per_serving.calories_kcal, " kcal")}
                  {" • "}
                  {formatNumber(prediction.nutrition_per_serving.protein_g, "g protein")}
                  {" • "}
                  {formatNumber(prediction.nutrition_per_serving.carbs_g, "g carbs")}
                  {" • "}
                  {formatNumber(prediction.nutrition_per_serving.fat_g, "g fat")}
                </p>
                <p className="nutrition-line">
                  Adjusted:{" "}
                  {formatNumber(prediction.nutrition_adjusted.calories_kcal, " kcal")}
                  {" • "}
                  {formatNumber(prediction.nutrition_adjusted.protein_g, "g protein")}
                  {" • "}
                  {formatNumber(prediction.nutrition_adjusted.carbs_g, "g carbs")}
                  {" • "}
                  {formatNumber(prediction.nutrition_adjusted.fat_g, "g fat")}
                </p>
              </article>
            ))}
          </div>

          {(result?.warnings ?? []).map((warning) => (
            <p className="warning-line" key={warning}>
              {warning}
            </p>
          ))}
        </aside>
      </section>
    </main>
  );
}
