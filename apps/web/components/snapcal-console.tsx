"use client";

import Image from "next/image";
import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";

type NutritionFacts = {
  calories_kcal: number | null;
};

type ClassPrediction = {
  class_name: string;
  nutrition_adjusted: NutritionFacts;
};

type PredictionResponse = {
  selected_class: string;
  top_predictions: ClassPrediction[];
  latency_ms?: {
    total?: number;
  };
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

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
    formData.append("portion_multiplier", "1.0");

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
  const latencyText = result?.latency_ms?.total
    ? `Fetched in ${result.latency_ms.total.toFixed(0)}ms.`
    : null;

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
                  {latencyText ? (
                    <p className="result-muted">{latencyText}</p>
                  ) : null}
                </div>
              </>
            ) : null}
          </button>

          {error ? <p className="warning-line">{error}</p> : null}

          <button className="button" type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Analyzing..." : "Run SnapCal"}
          </button>
        </form>
      </section>
    </main>
  );
}
