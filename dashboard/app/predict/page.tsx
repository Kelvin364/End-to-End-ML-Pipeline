"use client";

import { useState, useRef, DragEvent } from "react";
import { api, PredictionResponse } from "@/lib/api";
import { SectionHeader } from "@/components/SectionHeader";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorMessage } from "@/components/ErrorMessage";

const CLASS_COLORS: Record<string, string> = {
  EOSINOPHIL: "#EF4444",
  LYMPHOCYTE: "#3B82F6",
  MONOCYTE  : "#10B981",
  NEUTROPHIL: "#8B5CF6",
};

const CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"];

export default function PredictPage() {
  const [file,     setFile]     = useState<File | null>(null);
  const [preview,  setPreview]  = useState<string | null>(null);
  const [result,   setResult]   = useState<PredictionResponse | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = e => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const runPredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict(file);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-foreground tracking-tight">
          Predict
        </h1>
        <p className="mt-1 text-sm text-secondary">
          Upload a single Wright-Giemsa stained white blood cell image for classification
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload zone */}
        <div>
          <SectionHeader title="Image Input" />
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`
              relative rounded-lg border-2 border-dashed cursor-pointer
              transition-all duration-200 overflow-hidden
              ${dragging
                ? "border-accent bg-accent/5"
                : "border-border hover:border-muted bg-surface"
              }
              ${preview ? "aspect-square" : "aspect-video"}
            `}
          >
            {preview ? (
              <img
                src={preview}
                alt="Cell preview"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center
                justify-center gap-3 text-secondary p-8">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <p className="text-sm text-center">
                  Drag and drop or click to select an image
                </p>
                <p className="text-xs text-secondary/60 text-center">
                  JPEG or PNG — single cropped cell image
                </p>
              </div>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            className="hidden"
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
          />

          {file && (
            <div className="mt-3 flex items-center justify-between">
              <p className="text-xs font-mono text-secondary truncate">
                {file.name} — {(file.size / 1024).toFixed(1)} KB
              </p>
              <button
                onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                className="text-xs text-secondary hover:text-danger transition-colors ml-4 shrink-0"
              >
                Remove
              </button>
            </div>
          )}

          <button
            onClick={runPredict}
            disabled={!file || loading}
            className="
              mt-4 w-full py-2.5 px-4 rounded-md text-sm font-medium
              bg-accent text-white hover:bg-accent-hover
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-all duration-150 flex items-center justify-center gap-2
            "
          >
            {loading ? (
              <>
                <LoadingSpinner size={14} />
                Classifying...
              </>
            ) : (
              "Run Classification"
            )}
          </button>

          {error && <div className="mt-4"><ErrorMessage message={error} /></div>}
        </div>

        {/* Result */}
        <div>
          <SectionHeader title="Classification Result" />

          {!result && !loading && (
            <div className="rounded-lg border border-border bg-surface
              aspect-video flex items-center justify-center">
              <p className="text-sm text-secondary font-mono">
                Awaiting image input
              </p>
            </div>
          )}

          {loading && (
            <div className="rounded-lg border border-border bg-surface
              aspect-video flex flex-col items-center justify-center gap-3">
              <LoadingSpinner size={24} />
              <p className="text-xs font-mono text-secondary">
                Running inference...
              </p>
            </div>
          )}

          {result && !loading && (
            <div className="animate-slide-up space-y-4">
              {/* Primary result */}
              <div className="rounded-lg border bg-surface overflow-hidden"
                style={{
                  borderColor: CLASS_COLORS[result.prediction.label] + "40"
                }}>
                <div className="px-5 py-4 border-b border-border flex
                  items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ background: CLASS_COLORS[result.prediction.label] }}
                    />
                    <span className="text-lg font-semibold text-foreground">
                      {result.prediction.label}
                    </span>
                  </div>
                  <span className="text-2xl font-mono font-semibold"
                    style={{ color: CLASS_COLORS[result.prediction.label] }}>
                    {result.prediction.confidence.toFixed(1)}%
                  </span>
                </div>
                <div className="px-5 py-3">
                  <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${result.prediction.confidence}%`,
                        background: CLASS_COLORS[result.prediction.label]
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* All class scores */}
              <div className="rounded-lg border border-border bg-surface p-5">
                <p className="text-xs font-medium uppercase tracking-widest
                  text-secondary mb-4">
                  All Class Scores
                </p>
                <div className="space-y-3">
                  {CLASSES.map(cls => {
                    const score = result.prediction.all_scores[cls] ?? 0;
                    const isTop = cls === result.prediction.label;
                    return (
                      <div key={cls}>
                        <div className="flex justify-between items-center mb-1.5">
                          <span className={`text-xs font-mono ${
                            isTop ? "text-foreground font-medium" : "text-secondary"
                          }`}>
                            {cls}
                          </span>
                          <span className={`text-xs font-mono ${
                            isTop ? "text-foreground" : "text-secondary"
                          }`}>
                            {score.toFixed(2)}%
                          </span>
                        </div>
                        <div className="h-1 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-700"
                            style={{
                              width: `${score}%`,
                              background: isTop
                                ? CLASS_COLORS[cls]
                                : CLASS_COLORS[cls] + "60"
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* File info */}
              <div className="rounded-lg border border-border bg-surface p-4">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-secondary">File</span>
                  <span className="text-foreground truncate ml-4">
                    {result.filename}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
