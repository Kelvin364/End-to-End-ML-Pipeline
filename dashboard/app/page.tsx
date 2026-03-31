"use client";

import { useEffect, useState, useCallback } from "react";
import { api, StatusResponse, MetricsResponse } from "@/lib/api";
import { MetricCard } from "@/components/MetricCard";
import { SectionHeader } from "@/components/SectionHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { ProgressBar } from "@/components/ProgressBar";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorMessage } from "@/components/ErrorMessage";

const CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"];
const CLASS_COLORS: Record<string, string> = {
  EOSINOPHIL: "#EF4444",
  LYMPHOCYTE: "#3B82F6",
  MONOCYTE  : "#10B981",
  NEUTROPHIL: "#8B5CF6",
};

export default function Dashboard() {
  const [status,  setStatus]  = useState<StatusResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");

  const load = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([api.status(), api.metrics()]);
      setStatus(s);
      setMetrics(m);
      setError(null);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 30000);
    return () => clearInterval(interval);
  }, [load]);

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3 text-secondary">
      <LoadingSpinner size={16} />
      <span className="text-sm font-mono">Connecting to API...</span>
    </div>
  );

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-foreground tracking-tight">
            Dashboard
          </h1>
          <p className="mt-1 text-sm text-secondary">
            Model status, uptime, and performance metrics
          </p>
        </div>
        <div className="flex items-center gap-4">
          {lastUpdated && (
            <span className="text-xs font-mono text-secondary">
              Updated {lastUpdated}
            </span>
          )}
          <StatusBadge
            status={status ? "online" : "offline"}
            label={status ? "API Online" : "API Offline"}
          />
          <button
            onClick={load}
            className="text-xs font-mono text-secondary hover:text-foreground
              border border-border px-3 py-1.5 rounded hover:border-muted
              transition-all duration-150"
          >
            Refresh
          </button>
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {status && (
        <>
          {/* Core metrics */}
          <section className="mb-10">
            <SectionHeader
              title="Model Performance"
              description="Overall weighted metrics computed on the held-out test set of 1,244 images."
            />
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <MetricCard
                label="Test Accuracy"
                value={`${metrics?.overall.accuracy ?? status.accuracy ?? "—"}%`}
                sub="On 1,244 test images"
                highlight mono
              />
              <MetricCard
                label="Weighted F1"
                value={`${metrics?.overall.weighted_f1 ?? status.f1 ?? "—"}%`}
                sub="Harmonic mean P/R"
                mono
              />
              <MetricCard
                label="Weighted Precision"
                value={`${metrics?.overall.weighted_precision ?? "—"}%`}
                sub="Positive predictive value"
                mono
              />
              <MetricCard
                label="Weighted Recall"
                value={`${metrics?.overall.weighted_recall ?? "—"}%`}
                sub="Sensitivity"
                mono
              />
            </div>

            {/* Uptime + deployment */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                label="API Uptime"
                value={status.uptime_human}
                sub="Since last deploy"
                mono
              />
              <MetricCard
                label="Model"
                value={status.model_name || "Custom CNN"}
                sub={`${metrics?.dataset_info.image_size ?? "96x96"} input`}
              />
              <MetricCard
                label="Scheduler"
                value={status.scheduler_running ? "Running" : "Stopped"}
                sub="Autonomous retraining"
              />
              <MetricCard
                label="Dataset"
                value={`${metrics?.dataset_info.total_images?.toLocaleString() ?? "12,444"}`}
                sub={`${metrics?.dataset_info.stain ?? "Wright-Giemsa"} stain`}
                mono
              />
            </div>
          </section>

          {/* Per-class metrics */}
          {metrics && (
            <section className="mb-10">
              <SectionHeader
                title="Per-Class Metrics"
                description="Precision, recall, and F1 score for each white blood cell type on the test set."
              />
              <div className="rounded-lg border border-border overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border bg-muted/40">
                      {["Cell Type","Precision","Recall","F1 Score","Support"].map(h => (
                        <th key={h} className="
                          text-left px-5 py-3 text-xs font-medium
                          uppercase tracking-widest text-secondary
                        ">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {CLASSES.map((cls, i) => {
                      const m = metrics.per_class[cls];
                      if (!m) return null;
                      return (
                        <tr
                          key={cls}
                          className={`
                            border-b border-border last:border-0
                            hover:bg-muted/20 transition-colors
                          `}
                        >
                          <td className="px-5 py-4">
                            <div className="flex items-center gap-3">
                              <div
                                className="w-2 h-2 rounded-full shrink-0"
                                style={{ background: CLASS_COLORS[cls] }}
                              />
                              <span className="text-sm font-medium text-foreground">
                                {cls}
                              </span>
                            </div>
                          </td>
                          {[m.precision, m.recall, m.f1].map((val, j) => (
                            <td key={j} className="px-5 py-4">
                              <div className="flex items-center gap-3">
                                <span className="text-sm font-mono text-foreground w-14">
                                  {val.toFixed(2)}%
                                </span>
                                <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
                                  <div
                                    className="h-full rounded-full transition-all duration-700"
                                    style={{
                                      width: `${val}%`,
                                      background: CLASS_COLORS[cls]
                                    }}
                                  />
                                </div>
                              </div>
                            </td>
                          ))}
                          <td className="px-5 py-4">
                            <span className="text-sm font-mono text-secondary">
                              {m.support.toLocaleString()}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Auto-retrain progress */}
          <section>
            <SectionHeader
              title="Autonomous Retraining"
              description="Retraining triggers automatically via GitHub Actions when the pending image threshold is reached."
            />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="rounded-lg border border-border bg-surface p-5">
                <ProgressBar
                  value={status.pending_images_for_retrain}
                  max={status.retrain_threshold}
                  label="Pending images for retraining"
                  color="bg-accent"
                />
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-secondary mb-1">Threshold</p>
                    <p className="text-sm font-mono text-foreground">
                      {status.retrain_threshold} images
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-secondary mb-1">Next check</p>
                    <p className="text-sm font-mono text-foreground truncate">
                      {status.next_auto_retrain?.split("T").pop()?.slice(0,5) ?? "—"}
                    </p>
                  </div>
                </div>
              </div>
              <div className="rounded-lg border border-border bg-surface p-5">
                <p className="text-xs font-medium uppercase tracking-widest
                  text-secondary mb-4">
                  Infrastructure
                </p>
                <div className="space-y-3">
                  {[
                    ["Runner",    "GitHub Actions"],
                    ["RAM",       "7 GB (free tier)"],
                    ["Trigger",   "Repository dispatch"],
                    ["Storage",   "Supabase Storage"],
                    ["Database",  "Supabase PostgreSQL"],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center">
                      <span className="text-xs text-secondary">{k}</span>
                      <span className="text-xs font-mono text-foreground">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
