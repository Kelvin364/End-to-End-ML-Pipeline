"use client";

import { useEffect, useState, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from "recharts";
import { api, HistoryRun } from "@/lib/api";
import { SectionHeader } from "@/components/SectionHeader";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorMessage } from "@/components/ErrorMessage";
import { MetricCard } from "@/components/MetricCard";

const TOOLTIP_STYLE = {
  backgroundColor : "#111118",
  border          : "1px solid #1E1E2A",
  borderRadius    : "6px",
  color           : "#F4F4F5",
  fontSize        : "11px",
  fontFamily      : "DM Mono, monospace",
};

function fmt(v: number | null) {
  return v != null ? v.toFixed(4) : "—";
}

function fmtDate(s: string) {
  try {
    return new Date(s).toLocaleString("en-GB", {
      day: "2-digit", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit"
    });
  } catch { return s; }
}

export default function HistoryPage() {
  const [runs,    setRuns]    = useState<HistoryRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.history();
      setRuns(res.runs || []);
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const improved   = runs.filter(r => r.improved).length;
  const totalImgs  = runs.reduce((s, r) => s + (r.images_used ?? 0), 0);
  const avgDuration = runs.length
    ? (runs.reduce((s, r) => s + (r.duration_s ?? 0), 0) / runs.length).toFixed(0)
    : "—";

  // Chart data — F1 trend
  const chartData = [...runs]
    .reverse()
    .map((r, i) => ({
      run     : `Run ${i + 1}`,
      before  : r.f1_before,
      after   : r.f1_after,
    }))
    .filter(d => d.before != null || d.after != null);

  return (
    <div className="animate-fade-in">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-foreground tracking-tight">
            Retraining History
          </h1>
          <p className="mt-1 text-sm text-secondary">
            All past retraining runs — manual and autonomous — logged from Supabase
          </p>
        </div>
        <button
          onClick={load}
          className="text-xs font-mono text-secondary hover:text-foreground
            border border-border px-3 py-1.5 rounded hover:border-muted
            transition-all duration-150"
        >
          Refresh
        </button>
      </div>

      {loading && (
        <div className="flex items-center gap-3 text-secondary py-8">
          <LoadingSpinner size={16} />
          <span className="text-sm font-mono">Loading history...</span>
        </div>
      )}
      {error && <ErrorMessage message={error} />}

      {!loading && (
        <>
          {/* Summary metrics */}
          <section className="mb-10">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                label="Total Runs"
                value={runs.length}
                sub="Since deployment"
                mono
              />
              <MetricCard
                label="Improved"
                value={`${improved} / ${runs.length}`}
                sub="Model improved"
                mono
              />
              <MetricCard
                label="Images Processed"
                value={totalImgs.toLocaleString()}
                sub="Total training images"
                mono
              />
              <MetricCard
                label="Avg Duration"
                value={avgDuration !== "—" ? `${avgDuration}s` : "—"}
                sub="Per retraining run"
                mono
              />
            </div>
          </section>

          {/* F1 trend chart */}
          {chartData.length > 1 && (
            <section className="mb-10">
              <SectionHeader
                title="F1 Score Trend"
                description="Weighted F1 before and after each retraining run."
              />
              <div className="rounded-lg border border-border bg-surface p-6">
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart
                    data={chartData}
                    margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1E1E2A"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="run"
                      tick={{ fill: "#71717A", fontSize: 11 }}
                      axisLine={{ stroke: "#1E1E2A" }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fill: "#71717A", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                      domain={["auto", "auto"]}
                      tickFormatter={v => v.toFixed(3)}
                    />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      formatter={(v) => [v != null ? Number(v).toFixed(4) : "—"]}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: 11, fontFamily: "DM Mono" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="before"
                      name="F1 Before"
                      stroke="#71717A"
                      strokeWidth={1.5}
                      dot={{ fill: "#71717A", r: 3 }}
                      strokeDasharray="4 2"
                    />
                    <Line
                      type="monotone"
                      dataKey="after"
                      name="F1 After"
                      stroke="#2563EB"
                      strokeWidth={2}
                      dot={{ fill: "#2563EB", r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}

          {/* Runs table */}
          <section>
            <SectionHeader title="Run Log" />

            {runs.length === 0 ? (
              <div className="rounded-lg border border-border bg-surface p-12
                text-center">
                <p className="text-sm text-secondary font-mono">
                  No retraining runs recorded yet.
                </p>
                <p className="text-xs text-secondary/60 mt-1">
                  Upload images and trigger retraining to see results here.
                </p>
              </div>
            ) : (
              <div className="rounded-lg border border-border overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border bg-muted/30">
                      {[
                        "Timestamp","Triggered By","Images",
                        "F1 Before","F1 After","Improved","Epochs","Duration"
                      ].map(h => (
                        <th key={h} className="text-left px-4 py-3 text-xs
                          font-medium uppercase tracking-widest text-secondary">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map(run => (
                      <tr key={run.id}
                        className="border-b border-border last:border-0
                          hover:bg-muted/20 transition-colors">
                        <td className="px-4 py-3 text-xs font-mono text-secondary">
                          {run.triggered_at ? fmtDate(run.triggered_at) : "—"}
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs font-mono px-2 py-0.5 rounded
                            bg-muted/60 text-secondary border border-border">
                            {run.triggered_by ?? "—"}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-xs font-mono text-foreground">
                          {run.images_used ?? "—"}
                        </td>
                        <td className="px-4 py-3 text-xs font-mono text-secondary">
                          {fmt(run.f1_before)}
                        </td>
                        <td className="px-4 py-3 text-xs font-mono text-foreground">
                          {fmt(run.f1_after)}
                        </td>
                        <td className="px-4 py-3">
                          {run.improved == null ? (
                            <span className="text-xs text-secondary font-mono">—</span>
                          ) : run.improved ? (
                            <span className="text-xs px-2 py-0.5 rounded-full
                              bg-success/10 text-success border border-success/20
                              font-mono">
                              Yes
                            </span>
                          ) : (
                            <span className="text-xs px-2 py-0.5 rounded-full
                              bg-danger/10 text-danger border border-danger/20
                              font-mono">
                              No
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-xs font-mono text-secondary">
                          {run.epochs_run ?? "—"}
                        </td>
                        <td className="px-4 py-3 text-xs font-mono text-secondary">
                          {run.duration_s != null ? `${run.duration_s}s` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
