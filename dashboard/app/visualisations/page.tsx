"use client";

import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, Radar, Cell, Legend
} from "recharts";
import { api, MetricsResponse } from "@/lib/api";
import { SectionHeader } from "@/components/SectionHeader";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorMessage } from "@/components/ErrorMessage";

const COLORS = {
  EOSINOPHIL: "#EF4444",
  LYMPHOCYTE: "#3B82F6",
  MONOCYTE  : "#10B981",
  NEUTROPHIL: "#8B5CF6",
};

const CHART_STYLE = {
  background : "transparent",
  fontSize   : 11,
  fontFamily : "DM Mono, monospace",
};

const TOOLTIP_STYLE = {
  backgroundColor : "#111118",
  border          : "1px solid #1E1E2A",
  borderRadius    : "6px",
  color           : "#F4F4F5",
  fontSize        : "11px",
  fontFamily      : "DM Mono, monospace",
};

export default function VisualisationsPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  useEffect(() => {
    api.metrics()
      .then(setMetrics)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3 text-secondary">
      <LoadingSpinner size={16} />
      <span className="text-sm font-mono">Loading metrics...</span>
    </div>
  );

  if (error) return <ErrorMessage message={error} />;
  if (!metrics) return null;

  // Chart 1 data — model comparison
  const modelData = metrics.models_comparison.map(m => ({
    name      : m.name.replace(" FT", ""),
    Accuracy  : m.accuracy,
    F1        : m.f1,
    Precision : m.precision,
    Recall    : m.recall,
    best      : m.best,
  }));

  // Chart 2 data — class distribution
  const distData = Object.entries(metrics.class_distribution).map(([name, count]) => ({
    name,
    count,
    fill: COLORS[name as keyof typeof COLORS],
  }));

  // Chart 3 data — per-class radar
  const CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"];
  const radarData = ["Precision", "Recall", "F1"].map(metric => {
    const row: Record<string, string | number> = { metric };
    CLASSES.forEach(cls => {
      const m = metrics.per_class[cls];
      row[cls] = m
        ? metric === "Precision" ? m.precision
          : metric === "Recall"  ? m.recall
          : m.f1
        : 0;
    });
    return row;
  });

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-foreground tracking-tight">
          Visualisations
        </h1>
        <p className="mt-1 text-sm text-secondary">
          Dataset and model performance analysis — all values sourced dynamically from the API
        </p>
      </div>

      {/* Chart 1 — Model comparison */}
      <section className="mb-12">
        <SectionHeader
          title="1. Model Performance Comparison"
          description="Accuracy, F1, precision, and recall across all three trained architectures.
            The Custom CNN trained from scratch outperforms both transfer learning models
            because blood cell microscopy images are visually distinct from the ImageNet
            images that MobileNetV2 and EfficientNetB0 were pre-trained on. A domain gap
            reduces the effectiveness of transfer learning on this specific task."
        />
        <div className="rounded-lg border border-border bg-surface p-6">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={modelData}
              margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
              style={CHART_STYLE}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#1E1E2A"
                vertical={false}
              />
              <XAxis
                dataKey="name"
                tick={{ fill: "#71717A", fontSize: 11 }}
                axisLine={{ stroke: "#1E1E2A" }}
                tickLine={false}
              />
              <YAxis
                domain={[94, 101]}
                tick={{ fill: "#71717A", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={v => `${v}%`}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(v) => [v != null ? `${Number(v).toFixed(2)}%` : "—"]}
                cursor={{ fill: "rgba(255,255,255,0.03)" }}
              />
              <Legend
                wrapperStyle={{ fontSize: 11, fontFamily: "DM Mono" }}
              />
              <Bar dataKey="Accuracy"  fill="#2563EB" radius={[3,3,0,0]} />
              <Bar dataKey="F1"        fill="#10B981" radius={[3,3,0,0]} />
              <Bar dataKey="Precision" fill="#F59E0B" radius={[3,3,0,0]} />
              <Bar dataKey="Recall"    fill="#8B5CF6" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model comparison table */}
        <div className="mt-4 rounded-lg border border-border overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                {["Model","Accuracy","F1","Precision","Recall","Parameters"].map(h => (
                  <th key={h} className="text-left px-4 py-3 text-xs font-medium
                    uppercase tracking-widest text-secondary">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.models_comparison.map(m => (
                <tr key={m.name}
                  className="border-b border-border last:border-0 hover:bg-muted/20">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-foreground">{m.name}</span>
                      {m.best && (
                        <span className="text-xs px-1.5 py-0.5 rounded
                          bg-success/10 text-success border border-success/20
                          font-mono">
                          Best
                        </span>
                      )}
                    </div>
                  </td>
                  {[m.accuracy, m.f1, m.precision, m.recall].map((v, i) => (
                    <td key={i} className="px-4 py-3 text-sm font-mono text-foreground">
                      {v.toFixed(2)}%
                    </td>
                  ))}
                  <td className="px-4 py-3 text-sm font-mono text-secondary">
                    {m.params.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Chart 2 — Class distribution */}
      <section className="mb-12">
        <SectionHeader
          title="2. Class Distribution in Training Dataset"
          description="Number of images per white blood cell type across the 12,444-image dataset.
            The balance ratio is 1.01 — near-perfect balance — meaning no class weighting
            was required during training and the model cannot be biased toward any cell
            type by sheer volume. In clinical reality neutrophils represent 50–70% of
            circulating white cells, but equal representation ensures all four types
            are learned with equal thoroughness."
        />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-lg border border-border bg-surface p-6">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={distData}
                margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#1E1E2A"
                  vertical={false}
                />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#71717A", fontSize: 10 }}
                  axisLine={{ stroke: "#1E1E2A" }}
                  tickLine={false}
                />
                <YAxis
                  domain={[3000, 3200]}
                  tick={{ fill: "#71717A", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  cursor={{ fill: "rgba(255,255,255,0.03)" }}
                />
                <Bar dataKey="count" radius={[3,3,0,0]}>
                  {distData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Distribution stats */}
          <div className="rounded-lg border border-border bg-surface p-6">
            <p className="text-xs font-medium uppercase tracking-widest
              text-secondary mb-5">
              Distribution Statistics
            </p>
            <div className="space-y-4">
              {distData.map(d => {
                const total = Object.values(metrics.class_distribution)
                  .reduce((a, b) => a + b, 0);
                const pct = (d.count / total * 100).toFixed(1);
                return (
                  <div key={d.name}>
                    <div className="flex justify-between items-center mb-1.5">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full"
                          style={{ background: d.fill }} />
                        <span className="text-xs font-mono text-secondary">
                          {d.name}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-mono text-secondary">
                          {pct}%
                        </span>
                        <span className="text-xs font-mono text-foreground w-12 text-right">
                          {d.count.toLocaleString()}
                        </span>
                      </div>
                    </div>
                    <div className="h-1 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${pct}%`, background: d.fill }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-5 pt-4 border-t border-border flex justify-between">
              <span className="text-xs text-secondary font-mono">Total</span>
              <span className="text-xs font-mono text-foreground">
                {Object.values(metrics.class_distribution)
                  .reduce((a, b) => a + b, 0).toLocaleString()} images
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Chart 3 — Per-class precision/recall/F1 */}
      <section>
        <SectionHeader
          title="3. Per-Class Precision, Recall, and F1"
          description="Comparative view of all three classification metrics per cell type.
            Lymphocytes achieve near-perfect recall because their large, dense, round
            nucleus with minimal cytoplasm is visually distinctive. Eosinophils are
            identifiable by bright pink granules. Monocytes and neutrophils are the
            hardest pair — both are granulocytes with irregular lobed nuclei. This
            biological similarity creates the only source of confusion in the model,
            mirroring the difficulty experienced by trained haematologists."
        />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Grouped bar */}
          <div className="rounded-lg border border-border bg-surface p-6">
            <p className="text-xs font-medium uppercase tracking-widest
              text-secondary mb-4">
              Grouped by Metric
            </p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={radarData}
                margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#1E1E2A"
                  vertical={false}
                />
                <XAxis
                  dataKey="metric"
                  tick={{ fill: "#71717A", fontSize: 11 }}
                  axisLine={{ stroke: "#1E1E2A" }}
                  tickLine={false}
                />
                <YAxis
                  domain={[96, 101]}
                  tick={{ fill: "#71717A", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={v => `${v}%`}
                />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  formatter={(v) => [v != null ? `${Number(v).toFixed(2)}%` : "—"]}
                  cursor={{ fill: "rgba(255,255,255,0.03)" }}
                />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: "DM Mono" }} />
                {Object.entries(COLORS).map(([cls, color]) => (
                  <Bar key={cls} dataKey={cls} fill={color} radius={[2,2,0,0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Per-class metric table */}
          <div className="rounded-lg border border-border bg-surface p-6">
            <p className="text-xs font-medium uppercase tracking-widest
              text-secondary mb-4">
              Detailed Metrics Table
            </p>
            <div className="space-y-5">
              {Object.entries(COLORS).map(([cls, color]) => {
                const m = metrics.per_class[cls];
                if (!m) return null;
                return (
                  <div key={cls}>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-2 h-2 rounded-full"
                        style={{ background: color }} />
                      <span className="text-xs font-mono text-foreground">
                        {cls}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { label: "Precision", value: m.precision },
                        { label: "Recall",    value: m.recall },
                        { label: "F1",        value: m.f1 },
                      ].map(({ label, value }) => (
                        <div key={label}
                          className="bg-muted/30 rounded px-2 py-1.5 text-center">
                          <p className="text-xs text-secondary mb-0.5">{label}</p>
                          <p className="text-xs font-mono text-foreground">
                            {value.toFixed(2)}%
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
