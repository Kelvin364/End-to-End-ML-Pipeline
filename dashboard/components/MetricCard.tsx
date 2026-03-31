interface MetricCardProps {
  label     : string;
  value     : string | number;
  sub       ?: string;
  highlight ?: boolean;
  mono      ?: boolean;
}

export function MetricCard({
  label, value, sub, highlight, mono
}: MetricCardProps) {
  return (
    <div className={`
      rounded-lg border p-5 transition-all duration-200
      ${highlight
        ? "border-accent/40 bg-accent/5"
        : "border-border bg-surface"
      }
    `}>
      <p className="text-xs font-medium uppercase tracking-widest text-secondary mb-2">
        {label}
      </p>
      <p className={`
        text-2xl font-semibold text-foreground leading-none
        ${mono ? "font-mono" : ""}
      `}>
        {value}
      </p>
      {sub && (
        <p className="mt-2 text-xs text-secondary">{sub}</p>
      )}
    </div>
  );
}
