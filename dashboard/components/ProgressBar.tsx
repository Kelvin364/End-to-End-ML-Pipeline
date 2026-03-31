interface ProgressBarProps {
  value : number;
  max   : number;
  label?: string;
  color?: string;
}

export function ProgressBar({ value, max, label, color = "bg-accent" }: ProgressBarProps) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div>
      {label && (
        <div className="flex justify-between items-center mb-1.5">
          <span className="text-xs text-secondary">{label}</span>
          <span className="text-xs font-mono text-foreground">
            {value} / {max}
          </span>
        </div>
      )}
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
