interface StatusBadgeProps {
  status  : "online" | "offline" | "warning";
  label  ?: string;
}

export function StatusBadge({ status, label }: StatusBadgeProps) {
  const colors = {
    online  : "bg-success/10 text-success border-success/20",
    offline : "bg-danger/10  text-danger  border-danger/20",
    warning : "bg-warning/10 text-warning border-warning/20",
  };
  const dots = {
    online  : "bg-success",
    offline : "bg-danger",
    warning : "bg-warning",
  };
  return (
    <span className={`
      inline-flex items-center gap-2 px-3 py-1 rounded-full
      text-xs font-medium border ${colors[status]}
    `}>
      <span className={`w-1.5 h-1.5 rounded-full ${dots[status]}
        ${status === "online" ? "animate-pulse-slow" : ""}
      `} />
      {label || status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}
