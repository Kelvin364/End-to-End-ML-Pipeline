export function LoadingSpinner({ size = 20 }: { size?: number }) {
  return (
    <svg
      width={size} height={size}
      viewBox="0 0 24 24" fill="none"
      className="animate-spin text-secondary"
    >
      <circle
        cx="12" cy="12" r="10"
        stroke="currentColor" strokeWidth="2"
        strokeDasharray="60" strokeDashoffset="20"
        strokeLinecap="round"
      />
    </svg>
  );
}
