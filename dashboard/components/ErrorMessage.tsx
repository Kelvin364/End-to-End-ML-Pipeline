export function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-danger/20 bg-danger/5 p-4">
      <p className="text-sm text-danger font-mono">{message}</p>
    </div>
  );
}
