interface SectionHeaderProps {
  title       : string;
  description ?: string;
}

export function SectionHeader({ title, description }: SectionHeaderProps) {
  return (
    <div className="mb-8">
      <h2 className="text-lg font-semibold text-foreground tracking-tight">
        {title}
      </h2>
      {description && (
        <p className="mt-1 text-sm text-secondary leading-relaxed max-w-2xl">
          {description}
        </p>
      )}
    </div>
  );
}
