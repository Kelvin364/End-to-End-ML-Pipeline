"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/",               label: "Dashboard",     short: "01" },
  { href: "/predict",        label: "Predict",       short: "02" },
  { href: "/visualisations", label: "Visualisations",short: "03" },
  { href: "/upload",         label: "Upload & Retrain", short: "04" },
  { href: "/history",        label: "History",       short: "05" },
];

export function Sidebar() {
  const path = usePathname();
  return (
    <aside className="
      fixed left-0 top-0 h-full w-60 bg-surface
      border-r border-border flex flex-col z-20
    ">
      {/* Logo */}
      <div className="px-6 py-6 border-b border-border">
        <p className="text-xs font-mono uppercase tracking-widest text-secondary mb-1">
          ALU — BSE
        </p>
        <h1 className="text-sm font-semibold text-foreground leading-tight">
          Blood Cell<br />Classification
        </h1>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-3">
        {NAV.map(({ href, label, short }) => {
          const active = path === href;
          return (
            <Link
              key={href}
              href={href}
              className={`
                flex items-center gap-3 px-3 py-2.5 rounded-md mb-0.5
                text-sm transition-all duration-150 group
                ${active
                  ? "bg-accent text-white font-medium"
                  : "text-secondary hover:text-foreground hover:bg-muted"
                }
              `}
            >
              <span className={`
                font-mono text-xs w-6 shrink-0
                ${active ? "text-white/70" : "text-secondary group-hover:text-foreground/50"}
              `}>
                {short}
              </span>
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-border">
        <p className="text-xs text-secondary font-mono">
          Kelvin Rwihimba
        </p>
        <p className="text-xs text-secondary/60 mt-0.5">
          Machine Learning Pipeline
        </p>
      </div>
    </aside>
  );
}
