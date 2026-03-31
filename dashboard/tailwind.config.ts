import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans : ["DM Sans", "sans-serif"],
        mono : ["DM Mono", "monospace"],
      },
      colors: {
        background : "#0A0A0F",
        surface    : "#111118",
        border     : "#1E1E2A",
        muted      : "#2A2A38",
        accent     : "#2563EB",
        "accent-hover": "#1D4ED8",
        foreground : "#F4F4F5",
        secondary  : "#71717A",
        success    : "#10B981",
        warning    : "#F59E0B",
        danger     : "#EF4444",
      },
      animation: {
        "fade-in"  : "fadeIn 0.4s ease-out",
        "slide-up" : "slideUp 0.3s ease-out",
        "pulse-slow": "pulse 3s ease-in-out infinite",
      },
      keyframes: {
        fadeIn  : { from: { opacity: "0" }, to: { opacity: "1" } },
        slideUp : {
          from: { opacity: "0", transform: "translateY(8px)" },
          to  : { opacity: "1", transform: "translateY(0)" }
        },
      },
    },
  },
  plugins: [],
};

export default config;
