import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0f9ff",
          500: "#0284c7",
          600: "#0369a1",
        },
        slate: {
          850: "#151e2e",
          950: "#0b0f19",
        },
      },
    },
  },
  plugins: [],
};
export default config;
