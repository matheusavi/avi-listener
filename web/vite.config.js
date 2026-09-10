import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // `npm run dev` serves the interface here and forwards API calls to the
    // Python process, so both can be reloaded independently.
    proxy: { "/api": "http://127.0.0.1:8000" }
  },
  build: { outDir: "dist", emptyOutDir: true }
});
