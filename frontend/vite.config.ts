import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.LYRICTRACE_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: backend, changeOrigin: true },
      // Expose the backend's OpenAPI docs through the dev server.
      "/docs": { target: backend, changeOrigin: true },
      "/openapi.json": { target: backend, changeOrigin: true },
    },
  },
});
