import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  root: path.resolve(__dirname, "static"),
  publicDir: path.resolve(__dirname, "public"),
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, "dist", "client"),
    emptyOutDir: false,
    target: "es2022",
    sourcemap: false,
    rollupOptions: {
      output: {
        entryFileNames: "assets/editor-[hash].js",
        chunkFileNames: "assets/editor-chunk-[hash].js",
        assetFileNames: "assets/editor-[hash][extname]",
      },
    },
  },
});
