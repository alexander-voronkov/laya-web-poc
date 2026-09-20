import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { readdir, rm } from "node:fs/promises";
import { join } from "node:path";

// Cross-origin isolation is required for multi-threaded wasm (SharedArrayBuffer).
// require-corp (not credentialless): it works with the Hugging Face CDN even though
// the CDN sends no CORP header, because the weights are loaded via fetch() in cors
// mode and CORP enforcement only applies to no-cors subresource loads.
const ISOLATION_HEADERS = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

function crossOriginIsolation(): Plugin {
  // Dev server middleware: sets the isolation headers on every response.
  // (Preview builds use the native `preview.headers` config below -- Vite 8's
  // preview middleware stack rejects custom middlewares added this way.)
  return {
    name: "cross-origin-isolation",
    configureServer: (server) => {
      server.middlewares.use((_req, res, next) => {
        for (const [k, v] of Object.entries(ISOLATION_HEADERS)) res.setHeader(k, v);
        next();
      });
    },
  };
}

// The bundler emits ORT .wasm variants into dist/assets/ that the wasm-only entry
// never loads (env.wasm.wasmPaths points at /ort/ in production builds). Remove
// them so dist stays small.
function trimDist(): Plugin {
  return {
    name: "trim-dist",
    apply: "build",
    async closeBundle() {
      try {
        const stray = (await readdir("dist/assets")).filter((f) => f.endsWith(".wasm"));
        for (const f of stray) await rm(join("dist/assets", f));
        if (stray.length) console.log(`removed ${stray.length} unused ORT wasm from dist/assets`);
      } catch {
        /* assets dir may not exist */
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), crossOriginIsolation(), trimDist()],
  server: { headers: ISOLATION_HEADERS },
  preview: { headers: ISOLATION_HEADERS },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "iife" },
});
