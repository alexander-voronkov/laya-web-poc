// The bundler emits the ORT wasm variants into assets/ under hashed names, but
// env.wasm.wasmPaths points at /ort/, so a production build looks for files nobody
// wrote and hangs at session creation. Serve the runtime ourselves.
import { copyFile, mkdir, readdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const dist = dirname(createRequire(import.meta.url).resolve("onnxruntime-web"));
const out = new URL("../public/ort/", import.meta.url).pathname;
await mkdir(out, { recursive: true });

// Two runtimes, because the app has two backends:
//   ort-wasm-simd-threaded       — the wasm entry, used by the int8 and q8 models
//   ort-wasm-simd-threaded.jsep  — the webgpu entry, used by the fp16 model
// Asking for a runtime that is not here surfaces as "no available backend found",
// which names no file, so assert on the count rather than trusting the glob.
const STEMS = ["ort-wasm-simd-threaded", "ort-wasm-simd-threaded.jsep"];
const present = await readdir(dist);

// .mjs -> .js on the way out. nginx's stock mime.types has no entry for .mjs, so it
// goes out as application/octet-stream and the browser's strict MIME check on dynamic
// import() rejects the module. This shipped once and the model loaded for nobody.
const rename = (f) => f.replace(/\.mjs$/, ".js");

for (const stem of STEMS) {
  const files = present.filter((f) => f === `${stem}.wasm` || f === `${stem}.mjs`);
  if (files.length !== 2) {
    throw new Error(`expected ${stem}.{wasm,mjs} in ${dist}, found: ${files.join(", ") || "neither"}`);
  }
  for (const f of files) await copyFile(join(dist, f), join(out, rename(f)));
  console.log(`copied ${stem}.{wasm,mjs} -> public/ort/${stem}.{wasm,js}`);
}
