// The bundler emits the asyncify and jsep wasm variants but not the plain threaded one
// the wasm-only build actually loads, so a production build hangs forever looking for a
// file that was never written. Serve ORT's runtime files ourselves and point env.wasm at
// them. Ported from nvkudva/laya-web, with one change described below.
import { copyFile, mkdir, readdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const dist = dirname(createRequire(import.meta.url).resolve("onnxruntime-web"));
const out = new URL("../public/ort/", import.meta.url).pathname;
await mkdir(out, { recursive: true });

// Only the plain threaded runtime: src/laya/session.ts imports onnxruntime-web/wasm,
// which asks for this pair. Asking for a runtime that is not here surfaces as
// "no available backend found", which names no file -- so assert on the count.
const want = (f) => /^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f);
const files = (await readdir(dist)).filter(want);
if (files.length !== 2) throw new Error(`expected 2 ORT runtime files, found ${files}`);

// The loader is renamed .mjs -> .js on the way out, and session.ts points wasmPaths at
// the renamed file.
//
// nginx's stock mime.types has no entry for .mjs, so it is served as
// application/octet-stream -- and browsers apply a strict MIME check to dynamic
// import(), so the module is rejected. The file is there, 200, correct length; only the
// header is wrong. ORT reports it as "no available backend found", which names neither
// the file nor the reason. This shipped to production and stopped the model loading for
// every visitor; a status-code check passed the whole time.
//
// Fixing the server is the other half and worth doing, but the app should not depend on
// a static host knowing an extension it does not have to know. .js, every host knows.
const rename = (f) => (f.endsWith(".mjs") ? f.replace(/\.mjs$/, ".js") : f);
for (const f of files) await copyFile(join(dist, f), join(out, rename(f)));
console.log(`copied ${files.map((f) => `${f} -> ${rename(f)}`).join(", ")} into public/ort/`);
