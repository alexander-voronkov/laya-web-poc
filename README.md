# laya-web-poc

A single-page prototype that runs the [Laya](https://huggingface.co/convaiinnovations/laya) decision model **entirely in the browser** and answers calibrated probability questions about any text you paste.

You provide:

- a **text** (the model's *state*),
- a **task** prompt — how to interpret the text (e.g. "read it as a literary critic"),
- a list of typed **questions**: binary (`noul`), pick-one-of-many (`choice`), or expectation-over-ordered-levels (`score`).

Laya is not a chat model: it reads the state, scores the options you enumerate, and returns one calibrated probability distribution per question in a single forward pass — no sampling, no tokens out. Everything runs locally with `onnxruntime-web` (wasm backend); no data leaves the browser. Model weights (~524 MB) are downloaded once from the Hugging Face CDN and cached by the browser.

Built on the [nvkudva/laya-web](https://github.com/nvkudva/laya-web) model port (`src/laya/`: tokenizer port, sequence construction, temperature calibration, ONNX session glue). The checkpoint is [nvkudva/laya-web-q8](https://huggingface.co/nvkudva/laya-web-q8), a q8 weight-only export of the English [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) checkpoint. The UI here is original (plain TypeScript + DOM, no framework).

Hosted at **https://laya.voronkov.club** (static build under nginx with the
isolation headers from the hosting section below).

## The three question types

| UI type | Model type | Options scored | Result |
| --- | --- | --- | --- |
| Бинарный (0–100) | `noul` | `false: …`, `true: …` | p(true) as a percentage + bar |
| Выбор из списка | `choice` | one `[MASK]` per option | full probability distribution, top option highlighted |
| Шкала (ожидание по уровням) | `score` | one `[MASK]` per ordered level | distribution over levels + expected-value marker |

Every question is encoded byte-exactly as the Python original does:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
```

truncated to `max_len=512` tokens (`head_max_len=192` for the question head; the state keeps its beginning and the tail is cut). The head reads the `[MASK]` positions of the options; the answer is a softmax over the option scores **divided by the fitted temperature** for the (type, option-count) bucket from `rl_agent_config.json`.

The global task prompt and the optional per-question hint are joined into `<instructions>` (task → question → hint), so they frame every question the same way.

## Run it

```bash
npm install
npm run dev        # dev server, COOP/COEP headers set by the vite plugin
```

Production build:

```bash
npm run build      # copies the ORT wasm runtime into public/ort/, type-checks, bundles to dist/
npm run preview    # serves dist/ with the same isolation headers
```

`dist/` is fully static — deploy it to any static host. Weights are always fetched from the Hugging Face CDN; override the base URL with `VITE_MODELS_BASE` (see `src/config.ts`).

## Hosting requirement: cross-origin isolation

Multi-threaded wasm needs `SharedArrayBuffer`, so the page **must** be served with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

nginx:

```nginx
server {
    listen 443 ssl;
    server_name laya.example.org;

    location / {
        root /var/www/laya-web-poc;   # contents of dist/
        add_header Cross-Origin-Opener-Policy same-origin always;
        add_header Cross-Origin-Embedder-Policy require-corp always;
        try_files $uri $uri/ /index.html;
    }
}
```

Notes:

- `require-corp` (not `credentialless`) works with the Hugging Face CDN even though the CDN sends no CORP header: the weights are loaded with `fetch()` in CORS mode, and CORP enforcement only applies to no-cors subresource loads.
- Without these headers the page still works, but wasm falls back to a single thread and inference is several times slower.
- The dev server sets both headers itself — see the `crossOriginIsolation` plugin in `vite.config.ts`.

## Metrics

The metrics panel shows, per page load / per run:

- model download size, progress and time (471 MB encoder + 53 MB head external data, plus small graph files);
- ONNX session initialisation time (encoder / head separately);
- per-question inference time and the total run time;
- input token counts (real tokenizer counts) and the number of option scores produced;
- detected wasm thread count and `crossOriginIsolated` status;
- `performance.memory.usedJSHeapSize` where available (Chrome): the wasm heap lives **inside** the JS heap, so this number includes the model runtime;
- model cache status: weight files are stored in the browser's Cache Storage (`laya-weights-v1`), so the second load is nearly instant.

Every run can be exported as JSON (inputs, the exact Jev-shaped model request, answers, metrics) via the «Экспорт JSON» button.

## Parity check

The ported sequence builder is verified token-for-token (input ids + option marker positions) against the Python golden dump shipped with the reference repo:

```bash
curl -o /tmp/parity.json \
  https://raw.githubusercontent.com/nvkudva/laya-web/main/app/public/parity.json
npx esbuild scripts/parity-check.ts --bundle --platform=browser \
  --format=esm --external:node:* --outfile=/tmp/pc.mjs
node /tmp/pc.mjs /tmp/parity.json   # -> ALL 26 cases PASS
```

## Model notes

- English-only checkpoint — the UI is Russian, but English text works best (there is a hint in the UI).
- q8 weight-only quantisation, **wasm backend only**: WebGPU does not implement 8-bit `MatMulNBits`, so `onnxruntime-web/wasm` is imported instead of the default entry.
- Two ONNX sessions are created once and reused; questions run sequentially, one forward pass (encoder + head) per question, on the main thread.

## Credits & license

- Base model: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0)
- Browser port + quantised weights: [nvkudva/laya-web](https://github.com/nvkudva/laya-web), [nvkudva/laya-web-q8](https://huggingface.co/nvkudva/laya-web-q8)
- This repository: Apache-2.0 — see [LICENSE](LICENSE).
