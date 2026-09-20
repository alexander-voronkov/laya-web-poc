# laya-web-poc

A single-page prototype that runs the [Laya](https://huggingface.co/convaiinnovations/laya) decision model **entirely in the browser** and answers calibrated probability questions about any text you paste.

You provide:

- a **text** (the model's *state*),
- a **task** prompt — how to interpret the text (e.g. "read it as a literary critic"),
- a list of typed **questions**: binary (`noul`), pick-one-of-many (`choice`), or expectation-over-ordered-levels (`score`).

Laya is not a chat model: it reads the state, scores the options you enumerate, and returns one calibrated probability distribution per question in a single forward pass — no sampling, no tokens out. Everything runs locally with `onnxruntime-web` (wasm backend); no data leaves the browser. Model weights (~524 MB) are downloaded once from the Hugging Face CDN and cached by the browser.

There is **no backend**. The draft (text, task, questions, settings) is kept in `localStorage` under a single versioned key; clearing site data resets it to the example.

Built on the [nvkudva/laya-web](https://github.com/nvkudva/laya-web) model port (`src/laya/`: tokenizer port, sequence construction, temperature calibration, ONNX session glue). The checkpoint is [nvkudva/laya-web-q8](https://huggingface.co/nvkudva/laya-web-q8), a q8 weight-only export of the English [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) checkpoint. The UI is original (React + TypeScript + Vite).

Hosted at **https://laya.voronkov.club** (static build under nginx with the isolation headers from the hosting section below).

## The three question types

| UI type | Model type | Options scored | Result |
| --- | --- | --- | --- |
| Бинарный (0–100) | `noul` | `false: …`, `true: …` | p(true) as a percentage + bar |
| Выбор из списка | `choice` | one `[MASK]` per option | full probability distribution, top option highlighted |
| Шкала (ожидание по уровням) | `score` | one `[MASK]` per ordered level | distribution over levels + expected-value marker |

These are the only three primitives the model has; there is no fourth type to add. A binary question also takes optional descriptions of what *yes* and *no* mean — they replace the default wording, cost a handful of tokens, and measurably sharpen the distribution, so the editor nudges you to fill them in.

Every question is encoded byte-exactly as the Python original does:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
```

truncated to `max_len=512` tokens (`head_max_len=192` for the question head; the state keeps its beginning and the tail is cut). The head reads the `[MASK]` positions of the options; the answer is a softmax over the option scores **divided by the fitted temperature** for the (type, option-count) bucket from `rl_agent_config.json`.

### Where the task prompt goes, and why it is a setting

The question head has 192 tokens, shared between the instruction text and all of the option texts, and when it overflows the instruction is cut **from the end**. Put a long task prompt in front of the question and the thing that disappears is the question — silently, with a perfectly plausible probability still coming back.

So the placement is a choice you make in the UI:

- **в формулировку каждого вопроса** — `instructions = <task> <question> <hint>`. Semantically tight, re-encoded per question, and bounded by those 192 tokens.
- **в начало текста (state)** — the framing is prepended to the text instead, where it draws on the ~300 remaining tokens and leaves the question alone.
- **и туда, и туда** — for comparing the two.

Whatever you pick, each question card shows its real token breakdown (question / options / text) and says explicitly when the instruction was clipped or the text truncated. Those numbers come from running the actual sequence builder, not from an estimate.

## Run it

```bash
npm install
npm run dev        # dev server, COOP/COEP headers set by the vite plugin
```

Production build:

```bash
npm run build      # copies the ORT wasm runtime into public/ort/, type-checks, bundles to dist/
npm run preview    # serves dist/ with the same isolation headers
npm test           # token-parity gate, see below
```

`dist/` is fully static — deploy it to any static host. Weights are always fetched from the Hugging Face CDN; override the base URL with `VITE_MODELS_BASE` (see `src/config.ts`).

## Deploy

`.github/workflows/deploy.yml` builds and ships `dist/` to the nginx docroot on every push to `main`, then verifies the live site actually came back with the isolation headers and the ORT runtime — a deploy that does not check is a deploy that goes stale in silence.

It needs four repository secrets, and **fails loudly** rather than skipping if any is missing:

| Secret | Value |
| --- | --- |
| `DEPLOY_HOST` | `152.42.224.40` |
| `DEPLOY_USER` | the rsync user on that host |
| `DEPLOY_SSH_KEY` | private half of a deploy key authorised for that user |
| `DEPLOY_KNOWN_HOSTS` | `ssh-keyscan 152.42.224.40` output — the host key is pinned, because rsync runs with `--delete` |
| `DEPLOY_PATH` | optional; defaults to `/srv/laya-web-poc/site` |

Until those exist, deploys are done by hand on the host (`git fetch` + `npm run build` + rsync into the docroot).

## Hosting requirement: cross-origin isolation

Multi-threaded wasm needs `SharedArrayBuffer`, so the page **must** be served with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

nginx (this is the live config, trimmed):

```nginx
server {
    server_name laya.voronkov.club;
    root /srv/laya-web-poc/site;
    index index.html;

    add_header Cross-Origin-Opener-Policy same-origin always;
    add_header Cross-Origin-Embedder-Policy require-corp always;

    gzip on;
    gzip_types application/wasm application/javascript text/css application/json;
    gzip_min_length 1024;

    location /assets/ { expires 30d; }
    location / { try_files $uri $uri/ /index.html; }
}
```

Notes:

- `require-corp` (not `credentialless`) works with the Hugging Face CDN even though the CDN sends no CORP header: the weights are loaded with `fetch()` in CORS mode, and CORP enforcement only applies to no-cors subresource loads. `credentialless` would look equivalent and silently lose isolation in Safari, which does not implement it.
- The headers live in the `server` block, not in a `location` — an `add_header` inside `location /assets/` would otherwise drop the inherited ones.
- Without these headers the page still works, but wasm falls back to a single thread and inference is roughly 6× slower. The metrics panel says so when it happens rather than leaving you to wonder why it is slow.
- The dev server sets both headers itself — see the `crossOriginIsolation` plugin in `vite.config.ts`.

## Metrics

Per page load and per run:

- model download size, progress and time (471 MB encoder + 53 MB head external data, plus small graph files), and whether it came from cache;
- ONNX session initialisation time, encoder and head separately;
- wasm thread count, `crossOriginIsolated`, and an explicit warning when isolation is missing;
- per-question table: tokens in the sequence, how much of the text survived truncation, encoder time, head time, total, and the temperature bucket actually applied;
- share of wall time spent in the encoder, and encoder throughput in tokens/s;
- token accounting: input tokens, option scores produced, text tokens dropped, and a standing `0` for generated tokens — the model does not generate;
- `performance.memory.usedJSHeapSize` where available (Chromium): the wasm heap lives **inside** this number, so it includes the model;
- `navigator.deviceMemory` and `navigator.storage.estimate()` — whether a 524 MB cache will actually survive on this device;
- weight cache status (`laya-weights-v1` in Cache Storage), including the case where the cache write failed on quota and the next load will download again.

Each answer card also expands into a per-question breakdown, including `act_probability` — shown with the caveat that this head is saturated at 1.000 on this checkpoint and carries no signal.

Every run can be exported as JSON (inputs, the exact Jev-shaped model request, answers, metrics) via «Экспорт JSON прогона».

## Parity check

The ported sequence builder is verified token-for-token (input ids + option marker positions) against the Python golden dump from the reference repo, which is **vendored** at `test/fixtures/parity.json` so the gate cannot quietly skip itself when the network is unavailable:

```bash
npm test    # 24 cases / 26 questions
```

It runs on every push and pull request. This is the load-bearing test in the repository: if the token ids drift by one, nothing downstream fails — the page just answers a different question with the same confidence.

## Model notes and limits

These are properties of the base checkpoint, documented on its model card, not of this prototype:

- **English only.** On Khmer the checkpoint scores 0.000 accuracy at 0.952 mean confidence: it stays confident while being wrong, so a confidence threshold cannot catch it. The UI raises a banner as soon as it sees non-Latin characters in the text or the questions. A multilingual Laya exists, but nobody has published an ONNX/q8 build of it for the browser.
- **The probabilities are not recalibrated.** The temperatures in `rl_agent_config.json` were fitted by the original author on the fp32 model and were not refitted after quantisation. Refitting per (type, option count) on your own labelled data moved mean ECE from 0.466 to 0.081 on the base model. Until that is done, read the numbers as an ordering, not as frequencies.
- **`score` is the weakest primitive** (SST-5 0.372). Where a yes/no question will do, it separates better.
- **High-cardinality `choice` degrades**: the `choice:11+` temperature is 0.1006, which sharpens the distribution close to one-hot, and 192 head tokens leave only a few tokens per label.
- **`act_probability` is saturated** at 1.000 on this checkpoint and carries no signal.
- **Near chance zero-shot on typed decisions** (0.362 against a 0.461 majority-class baseline). Laya is a fast base to specialise, not a zero-shot decision engine.
- q8 weight-only quantisation, **wasm backend only**: WebGPU does not implement 8-bit `MatMulNBits`, so `onnxruntime-web/wasm` is imported instead of the default entry.
- Two ONNX sessions are created once and reused; questions run sequentially, one forward pass (encoder + head) per question, on the main thread. Batching several questions saves the download, not the compute: the state is re-encoded per question, so cost grows linearly.

## Credits & license

- Base model: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0)
- Browser port + quantised weights: [nvkudva/laya-web](https://github.com/nvkudva/laya-web), [nvkudva/laya-web-q8](https://huggingface.co/nvkudva/laya-web-q8)
- This repository: Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
