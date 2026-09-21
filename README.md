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
| Binary (0–100) | `noul` | `false: …`, `true: …` | p(true) as a percentage + bar |
| Pick one | `choice` | one `[MASK]` per option | full probability distribution, top option highlighted |
| Ordered scale (expected level) | `score` | one `[MASK]` per ordered level | distribution over levels + expected-value marker |

These are the only three primitives the model has; there is no fourth type to add. A binary question also takes optional descriptions of what *yes* and *no* mean — they replace the default wording, cost a handful of tokens, and measurably sharpen the distribution, so the editor nudges you to fill them in.

Every question is encoded byte-exactly as the Python original does:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
```

truncated to `max_len=512` tokens (`head_max_len=192` for the question head; the state keeps its beginning and the tail is cut). That 512 is a configuration value, not a property of the graph — the exported encoder declares a dynamic `seq_len` axis and uses rotary embeddings rather than a learned position table, and ModernBERT-large is an 8192-context backbone — so the UI offers larger budgets up to 8192.

The long range was measured rather than assumed, with a text built so the answer depends on its tail: ~650 tokens of unqualified praise for a product, then a closing paragraph retracting all of it. Asked whether the reviewer recommends the product:

| budget | text reaching the model | p(recommends) |
| --- | --- | --- |
| 512 | 477 tokens — the retraction is cut off | **64.3%** |
| 1024 | 653 tokens — the whole review | **3.1%** |

So the extra context is genuinely read, and moves the answer in the right direction. What that does **not** establish is that the probabilities stay calibrated past the length the temperatures were fitted at; only that the model sees the text. Past 512, read a number as an ordering. `scripts/` has no harness for this — it was a one-off, and the text above is enough to repeat it. The head reads the `[MASK]` positions of the options; the answer is a softmax over the option scores **divided by the fitted temperature** for the (type, option-count) bucket from `rl_agent_config.json`.

### Where the task prompt goes, and why it is a setting

The question head has 192 tokens, shared between the instruction text and all of the option texts, and when it overflows the instruction is cut **from the end**. Put a long task prompt in front of the question and the thing that disappears is the question — silently, with a perfectly plausible probability still coming back.

So the placement is a choice you make in the UI:

- **into every question's wording** — `instructions = <task> <question> <hint>`. Semantically tight, re-encoded per question, and bounded by those 192 tokens.
- **in front of the text (state)** — the framing is prepended to the text instead, where it draws on the ~300 remaining tokens and leaves the question alone.
- **both places** — for comparing the two.

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

It needs five repository secrets, and **fails loudly by name** rather than skipping if any is missing — a deploy job that quietly skips itself is indistinguishable from one that worked:

| Secret | Value |
| --- | --- |
| `DEPLOY_HOST` | `152.42.224.40` |
| `DEPLOY_USER` | the rsync user on that host (`laya-deploy`, owns only the docroot, no sudo) |
| `DEPLOY_SSH_KEY` | private half of a deploy key authorised for that user |
| `DEPLOY_KNOWN_HOSTS` | `ssh-keyscan 152.42.224.40` output — the host key is pinned, because rsync runs with `--delete` |
| `DEPLOY_PATH` | the docroot, `/srv/laya-web-poc/site`. Required and asserted absolute: rsync resolves a relative destination against the login user's home, so a mangled path builds a tree under `$HOME` while the real docroot goes on serving the old build |

The build is stamped with the commit SHA into `dist/version.txt`, and the job then checks that the live site serves *that* SHA. Every other assertion it makes — the isolation headers, the wasm runtime — would pass just as well against the previous deploy. Note also that nginx answers `try_files $uri $uri/ /index.html`, so a missing file comes back as 200 with the HTML page: the runtime check asserts `Content-Type: application/wasm`, not the status code.

## Hosting requirement: cross-origin isolation

Multi-threaded wasm needs `SharedArrayBuffer`, so the page **must** be served with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The live config is vendored at [`deploy/nginx-laya.voronkov.club.conf`](deploy/nginx-laya.voronkov.club.conf) — the server is not the only copy. To apply it:

```bash
scp deploy/nginx-laya.voronkov.club.conf user@host:/tmp/laya.conf
ssh user@host 'sudo cp -a /etc/nginx/sites-available/laya.voronkov.club{,.bak-$(date +%F-%H%M%S)} \
  && sudo cp /tmp/laya.conf /etc/nginx/sites-available/laya.voronkov.club \
  && sudo nginx -t && sudo systemctl reload nginx'
```

Notes:

- `require-corp` (not `credentialless`) works with the Hugging Face CDN even though the CDN sends no CORP header: the weights are loaded with `fetch()` in CORS mode, and CORP enforcement only applies to no-cors subresource loads. `credentialless` would look equivalent and silently lose isolation in Safari, which does not implement it.
- The ORT loader is served as `.js`, not `.mjs`, and `wasmPaths` names both runtime files explicitly. nginx's stock `mime.types` has no entry for `.mjs`, so it goes out as `application/octet-stream`; browsers apply a strict MIME check to dynamic `import()` and reject the module. ORT then reports `no available backend found`, naming neither the file nor the reason, while the file itself returns a perfectly healthy 200. This shipped, and the model loaded for nobody. Adding `.mjs` to the server's MIME map fixes it too and is worth doing — but the app no longer depends on the host knowing an extension it need not know.
- The headers live in the `server` block, not in a `location` — an `add_header` inside a location **replaces** the inherited set rather than extending it, so any location adding a header of its own must repeat these two. (`expires` is a different directive and does not have that effect, which is why the cache rules are safe.) The same trap applies to `types`: a `types { application/javascript mjs; }` inside the server block would replace the whole inherited MIME map and send CSS, PNG and WASM out as `application/octet-stream`.
- `/assets/` and `/ort/` use `try_files $uri =404`. Without it the SPA fallback answers **200 with the HTML page** for any missing build artefact, which is how a missing runtime file passed a status-code check while the model loaded for nobody. The deploy job asserts both directions: the real files by content type, and a deliberately absent one by its 404.
- `/assets/` is cached for 30 days because Vite content-hashes those filenames. `/ort/` deliberately is not: `ort-wasm-simd-threaded.{js,wasm}` keep the same names across builds, so a long-lived cached copy would outlive an onnxruntime upgrade and pair a new app with an old runtime.
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
- two memory numbers, because one of them is not the one you want. `performance.memory.usedJSHeapSize` (Chromium) counts the **JavaScript** heap, and the ~600 MB of dequantised weights sit in WebAssembly linear memory outside it. `performance.measureUserAgentSpecificMemory()` does include wasm, and is available here because the page is cross-origin isolated anyway — it sits behind a button, since the browser schedules the measurement when it likes and rate-limits it;
- `navigator.deviceMemory` and `navigator.storage.estimate()` — whether a 524 MB cache will actually survive on this device;
- weight cache status (`laya-weights-v1` in Cache Storage), counted against the four files expected: a `put()` that failed on quota leaves a working session behind a partial cache, and summing the bytes that did land would read as success.

Each answer card also expands into a per-question breakdown, including `act_probability` — shown with the caveat that this head is saturated at 1.000 on this checkpoint and carries no signal.

Every run can be exported as JSON (inputs, the exact Jev-shaped model request, answers, metrics) via "Export the run as JSON".

## Parity check

The ported sequence builder is verified token-for-token (input ids + option marker positions) against the Python golden dump from the reference repo, which is **vendored** at `test/fixtures/parity.json` so the gate cannot quietly skip itself when the network is unavailable:

```bash
npm test    # 24 cases / 26 questions
```

It runs on every push and pull request. This is the load-bearing test in the repository: if the token ids drift by one, nothing downstream fails — the page just answers a different question with the same confidence.

## Browser smoke check

The parity test and CI cover everything up to inference. They cannot cover inference itself: whether 524 MB of weights arrive, whether the wasm session builds, and whether a forward pass returns probabilities. That needs a real browser, so it is a script rather than a CI job — it downloads half a gigabyte and is not worth running on every push.

```bash
npm i -D playwright && npx playwright install chromium
node scripts/smoke.mjs --local                              # local Chromium
PLAYWRIGHT_WS=ws://host:9223/ node scripts/smoke.mjs        # a remote Playwright server
SITE=http://localhost:4173 node scripts/smoke.mjs --local   # against a preview build
```

It reports `crossOriginIsolated`, how long the weights took, the answers with their distributions, the whole metrics panel, the per-question table, and any console error or failed request. Two screenshots land next to it.

## Can any of them batch?

No, and it was measured rather than reasoned about. Laya answers a whole request in one forward pass natively — 39.5 ms for one question against 158.6 ms for ten — and none of the exports here can:

| | verdict | how it fails |
| --- | --- | --- |
| English q8 | **REFUSED** | broadcast error,  — 3 x 153 |
| Specialised q8 | **REFUSED** | broadcast error,  — 8 x 70 |
| Multilingual int8 | **DRIFTS** | accepts the batch, answers differ by up to 21 points |
| Multilingual fp16 | untested | nothing couples the rows in fp16, so it may work |

The two ModernBERT exports bake a batch-1 constant even though the export script declares the batch axis dynamic — and the specialised one was traced with torch dynamo rather than TorchScript, which was the reason to expect it might differ. It did not. The multilingual int8 build does accept a batch, and that is worse: dynamic quantization derives activation scales per tensor, so a question's numbers depend on which others share the pass.

The UI carries a questions-per-pass control anyway, defaulting to 1. It exists so the situation is visible rather than folklore, so a refusal names its cause instead of surfacing a raw broadcast error, and so the day an export can batch, nothing needs rebuilding.  runs on every export and prints REFUSED, DRIFTS or EQUIVALENT.

## Which checkpoint, and what each one costs

Three are selectable. They are not three qualities of the same thing; each trades a different axis, and the numbers below were measured here rather than quoted.

| | English q8 | **Specialised q8** | Multilingual int8 | Multilingual fp16 |
| --- | --- | --- | --- | --- |
| backbone | ModernBERT-large 421M | **ModernBERT-large 421M, fine-tuned** | mmBERT-base 322M | mmBERT-base 322M |
| languages | English; fails on non-Latin script | English; same failure | 100+, cross-lingual works | 100+, cross-lingual works |
| context | 512 trained | 1024 trained | 1024 trained | 1024 trained |
| download | 524 MB | 524 MB | 326 MB | 647 MB |
| runs on | wasm, anywhere | wasm, anywhere | wasm, anywhere | **WebGPU only** |
| speed, one laptop | ~11 s at 512 tokens | unmeasured | **~220 ms at 303 tokens** | unmeasured |
| fidelity vs fp32 | 100% argmax, 1.6 pp worst shift | **100% argmax, 0.9 pp worst shift** | **93.8% argmax, 16.9 pp worst shift** | exact by construction |
| temperatures | fitted | fitted | **none, all 1.0** | none, all 1.0 |
| typed-decisions accuracy | 0.362 | **0.766** | 0.342 | 0.342 |

The fidelity row is the one that is easy to miss. The English build uses *weight-only* quantization: weights are compressed, activations stay fp32, and its published parity against the fp32 model is 100% argmax agreement with a worst probability shift of 0.0158. The multilingual int8 build uses *dynamic* quantization, which derives activation scales from the tensor at run time. Measured here on 16 questions across an English and a Russian text, against that same repository's own fp32 export: 15 of 16 argmax decisions agree, and the worst shift is **16.9 percentage points** — ten times the English build's — with one decision flipped outright. Its publisher's claim of "100% classification accuracy preserved" is a claim about the latency work, not something this measurement supports.

That is also why it is fast. Dynamic quantization runs integer kernels end to end; weight-only dequantizes inside the kernel and computes in fp32, which is most of the twentyfold difference in the table. Faithful, fast, runs anywhere — pick two.

None of this is the dominant term in accuracy. On typed decisions, the shape of task this page is for, the upstream benchmark puts the base checkpoints at 0.362 (English) and 0.342 (multilingual) against a 0.461 majority-class baseline, with random guessing at 0.318 — both below the trivial answer — while the fine-tuned [`laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) scores 0.766. Choosing between the models here is choosing speed and language coverage. It is not choosing accuracy, and no amount of it substitutes for specialising the model on labelled data. `export/` holds the pipeline for building that checkpoint into a browser-ready graph.

## Model notes and limits

These are properties of the base checkpoint, documented on its model card, not of this prototype:

- **English root checkpoint, not an English-only model family.** Laya ships three checkpoints and covers 100+ languages — through [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) (mmBERT-base, 1024 context). The one running in the browser is the English root (ModernBERT-large, 512 context), which the upstream card sums up as "English only on root". It is not useless elsewhere — it clears 3x random in 23 of 51 languages — but it fails specifically on **non-Latin script**, and fails confidently: 0.000 accuracy at 0.952 mean confidence on Khmer, which no confidence threshold can catch. Cyrillic is in that bucket. The UI raises a banner as soon as it sees non-Latin script anywhere in the request. A browser-ready ONNX export of the multilingual checkpoint does exist ([`mizchi/laya-multilingual-onnx`](https://huggingface.co/mizchi/laya-multilingual-onnx), fp16, 647 MB, single graph, onnxruntime-web with WebGPU); wiring it up is open work.
- **The probabilities are not recalibrated.** The temperatures in `rl_agent_config.json` were fitted by the original author on the fp32 model and were not refitted after quantisation. Refitting per (type, option count) on your own labelled data moved mean ECE from 0.466 to 0.081 on the base model. Until that is done, read the numbers as an ordering, not as frequencies.
- **`score` is the weakest primitive** (SST-5 0.372). Where a yes/no question will do, it separates better.
- **High-cardinality `choice` degrades**: the `choice:11+` temperature is 0.1006, which sharpens the distribution close to one-hot, and 192 head tokens leave only a few tokens per label.
- **`act_probability` is saturated** at 1.000 on this checkpoint and carries no signal.
- **Near chance zero-shot on typed decisions** (0.362 against a 0.461 majority-class baseline). Laya is a fast base to specialise, not a zero-shot decision engine.
- q8 weight-only quantisation, **wasm backend only**: WebGPU does not implement 8-bit `MatMulNBits`, so `onnxruntime-web/wasm` is imported instead of the default entry.
- Two ONNX sessions are created once and reused; questions run sequentially, one forward pass (encoder + head) per question, on the main thread.

### Why questions are not batched

Laya answers a whole request in **one** forward pass — the reference `system_one` builds every sequence, pads them with `collate_items` and calls the model once with a batch dimension. That is where its published figures come from: 39.5 ms for one question, 158.6 ms for ten. Running them one at a time, as this port does, forfeits that.

It was implemented and reverted, because **this ONNX export does not support a batch greater than 1**:

```
failed to call OrtRun(). ERROR_CODE: 1 ... element_wise_ops.h:583
Attempting to broadcast an axis by a dimension other than 1. 153 by 459
```

459 is exactly 3 x 153 — three questions of 153 tokens. The export *declares* the batch axis dynamic (`dynamic_axes={"input_ids": {0: "b", ...}}` in the reference `export/export_onnx.py`), but declaring an axis dynamic only renames it: the traced graph still carries shape constants derived from the batch-1 example it was traced with. Checked and ruled out: ModernBERT unpadding, which would have explained it — the graph contains no `cu_seqlens` or `NonZero` ops at all.

So batching needs the model re-exported with a batch>1 example and re-quantised, or a different export. The multilingual build ([`mizchi/laya-multilingual-onnx`](https://huggingface.co/mizchi/laya-multilingual-onnx)) comes from another toolchain and declares `[batch, sequence]` inputs; whether it honours them is untested here.

The lesson is cheap to reuse: a speed change has to be proven not to be an answer change. The comparison harness that caught this ran the same questions with one pass per question and with batching, and diffed the rendered distributions — it failed on the first attempt, before anyone could believe the feature worked.

## Credits & license

- Base model: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0)
- Browser port + quantised weights: [nvkudva/laya-web](https://github.com/nvkudva/laya-web), [nvkudva/laya-web-q8](https://huggingface.co/nvkudva/laya-web-q8)
- This repository: Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
