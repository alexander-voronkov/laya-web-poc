// Adapted from nvkudva/laya-web (app/src/laya/session.ts).
// Differences from the reference: staged loading (tokenizer+config can be loaded
// separately so the UI can count tokens before the 524MB of weights arrive),
// per-stage timing callbacks for the metrics panel, and the thread count is kept
// on the instance. The inference path (systemOne) is a verbatim port.
// wasm-only entry: the default entry drags in the jsep runtime, whose .wasm is 28.3MB.
// We only ever use the wasm EP (WebGPU does not implement 8-bit MatMulNBits anyway).
import * as ort from "onnxruntime-web/wasm";
import { buildSequence, renderOptions, toInternal, type Tok } from "./sequence";
import { formatAnswer, softmax, tempBucket, temperatureFor } from "./postprocess";
import { loadTokenizer } from "./tokenizer";
import { weightFiles, type ModelSpec } from "../models";
import {
  QTYPES,
  type Answer,
  type LayaConfig,
  type LayaResponse,
  type QuestionTelemetry,
  type Questions,
  type State,
} from "./types";

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
  cached: boolean;
}

/** Fast-to-load part of the model: rl_agent_config.json + tokenizer. */
export interface Core {
  cfg: LayaConfig;
  tok: Tok;
}

export type LoadStage = "core" | "weights" | "encoder-init" | "head-init";

export interface LoadOptions {
  spec: ModelSpec;
  onProgress?: (p: LoadProgress) => void;
  onStage?: (stage: LoadStage, ms: number) => void;
  threads?: number;
  /** Skip the small-file fetches by passing a Core from loadCore(). */
  core?: Core;
}



/** Fetch with progress, backed by the Cache API so a reload does not re-download 524MB. */
async function fetchCached(url: string, cacheName: string, onProgress?: (p: LoadProgress) => void): Promise<Uint8Array> {
  const file = url.split("/").pop()!;
  let cache: Cache | undefined;
  try {
    cache = await caches.open(cacheName);
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onProgress?.({ file, loaded: buf.length, total: buf.length, cached: true });
      return buf;
    }
  } catch {
    // private mode / blocked storage: fall through and fetch without caching
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.({ file, loaded, total, cached: false });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  try { await cache?.put(url, new Response(buf)); } catch { /* quota */ }
  return buf;
}

/** What the weight cache actually holds -- reported rather than assumed, since a
 *  quota failure on put() leaves a loaded session with a partial cache. */
export async function cachedWeights(spec: ModelSpec): Promise<CacheState> {
  const want = weightFiles(spec);
  const empty: CacheState = { bytes: 0, files: 0, expected: want.length, unavailable: false };
  try {
    const cache = await caches.open(spec.cache);
    let bytes = 0;
    let files = 0;
    for (const k of await cache.keys()) {
      const r = await cache.match(k);
      if (!r) continue;
      const name = new URL(k.url).pathname.split("/").pop() ?? "";
      if (want.includes(name)) files++;
      bytes += Number(r.headers.get("content-length") ?? 0) || (await r.blob()).size;
    }
    return { ...empty, bytes, files };
  } catch {
    return { ...empty, unavailable: true };
  }
}

export async function deleteWeightCache(spec: ModelSpec): Promise<void> {
  try { await caches.delete(spec.cache); } catch { /* blocked storage */ }
}

export interface RunOptions {
  /** Checked between questions only -- see the note on systemOne. */
  signal?: AbortSignal;
  /** Fires as each answer lands, so a long batch fills the page progressively. */
  onAnswer?: (qid: string, answer: Answer, telemetry: QuestionTelemetry) => void;
  /** Question ids in the order they should run.
   *
   *  Without it the order comes from `Object.entries`, and JavaScript hoists
   *  integer-like keys to the front in ascending numeric order. Someone who names
   *  their questions "1", "2", "3" and then reorders them in the UI gets a run that
   *  ignores the reordering: answers appear in a different order from the editor and
   *  the progress counter jumps around. Answers stay correctly attributed either way,
   *  but the sequence the user asked for is not the one that happens. */
  order?: string[];
}

export interface RunResult {
  response: LayaResponse;
  telemetry: { questions: QuestionTelemetry[]; totalMs: number };
  /** True when the run stopped early because the signal aborted. */
  aborted: boolean;
}

/** Whether threaded wasm is possible at all here. Both conditions, not just the
 *  isolation flag: a browser can report `crossOriginIsolated` while SharedArrayBuffer
 *  is withheld by policy, and ORT needs the constructor, not the flag. */
export function threadsAvailable(): boolean {
  const isolated = typeof crossOriginIsolated === "boolean" ? crossOriginIsolated : false;
  return isolated && typeof SharedArrayBuffer !== "undefined";
}


export interface CacheState {
  bytes: number;
  /** How many of the model's weight files are present. A put() that failed on quota leaves a
   *  working session behind a partly-filled cache, and summing the bytes that did
   *  land reads as a complete cache -- so the next visit re-downloads a few hundred
   *  megabytes after the page promised it would not. */
  files: number;
  expected: number;
  /** Cache API unavailable (private mode, blocked storage): distinct from an empty cache. */
  unavailable: boolean;
}

/** usedJSHeapSize where the browser exposes it (Chromium), null elsewhere. Reported
 *  as "unavailable" rather than 0, because a 0 reads as "used no memory". */
export function jsHeapBytes(): number | null {
  const m = (performance as unknown as { memory?: { usedJSHeapSize?: number } }).memory;
  return typeof m?.usedJSHeapSize === "number" ? m.usedJSHeapSize : null;
}

/** Fetch rl_agent_config.json + tokenizer.json -- small enough to drive the live
 *  token counter before any weights are downloaded. */
export async function loadCore(spec: ModelSpec): Promise<Core> {
  const base = spec.base;
  const [cfg, tok] = await Promise.all([
    // r.json() on a 404 surfaces as "Unexpected token <", which sends the reader
    // looking for a parser bug instead of a wrong model base URL.
    fetch(`${base}/rl_agent_config.json`).then((r) => {
      if (!r.ok) throw new Error(`${base}/rl_agent_config.json: ${r.status} ${r.statusText}`);
      return r.json() as Promise<LayaConfig>;
    }),
    loadTokenizer(base, spec.tokenizerPath),
  ]);
  return { cfg, tok };
}

export class LayaSession {
  readonly cfg: LayaConfig;
  /** Sequence budget actually used, which may exceed `cfg.max_len`.
   *
   *  The 512 in rl_agent_config.json is a configuration value, not a property of the
   *  graph: the exported encoder declares a dynamic `seq_len` axis and uses rotary
   *  embeddings rather than a learned position table, so there is no length baked into
   *  the weights. ModernBERT-large is a 8192-context backbone.
   *
   *  What 512 *does* mark is the length Laya was trained and temperature-fitted at.
   *  Beyond it the model still runs and the numbers still look like probabilities;
   *  whether they mean anything is unmeasured. The UI labels that, and never raises
   *  this on its own. */
  maxLen: number;
  /** Threads asked for. */
  readonly requestedThreads: number;
  /** Threads the run can actually use.
   *
   *  ORT does not publish the number it settled on -- `env.wasm.numThreads` reads back
   *  whatever we wrote into it -- so this is derived from the one thing that decides it:
   *  threaded wasm needs SharedArrayBuffer, which exists only under cross-origin
   *  isolation. Without it ORT runs single-threaded regardless of the request, and
   *  reporting the request as the outcome puts "8 threads" on a benchmark that ran on
   *  one. Derived, not measured -- and the metrics panel says which. */
  readonly numThreads: number;
  readonly hardwareConcurrency: number;
  readonly spec: ModelSpec;
  private tok: Tok;
  /** : encoder then head. : one graph,  unused. */
  private enc: ort.InferenceSession;
  private head: ort.InferenceSession | null;

  private constructor(spec: ModelSpec, cfg: LayaConfig, tok: Tok, enc: ort.InferenceSession, head: ort.InferenceSession | null, requestedThreads: number) {
    this.spec = spec;
    this.cfg = cfg;
    this.maxLen = cfg.max_len;
    this.tok = tok;
    this.enc = enc;
    this.head = head;
    this.requestedThreads = requestedThreads;
    this.numThreads = threadsAvailable() ? requestedThreads : 1;
    this.hardwareConcurrency = navigator.hardwareConcurrency || 1;
  }

  static async load(opts: LoadOptions): Promise<LayaSession> {
    const spec = opts.spec;
    const base = spec.base;
    // wasmPaths is not optional in production: left to the bundler, the build emits the
    // asyncify and jsep variants but not the plain threaded one, and session creation
    // then hangs with no error rather than failing. In dev, ORT resolves the runtime
    // from node_modules, which Vite serves happily.
    // Explicit file paths, not a prefix. A prefix makes ORT ask for
    // `ort-wasm-simd-threaded.mjs`, and nginx's stock mime.types has no entry for
    // .mjs: it comes back as application/octet-stream, the browser's strict MIME
    // check on dynamic import() rejects it, and ORT reports "no available backend
    // found" -- naming neither the file nor the reason. scripts/copy-ort.mjs writes
    // the loader as .js for exactly this, so the app does not depend on the host
    // knowing an extension it need not know.
    if (import.meta.env.PROD) {
      ort.env.wasm.wasmPaths = {
        wasm: "/ort/ort-wasm-simd-threaded.wasm",
        mjs: "/ort/ort-wasm-simd-threaded.js",
      };
    }
    const threads = opts.threads ?? Math.min(navigator.hardwareConcurrency || 4, 8);
    ort.env.wasm.numThreads = threads;
    // Main thread, no proxy: threaded wasm initialises only there in a production bundle
    // (both a user-created worker and ORT's own env.wasm.proxy hang with no error after
    // the weights load). Inference blocks the UI for the length of one forward pass.
    let t0 = performance.now();
    const core = opts.core ?? await loadCore(spec);
    opts.onStage?.("core", performance.now() - t0);

    let weightsMs = 0;
    const mk = async (g: { name: string; externalData: boolean }, stage: LoadStage): Promise<ort.InferenceSession> => {
      let t = performance.now();
      const graph = await fetchCached(`${base}/${g.name}.onnx`, spec.cache, opts.onProgress);
      // The English build keeps its weights in a sibling .onnx.data; the multilingual
      // one carries them inline, and asking for a .data that does not exist would 404.
      const data = g.externalData
        ? await fetchCached(`${base}/${g.name}.onnx.data`, spec.cache, opts.onProgress)
        : null;
      weightsMs += performance.now() - t;
      t = performance.now();
      const s = await ort.InferenceSession.create(graph, {
        executionProviders: ["wasm"],
        ...(data ? { externalData: [{ data, path: `${g.name}.onnx.data` }] } : {}),
      });
      opts.onStage?.(stage, performance.now() - t);
      return s;
    };

    // Releasing on the way out matters more than it looks. The first session holds
    // hundreds of megabytes of wasm linear memory; if the second one's weights then
    // fail mid-download, dropping the JS reference does not free it -- only release()
    // does. A retry would stack a second encoder on the first, which is precisely the
    // out-of-memory this module's boot singleton exists to prevent.
    let partial: ort.InferenceSession | undefined;
    let enc: ort.InferenceSession;
    let head: ort.InferenceSession | null = null;
    try {
      enc = partial = await mk(spec.graphs[0], "encoder-init");
      if (spec.layout === "split") head = await mk(spec.graphs[1], "head-init");
    } catch (e) {
      await Promise.allSettled([partial?.release()]);
      throw e;
    }
    opts.onStage?.("weights", weightsMs);
    return new LayaSession(spec, core.cfg, core.tok, enc, head, threads);
  }

  /** questions: {id: {type, instructions, criteria}} -- the Jev request shape.
   *
   *  One shared state, N independent questions. Batching saves the 524MB download,
   *  not the forward passes: the head scores option markers inside the same sequence,
   *  so the state is re-encoded per question and cost grows linearly with N.
   *
   *  laya-web-poc: returns per-question telemetry alongside the answers, streams each
   *  answer through `onAnswer` as it lands, and honours an AbortSignal *between*
   *  questions. Deliberately not during one: a forward pass already inside wasm
   *  cannot be cancelled, and racing a timeout against it would abandon a session
   *  that keeps running and keeps holding its 524MB.
   */
  async systemOne(state: State, questions: Questions, opts: RunOptions = {}): Promise<RunResult> {
    const out: LayaResponse = { model: this.spec.id, answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    const perQuestion: QuestionTelemetry[] = [];
    const t0 = performance.now();
    let aborted = false;

    const order = opts.order?.filter((id) => id in questions) ?? Object.keys(questions);
    for (const qid of order) {
      if (opts.signal?.aborted) { aborted = true; break; }
      const qdef = questions[qid];
      const qt0 = performance.now();
      const q = toInternal(qdef);
      const k = renderOptions(q).length;
      const { ids, markers, stats } = buildSequence(this.tok, state, q, this.maxLen, this.cfg.head_max_len);
      if (markers.length !== k) {
        throw new Error(`question ${JSON.stringify(qid)}: options do not fit in head_max_len=${this.cfg.head_max_len} tokens`);
      }
      const buildMs = performance.now() - qt0;
      const L = ids.length;
      const att = new ort.Tensor("int64", new BigInt64Array(L).fill(1n), [1, L]);
      const inputIds = new ort.Tensor("int64", BigInt64Array.from(ids, (x) => BigInt(x)), [1, L]);
      const markerPos = new ort.Tensor("int64", BigInt64Array.from(markers, (x) => BigInt(x)), [1, markers.length]);
      const markerMask = new ort.Tensor("bool", new Uint8Array(markers.length).fill(1), [1, markers.length]);
      const qtypeT = new ort.Tensor("int64", BigInt64Array.from([QTYPES[q.t]], (x) => BigInt(x)), [1]);

      // Two shapes of the same model. The English build ships the encoder and the
      // decision head as separate graphs, so the hidden state crosses between two
      // sessions; the multilingual build carries encoder, head, scorer and act head in
      // one graph and takes every input at once. Where that leaves the split timings
      // undefined, they are reported as such rather than invented.
      let r: ort.InferenceSession.OnnxValueMapType;
      let encoderMs: number;
      let headMs: number;
      if (this.head) {
        const e0 = performance.now();
        const { hidden } = await this.enc.run({ input_ids: inputIds, attention_mask: att });
        encoderMs = performance.now() - e0;
        const h0 = performance.now();
        r = await this.head.run({ hidden, attention_mask: att, marker_pos: markerPos, marker_mask: markerMask, qtype: qtypeT });
        headMs = performance.now() - h0;
      } else {
        const e0 = performance.now();
        r = await this.enc.run({
          input_ids: inputIds,
          attention_mask: att,
          marker_pos: markerPos,
          marker_mask: markerMask,
          qtype: qtypeT,
        });
        encoderMs = performance.now() - e0;
        headMs = 0;
      }

      const logits = Array.from(r.logits.data as Float32Array).slice(0, k);
      const temperature = temperatureFor(this.cfg, QTYPES[q.t], k);
      const p = softmax(logits.map((v) => v / temperature));
      const act = softmax(Array.from(r.act_logits.data as Float32Array));
      const answer = formatAnswer(q, p, act[0]);
      out.answers[qid] = answer;
      out.usage.input_tokens += L;

      const tel: QuestionTelemetry = {
        qid,
        type: q.t,
        options: k,
        temperatureBucket: tempBucket(QTYPES[q.t], k),
        temperature,
        stats,
        buildMs,
        encoderMs,
        headMs,
        totalMs: performance.now() - qt0,
      };
      perQuestion.push(tel);
      opts.onAnswer?.(qid, answer, tel);
      // Inference runs on the main thread (see load()), so without a yield the page
      // never repaints and a six-question batch is indistinguishable from a hang.
      await new Promise((res) => setTimeout(res, 0));
    }

    return { response: out, aborted, telemetry: { questions: perQuestion, totalMs: performance.now() - t0 } };
  }
}
