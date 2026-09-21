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
  base?: string;
  onProgress?: (p: LoadProgress) => void;
  onStage?: (stage: LoadStage, ms: number) => void;
  threads?: number;
  /** Skip the small-file fetches by passing a Core from loadCore(). */
  core?: Core;
}

const CACHE = "laya-weights-v1";

/** Fetch with progress, backed by the Cache API so a reload does not re-download 524MB. */
async function fetchCached(url: string, onProgress?: (p: LoadProgress) => void): Promise<Uint8Array> {
  const file = url.split("/").pop()!;
  let cache: Cache | undefined;
  try {
    cache = await caches.open(CACHE);
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
export async function cachedWeights(): Promise<CacheState> {
  const empty: CacheState = { bytes: 0, files: 0, expected: WEIGHT_FILES.length, unavailable: false };
  try {
    const cache = await caches.open(CACHE);
    let bytes = 0;
    let files = 0;
    for (const k of await cache.keys()) {
      const r = await cache.match(k);
      if (!r) continue;
      const name = new URL(k.url).pathname.split("/").pop() ?? "";
      if (WEIGHT_FILES.includes(name)) files++;
      bytes += Number(r.headers.get("content-length") ?? 0) || (await r.blob()).size;
    }
    return { ...empty, bytes, files };
  } catch {
    return { ...empty, unavailable: true };
  }
}

export async function deleteWeightCache(): Promise<void> {
  try { await caches.delete(CACHE); } catch { /* blocked storage */ }
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
  /** Padded-token ceiling per batch. Questions are grouped up to it, so a request
   *  costs one forward pass per group rather than one per question. */
  batchTokens?: number;
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

/** The weight files a complete cache is expected to hold. */
export const WEIGHT_FILES = ["encoder_q8.onnx", "encoder_q8.onnx.data", "head_q8.onnx", "head_q8.onnx.data"];

export interface CacheState {
  bytes: number;
  /** How many of WEIGHT_FILES are present. A put() that failed on quota leaves a
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
export async function loadCore(base: string): Promise<Core> {
  const [cfg, tok] = await Promise.all([
    // r.json() on a 404 surfaces as "Unexpected token <", which sends the reader
    // looking for a parser bug instead of a wrong VITE_MODELS_BASE.
    fetch(`${base}/rl_agent_config.json`).then((r) => {
      if (!r.ok) throw new Error(`${base}/rl_agent_config.json: ${r.status} ${r.statusText}`);
      return r.json() as Promise<LayaConfig>;
    }),
    loadTokenizer(base),
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
  private tok: Tok;
  private enc: ort.InferenceSession;
  private head: ort.InferenceSession;

  private constructor(cfg: LayaConfig, tok: Tok, enc: ort.InferenceSession, head: ort.InferenceSession, requestedThreads: number) {
    this.cfg = cfg;
    this.maxLen = cfg.max_len;
    this.tok = tok;
    this.enc = enc;
    this.head = head;
    this.requestedThreads = requestedThreads;
    this.numThreads = threadsAvailable() ? requestedThreads : 1;
    this.hardwareConcurrency = navigator.hardwareConcurrency || 1;
  }

  static async load(opts: LoadOptions = {}): Promise<LayaSession> {
    const base = opts.base ?? "/models";
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
    const core = opts.core ?? await loadCore(base);
    opts.onStage?.("core", performance.now() - t0);

    let weightsMs = 0;
    const mk = async (name: string): Promise<ort.InferenceSession> => {
      let t = performance.now();
      const [graph, data] = [
        await fetchCached(`${base}/${name}.onnx`, opts.onProgress),
        await fetchCached(`${base}/${name}.onnx.data`, opts.onProgress),
      ];
      weightsMs += performance.now() - t;
      t = performance.now();
      const s = await ort.InferenceSession.create(graph, {
        executionProviders: ["wasm"],
        externalData: [{ data, path: `${name}.onnx.data` }],
      });
      opts.onStage?.(name === "encoder_q8" ? "encoder-init" : "head-init", performance.now() - t);
      return s;
    };
    // Releasing on the way out matters more than it looks. The encoder session is
    // built first and holds ~470MB of wasm linear memory; if the head's weights then
    // fail mid-download, dropping the JS reference does not free it -- only
    // release() does. A retry would stack a second encoder on top of the first, which
    // is precisely the out-of-memory this module's boot singleton exists to prevent.
    let partial: ort.InferenceSession | undefined;
    let enc: ort.InferenceSession;
    let head: ort.InferenceSession;
    try {
      enc = partial = await mk("encoder_q8");
      head = await mk("head_q8");
    } catch (e) {
      await Promise.allSettled([partial?.release()]);
      throw e;
    }
    opts.onStage?.("weights", weightsMs);
    return new LayaSession(core.cfg, core.tok, enc, head, threads);
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
    const out: LayaResponse = { model: "laya-web-q8", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    const perQuestion: QuestionTelemetry[] = [];
    const t0 = performance.now();
    let aborted = false;

    const order = opts.order?.filter((id) => id in questions) ?? Object.keys(questions);

    // Build every sequence first. This is pure tokenization -- no forward pass -- so
    // it also surfaces an impossible question before any compute is spent.
    const built = order.map((qid) => {
      const b0 = performance.now();
      const q = toInternal(questions[qid]);
      const k = renderOptions(q).length;
      const { ids, markers, stats } = buildSequence(this.tok, state, q, this.maxLen, this.cfg.head_max_len);
      if (markers.length !== k) {
        throw new Error(`question ${JSON.stringify(qid)}: options do not fit in head_max_len=${this.cfg.head_max_len} tokens`);
      }
      return { qid, q, k, ids, markers, stats, buildMs: performance.now() - b0 };
    });

    for (const group of this.groupIntoBatches(built, opts.batchTokens)) {
      if (opts.signal?.aborted) { aborted = true; break; }

      // Pad to the group's longest sequence and widest marker list, exactly as the
      // Python reference's collate_items does: pad ids with the pad token and mark
      // them 0 in the attention mask; pad marker_pos with 0 and marker_mask with
      // false, so position 0 ([CLS]) is written but never read.
      const n = group.length;
      const L = Math.max(...group.map((g) => g.ids.length));
      const kmax = Math.max(...group.map((g) => g.markers.length));

      const ids = new BigInt64Array(n * L).fill(BigInt(this.tok.padTokenId));
      const att = new BigInt64Array(n * L);
      const mpos = new BigInt64Array(n * kmax);
      const mmask = new Uint8Array(n * kmax);
      const qtype = new BigInt64Array(n);
      group.forEach((g, r) => {
        for (let i = 0; i < g.ids.length; i++) {
          ids[r * L + i] = BigInt(g.ids[i]);
          att[r * L + i] = 1n;
        }
        for (let i = 0; i < g.markers.length; i++) {
          mpos[r * kmax + i] = BigInt(g.markers[i]);
          mmask[r * kmax + i] = 1;
        }
        qtype[r] = BigInt(QTYPES[g.q.t]);
      });

      const attT = new ort.Tensor("int64", att, [n, L]);
      const e0 = performance.now();
      const { hidden } = await this.enc.run({
        input_ids: new ort.Tensor("int64", ids, [n, L]),
        attention_mask: attT,
      });
      const encoderMs = performance.now() - e0;

      const h0 = performance.now();
      const r = await this.head.run({
        hidden,
        attention_mask: attT,
        marker_pos: new ort.Tensor("int64", mpos, [n, kmax]),
        marker_mask: new ort.Tensor("bool", mmask, [n, kmax]),
        qtype: new ort.Tensor("int64", qtype, [n]),
      });
      const headMs = performance.now() - h0;

      // Read the row widths off the tensors rather than assuming kmax: the head is
      // free to return a wider logits tensor than the markers we supplied.
      const logits = r.logits.data as Float32Array;
      const logitsRow = Number(r.logits.dims[1] ?? kmax);
      const actAll = r.act_logits.data as Float32Array;
      const actRow = Number(r.act_logits.dims[1] ?? actAll.length / n);

      group.forEach((g, row) => {
        const z: number[] = [];
        for (let i = 0; i < g.k; i++) z.push(logits[row * logitsRow + i]);
        const temperature = temperatureFor(this.cfg, QTYPES[g.q.t], g.k);
        const p = softmax(z.map((v) => v / temperature));
        const act = softmax(Array.from(actAll.slice(row * actRow, (row + 1) * actRow)));
        const answer = formatAnswer(g.q, p, act[0]);
        out.answers[g.qid] = answer;
        // Real tokens, not padding -- the reference reports attention_mask.sum().
        out.usage.input_tokens += g.ids.length;

        const tel: QuestionTelemetry = {
          qid: g.qid,
          type: g.q.t,
          options: g.k,
          temperatureBucket: tempBucket(QTYPES[g.q.t], g.k),
          temperature,
          stats: g.stats,
          buildMs: g.buildMs,
          // Timings belong to the batch, not to this question: one forward pass
          // produced all `batchSize` of them, and dividing it up would invent a
          // per-question number that was never measured.
          batchSize: n,
          encoderMs,
          headMs,
          totalMs: g.buildMs + encoderMs + headMs,
        };
        perQuestion.push(tel);
        opts.onAnswer?.(g.qid, answer, tel);
      });

      // Inference runs on the main thread (see load()), so without a yield the page
      // never repaints and a long run is indistinguishable from a hang.
      await new Promise((res) => setTimeout(res, 0));
    }

    return { response: out, aborted, telemetry: { questions: perQuestion, totalMs: performance.now() - t0 } };
  }

  /** Split questions into batches the runtime can hold.
   *
   *  Padding waste is not a concern here and the grouping does not try to avoid it:
   *  every question in a request shares one state, so their sequences differ only by
   *  the question head and come out within a few tokens of each other.
   *
   *  What does matter is memory. The encoder materialises a hidden state of
   *  `n x L x 1024` floats, which at L=4096 is 16 MB per question on top of the
   *  attention intermediates, and this runs on the main thread of a browser tab that
   *  is already holding 524 MB of weights. So batches are bounded by padded tokens
   *  rather than by count, and a single question always forms a batch of its own
   *  rather than being refused for being too long. */
  private groupIntoBatches<T extends { ids: number[] }>(items: T[], budget = 8192): T[][] {
    const groups: T[][] = [];
    let current: T[] = [];
    let width = 0;
    for (const it of items) {
      const w = Math.max(width, it.ids.length);
      if (current.length && w * (current.length + 1) > budget) {
        groups.push(current);
        current = [];
        width = 0;
      }
      current.push(it);
      width = Math.max(width, it.ids.length);
    }
    if (current.length) groups.push(current);
    return groups;
  }
}
