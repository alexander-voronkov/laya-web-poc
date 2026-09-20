// Adapted from nvkudva/laya-web (app/src/laya/session.ts).
// Differences from the reference: staged loading (tokenizer+config can be loaded
// separately so the UI can count tokens before the 524MB of weights arrive),
// per-stage timing callbacks for the metrics panel, and the thread count is kept
// on the instance. The inference path (systemOne) is a verbatim port.
// wasm-only entry: the default entry drags in the jsep runtime, whose .wasm is 28.3MB.
// We only ever use the wasm EP (WebGPU does not implement 8-bit MatMulNBits anyway).
import * as ort from "onnxruntime-web/wasm";
import { buildSequence, renderOptions, toInternal, type Tok } from "./sequence";
import { formatAnswer, softmax, temperatureFor } from "./postprocess";
import { loadTokenizer } from "./tokenizer";
import { QTYPES, type LayaConfig, type LayaResponse, type Questions, type State } from "./types";

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

/** Bytes the weight cache actually holds -- reported rather than assumed, since a
 *  quota failure on put() leaves a loaded session with nothing cached. */
export async function cachedWeightBytes(): Promise<number> {
  try {
    const cache = await caches.open(CACHE);
    let n = 0;
    for (const k of await cache.keys()) {
      const r = await cache.match(k);
      if (!r) continue;
      n += Number(r.headers.get("content-length") ?? 0) || (await r.blob()).size;
    }
    return n;
  } catch {
    return 0;
  }
}

export async function deleteWeightCache(): Promise<void> {
  try { await caches.delete(CACHE); } catch { /* blocked storage */ }
}

/** Fetch rl_agent_config.json + tokenizer.json -- small enough to drive the live
 *  token counter before any weights are downloaded. */
export async function loadCore(base: string): Promise<Core> {
  const [cfg, tok] = await Promise.all([
    fetch(`${base}/rl_agent_config.json`).then((r) => r.json() as Promise<LayaConfig>),
    loadTokenizer(base),
  ]);
  return { cfg, tok };
}

export class LayaSession {
  readonly cfg: LayaConfig;
  readonly numThreads: number;
  readonly hardwareConcurrency: number;
  private tok: Tok;
  private enc: ort.InferenceSession;
  private head: ort.InferenceSession;

  private constructor(cfg: LayaConfig, tok: Tok, enc: ort.InferenceSession, head: ort.InferenceSession, numThreads: number) {
    this.cfg = cfg;
    this.tok = tok;
    this.enc = enc;
    this.head = head;
    this.numThreads = numThreads;
    this.hardwareConcurrency = navigator.hardwareConcurrency || 1;
  }

  static async load(opts: LoadOptions = {}): Promise<LayaSession> {
    const base = opts.base ?? "/models";
    // wasmPaths is not optional in production: left to the bundler, the build emits the
    // asyncify and jsep variants but not the plain threaded one, and session creation
    // then hangs with no error rather than failing. In dev, ORT resolves the runtime
    // from node_modules, which Vite serves happily.
    if (import.meta.env.PROD) ort.env.wasm.wasmPaths = "/ort/";
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
    const enc = await mk("encoder_q8");
    const head = await mk("head_q8");
    opts.onStage?.("weights", weightsMs);
    return new LayaSession(core.cfg, core.tok, enc, head, threads);
  }

  /** questions: {id: {type, instructions, criteria}} -- the Jev request shape. */
  async systemOne(state: State, questions: Questions): Promise<LayaResponse> {
    const out: LayaResponse = { model: "rl-agent", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    for (const [qid, qdef] of Object.entries(questions)) {
      const q = toInternal(qdef);
      const k = renderOptions(q).length;
      const { ids, markers } = buildSequence(this.tok, state, q, this.cfg.max_len, this.cfg.head_max_len);
      if (markers.length !== k) {
        throw new Error(`question ${JSON.stringify(qid)}: options do not fit in head_max_len=${this.cfg.head_max_len} tokens`);
      }
      const L = ids.length;
      const att = new ort.Tensor("int64", new BigInt64Array(L).fill(1n), [1, L]);
      const { hidden } = await this.enc.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, (x) => BigInt(x)), [1, L]),
        attention_mask: att,
      });
      const r = await this.head.run({
        hidden,
        attention_mask: att,
        marker_pos: new ort.Tensor("int64", BigInt64Array.from(markers, (x) => BigInt(x)), [1, markers.length]),
        marker_mask: new ort.Tensor("bool", new Uint8Array(markers.length).fill(1), [1, markers.length]),
        qtype: new ort.Tensor("int64", BigInt64Array.from([QTYPES[q.t]], (x) => BigInt(x)), [1]),
      });
      const logits = Array.from(r.logits.data as Float32Array).slice(0, k);
      const temp = temperatureFor(this.cfg, QTYPES[q.t], k);
      const p = softmax(logits.map((v) => v / temp));
      const act = softmax(Array.from(r.act_logits.data as Float32Array));
      out.answers[qid] = formatAnswer(q, p, act[0]);
      out.usage.input_tokens += L;
    }
    return out;
  }
}
