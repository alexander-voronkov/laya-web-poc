import { useEffect, useState, type ReactNode } from "react";
import { NOMINAL_TOTAL_BYTES } from "../config";
import { jsHeapBytes, type CacheState, type LayaSession, type LoadProgress, type LoadStage } from "../laya/session";
import type { QuestionTelemetry } from "../laya/types";
import { mb, ms, sec } from "../format";
import { MemoryProbe } from "./MemoryProbe";

interface RunSummary {
  questions: QuestionTelemetry[];
  totalMs: number;
  aborted: boolean;
  requested: number;
}

interface Props {
  files: LoadProgress[];
  stages: Partial<Record<LoadStage, number>>;
  cache: CacheState | null;
  session: LayaSession | null;
  ready: boolean;
  run: RunSummary | null;
}

interface StorageState {
  usage: number | null;
  quota: number | null;
  available: boolean;
}

/** Quota tells the user whether a 524MB cache can survive; usage tells them what is
 *  already spent. Both are estimates by specification -- labelled as such. */
function useStorageEstimate(ready: boolean): StorageState {
  const [s, setS] = useState<StorageState>({ usage: null, quota: null, available: false });
  useEffect(() => {
    if (!ready) return;
    let alive = true;
    navigator.storage?.estimate?.().then(
      (e) => { if (alive) setS({ usage: e.usage ?? null, quota: e.quota ?? null, available: true }); },
      () => { /* blocked by policy, or not implemented */ },
    );
    return () => { alive = false; };
  }, [ready]);
  return s;
}

export function MetricsPanel({ files, stages, cache, session, ready, run }: Props) {
  const storage = useStorageEstimate(ready);
  const heap = jsHeapBytes();

  const loaded = files.reduce((a, f) => a + f.loaded, 0);
  const total = files.reduce((a, f) => a + (f.total || 0), 0);
  const cachedAny = files.some((f) => f.cached);
  const weightsMs = stages.weights ?? 0;
  const encInit = stages["encoder-init"] ?? 0;
  const headInit = stages["head-init"] ?? 0;

  const qs = run?.questions ?? [];
  const inputTokens = qs.reduce((a, q) => a + q.stats.totalTokens, 0);
  const scored = qs.reduce((a, q) => a + q.options, 0);
  // Divided by batchSize on purpose: every question in a batch reports the same
  // forward-pass time, so a plain sum counts one pass once per question and a batch of
  // three would claim three times the compute that happened.
  const encoderMs = qs.reduce((a, q) => a + q.encoderMs / q.batchSize, 0);
  // Each batch contributes batchSize rows of 1/batchSize, so this is exactly the
  // number of forward passes, without needing to identify the groups.
  const batches = Math.round(qs.reduce((a, q) => a + 1 / q.batchSize, 0));
  const droppedTokens = qs.reduce((a, q) => a + (q.stats.stateTokens - q.stats.stateTokensUsed), 0);
  const totalMs = run?.totalMs ?? 0;

  const cores = navigator.hardwareConcurrency;
  const deviceMemory = (navigator as unknown as { deviceMemory?: number }).deviceMemory;
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  const threadsRequested = session?.requestedThreads ?? null;
  const threadsEffective = session?.numThreads ?? null;

  const rows: [string, ReactNode][] = [
    [
      "Weight download",
      ready || loaded
        ? `${mb(loaded)} MB${total ? ` of ${mb(total)} MB` : ""} · ${sec(weightsMs)}${cachedAny ? " · from the browser cache" : ""}`
        : `~${mb(NOMINAL_TOTAL_BYTES)} MB, downloading…`,
    ],
    [
      "ORT session init",
      encInit || headInit ? `${sec(encInit + headInit)} (encoder ${sec(encInit)} · head ${sec(headInit)})` : "—",
    ],
    [
      "Wasm threads",
      threadsEffective === null
        ? "—"
        : threadsEffective === threadsRequested
          ? `${threadsEffective} of ${cores ?? "?"} cores · crossOriginIsolated: ✓`
          : `${threadsEffective} (requested ${threadsRequested}, ${cores ?? "?"} cores) · crossOriginIsolated: ✗`,
    ],
    ...(ready && !isolated
      ? ([[
          "⚠ Isolation",
          "no cross-origin isolation — SharedArrayBuffer is unavailable, so wasm runs on one thread and takes roughly 6x longer. The thread count above is derived from that fact, not measured: ORT does not report how many threads it actually used. Check the COOP/COEP headers.",
        ]] as [string, ReactNode][])
      : []),
    [
      "Run time",
      qs.length
        ? `${sec(totalMs)} for ${qs.length} question${qs.length === 1 ? "" : "s"}${run && run.aborted ? ` of ${run.requested} (stopped)` : ""} · ${ms(totalMs / qs.length)} on average · ${((encoderMs / totalMs) * 100).toFixed(0)}% of it in the encoder`
        : "—",
    ],
    [
      "Tokens",
      qs.length
        ? `${inputTokens} in · ${scored} options scored${droppedTokens ? ` · ${droppedTokens} dropped from the text` : ""} · 0 generated (the model does not generate)`
        : "—",
    ],
    [
      "Forward passes",
      qs.length
        ? `${batches} for ${qs.length} question${qs.length === 1 ? "" : "s"}${batches < qs.length ? ` — batched, ${(qs.length / batches).toFixed(1)} per pass` : " — one per question"}`
        : "—",
    ],
    [
      "Throughput",
      qs.length ? `${(inputTokens / (encoderMs / 1000)).toFixed(0)} tokens/s through the encoder` : "—",
    ],
    [
      "Memory (JS heap)",
      // usedJSHeapSize measures the V8 heap. WebAssembly linear memory -- where the
      // ~600MB of dequantised weights actually live -- is a separate backing store and
      // is NOT counted here. Saying otherwise invites reading "142 MB" as the model
      // fitting in 142 MB.
      heap !== null
        ? `${mb(heap)} MB — this is the JavaScript heap; the wasm linear memory holding the weights is not part of it`
        : "performance.memory is unavailable in this browser (Chromium only)",
    ],
    ["Process memory (incl. wasm)", <MemoryProbe key="mem" />],
    [
      "Device memory",
      deviceMemory ? `${deviceMemory} GB (browser-rounded)` : "navigator.deviceMemory is unavailable",
    ],
    ["Weight cache", <CacheLine cache={cache} ready={ready} key="cache" />],
    [
      "Storage quota",
      !storage.available
        ? "navigator.storage.estimate() is unavailable"
        : storage.quota === null
          ? "the browser does not report a quota"
          : `${storage.usage === null ? "usage unknown" : `${mb(storage.usage)} MB used`} of ~${(storage.quota / 1e9).toFixed(1)} GB (browser estimate)`,
    ],
  ];

  return (
    <>
      <div className="metrics-rows">
        {rows.map(([k, v]) => (
          <div className="m-row" key={k}><span className="m-k">{k}</span><span className="m-v">{v}</span></div>
        ))}
      </div>
      {qs.length > 0 && <PerQuestionTable qs={qs} />}
    </>
  );
}

/** Partial is its own state. Summing whatever landed in the cache and calling it
 *  cached is how the page promises an instant second load and then downloads a few
 *  hundred megabytes: a put() that failed on quota leaves a working session behind an
 *  incomplete cache, and the bytes that did land look like success. */
function CacheLine({ cache, ready }: { cache: CacheState | null; ready: boolean }) {
  if (!cache) return <>{ready ? "checking…" : "—"}</>;
  if (cache.unavailable)
    return (
      <span className="warn">
        Cache Storage is unavailable (private mode, or site data blocked) — the weights will be
        downloaded on every visit
      </span>
    );
  if (cache.files === 0)
    return (
      <span className="warn">
        the weights were not cached (quota, or storage blocked) — they will download again on reload
      </span>
    );
  if (cache.files < cache.expected)
    return (
      <span className="warn">
        {cache.files} of {cache.expected} files cached ({mb(cache.bytes)} MB) — most likely the quota
        ran out; the rest will download again
      </span>
    );
  return <>{mb(cache.bytes)} MB, all {cache.expected} files — the next visit downloads nothing</>;
}

function PerQuestionTable({ qs }: { qs: QuestionTelemetry[] }) {
  return (
    <div className="table-wrap">
      <table className="metrics-table">
        <thead>
          <tr>
            <th>question</th><th>type</th><th>opts</th><th>tokens</th>
            <th>text</th><th>batch</th><th>encoder</th><th>head</th><th>temperature</th>
          </tr>
        </thead>
        <tbody>
          {qs.map((q) => (
            <tr key={q.qid}>
              <td>{q.qid}</td>
              <td>{q.type}</td>
              <td>{q.options}</td>
              <td>{q.stats.totalTokens}</td>
              <td className={q.stats.stateTokensUsed < q.stats.stateTokens ? "warn" : undefined}>
                {q.stats.stateTokensUsed}
                {q.stats.stateTokensUsed < q.stats.stateTokens && `/${q.stats.stateTokens}`}
              </td>
              <td>{q.batchSize}</td>
              <td>{ms(q.encoderMs)}</td>
              <td>{ms(q.headMs)}</td>
              <td>{q.temperature.toFixed(4)} <span className="muted">{q.temperatureBucket}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
