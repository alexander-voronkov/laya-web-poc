import { useCallback, useEffect, useRef, useState } from "react";
import {
  LayaSession,
  cachedWeights,
  loadCore,
  type CacheState,
  type Core,
  type LoadProgress,
  type LoadStage,
} from "./laya/session";
import { BootRegistry } from "./laya/bootRegistry";
import { MODELS, type ModelId, type ModelSpec } from "./models";

export type Phase = "core" | "weights" | "ready" | "error";

export interface LayaLoad {
  phase: Phase;
  error: string | null;
  core: Core | null;
  session: LayaSession | null;
  spec: ModelSpec;
  files: LoadProgress[];
  stages: Partial<Record<LoadStage, number>>;
  cache: CacheState | null;
}

interface Boot {
  core: Core;
  session: LayaSession;
}

/** One boot per model, kept at module scope.
 *
 *  React 19 StrictMode mounts effects twice in development, and a second boot would
 *  mean a second few-hundred-megabyte download and a second session — on these models
 *  that is not a wasted render, it is an out-of-memory tab. Keyed by model id rather
 *  than a single slot, because switching models and switching back should not pay for
 *  the weights twice in one session.
 */
const progressListeners = new Map<ModelId, ((p: LoadProgress) => void)[]>();
const stageListeners = new Map<ModelId, ((s: LoadStage, ms: number) => void)[]>();
// Replayed into late subscribers, so a remount does not show an empty progress list
// while a download that started before it is still running.
const seenProgress = new Map<ModelId, Map<string, LoadProgress>>();
const seenStages = new Map<ModelId, Partial<Record<LoadStage, number>>>();

const listOf = <T,>(m: Map<ModelId, T[]>, id: ModelId): T[] => {
  if (!m.has(id)) m.set(id, []);
  return m.get(id)!;
};
const progressOf = (id: ModelId) => {
  if (!seenProgress.has(id)) seenProgress.set(id, new Map());
  return seenProgress.get(id)!;
};
const stagesOf = (id: ModelId) => {
  if (!seenStages.has(id)) seenStages.set(id, {});
  return seenStages.get(id)!;
};

/** One boot per model, kept at module scope and abandoned when nobody is waiting.
 *
 *  The registry holds both halves of that: a second mount joins the download in flight
 *  rather than starting another, and switching away from a model that is still coming
 *  down stops it instead of paying for weights nobody asked for. What it cannot undo is
 *  a finished boot, which stays so that switching back is free. */
const boots = new BootRegistry<Boot>(
  async (id, signal) => {
    const spec = MODELS[id as ModelId];
    const core = await loadCore(spec, signal);
    const session = await LayaSession.load({
      spec,
      core,
      signal,
      onProgress: (pr) => {
        progressOf(spec.id).set(pr.file, pr);
        for (const l of listOf(progressListeners, spec.id)) l(pr);
      },
      onStage: (s, ms) => {
        stagesOf(spec.id)[s] = ms;
        for (const l of listOf(stageListeners, spec.id)) l(s, ms);
      },
    });
    return { core, session };
  },
  {
    // The bytes of an abandoned download were dropped unwritten -- fetchCached only
    // reaches the Cache API once a file is whole -- so the progress recorded for them
    // has to go too, or the next attempt starts its bar part-filled from a download
    // that no longer exists.
    onAbandon: (id) => {
      seenProgress.delete(id as ModelId);
      seenStages.delete(id as ModelId);
    },
  },
);

export function useLaya(modelId: ModelId): LayaLoad & { retry: () => void } {
  const spec = MODELS[modelId];
  const [phase, setPhase] = useState<Phase>("core");
  const [error, setError] = useState<string | null>(null);
  const [core, setCore] = useState<Core | null>(null);
  const [session, setSession] = useState<LayaSession | null>(null);
  const [files, setFiles] = useState<LoadProgress[]>([]);
  const [stages, setStages] = useState<Partial<Record<LoadStage, number>>>({});
  const [cache, setCache] = useState<CacheState | null>(null);
  const [attempt, setAttempt] = useState(0);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    // Switching models must not leave the previous one's numbers on screen: they would
    // read as this model's, and the two differ by hundreds of megabytes.
    setPhase("core");
    setError(null);
    setCore(null);
    setSession(null);
    setCache(null);
    setFiles([...progressOf(spec.id).values()]);
    setStages({ ...stagesOf(spec.id) });

    // Progress arrives thousands of times for a large file; repainting on each one
    // costs more than the download. Collect and flush on an animation frame.
    let pending = false;
    const onProgress = (p: LoadProgress) => {
      progressOf(spec.id).set(p.file, p);
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        if (alive.current) setFiles([...progressOf(spec.id).values()]);
      });
    };
    const onStage = (s: LoadStage, ms: number) => {
      if (!alive.current) return;
      setStages((prev) => ({ ...prev, [s]: ms }));
      if (s === "core") setPhase("weights");
    };
    listOf(progressListeners, spec.id).push(onProgress);
    listOf(stageListeners, spec.id).push(onStage);

    const held = boots.acquire(spec.id);
    held.promise.then(
      ({ core: c, session: s }) => {
        if (!alive.current) return;
        setCore(c);
        setSession(s);
        setPhase("ready");
        cachedWeights(spec).then((state) => { if (alive.current) setCache(state); });
      },
      (e: unknown) => {
        // An abandoned load is this component's own doing, not a fault to report --
        // and a live component should not reach here at all, since abandoning happens
        // only after its cleanup ran. Guarded anyway: were the bookkeeping ever wrong,
        // the symptom would be a permanent error screen for a model that is fine.
        if (!alive.current || (e as Error)?.name === "AbortError") return;
        setError(String((e as Error)?.message ?? e));
        setPhase("error");
      },
    );

    return () => {
      alive.current = false;
      progressListeners.set(spec.id, listOf(progressListeners, spec.id).filter((l) => l !== onProgress));
      stageListeners.set(spec.id, listOf(stageListeners, spec.id).filter((l) => l !== onStage));
      held.release();
    };
  }, [spec, attempt]);

  const retry = useCallback(() => {
    setError(null);
    setPhase("core");
    setAttempt((n) => n + 1);
  }, []);

  return { phase, error, core, session, spec, files, stages, cache, retry };
}
