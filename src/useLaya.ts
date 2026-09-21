import { useCallback, useEffect, useState } from "react";
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

  useEffect(() => {
    // Per effect run, deliberately not a ref.
    //
    // A ref is shared by every run: the cleanup sets it false and the next run sets it
    // true again, so a callback left over from the *previous* model passes the check
    // and writes its own session into state. That is not a cosmetic race. Measured on
    // the deployed site: selecting multilingual-fp16 downloaded english-q8 in full,
    // english's boot resolved half a minute later, and the page then answered with
    // english while the picker, the metrics panel and the run record all said
    // multilingual-fp16. Wrong answers under the right name are the one failure this
    // app must not have.
    //
    // A local closure variable belongs to one run and one model, so a late callback
    // from a superseded run can never satisfy it.
    let current = true;
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
        if (current) setFiles([...progressOf(spec.id).values()]);
      });
    };
    const onStage = (s: LoadStage, ms: number) => {
      if (!current) return;
      setStages((prev) => ({ ...prev, [s]: ms }));
      if (s === "core") setPhase("weights");
    };
    listOf(progressListeners, spec.id).push(onProgress);
    listOf(stageListeners, spec.id).push(onStage);

    const held = boots.acquire(spec.id);
    held.promise.then(
      ({ core: c, session: s }) => {
        if (!current) return;
        setCore(c);
        setSession(s);
        setPhase("ready");
        cachedWeights(spec).then((state) => { if (current) setCache(state); });
      },
      (e: unknown) => {
        // An abandoned load is this component's own doing, not a fault to report --
        // and a live component should not reach here at all, since abandoning happens
        // only after its cleanup ran. Guarded anyway: were the bookkeeping ever wrong,
        // the symptom would be a permanent error screen for a model that is fine.
        if (!current || (e as Error)?.name === "AbortError") return;
        setError(String((e as Error)?.message ?? e));
        setPhase("error");
      },
    );

    return () => {
      current = false;
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
