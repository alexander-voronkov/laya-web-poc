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
const boots = new Map<ModelId, Promise<Boot>>();
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

function boot(spec: ModelSpec): Promise<Boot> {
  const existing = boots.get(spec.id);
  if (existing) return existing;
  const p = (async () => {
    const core = await loadCore(spec);
    const session = await LayaSession.load({
      spec,
      core,
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
  })();
  boots.set(spec.id, p);
  // A failed boot must not be cached, or the retry button silently re-serves the same
  // rejection forever.
  p.catch(() => { boots.delete(spec.id); });
  return p;
}

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

    boot(spec).then(
      ({ core: c, session: s }) => {
        if (!alive.current) return;
        setCore(c);
        setSession(s);
        setPhase("ready");
        cachedWeights(spec).then((state) => { if (alive.current) setCache(state); });
      },
      (e: unknown) => {
        if (!alive.current) return;
        setError(String((e as Error)?.message ?? e));
        setPhase("error");
      },
    );

    return () => {
      alive.current = false;
      progressListeners.set(spec.id, listOf(progressListeners, spec.id).filter((l) => l !== onProgress));
      stageListeners.set(spec.id, listOf(stageListeners, spec.id).filter((l) => l !== onStage));
    };
  }, [spec, attempt]);

  const retry = useCallback(() => {
    setError(null);
    setPhase("core");
    setAttempt((n) => n + 1);
  }, []);

  return { phase, error, core, session, spec, files, stages, cache, retry };
}
