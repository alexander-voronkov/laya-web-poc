import { useCallback, useEffect, useRef, useState } from "react";
import { MODELS_BASE } from "./config";
import {
  LayaSession,
  cachedWeightBytes,
  loadCore,
  type Core,
  type LoadProgress,
  type LoadStage,
} from "./laya/session";

export type Phase = "core" | "weights" | "ready" | "error";

export interface LayaLoad {
  phase: Phase;
  error: string | null;
  core: Core | null;
  session: LayaSession | null;
  files: LoadProgress[];
  stages: Partial<Record<LoadStage, number>>;
  cacheBytes: number | null;
}

interface Boot {
  core: Core;
  session: LayaSession;
}

// Module scope on purpose. React 19 StrictMode mounts effects twice in development,
// and a second boot would mean a second 524MB download and a second 600MB session --
// on this model that is not a wasted render, it is an out-of-memory tab. Keeping the
// promise outside the component makes the second mount await the first boot.
let bootPromise: Promise<Boot> | null = null;
let listeners: ((p: LoadProgress) => void)[] = [];
let stageListeners: ((s: LoadStage, ms: number) => void)[] = [];
// Replayed into late subscribers, so a remount does not show an empty progress list
// while a download that started before it is still running.
const seenProgress = new Map<string, LoadProgress>();
const seenStages: Partial<Record<LoadStage, number>> = {};

function boot(): Promise<Boot> {
  if (bootPromise) return bootPromise;
  bootPromise = (async () => {
    const core = await loadCore(MODELS_BASE);
    const session = await LayaSession.load({
      base: MODELS_BASE,
      core,
      onProgress: (p) => {
        seenProgress.set(p.file, p);
        for (const l of listeners) l(p);
      },
      onStage: (s, ms) => {
        seenStages[s] = ms;
        for (const l of stageListeners) l(s, ms);
      },
    });
    return { core, session };
  })();
  // A failed boot must not be cached, or the retry button silently re-serves the
  // same rejection forever.
  bootPromise.catch(() => { bootPromise = null; });
  return bootPromise;
}

export function useLaya(): LayaLoad & { retry: () => void } {
  const [phase, setPhase] = useState<Phase>("core");
  const [error, setError] = useState<string | null>(null);
  const [core, setCore] = useState<Core | null>(null);
  const [session, setSession] = useState<LayaSession | null>(null);
  const [files, setFiles] = useState<LoadProgress[]>([...seenProgress.values()]);
  const [stages, setStages] = useState<Partial<Record<LoadStage, number>>>({ ...seenStages });
  const [cacheBytes, setCacheBytes] = useState<number | null>(null);
  const [attempt, setAttempt] = useState(0);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    // Progress arrives thousands of times for a 468MB file; repainting on each one
    // costs more than the download. Collect and flush on an animation frame.
    let pending = false;
    const onProgress = (p: LoadProgress) => {
      seenProgress.set(p.file, p);
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        if (alive.current) setFiles([...seenProgress.values()]);
      });
    };
    const onStage = (s: LoadStage, ms: number) => {
      if (alive.current) setStages((prev) => ({ ...prev, [s]: ms }));
      if (s === "core" && alive.current) setPhase("weights");
    };
    listeners.push(onProgress);
    stageListeners.push(onStage);

    boot().then(
      ({ core: c, session: s }) => {
        if (!alive.current) return;
        setCore(c);
        setSession(s);
        setPhase("ready");
        cachedWeightBytes().then((n) => { if (alive.current) setCacheBytes(n); });
      },
      (e: unknown) => {
        if (!alive.current) return;
        setError(String((e as Error)?.message ?? e));
        setPhase("error");
      },
    );

    return () => {
      alive.current = false;
      listeners = listeners.filter((l) => l !== onProgress);
      stageListeners = stageListeners.filter((l) => l !== onStage);
    };
  }, [attempt]);

  const retry = useCallback(() => {
    setError(null);
    setPhase("core");
    setAttempt((n) => n + 1);
  }, []);

  return { phase, error, core, session, files, stages, cacheBytes, retry };
}
