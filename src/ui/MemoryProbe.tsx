import { useState } from "react";
import { mb } from "../format";

/** The one memory number that actually includes the model.
 *
 *  `performance.memory.usedJSHeapSize` counts the V8 heap only, and the ~600MB of
 *  dequantised weights live in WebAssembly linear memory, outside it. This API does
 *  include wasm memory, and it is available precisely because the page is
 *  cross-origin isolated for wasm threads anyway.
 *
 *  It is behind a button rather than on the panel because the browser schedules the
 *  measurement at its own convenience -- it can take seconds, and it is rate-limited,
 *  so polling it would report stale numbers while looking live. */
interface MemoryBreakdown {
  bytes: number;
  attribution?: { url?: string }[];
  types?: string[];
}

interface MemoryResult {
  bytes: number;
  breakdown?: MemoryBreakdown[];
}

type Measure = () => Promise<MemoryResult>;

const measure = (performance as unknown as { measureUserAgentSpecificMemory?: Measure })
  .measureUserAgentSpecificMemory;

export function MemoryProbe() {
  const [state, setState] = useState<"idle" | "busy" | "done" | "error">("idle");
  const [result, setResult] = useState<MemoryResult | null>(null);
  const [error, setError] = useState<string>("");

  if (!measure) {
    return (
      <span className="muted">
        performance.measureUserAgentSpecificMemory() недоступен — это Chromium-only и требует
        cross-origin isolation
      </span>
    );
  }

  const run = async () => {
    setState("busy");
    try {
      // Must be called on performance, not detached: it reads the realm off `this`.
      const r = await measure.call(performance);
      setResult(r);
      setState("done");
    } catch (e) {
      // SecurityError without isolation, and the call can be refused outright.
      setError(String((e as Error)?.message ?? e));
      setState("error");
    }
  };

  const wasm = result?.breakdown?.filter((b) => b.types?.includes("WebAssembly")) ?? [];
  const wasmBytes = wasm.reduce((a, b) => a + b.bytes, 0);

  return (
    <span>
      <button className="link-btn" onClick={run} disabled={state === "busy"}>
        {state === "busy" ? "браузер измеряет…" : state === "done" ? "измерить ещё раз" : "измерить память"}
      </button>
      {state === "done" && result && (
        <>
          {" "}
          <b>{mb(result.bytes)} МБ</b> всего в этом процессе
          {wasmBytes > 0 && <> · из них WebAssembly {mb(wasmBytes)} МБ</>}
          {wasmBytes === 0 && result.breakdown && (
            <> · разбивка без записи WebAssembly — браузер сгруппировал её иначе</>
          )}
        </>
      )}
      {state === "error" && <span className="error"> не удалось: {error}</span>}
    </span>
  );
}
