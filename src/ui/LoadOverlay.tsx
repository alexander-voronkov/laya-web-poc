import type { ModelSpec } from "../models";
import type { LoadProgress } from "../laya/session";
import type { Phase } from "../useLaya";
import { mb } from "../format";

interface Props {
  spec: ModelSpec;
  phase: Phase;
  error: string | null;
  files: LoadProgress[];
  onRetry: () => void;
}

export function LoadOverlay({ spec, phase, error, files, onRetry }: Props) {
  if (phase === "ready") return null;
  return (
    <div id="overlay">
      <div className="overlay-card">
        <h3>Loading the model</h3>
        <p className="muted">
          {spec.label}: ~{mb(spec.nominalBytes)} MB, on the first visit only: the files are cached by the browser,
          so next time is almost instant. Everything runs locally; the text never leaves the machine.
        </p>
        <div id="ov-files">
          {files.map((f) => (
            <div className="ov-file" key={f.file}>
              <div className="ov-file-head">
                <span className="ov-name">{f.file}</span>
                <span className="ov-size">
                  {f.total ? `${mb(f.loaded)} / ${mb(f.total)} MB` : `${mb(f.loaded)} MB`}
                </span>
              </div>
              {/* Without Content-Length (chunked or compressed responses) there is no
                  denominator. A full bar would then be indistinguishable from a
                  finished file and reads as a hang; an indeterminate one is honest. */}
              <div className={`bar slim${f.total ? "" : " indeterminate"}`}>
                <div
                  className={`bar-fill${f.cached ? " cached" : ""}`}
                  style={f.total ? { width: `${Math.min(100, (f.loaded / f.total) * 100).toFixed(1)}%` } : undefined}
                />
              </div>
            </div>
          ))}
        </div>
        <div id="ov-status" className={error ? "error" : "muted"}>
          {error
            ? `Load failed: ${error}`
            : phase === "core"
              ? "loading the tokenizer and config…"
              : "downloading the model weights…"}
        </div>
        {error && <button className="btn" onClick={onRetry}>Retry</button>}
      </div>
    </div>
  );
}
