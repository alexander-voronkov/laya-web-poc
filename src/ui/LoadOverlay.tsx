import { NOMINAL_TOTAL_BYTES } from "../config";
import type { LoadProgress } from "../laya/session";
import type { Phase } from "../useLaya";
import { mb } from "../format";

interface Props {
  phase: Phase;
  error: string | null;
  files: LoadProgress[];
  onRetry: () => void;
}

export function LoadOverlay({ phase, error, files, onRetry }: Props) {
  if (phase === "ready") return null;
  return (
    <div id="overlay">
      <div className="overlay-card">
        <h3>Загрузка модели</h3>
        <p className="muted">
          ~{mb(NOMINAL_TOTAL_BYTES)} МБ, только при первом открытии: файлы кэшируются браузером,
          в следующий раз — почти мгновенно. Всё считается локально, текст никуда не уходит.
        </p>
        <div id="ov-files">
          {files.map((f) => (
            <div className="ov-file" key={f.file}>
              <div className="ov-file-head">
                <span className="ov-name">{f.file}</span>
                <span className="ov-size">
                  {f.total ? `${mb(f.loaded)} / ${mb(f.total)} МБ` : `${mb(f.loaded)} МБ`}
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
            ? `Ошибка загрузки: ${error}`
            : phase === "core"
              ? "загрузка токенизатора и конфига…"
              : "загрузка весов модели…"}
        </div>
        {error && <button className="btn" onClick={onRetry}>Повторить</button>}
      </div>
    </div>
  );
}
