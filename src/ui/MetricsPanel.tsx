import { useEffect, useState } from "react";
import { NOMINAL_TOTAL_BYTES } from "../config";
import { jsHeapBytes, type LoadProgress, type LoadStage } from "../laya/session";
import type { QuestionTelemetry } from "../laya/types";
import { mb, ms, sec } from "../format";

interface Props {
  files: LoadProgress[];
  stages: Partial<Record<LoadStage, number>>;
  cacheBytes: number | null;
  threads: number | null;
  ready: boolean;
  run: { questions: QuestionTelemetry[]; totalMs: number } | null;
  /** Non-null while a run is in flight, so the panel is live rather than final-only. */
  runningMs: number | null;
}

interface Storage {
  usage: number | null;
  quota: number | null;
}

/** Quota tells the user whether a 524MB cache will survive; usage tells them whether
 *  it is actually there. Both are estimates by specification -- labelled as such. */
function useStorageEstimate(ready: boolean): Storage {
  const [s, setS] = useState<Storage>({ usage: null, quota: null });
  useEffect(() => {
    if (!ready) return;
    let alive = true;
    navigator.storage?.estimate?.().then(
      (e) => { if (alive) setS({ usage: e.usage ?? null, quota: e.quota ?? null }); },
      () => { /* not supported, or blocked */ },
    );
    return () => { alive = false; };
  }, [ready]);
  return s;
}

export function MetricsPanel({ files, stages, cacheBytes, threads, ready, run, runningMs }: Props) {
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
  const encoderMs = qs.reduce((a, q) => a + q.encoderMs, 0);
  const droppedTokens = qs.reduce((a, q) => a + (q.stats.stateTokens - q.stats.stateTokensUsed), 0);
  const totalMs = run?.totalMs ?? runningMs ?? 0;

  const cores = navigator.hardwareConcurrency;
  const deviceMemory = (navigator as unknown as { deviceMemory?: number }).deviceMemory;
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;

  const rows: [string, React.ReactNode][] = [
    [
      "Загрузка весов",
      ready || loaded
        ? `${mb(loaded)} МБ${total ? ` из ${mb(total)} МБ` : ""} · ${sec(weightsMs)}${cachedAny ? " · из кэша браузера" : ""}`
        : `~${mb(NOMINAL_TOTAL_BYTES)} МБ, скачивается…`,
    ],
    [
      "Инициализация сессий ORT",
      encInit || headInit ? `${sec(encInit + headInit)} (энкодер ${sec(encInit)} · голова ${sec(headInit)})` : "—",
    ],
    [
      "Wasm-потоки",
      `${threads ?? "?"} из ${cores ?? "?"} ядер · crossOriginIsolated: ${isolated ? "✓" : "✗"}`,
    ],
    ...(isolated ? [] : [[
      "⚠ Изоляция",
      "нет cross-origin isolation — SharedArrayBuffer недоступен, wasm работает в один поток и считает примерно в 6 раз дольше. Проверьте заголовки COOP/COEP.",
    ] as [string, React.ReactNode]]),
    [
      "Время прогона",
      qs.length
        ? `${sec(totalMs)} на ${qs.length} вопр. · ${ms(totalMs / qs.length)} в среднем · энкодер ${((encoderMs / totalMs) * 100).toFixed(0)}% времени`
        : "—",
    ],
    [
      "Токены",
      qs.length
        ? `на вход ${inputTokens} · оценено вариантов ${scored}${droppedTokens ? ` · отброшено из текста ${droppedTokens}` : ""} · сгенерировано 0 (модель негенеративная)`
        : "—",
    ],
    [
      "Пропускная способность",
      qs.length ? `${(inputTokens / (encoderMs / 1000)).toFixed(0)} токенов/с через энкодер` : "—",
    ],
    [
      "Память (JS heap)",
      heap !== null
        ? `${mb(heap)} МБ · веса живут в wasm-куче внутри этого числа`
        : "performance.memory недоступен в этом браузере (есть только в Chromium)",
    ],
    [
      "Память устройства",
      deviceMemory ? `${deviceMemory} ГБ (округление браузера)` : "navigator.deviceMemory недоступен",
    ],
    [
      "Кэш весов",
      cacheBytes
        ? `${mb(cacheBytes)} МБ в Cache Storage — следующее открытие без скачивания`
        : ready
          ? "веса не закэшировались (квота или приватный режим) — при перезагрузке скачаются заново"
          : "—",
    ],
    [
      "Квота хранилища",
      storage.quota !== null
        ? `${mb(storage.usage ?? 0)} из ~${(storage.quota / 1e9).toFixed(1)} ГБ занято (оценка браузера)`
        : "navigator.storage.estimate() недоступен",
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

function PerQuestionTable({ qs }: { qs: QuestionTelemetry[] }) {
  return (
    <div className="table-wrap">
      <table className="metrics-table">
        <thead>
          <tr>
            <th>вопрос</th><th>тип</th><th>вар.</th><th>токенов</th>
            <th>текст</th><th>энкодер</th><th>голова</th><th>всего</th><th>температура</th>
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
              <td>{ms(q.encoderMs)}</td>
              <td>{ms(q.headMs)}</td>
              <td>{ms(q.totalMs)}</td>
              <td>{q.temperature.toFixed(4)} <span className="muted">{q.temperatureBucket}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
