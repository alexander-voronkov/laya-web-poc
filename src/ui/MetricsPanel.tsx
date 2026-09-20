import { useEffect, useState, type ReactNode } from "react";
import { NOMINAL_TOTAL_BYTES } from "../config";
import { jsHeapBytes, type CacheState, type LayaSession, type LoadProgress, type LoadStage } from "../laya/session";
import type { QuestionTelemetry } from "../laya/types";
import { mb, ms, sec } from "../format";

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
  const encoderMs = qs.reduce((a, q) => a + q.encoderMs, 0);
  const droppedTokens = qs.reduce((a, q) => a + (q.stats.stateTokens - q.stats.stateTokensUsed), 0);
  const totalMs = run?.totalMs ?? 0;

  const cores = navigator.hardwareConcurrency;
  const deviceMemory = (navigator as unknown as { deviceMemory?: number }).deviceMemory;
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  const threadsRequested = session?.requestedThreads ?? null;
  const threadsEffective = session?.numThreads ?? null;

  const rows: [string, ReactNode][] = [
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
      threadsEffective === null
        ? "—"
        : threadsEffective === threadsRequested
          ? `${threadsEffective} из ${cores ?? "?"} ядер · crossOriginIsolated: ✓`
          : `${threadsEffective} (запрошено ${threadsRequested}, ядер ${cores ?? "?"}) · crossOriginIsolated: ✗`,
    ],
    ...(ready && !isolated
      ? ([[
          "⚠ Изоляция",
          "нет cross-origin isolation — SharedArrayBuffer недоступен, wasm считает в один поток, примерно в 6 раз дольше. Число потоков выше выведено из этого факта, а не измерено: ORT не сообщает, на скольких потоках он фактически пошёл. Проверьте заголовки COOP/COEP.",
        ]] as [string, ReactNode][])
      : []),
    [
      "Время прогона",
      qs.length
        ? `${sec(totalMs)} на ${qs.length} вопр.${run && run.aborted ? ` из ${run.requested} (остановлено)` : ""} · ${ms(totalMs / qs.length)} в среднем · энкодер ${((encoderMs / totalMs) * 100).toFixed(0)}% времени`
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
      // usedJSHeapSize measures the V8 heap. WebAssembly linear memory -- where the
      // ~600MB of dequantised weights actually live -- is a separate backing store and
      // is NOT counted here. Saying otherwise invites reading "142 MB" as the model
      // fitting in 142 MB.
      heap !== null
        ? `${mb(heap)} МБ — это куча JavaScript; линейная память wasm, где лежат веса, сюда не входит и браузером не раскрывается`
        : "performance.memory недоступен в этом браузере (есть только в Chromium)",
    ],
    [
      "Память устройства",
      deviceMemory ? `${deviceMemory} ГБ (округление браузера)` : "navigator.deviceMemory недоступен",
    ],
    ["Кэш весов", <CacheLine cache={cache} ready={ready} key="cache" />],
    [
      "Квота хранилища",
      !storage.available
        ? "navigator.storage.estimate() недоступен"
        : storage.quota === null
          ? "браузер не сообщает квоту"
          : `${storage.usage === null ? "занято неизвестно" : `${mb(storage.usage)} МБ занято`} из ~${(storage.quota / 1e9).toFixed(1)} ГБ (оценка браузера)`,
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
  if (!cache) return <>{ready ? "проверяется…" : "—"}</>;
  if (cache.unavailable) return <span className="warn">Cache Storage недоступен (приватный режим или запрет на данные сайта) — веса будут скачиваться каждый раз</span>;
  if (cache.files === 0) return <span className="warn">веса не закэшировались (квота или запрет) — при перезагрузке скачаются заново</span>;
  if (cache.files < cache.expected)
    return (
      <span className="warn">
        закэшировано {cache.files} из {cache.expected} файлов ({mb(cache.bytes)} МБ) — скорее всего не хватило квоты;
        недостающие скачаются заново
      </span>
    );
  return <>{mb(cache.bytes)} МБ, все {cache.expected} файла — следующее открытие без скачивания</>;
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
