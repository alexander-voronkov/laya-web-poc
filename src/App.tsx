import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MODELS_BASE } from "./config";
import { analyse, type Budget } from "./budget";
import {
  FRAMING_LABELS,
  TYPE_LABELS,
  advice,
  buildState,
  newQuestion,
  toRequest,
  validateQuestions,
  type FramingMode,
  type QuestionItem,
  type QuestionType,
} from "./questions";
import { clearSession, defaultSession, loadSession, saveSession } from "./persist";
import { useLaya } from "./useLaya";
import type { Answer, QuestionTelemetry } from "./laya/types";
import { AnswerCard } from "./ui/AnswerCard";
import { LoadOverlay } from "./ui/LoadOverlay";
import { MetricsPanel } from "./ui/MetricsPanel";
import { QuestionCard } from "./ui/QuestionEditor";
import { sec } from "./format";

interface Landed {
  question: QuestionItem;
  index: number;
  answer: Answer;
  telemetry: QuestionTelemetry;
}

export default function App() {
  const laya = useLaya();
  const [session, setSession] = useState(loadSession);
  const { text, task, framing, questions, counter } = session;

  const [landed, setLanded] = useState<Landed[]>([]);
  const [run, setRun] = useState<{ questions: QuestionTelemetry[]; totalMs: number } | null>(null);
  const [running, setRunning] = useState(false);
  const [runningMs, setRunningMs] = useState<number | null>(null);
  const [status, setStatus] = useState<{ text: string; error?: boolean } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const patch = useCallback(
    (p: Partial<typeof session>) => setSession((s) => ({ ...s, ...p })),
    [],
  );

  // Persist on a timer rather than on every keystroke: a 3000-character text
  // serialised on each input event is a visible stutter on a slow machine.
  useEffect(() => {
    const t = setTimeout(() => saveSession(session), 400);
    return () => clearTimeout(t);
  }, [session]);

  const issues = useMemo(() => validateQuestions(questions), [questions]);
  const invalid = useMemo(() => new Map(issues.map((i) => [i.id, i.problem])), [issues]);
  const advices = useMemo(() => advice(task, text, questions), [task, text, questions]);
  const hardAdvice = advices.filter((a) => a.level === "hard");
  const runAdvice = advices.filter((a) => a.level === "soft" && a.id === null);

  // The budget runs the real tokenizer over every question, so it is debounced and
  // only exists once the small files have landed.
  const [budget, setBudget] = useState<Budget | null>(null);
  useEffect(() => {
    const core = laya.core;
    if (!core) return;
    const t = setTimeout(() => setBudget(analyse(core, task, text, questions, framing)), 250);
    return () => clearTimeout(t);
  }, [laya.core, task, text, questions, framing]);

  const byId = useMemo(
    () => new Map((budget?.perQuestion ?? []).map((b) => [b.id, b])),
    [budget],
  );

  const doRun = useCallback(async () => {
    const s = laya.session;
    if (!s || running) return;
    if (issues.length) {
      setStatus({ text: `исправьте: ${issues.map((i) => `${i.id} — ${i.problem}`).join("; ")}`, error: true });
      return;
    }
    const ctl = new AbortController();
    abortRef.current = ctl;
    setRunning(true);
    setLanded([]);
    setRun(null);
    setStatus({ text: `инференс: 0 из ${questions.length}…` });

    const order = new Map(questions.map((q, i) => [q.id, i]));
    const started = performance.now();
    const tick = setInterval(() => setRunningMs(performance.now() - started), 200);
    try {
      const result = await s.systemOne(
        buildState(task, text, framing),
        toRequest(task, questions, framing),
        {
          signal: ctl.signal,
          onAnswer: (qid, answer, telemetry) => {
            const index = order.get(qid) ?? 0;
            const question = questions[index];
            setLanded((prev) => [...prev, { question, index, answer, telemetry }]);
            setStatus({ text: `инференс: ${index + 1} из ${questions.length}…` });
          },
        },
      );
      setRun(result.telemetry);
      setStatus({
        text: result.aborted
          ? `остановлено после ${result.telemetry.questions.length} из ${questions.length}`
          : `готово за ${sec(result.telemetry.totalMs)}`,
      });
    } catch (e) {
      setStatus({ text: `ошибка: ${String((e as Error)?.message ?? e)}`, error: true });
    } finally {
      clearInterval(tick);
      setRunningMs(null);
      setRunning(false);
      abortRef.current = null;
    }
  }, [laya.session, running, issues, questions, task, text, framing]);

  const exportJson = useCallback(() => {
    if (!run) return;
    const payload = {
      generatedAt: new Date().toISOString(),
      model: {
        base: MODELS_BASE,
        quant: "q8 weight-only (MatMulNBits), onnxruntime-web/wasm",
        maxLen: laya.core?.cfg.max_len ?? null,
        headMaxLen: laya.core?.cfg.head_max_len ?? null,
        calibration:
          "температуры подобраны автором на fp32 и после квантования не перекалибровывались; " +
          "вероятности стоит читать как порядок, а не как абсолютную частоту",
      },
      input: { text, task, framing, questions },
      layaRequest: { state: buildState(task, text, framing), questions: toRequest(task, questions, framing) },
      answers: Object.fromEntries(landed.map((l) => [l.question.id, l.answer])),
      metrics: {
        perQuestion: run.questions,
        totalMs: run.totalMs,
        inputTokens: run.questions.reduce((a, q) => a + q.stats.totalTokens, 0),
        scoredOptions: run.questions.reduce((a, q) => a + q.options, 0),
        generatedTokens: 0,
        wasmThreads: laya.session?.numThreads ?? null,
        hardwareConcurrency: navigator.hardwareConcurrency ?? null,
        crossOriginIsolated: typeof crossOriginIsolated !== "undefined" ? crossOriginIsolated : null,
        weightCacheBytes: laya.cacheBytes,
        loadStages: laya.stages,
      },
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `laya-run-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [run, landed, text, task, framing, questions, laya]);

  const stateTokens = budget?.stateTokens ?? null;
  const worstLeft = budget ? budget.maxLen - budget.worstStateUsed : null;

  return (
    <>
      <div id="app">
        <header>
          <h1>Laya · web PoC</h1>
          <p className="tagline">
            Калиброванные вероятности по вашим вопросам. Модель (ModernBERT-large, q8) считается
            целиком в браузере через onnxruntime-web/wasm — текст не покидает компьютер, сервера у
            приложения нет, черновик хранится в localStorage.
          </p>
        </header>

        <main>
          {hardAdvice.map((a, i) => (
            <div className="banner hard" key={i}>{a.text}</div>
          ))}

          <section className="card">
            <h2>1 · Текст</h2>
            <label htmlFor="state-text">Текст для анализа (state)</label>
            <textarea
              id="state-text"
              rows={10}
              value={text}
              placeholder="Вставьте текст…"
              onChange={(e) => patch({ text: e.target.value })}
            />
            <div className="note-row">
              <span className="muted">
                {stateTokens === null
                  ? "токены текста: —"
                  : `токены текста: ${stateTokens} · до модели дойдёт ${budget!.worstStateUsed}` +
                    (worstLeft !== null && !budget!.anyTruncated ? ` (запас ${worstLeft} до предела ${budget!.maxLen})` : "")}
              </span>
              {budget?.anyTruncated && (
                <span className="warn">хвост текста обрезается — какой именно вопрос и насколько, видно в его карточке</span>
              )}
            </div>
          </section>

          <section className="card">
            <h2>2 · Задача</h2>
            <label htmlFor="task-text">Как интерпретировать текст</label>
            <textarea
              id="task-text"
              rows={2}
              value={task}
              placeholder="например: Read the text as a literary critic"
              onChange={(e) => patch({ task: e.target.value })}
            />
            <div className="field">
              <label>Куда подставлять задачу</label>
              <div className="radio-row">
                {(Object.keys(FRAMING_LABELS) as FramingMode[]).map((m) => (
                  <label key={m} className="radio">
                    <input
                      type="radio"
                      name="framing"
                      checked={framing === m}
                      onChange={() => patch({ framing: m })}
                    />
                    {FRAMING_LABELS[m]}
                  </label>
                ))}
              </div>
              <div className="field-note">
                Формулировка вопроса делит {budget?.headMaxLen ?? 192} токенов с вариантами ответа, и
                при нехватке обрезается <b>с конца</b> — то есть длинная задача впереди съедает сам
                вопрос. В тексте задача занимает место из общего бюджета, но вопрос не трогает.
              </div>
            </div>
          </section>

          <section className="card">
            <h2>3 · Вопросы</h2>
            {questions.length === 0 && (
              <div className="empty-note">Вопросов пока нет — добавьте первый ниже.</div>
            )}
            {questions.map((q, i) => (
              <QuestionCard
                key={q.id + i}
                question={q}
                index={i}
                total={questions.length}
                problem={invalid.get(q.id)}
                advice={advices.filter((a) => a.id === q.id).map((a) => a.text)}
                budget={byId.get(q.id)}
                onChange={(next) => patch({ questions: questions.map((x, j) => (j === i ? next : x)) })}
                onRemove={() => patch({ questions: questions.filter((_, j) => j !== i) })}
                onMove={(d) => {
                  const n = [...questions];
                  [n[i + d], n[i]] = [n[i], n[i + d]];
                  patch({ questions: n });
                }}
              />
            ))}
            <div className="add-q-row">
              <AddQuestion
                onAdd={(type) =>
                  patch({ questions: [...questions, newQuestion(type, counter + 1)], counter: counter + 1 })
                }
              />
              <button
                className="link-btn"
                onClick={() => { clearSession(); setSession(defaultSession()); setLanded([]); setRun(null); }}
              >
                сбросить к примеру
              </button>
            </div>
          </section>

          <section className="card run-card">
            <button className="btn primary" disabled={running || laya.phase !== "ready" || !questions.length} onClick={doRun}>
              Получить ответы
            </button>
            {running && (
              <button className="btn" onClick={() => abortRef.current?.abort()}>Остановить</button>
            )}
            <span className={`muted${status?.error ? " error" : ""}`}>
              {status?.text ?? (laya.phase === "ready" ? "модель готова" : "модель загружается…")}
            </span>
          </section>

          {runAdvice.map((a, i) => <div className="banner soft" key={i}>{a.text}</div>)}

          {(landed.length > 0 || running) && (
            <section className="card">
              <h2>Ответы</h2>
              <div className="calib-note">
                Температуры откалиброваны автором на fp32-модели и после квантования не
                перекалибровывались. Сравнивать вероятности между собой можно; читать их как
                абсолютную частоту — нет, пока они не перекалиброваны на ваших размеченных данных.
              </div>
              {landed.map((l) => (
                <AnswerCard
                  key={l.question.id}
                  question={l.question}
                  index={l.index}
                  answer={l.answer}
                  telemetry={l.telemetry}
                />
              ))}
            </section>
          )}

          <section className="card">
            <h2>Метрики</h2>
            <MetricsPanel
              files={laya.files}
              stages={laya.stages}
              cacheBytes={laya.cacheBytes}
              threads={laya.session?.numThreads ?? null}
              ready={laya.phase === "ready"}
              run={run}
              runningMs={runningMs}
            />
            <div className="metrics-actions">
              <button className="btn" disabled={!run} onClick={exportJson}>Экспорт JSON прогона</button>
            </div>
          </section>
        </main>

        <footer>
          <p>
            Прототип на базе <a href="https://github.com/nvkudva/laya-web" target="_blank" rel="noreferrer">nvkudva/laya-web</a>{" "}
            (порт рантайма) и весов <a href="https://huggingface.co/nvkudva/laya-web-q8" target="_blank" rel="noreferrer">nvkudva/laya-web-q8</a>{" "}
            · базовая модель <a href="https://huggingface.co/convaiinnovations/laya" target="_blank" rel="noreferrer">convaiinnovations/laya</a>{" "}
            · Apache-2.0
          </p>
        </footer>
      </div>

      <LoadOverlay phase={laya.phase} error={laya.error} files={laya.files} onRetry={laya.retry} />
    </>
  );
}

function AddQuestion({ onAdd }: { onAdd: (t: QuestionType) => void }) {
  const [type, setType] = useState<QuestionType>("noul");
  return (
    <>
      <select value={type} onChange={(e) => setType(e.target.value as QuestionType)}>
        {(Object.keys(TYPE_LABELS) as QuestionType[]).map((t) => (
          <option key={t} value={t}>{TYPE_LABELS[t]}</option>
        ))}
      </select>
      <button className="btn" onClick={() => onAdd(type)}>Добавить вопрос</button>
    </>
  );
}
