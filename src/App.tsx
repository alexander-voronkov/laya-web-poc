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
import { clearSession, defaultSession, hasSetAsideDraft, loadSession, saveSession } from "./persist";
import { useLaya } from "./useLaya";
import type { Answer, QuestionTelemetry, Questions } from "./laya/types";
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

/** Everything about one run, frozen at the moment it started.
 *
 *  The export used to rebuild `state` and `questions` from live editor state while
 *  taking the answers from the finished run, so fixing a typo after a run and then
 *  exporting produced a file pairing the corrected text with probabilities computed
 *  from the old one. For an artefact whose whole purpose is "this request produced
 *  these numbers", that is the one thing it must not do. */
interface RunRecord {
  at: string;
  state: string;
  request: Questions;
  input: { text: string; task: string; framing: FramingMode; questions: QuestionItem[] };
  requested: number;
  landed: Landed[];
  telemetry: { questions: QuestionTelemetry[]; totalMs: number };
  aborted: boolean;
}

export default function App() {
  const laya = useLaya();
  const [session, setSession] = useState(loadSession);
  const { text, task, framing, questions, counter } = session;

  const [live, setLive] = useState<Landed[]>([]);
  const [result, setResult] = useState<RunRecord | null>(null);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<{ text: string; error?: boolean } | null>(null);
  const [saveFailed, setSaveFailed] = useState(false);
  const [recovered] = useState(hasSetAsideDraft);
  const abortRef = useRef<AbortController | null>(null);

  const patch = useCallback((p: Partial<typeof session>) => setSession((s) => ({ ...s, ...p })), []);

  // Persist on a timer rather than on every keystroke: a long text serialised on each
  // input event is a visible stutter on a slow machine.
  useEffect(() => {
    const t = setTimeout(() => setSaveFailed(!saveSession(session)), 400);
    return () => clearTimeout(t);
  }, [session]);

  const issues = useMemo(() => validateQuestions(questions), [questions]);
  const invalid = useMemo(() => new Map(issues.map((i) => [i.uid, i.problem])), [issues]);
  const advices = useMemo(() => advice(task, text, questions), [task, text, questions]);
  const hardAdvice = advices.filter((a) => a.level === "hard");

  // The budget runs the real tokenizer over every question, so it is debounced and
  // only exists once the small files have landed.
  const [budget, setBudget] = useState<Budget | null>(null);
  const [budgetError, setBudgetError] = useState<string | null>(null);
  useEffect(() => {
    const core = laya.core;
    if (!core) return;
    const t = setTimeout(() => {
      // The tokenizer runs over arbitrary pasted input here. An exception inside a
      // render effect with no error boundary above it is a blank page, so this one
      // degrades to "budget unknown" instead.
      try {
        setBudget(analyse(core, task, text, questions, framing));
        setBudgetError(null);
      } catch (e) {
        setBudget(null);
        setBudgetError(String((e as Error)?.message ?? e));
      }
    }, 250);
    return () => clearTimeout(t);
  }, [laya.core, task, text, questions, framing]);

  const byUid = useMemo(() => new Map((budget?.perQuestion ?? []).map((b) => [b.uid, b])), [budget]);

  // A question whose options do not fit the head budget throws mid-run, which leaves
  // half the answers on screen and no metrics. Refuse before starting instead.
  const broken = useMemo(
    () => (budget?.perQuestion ?? []).filter((b) => b.optionsDontFit).map((b) => b.uid),
    [budget],
  );
  const blockers = issues.length + broken.length;

  const doRun = useCallback(async () => {
    const s = laya.session;
    if (!s || running) return;
    if (issues.length) {
      setStatus({ text: `исправьте: ${issues.map((i) => i.problem).join("; ")}`, error: true });
      return;
    }
    if (broken.length) {
      setStatus({ text: "есть вопросы, варианты которых не помещаются в бюджет головы", error: true });
      return;
    }

    // Freeze the request now. Everything the export reports comes from these values,
    // never from the editor as it stands when the export button is pressed.
    const snapshot = {
      at: new Date().toISOString(),
      state: buildState(task, text, framing),
      request: toRequest(task, questions, framing),
      input: { text, task, framing, questions: structuredClone(questions) },
      requested: questions.length,
    };
    const order = new Map(questions.map((q, i) => [q.id, i]));
    const landed: Landed[] = [];

    const ctl = new AbortController();
    abortRef.current = ctl;
    setRunning(true);
    setLive([]);
    setResult(null);
    setStatus({ text: `инференс: 0 из ${questions.length}…` });

    try {
      const run = await s.systemOne(snapshot.state, snapshot.request, {
        signal: ctl.signal,
        // Explicit order: Object.entries hoists integer-like keys, so questions named
        // "1", "2", "3" would run in numeric order regardless of how they are arranged.
        order: questions.map((q) => q.id),
        onAnswer: (qid, answer, telemetry) => {
          const index = order.get(qid) ?? 0;
          landed.push({ question: questions[index], index, answer, telemetry });
          setLive([...landed]);
          setStatus({ text: `инференс: ${landed.length} из ${questions.length}…` });
        },
      });
      setResult({ ...snapshot, landed, telemetry: run.telemetry, aborted: run.aborted });
      setStatus({
        text: run.aborted
          ? `остановлено: ${landed.length} из ${questions.length} за ${sec(run.telemetry.totalMs)}`
          : `готово за ${sec(run.telemetry.totalMs)}`,
      });
    } catch (e) {
      // Whatever landed before the throw stays on screen, but it is not a run: no
      // result record means no metrics table and no export, so a partial screen
      // cannot be mistaken for a finished one.
      setStatus({ text: `ошибка на вопросе ${landed.length + 1}: ${String((e as Error)?.message ?? e)}`, error: true });
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }, [laya.session, running, issues, broken, questions, task, text, framing]);

  const exportJson = useCallback(() => {
    if (!result) return;
    const payload = {
      generatedAt: result.at,
      model: {
        base: MODELS_BASE,
        quant: "q8 weight-only (MatMulNBits), onnxruntime-web/wasm",
        maxLen: laya.core?.cfg.max_len ?? null,
        headMaxLen: laya.core?.cfg.head_max_len ?? null,
        calibration:
          "температуры подобраны автором на fp32 и после квантования не перекалибровывались; " +
          "вероятности стоит читать как порядок, а не как абсолютную частоту",
        englishOnly: true,
      },
      input: result.input,
      layaRequest: { state: result.state, questions: result.request },
      answers: Object.fromEntries(result.landed.map((l) => [l.question.id, l.answer])),
      run: {
        aborted: result.aborted,
        questionsRequested: result.requested,
        questionsAnswered: result.landed.length,
      },
      metrics: {
        perQuestion: result.telemetry.questions,
        totalMs: result.telemetry.totalMs,
        inputTokens: result.telemetry.questions.reduce((a, q) => a + q.stats.totalTokens, 0),
        scoredOptions: result.telemetry.questions.reduce((a, q) => a + q.options, 0),
        generatedTokens: 0,
        wasmThreadsRequested: laya.session?.requestedThreads ?? null,
        wasmThreadsEffective: laya.session?.numThreads ?? null,
        hardwareConcurrency: navigator.hardwareConcurrency ?? null,
        crossOriginIsolated: typeof crossOriginIsolated !== "undefined" ? crossOriginIsolated : null,
        weightCache: laya.cache,
        loadStages: laya.stages,
      },
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `laya-run-${result.at.replace(/[:.]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [result, laya]);

  const shown = result ? result.landed : live;
  // What is left of max_len is measured against the longest sequence, not the text
  // count: the question head and its options take their share of the 512 first.
  const headroom = budget ? budget.maxLen - budget.worstTotalTokens : null;

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
          {hardAdvice.map((a, i) => <div className="banner hard" key={i}>{a.text}</div>)}
          {saveFailed && (
            <div className="banner hard">
              Черновик не сохраняется: браузер отказал в записи в localStorage (приватный режим или
              исчерпана квота). Всё на странице работает, но после перезагрузки вернётся пример.
            </div>
          )}
          {recovered && (
            <div className="banner soft">
              Предыдущий черновик не удалось прочитать, и он отложен в{" "}
              <code>laya-web-poc/session.bak</code> — его можно достать из DevTools → Application →
              Local Storage. Сейчас загружен пример.
            </div>
          )}

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
                {budget === null
                  ? budgetError
                    ? `не удалось посчитать бюджет: ${budgetError}`
                    : "токены текста: —"
                  : `токены текста: ${budget.stateTokens} · до модели дойдёт ${budget.worstStateUsed}` +
                    (!budget.anyTruncated && headroom !== null
                      ? ` (в самой длинной последовательности остаётся ${headroom} из ${budget.maxLen})`
                      : "")}
              </span>
              {budget?.anyTruncated && (
                <span className="warn">хвост текста обрезается — в каком вопросе и насколько, видно в его карточке</span>
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
                    <input type="radio" name="framing" checked={framing === m} onChange={() => patch({ framing: m })} />
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
            {questions.length === 0 && <div className="empty-note">Вопросов пока нет — добавьте первый ниже.</div>}
            {questions.map((q, i) => (
              <QuestionCard
                key={q.uid}
                question={q}
                index={i}
                total={questions.length}
                problem={invalid.get(q.uid)}
                advice={advices.filter((a) => a.uid === q.uid).map((a) => a.text)}
                budget={byUid.get(q.uid)}
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
                onClick={() => { clearSession(); setSession(defaultSession()); setLive([]); setResult(null); }}
              >
                сбросить к примеру
              </button>
            </div>
          </section>

          <section className="card run-card">
            <button
              className="btn primary"
              disabled={running || laya.phase !== "ready" || !questions.length || blockers > 0}
              onClick={doRun}
            >
              Получить ответы
            </button>
            {running && <button className="btn" onClick={() => abortRef.current?.abort()}>Остановить</button>}
            <span className={`muted${status?.error ? " error" : ""}`}>
              {status?.text ??
                (laya.phase !== "ready"
                  ? "модель загружается…"
                  : blockers > 0
                    ? `${blockers} вопр. требуют правки — см. карточки выше`
                    : "модель готова")}
            </span>
          </section>

          {(shown.length > 0 || running) && (
            <section className="card">
              <h2>
                Ответы
                {result?.aborted && (
                  <span className="ans-partial"> · прогон остановлен: {result.landed.length} из {result.requested}</span>
                )}
              </h2>
              <div className="calib-note">
                Температуры откалиброваны автором на fp32-модели и после квантования не
                перекалибровывались. Сравнивать вероятности между собой можно; читать их как
                абсолютную частоту — нет, пока они не перекалиброваны на ваших размеченных данных.
              </div>
              {shown.map((l) => (
                <AnswerCard key={l.question.uid} question={l.question} index={l.index} answer={l.answer} telemetry={l.telemetry} />
              ))}
            </section>
          )}

          <section className="card">
            <h2>Метрики</h2>
            <MetricsPanel
              files={laya.files}
              stages={laya.stages}
              cache={laya.cache}
              session={laya.session}
              ready={laya.phase === "ready"}
              run={result ? { ...result.telemetry, aborted: result.aborted, requested: result.requested } : null}
            />
            <div className="metrics-actions">
              <button className="btn" disabled={!result} onClick={exportJson}>Экспорт JSON прогона</button>
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
