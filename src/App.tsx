import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MODELS, type ModelId } from "./models";
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
import { webgpuAvailable } from "./laya/session";
import type { Answer, QuestionTelemetry, Questions } from "./laya/types";
import { AnswerCard } from "./ui/AnswerCard";
import { LoadOverlay } from "./ui/LoadOverlay";
import { MetricsPanel } from "./ui/MetricsPanel";
import { QuestionCard } from "./ui/QuestionEditor";
import { sec } from "./format";

const BUDGETS = [512, 1024, 2048, 4096, 8192];

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
  /** Sequence budget this run actually used. */
  maxLen: number;
  landed: Landed[];
  telemetry: { questions: QuestionTelemetry[]; totalMs: number };
  aborted: boolean;
}

export default function App() {
  const [session, setSession] = useState(loadSession);
  const { text, task, framing, questions, counter, maxLen, model, batchSize } = session;
  const laya = useLaya(model);

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
  const advices = useMemo(
    () => advice(task, text, questions, laya.spec.languages),
    [task, text, questions, laya.spec.languages],
  );
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
        setBudget(analyse(core, task, text, questions, framing, maxLen, BUDGETS));
        setBudgetError(null);
      } catch (e) {
        setBudget(null);
        setBudgetError(String((e as Error)?.message ?? e));
      }
    }, 250);
    return () => clearTimeout(t);
  }, [laya.core, task, text, questions, framing, maxLen]);

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
      setStatus({ text: `fix first: ${issues.map((i) => i.problem).join("; ")}`, error: true });
      return;
    }
    if (broken.length) {
      setStatus({ text: "some questions have options that do not fit the head budget", error: true });
      return;
    }

    // Resolved fresh rather than read off the debounced panel state: a run started
    // straight after a keystroke would otherwise use the previous text's budget.
    const core = laya.core;
    const resolved = core
      ? analyse(core, task, text, questions, framing, maxLen, BUDGETS).maxLen
      : (maxLen ?? s.cfg.max_len);
    s.maxLen = resolved;

    // Freeze the request now. Everything the export reports comes from these values,
    // never from the editor as it stands when the export button is pressed.
    const snapshot = {
      at: new Date().toISOString(),
      state: buildState(task, text, framing),
      request: toRequest(task, questions, framing),
      input: { text, task, framing, questions: structuredClone(questions) },
      requested: questions.length,
      maxLen: resolved,
    };
    const order = new Map(questions.map((q, i) => [q.id, i]));
    const landed: Landed[] = [];

    const ctl = new AbortController();
    abortRef.current = ctl;
    setRunning(true);
    setLive([]);
    setResult(null);
    setStatus({ text: `running: 0 of ${questions.length}…` });

    try {
      const run = await s.systemOne(snapshot.state, snapshot.request, {
        signal: ctl.signal,
        // Explicit order: Object.entries hoists integer-like keys, so questions named
        // "1", "2", "3" would run in numeric order regardless of how they are arranged.
        order: questions.map((q) => q.id),
        batchSize,
        onAnswer: (qid, answer, telemetry) => {
          const index = order.get(qid) ?? 0;
          landed.push({ question: questions[index], index, answer, telemetry });
          setLive([...landed]);
          setStatus({ text: `running: ${landed.length} of ${questions.length}…` });
        },
      });
      setResult({ ...snapshot, landed, telemetry: run.telemetry, aborted: run.aborted });
      setStatus({
        text: run.aborted
          ? `stopped: ${landed.length} of ${questions.length} in ${sec(run.telemetry.totalMs)}`
          : `done in ${sec(run.telemetry.totalMs)}`,
      });
    } catch (e) {
      // Whatever landed before the throw stays on screen, but it is not a run: no
      // result record means no metrics table and no export, so a partial screen
      // cannot be mistaken for a finished one.
      setStatus({ text: `failed on question ${landed.length + 1}: ${String((e as Error)?.message ?? e)}`, error: true });
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }, [laya.session, laya.core, running, issues, broken, questions, task, text, framing, maxLen, batchSize]);

  const exportJson = useCallback(() => {
    if (!result) return;
    const payload = {
      generatedAt: result.at,
      model: {
        id: laya.spec.id,
        base: laya.spec.base,
        layout: laya.spec.layout,
        maxLen: result.maxLen,
        maxLenChosenAutomatically: maxLen === null,
        trainedMaxLen: laya.core?.cfg.max_len ?? null,
        headMaxLen: laya.core?.cfg.head_max_len ?? null,
        calibration: laya.spec.languages === "multilingual"
          ? "this checkpoint ships with no fitted temperatures at all — all 1.0, temperature_by_options empty; read the probabilities as an ordering"
          : "temperatures were fitted by the original author on the fp32 model and not refitted after quantisation; read the probabilities as an ordering, not as frequencies",
        languages: laya.spec.languages,
        // Recorded because it is the reason a run has one forward pass per question.
        batchSafe: laya.spec.batchSafe,
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

  // How large the budget would have to be for the whole text to reach the model.
  // Reporting that a text is being cut without saying what would fix it leaves the
  // reader to work out the arithmetic of a head they cannot see.
  const neededBudget = useMemo(() => {
    if (!budget || !budget.anyTruncated) return null;
    const worstOverhead = Math.max(
      ...budget.perQuestion.map((p) => p.stats.totalTokens - p.stats.stateTokensUsed),
      0,
    );
    const need = budget.stateTokens + worstOverhead;
    return BUDGETS.find((b) => b >= need) ?? null;
  }, [budget]);

  return (
    <>
      <div id="app">
        <header>
          <h1>Laya · web PoC</h1>
          <p className="tagline">
            Calibrated probabilities for your own questions. The model (ModernBERT-large, q8) runs
            entirely in the browser on onnxruntime-web/wasm — the text never leaves your machine,
            there is no backend, and the draft is kept in localStorage.
          </p>
        </header>

        <main>
          {hardAdvice.map((a, i) => <div className="banner hard" key={i}>{a.text}</div>)}
          {saveFailed && (
            <div className="banner hard">
              The draft is not being saved: the browser refused to write to localStorage (private
              mode, or the quota is full). Everything here still works, but a reload will bring back
              the example.
            </div>
          )}
          {recovered && (
            <div className="banner soft">
              The previous draft could not be read and was set aside under{" "}
              <code>laya-web-poc/session.bak</code> — you can recover it from DevTools → Application →
              Local Storage. The example is loaded instead.
            </div>
          )}

          <ModelPicker value={model} busy={running} onChange={(m) => patch({ model: m })} />

          <section className="card">
            <h2>1 · Text</h2>
            <label htmlFor="state-text">The text to analyse (state)</label>
            <textarea
              id="state-text"
              rows={10}
              value={text}
              placeholder="Paste the text…"
              onChange={(e) => patch({ text: e.target.value })}
            />
            <div className="note-row">
              <span className="muted">
                {budget === null
                  ? budgetError
                    ? `could not compute the budget: ${budgetError}`
                    : "text tokens: —"
                  : `text tokens: ${budget.stateTokens} · ${budget.worstStateUsed} reach the model` +
                    (!budget.anyTruncated && headroom !== null
                      ? ` (${headroom} of ${budget.maxLen} spare in the longest sequence)`
                      : "")}
              </span>
              {budget?.anyTruncated && (
                <span className="warn">
                  the tail of the text is being cut — which question, and by how much, is on its card.
                  {neededBudget ? (
                    <>
                      {" "}
                      <button className="link-btn" onClick={() => patch({ maxLen: neededBudget })}>
                        raise the budget to {neededBudget} so all {budget.stateTokens} tokens fit
                      </button>
                      {neededBudget > budget.trainedMaxLen && " (past the trained length — see below)"}
                    </>
                  ) : (
                    ` even ${BUDGETS[BUDGETS.length - 1]} tokens would not hold all ${budget.stateTokens}.`
                  )}
                </span>
              )}
            </div>
            {budget && (
              <ContextBudget
                value={maxLen}
                resolved={budget.maxLen}
                trained={budget.trainedMaxLen}
                onChange={(v) => patch({ maxLen: v })}
              />
            )}
          </section>

          <section className="card">
            <h2>2 · Task</h2>
            <label htmlFor="task-text">How to read the text</label>
            <textarea
              id="task-text"
              rows={2}
              value={task}
              placeholder="e.g. Read the text as a literary critic"
              onChange={(e) => patch({ task: e.target.value })}
            />
            <div className="field">
              <label>Where the task goes</label>
              <div className="radio-row">
                {(Object.keys(FRAMING_LABELS) as FramingMode[]).map((m) => (
                  <label key={m} className="radio">
                    <input type="radio" name="framing" checked={framing === m} onChange={() => patch({ framing: m })} />
                    {FRAMING_LABELS[m]}
                  </label>
                ))}
              </div>
              <div className="field-note">
                The question wording shares {budget?.headMaxLen ?? 192} tokens with the options, and
                on overflow it is cut <b>from the end</b> — so a long task in front eats the question
                itself. In the text it draws on the shared budget instead, and leaves the question alone.
              </div>
            </div>
          </section>

          <section className="card">
            <h2>3 · Questions</h2>
            {questions.length === 0 && <div className="empty-note">No questions yet — add the first one below.</div>}
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
                reset to the example
              </button>
            </div>
          </section>

          <section className="card run-card">
            <button
              className="btn primary"
              disabled={running || laya.phase !== "ready" || !questions.length || blockers > 0}
              onClick={doRun}
            >
              Get answers
            </button>
            {running && <button className="btn" onClick={() => abortRef.current?.abort()}>Stop</button>}
            <BatchPicker
              value={batchSize}
              safe={laya.spec.batchSafe}
              disabled={running}
              onChange={(v) => patch({ batchSize: v })}
            />
            <span className={`muted${status?.error ? " error" : ""}`}>
              {status?.text ??
                (laya.phase !== "ready"
                  ? "the model is loading…"
                  : blockers > 0
                    ? `${blockers} question${blockers === 1 ? "" : "s"} need fixing — see the cards above`
                    : "the model is ready")}
            </span>
          </section>

          {(shown.length > 0 || running) && (
            <section className="card">
              <h2>
                Answers
                {result?.aborted && (
                  <span className="ans-partial"> · run stopped: {result.landed.length} of {result.requested}</span>
                )}
              </h2>
              <div className="calib-note">
                The temperatures were fitted by the original author on the fp32 model and were not
                refitted after quantisation. Comparing these probabilities against each other is fine;
                reading them as frequencies is not, until they are refitted on your own labelled data.
              </div>
              {shown.map((l) => (
                <AnswerCard key={l.question.uid} question={l.question} index={l.index} answer={l.answer} telemetry={l.telemetry} />
              ))}
            </section>
          )}

          <section className="card">
            <h2>Metrics</h2>
            <MetricsPanel
              files={laya.files}
              stages={laya.stages}
              cache={laya.cache}
              session={laya.session}
              spec={laya.spec}
              ready={laya.phase === "ready"}
              run={result ? { ...result.telemetry, aborted: result.aborted, requested: result.requested } : null}
            />
            <div className="metrics-actions">
              <button className="btn" disabled={!result} onClick={exportJson}>Export the run as JSON</button>
            </div>
          </section>
        </main>

        <footer>
          <p>
            Built on <a href="https://github.com/nvkudva/laya-web" target="_blank" rel="noreferrer">nvkudva/laya-web</a>{" "}
            (the runtime port) and the weights <a href="https://huggingface.co/nvkudva/laya-web-q8" target="_blank" rel="noreferrer">nvkudva/laya-web-q8</a>{" "}
            · base model <a href="https://huggingface.co/convaiinnovations/laya" target="_blank" rel="noreferrer">convaiinnovations/laya</a>{" "}
            · Apache-2.0
          </p>
        </footer>
      </div>

      <LoadOverlay spec={laya.spec} phase={laya.phase} error={laya.error} files={laya.files} onRetry={laya.retry} />
    </>
  );
}


/** Defaults to fitting the text, because truncating it silently is the worse default
 *  and the long range has now been measured rather than assumed.
 *
 *  The 512 is not a property of the graph: the exported encoder declares a dynamic
 *  `seq_len` axis and uses rotary embeddings instead of a learned position table, and
 *  ModernBERT-large is an 8192-context backbone. What 512 marks is the length Laya was
 *  trained and temperature-fitted at. Past it the forward pass still runs and still
 *  returns numbers between 0 and 1 — which is exactly the problem, because nothing
 *  about them has been *calibrated* there. What has been measured is that the model
 *  genuinely reads the extra tokens: on a 653-token review whose last paragraph
 *  retracts the rest, p(recommends) is 64.3% at 512 (retraction truncated away) and
 *  3.1% at 1024 (retraction in range). Cost measured on the same machine: 11 s at 512,
 *  16 s at 1024 — ModernBERT alternates local and global attention, so the growth is
 *  far gentler than the quadratic worst case. */
function ContextBudget({ value, resolved, trained, onChange }: {
  value: number | null; resolved: number; trained: number; onChange: (v: number | null) => void;
}) {
  return (
    <div className="field">
      <label htmlFor="ctx-budget">Sequence budget</label>
      <div className="radio-row">
        <select
          id="ctx-budget"
          value={value === null ? "auto" : value}
          onChange={(e) => onChange(e.target.value === "auto" ? null : Number(e.target.value))}
        >
          <option value="auto">Fit the text — now {resolved}</option>
          {BUDGETS.map((b) => (
            <option key={b} value={b}>
              {b} tokens{b === trained ? " — as trained" : " — beyond training"}
            </option>
          ))}
        </select>
        {resolved > trained && (
          <span className="warn">
            {resolved} &gt; {trained}: past the length this checkpoint was trained and
            temperature-fitted at. The long range does work — on a 653-token review whose closing
            paragraph retracts everything before it, p(recommends) went 64.3% at 512, where the
            retraction is truncated away, to 3.1% at 1024, where it is not. What that measurement
            does not show is whether the probabilities stay calibrated out here, only that the
            model reads the text. Treat a long-context number as an ordering.
          </span>
        )}
      </div>
      <div className="field-note">
        The text gets whatever this leaves after the question head. Longer costs more time, but
        less than quadratically: 11 s at 512 and 16 s at 1024 on a 16-core laptop, because
        ModernBERT alternates local and global attention layers.
      </div>
    </div>
  );
}


/** Switching checkpoint means another few hundred megabytes, so the picker says what
 *  each one costs and what it is for rather than listing two opaque names. */

/** Questions per forward pass. 1 is one at a time.
 *
 *  Laya answers a whole request in one pass natively, and that is where its published
 *  39.5 ms for one question against 158.6 ms for ten comes from. Whether a given export
 *  can is a property of the export, and the two measured here disagree: the English q8
 *  graph refuses a batch outright, and the multilingual int8 graph accepts one and
 *  answers differently, because dynamic quantization takes activation scales per tensor
 *  and so couples the rows. That one shifted probabilities by up to 21 points and
 *  flipped a decision.
 *
 *  So the control exists, defaults to off, and says which case the current model is in
 *  rather than letting a silent one look like a fast one. */
function BatchPicker({ value, safe, disabled, onChange }: {
  value: number; safe: boolean; disabled: boolean; onChange: (v: number) => void;
}) {
  return (
    <span className="batch-picker">
      <label htmlFor="batch-size">Questions per pass</label>
      <select
        id="batch-size"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        <option value={1}>1 — one at a time</option>
        {[2, 4, 8, 16].map((n) => <option key={n} value={n}>{n}</option>)}
      </select>
      {value > 1 && !safe && (
        <span className="warn">
          this export has not been shown to answer the same batched as it does alone. It
          will either refuse the batch with an error, or return numbers that differ —
          measured at up to 21 points elsewhere, enough to flip a decision. Compare
          against 1 before believing a batched answer.
        </span>
      )}
    </span>
  );
}

function ModelPicker({ value, onChange, busy }: {
  value: ModelId; onChange: (m: ModelId) => void; busy: boolean;
}) {
  const spec = MODELS[value];
  // Read once: navigator.gpu does not appear mid-session, and re-checking on every
  // render would only invite the reader to think it might.
  const [gpu] = useState(webgpuAvailable);
  return (
    <section className="card">
      <h2>0 · Model</h2>
      <div className="radio-row">
        <select
          id="model-picker"
          value={value}
          disabled={busy}
          onChange={(e) => onChange(e.target.value as ModelId)}
        >
          {Object.values(MODELS).map((m) => (
            <option key={m.id} value={m.id} disabled={m.requiresWebGPU && !gpu}>
              {m.label}{m.requiresWebGPU && !gpu ? " — unavailable here" : ""}
            </option>
          ))}
        </select>
        {busy && <span className="muted">finish the run before switching</span>}
      </div>
      <div className="field-note">{spec.note}</div>
      {!gpu && (
        <div className="field-note">
          This browser does not expose <code>navigator.gpu</code>, so the fp16 build is
          greyed out. Running it on wasm instead would emulate half precision in software
          at roughly 3.9 s per sequence, which is why it is refused rather than offered
          as a slow option.
        </div>
      )}
      <div className="field-note">
        Questions run one at a time, on both builds. Laya answers a whole batch in one
        pass natively, but neither export here can: the English one refuses a batch
        outright, and the multilingual one accepts it and quietly answers differently —
        its activation scales are derived per tensor, so one question's numbers depend
        on the others sharing the pass. Measured at up to 21 percentage points, enough
        to flip a decision.
      </div>
    </section>
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
      <button className="btn" onClick={() => onAdd(type)}>Add question</button>
    </>
  );
}
