import "./styles.css";
import { MODELS_BASE, NOMINAL_TOTAL_BYTES } from "./config";
import {
  LayaSession,
  cachedWeightBytes,
  loadCore,
  type Core,
  type LoadProgress,
  type LoadStage,
} from "./laya/session";
import type { Answer } from "./laya/types";
import { renderQuestions } from "./form";
import { answerCard } from "./render";
import { $, h, mb, nextPaint } from "./dom";
import {
  countOptions,
  newQuestion,
  seedQuestions,
  toQuestionDef,
  validateQuestions,
  type QuestionItem,
  type QuestionType,
} from "./state";

interface RunRecord {
  id: string;
  ms: number;
  inputTokens: number;
  options: number;
}

interface RunSnapshot {
  at: string;
  state: string;
  task: string;
  questions: QuestionItem[];
  answers: Record<string, Answer>;
  records: RunRecord[];
  totalMs: number;
}

const els = {
  stateText: $("state-text") as HTMLTextAreaElement,
  stateTokens: $("state-tokens"),
  stateWarn: $("state-warn"),
  taskText: $("task-text") as HTMLTextAreaElement,
  questionsRoot: $("questions"),
  newQType: $("new-q-type") as HTMLSelectElement,
  addQ: $("add-q") as HTMLButtonElement,
  run: $("run") as HTMLButtonElement,
  runStatus: $("run-status"),
  resultsCard: $("results-card"),
  results: $("results"),
  metrics: $("metrics"),
  exportJson: $("export-json") as HTMLButtonElement,
  overlay: $("overlay"),
  ovFiles: $("ov-files"),
  ovStatus: $("ov-status"),
  ovRetry: $("ov-retry") as HTMLButtonElement,
};

const seed = seedQuestions();
let questions: QuestionItem[] = seed.questions;
let qCounter = seed.counter;
let core: Core | null = null;
let session: LayaSession | null = null;
let running = false;
let lastRun: RunSnapshot | null = null;

// ---- load-time metrics ----------------------------------------------------
const loadFiles: Record<string, LoadProgress> = {};
const stages: Partial<Record<LoadStage, number>> = {};
let cacheBytes: number | null = null;
let loadFailed = false;

// ---- boot -----------------------------------------------------------------

boot().catch((e) => overlayError(e));

async function boot(): Promise<void> {
  renderQuestionsList();
  updateRunButton();

  els.stateText.addEventListener("input", () => {
    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(updateTokenCount, 150);
  });
  els.addQ.addEventListener("click", () => {
    questions.push(newQuestion(els.newQType.value as QuestionType, ++qCounter));
    renderQuestionsList();
  });
  els.run.addEventListener("click", run);
  els.exportJson.addEventListener("click", exportJson);
  els.ovRetry.addEventListener("click", () => location.reload());

  try {
    setOvStatus("загрузка токенизатора и конфига…");
    const t0 = performance.now();
    core = await loadCore(MODELS_BASE);
    stages.core = performance.now() - t0;
    updateTokenCount();

    setOvStatus("загрузка весов модели…");
    session = await LayaSession.load({
      base: MODELS_BASE,
      core,
      onProgress: onProgress,
      onStage: (stage, ms) => {
        stages[stage] = ms;
        updateOverlay();
      },
    });
    hideOverlay();
    updateRunButton();
    setRunStatus("модель готова");
    updateMetrics([], 0);
    cachedWeightBytes().then((n) => {
      cacheBytes = n;
      updateMetrics(lastRun?.records ?? [], 0);
    });
  } catch (e) {
    overlayError(e);
  }
}

// ---- questions editor -----------------------------------------------------

let debounceTimer: number | undefined;

function renderQuestionsList(): void {
  const issues = validateQuestions(questions);
  const invalid = new Map(issues.map((i) => [i.id, i.problem]));
  renderQuestions(els.questionsRoot, questions, { onChanged: renderQuestionsList }, invalid);
  updateRunButton();
}

// ---- token counting -------------------------------------------------------

function updateTokenCount(): void {
  if (!core) return;
  const n = core.tok.encode(els.stateText.value, { add_special_tokens: false }).length;
  const budget = core.cfg.max_len;
  els.stateTokens.textContent = `токены текста: ${n} / ${budget}`;
  // the question head ([CLS] instructions [SEP] options [SEP]) also needs room in max_len
  const headroom = 96;
  els.stateWarn.classList.toggle("hidden", n <= budget - headroom);
}

// ---- run ------------------------------------------------------------------

async function run(): Promise<void> {
  if (!session || running) return;
  const state = els.stateText.value;
  const task = els.taskText.value.trim();

  const issues = validateQuestions(questions);
  if (issues.length) {
    renderQuestionsList();
    setRunStatus(`исправьте: ${issues.map((i) => `${i.id} — ${i.problem}`).join("; ")}`, true);
    return;
  }

  running = true;
  updateRunButton();
  els.resultsCard.classList.remove("hidden");
  els.results.replaceChildren();
  const records: RunRecord[] = [];
  const answers: Record<string, Answer> = {};
  const t0 = performance.now();
  try {
    for (let i = 0; i < questions.length; i++) {
      const q = questions[i];
      els.run.textContent = `Получить ответы (${i + 1}/${questions.length})…`;
      setRunStatus(`инференс: вопрос ${i + 1} из ${questions.length}…`);
      updateMetrics(records, performance.now() - t0);
      await nextPaint(); // let the UI paint before the blocking forward pass
      const tq = performance.now();
      const resp = await session.systemOne(state, { [q.id]: toQuestionDef(task, q) });
      const ms = performance.now() - tq;
      records.push({ id: q.id, ms, inputTokens: resp.usage.input_tokens, options: countOptions(q) });
      answers[q.id] = resp.answers[q.id];
      els.results.append(answerCard(q, i, answers[q.id], ms));
      updateMetrics(records, performance.now() - t0);
    }
    const totalMs = performance.now() - t0;
    lastRun = { at: new Date().toISOString(), state, task, questions: structuredClone(questions), answers, records, totalMs };
    els.exportJson.disabled = false;
    setRunStatus(`готово за ${(totalMs / 1000).toFixed(1)} с`);
  } catch (e) {
    setRunStatus(`ошибка: ${String((e as Error)?.message ?? e)}`, true);
  } finally {
    running = false;
    els.run.textContent = "Получить ответы";
    updateRunButton();
  }
}

function updateRunButton(): void {
  els.run.disabled = running || !session || questions.length === 0;
  if (!session && !loadFailed) setRunStatus("модель загружается…");
}

function setRunStatus(text: string, isError = false): void {
  els.runStatus.textContent = text;
  els.runStatus.classList.toggle("error", isError);
}

// ---- metrics panel --------------------------------------------------------

function updateMetrics(records: RunRecord[], runningMs: number): void {
  const files = Object.values(loadFiles);
  const loaded = files.reduce((a, f) => a + f.loaded, 0);
  const total = files.reduce((a, f) => a + (f.total || 0), 0);
  const weightsMs = stages.weights ?? 0;
  const encInit = stages["encoder-init"] ?? 0;
  const headInit = stages["head-init"] ?? 0;
  const initMs = encInit + headInit;
  const cachedAny = files.some((f) => f.cached);

  const mem = (performance as PerfWithMemory).memory;
  const inputTokens = records.reduce((a, r) => a + r.inputTokens, 0);
  const scores = records.reduce((a, r) => a + r.options, 0);
  const totalMs = lastRun?.totalMs ?? runningMs;
  const threads = session?.numThreads;
  const cores = session?.hardwareConcurrency ?? navigator.hardwareConcurrency;

  const rows: [string, string][] = [
    [
      "Загрузка модели",
      session || loadFailed
        ? `${mb(loaded) || "0.0"} МБ${total ? ` из ${mb(total)} МБ (${((loaded / total) * 100).toFixed(0)}%)` : ` (~${mb(NOMINAL_TOTAL_BYTES)} МБ)`} · ${(weightsMs / 1000).toFixed(1)} с${cachedAny ? " · из кэша браузера" : ""}`
        : `~${mb(NOMINAL_TOTAL_BYTES)} МБ, скачивается…`,
    ],
    [
      "Инициализация сессий",
      initMs ? `${(initMs / 1000).toFixed(2)} с (энкодер ${(encInit / 1000).toFixed(2)} с · голова ${(headInit / 1000).toFixed(2)} с)` : "—",
    ],
    [
      "Wasm-потоки / изоляция",
      `${threads ?? "?"} (ядер: ${cores ?? "?"}) · crossOriginIsolated: ${typeof crossOriginIsolated !== "undefined" && crossOriginIsolated ? "✓" : "✗"}`,
    ],
    [
      "Инференс по вопросам",
      records.length ? records.map((r) => `${r.id}: ${r.ms.toFixed(0)} мс`).join(" · ") : "—",
    ],
    ["Общее время прогона", records.length ? `${(totalMs / 1000).toFixed(2)} с` : "—"],
    [
      "Токены / оценки",
      records.length ? `входных токенов: ${inputTokens} · оценено вариантов: ${scores}` : "—",
    ],
    [
      "Память (JS heap)",
      mem
        ? `${mb(mem.usedJSHeapSize)} МБ использовано · wasm-память входит в JS heap (доступно в Chrome)`
        : "performance.memory недоступен в этом браузере (wasm-память живёт внутри JS heap)",
    ],
    [
      "Кэш модели",
      cacheBytes != null && cacheBytes > 0
        ? `${mb(cacheBytes)} МБ в Cache Storage — при следующем открытии скачивание почти мгновенное`
        : `весах кэшируются браузером (${mb(NOMINAL_TOTAL_BYTES)} МБ), второй заход быстрый`,
    ],
  ];

  els.metrics.replaceChildren(
    ...rows.map(([k, v]) => h("div", { class: "m-row" }, h("span", { class: "m-k" }, k), h("span", { class: "m-v" }, v))),
  );
}

interface PerfWithMemory extends Performance {
  memory?: { usedJSHeapSize: number };
}

// ---- JSON export ----------------------------------------------------------

function exportJson(): void {
  const run = lastRun;
  if (!run || !core) return;
  const files = Object.values(loadFiles);
  const mem = (performance as PerfWithMemory).memory;
  const payload = {
    generatedAt: run.at,
    model: {
      base: MODELS_BASE,
      quant: "q8 weight-only, onnxruntime-web/wasm",
      maxLen: core?.cfg.max_len ?? null,
      headMaxLen: core?.cfg.head_max_len ?? null,
    },
    input: { state: run.state, task: run.task, questions: run.questions },
    layaRequest: {
      state: run.state,
      questions: Object.fromEntries(run.questions.map((q) => [q.id, toQuestionDef(run.task, q)])),
    },
    answers: run.answers,
    metrics: {
      download: {
        bytes: files.reduce((a, f) => a + f.loaded, 0),
        totalBytes: files.reduce((a, f) => a + (f.total || 0), 0) || null,
        ms: stages.weights ?? null,
        cachedFiles: Object.values(loadFiles).filter((f) => f.cached).length,
      },
      sessionInitMs: { encoder: stages["encoder-init"] ?? null, head: stages["head-init"] ?? null },
      perQuestion: run.records,
      totalMs: run.totalMs,
      inputTokens: run.records.reduce((a, r) => a + r.inputTokens, 0),
      outputScores: run.records.reduce((a, r) => a + r.options, 0),
      wasmThreads: session?.numThreads ?? null,
      hardwareConcurrency: navigator.hardwareConcurrency ?? null,
      crossOriginIsolated: typeof crossOriginIsolated !== "undefined" ? crossOriginIsolated : null,
      jsHeapBytes: mem?.usedJSHeapSize ?? null,
      weightCacheBytes: cacheBytes,
    },
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `laya-run-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---- load overlay ---------------------------------------------------------

let lastOverlayPaint = 0;

function onProgress(p: LoadProgress): void {
  loadFiles[p.file] = p;
  const now = performance.now();
  if (now - lastOverlayPaint > 100 || p.loaded === p.total) {
    lastOverlayPaint = now;
    updateOverlay();
  }
}

function updateOverlay(): void {
  const files = Object.values(loadFiles);
  const rows = files.map((f) =>
    h("div", { class: "ov-file" },
      h("div", { class: "ov-file-head" },
        h("span", { class: "ov-name" }, f.file),
        h("span", { class: "ov-size" }, f.total ? `${mb(f.loaded)} / ${mb(f.total)} МБ` : `${mb(f.loaded)} МБ`),
      ),
      h("div", { class: "bar slim" },
        h("div", { class: `bar-fill${f.cached ? " cached" : ""}`, style: `width:${f.total ? Math.min(100, (f.loaded / f.total) * 100).toFixed(1) : 100}%` }),
      ),
    ),
  );
  els.ovFiles.replaceChildren(...rows);
}

function setOvStatus(text: string): void {
  els.ovStatus.textContent = text;
}

function hideOverlay(): void {
  els.overlay.classList.add("hidden");
}

function overlayError(e: unknown): void {
  loadFailed = true;
  setOvStatus(`Ошибка загрузки модели: ${String((e as Error)?.message ?? e)}`);
  els.ovRetry.classList.remove("hidden");
  updateRunButton();
}
