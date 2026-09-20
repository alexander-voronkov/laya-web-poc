import type { QuestionDef, Questions } from "./laya/types";

export interface OptionItem {
  label: string;
  description: string;
}

export type QuestionType = "noul" | "choice" | "score";

export interface QuestionItem {
  id: string;
  type: QuestionType;
  text: string;
  /** Optional per-question hint appended to the question instructions. */
  hint: string;
  /** noul: what "yes" and "no" mean. Both optional; both cost a handful of tokens
   *  and consistently sharpen the answer, so the editor nudges towards filling them. */
  criteriaTrue: string;
  criteriaFalse: string;
  /** choice options. */
  options: OptionItem[];
  /** score levels, low to high. */
  levels: OptionItem[];
}

export const TYPE_LABELS: Record<QuestionType, string> = {
  noul: "Бинарный (0–100)",
  choice: "Выбор из списка",
  score: "Шкала (ожидание по уровням)",
};

/** Where the task prompt is spliced in.
 *
 *  It matters more than it looks. `instructions` shares a 192-token head budget with
 *  the options, and the clamp cuts from the *end* -- so a long task prompt sitting in
 *  front of the question deletes the question, silently. `state` puts the framing in
 *  the ~300 tokens left over instead, where it is also paid for once per question
 *  rather than re-encoded inside the head. */
export type FramingMode = "instructions" | "state" | "both";

export const FRAMING_LABELS: Record<FramingMode, string> = {
  instructions: "в формулировку каждого вопроса",
  state: "в начало текста (state)",
  both: "и туда, и туда",
};

export function newQuestion(type: QuestionType, n: number): QuestionItem {
  const q: QuestionItem = {
    id: `q${n}`,
    type,
    text: "",
    hint: "",
    criteriaTrue: "",
    criteriaFalse: "",
    options: [],
    levels: [],
  };
  if (type === "choice") q.options = [{ label: "", description: "" }, { label: "", description: "" }];
  if (type === "score")
    q.levels = [
      { label: "low", description: "" },
      { label: "medium", description: "" },
      { label: "high", description: "" },
    ];
  return q;
}

/** Join non-empty parts into one instruction string, each ending with a sentence mark. */
function sentences(parts: string[]): string {
  return parts
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => (/[.!?…]$/.test(p) ? p : p + "."))
    .join(" ");
}

/** instructions = [<задача>] <вопрос> <уточнение>. */
export function buildInstructions(task: string, q: QuestionItem, mode: FramingMode): string {
  const framing = mode === "instructions" || mode === "both" ? task : "";
  return sentences([framing, q.text, q.hint]);
}

/** state = [<задача>] + текст. A visible separator, not a JSON wrapper: the reference
 *  serializer would escape the whole thing and spend tokens on punctuation. */
export function buildState(task: string, text: string, mode: FramingMode): string {
  const framing = mode === "state" || mode === "both" ? task.trim() : "";
  const body = text.trim();
  if (!framing) return body;
  if (!body) return framing;
  return `${framing}\n\n---\n\n${body}`;
}

/** noul criteria. An empty string must become an absent key rather than an empty one:
 *  renderOptions falls back to the default wording either way, but an absent key is
 *  what the Python reference produces and what the golden dump covers. */
function noulCriteria(q: QuestionItem): Record<string, string> | null {
  const c: Record<string, string> = {};
  if (q.criteriaTrue.trim()) c["true"] = q.criteriaTrue.trim();
  if (q.criteriaFalse.trim()) c["false"] = q.criteriaFalse.trim();
  return Object.keys(c).length ? c : null;
}

/** Map our UI question onto the Jev request shape the model port accepts. */
export function toQuestionDef(task: string, q: QuestionItem, mode: FramingMode): QuestionDef {
  const instructions = buildInstructions(task, q, mode);
  if (q.type === "choice") {
    return {
      type: "choice",
      instructions,
      criteria: Object.fromEntries(q.options.map((o) => [o.label.trim(), o.description.trim() || null])),
    };
  }
  if (q.type === "score") {
    return {
      type: "score",
      instructions,
      criteria: q.levels.map((l) => {
        const d = l.description.trim();
        return d ? `${l.label.trim()}: ${d}` : l.label.trim();
      }),
    };
  }
  return { type: "noul", instructions, criteria: noulCriteria(q) };
}

export function toRequest(task: string, qs: QuestionItem[], mode: FramingMode): Questions {
  return Object.fromEntries(qs.map((q) => [q.id, toQuestionDef(task, q, mode)]));
}

export function countOptions(q: QuestionItem): number {
  if (q.type === "noul") return 2;
  return (q.type === "score" ? q.levels : q.options).filter((o) => o.label.trim()).length;
}

export interface ValidationIssue {
  id: string;
  problem: string;
}

export function validateQuestions(qs: QuestionItem[]): ValidationIssue[] {
  const out: ValidationIssue[] = [];
  const seen = new Set<string>();
  for (const q of qs) {
    if (seen.has(q.id)) out.push({ id: q.id, problem: "ключ вопроса повторяется" });
    seen.add(q.id);
    if (!q.text.trim()) out.push({ id: q.id, problem: "пустой текст вопроса" });
    if (q.type === "choice" || q.type === "score") {
      const items = q.type === "score" ? q.levels : q.options;
      const noun = q.type === "score" ? "уровня" : "варианта";
      const labels = items.map((o) => o.label.trim()).filter(Boolean);
      if (labels.length < 2) out.push({ id: q.id, problem: `нужно минимум 2 непустых ${noun}` });
      else if (new Set(labels).size !== labels.length)
        out.push({ id: q.id, problem: q.type === "score" ? "уровни повторяются" : "варианты повторяются" });
      // criteria is a dict keyed by label, and the documented ceiling is 255
      if (labels.length > 255) out.push({ id: q.id, problem: "больше 255 вариантов" });
    }
  }
  return out;
}

export interface Advice {
  /** Question this is about, or null for the whole run. */
  id: string | null;
  text: string;
  /** `hard` advice means the numbers below are not worth reading. */
  level: "hard" | "soft";
}

// Anything outside Latin-1 + Latin Extended-A/B, punctuation and currency.
const NON_LATIN = /[^\u0000-ɏ -⁯₠-₿]/;

/** Documented limits of this checkpoint, surfaced against what is actually typed.
 *  Every item corresponds to a measured number on the model card, not a hunch. */
export function advice(task: string, text: string, qs: QuestionItem[]): Advice[] {
  const out: Advice[] = [];

  // English-only, and it does not fail gracefully: the model card reports 0.000
  // accuracy at 0.952 mean confidence on Khmer. Being wrong while confident is
  // exactly the failure a confidence threshold cannot catch, so this is `hard`.
  const foreign = [text, task, ...qs.map((q) => q.text + q.hint)].some((s) => NON_LATIN.test(s));
  if (foreign) {
    out.push({
      id: null,
      level: "hard",
      text:
        "Найдены нелатинские символы. Этот чекпойнт обучен только на английском и на других " +
        "языках остаётся уверенным, будучи неправым (0.000 точности при 0.952 средней уверенности " +
        "на кхмерском) — отфильтровать такие ответы по confidence невозможно. Мультиязычная Laya " +
        "существует, но в ONNX/q8 для браузера её пока никто не выложил.",
    });
  }

  for (const q of qs) {
    if (q.type === "choice" && countOptions(q) > 10) {
      out.push({
        id: q.id,
        level: "soft",
        text:
          `${countOptions(q)} вариантов: температура бакета choice:11+ равна 0.1006, распределение ` +
          "сжимается почти в one-hot, и на 192 токена головы остаётся по несколько токенов на метку.",
      });
    }
    if (q.type === "score") {
      out.push({
        id: q.id,
        level: "soft",
        text: "score — самый слабый примитив этого чекпойнта (SST-5 0.372). Где хватает «да/нет», бинарный вопрос разделяет лучше.",
      });
    }
    if (q.type === "noul" && (!q.criteriaTrue.trim() || !q.criteriaFalse.trim())) {
      out.push({
        id: q.id,
        level: "soft",
        text: "не заданы описания «да» и «нет» — они стоят несколько токенов и заметно повышают разделимость.",
      });
    }
    if (q.text.trim().length > 80 && /\b(and|or)\b/i.test(q.text)) {
      out.push({
        id: q.id,
        level: "soft",
        text: "похоже на составной вопрос. Один предикат на вопрос: составная формулировка сглаживает шкалу — спросите отдельно и соедините результаты в коде.",
      });
    }
  }
  return out;
}

/** Example content so the PoC is runnable immediately. English, because the model is. */
export function seedQuestions(): { questions: QuestionItem[]; counter: number } {
  const q = (p: Partial<QuestionItem> & Pick<QuestionItem, "id" | "type" | "text">): QuestionItem => ({
    hint: "",
    criteriaTrue: "",
    criteriaFalse: "",
    options: [],
    levels: [],
    ...p,
  });
  return {
    counter: 3,
    questions: [
      q({
        id: "q1",
        type: "noul",
        text: "Does the narrator's account contradict itself?",
        criteriaTrue: "yes, the text undercuts its own claims",
        criteriaFalse: "no, the account is internally consistent",
      }),
      q({
        id: "q2",
        type: "choice",
        text: "Which register dominates the passage?",
        options: [
          { label: "ironic", description: "the tone undercuts what is stated" },
          { label: "earnest", description: "the tone supports what is stated" },
          { label: "clinical", description: "detached, without evaluation" },
        ],
      }),
      q({
        id: "q3",
        type: "score",
        text: "How strongly does the passage commit to its central claim?",
        levels: [
          { label: "hedged", description: "qualified throughout" },
          { label: "measured", description: "asserted with caveats" },
          { label: "emphatic", description: "asserted without reservation" },
        ],
      }),
    ],
  };
}

export const SEED_TASK = "Read the text as a literary critic.";

export const SEED_TEXT =
  "I have never, in all my years at the firm, seen a proposal handled with such care. " +
  "Every figure was checked twice, and the committee — men of unimpeachable judgement, " +
  "as they are so often described, not least by themselves — approved it in under nine minutes. " +
  "I am told the earlier version, the one I drafted, contained an error. I do not recall the error. " +
  "I recall only that it was mine, and that this was said to settle the matter.";
