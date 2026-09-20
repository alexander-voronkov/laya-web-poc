import type { QuestionDef } from "./laya/types";

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

export function newQuestion(type: QuestionType, n: number): QuestionItem {
  const q: QuestionItem = { id: `q${n}`, type, text: "", hint: "", options: [], levels: [] };
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

/** instructions = <задача> <вопрос> <уточнение> -- the task frames the whole run,
 *  the hint refines this particular question. */
export function buildInstructions(task: string, question: string, hint: string): string {
  return sentences([task, question, hint]);
}

/** Map our UI question onto the Jev request shape the model port accepts. */
export function toQuestionDef(task: string, q: QuestionItem): QuestionDef {
  const instructions = buildInstructions(task, q.text, q.hint);
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
  return { type: "noul", instructions, criteria: null };
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
  for (const q of qs) {
    if (!q.text.trim()) out.push({ id: q.id, problem: "пустой текст вопроса" });
    if (q.type === "choice" || q.type === "score") {
      const items = q.type === "score" ? q.levels : q.options;
      const noun = q.type === "score" ? "уровня" : "варианта";
      const labels = items.map((o) => o.label.trim()).filter(Boolean);
      if (labels.length < 2) out.push({ id: q.id, problem: `нужно минимум 2 непустых ${noun}` });
      else if (new Set(labels).size !== labels.length)
        out.push({ id: q.id, problem: q.type === "score" ? "уровни повторяются" : "варианты повторяются" });
    }
  }
  return out;
}

/** Example content so the PoC is runnable immediately. */
export function seedQuestions(): { questions: QuestionItem[]; counter: number } {
  return {
    questions: [
      {
        id: "q1",
        type: "noul",
        text: "Will the project meet its revised timeline?",
        hint: "",
        options: [],
        levels: [],
      },
      {
        id: "q2",
        type: "choice",
        text: "What is the main risk highlighted in the text?",
        hint: "",
        options: [
          { label: "schedule", description: "the timeline is unrealistic" },
          { label: "funding", description: "the budget cap is too low" },
          { label: "governance", description: "committee overhead slows decisions" },
        ],
        levels: [],
      },
      {
        id: "q3",
        type: "score",
        text: "How confident is the text about the project's success?",
        hint: "",
        options: [],
        levels: [
          { label: "low", description: "" },
          { label: "medium", description: "" },
          { label: "high", description: "" },
        ],
      },
    ],
    counter: 3,
  };
}
