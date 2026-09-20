import type { QuestionDef, Questions } from "./laya/types";

export interface OptionItem {
  label: string;
  description: string;
}

export type QuestionType = "noul" | "choice" | "score";

export interface QuestionItem {
  /** Stable React identity. Separate from `id` on purpose: `id` is an editable field,
   *  and keying the list by it remounts the card on every keystroke, which takes the
   *  focus out of the very input being typed into. Never shown to the model. */
  uid: string;
  /** The key this question's answer appears under, in the request and the export. */
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
  noul: "Binary (0–100)",
  choice: "Pick one",
  score: "Ordered scale (expected level)",
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
  instructions: "into every question's wording",
  state: "in front of the text (state)",
  both: "both places",
};

let uidSeq = 0;
export const newUid = () => `u${Date.now().toString(36)}-${(uidSeq++).toString(36)}`;

export function newQuestion(type: QuestionType, n: number): QuestionItem {
  const q: QuestionItem = {
    uid: newUid(),
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

/** instructions = [<task>] <question> <hint>. */
export function buildInstructions(task: string, q: QuestionItem, mode: FramingMode): string {
  const framing = mode === "instructions" || mode === "both" ? task : "";
  return sentences([framing, q.text, q.hint]);
}

/** state = [<task>] + text. A visible separator, not a JSON wrapper: the reference
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

/** The options that will actually be scored, in the order the model will see them.
 *
 *  One definition, used by the request builder, the budget analysis and the answer
 *  cards alike. They diverged before: validation counted `filter(Boolean)` labels
 *  while the request sent every row, so a blank row became a real scored option
 *  (`"level 3: "`) that the cards then filtered out of the display — and for `score`,
 *  filtering the *display* while the model indexed the *unfiltered* list put every
 *  label next to its neighbour's probability. Anything blank is dropped here, once. */
export function scoredItems(q: QuestionItem): OptionItem[] {
  const items = q.type === "score" ? q.levels : q.type === "choice" ? q.options : [];
  return items
    .map((o) => ({ label: o.label.trim(), description: o.description.trim() }))
    .filter((o) => o.label);
}

/** Map our UI question onto the Jev request shape the model port accepts. */
export function toQuestionDef(task: string, q: QuestionItem, mode: FramingMode): QuestionDef {
  const instructions = buildInstructions(task, q, mode);
  if (q.type === "choice") {
    return {
      type: "choice",
      instructions,
      criteria: Object.fromEntries(scoredItems(q).map((o) => [o.label, o.description || null])),
    };
  }
  if (q.type === "score") {
    return {
      type: "score",
      instructions,
      criteria: scoredItems(q).map((l) => (l.description ? `${l.label}: ${l.description}` : l.label)),
    };
  }
  return { type: "noul", instructions, criteria: noulCriteria(q) };
}

export function toRequest(task: string, qs: QuestionItem[], mode: FramingMode): Questions {
  return Object.fromEntries(qs.map((q) => [q.id, toQuestionDef(task, q, mode)]));
}

export function countOptions(q: QuestionItem): number {
  return q.type === "noul" ? 2 : scoredItems(q).length;
}

export interface ValidationIssue {
  /** The question's stable uid, not its editable key. */
  uid: string;
  problem: string;
}

export function validateQuestions(qs: QuestionItem[]): ValidationIssue[] {
  const out: ValidationIssue[] = [];
  const seen = new Set<string>();
  for (const q of qs) {
    // An empty id would become the key "" in the request and in the exported answers.
    if (!q.id.trim()) out.push({ uid: q.uid, problem: "the question key is empty" });
    else if (seen.has(q.id)) out.push({ uid: q.uid, problem: "the question key is used twice" });
    seen.add(q.id);
    if (!q.text.trim()) out.push({ uid: q.uid, problem: "the question text is empty" });
    if (q.type === "choice" || q.type === "score") {
      const items = q.type === "score" ? q.levels : q.options;
      const noun = q.type === "score" ? "levels" : "options";
      const labels = scoredItems(q).map((o) => o.label);
      if (labels.length < 2) out.push({ uid: q.uid, problem: `at least 2 labelled ${noun} are needed` });
      else if (new Set(labels).size !== labels.length)
        out.push({ uid: q.uid, problem: `duplicate ${noun}` });
      // Blank rows are dropped from the request rather than sent as empty options, so
      // this is not a crash -- but a row the user typed a description into and never
      // labelled would vanish without a word, which is the wrong kind of quiet.
      if (labels.length !== items.length) {
        const blanks = items.length - labels.length;
        out.push({
          uid: q.uid,
          problem: `${blanks} unlabelled ${blanks === 1 ? noun.slice(0, -1) : noun} — not sent to the model; label or remove`,
        });
      }
      // criteria is a dict keyed by label, and the documented ceiling is 255
      if (labels.length > 255) out.push({ uid: q.uid, problem: "more than 255 options" });
    }
  }
  return out;
}

export interface Advice {
  /** Question this is about, by stable uid, or null for the whole run. */
  uid: string | null;
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
  // Every string that ends up inside the sequence, not just the obvious two. Option
  // labels, their descriptions and the yes/no criteria all pass through renderOptions
  // and are scored, so a Russian option list under an English question was reaching
  // the model with no banner at all.
  const inRequest = [
    text,
    task,
    ...qs.flatMap((q) => [
      q.text,
      q.hint,
      q.criteriaTrue,
      q.criteriaFalse,
      ...scoredItems(q).flatMap((o) => [o.label, o.description]),
    ]),
  ];
  const foreign = inRequest.some((s) => NON_LATIN.test(s));
  if (foreign) {
    out.push({
      uid: null,
      level: "hard",
      text:
        "Non-Latin script found. Laya as a family covers 100+ languages — but that is the " +
        "multilingual checkpoint (mmBERT-base), and the one running here is the English root " +
        "(ModernBERT-large), which its own card sums up as \"English only on root\". It is not " +
        "useless elsewhere: it clears 3x random in 23 of 51 languages. What it does badly is " +
        "non-Latin script specifically, and it does it while staying confident — 0.000 accuracy " +
        "at 0.952 mean confidence on Khmer — so a confidence threshold cannot filter these " +
        "answers out. Cyrillic is in that bucket. The multilingual checkpoint also reads 1024 " +
        "tokens instead of 512.",
    });
  }

  for (const q of qs) {
    if (q.type === "choice" && countOptions(q) > 10) {
      out.push({
        uid: q.uid,
        level: "soft",
        text:
          `${countOptions(q)} options: the choice:11+ temperature is 0.1006, which sharpens the ` +
          "distribution close to one-hot, and 192 head tokens leave only a few tokens per label.",
      });
    }
    if (q.type === "score") {
      out.push({
        uid: q.uid,
        level: "soft",
        text: "score is the weakest primitive on this checkpoint (SST-5 0.372). Where a yes/no question will do, it separates better.",
      });
    }
    if (q.type === "noul" && (!q.criteriaTrue.trim() || !q.criteriaFalse.trim())) {
      out.push({
        uid: q.uid,
        level: "soft",
        text: "no descriptions for yes and no — they cost a handful of tokens and measurably sharpen the answer.",
      });
    }
    if (q.text.trim().length > 80 && /\b(and|or)\b/i.test(q.text)) {
      out.push({
        uid: q.uid,
        level: "soft",
        text: "looks like a compound question. One predicate per question: a compound wording flattens the distribution — ask separately and combine the results in code.",
      });
    }
  }
  return out;
}

/** Example content so the PoC is runnable immediately. English, because the model is. */
export function seedQuestions(): { questions: QuestionItem[]; counter: number } {
  const q = (p: Partial<QuestionItem> & Pick<QuestionItem, "id" | "type" | "text">): QuestionItem => ({
    uid: newUid(),
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
