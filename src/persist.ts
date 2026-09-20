// There is no server. Everything the page holds between sessions lives in
// localStorage, under one versioned key.
//
// Every accessor is wrapped: localStorage throws outright in Safari private mode and
// under a blocked-site-data policy, and an unhandled throw here would take the whole
// page down before the first render. A failed read means "no saved session", never
// an error the user has to deal with.
import {
  SEED_TASK,
  SEED_TEXT,
  seedQuestions,
  type FramingMode,
  type OptionItem,
  type QuestionItem,
} from "./questions";

const KEY = "laya-web-poc/session";
const VERSION = 1;

export interface Session {
  version: number;
  text: string;
  task: string;
  framing: FramingMode;
  questions: QuestionItem[];
  counter: number;
}

export function defaultSession(): Session {
  const seed = seedQuestions();
  return {
    version: VERSION,
    text: SEED_TEXT,
    task: SEED_TASK,
    framing: "instructions",
    questions: seed.questions,
    counter: seed.counter,
  };
}

/** Old or hand-edited payloads are discarded rather than migrated: this is a
 *  prototype, and a half-understood shape flowing into the request builder is a
 *  worse outcome than losing a draft. */
interface RawSession {
  version: unknown;
  text: unknown;
  task: unknown;
  framing: unknown;
  questions: unknown;
  counter: unknown;
}

function isRawSession(v: unknown): v is RawSession {
  if (typeof v !== "object" || v === null) return false;
  const s = v as RawSession;
  return (
    s.version === VERSION &&
    typeof s.text === "string" &&
    typeof s.task === "string" &&
    (s.framing === "instructions" || s.framing === "state" || s.framing === "both") &&
    Array.isArray(s.questions) &&
    typeof s.counter === "number"
  );
}

const str = (v: unknown, fallback = "") => (typeof v === "string" ? v : fallback);

function toOptions(v: unknown): OptionItem[] {
  if (!Array.isArray(v)) return [];
  return v.map((o) => {
    const r = (o ?? {}) as Record<string, unknown>;
    return { label: str(r.label), description: str(r.description) };
  });
}

/** Field by field, never by spread. A saved question written before a field existed
 *  arrives without it, and `{...defaults, ...saved}` cannot fix that: the saved object
 *  has the key absent, not undefined, only when the writer omitted it -- and TypeScript
 *  types it as present either way, so the defaults are dead code that typechecks. The
 *  failure it would cause is an `undefined.trim()` inside the request builder. */
function toQuestion(v: unknown, index: number): QuestionItem {
  const r = (v ?? {}) as Record<string, unknown>;
  const type = r.type === "choice" || r.type === "score" ? r.type : "noul";
  return {
    id: str(r.id) || `q${index + 1}`,
    type,
    text: str(r.text),
    hint: str(r.hint),
    criteriaTrue: str(r.criteriaTrue),
    criteriaFalse: str(r.criteriaFalse),
    options: toOptions(r.options),
    levels: toOptions(r.levels),
  };
}

export function loadSession(): Session {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaultSession();
    const parsed: unknown = JSON.parse(raw);
    if (!isRawSession(parsed)) return defaultSession();
    return {
      version: VERSION,
      text: parsed.text as string,
      task: parsed.task as string,
      framing: parsed.framing as FramingMode,
      counter: parsed.counter as number,
      questions: (parsed.questions as unknown[]).map(toQuestion),
    };
  } catch {
    return defaultSession();
  }
}

export function saveSession(s: Session): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // quota, or storage blocked entirely -- the page keeps working, it just forgets
  }
}

export function clearSession(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* storage blocked */
  }
}
