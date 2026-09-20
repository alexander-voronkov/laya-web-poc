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
function isSession(v: unknown): v is Session {
  if (typeof v !== "object" || v === null) return false;
  const s = v as Partial<Session>;
  return (
    s.version === VERSION &&
    typeof s.text === "string" &&
    typeof s.task === "string" &&
    (s.framing === "instructions" || s.framing === "state" || s.framing === "both") &&
    Array.isArray(s.questions) &&
    typeof s.counter === "number"
  );
}

export function loadSession(): Session {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaultSession();
    const parsed: unknown = JSON.parse(raw);
    if (!isSession(parsed)) return defaultSession();
    // Fields added after a session was saved would otherwise arrive as undefined and
    // reach .trim() in the request builder.
    return {
      ...parsed,
      questions: parsed.questions.map((q) => ({
        hint: "",
        criteriaTrue: "",
        criteriaFalse: "",
        options: [],
        levels: [],
        ...q,
      })),
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
