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
  newUid,
  seedQuestions,
  type FramingMode,
  type OptionItem,
  type QuestionItem,
} from "./questions";

const KEY = "laya-web-poc/session";
// Holds a draft that could not be parsed, so a bad read is recoverable rather than
// silently overwritten by the next autosave.
const BACKUP_KEY = "laya-web-poc/session.bak";
const VERSION = 1;

export interface Session {
  version: number;
  /** Sequence budget, or null to use the checkpoint default. */
  maxLen: number | null;
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
    maxLen: null,
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
  maxLen?: unknown;
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
    uid: str(r.uid) || newUid(),
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

/** A draft we could not read is not the same as no draft.
 *
 *  Autosave fires 400ms after the seed becomes state, so returning the example on a
 *  failed read does not merely fail to restore the draft -- it overwrites it, and the
 *  user sees an ordinary first run with no sign that a moment ago their text was
 *  there. Keeping the raw string under a second key makes that recoverable. */
function setAside(raw: string): void {
  try {
    localStorage.setItem(BACKUP_KEY, raw);
  } catch {
    /* nothing better to do: the draft is lost either way */
  }
}

/** True when a previous draft could not be read and was set aside. */
export function hasSetAsideDraft(): boolean {
  try {
    return localStorage.getItem(BACKUP_KEY) !== null;
  } catch {
    return false;
  }
}

export function loadSession(): Session {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaultSession();
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      setAside(raw);
      return defaultSession();
    }
    if (!isRawSession(parsed)) {
      setAside(raw);
      return defaultSession();
    }
    return {
      version: VERSION,
      text: parsed.text as string,
      task: parsed.task as string,
      framing: parsed.framing as FramingMode,
      counter: parsed.counter as number,
      // Absent in drafts saved before the setting existed, and a hand-edited value
      // must not reach the tensor shape: clamp to a sane window or fall back.
      maxLen:
        typeof parsed.maxLen === "number" && Number.isFinite(parsed.maxLen)
          ? Math.min(8192, Math.max(128, Math.round(parsed.maxLen)))
          : null,
      questions: (parsed.questions as unknown[]).map(toQuestion),
    };
  } catch {
    return defaultSession();
  }
}

/** Returns false when the draft was not stored. The page promises in its own tagline
 *  that it keeps the draft, so a write that quietly fails -- private mode, or a pasted
 *  text past the origin quota -- has to be visible. Losing an hour of work and finding
 *  the seed example on reload is not something to discover by reloading. */
export function saveSession(s: Session): boolean {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
    return true;
  } catch {
    return false;
  }
}

export function clearSession(): void {
  try {
    localStorage.removeItem(KEY);
    localStorage.removeItem(BACKUP_KEY);
  } catch {
    /* storage blocked */
  }
}
