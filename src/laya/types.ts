// Ported from nvkudva/laya-web (app/src/laya/types.ts).
export type QType = "choice" | "score" | "noul";

/** Jev request shape, as rl_agent_api.RLAgent.system_one accepts it. */
export interface QuestionDef {
  type: QType;
  instructions: string | unknown;
  criteria?: Record<string, string | null> | string[] | null;
}
export type Questions = Record<string, QuestionDef>;
export type State = string | Record<string, unknown> | unknown[];

/** Internal form after _to_internal: criteria normalised, instructions stringified. */
export interface InternalQ {
  t: QType;
  ins: string;
  crit: Record<string, string | null> | string[] | null;
}

export interface LayaConfig {
  max_len: number;
  head_max_len: number;
  temperature: [number, number, number];
  temperature_by_options: Record<string, number>;
}

export type Answer =
  | { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number; rl_agent: { act_probability: number } }
  | { type: "score"; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number; rl_agent: { act_probability: number } }
  | { type: "noul"; noul: number; rl_agent: { act_probability: number } };

export interface LayaResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
}

export const QTYPES: Record<QType, number> = { choice: 0, score: 1, noul: 2 };
export const QTYPE_NAMES: QType[] = ["choice", "score", "noul"];

// ---------------------------------------------------------------------------
// laya-web-poc: telemetry. None of this feeds back into inference -- it exists so
// the page can show where the 512-token budget, the time and the memory went.
// ---------------------------------------------------------------------------

/** How one question spent the sequence budget, and what did not fit. */
export interface SequenceStats {
  /** Instruction tokens that survived the head budget clamp. */
  headTokens: number;
  /** Instruction tokens before the clamp. Larger than headTokens means the wording
   *  was cut from the end -- and since the task prompt is prepended, the cut lands
   *  on the question itself. */
  headTokensFull: number;
  /** Option tokens actually sent, including one [MASK] marker each. */
  optionTokens: number;
  optionTokensFull: number;
  /** True when the options did not fit and every one of them was shortened. */
  optionsShrunk: boolean;
  /** State tokens the text would need in full. */
  stateTokens: number;
  /** State tokens that actually reached the model. */
  stateTokensUsed: number;
  /** Length of the sequence handed to the encoder. */
  totalTokens: number;
  /** Tokens dropped past max_len even after the state was truncated. */
  overflowTokens: number;
}

export interface QuestionTelemetry {
  qid: string;
  type: QType;
  options: number;
  /** Bucket the temperature was looked up under, e.g. "choice:3-5". */
  temperatureBucket: string;
  temperature: number;
  stats: SequenceStats;
  /** Tokenizing and assembling the sequence. */
  buildMs: number;
  /** How many questions shared the forward pass these timings describe. Laya answers
   *  a whole batch in one pass, so encoderMs and headMs belong to the batch; splitting
   *  them per question would invent a number nobody measured. */
  batchSize: number;
  /** The 28-layer ModernBERT encoder pass -- this is where the time goes. */
  encoderMs: number;
  /** The decision head. */
  headMs: number;
  totalMs: number;
}
