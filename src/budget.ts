// What each question will actually cost, computed by running the real sequence
// builder rather than estimating it.
//
// The estimate this replaces was a fixed 96-token headroom, which is wrong in both
// directions: a noul with short options leaves ~470 tokens for the text, while a
// twelve-option choice with a long task prompt leaves ~300. Worse, the head budget
// clamp cuts instructions from the end, so a long task prompt silently deletes the
// question it was prepended to -- and nothing in the output says so. Running the
// builder costs one tokenizer pass per question and tells us exactly.
import { buildSequence, renderOptions, toInternal } from "./laya/sequence";
import type { Core } from "./laya/session";
import type { SequenceStats } from "./laya/types";
import { buildState, toQuestionDef, type FramingMode, type QuestionItem } from "./questions";

export interface QuestionBudget {
  id: string;
  stats: SequenceStats;
  /** The instruction text did not fit the head budget and was cut from the end.
   *  Since the task prompt goes in front, what gets cut is the question itself. */
  instructionsClipped: boolean;
  /** The text did not fit and its tail was dropped. */
  stateTruncated: boolean;
  /** Options did not fit in head_max_len at all -- this question cannot run. */
  optionsDontFit: boolean;
}

export interface Budget {
  perQuestion: QuestionBudget[];
  /** Tokens the text needs in full, independent of any question. */
  stateTokens: number;
  maxLen: number;
  headMaxLen: number;
  /** Fewest state tokens any single question leaves room for. */
  worstStateUsed: number;
  anyTruncated: boolean;
  anyClipped: boolean;
  anyBroken: boolean;
}

export function analyse(
  core: Core,
  task: string,
  text: string,
  questions: QuestionItem[],
  mode: FramingMode,
): Budget {
  const state = buildState(task, text, mode);
  const stateTokens = core.tok.encode(state, { add_special_tokens: false }).length;
  const perQuestion: QuestionBudget[] = [];

  for (const q of questions) {
    const internal = toInternal(toQuestionDef(task, q, mode));
    const k = renderOptions(internal).length;
    const { markers, stats } = buildSequence(core.tok, state, internal, core.cfg.max_len, core.cfg.head_max_len);
    perQuestion.push({
      id: q.id,
      stats,
      instructionsClipped: stats.headTokens < stats.headTokensFull,
      stateTruncated: stats.stateTokensUsed < stats.stateTokens || stats.overflowTokens > 0,
      optionsDontFit: markers.length !== k,
    });
  }

  return {
    perQuestion,
    stateTokens,
    maxLen: core.cfg.max_len,
    headMaxLen: core.cfg.head_max_len,
    // Math.min of an empty list is Infinity, which would render as "∞ tokens left".
    worstStateUsed: perQuestion.length
      ? Math.min(...perQuestion.map((p) => p.stats.stateTokensUsed))
      : stateTokens,
    anyTruncated: perQuestion.some((p) => p.stateTruncated),
    anyClipped: perQuestion.some((p) => p.instructionsClipped),
    anyBroken: perQuestion.some((p) => p.optionsDontFit),
  };
}
