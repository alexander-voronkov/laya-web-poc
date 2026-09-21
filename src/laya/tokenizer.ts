// Ported from nvkudva/laya-web (app/src/laya/tokenizer.ts).
//
// laya-web-poc: the special tokens are read from tokenizer_config.json instead of being
// hardcoded as [CLS]/[SEP]/[MASK]/[PAD]. The English checkpoint is ModernBERT and does
// use those; the multilingual one is mmBERT, whose vocabulary has no [MASK] at all --
// it is <bos>/<eos>/<mask>/<pad>. Hardcoding them does not fail gracefully there: the
// literal string "[MASK]" tokenizes into three ordinary word pieces, so the sequence
// builder would cheerfully mark option positions with nonsense and the model would
// answer about something else.
import { PreTrainedTokenizer } from "@huggingface/transformers";
import type { Tok } from "./sequence";

/** A tokenizer_config.json special token: either a plain string or the object form
 *  transformers writes when it carries stripping flags. */
type SpecialToken = string | { content?: string } | undefined;

function tokenText(v: SpecialToken, field: string): string {
  const s = typeof v === "string" ? v : v?.content;
  if (!s) throw new Error(`tokenizer_config.json has no ${field}; cannot build sequences without it`);
  return s;
}

/** transformers.js tokenizer built straight from tokenizer.json -- no model needed. */
export async function loadTokenizer(base: string, tokenizerPath = "", signal?: AbortSignal): Promise<Tok> {
  const dir = tokenizerPath ? `${base}/${tokenizerPath}` : base;
  const [tj, tc] = await Promise.all([
    fetch(`${dir}/tokenizer.json`, { signal }).then((r) => {
      if (!r.ok) throw new Error(`${dir}/tokenizer.json: ${r.status}`);
      return r.json();
    }),
    fetch(`${dir}/tokenizer_config.json`, { signal }).then((r) => {
      if (!r.ok) throw new Error(`${dir}/tokenizer_config.json: ${r.status}`);
      return r.json();
    }),
  ]);
  const t = new PreTrainedTokenizer(tj, tc);
  // encode the literal special token rather than reaching into the model internals
  const id = (s: string) => {
    const ids = t.encode(s, { add_special_tokens: false }) as number[];
    if (ids.length !== 1) throw new Error(`${s} did not tokenize to a single id: ${ids}`);
    return ids[0];
  };
  const maskToken = tokenText(tc.mask_token, "mask_token");
  return {
    encode: (text, opts) => t.encode(text, opts) as number[],
    maskToken,
    maskTokenId: id(maskToken),
    clsTokenId: id(tokenText(tc.cls_token ?? tc.bos_token, "cls_token")),
    sepTokenId: id(tokenText(tc.sep_token ?? tc.eos_token, "sep_token")),
    padTokenId: id(tokenText(tc.pad_token, "pad_token")),
  };
}
