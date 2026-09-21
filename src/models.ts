/** The checkpoints this page can run, and what is true about each.
 *
 *  Both are Laya. They differ in backbone, language coverage, context, file layout and
 *  — the part that is easy to miss — in how they were quantized, which decides whether
 *  several questions may share a forward pass. See `batchSafe`.
 */
export interface ModelSpec {
  id: ModelId;
  label: string;
  /** Where the runtime files live. */
  base: string;
  /** Subdirectory holding tokenizer.json, relative to `base`; "" when alongside. */
  tokenizerPath: string;
  /** `split`: separate encoder and head graphs (the English q8 build).
   *  `single`: one graph carrying encoder, head, scorer and act head. */
  layout: "split" | "single";
  /** Graph files to fetch, in load order. A `.data` sibling is fetched for each entry
   *  that has one; see `externalData`. */
  graphs: { name: string; externalData: boolean }[];
  /** Cache Storage bucket. Changing the weights means changing this, or returning
   *  visitors keep the old ones — the browser caches on URL, and these URLs are stable. */
  cache: string;
  /** Approximate total download, for the loading overlay before Content-Length lands. */
  nominalBytes: number;
  /** What the checkpoint can read. */
  languages: "english" | "multilingual";
  /** True when the quantization leaves activations in a format whose scale does not
   *  depend on the rest of the batch.
   *
   *  It is false for every build here, for two different reasons, and both were
   *  measured rather than assumed:
   *
   *  - the English q8 export refuses batch > 1 outright: "Attempting to broadcast an
   *    axis by a dimension other than 1. 153 by 459" — 459 being 3 x 153.
   *  - the multilingual INT8 export accepts a batch and silently answers differently.
   *    Dynamic quantization derives the activation scale from the whole tensor, so one
   *    question's numbers depend on the others sharing the pass. On real text, eight
   *    questions batched moved probabilities by up to 21 percentage points against the
   *    same questions run alone, flipping one decision from 67.6% to 46.7%.
   *
   *  A weight-only quantized build (activations left in fp32, as nvkudva did for the
   *  English encoder) would be batch-exact. Until one exists, questions run one at a
   *  time and this flag stays false. */
  batchSafe: boolean;
  /** One line for the model picker. */
  note: string;
}

export type ModelId = "english" | "multilingual";

export const MODELS: Record<ModelId, ModelSpec> = {
  english: {
    id: "english",
    label: "English (ModernBERT-large, q8)",
    base: "https://huggingface.co/nvkudva/laya-web-q8/resolve/main/v1",
    tokenizerPath: "",
    layout: "split",
    graphs: [
      { name: "encoder_q8", externalData: true },
      { name: "head_q8", externalData: true },
    ],
    cache: "laya-weights-v1",
    nominalBytes: 524_100_000,
    languages: "english",
    batchSafe: false,
    note:
      "Sharper on English than the multilingual build (MASSIVE 0.783 vs 0.657, XNLI 0.860 vs 0.843) " +
      "and its temperatures were actually fitted. Fails on non-Latin script while staying confident. " +
      "512-token context, 524 MB.",
  },
  multilingual: {
    id: "multilingual",
    label: "Multilingual (mmBERT-base, int8)",
    base: "https://huggingface.co/soyelmismo/laya-multilingual-onnx/resolve/main",
    tokenizerPath: "tokenizer",
    layout: "single",
    graphs: [{ name: "model", externalData: false }],
    cache: "laya-weights-ml-int8-v1",
    // 326 MB graph + 34 MB tokenizer: mmBERT's vocabulary is 256k, so the tokenizer
    // alone is ten times the English one and worth counting in the progress bar.
    nominalBytes: 360_100_000,
    languages: "multilingual",
    batchSafe: false,
    note:
      "100+ languages, and a text in one language with questions in another works. " +
      "1024-token context. Smaller than the English build at 326 MB, and quantized for CPUs " +
      "rather than for WebGPU. Ships with no fitted temperatures at all, so read its " +
      "probabilities as an ordering.",
  },
};

export const DEFAULT_MODEL: ModelId = "english";

export function modelFor(id: string | null | undefined): ModelSpec {
  return MODELS[(id as ModelId) in MODELS ? (id as ModelId) : DEFAULT_MODEL];
}

/** Files a complete cache holds for this model, by basename. */
export function weightFiles(m: ModelSpec): string[] {
  return m.graphs.flatMap((g) => (g.externalData ? [`${g.name}.onnx`, `${g.name}.onnx.data`] : [`${g.name}.onnx`]));
}
