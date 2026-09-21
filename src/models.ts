/** The checkpoints this page can run, and what is true about each.
 *
 *  Both are Laya. They differ in backbone, language coverage, context, file layout and
 *  — the part that is easy to miss — in how they were quantized, which decides whether
 *  several questions may share a forward pass. See `batchSafe`.
 */
export interface ModelSpec {
  id: ModelId;
  label: string;
  /** Which onnxruntime-web entry to load.
   *
   *  `wasm` is the small one (14 MB runtime). `webgpu` pulls the jsep build, which is
   *  roughly twice that, so it is imported only when a model that needs it is chosen
   *  rather than charged to every visitor. */
  backend: "wasm" | "webgpu";
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
  /** True when wasm is not an acceptable fallback for this build. */
  requiresWebGPU?: boolean;
  /** One line for the model picker. */
  note: string;
}

export type ModelId = "english" | "typed-decisions" | "multilingual" | "multilingual-fp16";

export const MODELS: Record<ModelId, ModelSpec> = {
  english: {
    id: "english",
    label: "English (ModernBERT-large, q8)",
    backend: "wasm",
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
  "typed-decisions": {
    id: "typed-decisions",
    label: "English, specialised (typed-decisions, q8)",
    backend: "wasm",
    base: "https://huggingface.co/alfred361/laya-typed-decisions-web-q8/resolve/main/v1",
    tokenizerPath: "",
    layout: "split",
    graphs: [
      { name: "encoder_q8", externalData: true },
      { name: "head_q8", externalData: true },
    ],
    cache: "laya-weights-td-v1",
    nominalBytes: 524_100_000,
    languages: "english",
    // Measured, not assumed, and the assumption was wrong. Weight-only quantization
    // leaves activations in fp32 so nothing couples the rows, and this export used
    // torch dynamo rather than TorchScript -- both reasons to expect batching to work.
    // It does not: batch of 8 fails with "Attempting to broadcast an axis by a
    // dimension other than 1. 70 by 560", where 560 is 8 x 70. The ModernBERT export
    // bakes a batch-1 constant whichever exporter traces it.
    batchSafe: false,
    note:
      "The only checkpoint here that is actually good at this task. On typed decisions it " +
      "scores 0.766 where the general checkpoints score 0.362 and 0.342 and the " +
      "majority-class baseline is 0.461 — they are below the trivial answer, this one is " +
      "well above it. Exported and verified here: 100% argmax agreement with fp32, worst " +
      "probability shift 0.9 points. English only, 1024-token context, temperatures fitted. " +
      "Its 0.766 was measured on the four workflows it was tuned for — invoices, security " +
      "incidents, customer service, agent traces — so on questions shaped like something " +
      "else, measure before trusting it.",
  },
  multilingual: {
    id: "multilingual",
    label: "Multilingual (mmBERT-base, int8)",
    backend: "wasm",
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
      "1024-token context, 326 MB, and by far the fastest here — but the least faithful: " +
      "its dynamic quantization shifts probabilities by up to 17 points against the fp32 " +
      "reference, enough to flip roughly one decision in sixteen. It also ships with no " +
      "fitted temperatures at all. Good for triage and for languages; for a number you " +
      "intend to act on, check it against fp16 or English.",
  },
  "multilingual-fp16": {
    id: "multilingual-fp16",
    label: "Multilingual fp16 — WebGPU only (mmBERT-base)",
    backend: "webgpu",
    base: "https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main",
    tokenizerPath: "tokenizer",
    layout: "single",
    graphs: [{ name: "model", externalData: false }],
    cache: "laya-weights-ml-fp16-v1",
    // 647 MB graph + 34 MB tokenizer.
    nominalBytes: 681_200_000,
    languages: "multilingual",
    // Untested. Unlike the int8 build there is no dynamic activation scale to couple
    // the rows, so batching ought to be exact here -- but "ought to" is what was said
    // about the English export before it refused a batch outright, so this stays false
    // until somebody runs the comparison in scripts/batch-equivalence.mjs.
    batchSafe: false,
    requiresWebGPU: true,
    note:
      "Half precision, so no quantization loss at all -- the closest thing here to the " +
      "unquantised checkpoint, and the answer to the int8 build's 17-point drift. " +
      "unquantised checkpoint. Needs WebGPU: on a CPU without native fp16, ONNX Runtime " +
      "emulates it in software at roughly 3.9 s per sequence, which is why this build is " +
      "refused rather than silently run on wasm. 647 MB, the largest of the three.",
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
