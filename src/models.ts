/** The checkpoints this page can run, and what is true about each.
 *
 *  All three are Laya on the same multilingual backbone, mmBERT-base. The English
 *  checkpoints were removed: the demo needs to read the languages its questions are
 *  written in, and an English-only model that answers confidently in Russian is worse
 *  than no model. What remains differs in two things that matter more than they look --
 *  whether the checkpoint was fine-tuned for this kind of question at all (0.748 against
 *  0.342, a gap nothing else here comes close to), and how it was exported, which
 *  decides both its fidelity and whether several questions may share a forward pass.
 *  See `batchSafe`.
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
  /** `split`: separate encoder and head graphs, as our own exports produce.
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
  /** True when several questions may share one forward pass and get the same answers.
   *
   *  This is a property of the export, not of the checkpoint or its precision, and the
   *  three ways it can go wrong were all measured here rather than reasoned about:
   *
   *  - **The trace baked in a batch of one.** Every export from the standard pipeline
   *    refuses batch > 1: "Attempting to broadcast an axis by a dimension other than 1.
   *    69 by 552", where 552 is 8 x 69. These encoders pack sequences into one flat
   *    token buffer for attention, and at batch 1 the total token count is
   *    indistinguishable from the sequence length, so the tracer records the wrong one.
   *    `dynamic_axes` does not help: it renames an axis in the signature, it does not
   *    revisit folded arithmetic. Re-tracing at batch 3 with three different lengths
   *    fixes it — see export/trace_batched.py.
   *  - **Dynamic quantization couples the rows.** The multilingual int8 build accepts a
   *    batch and silently answers differently: its activation scale is derived from the
   *    whole tensor, so one question's numbers depend on the others sharing the pass.
   *    Eight questions batched moved probabilities by up to 21 points, flipping one
   *    decision from 67.6% to 46.7%.
   *  - **It is fine.** fp16 and weight-only int8 leave nothing to couple the rows, and a
   *    batched trace leaves nothing to break, so batching becomes purely a speed change.
   *
   *  Set from export/batch_probe.py, which every export runs against itself. Never from
   *  expectation: the assumption that weight-only quantization implied batch safety was
   *  made here, written down, and turned out to be about the wrong half of the problem. */
  batchSafe: boolean;
  /** True when wasm is not an acceptable fallback for this build. */
  requiresWebGPU?: boolean;
  /** One line for the model picker. */
  note: string;
}

/** Every build in one repository of ours, one folder per variant.
 *
 *  Three of the four came from other people's repositories, and the app fetched them
 *  there directly. That is a dependency on those repositories staying put -- and worse,
 *  on their contents staying the same, since a re-upload under an unchanged URL would
 *  not break anything, it would quietly change what the page answers. The copies are
 *  verified byte-for-byte by .github/workflows/mirror-models.yml; credits live in the
 *  mirror's README and in NOTICE.
 *
 *  The tokenizer sits beside the graphs in every variant, so tokenizerPath is empty
 *  throughout and the per-source layout differences stop at the mirror. */
const MIRROR = "https://huggingface.co/alfred361/laya-web/resolve/main";

export type ModelId = "multilingual-tuned-q8" | "multilingual-tuned" | "multilingual-fp16" | "multilingual";

export const MODELS: Record<ModelId, ModelSpec> = {
  "multilingual-tuned-q8": {
    id: "multilingual-tuned-q8",
    label: "Multilingual, fine-tuned (mmBERT-base, int8) — runs anywhere",
    backend: "wasm",
    base: MIRROR + "/multilingual-tuned-q8/v1",
    tokenizerPath: "",
    layout: "split",
    graphs: [
      { name: "encoder_q8", externalData: true },
      { name: "head_q8", externalData: true },
    ],
    cache: "laya-weights-mlt-q8-v1",
    // 509 MB encoder + 30 MB head + 34 MB tokenizer.
    nominalBytes: 574_000_000,
    languages: "multilingual",
    // EQUIVALENT from batch_probe.py: weight-only quantization leaves activations in
    // fp32, so nothing couples the rows, and this export was traced at batch 3.
    batchSafe: true,
    note:
      "The same fine-tune as the full-precision build, quantized to run without a GPU — " +
      "so this is the one that works on any machine or phone, and it is the default for " +
      "that reason. Every decision in the reference set is unchanged (100% argmax " +
      "agreement) and the worst single probability moves 2.4 points. That is above the " +
      "2-point limit the pipeline applies to the English checkpoint, and deliberately " +
      "so: weight-only int8 costs a 322M encoder more than it costs a 421M one, and " +
      "three block sizes measured 3.0, 2.8 and 2.4 points with no setting reaching 2. " +
      "For comparison, the untuned int8 build below moves probabilities by up to 17 " +
      "points and flips decisions. 574 MB.",
  },
  "multilingual-tuned": {
    id: "multilingual-tuned",
    label: "Multilingual, fine-tuned (mmBERT-base, fp32)",
    backend: "webgpu",
    base: MIRROR + "/multilingual-tuned-fp32-batched/v1",
    tokenizerPath: "",
    layout: "split",
    graphs: [
      { name: "encoder_fp32", externalData: true },
      { name: "head_fp32", externalData: true },
    ],
    cache: "laya-weights-mlt-fp32-v1",
    // 1230 MB encoder + 60 MB head + 34 MB tokenizer.
    nominalBytes: 1_324_400_000,
    languages: "multilingual",
    // Measured by batch_probe.py on this exact export: EQUIVALENT. Eight questions in
    // one pass answer the same as eight passes. It is the first build here of which
    // that is true, and it is true because of how it was traced rather than because of
    // its precision -- the fp32 build from the unmodified pipeline refuses a batch.
    batchSafe: true,
    requiresWebGPU: true,
    note:
      "The only checkpoint here that is good at this task and multilingual at once. " +
      "Fine-tuned from laya-multilingual on the typed-decisions benchmark: 0.352 before, " +
      "0.748 after, measured on the same 400-case test split through the same code. That " +
      "is above the 0.735 ceiling of the teachers who wrote the answers, above TypeSafe " +
      "Jev's published 0.727, and 1.8 points under the English fine-tune — which it beats " +
      "on calibration, 0.112 expected error against 0.213, because its temperatures were " +
      "fitted for it rather than inherited. Full precision, so nothing is lost to " +
      "quantization at all: 100% argmax agreement with the PyTorch reference and a worst " +
      "probability shift of 0.0001 points. Needs WebGPU, 1.3 GB.",
  },
  "multilingual-fp16": {
    id: "multilingual-fp16",
    label: "Multilingual, general (mmBERT-base, fp16)",
    backend: "webgpu",
    base: MIRROR + "/multilingual-fp16/v1",
    tokenizerPath: "",
    layout: "single",
    graphs: [{ name: "model", externalData: false }],
    cache: "laya-weights-ml-fp16-v1",
    // 647 MB graph + 34 MB tokenizer.
    nominalBytes: 681_200_000,
    languages: "multilingual",
    // Measured: eight questions batched against the same eight run alone moved
    // probabilities by at most 0.05 of a point, on both an English and a Russian text.
    // Half precision has no activation scale to derive, so nothing couples the rows,
    // and this export was traced with a batch.
    batchSafe: true,
    requiresWebGPU: true,
    note:
      "The general multilingual checkpoint, untuned, at half precision — so no " +
      "quantization loss, but also none of the fine-tuning. On typed decisions it scores " +
      "0.342, which is below the 0.461 you get by always answering the most common " +
      "option: useful for seeing what the base model does and for questions unlike the " +
      "four workflows the tuned build was trained on, not for numbers you intend to act " +
      "on. Ships with no fitted temperatures. Needs WebGPU, 647 MB — half the tuned " +
      "build and a good deal quicker to download.",
  },
  multilingual: {
    id: "multilingual",
    label: "Multilingual, general (mmBERT-base, int8) — runs without WebGPU",
    backend: "wasm",
    base: MIRROR + "/multilingual-int8/v1",
    tokenizerPath: "",
    layout: "single",
    graphs: [{ name: "model", externalData: false }],
    cache: "laya-weights-ml-int8-v1",
    // 326 MB graph + 34 MB tokenizer: mmBERT's vocabulary is 256k, so the tokenizer
    // alone is ten times the English one and worth counting in the progress bar.
    nominalBytes: 360_100_000,
    languages: "multilingual",
    // Accepts a batch and silently answers differently: dynamic quantization derives the
    // activation scale from the whole tensor, so one question's numbers depend on the
    // others sharing the pass. Eight questions batched moved probabilities by up to 21
    // points and flipped a decision from 67.6% to 46.7%.
    batchSafe: false,
    note:
      "Here for one reason: it is the only multilingual build that runs without WebGPU, " +
      "and it is by far the fastest — under a second where the others take three or four. " +
      "Everything else about it is a compromise. Its dynamic quantization shifts " +
      "probabilities by up to 17 points against full precision, enough to flip roughly " +
      "one decision in sixteen; it has no fitted temperatures; and it is the untuned " +
      "checkpoint, so 0.342 before any of that is counted. Fine for triage and for " +
      "seeing the page work on a device that cannot run the others. 326 MB.",
  },
};

/** The fine-tune, in the form that loads on anything.
 *
 *  Choosing between "the best numbers" and "it works here" stopped being necessary once
 *  the fine-tune existed as weight-only int8: it needs no GPU, batches, costs 574 MB
 *  rather than 1.3 GB, and gives the same decision on every case in the reference set as
 *  the full-precision build. What it costs is 2.4 points on one probability, which the
 *  model's note states rather than hides.
 *
 *  The full-precision build stays one click away for anyone who wants the exact numbers,
 *  and a browser without WebGPU is refused by it with a message naming this one. */
export const DEFAULT_MODEL: ModelId = "multilingual-tuned-q8";

export function modelFor(id: string | null | undefined): ModelSpec {
  return MODELS[(id as ModelId) in MODELS ? (id as ModelId) : DEFAULT_MODEL];
}

/** Files a complete cache holds for this model, by basename. */
export function weightFiles(m: ModelSpec): string[] {
  return m.graphs.flatMap((g) => (g.externalData ? [`${g.name}.onnx`, `${g.name}.onnx.data`] : [`${g.name}.onnx`]));
}
