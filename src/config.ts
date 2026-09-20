/** Where the browser fetches encoder_q8.onnx(.data), head_q8.onnx(.data),
 *  tokenizer.json, tokenizer_config.json and rl_agent_config.json from.
 *  Override at build time with VITE_MODELS_BASE (see README). */
export const MODELS_BASE: string =
  (import.meta.env.VITE_MODELS_BASE as string | undefined) ||
  "https://huggingface.co/nvkudva/laya-web-q8/resolve/main/v1";

/** Nominal size of the four weight files, shown before content-length is known. */
export const NOMINAL_TOTAL_BYTES = 524_100_000;
