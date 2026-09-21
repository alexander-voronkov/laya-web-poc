"""Bring the mirror's README up to what the mirror actually holds.

It was generated when the repository held four builds copied from other people, and it
still describes those four. Two of ours have been added since, and one of them is what
the page now serves by default -- so the document that carries the attribution and tells
a reader what each folder is has been wrong about both.

Regenerating it through the mirror workflow would re-download and re-verify two gigabytes
to rewrite one file, so this writes the README alone. The per-variant descriptions are
the measurements, not adjectives.
"""
import os

from huggingface_hub import HfApi

REPO = os.environ.get("DEST", "alfred361/laya-web")
TOKEN = os.environ["HF_TOKEN"]

VARIANTS = [
    ("multilingual-tuned-q8",
     "**Fine-tuned, and the one the demo serves by default.** `laya-multilingual` fine-tuned on "
     "[typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), then exported "
     "weight-only int8. Runs on wasm, so it needs no GPU. Accepts a batch and answers the same "
     "(`EQUIVALENT`). 100% argmax agreement with the PyTorch reference; worst single probability "
     "shift 0.024. 574 MB.",
     "Ours: [alfred361/laya-multilingual-typed-decisions]"
     "(https://huggingface.co/alfred361/laya-multilingual-typed-decisions), fine-tuned from "
     "[convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)."),
    ("multilingual-tuned-fp32-batched",
     "The same fine-tune at full precision, re-traced so it accepts a batch. Nothing is lost to "
     "quantization: 100% argmax and a worst shift of 0.0000011. Needs WebGPU. 1.3 GB. Use it when "
     "the exact numbers matter; on every case measured it makes the same decisions as the int8 "
     "build above.",
     "Ours, same checkpoint."),
    ("multilingual-tuned-fp32",
     "The first full-precision export of the fine-tune, before the batched re-trace. Identical "
     "fidelity, but refuses a batch greater than 1. Kept so the difference the re-trace made is "
     "still inspectable; prefer `multilingual-tuned-fp32-batched`.",
     "Ours, same checkpoint."),
    ("multilingual-fp16",
     "The general multilingual checkpoint, untuned, at half precision. No quantization loss, and it "
     "batches. Scores 0.342 on typed decisions, below the 0.461 of always answering the most common "
     "option -- it is here to show what the base model does, not to be relied on. No fitted "
     "temperatures. Needs WebGPU. 647 MB.",
     "fp16 ONNX export by [mizchi](https://huggingface.co/mizchi/laya-multilingual-onnx)."),
    ("multilingual-int8",
     "The same untuned checkpoint in dynamic int8. The fastest build here and the least faithful: "
     "activation scales are derived per tensor at run time, which measured 93.8% argmax agreement "
     "with full precision and a worst shift of 0.169. It accepts a batch and answers differently "
     "when it does, by up to 21 points. 326 MB.",
     "CPU-optimized export by [soyelmismo](https://huggingface.co/soyelmismo/laya-multilingual-onnx)."),
    ("english-q8",
     "Laya English base (ModernBERT-large, 421M), weight-only int8. 100% argmax agreement with "
     "fp32, worst shift 0.016. 512-token context. Refuses a batch. No longer served by the demo, "
     "which is multilingual only.",
     "Quantization and browser runtime by [nvkudva](https://huggingface.co/nvkudva/laya-web-q8)."),
    ("typed-decisions-q8",
     "Laya fine-tuned on typed decisions (ModernBERT-large, 421M, English only). Scores 0.766, the "
     "highest here, but reads English alone. Weight-only int8 exported by us: 100% argmax, worst "
     "shift 0.009. Refuses a batch. Note that its `temperature_by_options` is inherited from the "
     "base checkpoint and overrides the temperatures fitted for it -- upstream says so too.",
     "Exported from [convaiinnovations/laya-typed-decisions]"
     "(https://huggingface.co/convaiinnovations/laya-typed-decisions) with the pipeline from "
     "[nvkudva/laya-web](https://github.com/nvkudva/laya-web)."),
]

lines = [
    # Frontmatter, so the Hub shows the licence on the model page rather than only in
    # the prose. These builds are Apache-2.0 because the checkpoints they came from are.
    "---", "license: apache-2.0", "language: [multilingual]",
    "tags: [laya, system-one, onnx, onnxruntime-web, typed-decisions]", "---", "",
    "# Laya, packaged for the browser\n",
    "Every build [laya.voronkov.club](https://laya.voronkov.club) has served, in one place, so a "
    "demo does not depend on several separate repositories staying where they are. Each folder is "
    "self-contained: the graphs, the tokenizer that produced their token ids, and the config "
    "carrying the sequence limits and temperatures.\n",
    "All Apache-2.0, as upstream. Credits are per variant below; the base models are not ours.\n",
    "| folder | what it is |", "| --- | --- |",
]
for folder, about, _ in VARIANTS:
    lines.append(f"| `{folder}/v1/` | {about} |")

lines += [
    "\n## Where the numbers come from\n",
    "`argmax agreement` and `worst shift` are measured against the PyTorch model the export came "
    "from, over a fixed set of 26 questions, by `export/gate.py` in "
    "[laya-web-poc](https://github.com/alexander-voronkov/laya-web-poc). Whether a build accepts a "
    "batch is measured too, by `export/batch_probe.py`, because it is a property of the export "
    "rather than of the checkpoint: three of these refuse one outright and a fourth accepts one and "
    "silently answers differently.\n",
    "The benchmark scores are accuracy on the test split of "
    "[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), 400 "
    "cases and 2,000 decisions. For scale: always answering the most common option scores 0.461, "
    "the teachers who wrote the answers agree with themselves 0.735 of the time, and the fine-tune "
    "here scores 0.748.\n",
    "## Credits\n",
]
for folder, _, credit in VARIANTS:
    lines.append(f"- **{folder}** — {credit}")

lines.append(
    "\nBase models: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) and its "
    "family, by Nandakishor M / Convai Innovations. Benchmark by LocalLLaMA.\n"
    "\n## The `v1` segment\n\nBrowsers cache by URL. Re-quantizing publishes to `v2` rather than "
    "overwriting `v1`, or a returning visitor keeps weights the app no longer believes it is "
    "running.\n")

text = "\n".join(lines)
api = HfApi(token=TOKEN)

# Every folder described must exist, and every folder that exists must be described. A
# README that quietly omits a build is the attribution failing silently.
listed = {f.rfilename.split("/")[0] for f in api.model_info(REPO, files_metadata=False).siblings
          if "/" in f.rfilename}
described = {f for f, _, _ in VARIANTS}
missing = listed - described
extra = described - listed
if missing or extra:
    raise SystemExit(f"README does not match the repo: undescribed {sorted(missing)}, "
                     f"described but absent {sorted(extra)}")
print(f"{len(listed)} folders, all described")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(text)
api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id=REPO,
                repo_type="model")
print(f"https://huggingface.co/{REPO}")
