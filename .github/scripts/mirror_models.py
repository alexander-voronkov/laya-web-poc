"""Copy the browser builds this app serves into one repository, and prove the copy.

One folder per variant, each self-contained: the graphs the runtime loads, the
tokenizer that built their token ids, and the rl_agent_config.json that carries the
sequence limits and the calibration temperatures. A variant is useless without all
three, and pairing a graph with the wrong tokenizer produces confident answers to a
different question rather than an error.

The `v1` segment inside each variant is not decoration. Browsers cache by URL, so
re-quantizing has to publish to `v2` rather than overwrite `v1`, or returning visitors
keep weights the app no longer believes it is running.
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

DEST = os.environ["DEST"]
DRY = os.environ.get("DRY", "").lower() == "true"
TOKEN = os.environ.get("HF_TOKEN") or None

# dest folder -> where it comes from and what a complete variant needs.
VARIANTS = {
    "english-q8": {
        "repo": "nvkudva/laya-web-q8",
        "prefix": "v1/",
        "files": ["encoder_q8.onnx", "encoder_q8.onnx.data", "head_q8.onnx", "head_q8.onnx.data",
                  "rl_agent_config.json", "tokenizer.json", "tokenizer_config.json"],
        "about": "Laya English base (ModernBERT-large, 421M). Weight-only int8: weights are "
                 "compressed, activations stay fp32. 100% argmax agreement with fp32, worst "
                 "probability shift 0.016. 512-token context, temperatures fitted. Refuses a "
                 "batch greater than 1.",
        "credit": "Quantization and browser runtime by [nvkudva](https://huggingface.co/nvkudva/laya-web-q8).",
    },
    "typed-decisions-q8": {
        "repo": "alfred361/laya-typed-decisions-web-q8",
        "prefix": "v1/",
        "files": ["encoder_q8.onnx", "encoder_q8.onnx.data", "head_q8.onnx", "head_q8.onnx.data",
                  "rl_agent_config.json", "tokenizer.json", "tokenizer_config.json"],
        "about": "Laya fine-tuned on typed decisions (ModernBERT-large, 421M). Scores 0.766 on "
                 "that benchmark where the general checkpoints score 0.362 and 0.342 and the "
                 "majority-class baseline is 0.461. Weight-only int8, exported here: 100% argmax "
                 "agreement with fp32, worst shift 0.0091. English only. Refuses a batch.",
        "credit": "Exported from [convaiinnovations/laya-typed-decisions]"
                  "(https://huggingface.co/convaiinnovations/laya-typed-decisions) with the "
                  "pipeline from [nvkudva/laya-web](https://github.com/nvkudva/laya-web).",
    },
    "multilingual-int8": {
        "repo": "soyelmismo/laya-multilingual-onnx",
        "prefix": "",
        "files": ["model.onnx", "rl_agent_config.json",
                  "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"],
        "about": "Laya multilingual (mmBERT-base, 322M), dynamic int8. The fastest of the four "
                 "and the least faithful: activation scales are derived per tensor at run time, "
                 "which measured 93.8% argmax agreement with this repository's own fp32 export "
                 "and a worst shift of 16.9 points. It accepts a batch and answers differently "
                 "when it does, by up to 21 points. 1024-token context, no fitted temperatures.",
        "credit": "CPU-optimized export by [soyelmismo](https://huggingface.co/soyelmismo/laya-multilingual-onnx).",
    },
    "multilingual-fp16": {
        "repo": "mizchi/laya-multilingual-onnx",
        "prefix": "",
        "files": ["model.onnx", "rl_agent_config.json",
                  "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"],
        "about": "Laya multilingual (mmBERT-base, 322M), half precision. No quantization loss at "
                 "all, and the only one of the four that batches correctly — eight questions "
                 "batched moved probabilities by at most 0.05 of a point. Needs WebGPU: on a CPU "
                 "without native fp16 the runtime emulates it in software at seconds per "
                 "sequence. 1024-token context, no fitted temperatures.",
        "credit": "fp16 ONNX export by [mizchi](https://huggingface.co/mizchi/laya-multilingual-onnx).",
    },
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    api = HfApi(token=TOKEN)
    staged = {}

    for folder, v in VARIANTS.items():
        print(f"\n=== {folder}  <-  {v['repo']}")
        for name in v["files"]:
            src = v["prefix"] + name
            path = hf_hub_download(v["repo"], src, local_dir=f"work/{folder}")
            size = os.path.getsize(path)
            # Flatten tokenizer/ into the variant root: every variant then has the same
            # shape, and the app does not need a per-model path rule.
            dest = f"{folder}/v1/{os.path.basename(name)}"
            staged[dest] = (path, size, digest(path))
            print(f"    {src:<42} {size/1e6:8.1f} MB")

    total = sum(s for _, s, _ in staged.values())
    print(f"\n{len(staged)} files, {total/1e9:.2f} GB total")

    if DRY:
        print("\ndry run: sources all resolve, nothing uploaded")
        return 0
    if not TOKEN:
        print("::error::HF_TOKEN is not set", file=sys.stderr)
        return 1

    readme = ["# Laya, packaged for the browser\n",
              "Every build this project serves, in one place, so a demo does not depend on four "
              "separate repositories staying where they are. Each folder is self-contained: the "
              "graphs, the tokenizer that produced their token ids, and the config carrying the "
              "sequence limits and temperatures.\n",
              "All are Apache-2.0, as upstream. Credits are per variant below; none of the base "
              "models are ours.\n",
              "| folder | what it is |", "| --- | --- |"]
    for folder, v in VARIANTS.items():
        readme.append(f"| `{folder}/v1/` | {v['about']} |")
    readme.append("\n## Credits\n")
    for folder, v in VARIANTS.items():
        readme.append(f"- **{folder}** — {v['credit']}")
    readme.append(
        "\nBase models: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) "
        "and its family, by Nandakishor M / Convai Innovations.\n"
        "\n## The `v1` segment\n\nBrowsers cache by URL. Re-quantizing publishes to `v2` rather "
        "than overwriting `v1`, or a returning visitor keeps weights the app no longer believes "
        "it is running.\n")
    os.makedirs("out", exist_ok=True)
    with open("out/README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

    api.create_repo(DEST, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj="out/README.md", path_in_repo="README.md",
                    repo_id=DEST, repo_type="model")
    for dest, (path, _, _) in sorted(staged.items()):
        print(f"  uploading {dest}")
        api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=DEST, repo_type="model")

    # A mirror nobody checked is a claim, not a copy. Re-download each file from the
    # destination and compare digests: an upload that truncated or a path that collided
    # would otherwise surface later as a model that loads and answers wrongly.
    print("\nverifying the copy")
    bad = []
    for dest, (_, size, want) in sorted(staged.items()):
        got = hf_hub_download(DEST, dest, local_dir="verify")
        have = digest(got)
        ok = have == want
        print(f"  {'ok  ' if ok else 'FAIL'} {dest:<48} {size/1e6:8.1f} MB")
        if not ok:
            bad.append(dest)
        os.remove(got)
    if bad:
        print(f"::error::{len(bad)} file(s) differ after upload: {', '.join(bad)}", file=sys.stderr)
        return 1
    print(f"\nhttps://huggingface.co/{DEST}  — {len(staged)} files verified byte-for-byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
