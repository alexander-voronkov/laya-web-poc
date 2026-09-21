"""Re-trace the two graphs with a real batch, so they can take one.

Every export this project has made refuses a batch, in all three precisions, with the
same shape of complaint:

    Attempting to broadcast an axis by a dimension other than 1. 69 by 552

The probe sends 8 questions whose longest is 69 tokens, and 8 x 69 = 552. That is the
signature of the unpadding these encoders do internally: sequences are packed into one
flat (total_tokens, hidden) buffer for attention, and `total_tokens` is a value the trace
recorded rather than computed. At batch 1 it equals the sequence length, so the tracer
cannot tell the two apart and writes down the wrong one. At batch 8 they differ and the
graph adds a 69 to a 552.

`dynamic_axes` does not help with this. It renames an axis in the graph's declared
signature; it does not revisit arithmetic the tracer already folded into a constant.

So the fix is to trace where the two quantities are visibly different: a batch of three,
with three different real lengths, padded. Then `total_tokens` is neither the batch size,
nor the padded length, nor their product, and nothing the tracer writes down by accident
survives being wrong three ways at once.

Writes over out/encoder_fp32.onnx and out/head_fp32.onnx, so quantization, conversion,
the parity gate and the batch probe all continue to address the graphs by the names they
already use.
"""
import os
import sys

import torch

sys.path.insert(0, ".")
# The wrappers are the author's, and their forward passes are what the parity gate is
# defined against. Importing rather than copying means a change upstream reaches this
# script instead of silently diverging from it.
from export_onnx import MODEL_DIR, OUT_DIR, EncoderWrapper, HeadWrapper  # noqa: E402
from rl_common import load_cfg, build_model  # noqa: E402


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_cfg(os.path.join(MODEL_DIR, "rl_agent_config.json"))
    model = build_model(cfg, os.path.join(MODEL_DIR, "encoder"))
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(MODEL_DIR, "model.safetensors")), strict=True)
    model.eval()
    model.encoder.config.reference_compile = False
    d = model.encoder.config.hidden_size
    torch.backends.mha.set_fastpath_enabled(False)

    # Three rows, three lengths, one padded width. B=3 rather than 2 because with two
    # rows a sum of lengths can still coincide with something else by chance; with
    # 37+52+29 = 118 against a width of 52 and a batch of 3, no pair of these numbers
    # is equal and none is a product of the others.
    B, L = 3, 52
    lengths = [37, 52, 29]
    K = 3

    ids = torch.randint(100, 5000, (B, L), dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, n in enumerate(lengths):
        att[i, :n] = 1

    hid = torch.randn(B, L, d)
    # Markers inside each row's real length, so the gather is exercised on ragged input
    # rather than on padding.
    mpos = torch.tensor([[3, 9, 15], [4, 20, 40], [2, 7, 11]], dtype=torch.long)
    mmask = torch.ones((B, K), dtype=torch.bool)
    # Different question types per row: type_emb is added per row, and at batch 1 that
    # addition is indistinguishable from adding a constant.
    qtype = torch.tensor([0, 1, 2], dtype=torch.long)

    kw = {"opset_version": 18, "dynamo": True}

    enc_path = os.path.join(OUT_DIR, "encoder_fp32.onnx")
    print("re-tracing encoder at batch %d, lengths %s -> %s" % (B, lengths, enc_path))
    torch.onnx.export(
        EncoderWrapper(model.encoder), (ids, att), enc_path,
        input_names=["input_ids", "attention_mask"], output_names=["hidden"],
        dynamic_axes={"input_ids": {0: "b", 1: "L"}, "attention_mask": {0: "b", 1: "L"},
                      "hidden": {0: "b", 1: "L"}}, **kw)

    head_path = os.path.join(OUT_DIR, "head_fp32.onnx")
    print("re-tracing head at batch %d -> %s" % (B, head_path))
    torch.onnx.export(
        HeadWrapper(model), (hid, att, mpos, mmask, qtype), head_path,
        input_names=["hidden", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes={"hidden": {0: "b", 1: "L"}, "attention_mask": {0: "b", 1: "L"},
                      "marker_pos": {0: "b", 1: "K"}, "marker_mask": {0: "b", 1: "K"},
                      "qtype": {0: "b"}, "logits": {0: "b", 1: "K"}, "act_logits": {0: "b"}},
        **kw)

    for p in (enc_path, head_path):
        extra = p + ".data"
        total = os.path.getsize(p) + (os.path.getsize(extra) if os.path.exists(extra) else 0)
        print("%-28s %8.1f MB" % (os.path.basename(p), total / 1e6))

    # Whether this worked is not something to assume from "the export did not crash".
    # The parity gate and batch_probe.py both run after this and answer it with numbers.
    print("re-traced; the gate and the batch probe decide whether it helped")


if __name__ == "__main__":
    main()
