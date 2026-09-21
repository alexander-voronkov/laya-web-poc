"""Does this export answer the same when several questions share a forward pass?

Batching is the difference between Laya's published 39.5 ms for one question and
158.6 ms for ten. Whether an export can do it is not something to infer from the
export script: the English q8 build declares its batch axis dynamic and then refuses
batch > 1 outright, and the multilingual int8 build accepts a batch and quietly answers
differently. Both were found by running them, after one of them had shipped.

So this runs the graphs. Every row alone, then all rows together, and compares the
probabilities. Three outcomes, and only one of them means batching is available:

    REFUSED    the graph will not take batch > 1 at all
    DRIFTS     it takes one and the answers move
    EQUIVALENT it takes one and the answers hold

Usage:  python batch_probe.py --encoder out/encoder_q8.onnx --head out/head_q8.onnx
        python batch_probe.py --model out/model.onnx          # single-graph exports
"""
import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def collate(rows, pad_id):
    """Pad exactly as the reference rl_common.collate_items does: pad ids with the pad
    token and zero the attention mask; pad marker_pos with 0 and marker_mask with False,
    so position 0 is written but never read."""
    n = len(rows)
    L = max(len(r["ids"]) for r in rows)
    kmax = max(len(r["markers"]) for r in rows)
    ids = np.full((n, L), pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    qtype = np.zeros((n,), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, : len(r["ids"])] = r["ids"]
        att[i, : len(r["ids"])] = 1
        mpos[i, : len(r["markers"])] = r["markers"]
        mmask[i, : len(r["markers"])] = True
        qtype[i] = r["qtype"]
    return ids, att, mpos, mmask, qtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder")
    ap.add_argument("--head")
    ap.add_argument("--model", help="single-graph export instead of encoder+head")
    ap.add_argument("--golden", default="golden/fixtures.json")
    ap.add_argument("--pad-id", type=int, default=50283)
    # 0.02 is the same bar verify_onnx.py holds quantization to. A batching scheme that
    # drifts more than the quantization does is not a scheme, it is a second model.
    ap.add_argument("--tolerance", type=float, default=0.02)
    a = ap.parse_args()

    split = bool(a.encoder and a.head)
    if split:
        enc = ort.InferenceSession(a.encoder, providers=["CPUExecutionProvider"])
        head = ort.InferenceSession(a.head, providers=["CPUExecutionProvider"])
    else:
        enc = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
        head = None

    # Real sequences rather than synthetic ids: a random draw from the vocabulary
    # produces activations nothing was trained on, and on a dynamically quantized graph
    # that exaggerates exactly the effect being measured.
    with open(a.golden) as f:
        cases = json.load(f)
    rows = []
    for c in cases:
        exp = c.get("expect") or {}
        for qid, e in exp.items():
            if "input_ids" not in e:
                continue
            qt = c["questions"][qid]["type"]
            rows.append({"ids": e["input_ids"], "markers": e["marker_pos"],
                         "qtype": QTYPES[qt], "k": len(e["marker_pos"])})
        if len(rows) >= 8:
            break
    rows = rows[:8]
    if len(rows) < 2:
        print("need at least two fixtures with input_ids to compare", file=sys.stderr)
        return 2
    print(f"{len(rows)} questions, lengths {[len(r['ids']) for r in rows]}")

    def run(batch):
        ids, att, mpos, mmask, qtype = collate(batch, a.pad_id)
        if split:
            hidden = enc.run(["hidden"], {"input_ids": ids, "attention_mask": att})[0]
            out = head.run(["logits"], {"hidden": hidden, "attention_mask": att,
                                        "marker_pos": mpos, "marker_mask": mmask,
                                        "qtype": qtype})[0]
        else:
            out = enc.run(["logits"], {"input_ids": ids, "attention_mask": att,
                                       "marker_pos": mpos, "marker_mask": mmask,
                                       "qtype": qtype})[0]
        return [softmax(out[i, : r["k"]]) for i, r in enumerate(batch)]

    alone = [run([r])[0] for r in rows]
    try:
        together = run(rows)
    except Exception as e:  # noqa: BLE001 - the message is the finding
        print(f"\nREFUSED: {str(e).splitlines()[0]}")
        print("This export cannot take a batch. Questions must run one at a time.")
        return 1

    worst = 0.0
    for i, r in enumerate(rows):
        d = float(np.abs(np.asarray(together[i]) - np.asarray(alone[i])).max())
        worst = max(worst, d)
        flag = "" if d <= a.tolerance else "   <-- moved"
        print(f"  q{i}  k={r['k']}  worst |dp| {d:.4f}{flag}")

    print(f"\nworst probability shift from batching: {worst:.4f}  (tolerance {a.tolerance})")
    if worst <= a.tolerance:
        print("EQUIVALENT: batching is a speed change on this export.")
        return 0
    print("DRIFTS: batching changes the answers. Questions must run one at a time.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
