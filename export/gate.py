"""The acceptance gate, with its thresholds stated rather than assumed.

nvkudva's verify_onnx.py applies a fixed gate: argmax >= 99%, max |dp| <= 0.02, mean KL
<= 1e-3. Those numbers were fitted on ModernBERT-large, and one of them does not carry to
a smaller encoder. Four measurements on the 322M multilingual fine-tune, every one at
100% argmax agreement:

    block 32    max |dp| 0.0304
    block 64    max |dp| 0.0276   (identical with and without a batched trace)
    block 128   max |dp| 0.0244

Monotonic, and in the opposite direction to the textbook expectation, which is how we
know it is not a granularity problem to be tuned away. It is what int8 costs a model this
size: the same recipe costs the 421M English checkpoint 0.0091, because a larger model
has more redundancy to spend.

So this script exists to make one decision visible instead of hidden. The default is
nvkudva's gate unchanged. A run that needs a different threshold has to say so on the
command line, the value is printed beside the measurement, and it belongs in the model's
note in the app so a reader of the page sees the same number. What it must never become
is a threshold quietly widened until the build in hand passes -- which is why the
argmax and KL limits are not exposed at all. Those are the ones that catch a broken
export, and a broken export is what a gate is for: the dynamic-int8 build that shipped
before any of this collapsed to 93.8% argmax and 0.169, and it is argmax that says so.
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
from verify_onnx import GOLD, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--max-dp", type=float, default=0.02,
                    help="worst single probability shift allowed, in absolute terms")
    a = ap.parse_args()

    rows = run(a.encoder, a.head, json.load(open(GOLD)))
    n = len(rows)
    agree = sum(r["argmax_ok"] for r in rows) / n
    max_dp = max(r["d_p"] for r in rows)
    mean_kl = sum(r["kl"] for r in rows) / n

    worst = sorted(rows, key=lambda r: -r["d_p"])[:5]
    print("%-34s %3s %10s %9s %9s %6s" % ("worst cases", "k", "d_logit", "d_p", "kl", "argmx"))
    for r in worst:
        print("%-34s %3d %10.2e %9.2e %9.2e %6s" %
              (r["case"], r["k"], r["d_logit"], r["d_p"], r["kl"], r["argmax_ok"]))
    print("\n%d questions | argmax %.1f%% | max |dp| %.4f | mean KL %.2e" % (n, agree * 100, max_dp, mean_kl))

    # argmax and KL are fixed: they are what catches an export that is wrong rather than
    # merely coarse, and nothing about model size moves them.
    checks = [
        ("argmax >= 99%", agree >= 0.99, "%.1f%%" % (agree * 100)),
        ("max |dp| <= %.4f" % a.max_dp, max_dp <= a.max_dp, "%.4f" % max_dp),
        ("mean KL <= 1e-3", mean_kl <= 1e-3, "%.2e" % mean_kl),
    ]
    print()
    for name, ok, val in checks:
        print("  %-22s %-8s %s" % (name, "PASS" if ok else "FAIL", val))
    if a.max_dp != 0.02:
        print("\n  NOTE: max |dp| was raised from the default 0.02 to %.4f for this run." % a.max_dp)
        print("        Put %.4f in the model's note in src/models.ts, or the page will be" % max_dp)
        print("        claiming a fidelity nobody measured.")
    raise SystemExit(0 if all(c[1] for c in checks) else 1)


if __name__ == "__main__":
    main()
