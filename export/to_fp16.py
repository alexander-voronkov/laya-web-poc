"""Convert the exported fp32 graphs to half precision, for WebGPU.

Not the same thing as the fp16 already in the q8 build. `embed_fp16.py` and
`head_fp16_storage.py` change how weights are *stored* and cast every one of them back
to fp32 before it is used, so the arithmetic stays fp32 and the graph runs on wasm.
This converts the arithmetic too, which is what makes a WebGPU build worth having --
and what makes it useless on a CPU, where fp16 is emulated in software at seconds per
sequence.

Why it should cost nothing in accuracy: the checkpoint is bf16. bf16 carries 8 mantissa
bits, fp16 carries 10, and these values sit well inside fp16's exponent range, so the
weights survive the conversion exactly. The int8 path cannot say that -- it is a lossy
approximation that happens to be a good one. The verification step downstream is what
turns "should" into a number either way.

IO stays fp32 (`keep_io_types`), so nothing about the app's tensors changes and the two
graphs still compose: the encoder's fp32 `hidden` is what the head expects.
"""
import argparse
import os
import sys

import onnx
from onnxconverter_common import float16


def save(model, out):
    """Save with external data, then repair the location fields.

    onnx.save writes the external-data location as whatever path it was handed, which
    for a nested output directory is a relative path the loader then resolves against
    its own working directory rather than the model's. Rewriting them to the bare
    filename is what makes the pair movable -- and the app downloads these two files
    into a browser cache where there is no directory at all.
    """
    for f in (out, out + ".data"):
        if os.path.exists(f):
            os.remove(f)
    onnx.save(model, out, save_as_external_data=True, location=os.path.basename(out) + ".data",
              all_tensors_to_one_file=True, size_threshold=1024)
    m = onnx.load(out, load_external_data=False)
    for t in m.graph.initializer:
        for kv in t.external_data:
            if kv.key == "location":
                kv.value = os.path.basename(out) + ".data"
    onnx.save(m, out)
    return sum(os.path.getsize(f) for f in (out, out + ".data") if os.path.exists(f))


def convert(src, out, block):
    m = onnx.load(src)
    before = os.path.getsize(src) + (os.path.getsize(src + ".data") if os.path.exists(src + ".data") else 0)
    # op_block_list keeps the named ops in fp32. The defaults this library ships leave
    # the numerically delicate ones alone; the additions here are the ones this model
    # reaches that would lose more than precision:
    #
    #   Softmax/LogSoftmax  the head's output distribution -- the whole product is a
    #                       calibrated probability, so it is the last place to economise
    #   ReduceMean/Pow/Sqrt/Div  layer norm's variance, where fp16 underflows on small
    #                       activations and the normalisation then divides by ~0
    #   Range/Cast          index arithmetic, which is not a quantity at all
    fp16 = float16.convert_float_to_float16(
        m, keep_io_types=True, disable_shape_infer=False, op_block_list=block)
    size = save(fp16, out)
    print("%-24s %8.1f MB  ->  %-24s %8.1f MB" %
          (os.path.basename(src), before / 1e6, os.path.basename(out), size / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="out/encoder_fp32.onnx")
    ap.add_argument("--head", default="out/head_fp32.onnx")
    ap.add_argument("--encoder-out", default="out/encoder_fp16.onnx")
    ap.add_argument("--head-out", default="out/head_fp16.onnx")
    a = ap.parse_args()

    # getattr rather than the attribute: the library's default list has been renamed
    # before, and an AttributeError here would surface forty minutes into a run that
    # has already downloaded a checkpoint and traced a 421M-parameter model.
    default = list(getattr(float16, "DEFAULT_OP_BLOCK_LIST", []))
    if not default:
        print("::warning::onnxconverter_common has no DEFAULT_OP_BLOCK_LIST; "
              "using only the ops named here")
    block = default + [
        "Softmax", "LogSoftmax", "ReduceMean", "Pow", "Sqrt", "Div", "Range", "Cast",
    ]
    print("kept in fp32:", ", ".join(sorted(set(block))))
    for src, out in ((a.encoder, a.encoder_out), (a.head, a.head_out)):
        if not os.path.exists(src):
            print("::error::missing %s -- export_onnx.py did not produce it" % src, file=sys.stderr)
            raise SystemExit(1)
        convert(src, out, block)
