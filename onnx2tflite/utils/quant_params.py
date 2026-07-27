"""
Export the per-output dequantization parameters (scale, zero_point) of a
converted int8 TFLite model, keyed by the ORIGINAL ONNX output tensor name
(e.g. "box", "conf", "land") rather than whatever internal name TF assigned
during Keras/TFLite conversion (typically "StatefulPartitionedCall:N").

Why this needs care: `keras_builder()` builds `keras_model.outputs` as
`[tf_tensor[x.name] for x in model_graph.output]`, i.e. in the exact same
order as the ONNX graph's declared outputs. TFLiteConverter preserves that
order in the flatbuffer's SignatureDef, BUT the plain, unordered
`Interpreter.get_output_details()` list is sorted by internal tensor index,
which is NOT necessarily the same order (verified empirically: a 3-output
model came back index-sorted as [output_1, output_2, output_0]). The
reliable signal is the ":N" suffix TF appends to each output tensor's name
(e.g. "StatefulPartitionedCall:0", "...:1", ...), which reflects the
original position in the tuple the model returns -- exactly the order
`model_graph.output` was built in. This module sorts on that suffix (via
the SignatureDef runner when available, since it also gives a friendlier
name-to-tensor mapping) rather than trusting list order.
"""

import re
import logging

import numpy as np

LOG = logging.getLogger("quant_params:")


def _make_interpreter(tflite_bytes):
    """Prefer tf.lite.Interpreter (already a dependency of this package);
    fall back to lighter-weight standalone runtimes if tensorflow isn't
    importable in the environment this happens to run in."""
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_content=tflite_bytes)
    except Exception:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_content=tflite_bytes)
    except Exception:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_content=tflite_bytes)
    except Exception as e:
        raise ImportError(
            "Could not construct a TFLite Interpreter: tried tensorflow, "
            "ai_edge_litert, and tflite_runtime, none were importable."
        ) from e


def _suffix_index(tensor_name: str):
    """Extract the trailing ':N' position from a tensor name, if present."""
    m = re.search(r":(\d+)$", tensor_name or "")
    return int(m.group(1)) if m else None


def _ordered_output_details(interp):
    """Return output tensor detail dicts ordered to match the model's
    original output-tuple order (== ONNX graph.output order), using the
    ':N' name suffix as the authoritative position, with the SignatureDef
    runner used when available for a cleaner name-to-tensor mapping."""
    sig_list = None
    try:
        sig_list = interp.get_signature_list()
    except Exception as e:
        # Older TF/TFLite runtimes (pre ~2.5) don't expose
        # get_signature_list()/get_signature_runner() at all, and some
        # interpreter builds raise rather than return {} when a model has
        # no embedded SignatureDef. Either way, fall through to the plain
        # get_output_details() path below instead of aborting the whole
        # export.
        LOG.warning(f"interp.get_signature_list() unavailable/failed "
                    f"({type(e).__name__}: {e}); falling back to plain "
                    f"get_output_details() order.")

    if sig_list:
        try:
            sig_key = "serving_default" if "serving_default" in sig_list else next(iter(sig_list))
            runner = interp.get_signature_runner(sig_key)
            details_by_sig_name = runner.get_output_details()
            items = list(details_by_sig_name.items())
            if all(_suffix_index(d.get("name", "")) is not None for _, d in items):
                items.sort(key=lambda kv: _suffix_index(kv[1]["name"]))
            else:
                LOG.warning("Signature output tensor names lack a ':N' suffix; "
                            "falling back to signature dict order (unverified).")
            return [d for _, d in items]
        except Exception as e:
            LOG.warning(f"Signature-based output ordering failed "
                        f"({type(e).__name__}: {e}); falling back to plain "
                        f"get_output_details() order.")

    # No SignatureDef in the flatbuffer (e.g. an older/direct-IR-built
    # model) -- fall back to the plain output details list.
    details = interp.get_output_details()
    if all(_suffix_index(d.get("name", "")) is not None for d in details):
        details = sorted(details, key=lambda d: _suffix_index(d["name"]))
    else:
        LOG.warning("No SignatureDef and output tensor names lack a ':N' "
                    "suffix; falling back to raw get_output_details() order "
                    "-- verify the resulting JSON against the model "
                    "carefully, this order could not be independently "
                    "confirmed.")
    return details


def compute_output_dequant_params(model_proto, tflite_bytes) -> dict:
    """
    Returns an (insertion-ordered) dict:
        { onnx_output_name: {
              "tflite_tensor_name": str,
              "shape": [int, ...],
              "dtype": str,
              "quantized": bool,
              # present only when quantized is True:
              "scale": float,
              "zero_point": int,
              "dequant_formula": "real_value = (quantized_value - zero_point) * scale",
          }, ... }
    keyed by the ORIGINAL ONNX graph output names, in that same order.
    """
    onnx_output_names = [o.name for o in model_proto.graph.output]

    interp = _make_interpreter(tflite_bytes)
    interp.allocate_tensors()
    detail_list = _ordered_output_details(interp)

    if len(detail_list) != len(onnx_output_names):
        raise ValueError(
            f"Output count mismatch: ONNX model declares {len(onnx_output_names)} "
            f"output(s) {onnx_output_names!r}, but the TFLite model has "
            f"{len(detail_list)} output tensor(s). Cannot reliably map names -- "
            f"check that this .tflite was actually produced from this .onnx."
        )

    result = {}
    for onnx_name, det in zip(onnx_output_names, detail_list):
        dtype = det["dtype"]
        is_float = np.issubdtype(dtype, np.floating)
        dtype_name = getattr(dtype, "__name__", str(dtype))

        entry = {
            "tflite_tensor_name": det.get("name"),
            "shape": [int(x) for x in det.get("shape", [])],
            "dtype": dtype_name,
        }

        scales = det.get("quantization_parameters", {}).get("scales", [])
        zero_points = det.get("quantization_parameters", {}).get("zero_points", [])
        scale, zero_point = det.get("quantization", (0.0, 0))

        if is_float:
            entry["quantized"] = False
            entry["note"] = "Output tensor is already float; no dequantization needed."
        else:
            entry["quantized"] = True
            if len(np.atleast_1d(scales)) > 1:
                # Per-channel quantization (unusual for a model output, but
                # handle it rather than silently picking scales[0]).
                entry["scale"] = [float(s) for s in scales]
                entry["zero_point"] = [int(z) for z in zero_points]
                entry["quantized_dimension"] = int(
                    det["quantization_parameters"].get("quantized_dimension", 0))
            else:
                entry["scale"] = float(scale)
                entry["zero_point"] = int(zero_point)
            entry["dequant_formula"] = "real_value = (quantized_value - zero_point) * scale"

        result[onnx_name] = entry

    return result


def export_output_dequant_params(model_proto, tflite_bytes, output_json_path: str) -> dict:
    """Compute dequant params and write them to `output_json_path`. Returns
    the same dict that was written, in case the caller wants it in-memory
    too (e.g. to log a summary)."""
    import json

    params = compute_output_dequant_params(model_proto, tflite_bytes)
    with open(output_json_path, "w") as f:
        json.dump(params, f, indent=2)
    LOG.info(f"Wrote output dequantization parameters to {output_json_path}")
    return params
