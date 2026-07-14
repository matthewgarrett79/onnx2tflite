import os
import logging

from .onnx_loader import load_onnx_modelproto
from .output_check import get_elements_error, check_tflite_error
from .keras_backend.builder import keras_builder, tflite_builder
from .utils.relu_concat_fix import push_relu_before_concat

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("converter:")


def _resolve_output_path(onnx_model_path: str, output_path: str = None,
                         fp16: bool = False, int8: bool = False) -> str:
    onnx_dir, model_name = os.path.split(onnx_model_path)
    if output_path is None:
        output_path = onnx_dir
    base = os.path.join(output_path, model_name.split('.')[0])
    if fp16:   base += "_fp16"
    elif int8: base += "_int8"
    return base


def _convert_keras(model_proto, onnx_model_path, output_path,
                   native_groupconv, weight_quant, fp16_model, int8_model,
                   calibration_data, target_formats, float_output=False):
    """ONNX → Keras → TFLite (original two-stage path)."""
    keras_model, input_layout, output_layout = keras_builder(model_proto, native_groupconv)

    tflite_bytes = None
    if 'tflite' in target_formats:
        tflite_bytes = tflite_builder(keras_model, weight_quant, fp16_model,
                                      int8_model, calibration_data, float_output)

    out_base = _resolve_output_path(onnx_model_path, output_path, fp16_model, int8_model)

    keras_path = None
    if 'keras' in target_formats:
        keras_path = out_base + ".h5"
        keras_model.save(keras_path)
        LOG.info(f"keras model saved in {keras_path}")

    tflite_path = None
    if tflite_bytes is not None:
        tflite_path = out_base + ".tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)

    result = {"keras": keras_path, "tflite": tflite_path,
              "keras_error": 0, "tflite_error": 0}
    if int8_model:
        return result

    try:
        errors = get_elements_error(model_proto, keras_path, tflite_path,
                                    input_layout, output_layout)
    except Exception:
        LOG.warning("convert succeeded but model runtime check failed")
        return result

    for key, label in [('keras', 'h5'), ('tflite', 'tflite')]:
        err = errors.get(key)
        if err is None:
            continue
        result[f"{key}_error"] = err
        if err > 1e-2:
            LOG.error(f"{label} max error {err:.4E} — check model carefully!")
        elif err > 1e-4:
            LOG.warning(f"{label} max error {err:.4E}, pass")
        else:
            LOG.info(f"{label} max error {err:.4E}, pass")

    return result


def _convert_direct(model_proto, onnx_model_path, output_path):
    """ONNX → TFLite via direct IR builder (bypasses Keras)."""
    from .tflite_backend import build_tflite_ir

    tflite_bytes = build_tflite_ir(model_proto)
    out_base = _resolve_output_path(onnx_model_path, output_path)
    tflite_path = out_base + ".tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    LOG.info(f"tflite model (direct IR) saved in {tflite_path}")

    result = {"tflite": tflite_path, "tflite_error": None}
    try:
        result["tflite_error"] = check_tflite_error(model_proto, tflite_bytes)
        LOG.info(f"tflite model (direct IR) max error: {result['tflite_error']:.4E}")
    except Exception as e:
        LOG.warning(f"direct IR error check failed: {type(e).__name__}: {e}")
    return result


# ---- Public API ----

def onnx_converter(onnx_model_path: str, output_path: str = None,
                   input_node_names: list = None, output_node_names: list = None,
                   need_simplify: bool = True, target_formats: list = ['tflite'],
                   native_groupconv: bool = False,
                   use_direct_ir: bool = False,
                   weight_quant: bool = False, fp16_model: bool = False,
                   int8_model: bool = False, calibration_data: list = None,
                   float_output: bool = False) -> dict:
    """Convert ONNX model to TFLite (and optionally Keras .h5).

    Two backends available:
      - use_direct_ir=False (default): ONNX → Keras Model → TFLite
      - use_direct_ir=True:            ONNX → TFLite FlatBuffer (bypasses Keras)

    For INT8 quantization, use calibration_data (list of .npy file paths,
    one per model input). Data should already be preprocessed before saving.

    float_output: only applies when int8_model=True. Keeps the internal
    graph fully int8 (NPU-friendly) but appends a native DEQUANTIZE op
    after the last op, so the .tflite file's output tensor is declared
    float32 and any runtime reading tensor metadata straight from the
    model (e.g. Axis's larod) hands back real float values with no
    manual scale/zero_point dequantization needed downstream. Ignored
    when use_direct_ir=True (the direct IR backend doesn't do PTQ).

    Returns dict with keys: tflite (file path), tflite_error (max element error),
    and for Keras path: keras (file path), keras_error.
    """
    model_proto = load_onnx_modelproto(onnx_model_path, input_node_names,
                                       output_node_names, need_simplify)

    # Rewrite any Relu(Concat(...)) into Concat(Relu(...), ...) so each
    # branch's Relu becomes fusable into its own producer (usually a Conv2D)
    # during INT8 quantization, instead of surviving as a standalone
    # Activation layer with its own independently-calibrated (and typically
    # mismatched-scale) quantization -- which is what Ethos-N NPU backends
    # reject with "Layer of type Activation is not supported ... falling
    # back to the next backend". See utils/relu_concat_fix.py for details.
    if int8_model:
        push_relu_before_concat(model_proto)

    if use_direct_ir:
        return _convert_direct(model_proto, onnx_model_path, output_path)

    return _convert_keras(model_proto, onnx_model_path, output_path,
                          native_groupconv, weight_quant, fp16_model,
                          int8_model, calibration_data, target_formats,
                          float_output)
