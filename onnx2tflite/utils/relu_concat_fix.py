"""
Rewrite `Relu(Concat(a, b, c, ...))` into `Concat(Relu(a), Relu(b), Relu(c), ...)`.

Both are mathematically identical: ReLU is a pure elementwise function, and
Concat just juxtaposes tensors along an axis without any interaction between
the pieces, so applying ReLU to the whole concatenated tensor is exactly the
same as applying it to each piece individually before concatenating.

Why this matters for INT8 TFLite conversion (this is the root cause behind
the Arm NN / Ethos-N NPU warning: "Layer of type Activation is not supported
... falling back to the next backend"): TFLite's PTQ converter can only fuse
a ReLU into the preceding op (Conv2D/DepthwiseConv2D/Add/etc.'s
`fused_activation_function`) when the ReLU directly follows a single fusable
op. A standalone `Relu(Concat(...))` can't be fused into anything -- Concat
has no fused-activation slot in TFLite's schema -- so it stays a separate
"Activation" layer with its OWN independently-calibrated INT8 quantization,
which typically ends up with a different (tighter) scale than its input,
since post-ReLU values are non-negative and calibration picks a range to
match. Ethos-N's Activation-layer support check requires identical
input/output quantization for a pure clamp op and rejects it otherwise,
falling back to a slower backend for that one op.

Pushing the ReLU before the Concat means each per-branch ReLU now directly
follows whatever produced that branch (typically a Conv2D) -- the same
fusable pattern the rest of a typical model's ReLUs already use successfully
-- so it disappears as a standalone layer during INT8 conversion instead of
getting its own mismatched-scale quantization.

This is applied as a preprocessing pass on the ONNX model, in
onnx_converter() right after loading it, so it benefits both the Keras and
direct-IR backends uniformly.
"""

import logging

from onnx import helper

LOG = logging.getLogger("relu_concat_fix:")


def _find_producer(graph, tensor_name):
    for node in graph.node:
        if tensor_name in node.output:
            return node
    return None


def _find_consumers(graph, tensor_name):
    return [node for node in graph.node if tensor_name in node.input]


def push_relu_before_concat(model_proto) -> int:
    """
    Find every Relu node whose sole input is produced by a Concat node,
    where that Concat's output is consumed ONLY by that Relu (no other
    consumers, and not itself a graph output), and rewrite it in place:
    apply Relu to each of the Concat's branches individually instead, then
    Concat those. The Relu's original output tensor NAME is preserved (the
    Concat's output gets renamed to it), so nothing downstream needs to
    change.

    Concats with other consumers besides the Relu, or that are graph
    outputs, are left untouched (rewriting them would change what those
    OTHER consumers see) -- a debug message is logged explaining why each
    such case was skipped.

    Returns the number of patterns rewritten.
    """
    graph = model_proto.graph
    # Snapshot the Relu list up front since we'll be mutating graph.node.
    relu_nodes = [n for n in graph.node if n.op_type == "Relu"]

    fixed = 0
    for relu_node in relu_nodes:
        concat_node = _find_producer(graph, relu_node.input[0])
        if concat_node is None or concat_node.op_type != "Concat":
            continue

        concat_out = concat_node.output[0]
        consumers = _find_consumers(graph, concat_out)
        is_graph_output = any(o.name == concat_out for o in graph.output)
        if len(consumers) != 1 or is_graph_output:
            LOG.debug(f"Skipping '{relu_node.name}': its Concat '{concat_node.name}' "
                      f"has {len(consumers)} consumer(s) and "
                      f"{'is' if is_graph_output else 'is not'} a graph output "
                      f"(only rewriting Concats whose SOLE consumer is this Relu).")
            continue

        axis_attr = next((a for a in concat_node.attribute if a.name == "axis"), None)
        concat_index = list(graph.node).index(concat_node)
        original_branches = list(concat_node.input)

        new_branch_names = []
        new_relu_nodes = []
        for i, branch in enumerate(original_branches):
            new_name = f"{branch}_relu_pre_concat"
            new_relu = helper.make_node(
                "Relu", [branch], [new_name], name=f"{relu_node.name}_branch{i}"
            )
            new_relu_nodes.append(new_relu)
            new_branch_names.append(new_name)

        # Rewire the Concat to consume the per-branch Relu outputs, and
        # steal the ORIGINAL Relu node's output name for the Concat's own
        # output -- so every downstream consumer keeps working unchanged.
        del concat_node.input[:]
        concat_node.input.extend(new_branch_names)
        concat_node.output[0] = relu_node.output[0]

        # Insert the new per-branch Relu nodes right before the Concat's
        # position, to keep the graph topologically sorted (their
        # producers are earlier in the list; the Concat now consumes them
        # right after).
        for offset, new_relu in enumerate(new_relu_nodes):
            graph.node.insert(concat_index + offset, new_relu)

        graph.node.remove(relu_node)

        LOG.info(f"'{relu_node.name}': rewrote Relu(Concat(...)) into "
                 f"Concat(Relu(...) x{len(new_branch_names)}) before '{concat_node.name}' "
                 f"(axis={axis_attr.i if axis_attr else None}) so each branch's Relu can be "
                 f"fused into its own producer during INT8 conversion.")
        fixed += 1

    if fixed:
        LOG.info(f"push_relu_before_concat: rewrote {fixed} Relu(Concat(...)) pattern(s).")

    return fixed
