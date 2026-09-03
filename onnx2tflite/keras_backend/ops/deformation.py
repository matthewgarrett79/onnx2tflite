import logging
import tensorflow as tf

from onnx2tflite.utils.definitions import Layout
from onnx2tflite.utils import OPERATOR, dimension_utils

LOG = logging.getLogger("deformation_layers :")

# Axis/larod profile. Set by keras_builder() from onnx_converter(no_transpose=...); OFF by default, so
# upstream behaviour is unchanged unless a caller asks for it.
#
# larod's DLPU has no Transpose op, so under this flag a Transpose is only ever allowed to be a layout
# RELABEL (elided, see TFTranspose) -- never a data movement. Reshape and Concat then have to express
# their ONNX (channel-first) shapes/axes in channel-last terms themselves, since no physical transpose
# is inserted to put the tensor back into NCHW for them.
NO_TRANSPOSE = False

@OPERATOR.register_operator("Transpose")
class TFTranspose():
    """ONNX Transpose.

    With NO_TRANSPOSE set (the Axis/larod profile -- larod's DLPU has no Transpose op), a transpose is
    never allowed to move data. That's sound for the only transposes a NCHW->NHWC conversion actually
    needs: an ONNX graph in channel-first order gets its Conv outputs materialised channel-LAST by this
    converter, so an ONNX `Transpose(0,2,3,1)` sitting after one is describing a permutation that has
    ALREADY happened physically. It is pure bookkeeping, and eliding it is exact.

    What is NOT safe is eliding a transpose that genuinely reorders data (e.g. swapping two spatial
    axes). The previous revision returned `inputs` unconditionally, which silently produced a wrong
    graph in that case; we now raise instead, naming the node, so the failure is a build error rather
    than a model that converts cleanly and detects nothing.
    """
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.trans_in, self.perm_list = None, None
        self.elide = False

        if kwargs.get("perm_list"):
            self.perm_list = kwargs.get("perm_list")
            for nop in node_outputs:
                layout_dict[nop] = Layout.Channel_First
            return

        perm = [i for i in node_attribute['perm']]
        rank = len(perm)
        input_is_last = layout_dict[node_inputs[0]] == Layout.Channel_Last
        # The two permutations that are nothing but a layout relabel.
        nchw_to_nhwc = [0] + list(range(2, rank)) + [1]
        nhwc_to_nchw = [0, rank - 1] + list(range(1, rank - 1))

        if NO_TRANSPOSE:
            if input_is_last and perm == nchw_to_nhwc:
                # Data is already channel-last; this transpose only says so. Elide, stay channel-last.
                self.elide = True
                for nop in node_outputs:
                    layout_dict[nop] = Layout.Channel_Last
                return
            if input_is_last and perm == nhwc_to_nchw:
                # Elide the data movement but record that consumers should read this as channel-first.
                self.elide = True
                for nop in node_outputs:
                    layout_dict[nop] = Layout.Channel_First
                return
            raise NotImplementedError(
                f"Transpose perm={perm} on a "
                f"{'channel-last' if input_is_last else 'channel-first'} tensor can't be elided, but the "
                f"target runtime (larod) has no Transpose op. This permutation genuinely reorders data, "
                f"so it has to be removed upstream in the ONNX export rather than here. Offending node "
                f"output(s): {list(node_outputs)}")

        for nop in node_outputs:
            layout_dict[nop] = Layout.Channel_First
        self.perm_list = perm
        if input_is_last:
            # LOG.info("Transpose will process tensor after change back to NCHW format.")
            shape_len = len(tensor_grap[node_inputs[0]].shape)
            self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]

    def __call__(self, inputs):
        if self.elide:
            return inputs
        if self.trans_in:
            inputs = tf.transpose(inputs, perm=self.trans_in)
        return tf.transpose(inputs, perm=self.perm_list)

@OPERATOR.register_operator("Slice")
class TFSlice():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        if len(node_inputs) == 1:
            self.starts = node_attribute['starts'][0]
            self.ends = node_attribute['ends'][0]
            self.axis = node_attribute['axes'][0]
            self.steps = 1
        else:
            self.starts = node_weights[node_inputs[1]][0] if node_inputs[1] in node_weights else tensor_grap[node_inputs[1]][0]
            self.axis = node_weights[node_inputs[3]][0] if node_inputs[3] in node_weights else tensor_grap[node_inputs[3]][0]
            self.ends = node_weights[node_inputs[2]][0] if node_inputs[2] in node_weights else tensor_grap[node_inputs[2]][0]
            self.ends = min(self.ends, tensor_grap[node_inputs[0]].shape[self.axis])
            if len(node_inputs) < 5:
                self.steps = 1
            else:
                self.steps = node_weights[node_inputs[4]][0] if node_inputs[4] in node_weights else tensor_grap[node_inputs[4]][0]
        
        shape = tensor_grap[node_inputs[0]].shape.as_list()
        if self.starts < 0:
            self.starts = shape[self.axis] + self.starts
        if self.ends < 0:
            self.ends = shape[self.axis] + self.ends

        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)

    def __call__(self, inputs):
        indices = tf.keras.backend.arange(self.starts, self.ends, step=self.steps)
        return tf.gather(inputs, indices, axis=self.axis)

@OPERATOR.register_operator("Gather")
class TFGather():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        self.axis = node_attribute.get('axis', 0)
        self.indices = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)

    def __call__(self, inputs):
        return tf.gather(inputs, self.indices, axis=self.axis)

@OPERATOR.register_operator("Concat")
class TFConcat():
    """ONNX Concat.

    The concat axis in the ONNX graph is stated in CHANNEL-FIRST terms. When the tensors have been
    materialised channel-last, that axis index has to be remapped -- which is exactly what
    `dimension_utils.channel_to_last_dimension` does (`1`, the channel axis, becomes `-1`; anything
    past it shifts down by one). A previous revision hardcoded `_axis = 1` and then re-derived it from
    each input's rank (`rank == 4 -> last, else 1`); that happens to agree with the remap on the
    detector graphs it was written for, but it discards `node_attribute['axis']` entirely and so is
    wrong for any concat on a non-channel axis.

    Under NO_TRANSPOSE all inputs are aligned channel-LAST, because aligning to channel-first means
    calling tensor_NDC_to_NCD_format, which emits a tf.transpose -- the one thing the Axis path can't
    have.
    """
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        onnx_axis = node_attribute.get('axis', 0)

        # use `count` to count how much more for channel-last to channel-first
        count = 0
        for inp in node_inputs:
            if inp in node_weights:
                count -= 1
            elif layout_dict[inp] == Layout.Channel_Last:
                count += 1
            else:
                count -= 1

        self._gather = []
        if count < 0 and not NO_TRANSPOSE:
            # align to Channel_First
            layout_dict[node_outputs[0]] = Layout.Channel_First
            self._axis = onnx_axis
            for inp in node_inputs:
                if inp in tensor_grap:
                    if layout_dict[inp] == Layout.Channel_Last:
                        tensor_grap[inp] = dimension_utils.tensor_NDC_to_NCD_format(tensor_grap[inp])
                    self._gather.append(tensor_grap[inp])
                else:
                    self._gather.append(node_weights[inp])
        else:
            # align to Channel_Last
            layout_dict[node_outputs[0]] = Layout.Channel_Last
            self._axis = dimension_utils.channel_to_last_dimension(onnx_axis)
            for inp in node_inputs:
                if inp in tensor_grap:
                    if layout_dict[inp] != Layout.Channel_Last:
                        if NO_TRANSPOSE:
                            raise NotImplementedError(
                                f"Concat input {inp!r} is channel-first while its siblings are "
                                f"channel-last; aligning them would need a tf.transpose, which the "
                                f"target runtime (larod) has no op for. Node output(s): "
                                f"{list(node_outputs)}")
                        tensor_grap[inp] = dimension_utils.tensor_NCD_to_NDC_format(tensor_grap[inp])
                    self._gather.append(tensor_grap[inp])
                else:
                    # An initializer -- permuting a constant is free (folded at build time), no runtime op.
                    self._gather.append(dimension_utils.tensor_NCD_to_NDC_format(node_weights[inp]))

    def __call__(self, *args, **kwargs):
        return tf.concat(self._gather, axis=self._axis)

@OPERATOR.register_operator("Reshape")
class TFReshape():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        
        onnx_shape = [int(d) for d in node_weights[node_inputs[1]]]
        self.trans_in = None

        if NO_TRANSPOSE:
            # No transpose is inserted to put the tensor back into NCHW, so the target shape -- which
            # ONNX states in channel-first terms -- has to be expressed channel-last instead.
            #
            # ONNX allows 0 to mean "keep the input's extent at this index", indexed against the
            # CHANNEL-FIRST input. Resolve those before permuting, or they end up pointing at the wrong
            # axis. The input tensor is physically channel-last here, so read its ONNX-equivalent shape
            # back through NDC->NCD to look the extents up.
            if 0 in onnx_shape:
                nhwc_in = [None if d is None else int(d) for d in tensor_grap[node_inputs[0]].shape]
                nchw_in = dimension_utils.shape_NDC_to_NCD_format(nhwc_in)
                onnx_shape = [nchw_in[i] if d == 0 else d for i, d in enumerate(onnx_shape)]

            if onnx_shape.count(-1) > 1:
                raise ValueError(f"Reshape target {onnx_shape} has more than one -1; ONNX permits one. "
                                 f"Node output(s): {list(node_outputs)}")

            # (N, C, D...) -> (N, D..., C). This is the general form of what a previous revision
            # hardcoded as (1, 2, 2, 18920) for rank 4 and (s0, s2, s1) for rank 3 -- both are exactly
            # this helper applied to that graph's shapes.
            self.out_shape = dimension_utils.shape_NCD_to_NDC_format(onnx_shape)
            for nop in node_outputs:
                layout_dict[nop] = Layout.Channel_Last
        else:
            self.out_shape = tuple(onnx_shape)
            # LOG.info("Reshape will process tensor after change back to NCHW format.")
            if layout_dict[node_inputs[0]] == Layout.Channel_Last:
                shape_len = len(tensor_grap[node_inputs[0]].shape)
                self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
            for nop in node_outputs:
                layout_dict[nop] = Layout.Channel_First

    def __call__(self, inputs):
        if self.trans_in:
            inputs = tf.transpose(inputs, perm=self.trans_in)
        inputs = tf.reshape(inputs, shape=self.out_shape)
        return inputs
        
@OPERATOR.register_operator("Flatten")
class TFFlatten():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        num_elements = int(tensor_grap[node_inputs[0]].shape.num_elements()/tensor_grap[node_inputs[0]].shape[0])
        input_shape = tensor_grap[node_inputs[0]].shape
        self.flat = tf.keras.layers.Flatten()
        '''
            ensure memory order match, for example:
            onnx = (B, 2, 3, 4).reshape(B, -1)
            tflite = (B, 3, 4, 2).reshape(B, -1)
            we can observe that:
            onnx.shape == tflite.shape, but np.sum(onnx-tflite) != 0
            it's cause memory order of two vars is different, we must make tflite back to onnx by transpose.
            generally, this situation is general one, below is just special situation and most appear in cnn.
            onnx = (B, 512, 1, 1)
            tflite = (B, 1, 1, 512)
            or = (B, 1, 512, 1)
            these memory order are all same.
        '''
        self.perm = None
        if layout_dict[node_inputs[0]] == Layout.Channel_Last and  num_elements != max(input_shape[1:]):
            self.perm = [0, len(input_shape)-1]
            for i in range(len(input_shape)-2):
                self.perm.append(i+1)

    def __call__(self, inputs):
        if self.perm:
            inputs = tf.transpose(inputs, perm=self.perm)
        return self.flat(inputs)

@OPERATOR.register_operator("Split")
class TFSplit():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.outputs_nums = len(node_outputs)
        self.axis = node_attribute.get("axis", 0)
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)
        split_args = None
        if 'split' in node_attribute:
            split_args = node_attribute['split']
        else:
            assert len(node_inputs) == 2 and node_inputs[1] in node_weights
            split_args = node_weights[node_inputs[1]]
        
        self.indices = []
        start, end = 0, 0
        for i in range(self.outputs_nums):
            end = start + int(split_args[i])
            self.indices.append(tf.keras.backend.arange(start, end, 1))
            start = end

    def __call__(self, inputs):
        return [tf.gather(inputs, indices=indice, axis=self.axis) for indice in self.indices]

@OPERATOR.register_operator("Expand")
class TFExpand():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.shape = node_weights[node_inputs[1]]
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.shape = dimension_utils.shape_NCD_to_NDC_format(self.shape)
    def __call__(self, inputs):
        for i in range(len(self.shape)):
            if int(self.shape[i]//inputs.shape[i]) > 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]//inputs.shape[i]), axis=i)
            elif self.shape[i] < inputs.shape[i] and self.shape[i] != 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]), axis=i)
        return inputs
    
@OPERATOR.register_operator("GatherElements")
class TFGatherElements():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        self.axis = node_attribute.get("axis", 1)
        if 'indices' in node_attribute:
            self.indices = node_attribute['indices']
        elif node_inputs[1] in node_weights:
            self.indices = node_weights[node_inputs[1]]
        else:
            self.indices = tensor_grap[node_inputs[1]]
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)
            # Convert NCD-format indices to NDC to match NHWC input
            if isinstance(self.indices, np.ndarray) and self.indices.ndim > 2:
                self.indices = dimension_utils.tensor_NCD_to_NDC_format(self.indices)

    def gather_elements(self, input_tensor, indices, axis):
        # Get the shape of the input tensor and the indices tensor
        input_shape = tf.shape(input_tensor)
        indices_shape = tf.shape(indices)

        # Create indices for all dimensions
        idx = tf.meshgrid(*[tf.range(s) for s in indices_shape], indexing='ij')
        idx = [tf.cast(i, tf.int64) for i in idx]

        # Replace the axis index with the provided indices
        idx[axis] = tf.cast(indices, tf.int64)

        # Stack indices to form the final gather indices
        gather_indices = tf.stack(idx, axis=-1)

        # Use tf.gather_nd to gather elements
        output_tensor = tf.gather_nd(input_tensor, gather_indices)

        return output_tensor

    def __call__(self, inputs):
        return self.gather_elements(inputs, self.indices, self.axis)
    
@OPERATOR.register_operator("Tile")
class TFTile():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.repeats = node_attribute['repeats'] if 'repeats' in node_attribute else node_weights[node_inputs[1]]
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.repeats = dimension_utils.shape_NCD_to_NDC_format(self.repeats)

    def __call__(self, inputs):
        for i in range(len(self.repeats)):
            if self.repeats[i] > 1:
                inputs = tf.repeat(inputs, self.repeats[i], axis=i)
        return inputs

@OPERATOR.register_operator("Unsqueeze")
class TFUnsqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.axis = node_attribute['axes'] if 'axes' in node_attribute else node_weights[node_inputs[1]]
        if not isinstance(self.axis, int):
            self.axis = int(self.axis[0])
        input_shape = tensor_grap[node_inputs[0]].shape
        if len(input_shape) == 1:
            layout_dict[node_outputs[0]] = Layout.Channel_None
        elif len(input_shape) == 2:
            layout_dict[node_outputs[0]] = Layout.Channel_First
        else:
            layout_dict[node_outputs[0]] = layout_dict[node_inputs[0]]
            if layout_dict[node_inputs[0]] == Layout.Channel_Last:
                self.axis = dimension_utils.channel_to_last_dimension(self.axis)

    def __call__(self, inputs):
        return tf.expand_dims(inputs, self.axis)

@OPERATOR.register_operator("Squeeze")
class TFSqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.axis = node_attribute['axes'] if 'axes' in node_attribute else node_weights[node_inputs[1]]
        if not isinstance(self.axis, int):
            self.axis = int(self.axis[0])
        input_shape = tensor_grap[node_inputs[0]].shape
        if len(input_shape) <= 3:
            layout_dict[node_outputs[0]] = Layout.Channel_None
        if len(input_shape) > 2 and layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)

    def __call__(self, inputs):
        return tf.squeeze(inputs, self.axis)

@OPERATOR.register_operator("DepthToSpace")
class TFDepthToSpace():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.block_size = node_attribute.get("blocksize", 2)
        self.mode = node_attribute.get("mode", "DCR")
        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last

    def __call__(self, inputs):
        if not self.channel_last:
            inputs = dimension_utils.tensor_NDC_to_NCD_format(inputs)
        if self.mode == "DCR":
            return tf.nn.depth_to_space(inputs, self.block_size)
        elif self.mode == "CRD":
            # help want, native tensorflow is not support CRD mode, this way will generate 5 dims op.
            b, h, w, c = inputs.shape
            inputs = tf.reshape(inputs, [b, h, w, c//(self.block_size * self.block_size), self.block_size, self.block_size])
            inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 5, 3])
            inputs = tf.reshape(inputs, [b, h*self.block_size, w*self.block_size, c//(self.block_size * self.block_size)])
            return inputs
        else:
            raise KeyError(f"For DepthToSpace, mode must be [DCR, CRD], not {self.mode}")