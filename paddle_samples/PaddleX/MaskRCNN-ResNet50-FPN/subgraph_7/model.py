import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        parameter_4,
        parameter_5,
        parameter_6,
        parameter_7,
        parameter_8,
        parameter_9,
        parameter_10,
        parameter_11,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2147483647]

        # pd_op.slice: (5x4xf32) <- (5x6xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_0, full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (5xf32) <- (5x6xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del data_0

        # pd_op.cast: (5xi32) <- (5xf32)
        cast_0 = paddle._C_ops.cast(slice_1, paddle.int32)
        del slice_1

        # pd_op.distribute_fpn_proposals: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], [-1xi32, -1xi32, -1xi32, -1xi32], -1x1xi32) <- (5x4xf32, 1xi32)
        (
            distribute_fpn_proposals_0,
            distribute_fpn_proposals_1,
            distribute_fpn_proposals_2,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.distribute_fpn_proposals(
                slice_0, data_1, 2, 5, 4, 224, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_1, slice_0

        # builtin.split: (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = distribute_fpn_proposals_0
        del distribute_fpn_proposals_0

        # builtin.split: (-1xi32, -1xi32, -1xi32, -1xi32) <- ([-1xi32, -1xi32, -1xi32, -1xi32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = distribute_fpn_proposals_1
        del distribute_fpn_proposals_1

        # pd_op.roi_align: (-1x256x14x14xf32) <- (1x256x200x232xf32, -1x4xf32, -1xi32)
        roi_align_0 = paddle._C_ops.roi_align(
            data_2, split_0, split_4, 14, 14, float("0.25"), 0, True
        )
        del data_2, split_0, split_4

        # pd_op.roi_align: (-1x256x14x14xf32) <- (1x256x100x116xf32, -1x4xf32, -1xi32)
        roi_align_1 = paddle._C_ops.roi_align(
            data_3, split_1, split_5, 14, 14, float("0.125"), 0, True
        )
        del data_3, split_1, split_5

        # pd_op.roi_align: (-1x256x14x14xf32) <- (1x256x50x58xf32, -1x4xf32, -1xi32)
        roi_align_2 = paddle._C_ops.roi_align(
            data_4, split_2, split_6, 14, 14, float("0.0625"), 0, True
        )
        del data_4, split_2, split_6

        # pd_op.roi_align: (-1x256x14x14xf32) <- (1x256x25x29xf32, -1x4xf32, -1xi32)
        roi_align_3 = paddle._C_ops.roi_align(
            data_5, split_3, split_7, 14, 14, float("0.03125"), 0, True
        )
        del data_5, split_3, split_7

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32)
        combine_0 = [roi_align_0, roi_align_1, roi_align_2, roi_align_3]
        del roi_align_0, roi_align_1, roi_align_2, roi_align_3

        # pd_op.concat: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.gather: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x1xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(concat_0, distribute_fpn_proposals_2, full_0)
        del concat_0, distribute_fpn_proposals_2, full_0

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            gather_0, parameter_11, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del gather_0, parameter_11

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_10, full_int_array_4)
        del parameter_10

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu_0 = paddle._C_ops.relu(add_0)
        del add_0

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9, relu_0

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_8, full_int_array_4)
        del parameter_8

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 1x256x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_1)
        del conv2d_1, reshape_1

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu_1 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_7, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_7, relu_1

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_6, full_int_array_4)
        del parameter_6

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 1x256x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_2, reshape_2)
        del conv2d_2, reshape_2

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu_2 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_2, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_5, relu_2

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_4, full_int_array_4)
        del parameter_4

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 1x256x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_3, reshape_3)
        del conv2d_3, reshape_3

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu_3 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_5 = []

        # pd_op.conv2d_transpose: (-1x256x28x28xf32) <- (-1x256x14x14xf32, 256x256x2x2xf32, 0xi64)
        conv2d_transpose_0 = paddle._C_ops.conv2d_transpose(
            relu_3,
            parameter_3,
            [2, 2],
            [0, 0],
            [],
            full_int_array_5,
            "EXPLICIT",
            1,
            [1, 1],
            "NCHW",
        )
        del full_int_array_5, parameter_3, relu_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_2, full_int_array_6)
        del full_int_array_6, parameter_2

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_transpose_0, reshape_4)
        del conv2d_transpose_0, reshape_4

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu_4 = paddle._C_ops.relu(add_4)
        del add_4

        # pd_op.conv2d: (-1x2x28x28xf32) <- (-1x256x28x28xf32, 2x256x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_4, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1, relu_4

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_0, full_int_array_4)
        del full_int_array_4, parameter_0

        # pd_op.add: (-1x2x28x28xf32) <- (-1x2x28x28xf32, 1x2x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_4, reshape_5)
        del conv2d_4, reshape_5

        # pd_op.shape64: (4xi64) <- (-1x2x28x28xf32)
        shape64_0 = paddle._C_ops.shape64(add_5)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_1, slice_2, full_2, dtype="int64")
        del full_1, full_2, slice_2

        # pd_op.cast: (-1xi32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(arange_0, paddle.int32)
        del arange_0

        # builtin.combine: ([-1xi32, 5xi32]) <- (-1xi32, 5xi32)
        combine_1 = [cast_1, cast_0]
        del cast_0

        # pd_op.broadcast_tensors: ([5xi32, 5xi32]) <- ([-1xi32, 5xi32])
        broadcast_tensors_0 = paddle._C_ops.broadcast_tensors(combine_1)
        del combine_1

        # builtin.split: (5xi32, 5xi32) <- ([5xi32, 5xi32])
        (
            split_8,
            split_9,
        ) = broadcast_tensors_0
        del broadcast_tensors_0

        # builtin.combine: ([5xi32, 5xi32]) <- (5xi32, 5xi32)
        combine_2 = [split_8, split_9]
        del split_8, split_9

        # pd_op.stack: (5x2xi32) <- ([5xi32, 5xi32])
        stack_0 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.gather_nd: (5x28x28xf32) <- (-1x2x28x28xf32, 5x2xi32)
        gather_nd_0 = paddle._C_ops.gather_nd(add_5, stack_0)
        del add_5, stack_0

        # pd_op.shape64: (1xi64) <- (-1xi32)
        shape64_1 = paddle._C_ops.shape64(cast_1)
        del cast_1

        # pd_op.slice: (xi64) <- (1xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_1

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("28"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [slice_3, full_3, full_3]
        del full_3, slice_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.reshape: (-1x28x28xf32) <- (5x28x28xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(gather_nd_0, stack_1)
        del gather_nd_0, stack_1

        # pd_op.sigmoid: (-1x28x28xf32) <- (-1x28x28xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(reshape_6)
        del reshape_6

        return sigmoid_0
