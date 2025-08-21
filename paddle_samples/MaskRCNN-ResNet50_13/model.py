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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
    ):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (-1x2048x7x7xf32) <- (512x2048x7x7xf32, -1x1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_4, data_2, full_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.conv2d_transpose: (-1x256x14x14xf32) <- (-1x2048x7x7xf32, 2048x256x2x2xf32, 0xi64)
        conv2d_transpose_0 = paddle._C_ops.conv2d_transpose(
            gather_0,
            parameter_3,
            [2, 2],
            [0, 0],
            [],
            full_int_array_0,
            "EXPLICIT",
            1,
            [1, 1],
            "NCHW",
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_2, full_int_array_1)

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 1x256x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_transpose_0, reshape_0)

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu_0 = paddle._C_ops.relu(add_1)

        # pd_op.conv2d: (-1x2x14x14xf32) <- (-1x256x14x14xf32, 2x256x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            relu_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, -1, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_0, full_int_array_2)

        # pd_op.add: (-1x2x14x14xf32) <- (-1x2x14x14xf32, 1x2x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_1)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (-1x2xf32) <- (-1xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            data_0 % paddle.cast(full_1, data_0.dtype), full_1
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [2, 3]

        # pd_op.unsqueeze: (-1x2x1x1xf32) <- (-1x2xf32, 2xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(one_hot_0, full_int_array_3)

        # pd_op.expand_as: (-1x2x14x14xf32) <- (-1x2x1x1xf32, -1x2x14x14xf32)
        expand_as_0 = paddle._C_ops.expand_as(unsqueeze_1, add_0, [-1, 2, 14, 14])

        # pd_op.nonzero: (-1x4xi64) <- (-1x2x14x14xf32)
        nonzero_0 = paddle._C_ops.nonzero(expand_as_0)

        # pd_op.gather_nd: (-1xf32) <- (-1x2x14x14xf32, -1x4xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(add_0, nonzero_0)

        # pd_op.shape64: (4xi64) <- (-1x2x14x14xf32)
        shape64_0 = paddle._C_ops.shape64(add_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("14"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [slice_0, full_2, full_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (-1x14x14xf32) <- (-1xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(gather_nd_0, stack_0)

        # pd_op.cast: (-1x14x14xf32) <- (-1x14x14xi32)
        cast_0 = paddle._C_ops.cast(data_1, paddle.float32)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [1, 2]

        # pd_op.unsqueeze: (-1x1x1xf32) <- (-1xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_3, full_int_array_6)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.sigmoid_cross_entropy_with_logits: (-1x14x14xf32) <- (-1x14x14xf32, -1x14x14xf32, None)
        sigmoid_cross_entropy_with_logits_0 = (
            paddle._C_ops.sigmoid_cross_entropy_with_logits(
                reshape_2, cast_0, None, False, -100
            )
        )

        # pd_op.multiply: (-1x14x14xf32) <- (-1x14x14xf32, -1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(
            sigmoid_cross_entropy_with_logits_0, unsqueeze_0
        )

        # pd_op.mean_all: (xf32) <- (-1x14x14xf32)
        mean_all_0 = paddle._C_ops.mean_all(multiply_0)
        return (
            full_0,
            gather_0,
            full_int_array_0,
            conv2d_transpose_0,
            reshape_0,
            relu_0,
            conv2d_0,
            reshape_1,
            add_0,
            nonzero_0,
            gather_nd_0,
            reshape_2,
            cast_0,
            unsqueeze_0,
            sigmoid_cross_entropy_with_logits_0,
            multiply_0,
            mean_all_0,
        )
