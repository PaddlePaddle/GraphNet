import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
    ):
        # pd_op.full: (2xi64) <- ()
        full_3 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_3,
            [2],
            paddle.int64,
            [float("640"), float("640")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.divide: (8400x2xf32) <- (8400x2xf32, 8400x1xf32)
        divide_0 = paddle._C_ops.divide(data_2, data_3)

        # pd_op.shape64: (3xi64) <- (1x8400x68xf32)
        shape64_0 = paddle._C_ops.shape64(data_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("17"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_4, slice_1, full_5, full_6]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (-1x-1x4x17xf32) <- (1x8400x68xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(data_1, stack_3)

        # pd_op.softmax: (-1x-1x4x17xf32) <- (-1x-1x4x17xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_1, -1)

        # pd_op.transpose: (-1x17x-1x4xf32) <- (-1x-1x4x17xf32)
        transpose_0 = paddle._C_ops.transpose(softmax_0, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x1x-1x4xf32) <- (-1x17x-1x4xf32, 1x17x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            transpose_0, parameter_0, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.squeeze: (-1x-1x4xf32) <- (-1x1x-1x4xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(conv2d_0, full_int_array_1)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([-1x-1x2xf32, -1x-1x2xf32]) <- (-1x-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(squeeze_0, 2, full_0)

        # builtin.split: (-1x-1x2xf32, -1x-1x2xf32) <- ([-1x-1x2xf32, -1x-1x2xf32])
        (
            split_1,
            split_0,
        ) = split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1x2xf32) <- (-1x-1x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(split_1, full_1, float("0"), True)

        # pd_op.add: (-1x8400x2xf32) <- (-1x-1x2xf32, 8400x2xf32)
        add_0 = paddle._C_ops.add(scale_0, divide_0)

        # pd_op.add: (-1x8400x2xf32) <- (-1x-1x2xf32, 8400x2xf32)
        add_1 = paddle._C_ops.add(split_0, divide_0)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x8400x2xf32, -1x8400x2xf32]) <- (-1x8400x2xf32, -1x8400x2xf32)
        combine_1 = [add_0, add_1]

        # pd_op.concat: (-1x8400x4xf32) <- ([-1x8400x2xf32, -1x8400x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_2)

        # builtin.combine: ([1x1xi32]) <- (1x1xi32)
        combine_2 = [data_4]

        # pd_op.stack: (1x1x1xi32) <- ([1x1xi32])
        stack_0 = paddle._C_ops.stack(combine_2, 0)

        # builtin.combine: ([1x4xf32]) <- (1x4xf32)
        combine_3 = [data_5]

        # pd_op.stack: (1x1x4xf32) <- ([1x4xf32])
        stack_1 = paddle._C_ops.stack(combine_3, 0)

        # builtin.combine: ([1x1xf32]) <- (1x1xf32)
        combine_4 = [data_7]

        # pd_op.stack: (1x1x1xf32) <- ([1x1xf32])
        stack_2 = paddle._C_ops.stack(combine_4, 0)

        # builtin.combine: ([1x640x640xf32]) <- (1x640x640xf32)
        combine_5 = [data_6]

        # pd_op.stack: (1x1x640x640xf32) <- ([1x640x640xf32])
        stack_4 = paddle._C_ops.stack(combine_5, 0)

        # pd_op.cast: (1x1x640x640xf32) <- (1x1x640x640xf32)
        cast_0 = paddle._C_ops.cast(stack_4, paddle.float32)

        # pd_op.nearest_interp: (1x1x160x160xf32) <- (1x1x640x640xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            cast_0, None, None, None, "NCHW", -1, 160, 160, [], "nearest", False, 0
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [1, -1, 25600]

        # pd_op.reshape: (1x1x25600xf32) <- (1x1x160x160xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(nearest_interp_0, full_int_array_4)

        # pd_op.share_data_: (1x8400x2xf32) <- (1x8400x2xf32)
        share_data__0 = data_0.detach()

        # pd_op.share_data_: (-1x8400x4xf32) <- (-1x8400x4xf32)
        share_data__1 = concat_0.detach()

        # pd_op.multiply: (-1x8400x4xf32) <- (-1x8400x4xf32, 8400x1xf32)
        multiply_0 = paddle._C_ops.multiply(share_data__1, data_3)
        return (
            softmax_0,
            transpose_0,
            conv2d_0,
            assign_0,
            full_0,
            split_0,
            full_1,
            scale_0,
            add_0,
            add_1,
            full_2,
            share_data__0,
            multiply_0,
            stack_0,
            stack_1,
            stack_2,
            reshape_0,
            concat_0,
            divide_0,
            assign_value__0,
        )
