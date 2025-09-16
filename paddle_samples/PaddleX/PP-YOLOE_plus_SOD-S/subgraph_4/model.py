import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, data_0, data_1, data_2, data_3, data_4):
        # pd_op.divide: (-1x2xf32) <- (-1x2xf32, -1x1xf32)
        divide_0 = paddle._C_ops.divide(data_3, data_4)
        del data_3

        # pd_op.shape64: (3xi64) <- (2x-1x40xf32)
        shape64_0 = paddle._C_ops.shape64(data_2)

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
        del full_int_array_0

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
        del full_int_array_2, full_int_array_3, shape64_0

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("10"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, slice_1, full_1, full_2]
        del full_0, full_1, full_2, slice_1

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x-1x4x10xf32) <- (2x-1x40xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(data_2, stack_0)
        del data_2, stack_0

        # pd_op.softmax: (-1x-1x4x10xf32) <- (-1x-1x4x10xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_0, -1)
        del reshape_0

        # pd_op.transpose: (-1x10x-1x4xf32) <- (-1x-1x4x10xf32)
        transpose_0 = paddle._C_ops.transpose(softmax_0, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x1x-1x4xf32) <- (-1x10x-1x4xf32, 1x10x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            transpose_0, parameter_0, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_0

        # pd_op.squeeze: (-1x-1x4xf32) <- (-1x1x-1x4xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(conv2d_0, full_int_array_1)
        del full_int_array_1

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([-1x-1x2xf32, -1x-1x2xf32]) <- (-1x-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(squeeze_0, 2, full_3)
        del squeeze_0

        # builtin.split: (-1x-1x2xf32, -1x-1x2xf32) <- ([-1x-1x2xf32, -1x-1x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1x2xf32) <- (-1x-1x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(split_0, full_4, float("0"), True)
        del split_0

        # pd_op.add: (-1x-1x2xf32) <- (-1x-1x2xf32, -1x2xf32)
        add_0 = paddle._C_ops.add(scale_0, divide_0)

        # pd_op.add: (-1x-1x2xf32) <- (-1x-1x2xf32, -1x2xf32)
        add_1 = paddle._C_ops.add(split_1, divide_0)

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x-1x2xf32, -1x-1x2xf32]) <- (-1x-1x2xf32, -1x-1x2xf32)
        combine_1 = [add_0, add_1]

        # pd_op.concat: (-1x-1x4xf32) <- ([-1x-1x2xf32, -1x-1x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_5)
        del combine_1

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = paddle._C_ops.less_than(data_0, full_6)
        del data_0, full_6

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(less_than_0, paddle.int64)
        del less_than_0

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_7)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_1, full_7)
        del cast_1, full_7

        # pd_op.share_data_: (2x-1x10xf32) <- (2x-1x10xf32)
        share_data__0 = data_1.detach()
        del data_1

        # pd_op.share_data_: (-1x-1x4xf32) <- (-1x-1x4xf32)
        share_data__1 = concat_0.detach()

        # pd_op.multiply: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x1xf32)
        multiply_0 = paddle._C_ops.multiply(share_data__1, data_4)
        del (
            add_0,
            add_1,
            assign_0,
            concat_0,
            conv2d_0,
            data_4,
            divide_0,
            full_3,
            full_4,
            full_5,
            scale_0,
            share_data__1,
            softmax_0,
            split_1,
            transpose_0,
        )

        return share_data__0, multiply_0
