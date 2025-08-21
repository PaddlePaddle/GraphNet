import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x6x200x304xf32) <- (1x6x200x304xui8)
        cast_2 = paddle._C_ops.cast(data_2, paddle.float32)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [-1, 200, 304]

        # pd_op.reshape: (6x200x304xf32) <- (1x6x200x304xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(cast_2, full_int_array_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 2]

        # pd_op.sum: (6xf32) <- (6x200x304xf32, 2xi64)
        sum_2 = paddle._C_ops.sum(reshape_4, full_int_array_2, None, False)

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (6xb) <- (6xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(sum_2, full_6)

        # pd_op.cast: (6xf32) <- (6xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.sigmoid: (6x200x304xf32) <- (6x200x304xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(data_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [6, -1]

        # pd_op.reshape: (6x60800xf32) <- (6x200x304xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(sigmoid_0, full_int_array_3)

        # pd_op.reshape: (6x60800xf32) <- (6x200x304xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(reshape_4, full_int_array_3)

        # pd_op.multiply: (6x60800xf32) <- (6x60800xf32, 6x60800xf32)
        multiply_0 = paddle._C_ops.multiply(reshape_0, reshape_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.sum: (6xf32) <- (6x60800xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(multiply_0, full_int_array_0, None, False)

        # pd_op.multiply: (6x60800xf32) <- (6x60800xf32, 6x60800xf32)
        multiply_1 = paddle._C_ops.multiply(reshape_0, reshape_0)

        # pd_op.sum: (6xf32) <- (6x60800xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(multiply_1, full_int_array_0, None, False)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_0

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(sum_4, full_0, float("0.001"), True)

        # pd_op.multiply: (6x60800xf32) <- (6x60800xf32, 6x60800xf32)
        multiply_10 = paddle._C_ops.multiply(reshape_1, reshape_1)

        # pd_op.sum: (6xf32) <- (6x60800xf32, 1xi64)
        sum_5 = paddle._C_ops.sum(multiply_10, full_int_array_0, None, False)

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(sum_5, full_0, float("0.001"), True)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_1

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(sum_3, full_1, float("0"), True)

        # pd_op.add: (6xf32) <- (6xf32, 6xf32)
        add_0 = paddle._C_ops.add(scale_0, scale_1)

        # pd_op.divide: (6xf32) <- (6xf32, 6xf32)
        divide_0 = paddle._C_ops.divide(scale_2, add_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_2

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(divide_0, full_2, float("1"), True)

        # pd_op.multiply: (6xf32) <- (6xf32, 6xf32)
        multiply_2 = paddle._C_ops.multiply(scale_3, cast_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_4 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_7 = full_int_array_4

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_6 = full_int_array_4

        # pd_op.sum: (xf32) <- (6xf32, 0xi64)
        sum_6 = paddle._C_ops.sum(cast_0, full_int_array_4, None, False)

        # pd_op.add: (1xf32) <- (1xf32, xf32)
        add_5 = paddle._C_ops.add(full_5, sum_6)

        # pd_op.cast: (1x6x200x304xf32) <- (1x6x200x304xui8)
        cast_3 = paddle._C_ops.cast(data_3, paddle.float32)

        # pd_op.reshape: (6x200x304xf32) <- (1x6x200x304xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(cast_3, full_int_array_1)

        # pd_op.sum: (6xf32) <- (6x200x304xf32, 2xi64)
        sum_7 = paddle._C_ops.sum(reshape_5, full_int_array_2, None, False)

        # pd_op.greater_than: (6xb) <- (6xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(sum_7, full_6)

        # pd_op.cast: (6xf32) <- (6xb)
        cast_1 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.sigmoid: (6x200x304xf32) <- (6x200x304xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(data_1)

        # pd_op.reshape: (6x60800xf32) <- (6x200x304xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(sigmoid_1, full_int_array_3)

        # pd_op.reshape: (6x60800xf32) <- (6x200x304xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(reshape_5, full_int_array_3)

        # pd_op.multiply: (6x60800xf32) <- (6x60800xf32, 6x60800xf32)
        multiply_3 = paddle._C_ops.multiply(reshape_2, reshape_3)

        # pd_op.sum: (6xf32) <- (6x60800xf32, 1xi64)
        sum_8 = paddle._C_ops.sum(multiply_3, full_int_array_0, None, False)

        # pd_op.multiply: (6x60800xf32) <- (6x60800xf32, 6x60800xf32)
        multiply_4 = paddle._C_ops.multiply(reshape_2, reshape_2)

        # pd_op.sum: (6xf32) <- (6x60800xf32, 1xi64)
        sum_9 = paddle._C_ops.sum(multiply_4, full_int_array_0, None, False)

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(sum_9, full_0, float("0.001"), True)

        # pd_op.multiply: (6x60800xf32) <- (6x60800xf32, 6x60800xf32)
        multiply_11 = paddle._C_ops.multiply(reshape_3, reshape_3)

        # pd_op.sum: (6xf32) <- (6x60800xf32, 1xi64)
        sum_10 = paddle._C_ops.sum(multiply_11, full_int_array_0, None, False)

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(sum_10, full_0, float("0.001"), True)

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(sum_8, full_1, float("0"), True)

        # pd_op.add: (6xf32) <- (6xf32, 6xf32)
        add_1 = paddle._C_ops.add(scale_4, scale_5)

        # pd_op.divide: (6xf32) <- (6xf32, 6xf32)
        divide_1 = paddle._C_ops.divide(scale_6, add_1)

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(divide_1, full_2, float("1"), True)

        # pd_op.multiply: (6xf32) <- (6xf32, 6xf32)
        multiply_5 = paddle._C_ops.multiply(scale_7, cast_1)

        # pd_op.sum: (xf32) <- (6xf32, 0xi64)
        sum_11 = paddle._C_ops.sum(cast_1, full_int_array_4, None, False)

        # pd_op.add: (1xf32) <- (1xf32, xf32)
        add_2 = paddle._C_ops.add(add_5, sum_11)

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([6xf32, 6xf32]) <- (6xf32, 6xf32)
        combine_0 = [multiply_2, multiply_5]

        # pd_op.concat: (12xf32) <- ([6xf32, 6xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_3)

        # pd_op.sum: (xf32) <- (12xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(concat_0, full_int_array_4, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_2 = paddle._C_ops.divide(sum_0, add_2)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(divide_2, full_4, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (3872x3xf32) <- (3872xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            data_5 % paddle.cast(full_7, data_5.dtype), full_7
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.slice: (3872x2xf32) <- (3872x3xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            one_hot_0, [1], full_int_array_0, full_int_array_5, [1], []
        )

        # pd_op.cast: (xf32) <- (xi64)
        cast_4 = paddle._C_ops.cast(data_6, paddle.float32)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_4, full_0, float("1"), True)

        # pd_op.shape64: (2xi64) <- (3872x2xf32)
        shape64_0 = paddle._C_ops.shape64(data_4)

        # pd_op.full_with_tensor: (3872x2xf32) <- (1xf32, 2xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_0, shape64_0, paddle.float32
        )

        # pd_op.sigmoid_cross_entropy_with_logits: (3872x2xf32) <- (3872x2xf32, 3872x2xf32, None)
        sigmoid_cross_entropy_with_logits_0 = (
            paddle._C_ops.sigmoid_cross_entropy_with_logits(
                data_4, slice_0, None, False, -100
            )
        )

        # pd_op.sigmoid: (3872x2xf32) <- (3872x2xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(data_4)

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_6 = paddle._C_ops.multiply(sigmoid_2, slice_0)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        subtract_0 = paddle._C_ops.subtract(full_with_tensor_0, sigmoid_2)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        subtract_1 = paddle._C_ops.subtract(full_with_tensor_0, slice_0)

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_7 = paddle._C_ops.multiply(subtract_0, subtract_1)

        # pd_op.add: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        add_3 = paddle._C_ops.add(multiply_6, multiply_7)

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_8,
            [],
            paddle.float32,
            [float("0.25")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.multiply: (3872x2xf32) <- (xf32, 3872x2xf32)
        multiply_12 = paddle._C_ops.multiply(assign_value__0, slice_0)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, xf32)
        subtract_3 = paddle._C_ops.subtract(full_with_tensor_0, assign_value__0)

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_13 = paddle._C_ops.multiply(subtract_3, subtract_1)

        # pd_op.add: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        add_4 = paddle._C_ops.add(multiply_12, multiply_13)

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_8 = paddle._C_ops.multiply(add_4, sigmoid_cross_entropy_with_logits_0)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        subtract_2 = paddle._C_ops.subtract(full_with_tensor_0, add_3)

        # pd_op.pow: (3872x2xf32) <- (3872x2xf32)
        pow_0 = paddle._C_ops.pow(subtract_2, float("2"))

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_9 = paddle._C_ops.multiply(pow_0, multiply_8)

        # pd_op.divide: (3872x2xf32) <- (3872x2xf32, xf32)
        divide_3 = paddle._C_ops.divide(multiply_9, scale_8)

        # pd_op.sum: (xf32) <- (3872x2xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(divide_3, full_int_array_4, None, False)
        return (
            cast_0,
            sigmoid_0,
            reshape_0,
            reshape_1,
            multiply_0,
            full_int_array_0,
            multiply_1,
            assign_0,
            full_0,
            scale_0,
            scale_1,
            full_1,
            scale_2,
            add_0,
            divide_0,
            full_2,
            scale_3,
            multiply_2,
            cast_1,
            sigmoid_1,
            reshape_2,
            reshape_3,
            multiply_3,
            assign_1,
            multiply_4,
            assign_2,
            assign_3,
            scale_4,
            scale_5,
            assign_4,
            scale_6,
            add_1,
            divide_1,
            assign_5,
            scale_7,
            multiply_5,
            add_2,
            full_3,
            concat_0,
            assign_6,
            sum_0,
            divide_2,
            full_4,
            slice_0,
            scale_8,
            full_with_tensor_0,
            sigmoid_cross_entropy_with_logits_0,
            sigmoid_2,
            multiply_6,
            subtract_0,
            subtract_1,
            multiply_7,
            add_3,
            add_4,
            multiply_8,
            subtract_2,
            pow_0,
            multiply_9,
            divide_3,
            assign_7,
            scale_9,
            sum_1,
        )
