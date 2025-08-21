import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
        data_11,
        data_12,
        data_13,
        data_14,
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.reshape: (9xi32) <- (1x9xi32, 1xi64)
        reshape_0 = paddle._C_ops.reshape(data_6, full_int_array_0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [1, 256, -1]

        # pd_op.reshape: (1x256x144xf32) <- (1x256x12x12xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(data_13, full_int_array_1)

        # pd_op.transpose: (1x144x256xf32) <- (1x256x144xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_11, [0, 2, 1])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 256]

        # pd_op.reshape: (144x256xf32) <- (1x144x256xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_0, full_int_array_2)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.gather: (9x256xf32) <- (144x256xf32, 9xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(reshape_1, reshape_0, full_0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [1, -1, 256]

        # pd_op.reshape: (1x9x256xf32) <- (9x256xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(gather_0, full_int_array_3)

        # pd_op.shape64: (4xi64) <- (1x256x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(data_14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.reshape: (1x256x-1xf32) <- (1x256x-1x-1xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(data_14, full_int_array_1)

        # pd_op.matmul: (1x9x-1xf32) <- (1x9x256xf32, 1x256x-1xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_2, reshape_3, False, False)

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [full_5, slice_1, slice_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (-1x-1x-1xf32) <- (1x9x-1xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(matmul_0, stack_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_7 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_5 = full_int_array_7

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_4 = full_int_array_7

        # pd_op.sum: (xi64) <- (1xi64, 0xi64)
        sum_2 = paddle._C_ops.sum(data_7, full_int_array_7, None, False)

        # pd_op.transpose: (1x40x40x2xf32) <- (1x2x40x40xf32)
        transpose_1 = paddle._C_ops.transpose(data_8, [0, 2, 3, 1])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [-1, 2]

        # pd_op.reshape: (1600x2xf32) <- (1x40x40x2xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_1, full_int_array_8)

        # pd_op.transpose: (1x36x36x2xf32) <- (1x2x36x36xf32)
        transpose_2 = paddle._C_ops.transpose(data_9, [0, 2, 3, 1])

        # pd_op.reshape: (1296x2xf32) <- (1x36x36x2xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_2, full_int_array_8)

        # pd_op.transpose: (1x24x24x2xf32) <- (1x2x24x24xf32)
        transpose_3 = paddle._C_ops.transpose(data_10, [0, 2, 3, 1])

        # pd_op.reshape: (576x2xf32) <- (1x24x24x2xf32, 2xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_3, full_int_array_8)

        # pd_op.transpose: (1x16x16x2xf32) <- (1x2x16x16xf32)
        transpose_4 = paddle._C_ops.transpose(data_11, [0, 2, 3, 1])

        # pd_op.reshape: (256x2xf32) <- (1x16x16x2xf32, 2xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_4, full_int_array_8)

        # pd_op.transpose: (1x12x12x2xf32) <- (1x2x12x12xf32)
        transpose_5 = paddle._C_ops.transpose(data_12, [0, 2, 3, 1])

        # pd_op.reshape: (144x2xf32) <- (1x12x12x2xf32, 2xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_5, full_int_array_8)

        # builtin.combine: ([1600x2xf32, 1296x2xf32, 576x2xf32, 256x2xf32, 144x2xf32]) <- (1600x2xf32, 1296x2xf32, 576x2xf32, 256x2xf32, 144x2xf32)
        combine_1 = [reshape_4, reshape_5, reshape_6, reshape_7, reshape_8]

        # pd_op.concat: (3872x2xf32) <- ([1600x2xf32, 1296x2xf32, 576x2xf32, 256x2xf32, 144x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.reshape: (1600xi64) <- (1x1600xi64, 1xi64)
        reshape_13 = paddle._C_ops.reshape(data_0, full_int_array_0)

        # pd_op.reshape: (1296xi64) <- (1x1296xi64, 1xi64)
        reshape_14 = paddle._C_ops.reshape(data_1, full_int_array_0)

        # pd_op.reshape: (576xi64) <- (1x576xi64, 1xi64)
        reshape_15 = paddle._C_ops.reshape(data_2, full_int_array_0)

        # pd_op.reshape: (256xi64) <- (1x256xi64, 1xi64)
        reshape_16 = paddle._C_ops.reshape(data_3, full_int_array_0)

        # pd_op.reshape: (144xi64) <- (1x144xi64, 1xi64)
        reshape_17 = paddle._C_ops.reshape(data_4, full_int_array_0)

        # builtin.combine: ([1600xi64, 1296xi64, 576xi64, 256xi64, 144xi64]) <- (1600xi64, 1296xi64, 576xi64, 256xi64, 144xi64)
        combine_2 = [reshape_13, reshape_14, reshape_15, reshape_16, reshape_17]

        # pd_op.concat: (3872xi64) <- ([1600xi64, 1296xi64, 576xi64, 256xi64, 144xi64], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x9x-1x-1xf32) <- (1x9x-1x-1xui8)
        cast_1 = paddle._C_ops.cast(data_5, paddle.float32)

        # pd_op.shape64: (3xi64) <- (-1x-1x-1xf32)
        shape64_1 = paddle._C_ops.shape64(reshape_12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_10

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_9, full_int_array_10, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_10, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [full_5, slice_4, slice_5]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.reshape: (-1x-1x-1xf32) <- (1x9x-1x-1xf32, 3xi64)
        reshape_18 = paddle._C_ops.reshape(cast_1, stack_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [1, 2]

        # pd_op.sum: (-1xf32) <- (-1x-1x-1xf32, 2xi64)
        sum_3 = paddle._C_ops.sum(reshape_18, full_int_array_11, None, False)

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (-1xb) <- (-1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(sum_3, full_7)

        # pd_op.cast: (-1xf32) <- (-1xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.sigmoid: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(reshape_12)

        # pd_op.shape64: (3xi64) <- (-1x-1x-1xf32)
        shape64_2 = paddle._C_ops.shape64(sigmoid_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_9, full_int_array_10, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_10, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_4 = [slice_6, full_5]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_4, 0)

        # pd_op.reshape: (-1x-1xf32) <- (-1x-1x-1xf32, 2xi64)
        reshape_9 = paddle._C_ops.reshape(sigmoid_0, stack_2)

        # pd_op.shape64: (3xi64) <- (-1x-1x-1xf32)
        shape64_3 = paddle._C_ops.shape64(reshape_18)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_9, full_int_array_10, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_10, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_5 = [slice_9, full_5]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_5, 0)

        # pd_op.reshape: (-1x-1xf32) <- (-1x-1x-1xf32, 2xi64)
        reshape_10 = paddle._C_ops.reshape(reshape_18, stack_3)

        # pd_op.multiply: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        multiply_0 = paddle._C_ops.multiply(reshape_9, reshape_10)

        # pd_op.sum: (-1xf32) <- (-1x-1xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(multiply_0, full_int_array_10, None, False)

        # pd_op.multiply: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        multiply_1 = paddle._C_ops.multiply(reshape_9, reshape_9)

        # pd_op.sum: (-1xf32) <- (-1x-1xf32, 1xi64)
        sum_5 = paddle._C_ops.sum(multiply_1, full_int_array_10, None, False)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(sum_5, full_1, float("0.001"), True)

        # pd_op.multiply: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        multiply_7 = paddle._C_ops.multiply(reshape_10, reshape_10)

        # pd_op.sum: (-1xf32) <- (-1x-1xf32, 1xi64)
        sum_6 = paddle._C_ops.sum(multiply_7, full_int_array_10, None, False)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(sum_6, full_1, float("0.001"), True)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(sum_4, full_2, float("0"), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_0 = paddle._C_ops.add(scale_0, scale_1)

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_0 = paddle._C_ops.divide(scale_2, add_0)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(divide_0, full_3, float("1"), True)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_2 = paddle._C_ops.multiply(scale_3, cast_0)

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        sum_7 = paddle._C_ops.sum(cast_0, full_int_array_7, None, False)

        # pd_op.add: (1xf32) <- (1xf32, xf32)
        add_1 = paddle._C_ops.add(full_6, sum_7)

        # builtin.combine: ([-1xf32]) <- (-1xf32)
        combine_6 = [multiply_2]

        # pd_op.concat: (-1xf32) <- ([-1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_6, full_0)

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(concat_1, full_int_array_7, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_1 = paddle._C_ops.divide(sum_0, add_1)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(divide_1, full_4, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (3872x3xf32) <- (3872xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            concat_2 % paddle.cast(full_8, concat_2.dtype), full_8
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [2147483647]

        # pd_op.slice: (3872x2xf32) <- (3872x3xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            one_hot_0, [1], full_int_array_10, full_int_array_12, [1], []
        )

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(sum_2, paddle.float32)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_2, full_1, float("1"), True)

        # pd_op.shape64: (2xi64) <- (3872x2xf32)
        shape64_4 = paddle._C_ops.shape64(concat_0)

        # pd_op.full_with_tensor: (3872x2xf32) <- (1xf32, 2xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_1, shape64_4, paddle.float32
        )

        # pd_op.sigmoid_cross_entropy_with_logits: (3872x2xf32) <- (3872x2xf32, 3872x2xf32, None)
        sigmoid_cross_entropy_with_logits_0 = (
            paddle._C_ops.sigmoid_cross_entropy_with_logits(
                concat_0, slice_0, None, False, -100
            )
        )

        # pd_op.sigmoid: (3872x2xf32) <- (3872x2xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(concat_0)

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_3 = paddle._C_ops.multiply(sigmoid_1, slice_0)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        subtract_0 = paddle._C_ops.subtract(full_with_tensor_0, sigmoid_1)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        subtract_1 = paddle._C_ops.subtract(full_with_tensor_0, slice_0)

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_4 = paddle._C_ops.multiply(subtract_0, subtract_1)

        # pd_op.add: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        add_2 = paddle._C_ops.add(multiply_3, multiply_4)

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_9,
            [],
            paddle.float32,
            [float("0.25")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.multiply: (3872x2xf32) <- (xf32, 3872x2xf32)
        multiply_8 = paddle._C_ops.multiply(assign_value__0, slice_0)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, xf32)
        subtract_3 = paddle._C_ops.subtract(full_with_tensor_0, assign_value__0)

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_9 = paddle._C_ops.multiply(subtract_3, subtract_1)

        # pd_op.add: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        add_3 = paddle._C_ops.add(multiply_8, multiply_9)

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_5 = paddle._C_ops.multiply(add_3, sigmoid_cross_entropy_with_logits_0)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        subtract_2 = paddle._C_ops.subtract(full_with_tensor_0, add_2)

        # pd_op.pow: (3872x2xf32) <- (3872x2xf32)
        pow_0 = paddle._C_ops.pow(subtract_2, float("2"))

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        multiply_6 = paddle._C_ops.multiply(pow_0, multiply_5)

        # pd_op.divide: (3872x2xf32) <- (3872x2xf32, xf32)
        divide_2 = paddle._C_ops.divide(multiply_6, scale_4)

        # pd_op.sum: (xf32) <- (3872x2xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(divide_2, full_int_array_7, None, False)

        # builtin.combine: ([1xf32, xf32]) <- (1xf32, xf32)
        combine_7 = [scale_5, sum_1]

        # pd_op.add_n: (1xf32) <- ([1xf32, xf32])
        add_n_0 = paddle._C_ops.add_n(combine_7)
        return (
            reshape_0,
            transpose_0,
            reshape_1,
            full_0,
            gather_0,
            reshape_2,
            reshape_3,
            matmul_0,
            transpose_1,
            reshape_4,
            transpose_2,
            reshape_5,
            transpose_3,
            reshape_6,
            transpose_4,
            reshape_7,
            transpose_5,
            reshape_8,
            assign_0,
            concat_0,
            cast_0,
            sigmoid_0,
            reshape_9,
            reshape_10,
            multiply_0,
            assign_1,
            multiply_1,
            assign_2,
            full_1,
            scale_0,
            scale_1,
            full_2,
            scale_2,
            add_0,
            divide_0,
            full_3,
            scale_3,
            multiply_2,
            add_1,
            assign_3,
            concat_1,
            assign_4,
            sum_0,
            divide_1,
            full_4,
            slice_0,
            scale_4,
            full_with_tensor_0,
            sigmoid_cross_entropy_with_logits_0,
            sigmoid_1,
            multiply_3,
            subtract_0,
            subtract_1,
            multiply_4,
            add_2,
            add_3,
            multiply_5,
            subtract_2,
            pow_0,
            multiply_6,
            divide_2,
            assign_5,
            scale_5,
            sum_1,
            add_n_0,
        )
