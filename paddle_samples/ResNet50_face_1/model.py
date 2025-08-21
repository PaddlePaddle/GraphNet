import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.square: (-1x512xf32) <- (-1x512xf32)
        square_0 = paddle._C_ops.square(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.sum: (-1x1xf32) <- (-1x512xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(square_0, full_int_array_0, None, True)

        # pd_op.sqrt: (-1x1xf32) <- (-1x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(sum_0)

        # pd_op.divide: (-1x512xf32) <- (-1x512xf32, -1x1xf32)
        divide_0 = paddle._C_ops.divide(data_0, sqrt_0)

        # pd_op.square: (512x995xf32) <- (512x995xf32)
        square_1 = paddle._C_ops.square(data_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.sum: (1x995xf32) <- (512x995xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(square_1, full_int_array_1, None, True)

        # pd_op.sqrt: (1x995xf32) <- (1x995xf32)
        sqrt_1 = paddle._C_ops.sqrt(sum_1)

        # pd_op.divide: (512x995xf32) <- (512x995xf32, 1x995xf32)
        divide_1 = paddle._C_ops.divide(data_2, sqrt_1)

        # pd_op.matmul: (-1x995xf32) <- (-1x512xf32, 512x995xf32)
        matmul_0 = paddle._C_ops.matmul(divide_0, divide_1, False, False)

        # pd_op.square: (-1x995xf32) <- (-1x995xf32)
        square_2 = paddle._C_ops.square(matmul_0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x995xf32) <- (-1x995xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(square_2, full_0, float("1"), True)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_1

        # pd_op.scale: (-1x995xf32) <- (-1x995xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_1, float("1e-06"), True)

        # pd_op.sqrt: (-1x995xf32) <- (-1x995xf32)
        sqrt_2 = paddle._C_ops.sqrt(scale_7)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.877583"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x995xf32) <- (-1x995xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_0, full_2, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.479426"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x995xf32) <- (-1x995xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(sqrt_2, full_3, float("0"), True)

        # pd_op.subtract: (-1x995xf32) <- (-1x995xf32, -1x995xf32)
        subtract_0 = paddle._C_ops.subtract(scale_0, scale_1)

        # pd_op.scale: (-1x995xf32) <- (-1x995xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_0, full_1, float("-0.239713"), True)

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("-0.877583"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (-1x995xb) <- (-1x995xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(matmul_0, full_5)

        # pd_op.cast: (-1x995xf32) <- (-1x995xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.multiply: (-1x995xf32) <- (-1x995xf32, -1x995xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, subtract_0)

        # pd_op.scale: (-1x995xf32) <- (-1x995xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(cast_0, full_0, float("1"), True)

        # pd_op.multiply: (-1x995xf32) <- (-1x995xf32, -1x995xf32)
        multiply_1 = paddle._C_ops.multiply(scale_3, scale_2)

        # pd_op.add: (-1x995xf32) <- (-1x995xf32, -1x995xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("995"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (-1x1x995xf32) <- (-1x1xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            data_1 % paddle.cast(full_6, data_1.dtype), full_6
        )

        # pd_op.squeeze: (-1x995xf32) <- (-1x1x995xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(one_hot_0, full_int_array_0)

        # pd_op.multiply: (-1x995xf32) <- (-1x995xf32, -1x995xf32)
        multiply_2 = paddle._C_ops.multiply(squeeze_0, add_0)

        # pd_op.scale: (-1x995xf32) <- (-1x995xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(squeeze_0, full_0, float("1"), True)

        # pd_op.multiply: (-1x995xf32) <- (-1x995xf32, -1x995xf32)
        multiply_3 = paddle._C_ops.multiply(scale_4, matmul_0)

        # pd_op.add: (-1x995xf32) <- (-1x995xf32, -1x995xf32)
        add_1 = paddle._C_ops.add(multiply_2, multiply_3)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("64"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x995xf32) <- (-1x995xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(add_1, full_4, float("0"), True)
        return (
            square_0,
            full_int_array_0,
            sqrt_0,
            divide_0,
            square_1,
            full_int_array_1,
            sqrt_1,
            divide_1,
            matmul_0,
            full_0,
            full_1,
            sqrt_2,
            full_2,
            scale_0,
            full_3,
            scale_1,
            subtract_0,
            assign_0,
            scale_2,
            cast_0,
            multiply_0,
            scale_3,
            multiply_1,
            add_0,
            squeeze_0,
            multiply_2,
            scale_4,
            multiply_3,
            full_4,
            scale_5,
        )
