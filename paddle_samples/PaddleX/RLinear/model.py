import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0, data_1, data_2):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.mean: (-1x1x1xf32) <- (-1x96x1xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(data_2, full_int_array_0, True)

        # pd_op.share_data_: (-1x1x1xf32) <- (-1x1x1xf32)
        share_data__0 = mean_0.detach()
        del mean_0

        # pd_op.mean: (-1x1x1xf32) <- (-1x96x1xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(data_2, full_int_array_0, True)

        # pd_op.subtract: (-1x96x1xf32) <- (-1x96x1xf32, -1x1x1xf32)
        subtract_0 = paddle._C_ops.subtract(data_2, mean_1)
        del mean_1

        # pd_op.pow: (-1x96x1xf32) <- (-1x96x1xf32)
        pow_0 = paddle._C_ops.pow(subtract_0, float("2"))
        del subtract_0

        # pd_op.sum: (-1x1x1xf32) <- (-1x96x1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(pow_0, full_int_array_0, paddle.float32, True)
        del full_int_array_0, pow_0

        # pd_op.numel: (xi64) <- (-1x96x1xf32)
        numel_0 = paddle._C_ops.numel(data_2)

        # pd_op.cast: (xi64) <- (xi64)
        cast_0 = paddle._C_ops.cast(numel_0, paddle.int64)
        del numel_0

        # pd_op.numel: (xi64) <- (-1x1x1xf32)
        numel_1 = paddle._C_ops.numel(sum_0)

        # pd_op.cast: (xi64) <- (xi64)
        cast_1 = paddle._C_ops.cast(numel_1, paddle.int64)
        del numel_1

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(cast_0, paddle.float32)
        del cast_0

        # pd_op.cast: (xf32) <- (xi64)
        cast_3 = paddle._C_ops.cast(cast_1, paddle.float32)
        del cast_1

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(cast_2, cast_3)
        del cast_2, cast_3

        # pd_op.divide: (-1x1x1xf32) <- (-1x1x1xf32, xf32)
        divide_1 = paddle._C_ops.divide(sum_0, divide_0)
        del divide_0, sum_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1x1xf32) <- (-1x1x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_1, full_0, float("1e-05"), True)
        del divide_1

        # pd_op.sqrt: (-1x1x1xf32) <- (-1x1x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(scale_0)
        del scale_0

        # pd_op.share_data_: (-1x1x1xf32) <- (-1x1x1xf32)
        share_data__1 = sqrt_0.detach()
        del sqrt_0

        # pd_op.subtract: (-1x96x1xf32) <- (-1x96x1xf32, -1x1x1xf32)
        subtract_1 = paddle._C_ops.subtract(data_2, share_data__0)
        del data_2

        # pd_op.divide: (-1x96x1xf32) <- (-1x96x1xf32, -1x1x1xf32)
        divide_2 = paddle._C_ops.divide(subtract_1, share_data__1)
        del subtract_1

        # pd_op.multiply: (-1x96x1xf32) <- (-1x96x1xf32, 1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_2, data_0)
        del divide_2

        # pd_op.add: (-1x96x1xf32) <- (-1x96x1xf32, 1xf32)
        add_1 = paddle._C_ops.add(multiply_0, data_1)
        del multiply_0

        # pd_op.transpose: (-1x1x96xf32) <- (-1x96x1xf32)
        transpose_0 = paddle._C_ops.transpose(add_1, [0, 2, 1])
        del add_1

        # pd_op.matmul: (-1x1x96xf32) <- (-1x1x96xf32, 96x96xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_0, parameter_1, False, False)
        del parameter_1, transpose_0

        # pd_op.add: (-1x1x96xf32) <- (-1x1x96xf32, 96xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        # pd_op.transpose: (-1x96x1xf32) <- (-1x1x96xf32)
        transpose_1 = paddle._C_ops.transpose(add_2, [0, 2, 1])
        del add_2

        # pd_op.subtract: (-1x96x1xf32) <- (-1x96x1xf32, 1xf32)
        subtract_2 = paddle._C_ops.subtract(transpose_1, data_1)
        del data_1, transpose_1

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(data_0, full_0, float("1e-10"), True)
        del data_0, full_0

        # pd_op.divide: (-1x96x1xf32) <- (-1x96x1xf32, 1xf32)
        divide_3 = paddle._C_ops.divide(subtract_2, scale_1)
        del scale_1, subtract_2

        # pd_op.multiply: (-1x96x1xf32) <- (-1x96x1xf32, -1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_3, share_data__1)
        del divide_3

        # pd_op.add: (-1x96x1xf32) <- (-1x96x1xf32, -1x1x1xf32)
        add_0 = paddle._C_ops.add(multiply_1, share_data__0)
        del multiply_1, share_data__0, share_data__1

        return add_0
