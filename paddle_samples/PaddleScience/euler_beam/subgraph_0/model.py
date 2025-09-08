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
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.matmul: (100x20xf32) <- (100x1xf32, 1x20xf32)
        matmul_0 = paddle._C_ops.matmul(data_1, parameter_7, False, False)
        del data_1, parameter_7

        # pd_op.add: (100x20xf32) <- (100x20xf32, 20xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_6)
        del matmul_0, parameter_6

        # pd_op.tanh: (100x20xf32) <- (100x20xf32)
        tanh_0 = paddle._C_ops.tanh(add_0)
        del add_0

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_1 = paddle._C_ops.matmul(tanh_0, parameter_5, False, False)
        del parameter_5, tanh_0

        # pd_op.add: (100x20xf32) <- (100x20xf32, 20xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_4)
        del matmul_1, parameter_4

        # pd_op.tanh: (100x20xf32) <- (100x20xf32)
        tanh_1 = paddle._C_ops.tanh(add_1)
        del add_1

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_2 = paddle._C_ops.matmul(tanh_1, parameter_3, False, False)
        del parameter_3, tanh_1

        # pd_op.add: (100x20xf32) <- (100x20xf32, 20xf32)
        add_2 = paddle._C_ops.add(matmul_2, parameter_2)
        del matmul_2, parameter_2

        # pd_op.tanh: (100x20xf32) <- (100x20xf32)
        tanh_2 = paddle._C_ops.tanh(add_2)
        del add_2

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 20x1xf32)
        matmul_3 = paddle._C_ops.matmul(tanh_2, parameter_1, False, False)
        del parameter_1, tanh_2

        # pd_op.add: (100x1xf32) <- (100x1xf32, 1xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_0)
        del matmul_3, parameter_0

        # pd_op.subtract: (100x1xf32) <- (100x1xf32, 100x1xf32)
        subtract_0 = paddle._C_ops.subtract(add_3, data_2)
        del data_2

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("2"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.elementwise_pow: (100x1xf32) <- (100x1xf32, xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(subtract_0, full_0)
        del full_0, subtract_0

        # pd_op.multiply: (100x1xf32) <- (100x1xf32, 100x1xf32)
        multiply_0 = paddle._C_ops.multiply(elementwise_pow_0, data_3)
        del data_3, elementwise_pow_0

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [0, 1]

        # pd_op.sum: (xf32) <- (100x1xf32, 2xi64)
        sum_0 = paddle._C_ops.sum(multiply_0, full_int_array_1, paddle.float32, False)
        del full_int_array_1, multiply_0

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full(
            [], float("100"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_0, full_1)
        del add_3, full_1, sum_0

        return divide_0
