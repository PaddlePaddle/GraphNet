import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.square: (-1x512xf32) <- (-1x512xf32)
        square_0 = paddle._C_ops.square(data_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.sum: (-1x1xf32) <- (-1x512xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(square_0, full_int_array_0, None, True)
        del full_int_array_0, square_0

        # pd_op.sqrt: (-1x1xf32) <- (-1x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(sum_0)
        del sum_0

        # pd_op.divide: (-1x512xf32) <- (-1x512xf32, -1x1xf32)
        divide_0 = paddle._C_ops.divide(data_1, sqrt_0)
        del data_1, sqrt_0

        # pd_op.square: (512x995xf32) <- (512x995xf32)
        square_1 = paddle._C_ops.square(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.sum: (1x995xf32) <- (512x995xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(square_1, full_int_array_1, None, True)
        del full_int_array_1, square_1

        # pd_op.sqrt: (1x995xf32) <- (1x995xf32)
        sqrt_1 = paddle._C_ops.sqrt(sum_1)
        del sum_1

        # pd_op.divide: (512x995xf32) <- (512x995xf32, 1x995xf32)
        divide_1 = paddle._C_ops.divide(data_0, sqrt_1)
        del data_0, sqrt_1

        # pd_op.matmul: (-1x995xf32) <- (-1x512xf32, 512x995xf32)
        matmul_0 = paddle._C_ops.matmul(divide_0, divide_1, False, False)
        del divide_0, divide_1

        return matmul_0
