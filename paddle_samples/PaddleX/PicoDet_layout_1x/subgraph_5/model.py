import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 8]

        # pd_op.reshape: (-1x8xf32) <- (-1x-1xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(data_0, full_int_array_0)
        del data_0, full_int_array_0

        # pd_op.softmax: (-1x8xf32) <- (-1x8xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_1, 1)
        del reshape_1

        # pd_op.matmul: (-1xf32) <- (-1x8xf32, 8xf32)
        matmul_0 = paddle._C_ops.matmul(softmax_0, data_1, False, False)
        del data_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(matmul_0, full_int_array_1)
        del full_int_array_1, matmul_0, softmax_0

        return reshape_0
