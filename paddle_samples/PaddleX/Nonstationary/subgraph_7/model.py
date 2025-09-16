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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.matmul: (16x144x512xf32) <- (16x144x512xf32, 512x512xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_5, False, False)
        del data_0, parameter_5

        # pd_op.add: (16x144x512xf32) <- (16x144x512xf32, 512xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_4)
        del parameter_4

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [16, 144, 8, -1]

        # pd_op.reshape: (16x144x8x64xf32) <- (16x144x512xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_0, full_int_array_0)

        # pd_op.matmul: (16x144x512xf32) <- (16x144x512xf32, 512x512xf32)
        matmul_1 = paddle._C_ops.matmul(data_1, parameter_3, False, False)
        del data_1, parameter_3

        # pd_op.add: (16x144x512xf32) <- (16x144x512xf32, 512xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_2)
        del parameter_2

        # pd_op.reshape: (16x144x8x64xf32) <- (16x144x512xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_1, full_int_array_0)

        # pd_op.matmul: (16x144x512xf32) <- (16x144x512xf32, 512x512xf32)
        matmul_2 = paddle._C_ops.matmul(data_2, parameter_1, False, False)
        del data_2, parameter_1

        # pd_op.add: (16x144x512xf32) <- (16x144x512xf32, 512xf32)
        add_2 = paddle._C_ops.add(matmul_2, parameter_0)
        del parameter_0

        # pd_op.reshape: (16x144x8x64xf32) <- (16x144x512xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_2, full_int_array_0)
        del add_0, add_1, add_2, full_int_array_0, matmul_0, matmul_1, matmul_2

        return reshape_0, reshape_1, reshape_2
