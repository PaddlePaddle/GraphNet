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
        data_3,
    ):
        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        matmul_0 = paddle._C_ops.matmul(data_1, parameter_5, False, False)
        del data_1, parameter_5

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_4)
        del matmul_0, parameter_4

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("144"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [data_0, full_0, full_1, full_2]
        del data_0, full_0, full_1, full_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x144x8x-1xf32) <- (-1x144x512xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_0, stack_0)
        del add_0

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        matmul_1 = paddle._C_ops.matmul(data_2, parameter_3, False, False)
        del data_2, parameter_3

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_2)
        del matmul_1, parameter_2

        # pd_op.reshape: (-1x144x8x-1xf32) <- (-1x144x512xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_1, stack_0)
        del add_1

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        matmul_2 = paddle._C_ops.matmul(data_3, parameter_1, False, False)
        del data_3, parameter_1

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        add_2 = paddle._C_ops.add(matmul_2, parameter_0)
        del matmul_2, parameter_0

        # pd_op.reshape: (-1x144x8x-1xf32) <- (-1x144x512xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_2, stack_0)
        del add_2, stack_0

        return reshape_0, reshape_1, reshape_2
