import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0, data_1):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("144"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [data_0, full_0, full_1]
        del data_0, full_0, full_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x144x-1xf32) <- (-1x144x8x64xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(data_1, stack_0)
        del data_1, stack_0

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x-1xf32, 512x512xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_0, parameter_1, False, False)
        del parameter_1, reshape_0

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
