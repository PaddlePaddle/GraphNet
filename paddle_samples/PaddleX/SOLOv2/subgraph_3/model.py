import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_1, data_0, full_0)
        del data_1

        # pd_op.gather: (-1xf32) <- (-1xf32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(data_2, data_0, full_0)
        del data_2

        # pd_op.gather: (-1xi64) <- (-1xi64, -1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(data_3, data_0, full_0)
        del data_0, data_3, full_0

        return gather_0, gather_1, gather_2
