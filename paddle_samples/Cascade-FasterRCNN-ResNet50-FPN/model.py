import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (-1x4xf32) <- (1x4xf32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_3, data_2, full_0)

        # pd_op.shape64: (1xi64) <- (512xi64)
        shape64_0 = paddle._C_ops.shape64(data_0)
        return gather_0, shape64_0
