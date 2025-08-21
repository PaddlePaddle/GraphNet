import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.nonzero: (-1x1xi64) <- (8400xb)
        nonzero_0 = paddle._C_ops.nonzero(data_1)

        # pd_op.assign: (-1x1xi64) <- (-1x1xi64)
        assign_0 = nonzero_0

        # pd_op.gather_nd: (-1x25600xf32) <- (8400x25600xf32, -1x1xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(data_0, nonzero_0)

        # pd_op.gather_nd: (-1x32xf32) <- (8400x32xf32, -1x1xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(data_2, nonzero_0)

        # pd_op.gather_nd: (-1x4xf32) <- (8400x4xf32, -1x1xi64)
        gather_nd_2 = paddle._C_ops.gather_nd(data_3, nonzero_0)

        # pd_op.gather_nd: (-1x1xf32) <- (8400x1xf32, -1x1xi64)
        gather_nd_3 = paddle._C_ops.gather_nd(data_4, nonzero_0)
        return assign_0, gather_nd_0, gather_nd_1, gather_nd_2, gather_nd_3
