import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("4"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (4xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="int64")

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [data_0]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.tile: (-1xi64) <- (4xi64, 1xi64)
        tile_0 = paddle._C_ops.tile(arange_0, stack_0)
        return tile_0
