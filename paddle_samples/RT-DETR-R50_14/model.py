import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (1xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_1, dtype="int64")

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [100]

        # pd_op.tile: (100xi64) <- (1xi64, 1xi64)
        tile_0 = paddle._C_ops.tile(arange_0, full_int_array_0)

        # pd_op.assign: (100xi64) <- (100xi64)
        assign_2 = tile_0

        # pd_op.assign: (100xi64) <- (100xi64)
        assign_1 = tile_0

        # pd_op.assign: (100xi64) <- (100xi64)
        assign_0 = tile_0
        return tile_0, assign_0, assign_1, assign_2
