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
            [1], float("88"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (88xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")
        del full_0, full_1, full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (88xf32) <- (88xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(arange_0, full_3, float("0.5"), True)
        del arange_0, full_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (88xf32) <- (88xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, float("0"), True)
        del full_4, scale_0

        # builtin.combine: ([88xf32, 88xf32]) <- (88xf32, 88xf32)
        combine_0 = [scale_1, scale_1]
        del scale_1

        # pd_op.meshgrid: ([88x88xf32, 88x88xf32]) <- ([88xf32, 88xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (88x88xf32, 88x88xf32) <- ([88x88xf32, 88x88xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.flatten: (7744xf32) <- (88x88xf32)
        flatten_0 = paddle._C_ops.flatten(split_0, 0, 1)
        del split_0

        # pd_op.flatten: (7744xf32) <- (88x88xf32)
        flatten_1 = paddle._C_ops.flatten(split_1, 0, 1)
        del split_1

        return flatten_0, flatten_1
