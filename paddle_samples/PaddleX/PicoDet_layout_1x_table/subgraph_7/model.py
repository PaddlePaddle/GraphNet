import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.cast: (xf32) <- (xi64)
        cast_0 = paddle._C_ops.cast(data_2, paddle.float32)
        del data_2

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_0 = paddle.arange(full_0, cast_0, full_1, dtype="float32")
        del cast_0

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(arange_0, full_1, float("0.5"), True)
        del arange_0

        # pd_op.cast: (xf32) <- (xi64)
        cast_1 = paddle._C_ops.cast(data_0, paddle.float32)
        del data_0

        # pd_op.multiply: (-1xf32) <- (-1xf32, xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, cast_1)
        del scale_0

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(data_1, paddle.float32)
        del data_1

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_1 = paddle.arange(full_0, cast_2, full_1, dtype="float32")
        del cast_2, full_0

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(arange_1, full_1, float("0.5"), True)
        del arange_1, full_1

        # pd_op.multiply: (-1xf32) <- (-1xf32, xf32)
        multiply_1 = paddle._C_ops.multiply(scale_1, cast_1)
        del cast_1, scale_1

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [multiply_1, multiply_0]
        del multiply_0, multiply_1

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(split_0, 0, 1)
        del split_0

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(split_1, 0, 1)
        del split_1

        return flatten_0, flatten_1
