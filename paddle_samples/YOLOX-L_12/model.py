import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(data_2, full_0, float("1e-07"), True)

        # pd_op.log: (-1x-1xf32) <- (-1x-1xf32)
        log_0 = paddle._C_ops.log(scale_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(log_0, full_1, float("0"), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.squeeze: (-1xf32) <- (-1x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_3, full_int_array_0)

        # pd_op.cast: (-1xi64) <- (-1xf32)
        cast_0 = paddle._C_ops.cast(squeeze_0, paddle.int64)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("4"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (-1x4xf32) <- (-1xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            cast_0 % paddle.cast(full_2, cast_0.dtype), full_2
        )

        # pd_op.cast: (-1x4xf32) <- (-1x4xf32)
        cast_1 = paddle._C_ops.cast(one_hot_0, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.unsqueeze: (1x-1x4xf32) <- (-1x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(cast_1, full_int_array_1)

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [data_0, full_3, full_3]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.tile: (-1x-1x4xf32) <- (1x-1x4xf32, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, stack_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.unsqueeze: (-1x1x4xf32) <- (-1x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_4, full_int_array_2)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_1 = [full_3, data_1, full_3]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.tile: (-1x-1x4xf32) <- (-1x1x4xf32, 3xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_1, stack_1)

        # pd_op.bce_loss: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x4xf32)
        bce_loss_0 = paddle._C_ops.bce_loss(tile_1, tile_0)

        # pd_op.sum: (-1x-1xf32) <- (-1x-1x4xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(bce_loss_0, full_int_array_0, None, False)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(sum_0, full_0, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_1, full_4, float("0"), True)

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        add_1 = paddle._C_ops.add(scale_2, scale_3)

        # pd_op.logical_not: (-1x-1xb) <- (-1x-1xb)
        logical_not_0 = paddle._C_ops.logical_not(data_5)

        # pd_op.cast: (-1x-1xf32) <- (-1x-1xb)
        cast_2 = paddle._C_ops.cast(logical_not_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1e+08"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_2, full_5, float("0"), True)

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        add_0 = paddle._C_ops.add(add_1, scale_4)
        return add_0
