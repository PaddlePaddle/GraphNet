import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (11050xf32) <- (11050x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.unsqueeze: (11050x1xf32) <- (11050xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(slice_0, full_int_array_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 2]

        # pd_op.tile: (11050x2xf32) <- (11050x1xf32, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (11050xf32) <- (11050x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_3, [1], [1]
        )

        # pd_op.unsqueeze: (11050x1xf32) <- (11050xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(slice_1, full_int_array_1)

        # pd_op.tile: (11050x2xf32) <- (11050x1xf32, 2xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_1, full_int_array_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (11050xf32) <- (11050x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.unsqueeze: (11050x1xf32) <- (11050xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(slice_2, full_int_array_1)

        # pd_op.tile: (11050x2xf32) <- (11050x1xf32, 2xi64)
        tile_2 = paddle._C_ops.tile(unsqueeze_2, full_int_array_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        # pd_op.slice: (11050xf32) <- (11050x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_4, full_int_array_5, [1], [1]
        )

        # pd_op.unsqueeze: (11050x1xf32) <- (11050xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(slice_3, full_int_array_1)

        # pd_op.tile: (11050x2xf32) <- (11050x1xf32, 2xi64)
        tile_3 = paddle._C_ops.tile(unsqueeze_3, full_int_array_2)

        # pd_op.slice: (2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.subtract: (11050x2xf32) <- (11050x2xf32, 2xf32)
        subtract_0 = paddle._C_ops.subtract(tile_0, slice_4)

        # pd_op.slice: (2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (11050x2xf32) <- (11050x2xf32, 2xf32)
        subtract_1 = paddle._C_ops.subtract(tile_1, slice_5)

        # pd_op.slice: (2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.subtract: (11050x2xf32) <- (2xf32, 11050x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_6, tile_0)

        # pd_op.slice: (2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [1], full_int_array_4, full_int_array_5, [1], [1]
        )

        # pd_op.subtract: (11050x2xf32) <- (2xf32, 11050x2xf32)
        subtract_3 = paddle._C_ops.subtract(slice_7, tile_1)

        # builtin.combine: ([11050x2xf32, 11050x2xf32, 11050x2xf32, 11050x2xf32]) <- (11050x2xf32, 11050x2xf32, 11050x2xf32, 11050x2xf32)
        combine_0 = [subtract_0, subtract_1, subtract_2, subtract_3]

        # pd_op.stack: (11050x4x2xf32) <- ([11050x2xf32, 11050x2xf32, 11050x2xf32, 11050x2xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)

        # pd_op.min: (11050x2xf32) <- (11050x4x2xf32, 1xi64)
        min_0 = paddle._C_ops.min(stack_0, full_int_array_1, False)

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (11050x2xb) <- (11050x2xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(min_0, full_0)

        # pd_op.sum: (11050xi64) <- (11050x2xb, 1xi64)
        sum_0 = paddle._C_ops.sum(greater_than_0, full_int_array_1, None, False)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (11050xb) <- (11050xi64, xi64)
        greater_than_1 = paddle._C_ops.greater_than(sum_0, full_1)

        # pd_op.add: (2xf32) <- (2xf32, 2xf32)
        add_0 = paddle._C_ops.add(slice_4, slice_6)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2xf32) <- (2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_2, float("0"), True)

        # pd_op.add: (2xf32) <- (2xf32, 2xf32)
        add_1 = paddle._C_ops.add(slice_5, slice_7)

        # pd_op.scale: (2xf32) <- (2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_2, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("2.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (11050x2xf32) <- (11050x2xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(tile_2, full_3, float("0"), True)

        # pd_op.subtract: (11050x2xf32) <- (2xf32, 11050x2xf32)
        subtract_4 = paddle._C_ops.subtract(scale_0, scale_2)

        # pd_op.scale: (11050x2xf32) <- (11050x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(tile_3, full_3, float("0"), True)

        # pd_op.subtract: (11050x2xf32) <- (2xf32, 11050x2xf32)
        subtract_5 = paddle._C_ops.subtract(scale_1, scale_3)

        # pd_op.add: (11050x2xf32) <- (2xf32, 11050x2xf32)
        add_2 = paddle._C_ops.add(scale_0, scale_2)

        # pd_op.add: (11050x2xf32) <- (2xf32, 11050x2xf32)
        add_3 = paddle._C_ops.add(scale_1, scale_3)

        # pd_op.subtract: (11050x2xf32) <- (11050x2xf32, 11050x2xf32)
        subtract_6 = paddle._C_ops.subtract(tile_0, subtract_4)

        # pd_op.subtract: (11050x2xf32) <- (11050x2xf32, 11050x2xf32)
        subtract_7 = paddle._C_ops.subtract(tile_1, subtract_5)

        # pd_op.subtract: (11050x2xf32) <- (11050x2xf32, 11050x2xf32)
        subtract_8 = paddle._C_ops.subtract(add_2, tile_0)

        # pd_op.subtract: (11050x2xf32) <- (11050x2xf32, 11050x2xf32)
        subtract_9 = paddle._C_ops.subtract(add_3, tile_1)

        # builtin.combine: ([11050x2xf32, 11050x2xf32, 11050x2xf32, 11050x2xf32]) <- (11050x2xf32, 11050x2xf32, 11050x2xf32, 11050x2xf32)
        combine_1 = [subtract_6, subtract_7, subtract_8, subtract_9]

        # pd_op.stack: (11050x4x2xf32) <- ([11050x2xf32, 11050x2xf32, 11050x2xf32, 11050x2xf32])
        stack_1 = paddle._C_ops.stack(combine_1, 1)

        # pd_op.min: (11050x2xf32) <- (11050x4x2xf32, 1xi64)
        min_1 = paddle._C_ops.min(stack_1, full_int_array_1, False)

        # pd_op.greater_than: (11050x2xb) <- (11050x2xf32, xf32)
        greater_than_2 = paddle._C_ops.greater_than(min_1, full_0)

        # pd_op.sum: (11050xi64) <- (11050x2xb, 1xi64)
        sum_1 = paddle._C_ops.sum(greater_than_2, full_int_array_1, None, False)

        # pd_op.greater_than: (11050xb) <- (11050xi64, xi64)
        greater_than_3 = paddle._C_ops.greater_than(sum_1, full_1)

        # pd_op.logical_or: (11050xb) <- (11050xb, 11050xb)
        logical_or_0 = paddle._C_ops.logical_or(greater_than_1, greater_than_3)

        # pd_op.nonzero: (-1x1xi64) <- (11050xb)
        nonzero_0 = paddle._C_ops.nonzero(logical_or_0)

        # pd_op.squeeze: (-1xi64) <- (-1x1xi64, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(nonzero_0, full_int_array_1)

        # pd_op.cast: (11050x2xi64) <- (11050x2xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.int64)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (-1x2xi64) <- (11050x2xi64, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(cast_0, squeeze_0, full_4)

        # pd_op.cast: (-1x2xb) <- (-1x2xi64)
        cast_1 = paddle._C_ops.cast(gather_0, paddle.bool)

        # pd_op.cast: (11050x2xi64) <- (11050x2xb)
        cast_2 = paddle._C_ops.cast(greater_than_2, paddle.int64)

        # pd_op.gather: (-1x2xi64) <- (11050x2xi64, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(cast_2, squeeze_0, full_4)

        # pd_op.cast: (-1x2xb) <- (-1x2xi64)
        cast_3 = paddle._C_ops.cast(gather_1, paddle.bool)

        # pd_op.logical_and: (-1x2xb) <- (-1x2xb, -1x2xb)
        logical_and_0 = paddle._C_ops.logical_and(cast_1, cast_3)
        return logical_or_0, squeeze_0, logical_and_0
