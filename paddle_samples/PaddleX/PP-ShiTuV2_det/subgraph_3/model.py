import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (10285xf32) <- (10285x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.unsqueeze: (10285x1xf32) <- (10285xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(slice_1, full_int_array_1)
        del slice_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 1]

        # pd_op.tile: (10285x1xf32) <- (10285x1xf32, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_2)
        del unsqueeze_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (10285xf32) <- (10285x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_3, [1], [1]
        )

        # pd_op.unsqueeze: (10285x1xf32) <- (10285xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(slice_2, full_int_array_1)
        del slice_2

        # pd_op.tile: (10285x1xf32) <- (10285x1xf32, 2xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_1, full_int_array_2)
        del unsqueeze_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (10285xf32) <- (10285x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.unsqueeze: (10285x1xf32) <- (10285xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(slice_3, full_int_array_1)
        del slice_3

        # pd_op.tile: (10285x1xf32) <- (10285x1xf32, 2xi64)
        tile_2 = paddle._C_ops.tile(unsqueeze_2, full_int_array_2)
        del unsqueeze_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        # pd_op.slice: (10285xf32) <- (10285x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_1, [1], full_int_array_4, full_int_array_5, [1], [1]
        )
        del data_1

        # pd_op.unsqueeze: (10285x1xf32) <- (10285xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(slice_4, full_int_array_1)
        del slice_4

        # pd_op.tile: (10285x1xf32) <- (10285x1xf32, 2xi64)
        tile_3 = paddle._C_ops.tile(unsqueeze_3, full_int_array_2)
        del full_int_array_2, unsqueeze_3

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_3, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.subtract: (10285x1xf32) <- (10285x1xf32, 1xf32)
        subtract_0 = paddle._C_ops.subtract(tile_0, slice_5)

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_3, [1], full_int_array_1, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (10285x1xf32) <- (10285x1xf32, 1xf32)
        subtract_1 = paddle._C_ops.subtract(tile_1, slice_6)

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_3, [1], full_int_array_3, full_int_array_4, [1], [1]
        )
        del full_int_array_3

        # pd_op.subtract: (10285x1xf32) <- (1xf32, 10285x1xf32)
        subtract_2 = paddle._C_ops.subtract(slice_7, tile_0)

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_3, [1], full_int_array_4, full_int_array_5, [1], [1]
        )
        del data_3, full_int_array_4, full_int_array_5

        # pd_op.subtract: (10285x1xf32) <- (1xf32, 10285x1xf32)
        subtract_3 = paddle._C_ops.subtract(slice_8, tile_1)

        # builtin.combine: ([10285x1xf32, 10285x1xf32, 10285x1xf32, 10285x1xf32]) <- (10285x1xf32, 10285x1xf32, 10285x1xf32, 10285x1xf32)
        combine_0 = [subtract_0, subtract_1, subtract_2, subtract_3]
        del subtract_0, subtract_1, subtract_2, subtract_3

        # pd_op.stack: (10285x4x1xf32) <- ([10285x1xf32, 10285x1xf32, 10285x1xf32, 10285x1xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)
        del combine_0

        # pd_op.min: (10285x1xf32) <- (10285x4x1xf32, 1xi64)
        min_0 = paddle._C_ops.min(stack_0, full_int_array_1, False)
        del stack_0

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (10285x1xb) <- (10285x1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(min_0, full_0)
        del min_0

        # pd_op.sum: (10285xi64) <- (10285x1xb, 1xi64)
        sum_0 = paddle._C_ops.sum(greater_than_0, full_int_array_1, None, False)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (10285xb) <- (10285xi64, xi64)
        greater_than_1 = paddle._C_ops.greater_than(sum_0, full_1)
        del sum_0

        # pd_op.add: (1xf32) <- (1xf32, 1xf32)
        add_0 = paddle._C_ops.add(slice_5, slice_7)
        del slice_5, slice_7

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_2, float("0"), True)
        del add_0

        # pd_op.add: (1xf32) <- (1xf32, 1xf32)
        add_1 = paddle._C_ops.add(slice_6, slice_8)
        del slice_6, slice_8

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_2, float("0"), True)
        del add_1, full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("2.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (10285x1xf32) <- (10285x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(tile_2, full_3, float("0"), True)
        del tile_2

        # pd_op.subtract: (10285x1xf32) <- (1xf32, 10285x1xf32)
        subtract_4 = paddle._C_ops.subtract(scale_0, scale_2)

        # pd_op.scale: (10285x1xf32) <- (10285x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(tile_3, full_3, float("0"), True)
        del full_3, tile_3

        # pd_op.subtract: (10285x1xf32) <- (1xf32, 10285x1xf32)
        subtract_5 = paddle._C_ops.subtract(scale_1, scale_3)

        # pd_op.add: (10285x1xf32) <- (1xf32, 10285x1xf32)
        add_2 = paddle._C_ops.add(scale_0, scale_2)
        del scale_0, scale_2

        # pd_op.add: (10285x1xf32) <- (1xf32, 10285x1xf32)
        add_3 = paddle._C_ops.add(scale_1, scale_3)
        del scale_1, scale_3

        # pd_op.subtract: (10285x1xf32) <- (10285x1xf32, 10285x1xf32)
        subtract_6 = paddle._C_ops.subtract(tile_0, subtract_4)
        del subtract_4

        # pd_op.subtract: (10285x1xf32) <- (10285x1xf32, 10285x1xf32)
        subtract_7 = paddle._C_ops.subtract(tile_1, subtract_5)
        del subtract_5

        # pd_op.subtract: (10285x1xf32) <- (10285x1xf32, 10285x1xf32)
        subtract_8 = paddle._C_ops.subtract(add_2, tile_0)
        del add_2, tile_0

        # pd_op.subtract: (10285x1xf32) <- (10285x1xf32, 10285x1xf32)
        subtract_9 = paddle._C_ops.subtract(add_3, tile_1)
        del add_3, tile_1

        # builtin.combine: ([10285x1xf32, 10285x1xf32, 10285x1xf32, 10285x1xf32]) <- (10285x1xf32, 10285x1xf32, 10285x1xf32, 10285x1xf32)
        combine_1 = [subtract_6, subtract_7, subtract_8, subtract_9]
        del subtract_6, subtract_7, subtract_8, subtract_9

        # pd_op.stack: (10285x4x1xf32) <- ([10285x1xf32, 10285x1xf32, 10285x1xf32, 10285x1xf32])
        stack_1 = paddle._C_ops.stack(combine_1, 1)
        del combine_1

        # pd_op.min: (10285x1xf32) <- (10285x4x1xf32, 1xi64)
        min_1 = paddle._C_ops.min(stack_1, full_int_array_1, False)
        del stack_1

        # pd_op.greater_than: (10285x1xb) <- (10285x1xf32, xf32)
        greater_than_2 = paddle._C_ops.greater_than(min_1, full_0)
        del full_0, min_1

        # pd_op.sum: (10285xi64) <- (10285x1xb, 1xi64)
        sum_1 = paddle._C_ops.sum(greater_than_2, full_int_array_1, None, False)

        # pd_op.greater_than: (10285xb) <- (10285xi64, xi64)
        greater_than_3 = paddle._C_ops.greater_than(sum_1, full_1)
        del full_1, sum_1

        # pd_op.logical_or: (10285xb) <- (10285xb, 10285xb)
        logical_or_0 = paddle._C_ops.logical_or(greater_than_1, greater_than_3)
        del greater_than_1, greater_than_3

        # pd_op.nonzero: (-1x1xi64) <- (10285xb)
        nonzero_0 = paddle._C_ops.nonzero(logical_or_0)

        # pd_op.squeeze: (-1xi64) <- (-1x1xi64, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(nonzero_0, full_int_array_1)
        del nonzero_0

        # pd_op.cast: (10285x1xi64) <- (10285x1xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.int64)
        del greater_than_0

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (-1x1xi64) <- (10285x1xi64, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(cast_0, squeeze_0, full_4)
        del cast_0

        # pd_op.cast: (-1x1xb) <- (-1x1xi64)
        cast_1 = paddle._C_ops.cast(gather_0, paddle.bool)
        del gather_0

        # pd_op.cast: (10285x1xi64) <- (10285x1xb)
        cast_2 = paddle._C_ops.cast(greater_than_2, paddle.int64)
        del greater_than_2

        # pd_op.gather: (-1x1xi64) <- (10285x1xi64, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(cast_2, squeeze_0, full_4)
        del cast_2, full_4

        # pd_op.cast: (-1x1xb) <- (-1x1xi64)
        cast_3 = paddle._C_ops.cast(gather_1, paddle.bool)
        del gather_1

        # pd_op.logical_and: (-1x1xb) <- (-1x1xb, -1x1xb)
        logical_and_0 = paddle._C_ops.logical_and(cast_1, cast_3)
        del cast_1, cast_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [-1]

        # pd_op.unsqueeze: (-1x1xi64) <- (-1xi64, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_6)
        del full_int_array_6, squeeze_0

        # pd_op.gather_nd: (-1x4xf32) <- (10285x4xf32, -1x1xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(data_2, unsqueeze_4)
        del data_2

        # pd_op.gather_nd: (-1x1xf32) <- (10285x1xf32, -1x1xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(data_0, unsqueeze_4)
        del data_0, unsqueeze_4

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_0 = paddle._C_ops.shape64(gather_nd_1)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del full_int_array_0, full_int_array_1, gather_nd_1, logical_or_0, shape64_0

        return slice_0, gather_nd_0, logical_and_0
