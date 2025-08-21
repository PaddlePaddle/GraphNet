import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # builtin.combine: ([4x12800x4xf32, 4x9600x4xf32]) <- (4x12800x4xf32, 4x9600x4xf32)
        combine_0 = [data_0, data_1]

        # pd_op.concat: (4x22400x4xf32) <- ([4x12800x4xf32, 4x9600x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([4x12800x2xf32, 4x9600x2xf32]) <- (4x12800x2xf32, 4x9600x2xf32)
        combine_1 = [data_2, data_3]

        # pd_op.concat: (4x22400x2xf32) <- ([4x12800x2xf32, 4x9600x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.unsqueeze: (4x90x1xi32) <- (4x90xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_5, full_int_array_0)

        # pd_op.cast: (4x90x1xi64) <- (4x90x1xi32)
        cast_0 = paddle._C_ops.cast(unsqueeze_0, paddle.int64)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([12800x4xf32, 9600x4xf32]) <- (12800x4xf32, 9600x4xf32)
        combine_2 = [data_6, data_7]

        # pd_op.concat: (22400x4xf32) <- ([12800x4xf32, 9600x4xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 4]

        # pd_op.reshape: (360x4xf32) <- (4x90x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(data_4, full_int_array_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.unsqueeze: (360x1x4xf32) <- (360x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(reshape_1, full_int_array_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.unsqueeze: (1x22400x4xf32) <- (22400x4xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(concat_2, full_int_array_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.slice: (360x1x2xf32) <- (360x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            unsqueeze_1, [2], full_int_array_3, full_int_array_4, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.slice: (360x1x2xf32) <- (360x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_1, [2], full_int_array_4, full_int_array_5, [1], []
        )

        # pd_op.slice: (1x22400x2xf32) <- (1x22400x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_2, [2], full_int_array_3, full_int_array_4, [1], []
        )

        # pd_op.slice: (1x22400x2xf32) <- (1x22400x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_2, [2], full_int_array_4, full_int_array_5, [1], []
        )

        # pd_op.maximum: (360x22400x2xf32) <- (360x1x2xf32, 1x22400x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_0, slice_2)

        # pd_op.minimum: (360x22400x2xf32) <- (360x1x2xf32, 1x22400x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_1, slice_3)

        # pd_op.subtract: (360x22400x2xf32) <- (360x22400x2xf32, 360x22400x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (360x22400x2xf32) <- (360x22400x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_2, full_3)

        # pd_op.prod: (360x22400xf32) <- (360x22400x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_0, False, False)

        # pd_op.subtract: (360x1x2xf32) <- (360x1x2xf32, 360x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_1, slice_0)

        # pd_op.clip: (360x1x2xf32) <- (360x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_2, full_3)

        # pd_op.prod: (360x1xf32) <- (360x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_0, False, False)

        # pd_op.subtract: (1x22400x2xf32) <- (1x22400x2xf32, 1x22400x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_3, slice_2)

        # pd_op.clip: (1x22400x2xf32) <- (1x22400x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_2, full_3)

        # pd_op.prod: (1x22400xf32) <- (1x22400x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_0, False, False)

        # pd_op.add: (360x22400xf32) <- (360x1xf32, 1x22400xf32)
        add_0 = paddle._C_ops.add(prod_1, prod_2)

        # pd_op.subtract: (360x22400xf32) <- (360x22400xf32, 360x22400xf32)
        subtract_3 = paddle._C_ops.subtract(add_0, prod_0)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (360x22400xf32) <- (360x22400xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_3, full_4, float("1e-10"), True)

        # pd_op.divide: (360x22400xf32) <- (360x22400xf32, 360x22400xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [4, -1, 22400]

        # pd_op.reshape: (4x90x22400xf32) <- (360x22400xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(divide_0, full_int_array_6)

        # pd_op.max: (4x22400xf32) <- (4x90x22400xf32, 1xi64)
        max_0 = paddle._C_ops.max(reshape_2, full_int_array_2, False)

        # pd_op.full: (1xi64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.argmax: (4x22400xi64) <- (4x90x22400xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(reshape_2, full_5, False, False, paddle.int64)

        # pd_op.max: (4x90xf32) <- (4x90x22400xf32, 1xi64)
        max_1 = paddle._C_ops.max(reshape_2, full_int_array_4, False)

        # pd_op.full: (1xi64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.argmax: (4x90xi64) <- (4x90x22400xf32, 1xi64)
        argmax_1 = paddle._C_ops.argmax(reshape_2, full_6, False, False, paddle.int64)

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("4"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (4xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_7, full_8, full_9, dtype="int64")

        # pd_op.unsqueeze: (4x1xi64) <- (4xi64, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(arange_0, full_int_array_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [1, 22400]

        # pd_op.tile: (4x22400xi64) <- (4x1xi64, 2xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_3, full_int_array_7)

        # builtin.combine: ([4x22400xi64, 4x22400xi64]) <- (4x22400xi64, 4x22400xi64)
        combine_3 = [tile_1, argmax_0]

        # pd_op.stack: (4x22400x2xi64) <- ([4x22400xi64, 4x22400xi64])
        stack_0 = paddle._C_ops.stack(combine_3, -1)

        # pd_op.gather_nd: (4x22400x4xf32) <- (4x90x4xf32, 4x22400x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(data_4, stack_0)

        # pd_op.gather_nd: (4x22400x1xi64) <- (4x90x1xi64, 4x22400x2xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(cast_0, stack_0)

        # pd_op.full: (4x22400x1xi64) <- ()
        full_10 = paddle._C_ops.full(
            [4, 22400, 1],
            float("1"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.unsqueeze: (4x22400x1xf32) <- (4x22400xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(max_0, full_int_array_0)

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [],
            float("0.35"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.less_than: (4x22400x1xb) <- (4x22400x1xf32, xf32)
        less_than_0 = paddle._C_ops.less_than(unsqueeze_4, full_11)

        # pd_op.where: (4x22400x1xi64) <- (4x22400x1xb, 4x22400x1xi64, 4x22400x1xi64)
        where_0 = paddle._C_ops.where(less_than_0, full_10, gather_nd_1)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("22400"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x1xi64) <- (4x1xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(unsqueeze_3, full_12, float("0"), True)

        # pd_op.add: (4x90xi64) <- (4x1xi64, 4x90xi64)
        add_1 = paddle._C_ops.add(scale_1, argmax_1)

        # pd_op.flatten: (360xi64) <- (4x90xi64)
        flatten_0 = paddle._C_ops.flatten(add_1, 0, 1)

        # pd_op.reshape: (89600x4xf32) <- (4x22400x4xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(gather_nd_0, full_int_array_1)

        # pd_op.scatter: (89600x4xf32) <- (89600x4xf32, 360xi64, 360x4xf32)
        scatter_0 = paddle._C_ops.scatter(reshape_3, flatten_0, reshape_1, True)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [4, -1, 4]

        # pd_op.reshape: (4x22400x4xf32) <- (89600x4xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(scatter_0, full_int_array_8)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [-1, 1]

        # pd_op.reshape: (89600x1xi64) <- (4x22400x1xi64, 2xi64)
        reshape_5 = paddle._C_ops.reshape(where_0, full_int_array_9)

        # pd_op.reshape: (360x1xi64) <- (4x90x1xi64, 2xi64)
        reshape_6 = paddle._C_ops.reshape(cast_0, full_int_array_9)

        # pd_op.scatter: (89600x1xi64) <- (89600x1xi64, 360xi64, 360x1xi64)
        scatter_1 = paddle._C_ops.scatter(reshape_5, flatten_0, reshape_6, True)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_10 = [4, -1, 1]

        # pd_op.reshape: (4x22400x1xi64) <- (89600x1xi64, 3xi64)
        reshape_7 = paddle._C_ops.reshape(scatter_1, full_int_array_10)

        # pd_op.set_value_: (4x22400x1xi64) <- (4x22400x1xi64, 1xi64, 1xi64, 1xi64)
        set_value__0 = paddle._C_ops.set_value_(
            reshape_7,
            full_int_array_3,
            full_int_array_2,
            full_int_array_2,
            [1],
            [],
            [],
            [1],
            [float("1")],
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [4, 1, 1]

        # pd_op.tile: (4x22400x4xf32) <- (1x22400x4xf32, 3xi64)
        tile_2 = paddle._C_ops.tile(unsqueeze_2, full_int_array_11)

        # pd_op.reshape: (89600x4xf32) <- (4x22400x4xf32, 2xi64)
        reshape_8 = paddle._C_ops.reshape(tile_2, full_int_array_1)

        # pd_op.reshape: (89600x4xf32) <- (4x22400x4xf32, 2xi64)
        reshape_9 = paddle._C_ops.reshape(reshape_4, full_int_array_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [3]

        # pd_op.slice: (89600xf32) <- (89600x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            reshape_8, [1], full_int_array_4, full_int_array_12, [1], [1]
        )

        # pd_op.slice: (89600xf32) <- (89600x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            reshape_8, [1], full_int_array_3, full_int_array_2, [1], [1]
        )

        # pd_op.subtract: (89600xf32) <- (89600xf32, 89600xf32)
        subtract_4 = paddle._C_ops.subtract(slice_4, slice_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [4]

        # pd_op.slice: (89600xf32) <- (89600x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            reshape_8, [1], full_int_array_12, full_int_array_13, [1], [1]
        )

        # pd_op.slice: (89600xf32) <- (89600x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            reshape_8, [1], full_int_array_2, full_int_array_4, [1], [1]
        )

        # pd_op.subtract: (89600xf32) <- (89600xf32, 89600xf32)
        subtract_5 = paddle._C_ops.subtract(slice_6, slice_7)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (89600xf32) <- (89600xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(subtract_4, full_13, float("0"), True)

        # pd_op.add: (89600xf32) <- (89600xf32, 89600xf32)
        add_2 = paddle._C_ops.add(slice_5, scale_2)

        # pd_op.scale: (89600xf32) <- (89600xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(subtract_5, full_13, float("0"), True)

        # pd_op.add: (89600xf32) <- (89600xf32, 89600xf32)
        add_3 = paddle._C_ops.add(slice_7, scale_3)

        # pd_op.slice: (89600xf32) <- (89600x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            reshape_9, [1], full_int_array_4, full_int_array_12, [1], [1]
        )

        # pd_op.slice: (89600xf32) <- (89600x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            reshape_9, [1], full_int_array_3, full_int_array_2, [1], [1]
        )

        # pd_op.subtract: (89600xf32) <- (89600xf32, 89600xf32)
        subtract_6 = paddle._C_ops.subtract(slice_8, slice_9)

        # pd_op.slice: (89600xf32) <- (89600x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            reshape_9, [1], full_int_array_12, full_int_array_13, [1], [1]
        )

        # pd_op.slice: (89600xf32) <- (89600x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            reshape_9, [1], full_int_array_2, full_int_array_4, [1], [1]
        )

        # pd_op.subtract: (89600xf32) <- (89600xf32, 89600xf32)
        subtract_7 = paddle._C_ops.subtract(slice_10, slice_11)

        # pd_op.scale: (89600xf32) <- (89600xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(subtract_6, full_13, float("0"), True)

        # pd_op.add: (89600xf32) <- (89600xf32, 89600xf32)
        add_4 = paddle._C_ops.add(slice_9, scale_4)

        # pd_op.scale: (89600xf32) <- (89600xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(subtract_7, full_13, float("0"), True)

        # pd_op.add: (89600xf32) <- (89600xf32, 89600xf32)
        add_5 = paddle._C_ops.add(slice_11, scale_5)

        # pd_op.subtract: (89600xf32) <- (89600xf32, 89600xf32)
        subtract_8 = paddle._C_ops.subtract(add_4, add_2)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("10"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (89600xf32) <- (89600xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(subtract_8, full_14, float("0"), True)

        # pd_op.divide: (89600xf32) <- (89600xf32, 89600xf32)
        divide_1 = paddle._C_ops.divide(scale_6, subtract_4)

        # pd_op.subtract: (89600xf32) <- (89600xf32, 89600xf32)
        subtract_9 = paddle._C_ops.subtract(add_5, add_3)

        # pd_op.scale: (89600xf32) <- (89600xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(subtract_9, full_14, float("0"), True)

        # pd_op.divide: (89600xf32) <- (89600xf32, 89600xf32)
        divide_2 = paddle._C_ops.divide(scale_7, subtract_5)

        # pd_op.divide: (89600xf32) <- (89600xf32, 89600xf32)
        divide_3 = paddle._C_ops.divide(subtract_6, subtract_4)

        # pd_op.log: (89600xf32) <- (89600xf32)
        log_0 = paddle._C_ops.log(divide_3)

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (89600xf32) <- (89600xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(log_0, full_15, float("0"), True)

        # pd_op.divide: (89600xf32) <- (89600xf32, 89600xf32)
        divide_4 = paddle._C_ops.divide(subtract_7, subtract_5)

        # pd_op.log: (89600xf32) <- (89600xf32)
        log_1 = paddle._C_ops.log(divide_4)

        # pd_op.scale: (89600xf32) <- (89600xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(log_1, full_15, float("0"), True)

        # builtin.combine: ([89600xf32, 89600xf32, 89600xf32, 89600xf32]) <- (89600xf32, 89600xf32, 89600xf32, 89600xf32)
        combine_4 = [divide_1, divide_2, scale_8, scale_9]

        # pd_op.stack: (89600x4xf32) <- ([89600xf32, 89600xf32, 89600xf32, 89600xf32])
        stack_1 = paddle._C_ops.stack(combine_4, 1)

        # pd_op.reshape: (4x22400x4xf32) <- (89600x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(stack_1, full_int_array_8)

        # pd_op.full: (xi64) <- ()
        full_16 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (4x22400x1xb) <- (4x22400x1xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(set_value__0, full_16)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_14 = [1, 1, 4]

        # pd_op.tile: (4x22400x4xb) <- (4x22400x1xb, 3xi64)
        tile_0 = paddle._C_ops.tile(not_equal_0, full_int_array_14)

        # pd_op.cast: (4x22400x4xf32) <- (4x22400x4xb)
        cast_1 = paddle._C_ops.cast(tile_0, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_15 = []

        # pd_op.sum: (xf32) <- (4x22400x4xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(cast_1, full_int_array_15, None, False)

        # pd_op.full: (xf32) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(sum_0, full_17)
        return (
            full_0,
            assign_0,
            greater_than_0,
            concat_0,
            tile_0,
            reshape_0,
            concat_1,
            set_value__0,
        )
