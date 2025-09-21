import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x300x2xf32, 1x300x2xf32]) <- (1x300x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_0, 2, full_0)
        del data_0, full_0

        # builtin.split: (1x300x2xf32, 1x300x2xf32) <- ([1x300x2xf32, 1x300x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x300x2xf32) <- (1x300x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(split_1, full_1, float("0"), True)
        del full_1, split_1

        # pd_op.subtract: (1x300x2xf32) <- (1x300x2xf32, 1x300x2xf32)
        subtract_0 = paddle._C_ops.subtract(split_0, scale_0)

        # pd_op.add: (1x300x2xf32) <- (1x300x2xf32, 1x300x2xf32)
        add_0 = paddle._C_ops.add(split_0, scale_0)
        del scale_0, split_0

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x300x2xf32, 1x300x2xf32]) <- (1x300x2xf32, 1x300x2xf32)
        combine_0 = [subtract_0, add_0]
        del add_0, subtract_0

        # pd_op.concat: (1x300x4xf32) <- ([1x300x2xf32, 1x300x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0

        # pd_op.divide: (1x2xf32) <- (1x2xf32, 1x2xf32)
        divide_0 = paddle._C_ops.divide(data_3, data_4)
        del data_3, data_4

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x2xf32) <- (1x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(divide_0, full_3, float("0.5"), True)
        del divide_0, full_3

        # pd_op.floor: (1x2xf32) <- (1x2xf32)
        floor_0 = paddle._C_ops.floor(scale_1)
        del scale_1

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x1xf32, 1x1xf32]) <- (1x2xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(floor_0, 2, full_4)
        del full_4

        # builtin.split: (1x1xf32, 1x1xf32) <- ([1x1xf32, 1x1xf32])
        (
            split_2,
            split_3,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.flip: (1x2xf32) <- (1x2xf32)
        flip_0 = paddle._C_ops.flip(floor_0, [1])
        del floor_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2]

        # pd_op.tile: (1x4xf32) <- (1x2xf32, 2xi64)
        tile_1 = paddle._C_ops.tile(flip_0, full_int_array_0)
        del flip_0, full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.unsqueeze: (1x1x4xf32) <- (1x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(tile_1, full_int_array_1)
        del tile_1

        # pd_op.multiply: (1x300x4xf32) <- (1x300x4xf32, 1x1x4xf32)
        multiply_0 = paddle._C_ops.multiply(concat_0, unsqueeze_0)
        del concat_0, unsqueeze_0

        # pd_op.sigmoid: (1x300x2xf32) <- (1x300x2xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(data_1)
        del data_1

        # pd_op.flatten: (1x600xf32) <- (1x300x2xf32)
        flatten_0 = paddle._C_ops.flatten(sigmoid_0, 1, 2)
        del sigmoid_0

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("100"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (1x100xf32, 1x100xi64) <- (1x600xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(flatten_0, full_5, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del flatten_0, full_5

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.remainder: (1x100xi64) <- (1x100xi64, xi64)
        remainder_0 = paddle._C_ops.remainder(topk_1, full_6)

        # pd_op.floor_divide: (1x100xi64) <- (1x100xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(topk_1, full_6)
        del full_6, topk_1

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (1xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_7, full_8, full_8, dtype="int64")
        del full_7, full_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(arange_0, full_int_array_2)
        del arange_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [1, 100]

        # pd_op.tile: (1x100xi64) <- (1x1xi64, 2xi64)
        tile_2 = paddle._C_ops.tile(unsqueeze_1, full_int_array_3)
        del full_int_array_3, unsqueeze_1

        # builtin.combine: ([1x100xi64, 1x100xi64]) <- (1x100xi64, 1x100xi64)
        combine_1 = [tile_2, floor_divide_0]
        del floor_divide_0, tile_2

        # pd_op.stack: (1x100x2xi64) <- ([1x100xi64, 1x100xi64])
        stack_0 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.gather_nd: (1x100x4xf32) <- (1x300x4xf32, 1x100x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(multiply_0, stack_0)
        del multiply_0

        # pd_op.gather_nd: (1x100x160x160xf32) <- (1x300x160x160xf32, 1x100x2xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(data_2, stack_0)
        del data_2, stack_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.slice: (1xf32) <- (1x1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            split_2, [0], full_int_array_4, full_int_array_1, [1], [0]
        )
        del split_2

        # pd_op.cast: (1xi32) <- (1xf32)
        cast_1 = paddle._C_ops.cast(slice_0, paddle.int32)
        del slice_0

        # pd_op.slice: (1xf32) <- (1x1xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            split_3, [0], full_int_array_4, full_int_array_1, [1], [0]
        )
        del full_int_array_4, split_3

        # pd_op.cast: (1xi32) <- (1xf32)
        cast_2 = paddle._C_ops.cast(slice_1, paddle.int32)
        del slice_1

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_2 = [cast_1, cast_2]
        del cast_1, cast_2

        # pd_op.bilinear_interp: (1x100x-1x-1xf32) <- (1x100x160x160xf32, None, [1xi32, 1xi32], None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            gather_nd_1,
            None,
            combine_2,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [],
            "bilinear",
            False,
            0,
        )
        del combine_2, gather_nd_1

        # pd_op.sigmoid: (1x100x-1x-1xf32) <- (1x100x-1x-1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(bilinear_interp_0)
        del bilinear_interp_0

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0.5"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x100x-1x-1xb) <- (1x100x-1x-1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(sigmoid_1, full_9)
        del full_9, sigmoid_1

        # pd_op.cast: (1x100x-1x-1xf32) <- (1x100x-1x-1xb)
        cast_3 = paddle._C_ops.cast(greater_than_0, paddle.float32)
        del greater_than_0

        # pd_op.flatten: (100x-1x-1xf32) <- (1x100x-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(cast_3, 0, 1)
        del cast_3

        # pd_op.cast: (100x-1x-1xi32) <- (100x-1x-1xf32)
        cast_0 = paddle._C_ops.cast(flatten_1, paddle.int32)
        del flatten_1

        # pd_op.unsqueeze: (1x100x1xi64) <- (1x100xi64, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(remainder_0, full_int_array_2)
        del remainder_0

        # pd_op.cast: (1x100x1xf32) <- (1x100x1xi64)
        cast_4 = paddle._C_ops.cast(unsqueeze_2, paddle.float32)
        del unsqueeze_2

        # pd_op.unsqueeze: (1x100x1xf32) <- (1x100xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(topk_0, full_int_array_2)
        del full_int_array_2, topk_0

        # builtin.combine: ([1x100x1xf32, 1x100x1xf32, 1x100x4xf32]) <- (1x100x1xf32, 1x100x1xf32, 1x100x4xf32)
        combine_3 = [cast_4, unsqueeze_3, gather_nd_0]
        del cast_4, gather_nd_0, unsqueeze_3

        # pd_op.concat: (1x100x6xf32) <- ([1x100x1xf32, 1x100x1xf32, 1x100x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_3, full_2)
        del combine_3, full_2

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.int64,
            [float("100")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.cast: (xi32) <- (xi64)
        cast_5 = paddle._C_ops.cast(assign_value__0, paddle.int32)
        del assign_value__0

        # pd_op.tile: (1xi32) <- (xi32, 1xi64)
        tile_0 = paddle._C_ops.tile(cast_5, full_int_array_1)
        del cast_5, full_int_array_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [-1, 6]

        # pd_op.reshape: (100x6xf32) <- (1x100x6xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(concat_1, full_int_array_5)
        del concat_1, full_int_array_5

        return reshape_0, tile_0, cast_0
