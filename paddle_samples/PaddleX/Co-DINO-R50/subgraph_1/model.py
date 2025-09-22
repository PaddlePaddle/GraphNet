import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (900xi64) <- ()
        full_1 = paddle._C_ops.full(
            [900], float("4"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_3, full_int_array_0)
        del data_3, full_int_array_0

        # pd_op.gather_nd: (1x1xi32) <- (1x1xi32, 1x1xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(data_0, unsqueeze_0)
        del data_0, unsqueeze_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (1xi32) <- (1x1xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            gather_nd_0, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del full_int_array_2, gather_nd_0

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)
        del slice_0

        # builtin.combine: ([1xi64]) <- (1xi64)
        combine_0 = [data_2]
        del data_2

        # pd_op.index_put: (900xi64) <- (900xi64, [1xi64], 1xi64)
        index_put_0 = paddle._C_ops.index_put(full_1, combine_0, cast_0, False)
        del cast_0

        # pd_op.transpose: (900xi64) <- (900xi64)
        transpose_0 = paddle._C_ops.transpose(index_put_0, [0])
        del index_put_0

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_3 = []

        # pd_op.set_value_with_tensor_: (900xi64) <- (900xi64, 900xi64, 0xi64, 0xi64, 0xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            full_1,
            transpose_0,
            full_int_array_3,
            full_int_array_3,
            full_int_array_3,
            [],
            [],
            [],
        )
        del full_1, transpose_0

        # pd_op.full: (900xf32) <- ()
        full_0 = paddle._C_ops.full(
            [900],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (900x4xf32) <- (900x4xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_1, full_2, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (900x4xf32) <- (900x4xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_1, full_2, paddle.float32, paddle.framework._current_expected_place()
        )
        del data_1, full_2

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (1xf64) <- (1xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_3,
            [1],
            paddle.float64,
            [float("1")],
            paddle.framework._current_expected_place(),
        )
        del full_3

        # pd_op.cast: (1xf32) <- (1xf64)
        cast_1 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.index_put: (900x4xf32) <- (900x4xf32, [1xi64], 1xf32)
        index_put_1 = paddle._C_ops.index_put(full_like_1, combine_0, cast_1, False)
        del cast_1

        # pd_op.transpose: (900x4xf32) <- (900x4xf32)
        transpose_1 = paddle._C_ops.transpose(index_put_1, [0, 1])
        del index_put_1

        # pd_op.set_value_with_tensor_: (900x4xf32) <- (900x4xf32, 900x4xf32, 0xi64, 0xi64, 0xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor_(
            full_like_1,
            transpose_1,
            full_int_array_3,
            full_int_array_3,
            full_int_array_3,
            [],
            [],
            [],
        )
        del full_like_1, transpose_1

        # pd_op.full: (4xi64) <- ()
        full_4 = paddle._C_ops.full(
            [4], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xi64) <- (4xi64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_4,
            [4],
            paddle.int64,
            [float("640"), float("853"), float("640"), float("853")],
            paddle.framework._current_expected_place(),
        )
        del full_4

        # pd_op.cast: (4xf32) <- (4xi64)
        cast_2 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.unsqueeze: (1x4xf32) <- (4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(cast_2, full_int_array_1)
        del cast_2, full_int_array_1

        # pd_op.divide: (1x4xf32) <- (1x4xf32, 1x4xf32)
        divide_0 = paddle._C_ops.divide(data_4, unsqueeze_1)
        del data_4, unsqueeze_1

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(divide_0, 4, full_5)
        del divide_0, full_5

        # builtin.split: (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.add: (1x1xf32) <- (1x1xf32, 1x1xf32)
        add_0 = paddle._C_ops.add(split_0, split_2)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_6, float("0"), True)
        del add_0

        # pd_op.add: (1x1xf32) <- (1x1xf32, 1x1xf32)
        add_1 = paddle._C_ops.add(split_1, split_3)

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_6, float("0"), True)
        del add_1, full_6

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_0 = paddle._C_ops.subtract(split_2, split_0)
        del split_0, split_2

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_3, split_1)
        del split_1, split_3

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32)
        combine_1 = [scale_0, scale_1, subtract_0, subtract_1]
        del scale_0, scale_1, subtract_0, subtract_1

        # pd_op.concat: (1x4xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_7)
        del combine_1, full_7

        # pd_op.index_put: (900x4xf32) <- (900x4xf32, [1xi64], 1x4xf32)
        index_put_2 = paddle._C_ops.index_put(full_like_0, combine_0, concat_0, False)
        del combine_0, concat_0

        # pd_op.transpose: (900x4xf32) <- (900x4xf32)
        transpose_2 = paddle._C_ops.transpose(index_put_2, [0, 1])
        del index_put_2

        # pd_op.set_value_with_tensor_: (900x4xf32) <- (900x4xf32, 900x4xf32, 0xi64, 0xi64, 0xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(
            full_like_0,
            transpose_2,
            full_int_array_3,
            full_int_array_3,
            full_int_array_3,
            [],
            [],
            [],
        )
        del full_int_array_3, full_like_0, transpose_2

        return (
            set_value_with_tensor__0,
            full_0,
            set_value_with_tensor__1,
            set_value_with_tensor__2,
        )
