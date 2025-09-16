import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xf32) <- (2xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int32)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xf32) <- (2xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del data_4

        # pd_op.cast: (xi32) <- (xf32)
        cast_1 = paddle._C_ops.cast(slice_1, paddle.int32)
        del slice_1

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, float("0"), True)
        del data_0

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(data_1, full_0, float("0"), True)
        del data_1, full_0

        # pd_op.full: (3872x2xf32) <- ()
        full_1 = paddle._C_ops.full(
            [3872, 2],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0.1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (3872x2xb) <- (3872x2xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(data_2, full_2)
        del full_2

        # pd_op.where: (3872x2xf32) <- (3872x2xb, 3872x2xf32, 3872x2xf32)
        where_0 = paddle._C_ops.where(greater_than_0, data_2, full_1)
        del full_1, greater_than_0

        # pd_op.nonzero: (-1x2xi64) <- (3872x2xf32)
        nonzero_0 = paddle._C_ops.nonzero(where_0)
        del where_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.reshape: (7744xf32) <- (3872x2xf32, 1xi64)
        reshape_0 = paddle._C_ops.reshape(data_2, full_int_array_3)
        del data_2, full_int_array_3

        # pd_op.shape64: (2xi64) <- (3872x256xf32)
        shape64_0 = paddle._C_ops.shape64(data_3)

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del shape64_0

        # pd_op.cast: (1xi64) <- (1xi64)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)
        del slice_2

        # pd_op.full: (1xi64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1xi64, 1xi64]) <- (1xi64, 1xi64)
        combine_0 = [cast_2, full_3]
        del cast_2, full_3

        # pd_op.concat: (2xi64) <- ([1xi64, 1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_4)
        del combine_0

        # pd_op.unsqueeze: (1x2xi64) <- (2xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(concat_0, full_int_array_0)
        del concat_0

        # builtin.combine: ([-1x2xi64, 1x2xi64]) <- (-1x2xi64, 1x2xi64)
        combine_1 = [nonzero_0, unsqueeze_0]
        del nonzero_0, unsqueeze_0

        # pd_op.concat: (-1x2xi64) <- ([-1x2xi64, 1x2xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_4)
        del combine_1

        # pd_op.full: (1x256xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1, 256],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # builtin.combine: ([3872x256xf32, 1x256xf32]) <- (3872x256xf32, 1x256xf32)
        combine_2 = [data_3, full_5]
        del data_3, full_5

        # pd_op.concat: (3873x256xf32) <- ([3872x256xf32, 1x256xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_4)
        del combine_2

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # builtin.combine: ([7744xf32, 1xf32]) <- (7744xf32, 1xf32)
        combine_3 = [reshape_0, full_6]
        del full_6, reshape_0

        # pd_op.concat: (7745xf32) <- ([7744xf32, 1xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_4)
        del combine_3

        # pd_op.slice: (-1xi64) <- (-1x2xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del full_int_array_2

        # pd_op.slice: (-1xi64) <- (-1x2xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )
        del full_int_array_0, full_int_array_1

        # pd_op.gather: (-1x256xf32) <- (3873x256xf32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(concat_2, slice_4, full_4)
        del concat_2

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_4, full_7, float("0"), True)
        del full_7, slice_4

        # pd_op.add: (-1xi64) <- (-1xi64, -1xi64)
        add_0 = paddle._C_ops.add(scale_2, slice_3)
        del scale_2

        # pd_op.gather: (-1xf32) <- (7745xf32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(concat_3, add_0, full_4)
        del add_0, concat_1, concat_3, full_4, slice_3

        return gather_0, gather_1, scale_0, scale_1, cast_0, cast_1
