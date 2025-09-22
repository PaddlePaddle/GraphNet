import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
        data_11,
        data_12,
        data_13,
        data_14,
        data_15,
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1x3x-1x-1xf32) <- (4x3x-1x-1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_0

        # pd_op.slice: (1x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_5, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_5

        # pd_op.slice: (1x2xf32) <- (4x2xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_15, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_15

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1x4xf32) <- (-1x4xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_10, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x-1x-1xf32, 1x12x-1x-1xf32, 1x2xf32, -1x4xf32, -1x4xf32)
        generate_proposals_1, generate_proposals_2, generate_proposals_3 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                slice_0,
                slice_1,
                slice_2,
                data_10,
                full_like_0,
                2000,
                2000,
                float("0.7"),
                float("0"),
                float("1"),
                False,
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_10, full_like_0, slice_0, slice_1

        # pd_op.slice: (1x3x-1x-1xf32) <- (4x3x-1x-1xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_1

        # pd_op.slice: (1x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_6, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_6

        # pd_op.full_like: (-1x4xf32) <- (-1x4xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_11, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x-1x-1xf32, 1x12x-1x-1xf32, 1x2xf32, -1x4xf32, -1x4xf32)
        generate_proposals_4, generate_proposals_5, generate_proposals_6 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                slice_3,
                slice_4,
                slice_2,
                data_11,
                full_like_1,
                2000,
                2000,
                float("0.7"),
                float("0"),
                float("1"),
                False,
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_11, full_like_1, slice_3, slice_4

        # pd_op.slice: (1x3x-1x-1xf32) <- (4x3x-1x-1xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_2, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_2

        # pd_op.slice: (1x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_7, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_7

        # pd_op.full_like: (-1x4xf32) <- (-1x4xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            data_12, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x-1x-1xf32, 1x12x-1x-1xf32, 1x2xf32, -1x4xf32, -1x4xf32)
        generate_proposals_7, generate_proposals_8, generate_proposals_9 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                slice_5,
                slice_6,
                slice_2,
                data_12,
                full_like_2,
                2000,
                2000,
                float("0.7"),
                float("0"),
                float("1"),
                False,
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_12, full_like_2, slice_5, slice_6

        # pd_op.slice: (1x3x-1x-1xf32) <- (4x3x-1x-1xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_3, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_3

        # pd_op.slice: (1x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_8, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_8

        # pd_op.full_like: (-1x4xf32) <- (-1x4xf32, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            data_13, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x-1x-1xf32, 1x12x-1x-1xf32, 1x2xf32, -1x4xf32, -1x4xf32)
        generate_proposals_10, generate_proposals_11, generate_proposals_12 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                slice_7,
                slice_8,
                slice_2,
                data_13,
                full_like_3,
                2000,
                2000,
                float("0.7"),
                float("0"),
                float("1"),
                False,
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_13, full_like_3, slice_7, slice_8

        # pd_op.slice: (1x3x-1x-1xf32) <- (4x3x-1x-1xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_4, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_4

        # pd_op.slice: (1x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_9, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_9

        # pd_op.full_like: (-1x4xf32) <- (-1x4xf32, 1xf32)
        full_like_4 = paddle._C_ops.full_like(
            data_14, full_0, paddle.float32, paddle.framework._current_expected_place()
        )
        del full_0

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x-1x-1xf32, 1x12x-1x-1xf32, 1x2xf32, -1x4xf32, -1x4xf32)
        generate_proposals_13, generate_proposals_14, generate_proposals_0 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                slice_9,
                slice_10,
                slice_2,
                data_14,
                full_like_4,
                2000,
                2000,
                float("0.7"),
                float("0"),
                float("1"),
                False,
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_14, full_like_4, slice_10, slice_2, slice_9

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32]) <- (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32)
        combine_0 = [
            generate_proposals_1,
            generate_proposals_4,
            generate_proposals_7,
            generate_proposals_10,
            generate_proposals_13,
        ]
        del (
            generate_proposals_1,
            generate_proposals_10,
            generate_proposals_13,
            generate_proposals_4,
            generate_proposals_7,
        )

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)
        del combine_0

        # builtin.combine: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32)
        combine_1 = [
            generate_proposals_2,
            generate_proposals_5,
            generate_proposals_8,
            generate_proposals_11,
            generate_proposals_14,
        ]
        del (
            generate_proposals_11,
            generate_proposals_2,
            generate_proposals_5,
            generate_proposals_8,
        )

        # pd_op.concat: (-1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)
        del combine_1, full_1

        # pd_op.flatten: (-1xf32) <- (-1x1xf32)
        flatten_0 = paddle._C_ops.flatten(concat_1, 0, 1)
        del concat_1

        # pd_op.shape64: (1xi64) <- (-1xf32)
        shape64_0 = paddle._C_ops.shape64(flatten_0)

        # pd_op.slice: (xi64) <- (1xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del full_int_array_0, full_int_array_1, shape64_0

        # pd_op.cast: (xi32) <- (xi64)
        cast_0 = paddle._C_ops.cast(slice_11, paddle.int32)
        del slice_11

        # pd_op.full: (xi32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("2000"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi32, xi32)
        greater_than_0 = paddle._C_ops.greater_than(cast_0, full_2)
        del cast_0, flatten_0, full_2, generate_proposals_14

        return greater_than_0, concat_0, generate_proposals_0
