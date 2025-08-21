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
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (182400x4xf32) <- (182400x4xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_10, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x200x304xf32, 1x12x200x304xf32, 1x2xf32, 182400x4xf32, 182400x4xf32)
        generate_proposals_2, generate_proposals_3, generate_proposals_4 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                data_0,
                data_5,
                data_15,
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

        # pd_op.full_like: (45600x4xf32) <- (45600x4xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_11, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x100x152xf32, 1x12x100x152xf32, 1x2xf32, 45600x4xf32, 45600x4xf32)
        generate_proposals_5, generate_proposals_6, generate_proposals_7 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                data_1,
                data_6,
                data_15,
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

        # pd_op.full_like: (11400x4xf32) <- (11400x4xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            data_12, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x50x76xf32, 1x12x50x76xf32, 1x2xf32, 11400x4xf32, 11400x4xf32)
        generate_proposals_8, generate_proposals_9, generate_proposals_10 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                data_2,
                data_7,
                data_15,
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

        # pd_op.full_like: (2850x4xf32) <- (2850x4xf32, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            data_13, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x25x38xf32, 1x12x25x38xf32, 1x2xf32, 2850x4xf32, 2850x4xf32)
        generate_proposals_11, generate_proposals_12, generate_proposals_13 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                data_3,
                data_8,
                data_15,
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

        # pd_op.full_like: (741x4xf32) <- (741x4xf32, 1xf32)
        full_like_4 = paddle._C_ops.full_like(
            data_14, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x3x13x19xf32, 1x12x13x19xf32, 1x2xf32, 741x4xf32, 741x4xf32)
        generate_proposals_14, generate_proposals_0, generate_proposals_1 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                data_4,
                data_9,
                data_15,
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

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32]) <- (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32)
        combine_0 = [
            generate_proposals_2,
            generate_proposals_5,
            generate_proposals_8,
            generate_proposals_11,
            generate_proposals_14,
        ]

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)

        # builtin.combine: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32)
        combine_1 = [
            generate_proposals_3,
            generate_proposals_6,
            generate_proposals_9,
            generate_proposals_12,
            generate_proposals_0,
        ]

        # pd_op.concat: (-1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.flatten: (-1xf32) <- (-1x1xf32)
        flatten_0 = paddle._C_ops.flatten(concat_1, 0, 1)

        # pd_op.shape64: (1xi64) <- (-1xf32)
        shape64_0 = paddle._C_ops.shape64(flatten_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (1xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.cast: (xi32) <- (xi64)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("2000"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi32, xi32)
        greater_than_0 = paddle._C_ops.greater_than(cast_0, full_2)
        return (
            greater_than_0,
            flatten_0,
            concat_0,
            cast_0,
            generate_proposals_0,
            generate_proposals_1,
        )
