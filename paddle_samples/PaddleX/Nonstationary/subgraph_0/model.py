import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (16x8x144x144xf32) <- ()
        full_0 = paddle._C_ops.full(
            [16, 8, 144, 144],
            float("-inf"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (16x8x144x144xf32) <- (16x8x144x144xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            full_0, full_1, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (16x8x144x144xf32) <- (16x8x144x144xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_0, full_1, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (16x1x144x144xb) <- (16x1x144x144xb, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            data_2, full_1, paddle.bool, paddle.framework._current_expected_place()
        )
        del full_1

        # pd_op.cast: (16x1x144x144xf32) <- (16x1x144x144xb)
        cast_0 = paddle._C_ops.cast(full_like_2, paddle.float32)
        del full_like_2

        # pd_op.cast: (16x1x144x144xf32) <- (16x1x144x144xb)
        cast_1 = paddle._C_ops.cast(data_2, paddle.float32)
        del data_2

        # pd_op.add: (16x8x144x144xf32) <- (16x8x144x144xf32, 16x8x144x144xf32)
        add_0 = paddle._C_ops.add(full_like_0, full_like_1)
        del full_like_0, full_like_1

        # pd_op.add: (16x8x144x144xf32) <- (16x8x144x144xf32, 16x1x144x144xf32)
        add_1 = paddle._C_ops.add(add_0, cast_0)
        del add_0, cast_0

        # pd_op.add: (16x8x144x144xf32) <- (16x8x144x144xf32, 16x8x144x144xf32)
        add_2 = paddle._C_ops.add(full_0, add_1)

        # pd_op.add: (16x8x144x144xf32) <- (16x8x144x144xf32, 16x8x144x144xf32)
        add_3 = paddle._C_ops.add(data_0, add_1)
        del data_0

        # pd_op.add: (16x8x144x144xf32) <- (16x1x144x144xf32, 16x8x144x144xf32)
        add_4 = paddle._C_ops.add(cast_1, add_1)
        del cast_1

        # pd_op.cast: (16x8x144x144xb) <- (16x8x144x144xf32)
        cast_2 = paddle._C_ops.cast(add_4, paddle.bool)
        del add_4

        # pd_op.where: (16x8x144x144xf32) <- (16x8x144x144xb, 16x8x144x144xf32, 16x8x144x144xf32)
        where_0 = paddle._C_ops.where(cast_2, add_2, add_3)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x8x144x144xf32) <- (16x8x144x144xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(where_0, full_2, float("0"), True)
        del where_0

        # pd_op.softmax: (16x8x144x144xf32) <- (16x8x144x144xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (16x8x144x144xf32, 16x8x144x144xui8) <- (16x8x144x144xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_3, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # builtin.combine: ([16x8x144x144xf32, 16x144x8x64xf32]) <- (16x8x144x144xf32, 16x144x8x64xf32)
        combine_0 = [dropout_0, data_1]
        del data_1, dropout_0

        # pd_op.einsum: (16x144x8x64xf32, [0xf32, 0xf32], [16x8x144x144xf32, 16x144x8x64xf32]) <- ([16x8x144x144xf32, 16x144x8x64xf32])
        einsum_0, einsum_1, einsum_2 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_0, "bhls,bshd->blhd"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_0

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_0,
            split_1,
        ) = einsum_1
        del einsum_1

        # builtin.split: (16x8x144x144xf32, 16x144x8x64xf32) <- ([16x8x144x144xf32, 16x144x8x64xf32])
        (
            split_2,
            split_3,
        ) = einsum_2
        del (
            add_1,
            add_2,
            add_3,
            cast_2,
            dropout_1,
            einsum_2,
            full_0,
            full_2,
            full_3,
            softmax_0,
        )

        return einsum_0
