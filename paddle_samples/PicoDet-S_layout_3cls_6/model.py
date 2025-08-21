import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("9"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (2x8x9xf32, 2x8x9xi64) <- (2x8x4096xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_0, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x8x9xi64) <- (2x8x9xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(topk_1, full_1, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("4096"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x8x9x4096xf32) <- (2x8x9xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(full_2, topk_1.dtype), full_2
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-2]

        # pd_op.sum: (2x8x4096xf32) <- (2x8x9x4096xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(one_hot_0, full_int_array_0, None, False)

        # pd_op.multiply: (2x8x4096xf32) <- (2x8x4096xf32, 2x8x1xf32)
        multiply_0 = paddle._C_ops.multiply(sum_0, data_4)

        # pd_op.topk: (2x8x9xf32, 2x8x9xi64) <- (2x8x1024xf32, 1xi32)
        topk_2, topk_3 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_1, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.scale: (2x8x9xi64) <- (2x8x9xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(topk_3, full_1, float("4096"), True)

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1024"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x8x9x1024xf32) <- (2x8x9xi64, 1xi32)
        one_hot_1 = paddle._C_ops.one_hot(
            topk_3 % paddle.cast(full_3, topk_3.dtype), full_3
        )

        # pd_op.sum: (2x8x1024xf32) <- (2x8x9x1024xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(one_hot_1, full_int_array_0, None, False)

        # pd_op.multiply: (2x8x1024xf32) <- (2x8x1024xf32, 2x8x1xf32)
        multiply_1 = paddle._C_ops.multiply(sum_1, data_4)

        # pd_op.topk: (2x8x9xf32, 2x8x9xi64) <- (2x8x256xf32, 1xi32)
        topk_4, topk_5 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_2, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.scale: (2x8x9xi64) <- (2x8x9xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(topk_5, full_1, float("5120"), True)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("256"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x8x9x256xf32) <- (2x8x9xi64, 1xi32)
        one_hot_2 = paddle._C_ops.one_hot(
            topk_5 % paddle.cast(full_4, topk_5.dtype), full_4
        )

        # pd_op.sum: (2x8x256xf32) <- (2x8x9x256xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(one_hot_2, full_int_array_0, None, False)

        # pd_op.multiply: (2x8x256xf32) <- (2x8x256xf32, 2x8x1xf32)
        multiply_2 = paddle._C_ops.multiply(sum_2, data_4)

        # pd_op.topk: (2x8x9xf32, 2x8x9xi64) <- (2x8x64xf32, 1xi32)
        topk_6, topk_7 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_3, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.scale: (2x8x9xi64) <- (2x8x9xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(topk_7, full_1, float("5376"), True)

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("64"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x8x9x64xf32) <- (2x8x9xi64, 1xi32)
        one_hot_3 = paddle._C_ops.one_hot(
            topk_7 % paddle.cast(full_5, topk_7.dtype), full_5
        )

        # pd_op.sum: (2x8x64xf32) <- (2x8x9x64xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(one_hot_3, full_int_array_0, None, False)

        # pd_op.multiply: (2x8x64xf32) <- (2x8x64xf32, 2x8x1xf32)
        multiply_3 = paddle._C_ops.multiply(sum_3, data_4)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x8x4096xf32, 2x8x1024xf32, 2x8x256xf32, 2x8x64xf32]) <- (2x8x4096xf32, 2x8x1024xf32, 2x8x256xf32, 2x8x64xf32)
        combine_0 = [multiply_0, multiply_1, multiply_2, multiply_3]

        # pd_op.concat: (2x8x5440xf32) <- ([2x8x4096xf32, 2x8x1024xf32, 2x8x256xf32, 2x8x64xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_6)

        # builtin.combine: ([2x8x9xi64, 2x8x9xi64, 2x8x9xi64, 2x8x9xi64]) <- (2x8x9xi64, 2x8x9xi64, 2x8x9xi64, 2x8x9xi64)
        combine_1 = [scale_0, scale_1, scale_2, scale_3]

        # pd_op.concat: (2x8x36xi64) <- ([2x8x9xi64, 2x8x9xi64, 2x8x9xi64, 2x8x9xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_6)
        return concat_0, concat_1
