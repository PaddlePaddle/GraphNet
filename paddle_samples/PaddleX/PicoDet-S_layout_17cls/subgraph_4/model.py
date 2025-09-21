import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("9"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (2x2x9xf32, 2x2x9xi64) <- (2x2x4624xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_0, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x2x9xi64) <- (2x2x9xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(topk_1, full_1, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("4624"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x2x9x4624xf32) <- (2x2x9xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(full_2, topk_1.dtype), full_2
        )
        del full_2, topk_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-2]

        # pd_op.sum: (2x2x4624xf32) <- (2x2x9x4624xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(one_hot_0, full_int_array_0, None, False)
        del one_hot_0

        # pd_op.multiply: (2x2x4624xf32) <- (2x2x4624xf32, 2x2x1xf32)
        multiply_0 = paddle._C_ops.multiply(sum_0, data_4)
        del sum_0

        # pd_op.topk: (2x2x9xf32, 2x2x9xi64) <- (2x2x1156xf32, 1xi32)
        topk_2, topk_3 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_1, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_1

        # pd_op.scale: (2x2x9xi64) <- (2x2x9xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(topk_3, full_1, float("4624"), True)

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1156"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x2x9x1156xf32) <- (2x2x9xi64, 1xi32)
        one_hot_1 = paddle._C_ops.one_hot(
            topk_3 % paddle.cast(full_3, topk_3.dtype), full_3
        )
        del full_3, topk_3

        # pd_op.sum: (2x2x1156xf32) <- (2x2x9x1156xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(one_hot_1, full_int_array_0, None, False)
        del one_hot_1

        # pd_op.multiply: (2x2x1156xf32) <- (2x2x1156xf32, 2x2x1xf32)
        multiply_1 = paddle._C_ops.multiply(sum_1, data_4)
        del sum_1

        # pd_op.topk: (2x2x9xf32, 2x2x9xi64) <- (2x2x289xf32, 1xi32)
        topk_4, topk_5 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_2, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_2

        # pd_op.scale: (2x2x9xi64) <- (2x2x9xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(topk_5, full_1, float("5780"), True)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("289"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x2x9x289xf32) <- (2x2x9xi64, 1xi32)
        one_hot_2 = paddle._C_ops.one_hot(
            topk_5 % paddle.cast(full_4, topk_5.dtype), full_4
        )
        del full_4, topk_5

        # pd_op.sum: (2x2x289xf32) <- (2x2x9x289xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(one_hot_2, full_int_array_0, None, False)
        del one_hot_2

        # pd_op.multiply: (2x2x289xf32) <- (2x2x289xf32, 2x2x1xf32)
        multiply_2 = paddle._C_ops.multiply(sum_2, data_4)
        del sum_2

        # pd_op.topk: (2x2x9xf32, 2x2x9xi64) <- (2x2x81xf32, 1xi32)
        topk_6, topk_7 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_3, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_3, full_0

        # pd_op.scale: (2x2x9xi64) <- (2x2x9xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(topk_7, full_1, float("6069"), True)
        del full_1

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("81"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x2x9x81xf32) <- (2x2x9xi64, 1xi32)
        one_hot_3 = paddle._C_ops.one_hot(
            topk_7 % paddle.cast(full_5, topk_7.dtype), full_5
        )
        del full_5, topk_7

        # pd_op.sum: (2x2x81xf32) <- (2x2x9x81xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(one_hot_3, full_int_array_0, None, False)
        del full_int_array_0, one_hot_3

        # pd_op.multiply: (2x2x81xf32) <- (2x2x81xf32, 2x2x1xf32)
        multiply_3 = paddle._C_ops.multiply(sum_3, data_4)
        del data_4, sum_3

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x2x4624xf32, 2x2x1156xf32, 2x2x289xf32, 2x2x81xf32]) <- (2x2x4624xf32, 2x2x1156xf32, 2x2x289xf32, 2x2x81xf32)
        combine_0 = [multiply_0, multiply_1, multiply_2, multiply_3]
        del multiply_0, multiply_1, multiply_2, multiply_3

        # pd_op.concat: (2x2x6150xf32) <- ([2x2x4624xf32, 2x2x1156xf32, 2x2x289xf32, 2x2x81xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_6)
        del combine_0

        # builtin.combine: ([2x2x9xi64, 2x2x9xi64, 2x2x9xi64, 2x2x9xi64]) <- (2x2x9xi64, 2x2x9xi64, 2x2x9xi64, 2x2x9xi64)
        combine_1 = [scale_0, scale_1, scale_2, scale_3]
        del scale_0, scale_1, scale_2, scale_3

        # pd_op.concat: (2x2x36xi64) <- ([2x2x9xi64, 2x2x9xi64, 2x2x9xi64, 2x2x9xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_6)
        del combine_1, full_6

        return concat_0, concat_1
