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
    ):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("9"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (1x-1x9xf32, 1x-1x9xi64) <- (1x-1x-1xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_7, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_7

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x9xi64) <- (1x-1x9xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(topk_1, full_1, float("0"), True)
        del full_1

        # pd_op.one_hot: (1x-1x9x-1xf32) <- (1x-1x9xi64, xi64)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(data_3, topk_1.dtype), data_3
        )
        del data_3, topk_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-2]

        # pd_op.sum: (1x-1x-1xf32) <- (1x-1x9x-1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(one_hot_0, full_int_array_0, None, False)
        del one_hot_0

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x1xf32)
        multiply_0 = paddle._C_ops.multiply(sum_0, data_11)
        del sum_0

        # pd_op.topk: (1x-1x9xf32, 1x-1x9xi64) <- (1x-1x-1xf32, 1xi32)
        topk_2, topk_3 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_8, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_8

        # pd_op.add: (1x-1x9xi64) <- (1x-1x9xi64, xi64)
        add_0 = paddle._C_ops.add(topk_3, data_0)
        del data_0

        # pd_op.one_hot: (1x-1x9x-1xf32) <- (1x-1x9xi64, xi64)
        one_hot_1 = paddle._C_ops.one_hot(
            topk_3 % paddle.cast(data_4, topk_3.dtype), data_4
        )
        del data_4, topk_3

        # pd_op.sum: (1x-1x-1xf32) <- (1x-1x9x-1xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(one_hot_1, full_int_array_0, None, False)
        del one_hot_1

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x1xf32)
        multiply_1 = paddle._C_ops.multiply(sum_1, data_11)
        del sum_1

        # pd_op.topk: (1x-1x9xf32, 1x-1x9xi64) <- (1x-1x-1xf32, 1xi32)
        topk_4, topk_5 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_9, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_9

        # pd_op.add: (1x-1x9xi64) <- (1x-1x9xi64, xi64)
        add_1 = paddle._C_ops.add(topk_5, data_1)
        del data_1

        # pd_op.one_hot: (1x-1x9x-1xf32) <- (1x-1x9xi64, xi64)
        one_hot_2 = paddle._C_ops.one_hot(
            topk_5 % paddle.cast(data_5, topk_5.dtype), data_5
        )
        del data_5, topk_5

        # pd_op.sum: (1x-1x-1xf32) <- (1x-1x9x-1xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(one_hot_2, full_int_array_0, None, False)
        del one_hot_2

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x1xf32)
        multiply_2 = paddle._C_ops.multiply(sum_2, data_11)
        del sum_2

        # pd_op.topk: (1x-1x9xf32, 1x-1x9xi64) <- (1x-1x-1xf32, 1xi32)
        topk_6, topk_7 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_10, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_10, full_0

        # pd_op.add: (1x-1x9xi64) <- (1x-1x9xi64, xi64)
        add_2 = paddle._C_ops.add(topk_7, data_2)
        del data_2

        # pd_op.one_hot: (1x-1x9x-1xf32) <- (1x-1x9xi64, xi64)
        one_hot_3 = paddle._C_ops.one_hot(
            topk_7 % paddle.cast(data_6, topk_7.dtype), data_6
        )
        del data_6, topk_7

        # pd_op.sum: (1x-1x-1xf32) <- (1x-1x9x-1xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(one_hot_3, full_int_array_0, None, False)
        del full_int_array_0, one_hot_3

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x1xf32)
        multiply_3 = paddle._C_ops.multiply(sum_3, data_11)
        del data_11, sum_3

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x-1x-1xf32, 1x-1x-1xf32, 1x-1x-1xf32, 1x-1x-1xf32]) <- (1x-1x-1xf32, 1x-1x-1xf32, 1x-1x-1xf32, 1x-1x-1xf32)
        combine_0 = [multiply_0, multiply_1, multiply_2, multiply_3]
        del multiply_0, multiply_1, multiply_2, multiply_3

        # pd_op.concat: (1x-1x-1xf32) <- ([1x-1x-1xf32, 1x-1x-1xf32, 1x-1x-1xf32, 1x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0

        # builtin.combine: ([1x-1x9xi64, 1x-1x9xi64, 1x-1x9xi64, 1x-1x9xi64]) <- (1x-1x9xi64, 1x-1x9xi64, 1x-1x9xi64, 1x-1x9xi64)
        combine_1 = [scale_0, add_0, add_1, add_2]
        del add_0, add_1, add_2, scale_0

        # pd_op.concat: (1x-1x36xi64) <- ([1x-1x9xi64, 1x-1x9xi64, 1x-1x9xi64, 1x-1x9xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)
        del combine_1, full_2

        return concat_0, concat_1
