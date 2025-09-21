import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("9"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (4x1x9xf32, 4x1x9xi64) <- (4x1x441xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_0, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x1x9xi64) <- (4x1x9xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(topk_1, full_1, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("441"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (4x1x9x441xf32) <- (4x1x9xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(full_2, topk_1.dtype), full_2
        )
        del full_2, topk_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-2]

        # pd_op.sum: (4x1x441xf32) <- (4x1x9x441xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(one_hot_0, full_int_array_0, None, False)
        del one_hot_0

        # pd_op.multiply: (4x1x441xf32) <- (4x1x441xf32, 4x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(sum_0, data_3)
        del sum_0

        # pd_op.topk: (4x1x9xf32, 4x1x9xi64) <- (4x1x1764xf32, 1xi32)
        topk_2, topk_3 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_1, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_1

        # pd_op.scale: (4x1x9xi64) <- (4x1x9xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(topk_3, full_1, float("441"), True)

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1764"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (4x1x9x1764xf32) <- (4x1x9xi64, 1xi32)
        one_hot_1 = paddle._C_ops.one_hot(
            topk_3 % paddle.cast(full_3, topk_3.dtype), full_3
        )
        del full_3, topk_3

        # pd_op.sum: (4x1x1764xf32) <- (4x1x9x1764xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(one_hot_1, full_int_array_0, None, False)
        del one_hot_1

        # pd_op.multiply: (4x1x1764xf32) <- (4x1x1764xf32, 4x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(sum_1, data_3)
        del sum_1

        # pd_op.topk: (4x1x9xf32, 4x1x9xi64) <- (4x1x7056xf32, 1xi32)
        topk_4, topk_5 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_2, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_2, full_0

        # pd_op.scale: (4x1x9xi64) <- (4x1x9xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(topk_5, full_1, float("2205"), True)
        del full_1

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("7056"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (4x1x9x7056xf32) <- (4x1x9xi64, 1xi32)
        one_hot_2 = paddle._C_ops.one_hot(
            topk_5 % paddle.cast(full_4, topk_5.dtype), full_4
        )
        del full_4, topk_5

        # pd_op.sum: (4x1x7056xf32) <- (4x1x9x7056xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(one_hot_2, full_int_array_0, None, False)
        del full_int_array_0, one_hot_2

        # pd_op.multiply: (4x1x7056xf32) <- (4x1x7056xf32, 4x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(sum_2, data_3)
        del data_3, sum_2

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([4x1x441xf32, 4x1x1764xf32, 4x1x7056xf32]) <- (4x1x441xf32, 4x1x1764xf32, 4x1x7056xf32)
        combine_0 = [multiply_0, multiply_1, multiply_2]
        del multiply_0, multiply_1, multiply_2

        # pd_op.concat: (4x1x9261xf32) <- ([4x1x441xf32, 4x1x1764xf32, 4x1x7056xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_5)
        del combine_0

        # builtin.combine: ([4x1x9xi64, 4x1x9xi64, 4x1x9xi64]) <- (4x1x9xi64, 4x1x9xi64, 4x1x9xi64)
        combine_1 = [scale_0, scale_1, scale_2]
        del scale_0, scale_1, scale_2

        # pd_op.concat: (4x1x27xi64) <- ([4x1x9xi64, 4x1x9xi64, 4x1x9xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_5)
        del combine_1, full_5

        return concat_0, concat_1
