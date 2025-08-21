import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        # pd_op.shape64: (3xi64) <- (4x1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(data_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("9"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (4x1x9xf32, 4x1x9xi64) <- (4x1x-1xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_3, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x1x9xi64) <- (4x1x9xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(topk_1, full_1, float("0"), True)

        # pd_op.one_hot: (4x1x9x-1xf32) <- (4x1x9xi64, xi64)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(slice_0, topk_1.dtype), slice_0
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-2]

        # pd_op.sum: (4x1x-1xf32) <- (4x1x9x-1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(one_hot_0, full_int_array_2, None, False)

        # pd_op.multiply: (4x1x-1xf32) <- (4x1x-1xf32, 4x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(sum_0, data_7)

        # pd_op.shape64: (3xi64) <- (4x1x-1xf32)
        shape64_1 = paddle._C_ops.shape64(data_4)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.topk: (4x1x9xf32, 4x1x9xi64) <- (4x1x-1xf32, 1xi32)
        topk_2, topk_3 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_4, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (4x1x9xi64) <- (4x1x9xi64, xi64)
        add_0 = paddle._C_ops.add(topk_3, data_0)

        # pd_op.one_hot: (4x1x9x-1xf32) <- (4x1x9xi64, xi64)
        one_hot_1 = paddle._C_ops.one_hot(
            topk_3 % paddle.cast(slice_1, topk_3.dtype), slice_1
        )

        # pd_op.sum: (4x1x-1xf32) <- (4x1x9x-1xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(one_hot_1, full_int_array_2, None, False)

        # pd_op.multiply: (4x1x-1xf32) <- (4x1x-1xf32, 4x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(sum_1, data_7)

        # pd_op.shape64: (3xi64) <- (4x1x-1xf32)
        shape64_2 = paddle._C_ops.shape64(data_5)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.topk: (4x1x9xf32, 4x1x9xi64) <- (4x1x-1xf32, 1xi32)
        topk_4, topk_5 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_5, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (4x1x9xi64) <- (4x1x9xi64, xi64)
        add_1 = paddle._C_ops.add(topk_5, data_1)

        # pd_op.one_hot: (4x1x9x-1xf32) <- (4x1x9xi64, xi64)
        one_hot_2 = paddle._C_ops.one_hot(
            topk_5 % paddle.cast(slice_2, topk_5.dtype), slice_2
        )

        # pd_op.sum: (4x1x-1xf32) <- (4x1x9x-1xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(one_hot_2, full_int_array_2, None, False)

        # pd_op.multiply: (4x1x-1xf32) <- (4x1x-1xf32, 4x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(sum_2, data_7)

        # pd_op.topk: (4x1x9xf32, 4x1x9xi64) <- (4x1x36xf32, 1xi32)
        topk_6, topk_7 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_6, full_0, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (4x1x9xi64) <- (4x1x9xi64, xi64)
        add_2 = paddle._C_ops.add(topk_7, data_2)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("36"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (4x1x9x36xf32) <- (4x1x9xi64, 1xi32)
        one_hot_3 = paddle._C_ops.one_hot(
            topk_7 % paddle.cast(full_2, topk_7.dtype), full_2
        )

        # pd_op.sum: (4x1x36xf32) <- (4x1x9x36xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(one_hot_3, full_int_array_2, None, False)

        # pd_op.multiply: (4x1x36xf32) <- (4x1x36xf32, 4x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(sum_3, data_7)

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([4x1x-1xf32, 4x1x-1xf32, 4x1x-1xf32, 4x1x36xf32]) <- (4x1x-1xf32, 4x1x-1xf32, 4x1x-1xf32, 4x1x36xf32)
        combine_0 = [multiply_0, multiply_1, multiply_2, multiply_3]

        # pd_op.concat: (4x1x-1xf32) <- ([4x1x-1xf32, 4x1x-1xf32, 4x1x-1xf32, 4x1x36xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_3)

        # builtin.combine: ([4x1x9xi64, 4x1x9xi64, 4x1x9xi64, 4x1x9xi64]) <- (4x1x9xi64, 4x1x9xi64, 4x1x9xi64, 4x1x9xi64)
        combine_1 = [scale_0, add_0, add_1, add_2]

        # pd_op.concat: (4x1x36xi64) <- ([4x1x9xi64, 4x1x9xi64, 4x1x9xi64, 4x1x9xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_3)
        return concat_0, concat_1
