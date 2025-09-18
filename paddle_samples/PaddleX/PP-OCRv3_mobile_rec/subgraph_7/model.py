import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [data_1, data_2, full_0]
        del full_0

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x-1x-1xf32) <- (8x26x1x40x1xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(data_0, stack_0)
        del data_0, stack_0

        # pd_op.softmax: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_0, -1)
        del reshape_0

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_1 = [data_1, data_2, data_3, data_4, data_5]
        del data_2, data_3, data_4, data_5

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1xf32, 5xi64)
        reshape_1 = paddle._C_ops.reshape(softmax_0, stack_1)
        del stack_1

        # pd_op.transpose: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_1, [0, 1, 4, 2, 3])
        del reshape_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (8x1x512x1x40xf32) <- (8x512x1x40xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_6, full_int_array_0)
        del data_6

        # pd_op.multiply: (8x-1x512x-1x40xf32) <- (8x1x512x1x40xf32, -1x-1x-1x-1x-1xf32)
        multiply_0 = paddle._C_ops.multiply(unsqueeze_0, transpose_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [3, 4]

        # pd_op.sum: (8x-1x512xf32) <- (8x-1x512x-1x40xf32, 2xi64)
        sum_0 = paddle._C_ops.sum(multiply_0, full_int_array_1, None, False)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("26"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("512"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_2 = [data_1, full_1, full_2]
        del data_1, full_1, full_2

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.expand: (-1x26x512xf32) <- (8x1x512xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_7, stack_2)
        del data_7

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([8x26x512xf32, 8x-1x512xf32, -1x26x512xf32]) <- (8x26x512xf32, 8x-1x512xf32, -1x26x512xf32)
        combine_3 = [data_8, sum_0, expand_0]
        del data_8

        # pd_op.concat: (8x26x1536xf32) <- ([8x26x512xf32, 8x-1x512xf32, -1x26x512xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_3, full_3)
        del combine_3

        # pd_op.matmul: (8x26x6626xf32) <- (8x26x1536xf32, 1536x6626xf32)
        matmul_0 = paddle._C_ops.matmul(concat_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (8x26x6626xf32) <- (8x26x6626xf32, 6626xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del parameter_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (8x26x6626xf32, 8x26x6626xui8) <- (8x26x6626xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_0, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del (
            add_0,
            concat_0,
            expand_0,
            full_3,
            full_4,
            full_int_array_0,
            full_int_array_1,
            matmul_0,
            multiply_0,
            softmax_0,
            stack_2,
            sum_0,
            transpose_0,
            unsqueeze_0,
        )

        return dropout_0
