import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x21504x2xf32, 1x21504x2xf32]) <- (1x21504x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_1, 2, full_0)
        del data_1

        # builtin.split: (1x21504x2xf32, 1x21504x2xf32) <- ([1x21504x2xf32, 1x21504x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.multiply: (1x21504x2xf32) <- (1x21504x2xf32, 1x21504x1xf32)
        multiply_0 = paddle._C_ops.multiply(split_0, data_4)

        # pd_op.add: (1x21504x2xf32) <- (1x21504x2xf32, 1x21504x2xf32)
        add_0 = paddle._C_ops.add(multiply_0, data_3)
        del data_3

        # pd_op.elu: (1x21504x2xf32) <- (1x21504x2xf32)
        elu_0 = paddle._C_ops.elu(split_1, float("1"))

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21504x2xf32) <- (1x21504x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(elu_0, full_1, float("1"), True)

        # pd_op.multiply: (1x21504x2xf32) <- (1x21504x2xf32, 1x21504x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_0, data_4)
        del data_4

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 21504, 1, 91]

        # pd_op.reshape: (1x21504x1x91xf32) <- (1x21504x91xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(data_2, full_int_array_0)
        del data_2, full_int_array_0

        # pd_op.softmax: (1x21504x1x91xf32) <- (1x21504x1x91xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_0, -1)
        del reshape_0

        # pd_op.matmul: (1x21504x1xf32) <- (1x21504x1x91xf32, 91xf32)
        matmul_0 = paddle._C_ops.matmul(softmax_0, data_5, False, False)
        del data_5

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x21504x2xf32, 1x21504x2xf32, 1x21504x1xf32]) <- (1x21504x2xf32, 1x21504x2xf32, 1x21504x1xf32)
        combine_0 = [add_0, multiply_1, matmul_0]

        # pd_op.concat: (1x21504x5xf32) <- ([1x21504x2xf32, 1x21504x2xf32, 1x21504x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0

        # pd_op.share_data_: (1x21504x15xf32) <- (1x21504x15xf32)
        share_data__0 = data_0.detach()
        del data_0

        # pd_op.share_data_: (1x21504x5xf32) <- (1x21504x5xf32)
        share_data__1 = concat_0.detach()
        del (
            add_0,
            concat_0,
            elu_0,
            full_0,
            full_1,
            full_2,
            matmul_0,
            multiply_0,
            multiply_1,
            scale_0,
            softmax_0,
            split_0,
            split_1,
        )

        return share_data__0, share_data__1
