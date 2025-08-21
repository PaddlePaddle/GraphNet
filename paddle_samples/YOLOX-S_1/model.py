import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.multiply: (8x6804x4xf32) <- (8x6804x4xf32, 8x6804x1xf32)
        multiply_1 = paddle._C_ops.multiply(data_0, data_2)

        # pd_op.sqrt: (8x6804x4xf32) <- (8x6804x4xf32)
        sqrt_0 = paddle._C_ops.sqrt(multiply_1)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([6804x2xf32, 6804x1xf32, 6804x1xf32]) <- (6804x2xf32, 6804x1xf32, 6804x1xf32)
        combine_0 = [data_3, data_4, data_4]

        # pd_op.concat: (6804x4xf32) <- ([6804x2xf32, 6804x1xf32, 6804x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.share_data_: (8x6804x4xf32) <- (8x6804x4xf32)
        share_data__0 = sqrt_0.detach()

        # pd_op.share_data_: (8x6804x4xf32) <- (8x6804x4xf32)
        share_data__1 = data_1.detach()

        # pd_op.multiply: (8x6804x4xf32) <- (8x6804x4xf32, 6804x1xf32)
        multiply_0 = paddle._C_ops.multiply(share_data__1, data_4)
        return share_data__0, multiply_0, concat_0
