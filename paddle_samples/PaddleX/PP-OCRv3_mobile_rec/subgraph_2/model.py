import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, data_0, data_1):
        # pd_op.embedding: (8x25x512xf32) <- (8x25xi64, 6627x512xf32)
        embedding_0 = paddle._C_ops.embedding(data_1, parameter_0, 6626, False)
        del data_1, parameter_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (8x1x512xf32) <- (8x512xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)
        del data_0

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([8x1x512xf32, 8x25x512xf32]) <- (8x1x512xf32, 8x25x512xf32)
        combine_0 = [unsqueeze_0, embedding_0]

        # pd_op.concat: (8x26x512xf32) <- ([8x1x512xf32, 8x25x512xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, embedding_0, full_0, full_int_array_0, unsqueeze_0

        return concat_0
