import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([14x1xi64]) <- (14x1xi64)
        combine_0 = [data_0]

        # pd_op.concat: (14x1xi64) <- ([14x1xi64], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([1xi64]) <- (1xi64)
        combine_1 = [data_1]

        # pd_op.concat: (1xi64) <- ([1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_0)

        # builtin.combine: ([14xi32]) <- (14xi32)
        combine_2 = [data_2]

        # pd_op.concat: (14xi32) <- ([14xi32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_0)

        # builtin.combine: ([14x28x28xi32]) <- (14x28x28xi32)
        combine_3 = [data_3]

        # pd_op.concat: (14x28x28xi32) <- ([14x28x28xi32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_3, full_0)

        # builtin.combine: ([14xf32]) <- (14xf32)
        combine_4 = [data_4]

        # pd_op.concat: (14xf32) <- ([14xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)
        return concat_0, concat_1, concat_2, concat_3, concat_4
