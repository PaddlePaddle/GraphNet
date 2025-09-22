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
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([34240x4xf32, 8560x4xf32, 2160x4xf32, 540x4xf32, 140x4xf32, 35x4xf32]) <- (34240x4xf32, 8560x4xf32, 2160x4xf32, 540x4xf32, 140x4xf32, 35x4xf32)
        combine_0 = [data_0, data_1, data_2, data_3, data_4, data_5]
        del data_0, data_1, data_2, data_3, data_4, data_5

        # pd_op.concat: (45675x4xf32) <- ([34240x4xf32, 8560x4xf32, 2160x4xf32, 540x4xf32, 140x4xf32, 35x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # builtin.combine: ([34240xi32, 8560xi32, 2160xi32, 540xi32, 140xi32, 35xi32]) <- (34240xi32, 8560xi32, 2160xi32, 540xi32, 140xi32, 35xi32)
        combine_1 = [data_6, data_7, data_8, data_9, data_10, data_11]
        del data_10, data_11, data_6, data_7, data_8, data_9

        # pd_op.concat: (45675xi32) <- ([34240xi32, 8560xi32, 2160xi32, 540xi32, 140xi32, 35xi32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1, full_0

        return concat_0, concat_1
