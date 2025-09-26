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
        data_12,
        data_13,
        data_14,
        data_15,
        data_16,
        data_17,
        data_18,
        data_19,
        data_20,
        data_21,
        data_22,
        data_23,
        data_24,
        data_25,
        data_26,
        data_27,
        data_28,
        data_29,
        data_30,
        data_31,
        data_32,
        data_33,
        data_34,
    ):
        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        combine_0 = [data_0, data_1, data_2, data_3, data_4, data_5, data_6]
        del data_0, data_1, data_2, data_3, data_4, data_5, data_6

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        add_n_0 = paddle._C_ops.add_n(combine_0)
        del combine_0

        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        combine_1 = [data_7, data_8, data_9, data_10, data_11, data_12, data_13]
        del data_10, data_11, data_12, data_13, data_7, data_8, data_9

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        add_n_1 = paddle._C_ops.add_n(combine_1)
        del combine_1

        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        combine_2 = [data_14, data_15, data_16, data_17, data_18, data_19, data_20]
        del data_14, data_15, data_16, data_17, data_18, data_19, data_20

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        add_n_2 = paddle._C_ops.add_n(combine_2)
        del combine_2

        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        combine_3 = [data_21, data_22, data_23, data_24, data_25, data_26, data_27]
        del data_21, data_22, data_23, data_24, data_25, data_26, data_27

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        add_n_3 = paddle._C_ops.add_n(combine_3)
        del combine_3

        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        combine_4 = [data_28, data_29, data_30, data_31, data_32, data_33, data_34]
        del data_28, data_29, data_30, data_31, data_32, data_33, data_34

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        add_n_4 = paddle._C_ops.add_n(combine_4)
        del combine_4

        return add_n_0, add_n_1, add_n_2, add_n_3, add_n_4
