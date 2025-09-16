import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32)
        combine_0 = [data_0, data_1, data_2, data_3]
        del data_0, data_1, data_2, data_3

        # pd_op.stack: (4x1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.expand_as: (4x22400xf32) <- (4x1xf32, 4x22400xi64)
        expand_as_0 = paddle._C_ops.expand_as(stack_0, data_4, [4, 22400])
        del stack_0

        # pd_op.cast: (4x22400xf32) <- (4x22400xi64)
        cast_1 = paddle._C_ops.cast(data_4, paddle.float32)
        del data_4

        # pd_op.less_than: (4x22400xb) <- (4x22400xf32, 4x22400xf32)
        less_than_0 = paddle._C_ops.less_than(cast_1, expand_as_0)
        del cast_1, expand_as_0

        # pd_op.cast: (4x22400xf32) <- (4x22400xb)
        cast_2 = paddle._C_ops.cast(less_than_0, paddle.float32)
        del less_than_0

        # pd_op.add: (4x22400xf32) <- (4x22400xf32, 4x22400xf32)
        add_0 = paddle._C_ops.add(cast_2, data_5)
        del cast_2, data_5

        # pd_op.cast: (4x22400xb) <- (4x22400xf32)
        cast_0 = paddle._C_ops.cast(add_0, paddle.bool)
        del add_0

        return cast_0
