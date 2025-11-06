import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([-1x-1x2xf32, -1x-1x2xf32]) <- (-1x-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_0, 2, full_0)
        del data_0, full_0

        # builtin.split: (-1x-1x2xf32, -1x-1x2xf32) <- ([-1x-1x2xf32, -1x-1x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1x2xf32) <- (-1x-1x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(split_1, full_1, float("0"), True)
        del full_1, split_1

        # pd_op.subtract: (-1x-1x2xf32) <- (-1x-1x2xf32, -1x-1x2xf32)
        subtract_0 = paddle._C_ops.subtract(split_0, scale_0)

        # pd_op.add: (-1x-1x2xf32) <- (-1x-1x2xf32, -1x-1x2xf32)
        add_0 = paddle._C_ops.add(split_0, scale_0)
        del scale_0, split_0

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x-1x2xf32, -1x-1x2xf32]) <- (-1x-1x2xf32, -1x-1x2xf32)
        combine_0 = [subtract_0, add_0]
        del add_0, subtract_0

        # pd_op.concat: (-1x-1x4xf32) <- ([-1x-1x2xf32, -1x-1x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0, full_2

        return concat_0
