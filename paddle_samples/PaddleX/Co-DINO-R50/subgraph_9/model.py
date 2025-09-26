import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_0, 4, full_0)
        del data_0, full_0

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.add: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        add_0 = paddle._C_ops.add(split_0, split_2)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_1, float("0"), True)
        del add_0

        # pd_op.add: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        add_1 = paddle._C_ops.add(split_1, split_3)

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_1, float("0"), True)
        del add_1, full_1

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_0 = paddle._C_ops.subtract(split_2, split_0)
        del split_0, split_2

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_3, split_1)
        del split_1, split_3

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32)
        combine_0 = [scale_0, scale_1, subtract_0, subtract_1]
        del scale_0, scale_1, subtract_0, subtract_1

        # pd_op.concat: (-1x4xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0, full_2

        return concat_0
