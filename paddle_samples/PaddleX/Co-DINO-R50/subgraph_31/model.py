import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [0, -1, 4]

        # pd_op.reshape: (8x1x4xf32) <- (8x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(data_0, full_int_array_0)
        del data_0, full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (8x1x1xf32) <- (8x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            reshape_0, [2], full_int_array_1, full_int_array_2, [1], []
        )
        del full_int_array_1

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x1x1xf32) <- (8x1x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float("0"), True)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (8x1x1xf32) <- (8x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            reshape_0, [2], full_int_array_2, full_int_array_3, [1], []
        )
        del full_int_array_2

        # pd_op.scale: (8x1x1xf32) <- (8x1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, full_0, float("0"), True)
        del full_0, slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (8x1x1xf32) <- (8x1x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            reshape_0, [2], full_int_array_3, full_int_array_4, [1], []
        )
        del full_int_array_3

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x1x1xf32) <- (8x1x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_2, full_1, float("0"), True)
        del slice_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        # pd_op.slice: (8x1x1xf32) <- (8x1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            reshape_0, [2], full_int_array_4, full_int_array_5, [1], []
        )
        del full_int_array_4, full_int_array_5, reshape_0

        # pd_op.scale: (8x1x1xf32) <- (8x1x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(slice_3, full_1, float("0"), True)
        del full_1, slice_3

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1e+10"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("4.13517"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (8x1x1xf32) <- (8x1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(scale_2, full_2, full_3)
        del scale_2

        # pd_op.clip: (8x1x1xf32) <- (8x1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(scale_3, full_2, full_3)
        del full_2, full_3, scale_3

        # pd_op.exp: (8x1x1xf32) <- (8x1x1xf32)
        exp_0 = paddle._C_ops.exp(clip_0)
        del clip_0

        # pd_op.exp: (8x1x1xf32) <- (8x1x1xf32)
        exp_1 = paddle._C_ops.exp(clip_1)
        del clip_1

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x1x1xf32) <- (8x1x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(exp_0, full_4, float("0"), True)
        del exp_0

        # pd_op.subtract: (8x1x1xf32) <- (8x1x1xf32, 8x1x1xf32)
        subtract_0 = paddle._C_ops.subtract(scale_0, scale_4)

        # pd_op.scale: (8x1x1xf32) <- (8x1x1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(exp_1, full_4, float("0"), True)
        del exp_1, full_4

        # pd_op.subtract: (8x1x1xf32) <- (8x1x1xf32, 8x1x1xf32)
        subtract_1 = paddle._C_ops.subtract(scale_1, scale_5)

        # pd_op.add: (8x1x1xf32) <- (8x1x1xf32, 8x1x1xf32)
        add_0 = paddle._C_ops.add(scale_0, scale_4)
        del scale_0, scale_4

        # pd_op.add: (8x1x1xf32) <- (8x1x1xf32, 8x1x1xf32)
        add_1 = paddle._C_ops.add(scale_1, scale_5)
        del scale_1, scale_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [-1]

        # pd_op.reshape: (8xf32) <- (8x1x1xf32, 1xi64)
        reshape_1 = paddle._C_ops.reshape(subtract_0, full_int_array_6)
        del subtract_0

        # pd_op.reshape: (8xf32) <- (8x1x1xf32, 1xi64)
        reshape_2 = paddle._C_ops.reshape(subtract_1, full_int_array_6)
        del subtract_1

        # pd_op.reshape: (8xf32) <- (8x1x1xf32, 1xi64)
        reshape_3 = paddle._C_ops.reshape(add_0, full_int_array_6)
        del add_0

        # pd_op.reshape: (8xf32) <- (8x1x1xf32, 1xi64)
        reshape_4 = paddle._C_ops.reshape(add_1, full_int_array_6)
        del add_1, full_int_array_6

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([8xf32, 8xf32, 8xf32, 8xf32]) <- (8xf32, 8xf32, 8xf32, 8xf32)
        combine_0 = [reshape_1, reshape_2, reshape_3, reshape_4]
        del reshape_1, reshape_2, reshape_3, reshape_4

        # pd_op.concat: (32xf32) <- ([8xf32, 8xf32, 8xf32, 8xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_5)
        del combine_0, full_5

        return concat_0
