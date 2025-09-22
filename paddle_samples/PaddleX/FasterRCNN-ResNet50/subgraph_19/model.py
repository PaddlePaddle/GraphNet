import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (1024xf32) <- (1024x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (1024xf32) <- (1024x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (1024xf32) <- (1024xf32, 1024xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (1024xf32) <- (1024x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (1024xf32) <- (1024x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_0

        # pd_op.subtract: (1024xf32) <- (1024xf32, 1024xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_2

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1024xf32) <- (1024xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_0, float("0"), True)

        # pd_op.add: (1024xf32) <- (1024xf32, 1024xf32)
        add_0 = paddle._C_ops.add(slice_1, scale_0)
        del scale_0, slice_1

        # pd_op.scale: (1024xf32) <- (1024xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_1, full_0, float("0"), True)

        # pd_op.add: (1024xf32) <- (1024xf32, 1024xf32)
        add_1 = paddle._C_ops.add(slice_3, scale_1)
        del scale_1, slice_3

        # pd_op.slice: (1024xf32) <- (1024x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (1024xf32) <- (1024x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del full_int_array_2

        # pd_op.subtract: (1024xf32) <- (1024xf32, 1024xf32)
        subtract_2 = paddle._C_ops.subtract(slice_4, slice_5)
        del slice_4

        # pd_op.slice: (1024xf32) <- (1024x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_4, [1], [1]
        )
        del full_int_array_1, full_int_array_4

        # pd_op.slice: (1024xf32) <- (1024x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_1, full_int_array_0, full_int_array_3

        # pd_op.subtract: (1024xf32) <- (1024xf32, 1024xf32)
        subtract_3 = paddle._C_ops.subtract(slice_6, slice_7)
        del slice_6

        # pd_op.scale: (1024xf32) <- (1024xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(subtract_2, full_0, float("0"), True)

        # pd_op.add: (1024xf32) <- (1024xf32, 1024xf32)
        add_2 = paddle._C_ops.add(slice_5, scale_2)
        del scale_2, slice_5

        # pd_op.scale: (1024xf32) <- (1024xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(subtract_3, full_0, float("0"), True)
        del full_0

        # pd_op.add: (1024xf32) <- (1024xf32, 1024xf32)
        add_3 = paddle._C_ops.add(slice_7, scale_3)
        del scale_3, slice_7

        # pd_op.subtract: (1024xf32) <- (1024xf32, 1024xf32)
        subtract_4 = paddle._C_ops.subtract(add_2, add_0)
        del add_0, add_2

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("10"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1024xf32) <- (1024xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(subtract_4, full_1, float("0"), True)
        del subtract_4

        # pd_op.divide: (1024xf32) <- (1024xf32, 1024xf32)
        divide_0 = paddle._C_ops.divide(scale_4, subtract_0)
        del scale_4

        # pd_op.subtract: (1024xf32) <- (1024xf32, 1024xf32)
        subtract_5 = paddle._C_ops.subtract(add_3, add_1)
        del add_1, add_3

        # pd_op.scale: (1024xf32) <- (1024xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(subtract_5, full_1, float("0"), True)
        del full_1, subtract_5

        # pd_op.divide: (1024xf32) <- (1024xf32, 1024xf32)
        divide_1 = paddle._C_ops.divide(scale_5, subtract_1)
        del scale_5

        # pd_op.divide: (1024xf32) <- (1024xf32, 1024xf32)
        divide_2 = paddle._C_ops.divide(subtract_2, subtract_0)
        del subtract_0, subtract_2

        # pd_op.log: (1024xf32) <- (1024xf32)
        log_0 = paddle._C_ops.log(divide_2)
        del divide_2

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1024xf32) <- (1024xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(log_0, full_2, float("0"), True)
        del log_0

        # pd_op.divide: (1024xf32) <- (1024xf32, 1024xf32)
        divide_3 = paddle._C_ops.divide(subtract_3, subtract_1)
        del subtract_1, subtract_3

        # pd_op.log: (1024xf32) <- (1024xf32)
        log_1 = paddle._C_ops.log(divide_3)
        del divide_3

        # pd_op.scale: (1024xf32) <- (1024xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(log_1, full_2, float("0"), True)
        del full_2, log_1

        # builtin.combine: ([1024xf32, 1024xf32, 1024xf32, 1024xf32]) <- (1024xf32, 1024xf32, 1024xf32, 1024xf32)
        combine_0 = [divide_0, divide_1, scale_6, scale_7]
        del divide_0, divide_1, scale_6, scale_7

        # pd_op.stack: (1024x4xf32) <- ([1024xf32, 1024xf32, 1024xf32, 1024xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)
        del combine_0

        return stack_0
