import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_3

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.assign: (2048xf32) <- (2048xf32)
        assign_7 = slice_1

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_0

        # pd_op.assign: (2048xf32) <- (2048xf32)
        assign_8 = slice_3

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_0

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_0, float("0"), True)

        # pd_op.add: (2048xf32) <- (2048xf32, 2048xf32)
        add_0 = paddle._C_ops.add(slice_1, scale_0)

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_1, full_0, float("0"), True)

        # pd_op.add: (2048xf32) <- (2048xf32, 2048xf32)
        add_1 = paddle._C_ops.add(slice_3, scale_1)

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_2 = paddle._C_ops.subtract(slice_4, slice_5)
        del slice_4

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_1

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_3 = paddle._C_ops.subtract(slice_6, slice_7)
        del slice_6

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(subtract_2, full_0, float("0"), True)

        # pd_op.add: (2048xf32) <- (2048xf32, 2048xf32)
        add_2 = paddle._C_ops.add(slice_5, scale_2)
        del scale_2, slice_5

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(subtract_3, full_0, float("0"), True)

        # pd_op.add: (2048xf32) <- (2048xf32, 2048xf32)
        add_3 = paddle._C_ops.add(slice_7, scale_3)
        del scale_3, slice_7

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_4 = paddle._C_ops.subtract(add_2, add_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("30"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_1

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(subtract_4, full_1, float("0"), True)
        del subtract_4

        # pd_op.divide: (2048xf32) <- (2048xf32, 2048xf32)
        divide_0 = paddle._C_ops.divide(scale_4, subtract_0)

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_5 = paddle._C_ops.subtract(add_3, add_1)

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(subtract_5, full_1, float("0"), True)
        del subtract_5

        # pd_op.divide: (2048xf32) <- (2048xf32, 2048xf32)
        divide_1 = paddle._C_ops.divide(scale_5, subtract_1)

        # pd_op.divide: (2048xf32) <- (2048xf32, 2048xf32)
        divide_2 = paddle._C_ops.divide(subtract_2, subtract_0)

        # pd_op.log: (2048xf32) <- (2048xf32)
        log_0 = paddle._C_ops.log(divide_2)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("15"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_2

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(log_0, full_2, float("0"), True)
        del log_0

        # pd_op.divide: (2048xf32) <- (2048xf32, 2048xf32)
        divide_3 = paddle._C_ops.divide(subtract_3, subtract_1)

        # pd_op.log: (2048xf32) <- (2048xf32)
        log_1 = paddle._C_ops.log(divide_3)

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(log_1, full_2, float("0"), True)
        del log_1

        # builtin.combine: ([2048xf32, 2048xf32, 2048xf32, 2048xf32]) <- (2048xf32, 2048xf32, 2048xf32, 2048xf32)
        combine_0 = [divide_0, divide_1, scale_6, scale_7]

        # pd_op.stack: (2048x4xf32) <- ([2048xf32, 2048xf32, 2048xf32, 2048xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)
        del (
            add_0,
            add_1,
            add_2,
            add_3,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            combine_0,
            divide_0,
            divide_1,
            divide_2,
            divide_3,
            full_0,
            full_1,
            full_2,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            scale_0,
            scale_1,
            scale_4,
            scale_5,
            scale_6,
            scale_7,
            slice_0,
            slice_1,
            slice_2,
            slice_3,
            subtract_0,
            subtract_1,
            subtract_2,
            subtract_3,
        )

        return stack_0
