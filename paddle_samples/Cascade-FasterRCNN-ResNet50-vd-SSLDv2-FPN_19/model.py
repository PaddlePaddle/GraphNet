import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
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
        assign_6 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_3

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_5 = slice_1

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_8 = slice_3

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_0

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_0, float("0"), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_0 = paddle._C_ops.add(slice_1, scale_0)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_1, full_0, float("0"), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_1 = paddle._C_ops.add(slice_3, scale_1)

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_2 = paddle._C_ops.subtract(slice_4, slice_5)

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_3 = paddle._C_ops.subtract(slice_6, slice_7)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(subtract_2, full_0, float("0"), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_2 = paddle._C_ops.add(slice_5, scale_6)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(subtract_3, full_0, float("0"), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_3 = paddle._C_ops.add(slice_7, scale_7)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_4 = paddle._C_ops.subtract(add_2, add_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("30"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(subtract_4, full_1, float("0"), True)

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_0 = paddle._C_ops.divide(scale_2, subtract_0)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_5 = paddle._C_ops.subtract(add_3, add_1)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(subtract_5, full_1, float("0"), True)

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_1 = paddle._C_ops.divide(scale_3, subtract_1)

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_2 = paddle._C_ops.divide(subtract_2, subtract_0)

        # pd_op.log: (-1xf32) <- (-1xf32)
        log_0 = paddle._C_ops.log(divide_2)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("15"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_2

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(log_0, full_2, float("0"), True)

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_3 = paddle._C_ops.divide(subtract_3, subtract_1)

        # pd_op.log: (-1xf32) <- (-1xf32)
        log_1 = paddle._C_ops.log(divide_3)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(log_1, full_2, float("0"), True)

        # builtin.combine: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, -1xf32, -1xf32, -1xf32)
        combine_0 = [divide_0, divide_1, scale_4, scale_5]

        # pd_op.stack: (-1x4xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)
        return (
            full_int_array_0,
            full_int_array_1,
            slice_0,
            full_int_array_2,
            full_int_array_3,
            slice_1,
            subtract_0,
            assign_0,
            full_int_array_4,
            slice_2,
            assign_1,
            assign_2,
            slice_3,
            subtract_1,
            assign_3,
            assign_4,
            assign_5,
            full_0,
            scale_0,
            add_0,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            scale_1,
            add_1,
            subtract_2,
            subtract_3,
            add_2,
            add_3,
            full_1,
            scale_2,
            divide_0,
            assign_10,
            scale_3,
            divide_1,
            divide_2,
            full_2,
            scale_4,
            divide_3,
            assign_11,
            scale_5,
            stack_0,
        )
