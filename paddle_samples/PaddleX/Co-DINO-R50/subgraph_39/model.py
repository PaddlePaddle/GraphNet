import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_3

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (9xf32) <- (9xf32, 9xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_4

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_1

        # pd_op.subtract: (9xf32) <- (9xf32, 9xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_2

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_0

        # pd_op.scale: (9xf32) <- (9xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_0, float("0"), True)

        # pd_op.add: (9xf32) <- (9xf32, 9xf32)
        add_0 = paddle._C_ops.add(slice_1, scale_0)
        del scale_0, slice_1

        # pd_op.scale: (9xf32) <- (9xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_1, full_0, float("0"), True)

        # pd_op.add: (9xf32) <- (9xf32, 9xf32)
        add_1 = paddle._C_ops.add(slice_3, scale_1)
        del scale_1, slice_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_5

        # pd_op.strided_slice: (9x1xf32) <- (9x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_2, full_int_array_5, full_int_array_4
        )
        del full_int_array_2

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_1

        # pd_op.scale: (9x1xf32) <- (9x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(strided_slice_0, full_1, float("0"), True)
        del strided_slice_0

        # pd_op.strided_slice: (9x1xf32) <- (9x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_3, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (9x1xf32) <- (9x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(strided_slice_1, full_1, float("0"), True)
        del strided_slice_1

        # pd_op.strided_slice: (9x1xf32) <- (9x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_0, full_int_array_5, full_int_array_4
        )
        del full_int_array_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_2

        # pd_op.scale: (9x1xf32) <- (9x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(strided_slice_2, full_2, float("0"), True)
        del strided_slice_2

        # pd_op.strided_slice: (9x1xf32) <- (9x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_1, full_int_array_5, full_int_array_4
        )
        del data_0, full_int_array_1, full_int_array_4

        # pd_op.scale: (9x1xf32) <- (9x1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(strided_slice_3, full_2, float("0"), True)
        del strided_slice_3

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("4.13517"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_4

        # pd_op.clip: (9x1xf32) <- (9x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(scale_4, full_3, full_4)

        # pd_op.clip: (9x1xf32) <- (9x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(scale_5, full_3, full_4)

        # pd_op.unsqueeze: (9x1xf32) <- (9xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(subtract_0, full_int_array_3)
        del subtract_0

        # pd_op.assign: (9x1xf32) <- (9x1xf32)
        assign_19 = unsqueeze_0

        # pd_op.multiply: (9x1xf32) <- (9x1xf32, 9x1xf32)
        multiply_0 = paddle._C_ops.multiply(scale_2, unsqueeze_0)

        # pd_op.unsqueeze: (9x1xf32) <- (9xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(add_0, full_int_array_3)
        del add_0

        # pd_op.add: (9x1xf32) <- (9x1xf32, 9x1xf32)
        add_2 = paddle._C_ops.add(multiply_0, unsqueeze_1)

        # pd_op.unsqueeze: (9x1xf32) <- (9xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(subtract_1, full_int_array_3)
        del subtract_1

        # pd_op.assign: (9x1xf32) <- (9x1xf32)
        assign_20 = unsqueeze_2

        # pd_op.multiply: (9x1xf32) <- (9x1xf32, 9x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_3, unsqueeze_2)

        # pd_op.unsqueeze: (9x1xf32) <- (9xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(add_1, full_int_array_3)
        del add_1, full_int_array_3

        # pd_op.add: (9x1xf32) <- (9x1xf32, 9x1xf32)
        add_3 = paddle._C_ops.add(multiply_1, unsqueeze_3)

        # pd_op.exp: (9x1xf32) <- (9x1xf32)
        exp_0 = paddle._C_ops.exp(clip_0)
        del clip_0

        # pd_op.multiply: (9x1xf32) <- (9x1xf32, 9x1xf32)
        multiply_2 = paddle._C_ops.multiply(exp_0, unsqueeze_0)

        # pd_op.exp: (9x1xf32) <- (9x1xf32)
        exp_1 = paddle._C_ops.exp(clip_1)
        del clip_1

        # pd_op.multiply: (9x1xf32) <- (9x1xf32, 9x1xf32)
        multiply_3 = paddle._C_ops.multiply(exp_1, unsqueeze_2)

        # pd_op.scale: (9x1xf32) <- (9x1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(multiply_2, full_0, float("0"), True)
        del multiply_2

        # pd_op.assign: (9x1xf32) <- (9x1xf32)
        assign_21 = scale_6

        # pd_op.subtract: (9x1xf32) <- (9x1xf32, 9x1xf32)
        subtract_2 = paddle._C_ops.subtract(add_2, scale_6)

        # pd_op.scale: (9x1xf32) <- (9x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(multiply_3, full_0, float("0"), True)
        del full_0, multiply_3

        # pd_op.assign: (9x1xf32) <- (9x1xf32)
        assign_22 = scale_7

        # pd_op.subtract: (9x1xf32) <- (9x1xf32, 9x1xf32)
        subtract_3 = paddle._C_ops.subtract(add_3, scale_7)

        # pd_op.add: (9x1xf32) <- (9x1xf32, 9x1xf32)
        add_4 = paddle._C_ops.add(add_2, scale_6)

        # pd_op.add: (9x1xf32) <- (9x1xf32, 9x1xf32)
        add_5 = paddle._C_ops.add(add_3, scale_7)

        # builtin.combine: ([9x1xf32, 9x1xf32, 9x1xf32, 9x1xf32]) <- (9x1xf32, 9x1xf32, 9x1xf32, 9x1xf32)
        combine_0 = [subtract_2, subtract_3, add_4, add_5]

        # pd_op.stack: (9x1x4xf32) <- ([9x1xf32, 9x1xf32, 9x1xf32, 9x1xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del (
            add_2,
            add_3,
            add_4,
            add_5,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            combine_0,
            exp_0,
            exp_1,
            full_1,
            full_2,
            full_3,
            full_4,
            full_int_array_5,
            multiply_0,
            multiply_1,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            scale_6,
            scale_7,
            subtract_2,
            subtract_3,
            unsqueeze_0,
            unsqueeze_1,
            unsqueeze_2,
            unsqueeze_3,
        )

        return stack_0
