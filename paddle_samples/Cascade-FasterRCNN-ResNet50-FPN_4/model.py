import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_4

        # builtin.combine: ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32]) <- (512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32)
        combine_0 = [data_1, data_2, data_3, data_4]

        # pd_op.concat: (2048x4xf32) <- ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_2

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_1, full_int_array_2, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_4

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_2 = paddle._C_ops.subtract(slice_0, slice_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_5

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_2, full_int_array_5, [1], [1]
        )

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_4, full_int_array_1, [1], [1]
        )

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_3 = paddle._C_ops.subtract(slice_2, slice_3)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_21 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_19 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_5

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(subtract_2, full_5, float("0"), True)

        # pd_op.add: (2048xf32) <- (2048xf32, 2048xf32)
        add_4 = paddle._C_ops.add(slice_1, scale_6)

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(subtract_3, full_5, float("0"), True)

        # pd_op.add: (2048xf32) <- (2048xf32, 2048xf32)
        add_5 = paddle._C_ops.add(slice_3, scale_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_0

        # pd_op.strided_slice: (2048x1xf32) <- (2048x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_3, full_int_array_0, full_int_array_5
        )

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_0

        # pd_op.scale: (2048x1xf32) <- (2048x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(strided_slice_0, full_0, float("0"), True)

        # pd_op.strided_slice: (2048x1xf32) <- (2048x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_4, full_int_array_0, full_int_array_5
        )

        # pd_op.scale: (2048x1xf32) <- (2048x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(strided_slice_1, full_0, float("0"), True)

        # pd_op.strided_slice: (2048x1xf32) <- (2048x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_1, full_int_array_0, full_int_array_5
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_1

        # pd_op.scale: (2048x1xf32) <- (2048x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(strided_slice_2, full_1, float("0"), True)

        # pd_op.strided_slice: (2048x1xf32) <- (2048x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_2, full_int_array_0, full_int_array_5
        )

        # pd_op.scale: (2048x1xf32) <- (2048x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(strided_slice_3, full_1, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("4.13517"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_3

        # pd_op.clip: (2048x1xf32) <- (2048x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(scale_2, full_2, full_3)

        # pd_op.clip: (2048x1xf32) <- (2048x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(scale_3, full_2, full_3)

        # pd_op.unsqueeze: (2048x1xf32) <- (2048xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(subtract_2, full_int_array_4)

        # pd_op.assign: (2048x1xf32) <- (2048x1xf32)
        assign_15 = unsqueeze_0

        # pd_op.multiply: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, unsqueeze_0)

        # pd_op.unsqueeze: (2048x1xf32) <- (2048xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(add_4, full_int_array_4)

        # pd_op.add: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        add_0 = paddle._C_ops.add(multiply_0, unsqueeze_1)

        # pd_op.unsqueeze: (2048x1xf32) <- (2048xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(subtract_3, full_int_array_4)

        # pd_op.assign: (2048x1xf32) <- (2048x1xf32)
        assign_16 = unsqueeze_2

        # pd_op.multiply: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_1, unsqueeze_2)

        # pd_op.unsqueeze: (2048x1xf32) <- (2048xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(add_5, full_int_array_4)

        # pd_op.add: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        add_1 = paddle._C_ops.add(multiply_1, unsqueeze_3)

        # pd_op.exp: (2048x1xf32) <- (2048x1xf32)
        exp_0 = paddle._C_ops.exp(clip_0)

        # pd_op.multiply: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        multiply_2 = paddle._C_ops.multiply(exp_0, unsqueeze_0)

        # pd_op.exp: (2048x1xf32) <- (2048x1xf32)
        exp_1 = paddle._C_ops.exp(clip_1)

        # pd_op.multiply: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        multiply_3 = paddle._C_ops.multiply(exp_1, unsqueeze_2)

        # pd_op.scale: (2048x1xf32) <- (2048x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(multiply_2, full_5, float("0"), True)

        # pd_op.assign: (2048x1xf32) <- (2048x1xf32)
        assign_20 = scale_4

        # pd_op.subtract: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        subtract_0 = paddle._C_ops.subtract(add_0, scale_4)

        # pd_op.scale: (2048x1xf32) <- (2048x1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(multiply_3, full_5, float("0"), True)

        # pd_op.assign: (2048x1xf32) <- (2048x1xf32)
        assign_22 = scale_5

        # pd_op.subtract: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        subtract_1 = paddle._C_ops.subtract(add_1, scale_5)

        # pd_op.add: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        add_2 = paddle._C_ops.add(add_0, scale_4)

        # pd_op.add: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        add_3 = paddle._C_ops.add(add_1, scale_5)

        # builtin.combine: ([2048x1xf32, 2048x1xf32, 2048x1xf32, 2048x1xf32]) <- (2048x1xf32, 2048x1xf32, 2048x1xf32, 2048x1xf32)
        combine_1 = [subtract_0, subtract_1, add_2, add_3]

        # pd_op.stack: (2048x1x4xf32) <- ([2048x1xf32, 2048x1xf32, 2048x1xf32, 2048x1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [-1, 4]

        # pd_op.reshape: (2048x4xf32) <- (2048x1x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_6)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [512, 512, 512, 512]

        # pd_op.split: ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32]) <- (2048x4xf32, 4xi64, 1xi32)
        split_4 = paddle._C_ops.split(reshape_0, full_int_array_7, full_4)

        # builtin.split: (512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32) <- ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_4
        return (
            assign_0,
            full_int_array_0,
            assign_1,
            full_0,
            scale_0,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            scale_1,
            assign_6,
            assign_7,
            assign_8,
            full_1,
            scale_2,
            assign_9,
            assign_10,
            assign_11,
            assign_12,
            scale_3,
            full_2,
            full_3,
            assign_13,
            assign_14,
            unsqueeze_0,
            multiply_0,
            unsqueeze_1,
            add_0,
            unsqueeze_2,
            multiply_1,
            unsqueeze_3,
            add_1,
            exp_0,
            assign_15,
            exp_1,
            assign_16,
            assign_17,
            scale_4,
            subtract_0,
            assign_18,
            scale_5,
            subtract_1,
            assign_19,
            assign_20,
            add_2,
            assign_21,
            assign_22,
            add_3,
            stack_0,
            assign_23,
            split_0,
            split_1,
            split_2,
            split_3,
        )
