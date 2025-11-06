import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (1xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_3, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del data_3

        # pd_op.slice: (2xf32) <- (1x2xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del data_4

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [slice_1, full_0]
        del full_0, slice_1

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.expand: (-1x2xf32) <- (2xf32, 2xi64)
        expand_0 = paddle._C_ops.expand(slice_2, stack_1)
        del slice_2, stack_1

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x2xf32]) <- (-1x2xf32)
        combine_1 = [expand_0]
        del expand_0

        # pd_op.concat: (-1x2xf32) <- ([-1x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_1)
        del combine_1

        # builtin.combine: ([1000x4xf32]) <- (1000x4xf32)
        combine_2 = [data_2]
        del data_2

        # pd_op.concat: (1000x4xf32) <- ([1000x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_1)
        del combine_2, full_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.subtract: (1000xf32) <- (1000xf32, 1000xf32)
        subtract_0 = paddle._C_ops.subtract(slice_3, slice_4)
        del slice_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del concat_1

        # pd_op.subtract: (1000xf32) <- (1000xf32, 1000xf32)
        subtract_1 = paddle._C_ops.subtract(slice_5, slice_6)
        del slice_5

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1000xf32) <- (1000xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_2, float("0"), True)

        # pd_op.add: (1000xf32) <- (1000xf32, 1000xf32)
        add_0 = paddle._C_ops.add(slice_4, scale_0)
        del scale_0, slice_4

        # pd_op.scale: (1000xf32) <- (1000xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_1, full_2, float("0"), True)

        # pd_op.add: (1000xf32) <- (1000xf32, 1000xf32)
        add_1 = paddle._C_ops.add(slice_6, scale_1)
        del scale_1, slice_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.strided_slice: (1000x2xf32) <- (1000x8xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_0, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1000x2xf32) <- (1000x2xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(strided_slice_0, full_3, float("0"), True)
        del strided_slice_0

        # pd_op.strided_slice: (1000x2xf32) <- (1000x8xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_1, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (1000x2xf32) <- (1000x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(strided_slice_1, full_3, float("0"), True)
        del full_3, strided_slice_1

        # pd_op.strided_slice: (1000x2xf32) <- (1000x8xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_2, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1000x2xf32) <- (1000x2xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(strided_slice_2, full_4, float("0"), True)
        del strided_slice_2

        # pd_op.strided_slice: (1000x2xf32) <- (1000x8xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_3, full_int_array_5, full_int_array_4
        )
        del data_0, full_int_array_5

        # pd_op.scale: (1000x2xf32) <- (1000x2xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(strided_slice_3, full_4, float("0"), True)
        del full_4, strided_slice_3

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("-3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("4.13517"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1000x2xf32) <- (1000x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(scale_4, full_5, full_6)
        del scale_4

        # pd_op.clip: (1000x2xf32) <- (1000x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(scale_5, full_5, full_6)
        del full_5, full_6, scale_5

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(subtract_0, full_int_array_1)
        del subtract_0

        # pd_op.multiply: (1000x2xf32) <- (1000x2xf32, 1000x1xf32)
        multiply_0 = paddle._C_ops.multiply(scale_2, unsqueeze_0)
        del scale_2

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(add_0, full_int_array_1)
        del add_0

        # pd_op.add: (1000x2xf32) <- (1000x2xf32, 1000x1xf32)
        add_2 = paddle._C_ops.add(multiply_0, unsqueeze_1)
        del multiply_0, unsqueeze_1

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(subtract_1, full_int_array_1)
        del subtract_1

        # pd_op.multiply: (1000x2xf32) <- (1000x2xf32, 1000x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_3, unsqueeze_2)
        del scale_3

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(add_1, full_int_array_1)
        del add_1

        # pd_op.add: (1000x2xf32) <- (1000x2xf32, 1000x1xf32)
        add_3 = paddle._C_ops.add(multiply_1, unsqueeze_3)
        del multiply_1, unsqueeze_3

        # pd_op.exp: (1000x2xf32) <- (1000x2xf32)
        exp_0 = paddle._C_ops.exp(clip_0)
        del clip_0

        # pd_op.multiply: (1000x2xf32) <- (1000x2xf32, 1000x1xf32)
        multiply_2 = paddle._C_ops.multiply(exp_0, unsqueeze_0)
        del exp_0, unsqueeze_0

        # pd_op.exp: (1000x2xf32) <- (1000x2xf32)
        exp_1 = paddle._C_ops.exp(clip_1)
        del clip_1

        # pd_op.multiply: (1000x2xf32) <- (1000x2xf32, 1000x1xf32)
        multiply_3 = paddle._C_ops.multiply(exp_1, unsqueeze_2)
        del exp_1, unsqueeze_2

        # pd_op.scale: (1000x2xf32) <- (1000x2xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(multiply_2, full_2, float("0"), True)
        del multiply_2

        # pd_op.subtract: (1000x2xf32) <- (1000x2xf32, 1000x2xf32)
        subtract_2 = paddle._C_ops.subtract(add_2, scale_6)

        # pd_op.scale: (1000x2xf32) <- (1000x2xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(multiply_3, full_2, float("0"), True)
        del full_2, multiply_3

        # pd_op.subtract: (1000x2xf32) <- (1000x2xf32, 1000x2xf32)
        subtract_3 = paddle._C_ops.subtract(add_3, scale_7)

        # pd_op.add: (1000x2xf32) <- (1000x2xf32, 1000x2xf32)
        add_4 = paddle._C_ops.add(add_2, scale_6)
        del add_2, scale_6

        # pd_op.add: (1000x2xf32) <- (1000x2xf32, 1000x2xf32)
        add_5 = paddle._C_ops.add(add_3, scale_7)
        del add_3, scale_7

        # builtin.combine: ([1000x2xf32, 1000x2xf32, 1000x2xf32, 1000x2xf32]) <- (1000x2xf32, 1000x2xf32, 1000x2xf32, 1000x2xf32)
        combine_3 = [subtract_2, subtract_3, add_4, add_5]
        del add_4, add_5, subtract_2, subtract_3

        # pd_op.stack: (1000x2x4xf32) <- ([1000x2xf32, 1000x2xf32, 1000x2xf32, 1000x2xf32])
        stack_2 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [-1]

        # pd_op.slice: (1000x2xf32) <- (1000x3xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_6, [1], []
        )
        del data_1, full_int_array_6

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [1000, 2, 4]

        # pd_op.expand: (1000x2x4xf32) <- (1000x2x4xf32, 3xi64)
        expand_1 = paddle._C_ops.expand(stack_2, full_int_array_7)
        del full_int_array_7, stack_2

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.unsqueeze: (-1x1xf32) <- (-1xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(slice_7, full_int_array_1)
        del slice_7

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del concat_0

        # pd_op.unsqueeze: (-1x1xf32) <- (-1xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(slice_8, full_int_array_1)
        del slice_8

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1x1xf32) <- (-1x1xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            unsqueeze_4,
            full_7,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.slice: (1000x2xf32) <- (1000x2x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            expand_1, [2], full_int_array_0, full_int_array_1, [1], [2]
        )
        del full_int_array_0

        # pd_op.minimum: (1000x2xf32) <- (1000x2xf32, -1x1xf32)
        minimum_0 = paddle._C_ops.minimum(slice_9, unsqueeze_5)
        del slice_9

        # pd_op.maximum: (1000x2xf32) <- (1000x2xf32, -1x1xf32)
        maximum_0 = paddle._C_ops.maximum(minimum_0, full_like_0)
        del minimum_0

        # pd_op.slice: (1000x2xf32) <- (1000x2x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            expand_1, [2], full_int_array_1, full_int_array_2, [1], [2]
        )
        del full_int_array_1

        # pd_op.minimum: (1000x2xf32) <- (1000x2xf32, -1x1xf32)
        minimum_1 = paddle._C_ops.minimum(slice_10, unsqueeze_4)
        del slice_10

        # pd_op.maximum: (1000x2xf32) <- (1000x2xf32, -1x1xf32)
        maximum_1 = paddle._C_ops.maximum(minimum_1, full_like_0)
        del minimum_1

        # pd_op.slice: (1000x2xf32) <- (1000x2x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            expand_1, [2], full_int_array_2, full_int_array_3, [1], [2]
        )
        del full_int_array_2

        # pd_op.minimum: (1000x2xf32) <- (1000x2xf32, -1x1xf32)
        minimum_2 = paddle._C_ops.minimum(slice_11, unsqueeze_5)
        del slice_11, unsqueeze_5

        # pd_op.maximum: (1000x2xf32) <- (1000x2xf32, -1x1xf32)
        maximum_2 = paddle._C_ops.maximum(minimum_2, full_like_0)
        del minimum_2

        # pd_op.slice: (1000x2xf32) <- (1000x2x4xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            expand_1, [2], full_int_array_3, full_int_array_4, [1], [2]
        )
        del expand_1, full_int_array_3, full_int_array_4

        # pd_op.minimum: (1000x2xf32) <- (1000x2xf32, -1x1xf32)
        minimum_3 = paddle._C_ops.minimum(slice_12, unsqueeze_4)
        del slice_12, unsqueeze_4

        # pd_op.maximum: (1000x2xf32) <- (1000x2xf32, -1x1xf32)
        maximum_3 = paddle._C_ops.maximum(minimum_3, full_like_0)
        del full_like_0, minimum_3

        # builtin.combine: ([1000x2xf32, 1000x2xf32, 1000x2xf32, 1000x2xf32]) <- (1000x2xf32, 1000x2xf32, 1000x2xf32, 1000x2xf32)
        combine_4 = [maximum_0, maximum_1, maximum_2, maximum_3]
        del maximum_0, maximum_1, maximum_2, maximum_3

        # pd_op.stack: (1000x2x4xf32) <- ([1000x2xf32, 1000x2xf32, 1000x2xf32, 1000x2xf32])
        stack_0 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        return stack_0, slice_0
