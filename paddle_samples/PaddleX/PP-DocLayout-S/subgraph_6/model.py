import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_1, full_0)
        del data_1

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.int64)
        del equal_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_0)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_1 = paddle._C_ops.equal(cast_1, full_0)
        del cast_1, full_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (1x-1x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(data_3, full_int_array_0)
        del data_3, full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.unsqueeze: (-1x1x4xf32) <- (-1x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(reshape_2, full_int_array_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.unsqueeze: (1x-1x4xf32) <- (-1x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_2, full_int_array_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (-1x1x2xf32) <- (-1x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            unsqueeze_0, [2], full_int_array_2, full_int_array_3, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.slice: (-1x1x2xf32) <- (-1x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_0, [2], full_int_array_3, full_int_array_4, [1], []
        )
        del unsqueeze_0

        # pd_op.slice: (1x-1x2xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_1, [2], full_int_array_2, full_int_array_3, [1], []
        )

        # pd_op.slice: (1x-1x2xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_1, [2], full_int_array_3, full_int_array_4, [1], []
        )
        del full_int_array_4, unsqueeze_1

        # pd_op.maximum: (-1x-1x2xf32) <- (-1x1x2xf32, 1x-1x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_0, slice_2)

        # pd_op.minimum: (-1x-1x2xf32) <- (-1x1x2xf32, 1x-1x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_1, slice_3)

        # pd_op.subtract: (-1x-1x2xf32) <- (-1x-1x2xf32, -1x-1x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (-1x-1x2xf32) <- (-1x-1x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_1, full_2)
        del subtract_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-1]

        # pd_op.prod: (-1x-1xf32) <- (-1x-1x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_5, False, False)
        del clip_0

        # pd_op.subtract: (-1x1x2xf32) <- (-1x1x2xf32, -1x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_1, slice_0)
        del slice_0, slice_1

        # pd_op.clip: (-1x1x2xf32) <- (-1x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_1, full_2)
        del subtract_1

        # pd_op.prod: (-1x1xf32) <- (-1x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_5, False, False)
        del clip_1

        # pd_op.subtract: (1x-1x2xf32) <- (1x-1x2xf32, 1x-1x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_3, slice_2)
        del slice_2, slice_3

        # pd_op.clip: (1x-1x2xf32) <- (1x-1x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_1, full_2)
        del full_1, full_2, subtract_2

        # pd_op.prod: (1x-1xf32) <- (1x-1x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_5, False, False)
        del clip_2, full_int_array_5

        # pd_op.add: (-1x-1xf32) <- (-1x1xf32, 1x-1xf32)
        add_0 = paddle._C_ops.add(prod_1, prod_2)
        del prod_1, prod_2

        # pd_op.subtract: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        subtract_3 = paddle._C_ops.subtract(add_0, prod_0)
        del add_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_3, full_3, float("1e-10"), True)
        del full_3, subtract_3

        # pd_op.divide: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_0)
        del prod_0, scale_0

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [full_4, full_5, data_0]
        del data_0, full_4, full_5

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (1x-1x-1xf32) <- (-1x-1xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(divide_0, stack_0)
        del divide_0

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_2, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [3]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_3, full_int_array_6, [1], [1]
        )

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_1 = paddle._C_ops.add(slice_4, slice_5)
        del slice_4, slice_5

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_6, float("0"), True)
        del add_1

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_1, full_int_array_3, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [4]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_6, full_int_array_7, [1], [1]
        )
        del reshape_2

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_2 = paddle._C_ops.add(slice_6, slice_7)
        del slice_6, slice_7

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(add_2, full_6, float("0"), True)
        del add_2

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_1 = [scale_1, scale_2]
        del scale_1, scale_2

        # pd_op.stack: (-1x2xf32) <- ([-1xf32, -1xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.unsqueeze: (-1x1x2xf32) <- (-1x2xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(stack_1, full_int_array_1)
        del stack_1

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_2, [1], full_int_array_2, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_2, [1], full_int_array_3, full_int_array_6, [1], [1]
        )

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_3 = paddle._C_ops.add(slice_8, slice_9)
        del slice_8, slice_9

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_3, full_6, float("0"), True)
        del add_3

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_2, [1], full_int_array_1, full_int_array_3, [1], [1]
        )
        del full_int_array_1, full_int_array_3

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_2, [1], full_int_array_6, full_int_array_7, [1], [1]
        )
        del data_2, full_int_array_6, full_int_array_7

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_4 = paddle._C_ops.add(slice_10, slice_11)
        del slice_10, slice_11

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(add_4, full_6, float("0"), True)
        del add_4, full_6

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_2 = [scale_3, scale_4]
        del scale_3, scale_4

        # pd_op.stack: (-1x2xf32) <- ([-1xf32, -1xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.unsqueeze: (1x-1x2xf32) <- (-1x2xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(stack_2, full_int_array_2)
        del full_int_array_2

        # pd_op.subtract: (-1x-1x2xf32) <- (-1x1x2xf32, 1x-1x2xf32)
        subtract_4 = paddle._C_ops.subtract(unsqueeze_2, unsqueeze_3)
        del unsqueeze_2, unsqueeze_3

        # pd_op.p_norm: (-1x-1xf32) <- (-1x-1x2xf32)
        p_norm_0 = paddle._C_ops.p_norm(
            subtract_4, float("2"), -1, float("1e-12"), False, False
        )
        del subtract_4

        # pd_op.reshape: (1x-1x-1xf32) <- (-1x-1xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(p_norm_0, stack_0)
        del p_norm_0, stack_0, stack_2

        return reshape_0, reshape_1
