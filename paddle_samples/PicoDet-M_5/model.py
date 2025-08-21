import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 4]

        # pd_op.reshape: (4x4xf32) <- (4x1x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(data_1, full_int_array_2)

        # pd_op.unsqueeze: (4x1x4xf32) <- (4x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(reshape_2, full_int_array_1)

        # pd_op.unsqueeze: (1x-1x4xf32) <- (-1x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (4x1x2xf32) <- (4x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_0, [2], full_int_array_0, full_int_array_3, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.slice: (4x1x2xf32) <- (4x1x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_0, [2], full_int_array_3, full_int_array_4, [1], []
        )

        # pd_op.slice: (1x-1x2xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_1, [2], full_int_array_0, full_int_array_3, [1], []
        )

        # pd_op.slice: (1x-1x2xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            unsqueeze_1, [2], full_int_array_3, full_int_array_4, [1], []
        )

        # pd_op.maximum: (4x-1x2xf32) <- (4x1x2xf32, 1x-1x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_1, slice_3)

        # pd_op.minimum: (4x-1x2xf32) <- (4x1x2xf32, 1x-1x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_2, slice_4)

        # pd_op.subtract: (4x-1x2xf32) <- (4x-1x2xf32, 4x-1x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (4x-1x2xf32) <- (4x-1x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_0, full_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-1]

        # pd_op.prod: (4x-1xf32) <- (4x-1x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_5, False, False)

        # pd_op.subtract: (4x1x2xf32) <- (4x1x2xf32, 4x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_1)

        # pd_op.clip: (4x1x2xf32) <- (4x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_0, full_1)

        # pd_op.prod: (4x1xf32) <- (4x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_5, False, False)

        # pd_op.subtract: (1x-1x2xf32) <- (1x-1x2xf32, 1x-1x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_4, slice_3)

        # pd_op.clip: (1x-1x2xf32) <- (1x-1x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_0, full_1)

        # pd_op.prod: (1x-1xf32) <- (1x-1x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_5, False, False)

        # pd_op.add: (4x-1xf32) <- (4x1xf32, 1x-1xf32)
        add_0 = paddle._C_ops.add(prod_1, prod_2)

        # pd_op.subtract: (4x-1xf32) <- (4x-1xf32, 4x-1xf32)
        subtract_3 = paddle._C_ops.subtract(add_0, prod_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x-1xf32) <- (4x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_3, full_2, float("1e-10"), True)

        # pd_op.divide: (4x-1xf32) <- (4x-1xf32, 4x-1xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_0)

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [full_3, full_4, slice_0]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (4x-1x-1xf32) <- (4x-1xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(divide_0, stack_1)

        # pd_op.slice: (4xf32) <- (4x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [3]

        # pd_op.slice: (4xf32) <- (4x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_3, full_int_array_6, [1], [1]
        )

        # pd_op.add: (4xf32) <- (4xf32, 4xf32)
        add_1 = paddle._C_ops.add(slice_5, slice_6)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4xf32) <- (4xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_5, float("0"), True)

        # pd_op.slice: (4xf32) <- (4x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_1, full_int_array_3, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [4]

        # pd_op.slice: (4xf32) <- (4x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_6, full_int_array_7, [1], [1]
        )

        # pd_op.add: (4xf32) <- (4xf32, 4xf32)
        add_2 = paddle._C_ops.add(slice_7, slice_8)

        # pd_op.scale: (4xf32) <- (4xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(add_2, full_5, float("0"), True)

        # builtin.combine: ([4xf32, 4xf32]) <- (4xf32, 4xf32)
        combine_1 = [scale_1, scale_2]

        # pd_op.stack: (4x2xf32) <- ([4xf32, 4xf32])
        stack_2 = paddle._C_ops.stack(combine_1, -1)

        # pd_op.unsqueeze: (4x1x2xf32) <- (4x2xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(stack_2, full_int_array_1)

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_6, [1], [1]
        )

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_3 = paddle._C_ops.add(slice_9, slice_10)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_3, full_5, float("0"), True)

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_3, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            data_0, [1], full_int_array_6, full_int_array_7, [1], [1]
        )

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_4 = paddle._C_ops.add(slice_11, slice_12)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(add_4, full_5, float("0"), True)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_2 = [scale_3, scale_4]

        # pd_op.stack: (-1x2xf32) <- ([-1xf32, -1xf32])
        stack_0 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.unsqueeze: (1x-1x2xf32) <- (-1x2xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(stack_0, full_int_array_0)

        # pd_op.subtract: (4x-1x2xf32) <- (4x1x2xf32, 1x-1x2xf32)
        subtract_4 = paddle._C_ops.subtract(unsqueeze_2, unsqueeze_3)

        # pd_op.p_norm: (4x-1xf32) <- (4x-1x2xf32)
        p_norm_0 = paddle._C_ops.p_norm(
            subtract_4, float("2"), -1, float("1e-12"), False, False
        )

        # pd_op.reshape: (4x-1x-1xf32) <- (4x-1xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(p_norm_0, stack_1)
        return reshape_0, reshape_1, stack_0, slice_0
