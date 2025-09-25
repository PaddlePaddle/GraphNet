import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (6x4xf32) <- (2x3x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(data_1, full_int_array_0)
        del data_1, full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.unsqueeze: (6x1x4xf32) <- (6x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(reshape_2, full_int_array_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.unsqueeze: (1x7681x4xf32) <- (7681x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_0, full_int_array_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (6x1x2xf32) <- (6x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            unsqueeze_0, [2], full_int_array_2, full_int_array_3, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.slice: (6x1x2xf32) <- (6x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_0, [2], full_int_array_3, full_int_array_4, [1], []
        )
        del unsqueeze_0

        # pd_op.slice: (1x7681x2xf32) <- (1x7681x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_1, [2], full_int_array_2, full_int_array_3, [1], []
        )

        # pd_op.slice: (1x7681x2xf32) <- (1x7681x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_1, [2], full_int_array_3, full_int_array_4, [1], []
        )
        del full_int_array_4, unsqueeze_1

        # pd_op.maximum: (6x7681x2xf32) <- (6x1x2xf32, 1x7681x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_0, slice_2)

        # pd_op.minimum: (6x7681x2xf32) <- (6x1x2xf32, 1x7681x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_1, slice_3)

        # pd_op.subtract: (6x7681x2xf32) <- (6x7681x2xf32, 6x7681x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (6x7681x2xf32) <- (6x7681x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_0, full_1)
        del subtract_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-1]

        # pd_op.prod: (6x7681xf32) <- (6x7681x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_5, False, False)
        del clip_0

        # pd_op.subtract: (6x1x2xf32) <- (6x1x2xf32, 6x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_1, slice_0)
        del slice_0, slice_1

        # pd_op.clip: (6x1x2xf32) <- (6x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_0, full_1)
        del subtract_1

        # pd_op.prod: (6x1xf32) <- (6x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_5, False, False)
        del clip_1

        # pd_op.subtract: (1x7681x2xf32) <- (1x7681x2xf32, 1x7681x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_3, slice_2)
        del slice_2, slice_3

        # pd_op.clip: (1x7681x2xf32) <- (1x7681x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_0, full_1)
        del full_0, full_1, subtract_2

        # pd_op.prod: (1x7681xf32) <- (1x7681x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_5, False, False)
        del clip_2, full_int_array_5

        # pd_op.add: (6x7681xf32) <- (6x1xf32, 1x7681xf32)
        add_0 = paddle._C_ops.add(prod_1, prod_2)
        del prod_1, prod_2

        # pd_op.subtract: (6x7681xf32) <- (6x7681xf32, 6x7681xf32)
        subtract_3 = paddle._C_ops.subtract(add_0, prod_0)
        del add_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (6x7681xf32) <- (6x7681xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_3, full_2, float("1e-10"), True)
        del full_2, subtract_3

        # pd_op.divide: (6x7681xf32) <- (6x7681xf32, 6x7681xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_0)
        del prod_0, scale_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [2, -1, 7681]

        # pd_op.reshape: (2x3x7681xf32) <- (6x7681xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(divide_0, full_int_array_6)
        del divide_0

        # pd_op.slice: (6xf32) <- (6x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_2, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [3]

        # pd_op.slice: (6xf32) <- (6x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_3, full_int_array_7, [1], [1]
        )

        # pd_op.add: (6xf32) <- (6xf32, 6xf32)
        add_1 = paddle._C_ops.add(slice_4, slice_5)
        del slice_4, slice_5

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_3, float("0"), True)
        del add_1

        # pd_op.slice: (6xf32) <- (6x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_1, full_int_array_3, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [4]

        # pd_op.slice: (6xf32) <- (6x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            reshape_2, [1], full_int_array_7, full_int_array_8, [1], [1]
        )
        del reshape_2

        # pd_op.add: (6xf32) <- (6xf32, 6xf32)
        add_2 = paddle._C_ops.add(slice_6, slice_7)
        del slice_6, slice_7

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(add_2, full_3, float("0"), True)
        del add_2

        # builtin.combine: ([6xf32, 6xf32]) <- (6xf32, 6xf32)
        combine_0 = [scale_1, scale_2]
        del scale_1, scale_2

        # pd_op.stack: (6x2xf32) <- ([6xf32, 6xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.unsqueeze: (6x1x2xf32) <- (6x2xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(stack_0, full_int_array_1)
        del stack_0

        # pd_op.slice: (7681xf32) <- (7681x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (7681xf32) <- (7681x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_7, [1], [1]
        )

        # pd_op.add: (7681xf32) <- (7681xf32, 7681xf32)
        add_3 = paddle._C_ops.add(slice_8, slice_9)
        del slice_8, slice_9

        # pd_op.scale: (7681xf32) <- (7681xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_3, full_3, float("0"), True)
        del add_3

        # pd_op.slice: (7681xf32) <- (7681x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_3, [1], [1]
        )
        del full_int_array_1, full_int_array_3

        # pd_op.slice: (7681xf32) <- (7681x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_0, [1], full_int_array_7, full_int_array_8, [1], [1]
        )
        del data_0, full_int_array_7, full_int_array_8

        # pd_op.add: (7681xf32) <- (7681xf32, 7681xf32)
        add_4 = paddle._C_ops.add(slice_10, slice_11)
        del slice_10, slice_11

        # pd_op.scale: (7681xf32) <- (7681xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(add_4, full_3, float("0"), True)
        del add_4, full_3

        # builtin.combine: ([7681xf32, 7681xf32]) <- (7681xf32, 7681xf32)
        combine_1 = [scale_3, scale_4]
        del scale_3, scale_4

        # pd_op.stack: (7681x2xf32) <- ([7681xf32, 7681xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.unsqueeze: (1x7681x2xf32) <- (7681x2xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(stack_1, full_int_array_2)
        del full_int_array_2

        # pd_op.subtract: (6x7681x2xf32) <- (6x1x2xf32, 1x7681x2xf32)
        subtract_4 = paddle._C_ops.subtract(unsqueeze_2, unsqueeze_3)
        del unsqueeze_2, unsqueeze_3

        # pd_op.p_norm: (6x7681xf32) <- (6x7681x2xf32)
        p_norm_0 = paddle._C_ops.p_norm(
            subtract_4, float("2"), -1, float("1e-12"), False, False
        )
        del subtract_4

        # pd_op.reshape: (2x3x7681xf32) <- (6x7681xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(p_norm_0, full_int_array_6)
        del full_int_array_6, p_norm_0, stack_1

        return reshape_0, reshape_1
