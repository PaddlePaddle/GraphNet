import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.add: (9xf32) <- (9xf32, 9xf32)
        add_0 = paddle._C_ops.add(slice_0, slice_1)
        del slice_0, slice_1

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (9xf32) <- (9xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_0, float("0"), True)
        del add_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_0

        # pd_op.add: (9xf32) <- (9xf32, 9xf32)
        add_1 = paddle._C_ops.add(slice_2, slice_3)
        del slice_2, slice_3

        # pd_op.scale: (9xf32) <- (9xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_0, float("0"), True)
        del add_1, full_0

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del full_int_array_2

        # pd_op.subtract: (9xf32) <- (9xf32, 9xf32)
        subtract_0 = paddle._C_ops.subtract(scale_0, slice_4)
        del slice_4

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del full_int_array_3

        # pd_op.subtract: (9xf32) <- (9xf32, 9xf32)
        subtract_1 = paddle._C_ops.subtract(scale_1, slice_5)
        del slice_5

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )
        del full_int_array_0

        # pd_op.subtract: (9xf32) <- (9xf32, 9xf32)
        subtract_2 = paddle._C_ops.subtract(slice_6, scale_0)
        del scale_0, slice_6

        # pd_op.slice: (9xf32) <- (9x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_4, [1], [1]
        )
        del data_1, full_int_array_1, full_int_array_4

        # pd_op.subtract: (9xf32) <- (9xf32, 9xf32)
        subtract_3 = paddle._C_ops.subtract(slice_7, scale_1)
        del scale_1, slice_7

        # builtin.combine: ([9xf32, 9xf32]) <- (9xf32, 9xf32)
        combine_0 = [subtract_0, subtract_2]
        del subtract_0, subtract_2

        # pd_op.stack: (9x2xf32) <- ([9xf32, 9xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)
        del combine_0

        # builtin.combine: ([9xf32, 9xf32]) <- (9xf32, 9xf32)
        combine_1 = [subtract_1, subtract_3]
        del subtract_1, subtract_3

        # pd_op.stack: (9x2xf32) <- ([9xf32, 9xf32])
        stack_1 = paddle._C_ops.stack(combine_1, 1)
        del combine_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-1]

        # pd_op.min: (9xf32) <- (9x2xf32, 1xi64)
        min_0 = paddle._C_ops.min(stack_0, full_int_array_5, False)

        # pd_op.max: (9xf32) <- (9x2xf32, 1xi64)
        max_0 = paddle._C_ops.max(stack_0, full_int_array_5, False)
        del stack_0

        # pd_op.divide: (9xf32) <- (9xf32, 9xf32)
        divide_0 = paddle._C_ops.divide(min_0, max_0)
        del max_0, min_0

        # pd_op.min: (9xf32) <- (9x2xf32, 1xi64)
        min_1 = paddle._C_ops.min(stack_1, full_int_array_5, False)

        # pd_op.max: (9xf32) <- (9x2xf32, 1xi64)
        max_1 = paddle._C_ops.max(stack_1, full_int_array_5, False)
        del full_int_array_5, stack_1

        # pd_op.divide: (9xf32) <- (9xf32, 9xf32)
        divide_1 = paddle._C_ops.divide(min_1, max_1)
        del max_1, min_1

        # pd_op.multiply: (9xf32) <- (9xf32, 9xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, divide_1)
        del divide_0, divide_1

        # pd_op.sqrt: (9xf32) <- (9xf32)
        sqrt_0 = paddle._C_ops.sqrt(multiply_0)
        del multiply_0

        # pd_op.isnan: (9xb) <- (9xf32)
        isnan_0 = paddle._C_ops.isnan(sqrt_0)

        # pd_op.any: (xb) <- (9xb)
        any_0 = paddle._C_ops.any(isnan_0, [], False)
        del isnan_0, sqrt_0

        return any_0
