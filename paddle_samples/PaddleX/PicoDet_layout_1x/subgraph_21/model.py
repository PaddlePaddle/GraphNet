import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (5832xf32) <- (5832x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (5832xf32) <- (5832x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (5832xf32) <- (5832xf32, 5832xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_0, slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (5832xf32) <- (5832x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (5832xf32) <- (5832x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (5832xf32) <- (5832xf32, 5832xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_2, slice_3

        # pd_op.multiply: (5832xf32) <- (5832xf32, 5832xf32)
        multiply_0 = paddle._C_ops.multiply(subtract_0, subtract_1)
        del subtract_0, subtract_1

        # pd_op.slice: (2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (2xf32) <- (2xf32, 2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_4, slice_5)
        del slice_4, slice_5

        # pd_op.slice: (2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_4, [1], [1]
        )
        del full_int_array_1, full_int_array_4

        # pd_op.slice: (2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (2xf32) <- (2xf32, 2xf32)
        subtract_3 = paddle._C_ops.subtract(slice_6, slice_7)
        del slice_6, slice_7

        # pd_op.multiply: (2xf32) <- (2xf32, 2xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_2, subtract_3)
        del subtract_2, subtract_3

        # pd_op.slice: (5832x2xf32) <- (5832x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [5832, 1, 2]

        # pd_op.reshape: (5832x1x2xf32) <- (5832x2xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(slice_8, full_int_array_5)
        del slice_8

        # pd_op.slice: (2x2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_1, [1], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.maximum: (5832x2x2xf32) <- (5832x1x2xf32, 2x2xf32)
        maximum_0 = paddle._C_ops.maximum(reshape_0, slice_9)
        del reshape_0, slice_9

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2147483647]

        # pd_op.slice: (5832x2xf32) <- (5832x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_6, [1], []
        )
        del data_0

        # pd_op.reshape: (5832x1x2xf32) <- (5832x2xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(slice_10, full_int_array_5)
        del full_int_array_5, slice_10

        # pd_op.slice: (2x2xf32) <- (2x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_6, [1], []
        )
        del data_1, full_int_array_6

        # pd_op.minimum: (5832x2x2xf32) <- (5832x1x2xf32, 2x2xf32)
        minimum_0 = paddle._C_ops.minimum(reshape_1, slice_11)
        del reshape_1, slice_11

        # pd_op.subtract: (5832x2x2xf32) <- (5832x2x2xf32, 5832x2x2xf32)
        subtract_4 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (5832x2x2xf32) <- (5832x2x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_4, full_0, full_1)
        del full_0, full_1, subtract_4

        # pd_op.slice: (5832x2xf32) <- (5832x2x2xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            clip_0, [2], full_int_array_2, full_int_array_3, [1], [2]
        )
        del full_int_array_2

        # pd_op.slice: (5832x2xf32) <- (5832x2x2xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            clip_0, [2], full_int_array_3, full_int_array_0, [1], [2]
        )
        del clip_0, full_int_array_0, full_int_array_3

        # pd_op.multiply: (5832x2xf32) <- (5832x2xf32, 5832x2xf32)
        multiply_2 = paddle._C_ops.multiply(slice_12, slice_13)
        del slice_12, slice_13

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [5832, 1]

        # pd_op.reshape: (5832x1xf32) <- (5832xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(multiply_0, full_int_array_7)
        del full_int_array_7, multiply_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [1, 2]

        # pd_op.reshape: (1x2xf32) <- (2xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(multiply_1, full_int_array_8)
        del full_int_array_8, multiply_1

        # pd_op.add: (5832x2xf32) <- (5832x1xf32, 1x2xf32)
        add_0 = paddle._C_ops.add(reshape_2, reshape_3)
        del reshape_2, reshape_3

        # pd_op.subtract: (5832x2xf32) <- (5832x2xf32, 5832x2xf32)
        subtract_5 = paddle._C_ops.subtract(add_0, multiply_2)
        del add_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (1xf32) <- (1xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_2,
            [1],
            paddle.float32,
            [float("1e-06")],
            paddle.framework._current_expected_place(),
        )
        del full_2

        # pd_op.maximum: (5832x2xf32) <- (5832x2xf32, 1xf32)
        maximum_1 = paddle._C_ops.maximum(subtract_5, assign_value__0)
        del assign_value__0, subtract_5

        # pd_op.divide: (5832x2xf32) <- (5832x2xf32, 5832x2xf32)
        divide_0 = paddle._C_ops.divide(multiply_2, maximum_1)
        del maximum_1, multiply_2

        return divide_0
