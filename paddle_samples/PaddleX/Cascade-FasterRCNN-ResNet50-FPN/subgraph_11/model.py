import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, float("0"), True)
        del data_0, full_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(scale_0, full_1)
        del scale_0

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.int64)
        del equal_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_1)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_1 = paddle._C_ops.equal(cast_1, full_1)
        del cast_1, full_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_1

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_3

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (1xf32) <- (1xf32, 1xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_0, slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_4

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (1xf32) <- (1xf32, 1xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_2, slice_3

        # pd_op.multiply: (1xf32) <- (1xf32, 1xf32)
        multiply_0 = paddle._C_ops.multiply(subtract_0, subtract_1)
        del subtract_0, subtract_1

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_2, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_2 = paddle._C_ops.subtract(slice_4, slice_5)

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_2, [1], full_int_array_1, full_int_array_4, [1], [1]
        )
        del full_int_array_1, full_int_array_4

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_2, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_3 = paddle._C_ops.subtract(slice_6, slice_7)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_2, subtract_3)

        # pd_op.unsqueeze: (1x1x4xf32) <- (1x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_3)
        del data_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_5

        # pd_op.slice: (1x1x2xf32) <- (1x1x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            unsqueeze_0, [2], full_int_array_0, full_int_array_5, [1], []
        )

        # pd_op.slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_5, [1], []
        )
        del full_int_array_5

        # pd_op.minimum: (1x-1x2xf32) <- (1x1x2xf32, -1x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_8, slice_9)

        # pd_op.slice: (1x1x2xf32) <- (1x1x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            unsqueeze_0, [2], full_int_array_2, full_int_array_0, [1], []
        )
        del unsqueeze_0

        # pd_op.slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_2, [1], full_int_array_2, full_int_array_0, [1], []
        )
        del data_2, full_int_array_2

        # pd_op.maximum: (1x-1x2xf32) <- (1x1x2xf32, -1x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_10, slice_11)

        # pd_op.subtract: (1x-1x2xf32) <- (1x-1x2xf32, 1x-1x2xf32)
        subtract_4 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x-1x2xf32) <- (1x-1x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_4, full_2, full_3)

        # pd_op.prod: (1x-1xf32) <- (1x-1x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_0, False, False)
        del full_int_array_0

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x-1xb) <- (1x-1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(prod_0, full_4)
        del full_4

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(multiply_0, full_int_array_3)
        del full_int_array_3, multiply_0

        # pd_op.add: (1x-1xf32) <- (1x1xf32, -1xf32)
        add_0 = paddle._C_ops.add(unsqueeze_1, multiply_1)

        # pd_op.subtract: (1x-1xf32) <- (1x-1xf32, 1x-1xf32)
        subtract_5 = paddle._C_ops.subtract(add_0, prod_0)

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x-1xf32)
        divide_0 = paddle._C_ops.divide(prod_0, subtract_5)

        # pd_op.full_like: (1x-1xf32) <- (1x-1xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            prod_0, full_2, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (1x-1xf32) <- (1x-1xb, 1x-1xf32, 1x-1xf32)
        where_0 = paddle._C_ops.where(greater_than_0, divide_0, full_like_0)
        del (
            add_0,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            clip_0,
            divide_0,
            full_2,
            full_3,
            full_like_0,
            greater_than_0,
            maximum_0,
            minimum_0,
            multiply_1,
            prod_0,
            slice_10,
            slice_11,
            slice_4,
            slice_5,
            slice_6,
            slice_7,
            slice_8,
            slice_9,
            subtract_2,
            subtract_3,
            subtract_4,
            subtract_5,
            unsqueeze_1,
        )

        return where_0
