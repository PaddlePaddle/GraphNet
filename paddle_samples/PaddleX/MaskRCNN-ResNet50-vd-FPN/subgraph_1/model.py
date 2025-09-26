import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (153450xi32) <- ()
        full_0 = paddle._C_ops.full(
            [153450],
            float("-1"),
            paddle.int32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (255xi32) <- (255xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_1, full_1, paddle.int32, paddle.framework._current_expected_place()
        )
        del full_1

        # pd_op.scatter: (153450xi32) <- (153450xi32, 255xi32, 255xi32)
        scatter_1 = paddle._C_ops.scatter(full_0, data_1, full_like_0, True)
        del data_1, full_0, full_like_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1xi32) <- (1xi32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_0, full_2, paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.scatter: (153450xi32) <- (153450xi32, 1xi32, 1xi32)
        scatter_0 = paddle._C_ops.scatter(scatter_1, data_0, full_like_1, True)
        del data_0, full_like_1, scatter_1

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (153450x4xf32) <- (1x4xf32, 153450xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_2, data_3, full_3)
        del data_2, data_3, full_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (153450xf32) <- (153450x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_4, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (153450xf32) <- (153450x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_4, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (153450xf32) <- (153450xf32, 153450xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (153450xf32) <- (153450x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_4, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (153450xf32) <- (153450x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_4, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_4

        # pd_op.subtract: (153450xf32) <- (153450xf32, 153450xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_2

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (153450xf32) <- (153450xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_4, float("0"), True)

        # pd_op.add: (153450xf32) <- (153450xf32, 153450xf32)
        add_0 = paddle._C_ops.add(slice_1, scale_0)
        del scale_0, slice_1

        # pd_op.scale: (153450xf32) <- (153450xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_1, full_4, float("0"), True)

        # pd_op.add: (153450xf32) <- (153450xf32, 153450xf32)
        add_1 = paddle._C_ops.add(slice_3, scale_1)
        del scale_1, slice_3

        # pd_op.slice: (153450xf32) <- (153450x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            gather_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (153450xf32) <- (153450x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            gather_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del full_int_array_2

        # pd_op.subtract: (153450xf32) <- (153450xf32, 153450xf32)
        subtract_2 = paddle._C_ops.subtract(slice_4, slice_5)
        del slice_4

        # pd_op.slice: (153450xf32) <- (153450x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            gather_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )
        del full_int_array_1, full_int_array_4

        # pd_op.slice: (153450xf32) <- (153450x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            gather_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del full_int_array_0, full_int_array_3

        # pd_op.subtract: (153450xf32) <- (153450xf32, 153450xf32)
        subtract_3 = paddle._C_ops.subtract(slice_6, slice_7)
        del slice_6

        # pd_op.scale: (153450xf32) <- (153450xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(subtract_2, full_4, float("0"), True)

        # pd_op.add: (153450xf32) <- (153450xf32, 153450xf32)
        add_2 = paddle._C_ops.add(slice_5, scale_2)
        del scale_2, slice_5

        # pd_op.scale: (153450xf32) <- (153450xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(subtract_3, full_4, float("0"), True)
        del full_4

        # pd_op.add: (153450xf32) <- (153450xf32, 153450xf32)
        add_3 = paddle._C_ops.add(slice_7, scale_3)
        del scale_3, slice_7

        # pd_op.subtract: (153450xf32) <- (153450xf32, 153450xf32)
        subtract_4 = paddle._C_ops.subtract(add_2, add_0)
        del add_0, add_2

        # pd_op.scale: (153450xf32) <- (153450xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(subtract_4, full_2, float("0"), True)
        del subtract_4

        # pd_op.divide: (153450xf32) <- (153450xf32, 153450xf32)
        divide_0 = paddle._C_ops.divide(scale_4, subtract_0)
        del scale_4

        # pd_op.subtract: (153450xf32) <- (153450xf32, 153450xf32)
        subtract_5 = paddle._C_ops.subtract(add_3, add_1)
        del add_1, add_3

        # pd_op.scale: (153450xf32) <- (153450xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(subtract_5, full_2, float("0"), True)
        del subtract_5

        # pd_op.divide: (153450xf32) <- (153450xf32, 153450xf32)
        divide_1 = paddle._C_ops.divide(scale_5, subtract_1)
        del scale_5

        # pd_op.divide: (153450xf32) <- (153450xf32, 153450xf32)
        divide_2 = paddle._C_ops.divide(subtract_2, subtract_0)
        del subtract_0, subtract_2

        # pd_op.log: (153450xf32) <- (153450xf32)
        log_0 = paddle._C_ops.log(divide_2)
        del divide_2

        # pd_op.scale: (153450xf32) <- (153450xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(log_0, full_2, float("0"), True)
        del log_0

        # pd_op.divide: (153450xf32) <- (153450xf32, 153450xf32)
        divide_3 = paddle._C_ops.divide(subtract_3, subtract_1)
        del subtract_1, subtract_3

        # pd_op.log: (153450xf32) <- (153450xf32)
        log_1 = paddle._C_ops.log(divide_3)
        del divide_3

        # pd_op.scale: (153450xf32) <- (153450xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(log_1, full_2, float("0"), True)
        del full_2, log_1

        # builtin.combine: ([153450xf32, 153450xf32, 153450xf32, 153450xf32]) <- (153450xf32, 153450xf32, 153450xf32, 153450xf32)
        combine_0 = [divide_0, divide_1, scale_6, scale_7]
        del divide_0, divide_1, scale_6, scale_7

        # pd_op.stack: (153450x4xf32) <- ([153450xf32, 153450xf32, 153450xf32, 153450xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)
        del combine_0, gather_0

        return scatter_0, stack_0
