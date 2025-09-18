import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("16"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (1x21504x16xf32) <- (1x21504xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            data_0 % paddle.cast(full_0, data_0.dtype), full_0
        )
        del data_0, full_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.slice: (1x21504x15xf32) <- (1x21504x16xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            one_hot_0, [2], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_0, full_int_array_1, one_hot_0

        # pd_op.pow: (1x21504x15xf32) <- (1x21504x15xf32)
        pow_0 = paddle._C_ops.pow(data_2, float("2"))

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.75"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21504x15xf32) <- (1x21504x15xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(pow_0, full_1, float("0"), True)
        del pow_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21504x15xf32) <- (1x21504x15xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_2, float("1"), True)
        del full_2

        # pd_op.multiply: (1x21504x15xf32) <- (1x21504x15xf32, 1x21504x15xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, scale_1)

        # pd_op.multiply: (1x21504x15xf32) <- (1x21504x15xf32, 1x21504x15xf32)
        multiply_1 = paddle._C_ops.multiply(data_1, slice_0)
        del slice_0

        # pd_op.add: (1x21504x15xf32) <- (1x21504x15xf32, 1x21504x15xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.bce_loss: (1x21504x15xf32) <- (1x21504x15xf32, 1x21504x15xf32)
        bce_loss_0 = paddle._C_ops.bce_loss(data_2, data_1)
        del data_2

        # pd_op.multiply: (1x21504x15xf32) <- (1x21504x15xf32, 1x21504x15xf32)
        multiply_2 = paddle._C_ops.multiply(bce_loss_0, add_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_2 = []

        # pd_op.sum: (xf32) <- (1x21504x15xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_2, None, False)

        # pd_op.sum: (xf32) <- (1x21504x15xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(data_1, full_int_array_2, None, False)
        del data_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (xf32) <- (xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(sum_1, full_3, full_4)
        del full_3, full_4, sum_1

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_0, clip_0)
        del (
            add_0,
            bce_loss_0,
            clip_0,
            full_1,
            full_int_array_2,
            multiply_0,
            multiply_1,
            multiply_2,
            scale_0,
            scale_1,
            sum_0,
        )

        return divide_0
