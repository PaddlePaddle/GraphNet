import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("14"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (14xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")

        # builtin.combine: ([14xf32, 14xf32]) <- (14xf32, 14xf32)
        combine_0 = [arange_0, arange_0]

        # pd_op.meshgrid: ([14x14xf32, 14x14xf32]) <- ([14xf32, 14xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (14x14xf32, 14x14xf32) <- ([14x14xf32, 14x14xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("256"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (256xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_3, full_2, dtype="float32")

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.00390625"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (256xf32) <- (256xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(arange_1, full_4, float("0"), True)

        # pd_op.full: (256xf32) <- ()
        full_5 = paddle._C_ops.full(
            [256],
            float("10000"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.elementwise_pow: (256xf32) <- (256xf32, 256xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(full_5, scale_0)

        # pd_op.full: (256xf32) <- ()
        full_6 = paddle._C_ops.full(
            [256],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.divide: (256xf32) <- (256xf32, 256xf32)
        divide_0 = paddle._C_ops.divide(full_6, elementwise_pow_0)

        # pd_op.flatten: (196xf32) <- (14x14xf32)
        flatten_0 = paddle._C_ops.flatten(split_0, 0, 1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (196x1xf32) <- (196xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(flatten_0, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.unsqueeze: (1x256xf32) <- (256xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(divide_0, full_int_array_1)

        # pd_op.matmul: (196x256xf32) <- (196x1xf32, 1x256xf32)
        matmul_0 = paddle._C_ops.matmul(unsqueeze_1, unsqueeze_2, False, False)

        # pd_op.flatten: (196xf32) <- (14x14xf32)
        flatten_1 = paddle._C_ops.flatten(split_1, 0, 1)

        # pd_op.unsqueeze: (196x1xf32) <- (196xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(flatten_1, full_int_array_0)

        # pd_op.matmul: (196x256xf32) <- (196x1xf32, 1x256xf32)
        matmul_1 = paddle._C_ops.matmul(unsqueeze_3, unsqueeze_2, False, False)

        # pd_op.sin: (196x256xf32) <- (196x256xf32)
        sin_0 = paddle._C_ops.sin(matmul_0)

        # pd_op.cos: (196x256xf32) <- (196x256xf32)
        cos_0 = paddle._C_ops.cos(matmul_0)

        # pd_op.sin: (196x256xf32) <- (196x256xf32)
        sin_1 = paddle._C_ops.sin(matmul_1)

        # pd_op.cos: (196x256xf32) <- (196x256xf32)
        cos_1 = paddle._C_ops.cos(matmul_1)

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([196x256xf32, 196x256xf32, 196x256xf32, 196x256xf32]) <- (196x256xf32, 196x256xf32, 196x256xf32, 196x256xf32)
        combine_1 = [sin_0, cos_0, sin_1, cos_1]

        # pd_op.concat: (196x1024xf32) <- ([196x256xf32, 196x256xf32, 196x256xf32, 196x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_7)

        # pd_op.unsqueeze: (1x196x1024xf32) <- (196x1024xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(concat_0, full_int_array_1)
        return unsqueeze_0
