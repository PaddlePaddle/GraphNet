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
            [1], float("17"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (17xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")
        del full_1

        # builtin.combine: ([17xf32, 17xf32]) <- (17xf32, 17xf32)
        combine_0 = [arange_0, arange_0]
        del arange_0

        # pd_op.meshgrid: ([17x17xf32, 17x17xf32]) <- ([17xf32, 17xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (17x17xf32, 17x17xf32) <- ([17x17xf32, 17x17xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("64"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (64xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_3, full_2, dtype="float32")
        del full_0, full_2, full_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.015625"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (64xf32) <- (64xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(arange_1, full_4, float("0"), True)
        del arange_1, full_4

        # pd_op.full: (64xf32) <- ()
        full_5 = paddle._C_ops.full(
            [64],
            float("10000"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.elementwise_pow: (64xf32) <- (64xf32, 64xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(full_5, scale_0)
        del full_5, scale_0

        # pd_op.full: (64xf32) <- ()
        full_6 = paddle._C_ops.full(
            [64], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.divide: (64xf32) <- (64xf32, 64xf32)
        divide_0 = paddle._C_ops.divide(full_6, elementwise_pow_0)
        del elementwise_pow_0, full_6

        # pd_op.flatten: (289xf32) <- (17x17xf32)
        flatten_0 = paddle._C_ops.flatten(split_0, 0, 1)
        del split_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (289x1xf32) <- (289xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(flatten_0, full_int_array_0)
        del flatten_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.unsqueeze: (1x64xf32) <- (64xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(divide_0, full_int_array_1)
        del divide_0

        # pd_op.matmul: (289x64xf32) <- (289x1xf32, 1x64xf32)
        matmul_0 = paddle._C_ops.matmul(unsqueeze_1, unsqueeze_2, False, False)
        del unsqueeze_1

        # pd_op.flatten: (289xf32) <- (17x17xf32)
        flatten_1 = paddle._C_ops.flatten(split_1, 0, 1)
        del split_1

        # pd_op.unsqueeze: (289x1xf32) <- (289xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(flatten_1, full_int_array_0)
        del flatten_1, full_int_array_0

        # pd_op.matmul: (289x64xf32) <- (289x1xf32, 1x64xf32)
        matmul_1 = paddle._C_ops.matmul(unsqueeze_3, unsqueeze_2, False, False)
        del unsqueeze_2, unsqueeze_3

        # pd_op.sin: (289x64xf32) <- (289x64xf32)
        sin_0 = paddle._C_ops.sin(matmul_0)

        # pd_op.cos: (289x64xf32) <- (289x64xf32)
        cos_0 = paddle._C_ops.cos(matmul_0)
        del matmul_0

        # pd_op.sin: (289x64xf32) <- (289x64xf32)
        sin_1 = paddle._C_ops.sin(matmul_1)

        # pd_op.cos: (289x64xf32) <- (289x64xf32)
        cos_1 = paddle._C_ops.cos(matmul_1)
        del matmul_1

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([289x64xf32, 289x64xf32, 289x64xf32, 289x64xf32]) <- (289x64xf32, 289x64xf32, 289x64xf32, 289x64xf32)
        combine_1 = [sin_0, cos_0, sin_1, cos_1]
        del cos_0, cos_1, sin_0, sin_1

        # pd_op.concat: (289x256xf32) <- ([289x64xf32, 289x64xf32, 289x64xf32, 289x64xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_7)
        del combine_1, full_7

        # pd_op.unsqueeze: (1x289x256xf32) <- (289x256xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(concat_0, full_int_array_1)
        del concat_0, full_int_array_1

        return unsqueeze_0
