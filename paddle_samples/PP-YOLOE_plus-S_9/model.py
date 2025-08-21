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
            [1], float("15"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (15xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="int64")

        # pd_op.cast: (15xf32) <- (15xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (15xf32) <- (15xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (15xf32) <- (15xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, float("0"), True)

        # builtin.combine: ([15xf32, 15xf32]) <- (15xf32, 15xf32)
        combine_0 = [scale_1, scale_1]

        # pd_op.meshgrid: ([15x15xf32, 15x15xf32]) <- ([15xf32, 15xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (15x15xf32, 15x15xf32) <- ([15x15xf32, 15x15xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0

        # pd_op.scale: (15x15xf32) <- (15x15xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(split_1, full_3, float("-80"), True)

        # pd_op.scale: (15x15xf32) <- (15x15xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_0, full_3, float("-80"), True)

        # pd_op.scale: (15x15xf32) <- (15x15xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_3, float("80"), True)

        # pd_op.scale: (15x15xf32) <- (15x15xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_3, float("80"), True)

        # builtin.combine: ([15x15xf32, 15x15xf32, 15x15xf32, 15x15xf32]) <- (15x15xf32, 15x15xf32, 15x15xf32, 15x15xf32)
        combine_1 = [scale_2, scale_3, scale_4, scale_5]

        # pd_op.stack: (15x15x4xf32) <- ([15x15xf32, 15x15xf32, 15x15xf32, 15x15xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # builtin.combine: ([15x15xf32, 15x15xf32]) <- (15x15xf32, 15x15xf32)
        combine_2 = [split_1, split_0]

        # pd_op.stack: (15x15x2xf32) <- ([15x15xf32, 15x15xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (225x4xf32) <- (15x15x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (225x2xf32) <- (15x15x2xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(stack_1, full_int_array_1)

        # pd_op.full: (225x1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [225, 1],
            float("32"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("30"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (30xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_6, full_2, dtype="int64")

        # pd_op.cast: (30xf32) <- (30xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (30xf32) <- (30xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_1, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (30xf32) <- (30xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_7, float("0"), True)

        # builtin.combine: ([30xf32, 30xf32]) <- (30xf32, 30xf32)
        combine_3 = [scale_7, scale_7]

        # pd_op.meshgrid: ([30x30xf32, 30x30xf32]) <- ([30xf32, 30xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (30x30xf32, 30x30xf32) <- ([30x30xf32, 30x30xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1

        # pd_op.scale: (30x30xf32) <- (30x30xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(split_3, full_3, float("-40"), True)

        # pd_op.scale: (30x30xf32) <- (30x30xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(split_2, full_3, float("-40"), True)

        # pd_op.scale: (30x30xf32) <- (30x30xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(split_3, full_3, float("40"), True)

        # pd_op.scale: (30x30xf32) <- (30x30xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(split_2, full_3, float("40"), True)

        # builtin.combine: ([30x30xf32, 30x30xf32, 30x30xf32, 30x30xf32]) <- (30x30xf32, 30x30xf32, 30x30xf32, 30x30xf32)
        combine_4 = [scale_8, scale_9, scale_10, scale_11]

        # pd_op.stack: (30x30x4xf32) <- ([30x30xf32, 30x30xf32, 30x30xf32, 30x30xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)

        # builtin.combine: ([30x30xf32, 30x30xf32]) <- (30x30xf32, 30x30xf32)
        combine_5 = [split_3, split_2]

        # pd_op.stack: (30x30x2xf32) <- ([30x30xf32, 30x30xf32])
        stack_3 = paddle._C_ops.stack(combine_5, -1)

        # pd_op.reshape: (900x4xf32) <- (30x30x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_2, full_int_array_0)

        # pd_op.reshape: (900x2xf32) <- (30x30x2xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(stack_3, full_int_array_1)

        # pd_op.full: (900x1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [900, 1],
            float("16"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("60"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (60xi64) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_9, full_2, dtype="int64")

        # pd_op.cast: (60xf32) <- (60xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (60xf32) <- (60xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(cast_2, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (60xf32) <- (60xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(scale_12, full_10, float("0"), True)

        # builtin.combine: ([60xf32, 60xf32]) <- (60xf32, 60xf32)
        combine_6 = [scale_13, scale_13]

        # pd_op.meshgrid: ([60x60xf32, 60x60xf32]) <- ([60xf32, 60xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (60x60xf32, 60x60xf32) <- ([60x60xf32, 60x60xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2

        # pd_op.scale: (60x60xf32) <- (60x60xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_5, full_3, float("-20"), True)

        # pd_op.scale: (60x60xf32) <- (60x60xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_4, full_3, float("-20"), True)

        # pd_op.scale: (60x60xf32) <- (60x60xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(split_5, full_3, float("20"), True)

        # pd_op.scale: (60x60xf32) <- (60x60xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(split_4, full_3, float("20"), True)

        # builtin.combine: ([60x60xf32, 60x60xf32, 60x60xf32, 60x60xf32]) <- (60x60xf32, 60x60xf32, 60x60xf32, 60x60xf32)
        combine_7 = [scale_14, scale_15, scale_16, scale_17]

        # pd_op.stack: (60x60x4xf32) <- ([60x60xf32, 60x60xf32, 60x60xf32, 60x60xf32])
        stack_4 = paddle._C_ops.stack(combine_7, -1)

        # builtin.combine: ([60x60xf32, 60x60xf32]) <- (60x60xf32, 60x60xf32)
        combine_8 = [split_5, split_4]

        # pd_op.stack: (60x60x2xf32) <- ([60x60xf32, 60x60xf32])
        stack_5 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.reshape: (3600x4xf32) <- (60x60x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(stack_4, full_int_array_0)

        # pd_op.reshape: (3600x2xf32) <- (60x60x2xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(stack_5, full_int_array_1)

        # pd_op.full: (3600x1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [3600, 1],
            float("8"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([225x4xf32, 900x4xf32, 3600x4xf32]) <- (225x4xf32, 900x4xf32, 3600x4xf32)
        combine_9 = [reshape_0, reshape_1, reshape_2]

        # pd_op.concat: (4725x4xf32) <- ([225x4xf32, 900x4xf32, 3600x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_9, full_12)

        # builtin.combine: ([225x2xf32, 900x2xf32, 3600x2xf32]) <- (225x2xf32, 900x2xf32, 3600x2xf32)
        combine_10 = [reshape_3, reshape_4, reshape_5]

        # pd_op.concat: (4725x2xf32) <- ([225x2xf32, 900x2xf32, 3600x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_10, full_12)

        # builtin.combine: ([225x1xf32, 900x1xf32, 3600x1xf32]) <- (225x1xf32, 900x1xf32, 3600x1xf32)
        combine_11 = [full_5, full_8, full_11]

        # pd_op.concat: (4725x1xf32) <- ([225x1xf32, 900x1xf32, 3600x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_11, full_12)
        return concat_0, concat_1, reshape_0, reshape_1, reshape_2, concat_2
