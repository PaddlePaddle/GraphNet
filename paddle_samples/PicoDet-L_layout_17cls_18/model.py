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
            [1], float("72"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (72xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="int64")

        # pd_op.cast: (72xf32) <- (72xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (72xf32) <- (72xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (72xf32) <- (72xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, float("0"), True)

        # builtin.combine: ([72xf32, 72xf32]) <- (72xf32, 72xf32)
        combine_0 = [scale_1, scale_1]

        # pd_op.meshgrid: ([72x72xf32, 72x72xf32]) <- ([72xf32, 72xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (72x72xf32, 72x72xf32) <- ([72x72xf32, 72x72xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0

        # pd_op.scale: (72x72xf32) <- (72x72xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(split_1, full_3, float("-20"), True)

        # pd_op.scale: (72x72xf32) <- (72x72xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_0, full_3, float("-20"), True)

        # pd_op.scale: (72x72xf32) <- (72x72xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_3, float("20"), True)

        # pd_op.scale: (72x72xf32) <- (72x72xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_3, float("20"), True)

        # builtin.combine: ([72x72xf32, 72x72xf32, 72x72xf32, 72x72xf32]) <- (72x72xf32, 72x72xf32, 72x72xf32, 72x72xf32)
        combine_1 = [scale_2, scale_3, scale_4, scale_5]

        # pd_op.stack: (72x72x4xf32) <- ([72x72xf32, 72x72xf32, 72x72xf32, 72x72xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # builtin.combine: ([72x72xf32, 72x72xf32]) <- (72x72xf32, 72x72xf32)
        combine_2 = [split_1, split_0]

        # pd_op.stack: (72x72x2xf32) <- ([72x72xf32, 72x72xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (5184x4xf32) <- (72x72x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (5184x2xf32) <- (72x72x2xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(stack_1, full_int_array_1)

        # pd_op.full: (5184x1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [5184, 1],
            float("8"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("36"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (36xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_6, full_2, dtype="int64")

        # pd_op.cast: (36xf32) <- (36xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (36xf32) <- (36xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_1, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (36xf32) <- (36xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_7, float("0"), True)

        # builtin.combine: ([36xf32, 36xf32]) <- (36xf32, 36xf32)
        combine_3 = [scale_7, scale_7]

        # pd_op.meshgrid: ([36x36xf32, 36x36xf32]) <- ([36xf32, 36xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (36x36xf32, 36x36xf32) <- ([36x36xf32, 36x36xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1

        # pd_op.scale: (36x36xf32) <- (36x36xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(split_3, full_3, float("-40"), True)

        # pd_op.scale: (36x36xf32) <- (36x36xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(split_2, full_3, float("-40"), True)

        # pd_op.scale: (36x36xf32) <- (36x36xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(split_3, full_3, float("40"), True)

        # pd_op.scale: (36x36xf32) <- (36x36xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(split_2, full_3, float("40"), True)

        # builtin.combine: ([36x36xf32, 36x36xf32, 36x36xf32, 36x36xf32]) <- (36x36xf32, 36x36xf32, 36x36xf32, 36x36xf32)
        combine_4 = [scale_8, scale_9, scale_10, scale_11]

        # pd_op.stack: (36x36x4xf32) <- ([36x36xf32, 36x36xf32, 36x36xf32, 36x36xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)

        # builtin.combine: ([36x36xf32, 36x36xf32]) <- (36x36xf32, 36x36xf32)
        combine_5 = [split_3, split_2]

        # pd_op.stack: (36x36x2xf32) <- ([36x36xf32, 36x36xf32])
        stack_3 = paddle._C_ops.stack(combine_5, -1)

        # pd_op.reshape: (1296x4xf32) <- (36x36x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_2, full_int_array_0)

        # pd_op.reshape: (1296x2xf32) <- (36x36x2xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(stack_3, full_int_array_1)

        # pd_op.full: (1296x1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1296, 1],
            float("16"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("18"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (18xi64) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_9, full_2, dtype="int64")

        # pd_op.cast: (18xf32) <- (18xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (18xf32) <- (18xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(cast_2, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (18xf32) <- (18xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(scale_12, full_10, float("0"), True)

        # builtin.combine: ([18xf32, 18xf32]) <- (18xf32, 18xf32)
        combine_6 = [scale_13, scale_13]

        # pd_op.meshgrid: ([18x18xf32, 18x18xf32]) <- ([18xf32, 18xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (18x18xf32, 18x18xf32) <- ([18x18xf32, 18x18xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2

        # pd_op.scale: (18x18xf32) <- (18x18xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_5, full_3, float("-80"), True)

        # pd_op.scale: (18x18xf32) <- (18x18xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_4, full_3, float("-80"), True)

        # pd_op.scale: (18x18xf32) <- (18x18xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(split_5, full_3, float("80"), True)

        # pd_op.scale: (18x18xf32) <- (18x18xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(split_4, full_3, float("80"), True)

        # builtin.combine: ([18x18xf32, 18x18xf32, 18x18xf32, 18x18xf32]) <- (18x18xf32, 18x18xf32, 18x18xf32, 18x18xf32)
        combine_7 = [scale_14, scale_15, scale_16, scale_17]

        # pd_op.stack: (18x18x4xf32) <- ([18x18xf32, 18x18xf32, 18x18xf32, 18x18xf32])
        stack_4 = paddle._C_ops.stack(combine_7, -1)

        # builtin.combine: ([18x18xf32, 18x18xf32]) <- (18x18xf32, 18x18xf32)
        combine_8 = [split_5, split_4]

        # pd_op.stack: (18x18x2xf32) <- ([18x18xf32, 18x18xf32])
        stack_5 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.reshape: (324x4xf32) <- (18x18x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(stack_4, full_int_array_0)

        # pd_op.reshape: (324x2xf32) <- (18x18x2xf32, 2xi64)
        reshape_6 = paddle._C_ops.reshape(stack_5, full_int_array_1)

        # pd_op.full: (324x1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [324, 1],
            float("32"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("9"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (9xi64) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_0, full_12, full_2, dtype="int64")

        # pd_op.cast: (9xf32) <- (9xi64)
        cast_3 = paddle._C_ops.cast(arange_3, paddle.float32)

        # pd_op.scale: (9xf32) <- (9xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_3, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("64"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (9xf32) <- (9xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_13, float("0"), True)

        # builtin.combine: ([9xf32, 9xf32]) <- (9xf32, 9xf32)
        combine_9 = [scale_19, scale_19]

        # pd_op.meshgrid: ([9x9xf32, 9x9xf32]) <- ([9xf32, 9xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_9)

        # builtin.split: (9x9xf32, 9x9xf32) <- ([9x9xf32, 9x9xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3

        # pd_op.scale: (9x9xf32) <- (9x9xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(split_7, full_3, float("-160"), True)

        # pd_op.scale: (9x9xf32) <- (9x9xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(split_6, full_3, float("-160"), True)

        # pd_op.scale: (9x9xf32) <- (9x9xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(split_7, full_3, float("160"), True)

        # pd_op.scale: (9x9xf32) <- (9x9xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(split_6, full_3, float("160"), True)

        # builtin.combine: ([9x9xf32, 9x9xf32, 9x9xf32, 9x9xf32]) <- (9x9xf32, 9x9xf32, 9x9xf32, 9x9xf32)
        combine_10 = [scale_20, scale_21, scale_22, scale_23]

        # pd_op.stack: (9x9x4xf32) <- ([9x9xf32, 9x9xf32, 9x9xf32, 9x9xf32])
        stack_6 = paddle._C_ops.stack(combine_10, -1)

        # builtin.combine: ([9x9xf32, 9x9xf32]) <- (9x9xf32, 9x9xf32)
        combine_11 = [split_7, split_6]

        # pd_op.stack: (9x9x2xf32) <- ([9x9xf32, 9x9xf32])
        stack_7 = paddle._C_ops.stack(combine_11, -1)

        # pd_op.reshape: (81x4xf32) <- (9x9x4xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(stack_6, full_int_array_0)

        # pd_op.reshape: (81x2xf32) <- (9x9x2xf32, 2xi64)
        reshape_7 = paddle._C_ops.reshape(stack_7, full_int_array_1)

        # pd_op.full: (81x1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [81, 1],
            float("64"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([5184x4xf32, 1296x4xf32, 324x4xf32, 81x4xf32]) <- (5184x4xf32, 1296x4xf32, 324x4xf32, 81x4xf32)
        combine_12 = [reshape_0, reshape_1, reshape_2, reshape_3]

        # pd_op.concat: (6885x4xf32) <- ([5184x4xf32, 1296x4xf32, 324x4xf32, 81x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_12, full_15)

        # builtin.combine: ([5184x2xf32, 1296x2xf32, 324x2xf32, 81x2xf32]) <- (5184x2xf32, 1296x2xf32, 324x2xf32, 81x2xf32)
        combine_13 = [reshape_4, reshape_5, reshape_6, reshape_7]

        # pd_op.concat: (6885x2xf32) <- ([5184x2xf32, 1296x2xf32, 324x2xf32, 81x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_13, full_15)

        # builtin.combine: ([5184x1xf32, 1296x1xf32, 324x1xf32, 81x1xf32]) <- (5184x1xf32, 1296x1xf32, 324x1xf32, 81x1xf32)
        combine_14 = [full_5, full_8, full_11, full_14]

        # pd_op.concat: (6885x1xf32) <- ([5184x1xf32, 1296x1xf32, 324x1xf32, 81x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_14, full_15)
        return concat_0, concat_1, reshape_0, reshape_1, reshape_2, reshape_3, concat_2
