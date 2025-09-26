import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
    ):
        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_0, data_2, full_1, dtype="int64")
        del data_2

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)
        del arange_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_2, float("0.5"), True)
        del cast_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_3, float("0"), True)
        del scale_0

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_0, data_1, full_1, dtype="int64")
        del data_1

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)
        del arange_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_1, full_2, float("0.5"), True)
        del cast_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_3, float("0"), True)
        del scale_2

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [scale_3, scale_1]
        del scale_1, scale_3

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_2, float("-20"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_2, float("-20"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(split_1, full_2, float("20"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(split_0, full_2, float("20"), True)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32)
        combine_1 = [scale_4, scale_5, scale_6, scale_7]
        del scale_4, scale_5, scale_6, scale_7

        # pd_op.stack: (-1x-1x4xf32) <- ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_2 = [split_1, split_0]
        del split_0, split_1

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1x-1x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_1, full_int_array_0)
        del stack_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (-1x2xf32) <- (-1x-1x2xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_2, full_int_array_1)
        del stack_2

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_0 = paddle._C_ops.shape64(reshape_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_3 = [slice_0, full_4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.full_with_tensor: (-1x1xf32) <- (1xf32, 2xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_3, stack_3, paddle.float32
        )
        del full_3, stack_3

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_2 = paddle.arange(full_0, data_4, full_1, dtype="int64")
        del data_4

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)
        del arange_2

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_2, full_2, float("0.5"), True)
        del cast_2

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_5, float("0"), True)
        del scale_8

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_3 = paddle.arange(full_0, data_3, full_1, dtype="int64")
        del data_3

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_3 = paddle._C_ops.cast(arange_3, paddle.float32)
        del arange_3

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(cast_3, full_2, float("0.5"), True)
        del cast_3

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(scale_10, full_5, float("0"), True)
        del scale_10

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_4 = [scale_11, scale_9]
        del scale_11, scale_9

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_4)
        del combine_4

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(split_3, full_2, float("-40"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(split_2, full_2, float("-40"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_3, full_2, float("40"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_2, full_2, float("40"), True)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32)
        combine_5 = [scale_12, scale_13, scale_14, scale_15]
        del scale_12, scale_13, scale_14, scale_15

        # pd_op.stack: (-1x-1x4xf32) <- ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32])
        stack_4 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_6 = [split_3, split_2]
        del split_2, split_3

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_5 = paddle._C_ops.stack(combine_6, -1)
        del combine_6

        # pd_op.reshape: (-1x4xf32) <- (-1x-1x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(stack_4, full_int_array_0)
        del stack_4

        # pd_op.reshape: (-1x2xf32) <- (-1x-1x2xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(stack_5, full_int_array_1)
        del stack_5

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_1 = paddle._C_ops.shape64(reshape_2)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_7 = [slice_1, full_4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.full_with_tensor: (-1x1xf32) <- (1xf32, 2xi64)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(
            full_5, stack_6, paddle.float32
        )
        del full_5, stack_6

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_4 = paddle.arange(full_0, data_6, full_1, dtype="int64")
        del data_6

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_4 = paddle._C_ops.cast(arange_4, paddle.float32)
        del arange_4

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(cast_4, full_2, float("0.5"), True)
        del cast_4

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(scale_16, full_6, float("0"), True)
        del scale_16

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_5 = paddle.arange(full_0, data_5, full_1, dtype="int64")
        del data_5

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_5 = paddle._C_ops.cast(arange_5, paddle.float32)
        del arange_5

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_5, full_2, float("0.5"), True)
        del cast_5

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_6, float("0"), True)
        del scale_18

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_8 = [scale_19, scale_17]
        del scale_17, scale_19

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_8)
        del combine_8

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(split_5, full_2, float("-80"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(split_4, full_2, float("-80"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(split_5, full_2, float("80"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(split_4, full_2, float("80"), True)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32)
        combine_9 = [scale_20, scale_21, scale_22, scale_23]
        del scale_20, scale_21, scale_22, scale_23

        # pd_op.stack: (-1x-1x4xf32) <- ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32])
        stack_7 = paddle._C_ops.stack(combine_9, -1)
        del combine_9

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_10 = [split_5, split_4]
        del split_4, split_5

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_8 = paddle._C_ops.stack(combine_10, -1)
        del combine_10

        # pd_op.reshape: (-1x4xf32) <- (-1x-1x4xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(stack_7, full_int_array_0)
        del stack_7

        # pd_op.reshape: (-1x2xf32) <- (-1x-1x2xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(stack_8, full_int_array_1)
        del stack_8

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_2 = paddle._C_ops.shape64(reshape_4)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_2

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_11 = [slice_2, full_4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.full_with_tensor: (-1x1xf32) <- (1xf32, 2xi64)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(
            full_6, stack_9, paddle.float32
        )
        del full_6, stack_9

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_6 = paddle.arange(full_0, data_8, full_1, dtype="int64")
        del data_8

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_6 = paddle._C_ops.cast(arange_6, paddle.float32)
        del arange_6

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(cast_6, full_2, float("0.5"), True)
        del cast_6

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("64"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(scale_24, full_7, float("0"), True)
        del scale_24

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_7 = paddle.arange(full_0, data_7, full_1, dtype="int64")
        del data_7, full_0, full_1

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_7 = paddle._C_ops.cast(arange_7, paddle.float32)
        del arange_7

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(cast_7, full_2, float("0.5"), True)
        del cast_7

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(scale_26, full_7, float("0"), True)
        del scale_26

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_12 = [scale_27, scale_25]
        del scale_25, scale_27

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_12)
        del combine_12

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3
        del meshgrid_3

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(split_7, full_2, float("-160"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(split_6, full_2, float("-160"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(split_7, full_2, float("160"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(split_6, full_2, float("160"), True)
        del full_2

        # builtin.combine: ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32)
        combine_13 = [scale_28, scale_29, scale_30, scale_31]
        del scale_28, scale_29, scale_30, scale_31

        # pd_op.stack: (-1x-1x4xf32) <- ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32])
        stack_10 = paddle._C_ops.stack(combine_13, -1)
        del combine_13

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_14 = [split_7, split_6]
        del split_6, split_7

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_11 = paddle._C_ops.stack(combine_14, -1)
        del combine_14

        # pd_op.reshape: (-1x4xf32) <- (-1x-1x4xf32, 2xi64)
        reshape_6 = paddle._C_ops.reshape(stack_10, full_int_array_0)
        del full_int_array_0, stack_10

        # pd_op.reshape: (-1x2xf32) <- (-1x-1x2xf32, 2xi64)
        reshape_7 = paddle._C_ops.reshape(stack_11, full_int_array_1)
        del full_int_array_1, stack_11

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_3 = paddle._C_ops.shape64(reshape_6)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_3

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_15 = [slice_3, full_4]
        del full_4

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.full_with_tensor: (-1x1xf32) <- (1xf32, 2xi64)
        full_with_tensor_3 = paddle._C_ops.full_with_tensor(
            full_7, stack_12, paddle.float32
        )
        del full_7, stack_12

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32]) <- (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32)
        combine_16 = [reshape_0, reshape_2, reshape_4, reshape_6]
        del reshape_0, reshape_2, reshape_4, reshape_6

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_16, full_8)
        del combine_16

        # builtin.combine: ([-1x2xf32, -1x2xf32, -1x2xf32, -1x2xf32]) <- (-1x2xf32, -1x2xf32, -1x2xf32, -1x2xf32)
        combine_17 = [reshape_1, reshape_3, reshape_5, reshape_7]
        del reshape_1, reshape_3, reshape_5, reshape_7

        # pd_op.concat: (-1x2xf32) <- ([-1x2xf32, -1x2xf32, -1x2xf32, -1x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_17, full_8)
        del combine_17

        # builtin.combine: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32)
        combine_18 = [
            full_with_tensor_0,
            full_with_tensor_1,
            full_with_tensor_2,
            full_with_tensor_3,
        ]
        del (
            full_with_tensor_0,
            full_with_tensor_1,
            full_with_tensor_2,
            full_with_tensor_3,
        )

        # pd_op.concat: (-1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_18, full_8)
        del combine_18, full_8

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_4, full_int_array_5, [1], [1]
        )

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_0 = paddle._C_ops.add(slice_4, slice_5)
        del slice_4, slice_5

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(add_0, full_9, float("0"), True)
        del add_0

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_3, full_int_array_4, [1], [1]
        )
        del full_int_array_3, full_int_array_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_5, full_int_array_6, [1], [1]
        )
        del concat_0, full_int_array_5, full_int_array_6

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_1 = paddle._C_ops.add(slice_6, slice_7)
        del slice_6, slice_7

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(add_1, full_9, float("0"), True)
        del add_1, full_9

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_19 = [scale_32, scale_33]
        del scale_32, scale_33

        # pd_op.stack: (-1x2xf32) <- ([-1xf32, -1xf32])
        stack_0 = paddle._C_ops.stack(combine_19, -1)
        del combine_19

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = paddle._C_ops.less_than(data_0, full_10)
        del data_0, full_10

        # pd_op.cast: (xi64) <- (xb)
        cast_8 = paddle._C_ops.cast(less_than_0, paddle.int64)
        del less_than_0

        # pd_op.full: (xi64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_8, full_11)
        del cast_8

        # pd_op.cast: (xi64) <- (xb)
        cast_9 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_9, full_11)
        del cast_9, full_11

        # pd_op.share_data_: (16x-1x4xf32) <- (16x-1x4xf32)
        share_data__0 = data_9.detach()
        del data_9

        # pd_op.share_data_: (16x-1x4xf32) <- (16x-1x4xf32)
        share_data__1 = data_10.detach()
        del data_10

        # pd_op.multiply: (16x-1x4xf32) <- (16x-1x4xf32, -1x1xf32)
        multiply_0 = paddle._C_ops.multiply(share_data__1, concat_2)
        del concat_2, share_data__1, slice_0, slice_1, slice_2, slice_3

        return share_data__0, multiply_0, stack_0
