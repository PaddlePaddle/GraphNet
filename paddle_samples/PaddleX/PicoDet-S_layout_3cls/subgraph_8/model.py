import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("56"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (56xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="int64")
        del full_1

        # pd_op.cast: (56xf32) <- (56xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)
        del arange_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (56xf32) <- (56xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_3, float("0.5"), True)
        del cast_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (56xf32) <- (56xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, float("0"), True)
        del full_4, scale_0

        # builtin.combine: ([56xf32, 56xf32]) <- (56xf32, 56xf32)
        combine_0 = [scale_1, scale_1]
        del scale_1

        # pd_op.meshgrid: ([56x56xf32, 56x56xf32]) <- ([56xf32, 56xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (56x56xf32, 56x56xf32) <- ([56x56xf32, 56x56xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.scale: (56x56xf32) <- (56x56xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(split_1, full_3, float("-20"), True)

        # pd_op.scale: (56x56xf32) <- (56x56xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_0, full_3, float("-20"), True)

        # pd_op.scale: (56x56xf32) <- (56x56xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_3, float("20"), True)

        # pd_op.scale: (56x56xf32) <- (56x56xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_3, float("20"), True)

        # builtin.combine: ([56x56xf32, 56x56xf32, 56x56xf32, 56x56xf32]) <- (56x56xf32, 56x56xf32, 56x56xf32, 56x56xf32)
        combine_1 = [scale_2, scale_3, scale_4, scale_5]
        del scale_2, scale_3, scale_4, scale_5

        # pd_op.stack: (56x56x4xf32) <- ([56x56xf32, 56x56xf32, 56x56xf32, 56x56xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # builtin.combine: ([56x56xf32, 56x56xf32]) <- (56x56xf32, 56x56xf32)
        combine_2 = [split_1, split_0]
        del split_0, split_1

        # pd_op.stack: (56x56x2xf32) <- ([56x56xf32, 56x56xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (3136x4xf32) <- (56x56x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_1, full_int_array_0)
        del stack_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (3136x2xf32) <- (56x56x2xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_2, full_int_array_1)
        del stack_2

        # pd_op.full: (3136x1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [3136, 1],
            float("8"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("28"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (28xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_6, full_2, dtype="int64")
        del full_6

        # pd_op.cast: (28xf32) <- (28xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)
        del arange_1

        # pd_op.scale: (28xf32) <- (28xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_1, full_3, float("0.5"), True)
        del cast_1

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (28xf32) <- (28xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_7, float("0"), True)
        del full_7, scale_6

        # builtin.combine: ([28xf32, 28xf32]) <- (28xf32, 28xf32)
        combine_3 = [scale_7, scale_7]
        del scale_7

        # pd_op.meshgrid: ([28x28xf32, 28x28xf32]) <- ([28xf32, 28xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)
        del combine_3

        # builtin.split: (28x28xf32, 28x28xf32) <- ([28x28xf32, 28x28xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # pd_op.scale: (28x28xf32) <- (28x28xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(split_3, full_3, float("-40"), True)

        # pd_op.scale: (28x28xf32) <- (28x28xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(split_2, full_3, float("-40"), True)

        # pd_op.scale: (28x28xf32) <- (28x28xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(split_3, full_3, float("40"), True)

        # pd_op.scale: (28x28xf32) <- (28x28xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(split_2, full_3, float("40"), True)

        # builtin.combine: ([28x28xf32, 28x28xf32, 28x28xf32, 28x28xf32]) <- (28x28xf32, 28x28xf32, 28x28xf32, 28x28xf32)
        combine_4 = [scale_8, scale_9, scale_10, scale_11]
        del scale_10, scale_11, scale_8, scale_9

        # pd_op.stack: (28x28x4xf32) <- ([28x28xf32, 28x28xf32, 28x28xf32, 28x28xf32])
        stack_3 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        # builtin.combine: ([28x28xf32, 28x28xf32]) <- (28x28xf32, 28x28xf32)
        combine_5 = [split_3, split_2]
        del split_2, split_3

        # pd_op.stack: (28x28x2xf32) <- ([28x28xf32, 28x28xf32])
        stack_4 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.reshape: (784x4xf32) <- (28x28x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(stack_3, full_int_array_0)
        del stack_3

        # pd_op.reshape: (784x2xf32) <- (28x28x2xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(stack_4, full_int_array_1)
        del stack_4

        # pd_op.full: (784x1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [784, 1],
            float("16"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("14"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (14xi64) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_9, full_2, dtype="int64")
        del full_9

        # pd_op.cast: (14xf32) <- (14xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)
        del arange_2

        # pd_op.scale: (14xf32) <- (14xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(cast_2, full_3, float("0.5"), True)
        del cast_2

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (14xf32) <- (14xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(scale_12, full_10, float("0"), True)
        del full_10, scale_12

        # builtin.combine: ([14xf32, 14xf32]) <- (14xf32, 14xf32)
        combine_6 = [scale_13, scale_13]
        del scale_13

        # pd_op.meshgrid: ([14x14xf32, 14x14xf32]) <- ([14xf32, 14xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)
        del combine_6

        # builtin.split: (14x14xf32, 14x14xf32) <- ([14x14xf32, 14x14xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # pd_op.scale: (14x14xf32) <- (14x14xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_5, full_3, float("-80"), True)

        # pd_op.scale: (14x14xf32) <- (14x14xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_4, full_3, float("-80"), True)

        # pd_op.scale: (14x14xf32) <- (14x14xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(split_5, full_3, float("80"), True)

        # pd_op.scale: (14x14xf32) <- (14x14xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(split_4, full_3, float("80"), True)

        # builtin.combine: ([14x14xf32, 14x14xf32, 14x14xf32, 14x14xf32]) <- (14x14xf32, 14x14xf32, 14x14xf32, 14x14xf32)
        combine_7 = [scale_14, scale_15, scale_16, scale_17]
        del scale_14, scale_15, scale_16, scale_17

        # pd_op.stack: (14x14x4xf32) <- ([14x14xf32, 14x14xf32, 14x14xf32, 14x14xf32])
        stack_5 = paddle._C_ops.stack(combine_7, -1)
        del combine_7

        # builtin.combine: ([14x14xf32, 14x14xf32]) <- (14x14xf32, 14x14xf32)
        combine_8 = [split_5, split_4]
        del split_4, split_5

        # pd_op.stack: (14x14x2xf32) <- ([14x14xf32, 14x14xf32])
        stack_6 = paddle._C_ops.stack(combine_8, -1)
        del combine_8

        # pd_op.reshape: (196x4xf32) <- (14x14x4xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(stack_5, full_int_array_0)
        del stack_5

        # pd_op.reshape: (196x2xf32) <- (14x14x2xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(stack_6, full_int_array_1)
        del stack_6

        # pd_op.full: (196x1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [196, 1],
            float("32"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("7"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (7xi64) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_0, full_12, full_2, dtype="int64")
        del full_0, full_12, full_2

        # pd_op.cast: (7xf32) <- (7xi64)
        cast_3 = paddle._C_ops.cast(arange_3, paddle.float32)
        del arange_3

        # pd_op.scale: (7xf32) <- (7xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_3, full_3, float("0.5"), True)
        del cast_3

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("64"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (7xf32) <- (7xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_13, float("0"), True)
        del full_13, scale_18

        # builtin.combine: ([7xf32, 7xf32]) <- (7xf32, 7xf32)
        combine_9 = [scale_19, scale_19]
        del scale_19

        # pd_op.meshgrid: ([7x7xf32, 7x7xf32]) <- ([7xf32, 7xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_9)
        del combine_9

        # builtin.split: (7x7xf32, 7x7xf32) <- ([7x7xf32, 7x7xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3
        del meshgrid_3

        # pd_op.scale: (7x7xf32) <- (7x7xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(split_7, full_3, float("-160"), True)

        # pd_op.scale: (7x7xf32) <- (7x7xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(split_6, full_3, float("-160"), True)

        # pd_op.scale: (7x7xf32) <- (7x7xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(split_7, full_3, float("160"), True)

        # pd_op.scale: (7x7xf32) <- (7x7xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(split_6, full_3, float("160"), True)
        del full_3

        # builtin.combine: ([7x7xf32, 7x7xf32, 7x7xf32, 7x7xf32]) <- (7x7xf32, 7x7xf32, 7x7xf32, 7x7xf32)
        combine_10 = [scale_20, scale_21, scale_22, scale_23]
        del scale_20, scale_21, scale_22, scale_23

        # pd_op.stack: (7x7x4xf32) <- ([7x7xf32, 7x7xf32, 7x7xf32, 7x7xf32])
        stack_7 = paddle._C_ops.stack(combine_10, -1)
        del combine_10

        # builtin.combine: ([7x7xf32, 7x7xf32]) <- (7x7xf32, 7x7xf32)
        combine_11 = [split_7, split_6]
        del split_6, split_7

        # pd_op.stack: (7x7x2xf32) <- ([7x7xf32, 7x7xf32])
        stack_8 = paddle._C_ops.stack(combine_11, -1)
        del combine_11

        # pd_op.reshape: (49x4xf32) <- (7x7x4xf32, 2xi64)
        reshape_6 = paddle._C_ops.reshape(stack_7, full_int_array_0)
        del full_int_array_0, stack_7

        # pd_op.reshape: (49x2xf32) <- (7x7x2xf32, 2xi64)
        reshape_7 = paddle._C_ops.reshape(stack_8, full_int_array_1)
        del full_int_array_1, stack_8

        # pd_op.full: (49x1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [49, 1],
            float("64"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([3136x4xf32, 784x4xf32, 196x4xf32, 49x4xf32]) <- (3136x4xf32, 784x4xf32, 196x4xf32, 49x4xf32)
        combine_12 = [reshape_0, reshape_2, reshape_4, reshape_6]

        # pd_op.concat: (4165x4xf32) <- ([3136x4xf32, 784x4xf32, 196x4xf32, 49x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_12, full_15)
        del combine_12

        # builtin.combine: ([3136x2xf32, 784x2xf32, 196x2xf32, 49x2xf32]) <- (3136x2xf32, 784x2xf32, 196x2xf32, 49x2xf32)
        combine_13 = [reshape_1, reshape_3, reshape_5, reshape_7]
        del reshape_1, reshape_3, reshape_5, reshape_7

        # pd_op.concat: (4165x2xf32) <- ([3136x2xf32, 784x2xf32, 196x2xf32, 49x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_13, full_15)
        del combine_13

        # builtin.combine: ([3136x1xf32, 784x1xf32, 196x1xf32, 49x1xf32]) <- (3136x1xf32, 784x1xf32, 196x1xf32, 49x1xf32)
        combine_14 = [full_5, full_8, full_11, full_14]
        del full_11, full_14, full_5, full_8

        # pd_op.concat: (4165x1xf32) <- ([3136x1xf32, 784x1xf32, 196x1xf32, 49x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_14, full_15)
        del combine_14, full_15

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (4165xf32) <- (4165x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (4165xf32) <- (4165x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_4, full_int_array_5, [1], [1]
        )

        # pd_op.add: (4165xf32) <- (4165xf32, 4165xf32)
        add_0 = paddle._C_ops.add(slice_0, slice_1)
        del slice_0, slice_1

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4165xf32) <- (4165xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(add_0, full_16, float("0"), True)
        del add_0

        # pd_op.slice: (4165xf32) <- (4165x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_3, full_int_array_4, [1], [1]
        )
        del full_int_array_3, full_int_array_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (4165xf32) <- (4165x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_5, full_int_array_6, [1], [1]
        )
        del full_int_array_5, full_int_array_6

        # pd_op.add: (4165xf32) <- (4165xf32, 4165xf32)
        add_1 = paddle._C_ops.add(slice_2, slice_3)
        del slice_2, slice_3

        # pd_op.scale: (4165xf32) <- (4165xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(add_1, full_16, float("0"), True)
        del add_1, full_16

        # builtin.combine: ([4165xf32, 4165xf32]) <- (4165xf32, 4165xf32)
        combine_15 = [scale_24, scale_25]
        del scale_24, scale_25

        # pd_op.stack: (4165x2xf32) <- ([4165xf32, 4165xf32])
        stack_0 = paddle._C_ops.stack(combine_15, -1)
        del combine_15

        # pd_op.share_data_: (2x4165x4xf32) <- (2x4165x4xf32)
        share_data__0 = data_0.detach()
        del data_0

        # pd_op.multiply: (2x4165x4xf32) <- (2x4165x4xf32, 4165x1xf32)
        multiply_0 = paddle._C_ops.multiply(share_data__0, concat_2)
        del (
            concat_0,
            concat_2,
            reshape_0,
            reshape_2,
            reshape_4,
            reshape_6,
            share_data__0,
        )

        return multiply_0, stack_0
