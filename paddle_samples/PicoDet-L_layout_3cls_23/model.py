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
            [1], float("84"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (84xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="int64")

        # pd_op.cast: (84xf32) <- (84xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (84xf32) <- (84xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (84xf32) <- (84xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, float("0"), True)

        # builtin.combine: ([84xf32, 84xf32]) <- (84xf32, 84xf32)
        combine_0 = [scale_1, scale_1]

        # pd_op.meshgrid: ([84x84xf32, 84x84xf32]) <- ([84xf32, 84xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (84x84xf32, 84x84xf32) <- ([84x84xf32, 84x84xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0

        # pd_op.scale: (84x84xf32) <- (84x84xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(split_1, full_3, float("-20"), True)

        # pd_op.scale: (84x84xf32) <- (84x84xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_0, full_3, float("-20"), True)

        # pd_op.scale: (84x84xf32) <- (84x84xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_3, float("20"), True)

        # pd_op.scale: (84x84xf32) <- (84x84xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_3, float("20"), True)

        # builtin.combine: ([84x84xf32, 84x84xf32, 84x84xf32, 84x84xf32]) <- (84x84xf32, 84x84xf32, 84x84xf32, 84x84xf32)
        combine_1 = [scale_2, scale_3, scale_4, scale_5]

        # pd_op.stack: (84x84x4xf32) <- ([84x84xf32, 84x84xf32, 84x84xf32, 84x84xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)

        # builtin.combine: ([84x84xf32, 84x84xf32]) <- (84x84xf32, 84x84xf32)
        combine_2 = [split_1, split_0]

        # pd_op.stack: (84x84x2xf32) <- ([84x84xf32, 84x84xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (7056x4xf32) <- (84x84x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_1, full_int_array_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (7056x2xf32) <- (84x84x2xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(stack_2, full_int_array_1)

        # pd_op.full: (7056x1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [7056, 1],
            float("8"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("42"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (42xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_6, full_2, dtype="int64")

        # pd_op.cast: (42xf32) <- (42xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (42xf32) <- (42xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_1, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (42xf32) <- (42xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_7, float("0"), True)

        # builtin.combine: ([42xf32, 42xf32]) <- (42xf32, 42xf32)
        combine_3 = [scale_7, scale_7]

        # pd_op.meshgrid: ([42x42xf32, 42x42xf32]) <- ([42xf32, 42xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (42x42xf32, 42x42xf32) <- ([42x42xf32, 42x42xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1

        # pd_op.scale: (42x42xf32) <- (42x42xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(split_3, full_3, float("-40"), True)

        # pd_op.scale: (42x42xf32) <- (42x42xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(split_2, full_3, float("-40"), True)

        # pd_op.scale: (42x42xf32) <- (42x42xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(split_3, full_3, float("40"), True)

        # pd_op.scale: (42x42xf32) <- (42x42xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(split_2, full_3, float("40"), True)

        # builtin.combine: ([42x42xf32, 42x42xf32, 42x42xf32, 42x42xf32]) <- (42x42xf32, 42x42xf32, 42x42xf32, 42x42xf32)
        combine_4 = [scale_8, scale_9, scale_10, scale_11]

        # pd_op.stack: (42x42x4xf32) <- ([42x42xf32, 42x42xf32, 42x42xf32, 42x42xf32])
        stack_3 = paddle._C_ops.stack(combine_4, -1)

        # builtin.combine: ([42x42xf32, 42x42xf32]) <- (42x42xf32, 42x42xf32)
        combine_5 = [split_3, split_2]

        # pd_op.stack: (42x42x2xf32) <- ([42x42xf32, 42x42xf32])
        stack_4 = paddle._C_ops.stack(combine_5, -1)

        # pd_op.reshape: (1764x4xf32) <- (42x42x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_3, full_int_array_0)

        # pd_op.reshape: (1764x2xf32) <- (42x42x2xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(stack_4, full_int_array_1)

        # pd_op.full: (1764x1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1764, 1],
            float("16"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("21"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (21xi64) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_9, full_2, dtype="int64")

        # pd_op.cast: (21xf32) <- (21xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (21xf32) <- (21xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(cast_2, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (21xf32) <- (21xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(scale_12, full_10, float("0"), True)

        # builtin.combine: ([21xf32, 21xf32]) <- (21xf32, 21xf32)
        combine_6 = [scale_13, scale_13]

        # pd_op.meshgrid: ([21x21xf32, 21x21xf32]) <- ([21xf32, 21xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (21x21xf32, 21x21xf32) <- ([21x21xf32, 21x21xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2

        # pd_op.scale: (21x21xf32) <- (21x21xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_5, full_3, float("-80"), True)

        # pd_op.scale: (21x21xf32) <- (21x21xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_4, full_3, float("-80"), True)

        # pd_op.scale: (21x21xf32) <- (21x21xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(split_5, full_3, float("80"), True)

        # pd_op.scale: (21x21xf32) <- (21x21xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(split_4, full_3, float("80"), True)

        # builtin.combine: ([21x21xf32, 21x21xf32, 21x21xf32, 21x21xf32]) <- (21x21xf32, 21x21xf32, 21x21xf32, 21x21xf32)
        combine_7 = [scale_14, scale_15, scale_16, scale_17]

        # pd_op.stack: (21x21x4xf32) <- ([21x21xf32, 21x21xf32, 21x21xf32, 21x21xf32])
        stack_5 = paddle._C_ops.stack(combine_7, -1)

        # builtin.combine: ([21x21xf32, 21x21xf32]) <- (21x21xf32, 21x21xf32)
        combine_8 = [split_5, split_4]

        # pd_op.stack: (21x21x2xf32) <- ([21x21xf32, 21x21xf32])
        stack_6 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.reshape: (441x4xf32) <- (21x21x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(stack_5, full_int_array_0)

        # pd_op.reshape: (441x2xf32) <- (21x21x2xf32, 2xi64)
        reshape_6 = paddle._C_ops.reshape(stack_6, full_int_array_1)

        # pd_op.full: (441x1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [441, 1],
            float("32"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("11"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (11xi64) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_0, full_12, full_2, dtype="int64")

        # pd_op.cast: (11xf32) <- (11xi64)
        cast_3 = paddle._C_ops.cast(arange_3, paddle.float32)

        # pd_op.scale: (11xf32) <- (11xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_3, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("64"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (11xf32) <- (11xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_13, float("0"), True)

        # builtin.combine: ([11xf32, 11xf32]) <- (11xf32, 11xf32)
        combine_9 = [scale_19, scale_19]

        # pd_op.meshgrid: ([11x11xf32, 11x11xf32]) <- ([11xf32, 11xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_9)

        # builtin.split: (11x11xf32, 11x11xf32) <- ([11x11xf32, 11x11xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3

        # pd_op.scale: (11x11xf32) <- (11x11xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(split_7, full_3, float("-160"), True)

        # pd_op.scale: (11x11xf32) <- (11x11xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(split_6, full_3, float("-160"), True)

        # pd_op.scale: (11x11xf32) <- (11x11xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(split_7, full_3, float("160"), True)

        # pd_op.scale: (11x11xf32) <- (11x11xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(split_6, full_3, float("160"), True)

        # builtin.combine: ([11x11xf32, 11x11xf32, 11x11xf32, 11x11xf32]) <- (11x11xf32, 11x11xf32, 11x11xf32, 11x11xf32)
        combine_10 = [scale_20, scale_21, scale_22, scale_23]

        # pd_op.stack: (11x11x4xf32) <- ([11x11xf32, 11x11xf32, 11x11xf32, 11x11xf32])
        stack_7 = paddle._C_ops.stack(combine_10, -1)

        # builtin.combine: ([11x11xf32, 11x11xf32]) <- (11x11xf32, 11x11xf32)
        combine_11 = [split_7, split_6]

        # pd_op.stack: (11x11x2xf32) <- ([11x11xf32, 11x11xf32])
        stack_8 = paddle._C_ops.stack(combine_11, -1)

        # pd_op.reshape: (121x4xf32) <- (11x11x4xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(stack_7, full_int_array_0)

        # pd_op.reshape: (121x2xf32) <- (11x11x2xf32, 2xi64)
        reshape_7 = paddle._C_ops.reshape(stack_8, full_int_array_1)

        # pd_op.full: (121x1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [121, 1],
            float("64"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([7056x4xf32, 1764x4xf32, 441x4xf32, 121x4xf32]) <- (7056x4xf32, 1764x4xf32, 441x4xf32, 121x4xf32)
        combine_12 = [reshape_0, reshape_1, reshape_2, reshape_3]

        # pd_op.concat: (9382x4xf32) <- ([7056x4xf32, 1764x4xf32, 441x4xf32, 121x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_12, full_15)

        # builtin.combine: ([7056x2xf32, 1764x2xf32, 441x2xf32, 121x2xf32]) <- (7056x2xf32, 1764x2xf32, 441x2xf32, 121x2xf32)
        combine_13 = [reshape_4, reshape_5, reshape_6, reshape_7]

        # pd_op.concat: (9382x2xf32) <- ([7056x2xf32, 1764x2xf32, 441x2xf32, 121x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_13, full_15)

        # builtin.combine: ([7056x1xf32, 1764x1xf32, 441x1xf32, 121x1xf32]) <- (7056x1xf32, 1764x1xf32, 441x1xf32, 121x1xf32)
        combine_14 = [full_5, full_8, full_11, full_14]

        # pd_op.concat: (9382x1xf32) <- ([7056x1xf32, 1764x1xf32, 441x1xf32, 121x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_14, full_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (9382xf32) <- (9382x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (9382xf32) <- (9382x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_4, full_int_array_5, [1], [1]
        )

        # pd_op.add: (9382xf32) <- (9382xf32, 9382xf32)
        add_0 = paddle._C_ops.add(slice_0, slice_1)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (9382xf32) <- (9382xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(add_0, full_16, float("0"), True)

        # pd_op.slice: (9382xf32) <- (9382x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (9382xf32) <- (9382x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_5, full_int_array_6, [1], [1]
        )

        # pd_op.add: (9382xf32) <- (9382xf32, 9382xf32)
        add_1 = paddle._C_ops.add(slice_2, slice_3)

        # pd_op.scale: (9382xf32) <- (9382xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(add_1, full_16, float("0"), True)

        # builtin.combine: ([9382xf32, 9382xf32]) <- (9382xf32, 9382xf32)
        combine_15 = [scale_24, scale_25]

        # pd_op.stack: (9382x2xf32) <- ([9382xf32, 9382xf32])
        stack_0 = paddle._C_ops.stack(combine_15, -1)

        # pd_op.share_data_: (2x9382x4xf32) <- (2x9382x4xf32)
        share_data__0 = data_0.detach()

        # pd_op.multiply: (2x9382x4xf32) <- (2x9382x4xf32, 9382x1xf32)
        multiply_0 = paddle._C_ops.multiply(share_data__0, concat_1)
        return (
            concat_0,
            reshape_0,
            reshape_1,
            reshape_2,
            reshape_3,
            multiply_0,
            concat_1,
            stack_0,
        )
