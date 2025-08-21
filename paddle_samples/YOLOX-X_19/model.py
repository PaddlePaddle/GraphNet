import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([2x6069x2xf32, 2x6069x2xf32]) <- (2x6069x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_2, 2, full_0)

        # builtin.split: (2x6069x2xf32, 2x6069x2xf32) <- ([2x6069x2xf32, 2x6069x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0

        # pd_op.divide: (6069x2xf32) <- (6069x2xf32, 6069x1xf32)
        divide_0 = paddle._C_ops.divide(data_0, data_1)

        # pd_op.add: (2x6069x2xf32) <- (2x6069x2xf32, 6069x2xf32)
        add_0 = paddle._C_ops.add(split_0, divide_0)

        # pd_op.exp: (2x6069x2xf32) <- (2x6069x2xf32)
        exp_0 = paddle._C_ops.exp(split_1)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x6069x2xf32) <- (2x6069x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(exp_0, full_1, float("0"), True)

        # pd_op.subtract: (2x6069x2xf32) <- (2x6069x2xf32, 2x6069x2xf32)
        subtract_0 = paddle._C_ops.subtract(add_0, scale_0)

        # pd_op.add: (2x6069x2xf32) <- (2x6069x2xf32, 2x6069x2xf32)
        add_1 = paddle._C_ops.add(add_0, scale_0)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x6069x2xf32, 2x6069x2xf32]) <- (2x6069x2xf32, 2x6069x2xf32)
        combine_0 = [subtract_0, add_1]

        # pd_op.concat: (2x6069x4xf32) <- ([2x6069x2xf32, 2x6069x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("68"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (68xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_3, full_4, full_5, dtype="int64")

        # pd_op.cast: (68xf32) <- (68xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (68xf32) <- (68xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(cast_0, full_6, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (68xf32) <- (68xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_7, float("0"), True)

        # builtin.combine: ([68xf32, 68xf32]) <- (68xf32, 68xf32)
        combine_1 = [scale_2, scale_2]

        # pd_op.meshgrid: ([68x68xf32, 68x68xf32]) <- ([68xf32, 68xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_1)

        # builtin.split: (68x68xf32, 68x68xf32) <- ([68x68xf32, 68x68xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_0

        # builtin.combine: ([68x68xf32, 68x68xf32]) <- (68x68xf32, 68x68xf32)
        combine_2 = [split_3, split_2]

        # pd_op.stack: (68x68x2xf32) <- ([68x68xf32, 68x68xf32])
        stack_0 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 2]

        # pd_op.reshape: (4624x2xf32) <- (68x68x2xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_0)

        # pd_op.full: (4624x1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [4624, 1],
            float("8"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("34"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (34xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_3, full_9, full_5, dtype="int64")

        # pd_op.cast: (34xf32) <- (34xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (34xf32) <- (34xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(cast_1, full_6, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (34xf32) <- (34xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(scale_3, full_10, float("0"), True)

        # builtin.combine: ([34xf32, 34xf32]) <- (34xf32, 34xf32)
        combine_3 = [scale_4, scale_4]

        # pd_op.meshgrid: ([34x34xf32, 34x34xf32]) <- ([34xf32, 34xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (34x34xf32, 34x34xf32) <- ([34x34xf32, 34x34xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_1

        # builtin.combine: ([34x34xf32, 34x34xf32]) <- (34x34xf32, 34x34xf32)
        combine_4 = [split_5, split_4]

        # pd_op.stack: (34x34x2xf32) <- ([34x34xf32, 34x34xf32])
        stack_1 = paddle._C_ops.stack(combine_4, -1)

        # pd_op.reshape: (1156x2xf32) <- (34x34x2xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_1, full_int_array_0)

        # pd_op.full: (1156x1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1156, 1],
            float("16"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("17"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (17xi64) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_3, full_12, full_5, dtype="int64")

        # pd_op.cast: (17xf32) <- (17xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (17xf32) <- (17xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(cast_2, full_6, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (17xf32) <- (17xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(scale_5, full_13, float("0"), True)

        # builtin.combine: ([17xf32, 17xf32]) <- (17xf32, 17xf32)
        combine_5 = [scale_6, scale_6]

        # pd_op.meshgrid: ([17x17xf32, 17x17xf32]) <- ([17xf32, 17xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_5)

        # builtin.split: (17x17xf32, 17x17xf32) <- ([17x17xf32, 17x17xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_2

        # builtin.combine: ([17x17xf32, 17x17xf32]) <- (17x17xf32, 17x17xf32)
        combine_6 = [split_7, split_6]

        # pd_op.stack: (17x17x2xf32) <- ([17x17xf32, 17x17xf32])
        stack_2 = paddle._C_ops.stack(combine_6, -1)

        # pd_op.reshape: (289x2xf32) <- (17x17x2xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(stack_2, full_int_array_0)

        # pd_op.full: (289x1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [289, 1],
            float("32"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([4624x2xf32, 1156x2xf32, 289x2xf32]) <- (4624x2xf32, 1156x2xf32, 289x2xf32)
        combine_7 = [reshape_0, reshape_1, reshape_2]

        # pd_op.concat: (6069x2xf32) <- ([4624x2xf32, 1156x2xf32, 289x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_7, full_15)

        # builtin.combine: ([4624x1xf32, 1156x1xf32, 289x1xf32]) <- (4624x1xf32, 1156x1xf32, 289x1xf32)
        combine_8 = [full_8, full_11, full_14]

        # pd_op.concat: (6069x1xf32) <- ([4624x1xf32, 1156x1xf32, 289x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_8, full_15)
        return (
            full_0,
            split_0,
            divide_0,
            add_0,
            exp_0,
            full_1,
            scale_0,
            subtract_0,
            add_1,
            full_2,
            concat_0,
            concat_1,
            concat_2,
            reshape_0,
            reshape_1,
            reshape_2,
        )
