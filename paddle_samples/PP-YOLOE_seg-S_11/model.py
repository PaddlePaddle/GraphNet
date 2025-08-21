import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        parameter_4,
        parameter_5,
        parameter_6,
        parameter_7,
        parameter_8,
        parameter_9,
        parameter_10,
        parameter_11,
        parameter_12,
        parameter_13,
        parameter_14,
        parameter_15,
        parameter_16,
        data_0,
    ):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("20"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (20xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="int64")

        # pd_op.cast: (20xf32) <- (20xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, float("0"), True)

        # builtin.combine: ([20xf32, 20xf32]) <- (20xf32, 20xf32)
        combine_0 = [scale_1, scale_1]

        # pd_op.meshgrid: ([20x20xf32, 20x20xf32]) <- ([20xf32, 20xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (20x20xf32, 20x20xf32) <- ([20x20xf32, 20x20xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0

        # pd_op.scale: (20x20xf32) <- (20x20xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(split_1, full_3, float("-80"), True)

        # pd_op.scale: (20x20xf32) <- (20x20xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_0, full_3, float("-80"), True)

        # pd_op.scale: (20x20xf32) <- (20x20xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_3, float("80"), True)

        # pd_op.scale: (20x20xf32) <- (20x20xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_3, float("80"), True)

        # builtin.combine: ([20x20xf32, 20x20xf32, 20x20xf32, 20x20xf32]) <- (20x20xf32, 20x20xf32, 20x20xf32, 20x20xf32)
        combine_1 = [scale_2, scale_3, scale_4, scale_5]

        # pd_op.stack: (20x20x4xf32) <- ([20x20xf32, 20x20xf32, 20x20xf32, 20x20xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # builtin.combine: ([20x20xf32, 20x20xf32]) <- (20x20xf32, 20x20xf32)
        combine_2 = [split_1, split_0]

        # pd_op.stack: (20x20x2xf32) <- ([20x20xf32, 20x20xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 4]

        # pd_op.reshape: (400x4xf32) <- (20x20x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_0, full_int_array_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 2]

        # pd_op.reshape: (400x2xf32) <- (20x20x2xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(stack_1, full_int_array_2)

        # pd_op.full: (400x1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [400, 1],
            float("32"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("40"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (40xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_6, full_2, dtype="int64")

        # pd_op.cast: (40xf32) <- (40xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_1, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_7, float("0"), True)

        # builtin.combine: ([40xf32, 40xf32]) <- (40xf32, 40xf32)
        combine_3 = [scale_7, scale_7]

        # pd_op.meshgrid: ([40x40xf32, 40x40xf32]) <- ([40xf32, 40xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (40x40xf32, 40x40xf32) <- ([40x40xf32, 40x40xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1

        # pd_op.scale: (40x40xf32) <- (40x40xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(split_3, full_3, float("-40"), True)

        # pd_op.scale: (40x40xf32) <- (40x40xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(split_2, full_3, float("-40"), True)

        # pd_op.scale: (40x40xf32) <- (40x40xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(split_3, full_3, float("40"), True)

        # pd_op.scale: (40x40xf32) <- (40x40xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(split_2, full_3, float("40"), True)

        # builtin.combine: ([40x40xf32, 40x40xf32, 40x40xf32, 40x40xf32]) <- (40x40xf32, 40x40xf32, 40x40xf32, 40x40xf32)
        combine_4 = [scale_8, scale_9, scale_10, scale_11]

        # pd_op.stack: (40x40x4xf32) <- ([40x40xf32, 40x40xf32, 40x40xf32, 40x40xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)

        # builtin.combine: ([40x40xf32, 40x40xf32]) <- (40x40xf32, 40x40xf32)
        combine_5 = [split_3, split_2]

        # pd_op.stack: (40x40x2xf32) <- ([40x40xf32, 40x40xf32])
        stack_3 = paddle._C_ops.stack(combine_5, -1)

        # pd_op.reshape: (1600x4xf32) <- (40x40x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(stack_2, full_int_array_1)

        # pd_op.reshape: (1600x2xf32) <- (40x40x2xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(stack_3, full_int_array_2)

        # pd_op.full: (1600x1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1600, 1],
            float("16"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("80"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (80xi64) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_9, full_2, dtype="int64")

        # pd_op.cast: (80xf32) <- (80xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(cast_2, full_3, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(scale_12, full_10, float("0"), True)

        # builtin.combine: ([80xf32, 80xf32]) <- (80xf32, 80xf32)
        combine_6 = [scale_13, scale_13]

        # pd_op.meshgrid: ([80x80xf32, 80x80xf32]) <- ([80xf32, 80xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (80x80xf32, 80x80xf32) <- ([80x80xf32, 80x80xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2

        # pd_op.scale: (80x80xf32) <- (80x80xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_5, full_3, float("-20"), True)

        # pd_op.scale: (80x80xf32) <- (80x80xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_4, full_3, float("-20"), True)

        # pd_op.scale: (80x80xf32) <- (80x80xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(split_5, full_3, float("20"), True)

        # pd_op.scale: (80x80xf32) <- (80x80xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(split_4, full_3, float("20"), True)

        # builtin.combine: ([80x80xf32, 80x80xf32, 80x80xf32, 80x80xf32]) <- (80x80xf32, 80x80xf32, 80x80xf32, 80x80xf32)
        combine_7 = [scale_14, scale_15, scale_16, scale_17]

        # pd_op.stack: (80x80x4xf32) <- ([80x80xf32, 80x80xf32, 80x80xf32, 80x80xf32])
        stack_4 = paddle._C_ops.stack(combine_7, -1)

        # builtin.combine: ([80x80xf32, 80x80xf32]) <- (80x80xf32, 80x80xf32)
        combine_8 = [split_5, split_4]

        # pd_op.stack: (80x80x2xf32) <- ([80x80xf32, 80x80xf32])
        stack_5 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.reshape: (6400x4xf32) <- (80x80x4xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(stack_4, full_int_array_1)

        # pd_op.reshape: (6400x2xf32) <- (80x80x2xf32, 2xi64)
        reshape_6 = paddle._C_ops.reshape(stack_5, full_int_array_2)

        # pd_op.full: (6400x1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [6400, 1],
            float("8"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([400x4xf32, 1600x4xf32, 6400x4xf32]) <- (400x4xf32, 1600x4xf32, 6400x4xf32)
        combine_9 = [reshape_1, reshape_2, reshape_3]

        # pd_op.concat: (8400x4xf32) <- ([400x4xf32, 1600x4xf32, 6400x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_9, full_12)

        # builtin.combine: ([400x2xf32, 1600x2xf32, 6400x2xf32]) <- (400x2xf32, 1600x2xf32, 6400x2xf32)
        combine_10 = [reshape_4, reshape_5, reshape_6]

        # pd_op.concat: (8400x2xf32) <- ([400x2xf32, 1600x2xf32, 6400x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_10, full_12)

        # builtin.combine: ([400x1xf32, 1600x1xf32, 6400x1xf32]) <- (400x1xf32, 1600x1xf32, 6400x1xf32)
        combine_11 = [full_5, full_8, full_11]

        # pd_op.concat: (8400x1xf32) <- ([400x1xf32, 1600x1xf32, 6400x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_11, full_12)

        # pd_op.conv2d: (1x128x80x80xf32) <- (1x96x80x80xf32, 128x96x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x80x80xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x80x80xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
                parameter_15,
                parameter_14,
                parameter_13,
                parameter_12,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (1x128x80x80xf32) <- (1x128x80x80xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(batch_norm__0)

        # pd_op.multiply: (1x128x80x80xf32) <- (1x128x80x80xf32, 1x128x80x80xf32)
        multiply_0 = paddle._C_ops.multiply(batch_norm__0, sigmoid_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.conv2d_transpose: (1x128x160x160xf32) <- (1x128x80x80xf32, 128x128x2x2xf32, 0xi64)
        conv2d_transpose_0 = paddle._C_ops.conv2d_transpose(
            multiply_0,
            parameter_11,
            [2, 2],
            [0, 0],
            [],
            full_int_array_0,
            "EXPLICIT",
            1,
            [1, 1],
            "NCHW",
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_10, full_int_array_3)

        # pd_op.add: (1x128x160x160xf32) <- (1x128x160x160xf32, 1x128x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_transpose_0, reshape_0)

        # pd_op.conv2d: (1x128x160x160xf32) <- (1x128x160x160xf32, 128x128x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            add_0, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_8,
                parameter_7,
                parameter_6,
                parameter_5,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (1x128x160x160xf32) <- (1x128x160x160xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(batch_norm__6)

        # pd_op.multiply: (1x128x160x160xf32) <- (1x128x160x160xf32, 1x128x160x160xf32)
        multiply_1 = paddle._C_ops.multiply(batch_norm__6, sigmoid_1)

        # pd_op.conv2d: (1x32x160x160xf32) <- (1x128x160x160xf32, 32x128x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            multiply_1, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (1x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_3,
                parameter_2,
                parameter_1,
                parameter_0,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (1x32x160x160xf32) <- (1x32x160x160xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(batch_norm__12)

        # pd_op.multiply: (1x32x160x160xf32) <- (1x32x160x160xf32, 1x32x160x160xf32)
        multiply_2 = paddle._C_ops.multiply(batch_norm__12, sigmoid_2)
        return (
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            sigmoid_0,
            multiply_0,
            full_int_array_0,
            conv2d_transpose_0,
            reshape_0,
            add_0,
            conv2d_1,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
            sigmoid_1,
            multiply_1,
            conv2d_2,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            sigmoid_2,
            multiply_2,
            concat_0,
            concat_1,
            reshape_1,
            reshape_2,
            reshape_3,
            concat_2,
        )
