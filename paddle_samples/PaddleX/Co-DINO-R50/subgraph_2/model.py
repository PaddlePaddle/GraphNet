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
        data_11,
    ):
        # pd_op.conv2d: (1x256x214x160xf32) <- (1x256x214x160xf32, 256x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_6, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_12, full_int_array_0)
        del parameter_12

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_0 = reshape_0

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_1 = reshape_0

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_2 = reshape_0

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_3 = reshape_0

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_4 = reshape_0

        # pd_op.add: (1x256x214x160xf32) <- (1x256x214x160xf32, 1x256x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.group_norm: (1x256x214x160xf32, 1x32xf32, 1x32xf32) <- (1x256x214x160xf32, 256xf32, 256xf32)
        group_norm_24, group_norm_0, group_norm_1 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_12, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x214x160xf32) <- (1x256x214x160xf32)
        relu_0 = paddle._C_ops.relu(group_norm_24)

        # pd_op.conv2d: (1x256x214x160xf32) <- (1x256x214x160xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_6, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_6

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_8, full_int_array_0)
        del parameter_8

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_5 = reshape_1

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_6 = reshape_1

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_7 = reshape_1

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_8 = reshape_1

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_9 = reshape_1

        # pd_op.add: (1x256x214x160xf32) <- (1x256x214x160xf32, 1x256x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_1, reshape_1)

        # pd_op.group_norm: (1x256x214x160xf32, 1x32xf32, 1x32xf32) <- (1x256x214x160xf32, 256xf32, 256xf32)
        group_norm_25, group_norm_2, group_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_13, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x214x160xf32) <- (1x256x214x160xf32)
        relu_1 = paddle._C_ops.relu(group_norm_25)

        # pd_op.conv2d: (1x4x214x160xf32) <- (1x256x214x160xf32, 4x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_0, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_4, full_int_array_0)
        del parameter_4

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_10 = reshape_2

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_11 = reshape_2

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_12 = reshape_2

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_13 = reshape_2

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_14 = reshape_2

        # pd_op.add: (1x4x214x160xf32) <- (1x4x214x160xf32, 1x4x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_2, reshape_2)

        # pd_op.conv2d: (1x4x214x160xf32) <- (1x256x214x160xf32, 4x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_1, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_2, full_int_array_0)
        del parameter_2

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_15 = reshape_3

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_16 = reshape_3

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_17 = reshape_3

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_18 = reshape_3

        # pd_op.assign: (1x4x1x1xf32) <- (1x4x1x1xf32)
        assign_19 = reshape_3

        # pd_op.add: (1x4x214x160xf32) <- (1x4x214x160xf32, 1x4x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_3, reshape_3)

        # pd_op.multiply: (1x4x214x160xf32) <- (1x4x214x160xf32, xf32)
        multiply_0 = paddle._C_ops.multiply(add_14, data_0)
        del data_0

        # pd_op.conv2d: (1x1x214x160xf32) <- (1x256x214x160xf32, 1x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_1, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_0, full_int_array_0)
        del full_int_array_0, parameter_0

        # pd_op.assign: (1x1x1x1xf32) <- (1x1x1x1xf32)
        assign_20 = reshape_4

        # pd_op.assign: (1x1x1x1xf32) <- (1x1x1x1xf32)
        assign_21 = reshape_4

        # pd_op.assign: (1x1x1x1xf32) <- (1x1x1x1xf32)
        assign_22 = reshape_4

        # pd_op.assign: (1x1x1x1xf32) <- (1x1x1x1xf32)
        assign_23 = reshape_4

        # pd_op.assign: (1x1x1x1xf32) <- (1x1x1x1xf32)
        assign_24 = reshape_4

        # pd_op.add: (1x1x214x160xf32) <- (1x1x214x160xf32, 1x1x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_4, reshape_4)

        # pd_op.conv2d: (1x256x107x80xf32) <- (1x256x107x80xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            data_7, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x256x107x80xf32) <- (1x256x107x80xf32, 1x256x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_5, reshape_0)

        # pd_op.group_norm: (1x256x107x80xf32, 1x32xf32, 1x32xf32) <- (1x256x107x80xf32, 256xf32, 256xf32)
        group_norm_26, group_norm_4, group_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_15, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x107x80xf32) <- (1x256x107x80xf32)
        relu_2 = paddle._C_ops.relu(group_norm_26)

        # pd_op.conv2d: (1x256x107x80xf32) <- (1x256x107x80xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            data_7, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_7

        # pd_op.add: (1x256x107x80xf32) <- (1x256x107x80xf32, 1x256x1x1xf32)
        add_16 = paddle._C_ops.add(conv2d_6, reshape_1)

        # pd_op.group_norm: (1x256x107x80xf32, 1x32xf32, 1x32xf32) <- (1x256x107x80xf32, 256xf32, 256xf32)
        group_norm_27, group_norm_6, group_norm_7 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_16, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x107x80xf32) <- (1x256x107x80xf32)
        relu_3 = paddle._C_ops.relu(group_norm_27)

        # pd_op.conv2d: (1x4x107x80xf32) <- (1x256x107x80xf32, 4x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_2, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x107x80xf32) <- (1x4x107x80xf32, 1x4x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_7, reshape_2)

        # pd_op.conv2d: (1x4x107x80xf32) <- (1x256x107x80xf32, 4x256x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_3, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x107x80xf32) <- (1x4x107x80xf32, 1x4x1x1xf32)
        add_17 = paddle._C_ops.add(conv2d_8, reshape_3)

        # pd_op.multiply: (1x4x107x80xf32) <- (1x4x107x80xf32, xf32)
        multiply_1 = paddle._C_ops.multiply(add_17, data_1)
        del data_1

        # pd_op.conv2d: (1x1x107x80xf32) <- (1x256x107x80xf32, 1x256x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_3, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x1x107x80xf32) <- (1x1x107x80xf32, 1x1x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_9, reshape_4)

        # pd_op.conv2d: (1x256x54x40xf32) <- (1x256x54x40xf32, 256x256x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            data_8, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x256x54x40xf32) <- (1x256x54x40xf32, 1x256x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_10, reshape_0)

        # pd_op.group_norm: (1x256x54x40xf32, 1x32xf32, 1x32xf32) <- (1x256x54x40xf32, 256xf32, 256xf32)
        group_norm_28, group_norm_8, group_norm_9 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_18, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x54x40xf32) <- (1x256x54x40xf32)
        relu_4 = paddle._C_ops.relu(group_norm_28)

        # pd_op.conv2d: (1x256x54x40xf32) <- (1x256x54x40xf32, 256x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            data_8, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_8

        # pd_op.add: (1x256x54x40xf32) <- (1x256x54x40xf32, 1x256x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_11, reshape_1)

        # pd_op.group_norm: (1x256x54x40xf32, 1x32xf32, 1x32xf32) <- (1x256x54x40xf32, 256xf32, 256xf32)
        group_norm_29, group_norm_10, group_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_19, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x54x40xf32) <- (1x256x54x40xf32)
        relu_5 = paddle._C_ops.relu(group_norm_29)

        # pd_op.conv2d: (1x4x54x40xf32) <- (1x256x54x40xf32, 4x256x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_4, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x54x40xf32) <- (1x4x54x40xf32, 1x4x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_12, reshape_2)

        # pd_op.conv2d: (1x4x54x40xf32) <- (1x256x54x40xf32, 4x256x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_5, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x54x40xf32) <- (1x4x54x40xf32, 1x4x1x1xf32)
        add_20 = paddle._C_ops.add(conv2d_13, reshape_3)

        # pd_op.multiply: (1x4x54x40xf32) <- (1x4x54x40xf32, xf32)
        multiply_2 = paddle._C_ops.multiply(add_20, data_2)
        del data_2

        # pd_op.conv2d: (1x1x54x40xf32) <- (1x256x54x40xf32, 1x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_5, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x1x54x40xf32) <- (1x1x54x40xf32, 1x1x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_14, reshape_4)

        # pd_op.conv2d: (1x256x27x20xf32) <- (1x256x27x20xf32, 256x256x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            data_9, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x256x27x20xf32) <- (1x256x27x20xf32, 1x256x1x1xf32)
        add_21 = paddle._C_ops.add(conv2d_15, reshape_0)

        # pd_op.group_norm: (1x256x27x20xf32, 1x32xf32, 1x32xf32) <- (1x256x27x20xf32, 256xf32, 256xf32)
        group_norm_30, group_norm_12, group_norm_13 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_21, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x27x20xf32) <- (1x256x27x20xf32)
        relu_6 = paddle._C_ops.relu(group_norm_30)

        # pd_op.conv2d: (1x256x27x20xf32) <- (1x256x27x20xf32, 256x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            data_9, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_9

        # pd_op.add: (1x256x27x20xf32) <- (1x256x27x20xf32, 1x256x1x1xf32)
        add_22 = paddle._C_ops.add(conv2d_16, reshape_1)

        # pd_op.group_norm: (1x256x27x20xf32, 1x32xf32, 1x32xf32) <- (1x256x27x20xf32, 256xf32, 256xf32)
        group_norm_31, group_norm_14, group_norm_15 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_22, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x27x20xf32) <- (1x256x27x20xf32)
        relu_7 = paddle._C_ops.relu(group_norm_31)

        # pd_op.conv2d: (1x4x27x20xf32) <- (1x256x27x20xf32, 4x256x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_6, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x27x20xf32) <- (1x4x27x20xf32, 1x4x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_17, reshape_2)

        # pd_op.conv2d: (1x4x27x20xf32) <- (1x256x27x20xf32, 4x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_7, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x27x20xf32) <- (1x4x27x20xf32, 1x4x1x1xf32)
        add_23 = paddle._C_ops.add(conv2d_18, reshape_3)

        # pd_op.multiply: (1x4x27x20xf32) <- (1x4x27x20xf32, xf32)
        multiply_3 = paddle._C_ops.multiply(add_23, data_3)
        del data_3

        # pd_op.conv2d: (1x1x27x20xf32) <- (1x256x27x20xf32, 1x256x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_7, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x1x27x20xf32) <- (1x1x27x20xf32, 1x1x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_19, reshape_4)

        # pd_op.conv2d: (1x256x14x10xf32) <- (1x256x14x10xf32, 256x256x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            data_10, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x256x14x10xf32) <- (1x256x14x10xf32, 1x256x1x1xf32)
        add_24 = paddle._C_ops.add(conv2d_20, reshape_0)

        # pd_op.group_norm: (1x256x14x10xf32, 1x32xf32, 1x32xf32) <- (1x256x14x10xf32, 256xf32, 256xf32)
        group_norm_32, group_norm_16, group_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_24, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x14x10xf32) <- (1x256x14x10xf32)
        relu_8 = paddle._C_ops.relu(group_norm_32)

        # pd_op.conv2d: (1x256x14x10xf32) <- (1x256x14x10xf32, 256x256x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            data_10, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_10

        # pd_op.add: (1x256x14x10xf32) <- (1x256x14x10xf32, 1x256x1x1xf32)
        add_25 = paddle._C_ops.add(conv2d_21, reshape_1)

        # pd_op.group_norm: (1x256x14x10xf32, 1x32xf32, 1x32xf32) <- (1x256x14x10xf32, 256xf32, 256xf32)
        group_norm_33, group_norm_18, group_norm_19 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_25, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.relu: (1x256x14x10xf32) <- (1x256x14x10xf32)
        relu_9 = paddle._C_ops.relu(group_norm_33)

        # pd_op.conv2d: (1x4x14x10xf32) <- (1x256x14x10xf32, 4x256x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_8, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x14x10xf32) <- (1x4x14x10xf32, 1x4x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_22, reshape_2)

        # pd_op.conv2d: (1x4x14x10xf32) <- (1x256x14x10xf32, 4x256x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_9, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x14x10xf32) <- (1x4x14x10xf32, 1x4x1x1xf32)
        add_26 = paddle._C_ops.add(conv2d_23, reshape_3)

        # pd_op.multiply: (1x4x14x10xf32) <- (1x4x14x10xf32, xf32)
        multiply_4 = paddle._C_ops.multiply(add_26, data_4)
        del data_4

        # pd_op.conv2d: (1x1x14x10xf32) <- (1x256x14x10xf32, 1x256x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            relu_9, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x1x14x10xf32) <- (1x1x14x10xf32, 1x1x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_24, reshape_4)

        # pd_op.conv2d: (1x256x7x5xf32) <- (1x256x7x5xf32, 256x256x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            data_11, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_13

        # pd_op.add: (1x256x7x5xf32) <- (1x256x7x5xf32, 1x256x1x1xf32)
        add_27 = paddle._C_ops.add(conv2d_25, reshape_0)

        # pd_op.group_norm: (1x256x7x5xf32, 1x32xf32, 1x32xf32) <- (1x256x7x5xf32, 256xf32, 256xf32)
        group_norm_34, group_norm_20, group_norm_21 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_27, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_10, parameter_11

        # pd_op.relu: (1x256x7x5xf32) <- (1x256x7x5xf32)
        relu_10 = paddle._C_ops.relu(group_norm_34)

        # pd_op.conv2d: (1x256x7x5xf32) <- (1x256x7x5xf32, 256x256x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            data_11, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_11, parameter_9

        # pd_op.add: (1x256x7x5xf32) <- (1x256x7x5xf32, 1x256x1x1xf32)
        add_28 = paddle._C_ops.add(conv2d_26, reshape_1)

        # pd_op.group_norm: (1x256x7x5xf32, 1x32xf32, 1x32xf32) <- (1x256x7x5xf32, 256xf32, 256xf32)
        group_norm_35, group_norm_22, group_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_28, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_6, parameter_7

        # pd_op.relu: (1x256x7x5xf32) <- (1x256x7x5xf32)
        relu_11 = paddle._C_ops.relu(group_norm_35)

        # pd_op.conv2d: (1x4x7x5xf32) <- (1x256x7x5xf32, 4x256x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            relu_10, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_5

        # pd_op.add: (1x4x7x5xf32) <- (1x4x7x5xf32, 1x4x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_27, reshape_2)

        # pd_op.conv2d: (1x4x7x5xf32) <- (1x256x7x5xf32, 4x256x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            relu_11, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.add: (1x4x7x5xf32) <- (1x4x7x5xf32, 1x4x1x1xf32)
        add_29 = paddle._C_ops.add(conv2d_28, reshape_3)

        # pd_op.multiply: (1x4x7x5xf32) <- (1x4x7x5xf32, xf32)
        multiply_5 = paddle._C_ops.multiply(add_29, data_5)
        del data_5

        # pd_op.conv2d: (1x1x7x5xf32) <- (1x256x7x5xf32, 1x256x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_11, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.add: (1x1x7x5xf32) <- (1x1x7x5xf32, 1x1x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_29, reshape_4)
        del (
            add_12,
            add_13,
            add_14,
            add_15,
            add_16,
            add_17,
            add_18,
            add_19,
            add_20,
            add_21,
            add_22,
            add_23,
            add_24,
            add_25,
            add_26,
            add_27,
            add_28,
            add_29,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
            assign_23,
            assign_24,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_12,
            conv2d_13,
            conv2d_14,
            conv2d_15,
            conv2d_16,
            conv2d_17,
            conv2d_18,
            conv2d_19,
            conv2d_2,
            conv2d_20,
            conv2d_21,
            conv2d_22,
            conv2d_23,
            conv2d_24,
            conv2d_25,
            conv2d_26,
            conv2d_27,
            conv2d_28,
            conv2d_29,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            group_norm_24,
            group_norm_25,
            group_norm_26,
            group_norm_27,
            group_norm_28,
            group_norm_29,
            group_norm_30,
            group_norm_31,
            group_norm_32,
            group_norm_33,
            group_norm_34,
            group_norm_35,
            relu_0,
            relu_1,
            relu_10,
            relu_11,
            relu_2,
            relu_3,
            relu_4,
            relu_5,
            relu_6,
            relu_7,
            relu_8,
            relu_9,
            reshape_0,
            reshape_1,
            reshape_2,
            reshape_3,
            reshape_4,
        )

        return (
            group_norm_0,
            group_norm_1,
            group_norm_2,
            group_norm_3,
            group_norm_4,
            group_norm_5,
            group_norm_6,
            group_norm_7,
            group_norm_8,
            group_norm_9,
            group_norm_10,
            group_norm_11,
            group_norm_12,
            group_norm_13,
            group_norm_14,
            group_norm_15,
            group_norm_16,
            group_norm_17,
            group_norm_18,
            group_norm_19,
            group_norm_20,
            group_norm_21,
            group_norm_22,
            group_norm_23,
            add_0,
            add_1,
            add_2,
            add_3,
            add_4,
            add_5,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            add_6,
            add_7,
            add_8,
            add_9,
            add_10,
            add_11,
        )
