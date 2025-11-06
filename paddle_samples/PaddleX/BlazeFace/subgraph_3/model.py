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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.conv2d: (4x8x80x80xf32) <- (4x96x80x80xf32, 8x96x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_7, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_7

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32) <- (8xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_6, full_int_array_0)
        del parameter_6

        # pd_op.add: (4x8x80x80xf32) <- (4x8x80x80xf32, 1x8x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_6)

        # pd_op.transpose: (4x80x80x8xf32) <- (4x8x80x80xf32)
        transpose_0 = paddle._C_ops.transpose(add_0, [0, 2, 3, 1])
        del add_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [0, -1, 4]

        # pd_op.reshape: (4x12800x4xf32) <- (4x80x80x8xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, full_int_array_1)

        # pd_op.conv2d: (4x4x80x80xf32) <- (4x96x80x80xf32, 4x96x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_0, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_5

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_4, full_int_array_0)
        del parameter_4

        # pd_op.add: (4x4x80x80xf32) <- (4x4x80x80xf32, 1x4x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_7)

        # pd_op.transpose: (4x80x80x4xf32) <- (4x4x80x80xf32)
        transpose_1 = paddle._C_ops.transpose(add_1, [0, 2, 3, 1])
        del add_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, -1, 2]

        # pd_op.reshape: (4x12800x2xf32) <- (4x80x80x4xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_1, full_int_array_2)

        # pd_op.conv2d: (4x24x40x40xf32) <- (4x96x40x40xf32, 24x96x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_1, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_2, full_int_array_0)
        del parameter_2

        # pd_op.add: (4x24x40x40xf32) <- (4x24x40x40xf32, 1x24x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_2, reshape_8)

        # pd_op.transpose: (4x40x40x24xf32) <- (4x24x40x40xf32)
        transpose_2 = paddle._C_ops.transpose(add_2, [0, 2, 3, 1])
        del add_2

        # pd_op.reshape: (4x9600x4xf32) <- (4x40x40x24xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_2, full_int_array_1)
        del full_int_array_1

        # pd_op.conv2d: (4x12x40x40xf32) <- (4x96x40x40xf32, 12x96x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            data_1, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.reshape: (1x12x1x1xf32) <- (12xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_0, full_int_array_0)
        del full_int_array_0, parameter_0

        # pd_op.add: (4x12x40x40xf32) <- (4x12x40x40xf32, 1x12x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_3, reshape_9)

        # pd_op.transpose: (4x40x40x12xf32) <- (4x12x40x40xf32)
        transpose_3 = paddle._C_ops.transpose(add_3, [0, 2, 3, 1])
        del add_3

        # pd_op.reshape: (4x9600x2xf32) <- (4x40x40x12xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_2)
        del full_int_array_2

        # pd_op.prior_box: (80x80x2x4xf32, 80x80x2x4xf32) <- (4x96x80x80xf32, 4x3x640x640xf32)
        prior_box_0, prior_box_1 = (lambda x, f: f(x))(
            paddle._C_ops.prior_box(
                data_0,
                data_2,
                [float("16"), float("24")],
                [],
                [float("1")],
                [float("0.1"), float("0.1"), float("0.2"), float("0.2")],
                False,
                False,
                float("8"),
                float("8"),
                float("0.5"),
                False,
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [-1, 4]

        # pd_op.reshape: (12800x4xf32) <- (80x80x2x4xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(prior_box_0, full_int_array_3)
        del prior_box_0

        # pd_op.prior_box: (40x40x6x4xf32, 40x40x6x4xf32) <- (4x96x40x40xf32, 4x3x640x640xf32)
        prior_box_2, prior_box_3 = (lambda x, f: f(x))(
            paddle._C_ops.prior_box(
                data_1,
                data_2,
                [
                    float("32"),
                    float("48"),
                    float("64"),
                    float("80"),
                    float("96"),
                    float("128"),
                ],
                [],
                [float("1")],
                [float("0.1"), float("0.1"), float("0.2"), float("0.2")],
                False,
                False,
                float("16"),
                float("16"),
                float("0.5"),
                False,
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_1, data_2

        # pd_op.reshape: (9600x4xf32) <- (40x40x6x4xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(prior_box_2, full_int_array_3)
        del (
            conv2d_0,
            conv2d_1,
            conv2d_2,
            conv2d_3,
            full_int_array_3,
            prior_box_2,
            reshape_6,
            reshape_7,
            reshape_8,
            reshape_9,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
        )

        return reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5
