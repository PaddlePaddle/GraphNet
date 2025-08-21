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
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.conv2d: (2x256x240x176xf32) <- (2x96x240x176xf32, 256x96x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_14, full_int_array_1)

        # pd_op.add: (2x256x240x176xf32) <- (2x256x240x176xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.conv2d: (2x256x120x88xf32) <- (2x192x120x88xf32, 256x192x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_1, parameter_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_12, full_int_array_1)

        # pd_op.add: (2x256x120x88xf32) <- (2x256x120x88xf32, 1x256x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_1)

        # pd_op.conv2d: (2x256x60x44xf32) <- (2x384x60x44xf32, 256x384x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_2, parameter_11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_10, full_int_array_1)

        # pd_op.add: (2x256x60x44xf32) <- (2x256x60x44xf32, 1x256x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_2, reshape_2)

        # pd_op.conv2d: (2x256x30x22xf32) <- (2x768x30x22xf32, 256x768x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            data_3, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_8, full_int_array_1)

        # pd_op.add: (2x256x30x22xf32) <- (2x256x30x22xf32, 1x256x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_3, reshape_3)

        # pd_op.nearest_interp: (2x256x60x44xf32) <- (2x256x30x22xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            add_3,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.add: (2x256x60x44xf32) <- (2x256x60x44xf32, 2x256x60x44xf32)
        add_4 = paddle._C_ops.add(add_2, nearest_interp_0)

        # pd_op.nearest_interp: (2x256x120x88xf32) <- (2x256x60x44xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            add_4,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.add: (2x256x120x88xf32) <- (2x256x120x88xf32, 2x256x120x88xf32)
        add_5 = paddle._C_ops.add(add_1, nearest_interp_1)

        # pd_op.nearest_interp: (2x256x240x176xf32) <- (2x256x120x88xf32, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(
            add_5,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.add: (2x256x240x176xf32) <- (2x256x240x176xf32, 2x256x240x176xf32)
        add_6 = paddle._C_ops.add(add_0, nearest_interp_2)

        # pd_op.conv2d: (2x256x240x176xf32) <- (2x256x240x176xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            add_6, parameter_7, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_6, full_int_array_1)

        # pd_op.add: (2x256x240x176xf32) <- (2x256x240x176xf32, 1x256x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_4, reshape_4)

        # pd_op.conv2d: (2x256x120x88xf32) <- (2x256x120x88xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            add_5, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_4, full_int_array_1)

        # pd_op.add: (2x256x120x88xf32) <- (2x256x120x88xf32, 1x256x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_5, reshape_5)

        # pd_op.conv2d: (2x256x60x44xf32) <- (2x256x60x44xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            add_4, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_2, full_int_array_1)

        # pd_op.add: (2x256x60x44xf32) <- (2x256x60x44xf32, 1x256x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_6, reshape_6)

        # pd_op.conv2d: (2x256x30x22xf32) <- (2x256x30x22xf32, 256x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            add_3, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_0, full_int_array_1)

        # pd_op.add: (2x256x30x22xf32) <- (2x256x30x22xf32, 1x256x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_7, reshape_7)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (2x256x15x11xf32) <- (2x256x30x22xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_10,
            full_int_array_0,
            [2, 2],
            [0, 0],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )
        return (
            conv2d_0,
            reshape_0,
            add_0,
            conv2d_1,
            reshape_1,
            add_1,
            conv2d_2,
            reshape_2,
            add_2,
            conv2d_3,
            reshape_3,
            add_3,
            nearest_interp_0,
            add_4,
            nearest_interp_1,
            add_5,
            nearest_interp_2,
            add_6,
            conv2d_4,
            reshape_4,
            conv2d_5,
            reshape_5,
            conv2d_6,
            reshape_6,
            conv2d_7,
            reshape_7,
            full_int_array_0,
            add_7,
            add_8,
            add_9,
            add_10,
            pool2d_0,
        )
