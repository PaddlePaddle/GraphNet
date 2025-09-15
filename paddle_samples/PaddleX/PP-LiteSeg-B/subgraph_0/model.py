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
        parameter_17,
        parameter_18,
        parameter_19,
        parameter_20,
        parameter_21,
        parameter_22,
        parameter_23,
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            data_2,
            full_int_array_0,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )
        del full_int_array_0

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x1024x1x1xf32, 128x1024x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            pool2d_0, parameter_23, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_23, pool2d_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_22, full_int_array_1)
        del full_int_array_1, parameter_22

        # pd_op.add: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_0,
                parameter_21,
                parameter_20,
                parameter_19,
                parameter_18,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del add_0, parameter_18, parameter_19, parameter_20, parameter_21

        # pd_op.relu: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [data_0, data_1]
        del data_0, data_1

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x1x1xf32, None, [xi64, xi64], None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            relu_1, None, combine_0, None, "NCHW", -1, -1, -1, [], "bilinear", False, 0
        )
        del relu_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (-1x1024x2x2xf32) <- (-1x1024x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            data_2,
            full_int_array_2,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )
        del full_int_array_2

        # pd_op.conv2d: (-1x128x2x2xf32) <- (-1x1024x2x2xf32, 128x1024x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            pool2d_1, parameter_17, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_17, pool2d_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_16, full_int_array_3)
        del full_int_array_3, parameter_16

        # pd_op.add: (-1x128x2x2xf32) <- (-1x128x2x2xf32, 1x128x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_1)
        del conv2d_1, reshape_1

        # pd_op.batch_norm_: (-1x128x2x2xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x2x2xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_1,
                parameter_15,
                parameter_14,
                parameter_13,
                parameter_12,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del add_1, parameter_12, parameter_13, parameter_14, parameter_15

        # pd_op.relu: (-1x128x2x2xf32) <- (-1x128x2x2xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x2x2xf32, None, [xi64, xi64], None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(
            relu_2, None, combine_0, None, "NCHW", -1, -1, -1, [], "bilinear", False, 0
        )
        del relu_2

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_2 = paddle._C_ops.add(bilinear_interp_0, bilinear_interp_1)
        del bilinear_interp_0, bilinear_interp_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [4, 4]

        # pd_op.pool2d: (-1x1024x4x4xf32) <- (-1x1024x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            data_2,
            full_int_array_4,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )
        del data_2, full_int_array_4

        # pd_op.conv2d: (-1x128x4x4xf32) <- (-1x1024x4x4xf32, 128x1024x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            pool2d_2, parameter_11, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_11, pool2d_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_10, full_int_array_5)
        del full_int_array_5, parameter_10

        # pd_op.add: (-1x128x4x4xf32) <- (-1x128x4x4xf32, 1x128x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_2, reshape_2)
        del conv2d_2, reshape_2

        # pd_op.batch_norm_: (-1x128x4x4xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x4x4xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_3,
                parameter_9,
                parameter_8,
                parameter_7,
                parameter_6,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del add_3, parameter_6, parameter_7, parameter_8, parameter_9

        # pd_op.relu: (-1x128x4x4xf32) <- (-1x128x4x4xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x4x4xf32, None, [xi64, xi64], None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(
            relu_3, None, combine_0, None, "NCHW", -1, -1, -1, [], "bilinear", False, 0
        )
        del combine_0, relu_3

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_4 = paddle._C_ops.add(add_2, bilinear_interp_2)
        del add_2, bilinear_interp_2

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            add_4, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_4, parameter_5

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_4, full_int_array_6)
        del full_int_array_6, parameter_4

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_3, reshape_3)
        del conv2d_3, reshape_3

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_5,
                parameter_3,
                parameter_2,
                parameter_1,
                parameter_0,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del add_5, parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        return relu_0
