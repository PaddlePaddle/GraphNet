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
        parameter_24,
        parameter_25,
        parameter_26,
        parameter_27,
        parameter_28,
        parameter_29,
        parameter_30,
        parameter_31,
        parameter_32,
        parameter_33,
        parameter_34,
        parameter_35,
        parameter_36,
        parameter_37,
        parameter_38,
        parameter_39,
        parameter_40,
        parameter_41,
        parameter_42,
        parameter_43,
        parameter_44,
        data_0,
    ):
        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x512x-1x-1xf32, 128x512x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_44

        # pd_op.batch_norm_: (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_43,
                parameter_42,
                parameter_41,
                parameter_40,
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
        del conv2d_0, parameter_40, parameter_41, parameter_42, parameter_43

        # pd_op.sigmoid: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(batch_norm__0)

        # pd_op.multiply: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 1x128x-1x-1xf32)
        multiply_1 = paddle._C_ops.multiply(batch_norm__0, sigmoid_0)
        del batch_norm__0, sigmoid_0

        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            multiply_1, parameter_39, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_39

        # pd_op.batch_norm_: (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_38,
                parameter_37,
                parameter_36,
                parameter_35,
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
        del conv2d_1, parameter_35, parameter_36, parameter_37, parameter_38

        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 128x128x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            multiply_1, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_1, parameter_34

        # pd_op.batch_norm_: (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
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
        del conv2d_2, parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.add: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 1x128x-1x-1xf32)
        add_0 = paddle._C_ops.add(batch_norm__6, batch_norm__12)
        del batch_norm__12, batch_norm__6

        # pd_op.silu: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32)
        silu_0 = paddle._C_ops.silu(add_0)
        del add_0

        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            silu_0, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.batch_norm_: (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_28,
                parameter_27,
                parameter_26,
                parameter_25,
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
        del conv2d_3, parameter_25, parameter_26, parameter_27, parameter_28

        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 128x128x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            silu_0, parameter_24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_24, silu_0

        # pd_op.batch_norm_: (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
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
        del conv2d_4, parameter_20, parameter_21, parameter_22, parameter_23

        # pd_op.add: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 1x128x-1x-1xf32)
        add_1 = paddle._C_ops.add(batch_norm__18, batch_norm__24)
        del batch_norm__18, batch_norm__24

        # pd_op.silu: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32)
        silu_1 = paddle._C_ops.silu(add_1)
        del add_1

        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            silu_1, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_19

        # pd_op.batch_norm_: (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_18,
                parameter_17,
                parameter_16,
                parameter_15,
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
        del conv2d_5, parameter_15, parameter_16, parameter_17, parameter_18

        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 128x128x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            silu_1, parameter_14, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_14, silu_1

        # pd_op.batch_norm_: (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
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
        del conv2d_6, parameter_10, parameter_11, parameter_12, parameter_13

        # pd_op.add: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 1x128x-1x-1xf32)
        add_2 = paddle._C_ops.add(batch_norm__30, batch_norm__36)
        del batch_norm__30, batch_norm__36

        # pd_op.silu: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32)
        silu_2 = paddle._C_ops.silu(add_2)
        del add_2

        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x512x-1x-1xf32, 128x512x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            data_0, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_9

        # pd_op.batch_norm_: (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_8,
                parameter_7,
                parameter_6,
                parameter_5,
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
        del conv2d_7, parameter_5, parameter_6, parameter_7, parameter_8

        # pd_op.sigmoid: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(batch_norm__42)

        # pd_op.multiply: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 1x128x-1x-1xf32)
        multiply_2 = paddle._C_ops.multiply(batch_norm__42, sigmoid_1)
        del batch_norm__42, sigmoid_1

        # pd_op.add: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 1x128x-1x-1xf32)
        add_3 = paddle._C_ops.add(silu_2, multiply_2)
        del multiply_2, silu_2

        # pd_op.conv2d: (1x256x-1x-1xf32) <- (1x128x-1x-1xf32, 256x128x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            add_3, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_3, parameter_4

        # pd_op.batch_norm_: (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
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
        del conv2d_8, parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.sigmoid: (1x256x-1x-1xf32) <- (1x256x-1x-1xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(batch_norm__48)

        # pd_op.multiply: (1x256x-1x-1xf32) <- (1x256x-1x-1xf32, 1x256x-1x-1xf32)
        multiply_0 = paddle._C_ops.multiply(batch_norm__48, sigmoid_2)
        del batch_norm__48, sigmoid_2

        return multiply_0
