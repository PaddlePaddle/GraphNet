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
        parameter_45,
        parameter_46,
        parameter_47,
        parameter_48,
        parameter_49,
        parameter_50,
        parameter_51,
        parameter_52,
        parameter_53,
        parameter_54,
        parameter_55,
        parameter_56,
        parameter_57,
        parameter_58,
        parameter_59,
        parameter_60,
        parameter_61,
        parameter_62,
        parameter_63,
        parameter_64,
        parameter_65,
        parameter_66,
        parameter_67,
        parameter_68,
        parameter_69,
        parameter_70,
        parameter_71,
        parameter_72,
        parameter_73,
        parameter_74,
        parameter_75,
        parameter_76,
        parameter_77,
        parameter_78,
        parameter_79,
        parameter_80,
        parameter_81,
        parameter_82,
        parameter_83,
        parameter_84,
        parameter_85,
        parameter_86,
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.conv2d: (4x256x80x80xf32) <- (4x256x80x80xf32, 256x256x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_85,
                parameter_84,
                parameter_83,
                parameter_82,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x80x80xf32) <- (4x256x80x80xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(batch_norm__0)

        # pd_op.multiply: (4x256x80x80xf32) <- (4x256x80x80xf32, 4x256x80x80xf32)
        multiply_0 = paddle._C_ops.multiply(batch_norm__0, sigmoid_0)

        # pd_op.conv2d: (4x256x80x80xf32) <- (4x256x80x80xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            multiply_0, parameter_81, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_80,
                parameter_79,
                parameter_78,
                parameter_77,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x80x80xf32) <- (4x256x80x80xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(batch_norm__6)

        # pd_op.multiply: (4x256x80x80xf32) <- (4x256x80x80xf32, 4x256x80x80xf32)
        multiply_1 = paddle._C_ops.multiply(batch_norm__6, sigmoid_1)

        # pd_op.conv2d: (4x256x80x80xf32) <- (4x256x80x80xf32, 256x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            multiply_1, parameter_76, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x80x80xf32) <- (4x256x80x80xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(batch_norm__12)

        # pd_op.multiply: (4x256x80x80xf32) <- (4x256x80x80xf32, 4x256x80x80xf32)
        multiply_2 = paddle._C_ops.multiply(batch_norm__12, sigmoid_2)

        # pd_op.conv2d: (4x4x80x80xf32) <- (4x256x80x80xf32, 4x256x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            multiply_2, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_70, full_int_array_0)

        # pd_op.add: (4x4x80x80xf32) <- (4x4x80x80xf32, 1x4x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_3, reshape_0)

        # pd_op.conv2d: (4x256x80x80xf32) <- (4x256x80x80xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            multiply_0, parameter_69, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x80x80xf32) <- (4x256x80x80xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(batch_norm__18)

        # pd_op.multiply: (4x256x80x80xf32) <- (4x256x80x80xf32, 4x256x80x80xf32)
        multiply_3 = paddle._C_ops.multiply(batch_norm__18, sigmoid_3)

        # pd_op.conv2d: (4x256x80x80xf32) <- (4x256x80x80xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            multiply_3, parameter_64, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_63,
                parameter_62,
                parameter_61,
                parameter_60,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x80x80xf32) <- (4x256x80x80xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(batch_norm__24)

        # pd_op.multiply: (4x256x80x80xf32) <- (4x256x80x80xf32, 4x256x80x80xf32)
        multiply_4 = paddle._C_ops.multiply(batch_norm__24, sigmoid_4)

        # pd_op.conv2d: (4x5x80x80xf32) <- (4x256x80x80xf32, 5x256x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            multiply_4, parameter_59, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x5x1x1xf32) <- (5xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_58, full_int_array_0)

        # pd_op.add: (4x5x80x80xf32) <- (4x5x80x80xf32, 1x5x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_6, reshape_1)

        # pd_op.sigmoid: (4x4x80x80xf32) <- (4x4x80x80xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(add_2)

        # pd_op.flatten: (4x4x6400xf32) <- (4x4x80x80xf32)
        flatten_0 = paddle._C_ops.flatten(sigmoid_5, 2, 3)

        # pd_op.transpose: (4x6400x4xf32) <- (4x4x6400xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [4, 1]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.split: ([4x4x80x80xf32, 4x1x80x80xf32]) <- (4x5x80x80xf32, 2xi64, 1xi32)
        split_4 = paddle._C_ops.split(add_3, full_int_array_1, full_0)

        # builtin.split: (4x4x80x80xf32, 4x1x80x80xf32) <- ([4x4x80x80xf32, 4x1x80x80xf32])
        (
            split_0,
            split_5,
        ) = split_4

        # pd_op.flatten: (4x4x6400xf32) <- (4x4x80x80xf32)
        flatten_1 = paddle._C_ops.flatten(split_0, 2, 3)

        # pd_op.transpose: (4x6400x4xf32) <- (4x4x6400xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])

        # pd_op.sigmoid: (4x1x80x80xf32) <- (4x1x80x80xf32)
        sigmoid_6 = paddle._C_ops.sigmoid(split_5)

        # pd_op.flatten: (4x1x6400xf32) <- (4x1x80x80xf32)
        flatten_2 = paddle._C_ops.flatten(sigmoid_6, 2, 3)

        # pd_op.transpose: (4x6400x1xf32) <- (4x1x6400xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])

        # pd_op.conv2d: (4x256x40x40xf32) <- (4x512x40x40xf32, 256x512x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            data_1, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_56,
                parameter_55,
                parameter_54,
                parameter_53,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x40x40xf32) <- (4x256x40x40xf32)
        sigmoid_7 = paddle._C_ops.sigmoid(batch_norm__30)

        # pd_op.multiply: (4x256x40x40xf32) <- (4x256x40x40xf32, 4x256x40x40xf32)
        multiply_5 = paddle._C_ops.multiply(batch_norm__30, sigmoid_7)

        # pd_op.conv2d: (4x256x40x40xf32) <- (4x256x40x40xf32, 256x256x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            multiply_5, parameter_52, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_51,
                parameter_50,
                parameter_49,
                parameter_48,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x40x40xf32) <- (4x256x40x40xf32)
        sigmoid_8 = paddle._C_ops.sigmoid(batch_norm__36)

        # pd_op.multiply: (4x256x40x40xf32) <- (4x256x40x40xf32, 4x256x40x40xf32)
        multiply_6 = paddle._C_ops.multiply(batch_norm__36, sigmoid_8)

        # pd_op.conv2d: (4x256x40x40xf32) <- (4x256x40x40xf32, 256x256x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            multiply_6, parameter_47, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_46,
                parameter_45,
                parameter_44,
                parameter_43,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x40x40xf32) <- (4x256x40x40xf32)
        sigmoid_9 = paddle._C_ops.sigmoid(batch_norm__42)

        # pd_op.multiply: (4x256x40x40xf32) <- (4x256x40x40xf32, 4x256x40x40xf32)
        multiply_7 = paddle._C_ops.multiply(batch_norm__42, sigmoid_9)

        # pd_op.conv2d: (4x4x40x40xf32) <- (4x256x40x40xf32, 4x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            multiply_7, parameter_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_41, full_int_array_0)

        # pd_op.add: (4x4x40x40xf32) <- (4x4x40x40xf32, 1x4x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_10, reshape_2)

        # pd_op.conv2d: (4x256x40x40xf32) <- (4x256x40x40xf32, 256x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            multiply_5, parameter_40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_39,
                parameter_38,
                parameter_37,
                parameter_36,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x40x40xf32) <- (4x256x40x40xf32)
        sigmoid_10 = paddle._C_ops.sigmoid(batch_norm__48)

        # pd_op.multiply: (4x256x40x40xf32) <- (4x256x40x40xf32, 4x256x40x40xf32)
        multiply_8 = paddle._C_ops.multiply(batch_norm__48, sigmoid_10)

        # pd_op.conv2d: (4x256x40x40xf32) <- (4x256x40x40xf32, 256x256x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            multiply_8, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_34,
                parameter_33,
                parameter_32,
                parameter_31,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x40x40xf32) <- (4x256x40x40xf32)
        sigmoid_11 = paddle._C_ops.sigmoid(batch_norm__54)

        # pd_op.multiply: (4x256x40x40xf32) <- (4x256x40x40xf32, 4x256x40x40xf32)
        multiply_9 = paddle._C_ops.multiply(batch_norm__54, sigmoid_11)

        # pd_op.conv2d: (4x5x40x40xf32) <- (4x256x40x40xf32, 5x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            multiply_9, parameter_30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x5x1x1xf32) <- (5xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_29, full_int_array_0)

        # pd_op.add: (4x5x40x40xf32) <- (4x5x40x40xf32, 1x5x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_13, reshape_3)

        # pd_op.sigmoid: (4x4x40x40xf32) <- (4x4x40x40xf32)
        sigmoid_12 = paddle._C_ops.sigmoid(add_4)

        # pd_op.flatten: (4x4x1600xf32) <- (4x4x40x40xf32)
        flatten_3 = paddle._C_ops.flatten(sigmoid_12, 2, 3)

        # pd_op.transpose: (4x1600x4xf32) <- (4x4x1600xf32)
        transpose_3 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])

        # pd_op.split: ([4x4x40x40xf32, 4x1x40x40xf32]) <- (4x5x40x40xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(add_5, full_int_array_1, full_0)

        # builtin.split: (4x4x40x40xf32, 4x1x40x40xf32) <- ([4x4x40x40xf32, 4x1x40x40xf32])
        (
            split_1,
            split_7,
        ) = split_6

        # pd_op.flatten: (4x4x1600xf32) <- (4x4x40x40xf32)
        flatten_4 = paddle._C_ops.flatten(split_1, 2, 3)

        # pd_op.transpose: (4x1600x4xf32) <- (4x4x1600xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])

        # pd_op.sigmoid: (4x1x40x40xf32) <- (4x1x40x40xf32)
        sigmoid_13 = paddle._C_ops.sigmoid(split_7)

        # pd_op.flatten: (4x1x1600xf32) <- (4x1x40x40xf32)
        flatten_5 = paddle._C_ops.flatten(sigmoid_13, 2, 3)

        # pd_op.transpose: (4x1600x1xf32) <- (4x1x1600xf32)
        transpose_5 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])

        # pd_op.conv2d: (4x256x20x20xf32) <- (4x1024x20x20xf32, 256x1024x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            data_2, parameter_28, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_27,
                parameter_26,
                parameter_25,
                parameter_24,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x20x20xf32) <- (4x256x20x20xf32)
        sigmoid_14 = paddle._C_ops.sigmoid(batch_norm__60)

        # pd_op.multiply: (4x256x20x20xf32) <- (4x256x20x20xf32, 4x256x20x20xf32)
        multiply_10 = paddle._C_ops.multiply(batch_norm__60, sigmoid_14)

        # pd_op.conv2d: (4x256x20x20xf32) <- (4x256x20x20xf32, 256x256x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            multiply_10, parameter_23, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_22,
                parameter_21,
                parameter_20,
                parameter_19,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x20x20xf32) <- (4x256x20x20xf32)
        sigmoid_15 = paddle._C_ops.sigmoid(batch_norm__66)

        # pd_op.multiply: (4x256x20x20xf32) <- (4x256x20x20xf32, 4x256x20x20xf32)
        multiply_11 = paddle._C_ops.multiply(batch_norm__66, sigmoid_15)

        # pd_op.conv2d: (4x256x20x20xf32) <- (4x256x20x20xf32, 256x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            multiply_11, parameter_18, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_17,
                parameter_16,
                parameter_15,
                parameter_14,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x20x20xf32) <- (4x256x20x20xf32)
        sigmoid_16 = paddle._C_ops.sigmoid(batch_norm__72)

        # pd_op.multiply: (4x256x20x20xf32) <- (4x256x20x20xf32, 4x256x20x20xf32)
        multiply_12 = paddle._C_ops.multiply(batch_norm__72, sigmoid_16)

        # pd_op.conv2d: (4x4x20x20xf32) <- (4x256x20x20xf32, 4x256x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            multiply_12, parameter_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_12, full_int_array_0)

        # pd_op.add: (4x4x20x20xf32) <- (4x4x20x20xf32, 1x4x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_17, reshape_4)

        # pd_op.conv2d: (4x256x20x20xf32) <- (4x256x20x20xf32, 256x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            multiply_10, parameter_11, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_10,
                parameter_9,
                parameter_8,
                parameter_7,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x20x20xf32) <- (4x256x20x20xf32)
        sigmoid_17 = paddle._C_ops.sigmoid(batch_norm__78)

        # pd_op.multiply: (4x256x20x20xf32) <- (4x256x20x20xf32, 4x256x20x20xf32)
        multiply_13 = paddle._C_ops.multiply(batch_norm__78, sigmoid_17)

        # pd_op.conv2d: (4x256x20x20xf32) <- (4x256x20x20xf32, 256x256x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            multiply_13, parameter_6, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x20x20xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
                False,
                float("0.97"),
                float("0.001"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.sigmoid: (4x256x20x20xf32) <- (4x256x20x20xf32)
        sigmoid_18 = paddle._C_ops.sigmoid(batch_norm__84)

        # pd_op.multiply: (4x256x20x20xf32) <- (4x256x20x20xf32, 4x256x20x20xf32)
        multiply_14 = paddle._C_ops.multiply(batch_norm__84, sigmoid_18)

        # pd_op.conv2d: (4x5x20x20xf32) <- (4x256x20x20xf32, 5x256x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            multiply_14, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x5x1x1xf32) <- (5xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_0, full_int_array_0)

        # pd_op.add: (4x5x20x20xf32) <- (4x5x20x20xf32, 1x5x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_20, reshape_5)

        # pd_op.sigmoid: (4x4x20x20xf32) <- (4x4x20x20xf32)
        sigmoid_19 = paddle._C_ops.sigmoid(add_6)

        # pd_op.flatten: (4x4x400xf32) <- (4x4x20x20xf32)
        flatten_6 = paddle._C_ops.flatten(sigmoid_19, 2, 3)

        # pd_op.transpose: (4x400x4xf32) <- (4x4x400xf32)
        transpose_6 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])

        # pd_op.split: ([4x4x20x20xf32, 4x1x20x20xf32]) <- (4x5x20x20xf32, 2xi64, 1xi32)
        split_8 = paddle._C_ops.split(add_7, full_int_array_1, full_0)

        # builtin.split: (4x4x20x20xf32, 4x1x20x20xf32) <- ([4x4x20x20xf32, 4x1x20x20xf32])
        (
            split_2,
            split_9,
        ) = split_8

        # pd_op.flatten: (4x4x400xf32) <- (4x4x20x20xf32)
        flatten_7 = paddle._C_ops.flatten(split_2, 2, 3)

        # pd_op.transpose: (4x400x4xf32) <- (4x4x400xf32)
        transpose_7 = paddle._C_ops.transpose(flatten_7, [0, 2, 1])

        # pd_op.sigmoid: (4x1x20x20xf32) <- (4x1x20x20xf32)
        sigmoid_20 = paddle._C_ops.sigmoid(split_9)

        # pd_op.flatten: (4x1x400xf32) <- (4x1x20x20xf32)
        flatten_8 = paddle._C_ops.flatten(sigmoid_20, 2, 3)

        # pd_op.transpose: (4x400x1xf32) <- (4x1x400xf32)
        transpose_8 = paddle._C_ops.transpose(flatten_8, [0, 2, 1])

        # builtin.combine: ([4x6400x4xf32, 4x1600x4xf32, 4x400x4xf32]) <- (4x6400x4xf32, 4x1600x4xf32, 4x400x4xf32)
        combine_0 = [transpose_0, transpose_3, transpose_6]

        # pd_op.concat: (4x8400x4xf32) <- ([4x6400x4xf32, 4x1600x4xf32, 4x400x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([4x6400x4xf32, 4x1600x4xf32, 4x400x4xf32]) <- (4x6400x4xf32, 4x1600x4xf32, 4x400x4xf32)
        combine_1 = [transpose_1, transpose_4, transpose_7]

        # pd_op.concat: (4x8400x4xf32) <- ([4x6400x4xf32, 4x1600x4xf32, 4x400x4xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_1, full_0)

        # builtin.combine: ([4x6400x1xf32, 4x1600x1xf32, 4x400x1xf32]) <- (4x6400x1xf32, 4x1600x1xf32, 4x400x1xf32)
        combine_2 = [transpose_2, transpose_5, transpose_8]

        # pd_op.concat: (4x8400x1xf32) <- ([4x6400x1xf32, 4x1600x1xf32, 4x400x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("80"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (80xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_4, full_5, full_6, dtype="int64")

        # pd_op.cast: (80xf32) <- (80xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(cast_0, full_7, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_8, float("0"), True)

        # builtin.combine: ([80xf32, 80xf32]) <- (80xf32, 80xf32)
        combine_3 = [scale_2, scale_2]

        # pd_op.meshgrid: ([80x80xf32, 80x80xf32]) <- ([80xf32, 80xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (80x80xf32, 80x80xf32) <- ([80x80xf32, 80x80xf32])
        (
            split_10,
            split_11,
        ) = meshgrid_0

        # builtin.combine: ([80x80xf32, 80x80xf32]) <- (80x80xf32, 80x80xf32)
        combine_4 = [split_11, split_10]

        # pd_op.stack: (80x80x2xf32) <- ([80x80xf32, 80x80xf32])
        stack_0 = paddle._C_ops.stack(combine_4, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 2]

        # pd_op.reshape: (6400x2xf32) <- (80x80x2xf32, 2xi64)
        reshape_9 = paddle._C_ops.reshape(stack_0, full_int_array_2)

        # pd_op.full: (6400x1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [6400, 1],
            float("8"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("40"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (40xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_4, full_10, full_6, dtype="int64")

        # pd_op.cast: (40xf32) <- (40xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(cast_1, full_7, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(scale_3, full_11, float("0"), True)

        # builtin.combine: ([40xf32, 40xf32]) <- (40xf32, 40xf32)
        combine_5 = [scale_4, scale_4]

        # pd_op.meshgrid: ([40x40xf32, 40x40xf32]) <- ([40xf32, 40xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_5)

        # builtin.split: (40x40xf32, 40x40xf32) <- ([40x40xf32, 40x40xf32])
        (
            split_12,
            split_13,
        ) = meshgrid_1

        # builtin.combine: ([40x40xf32, 40x40xf32]) <- (40x40xf32, 40x40xf32)
        combine_6 = [split_13, split_12]

        # pd_op.stack: (40x40x2xf32) <- ([40x40xf32, 40x40xf32])
        stack_1 = paddle._C_ops.stack(combine_6, -1)

        # pd_op.reshape: (1600x2xf32) <- (40x40x2xf32, 2xi64)
        reshape_10 = paddle._C_ops.reshape(stack_1, full_int_array_2)

        # pd_op.full: (1600x1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1600, 1],
            float("16"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("20"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (20xi64) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_4, full_13, full_6, dtype="int64")

        # pd_op.cast: (20xf32) <- (20xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(cast_2, full_7, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(scale_5, full_14, float("0"), True)

        # builtin.combine: ([20xf32, 20xf32]) <- (20xf32, 20xf32)
        combine_7 = [scale_6, scale_6]

        # pd_op.meshgrid: ([20x20xf32, 20x20xf32]) <- ([20xf32, 20xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_7)

        # builtin.split: (20x20xf32, 20x20xf32) <- ([20x20xf32, 20x20xf32])
        (
            split_14,
            split_15,
        ) = meshgrid_2

        # builtin.combine: ([20x20xf32, 20x20xf32]) <- (20x20xf32, 20x20xf32)
        combine_8 = [split_15, split_14]

        # pd_op.stack: (20x20x2xf32) <- ([20x20xf32, 20x20xf32])
        stack_2 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.reshape: (400x2xf32) <- (20x20x2xf32, 2xi64)
        reshape_11 = paddle._C_ops.reshape(stack_2, full_int_array_2)

        # pd_op.full: (400x1xf32) <- ()
        full_15 = paddle._C_ops.full(
            [400, 1],
            float("32"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([6400x2xf32, 1600x2xf32, 400x2xf32]) <- (6400x2xf32, 1600x2xf32, 400x2xf32)
        combine_9 = [reshape_9, reshape_10, reshape_11]

        # pd_op.concat: (8400x2xf32) <- ([6400x2xf32, 1600x2xf32, 400x2xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_9, full_16)

        # builtin.combine: ([6400x1xf32, 1600x1xf32, 400x1xf32]) <- (6400x1xf32, 1600x1xf32, 400x1xf32)
        combine_10 = [full_9, full_12, full_15]

        # pd_op.concat: (8400x1xf32) <- ([6400x1xf32, 1600x1xf32, 400x1xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_10, full_16)

        # pd_op.assign: (8400x1xf32) <- (8400x1xf32)
        assign_5 = concat_6

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([4x8400x2xf32, 4x8400x2xf32]) <- (4x8400x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_4, 2, full_1)

        # builtin.split: (4x8400x2xf32, 4x8400x2xf32) <- ([4x8400x2xf32, 4x8400x2xf32])
        (
            split_3,
            split_16,
        ) = split_with_num_0

        # pd_op.divide: (8400x2xf32) <- (8400x2xf32, 8400x1xf32)
        divide_0 = paddle._C_ops.divide(concat_5, concat_6)

        # pd_op.add: (4x8400x2xf32) <- (4x8400x2xf32, 8400x2xf32)
        add_0 = paddle._C_ops.add(split_3, divide_0)

        # pd_op.exp: (4x8400x2xf32) <- (4x8400x2xf32)
        exp_0 = paddle._C_ops.exp(split_16)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x8400x2xf32) <- (4x8400x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(exp_0, full_2, float("0"), True)

        # pd_op.subtract: (4x8400x2xf32) <- (4x8400x2xf32, 4x8400x2xf32)
        subtract_0 = paddle._C_ops.subtract(add_0, scale_0)

        # pd_op.add: (4x8400x2xf32) <- (4x8400x2xf32, 4x8400x2xf32)
        add_1 = paddle._C_ops.add(add_0, scale_0)

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([4x8400x2xf32, 4x8400x2xf32]) <- (4x8400x2xf32, 4x8400x2xf32)
        combine_11 = [subtract_0, add_1]

        # pd_op.concat: (4x8400x4xf32) <- ([4x8400x2xf32, 4x8400x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_11, full_3)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(cast_0, full_7, float("0.5"), True)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(scale_7, full_8, float("0"), True)

        # builtin.combine: ([80xf32, 80xf32]) <- (80xf32, 80xf32)
        combine_12 = [scale_8, scale_8]

        # pd_op.meshgrid: ([80x80xf32, 80x80xf32]) <- ([80xf32, 80xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_12)

        # builtin.split: (80x80xf32, 80x80xf32) <- ([80x80xf32, 80x80xf32])
        (
            split_17,
            split_18,
        ) = meshgrid_3

        # builtin.combine: ([80x80xf32, 80x80xf32]) <- (80x80xf32, 80x80xf32)
        combine_13 = [split_18, split_17]

        # pd_op.stack: (80x80x2xf32) <- ([80x80xf32, 80x80xf32])
        stack_3 = paddle._C_ops.stack(combine_13, -1)

        # pd_op.reshape: (6400x2xf32) <- (80x80x2xf32, 2xi64)
        reshape_6 = paddle._C_ops.reshape(stack_3, full_int_array_2)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(cast_1, full_7, float("0.5"), True)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(scale_9, full_11, float("0"), True)

        # builtin.combine: ([40xf32, 40xf32]) <- (40xf32, 40xf32)
        combine_14 = [scale_10, scale_10]

        # pd_op.meshgrid: ([40x40xf32, 40x40xf32]) <- ([40xf32, 40xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_14)

        # builtin.split: (40x40xf32, 40x40xf32) <- ([40x40xf32, 40x40xf32])
        (
            split_19,
            split_20,
        ) = meshgrid_4

        # builtin.combine: ([40x40xf32, 40x40xf32]) <- (40x40xf32, 40x40xf32)
        combine_15 = [split_20, split_19]

        # pd_op.stack: (40x40x2xf32) <- ([40x40xf32, 40x40xf32])
        stack_4 = paddle._C_ops.stack(combine_15, -1)

        # pd_op.reshape: (1600x2xf32) <- (40x40x2xf32, 2xi64)
        reshape_7 = paddle._C_ops.reshape(stack_4, full_int_array_2)

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(cast_2, full_7, float("0.5"), True)

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(scale_11, full_14, float("0"), True)

        # builtin.combine: ([20xf32, 20xf32]) <- (20xf32, 20xf32)
        combine_16 = [scale_12, scale_12]

        # pd_op.meshgrid: ([20x20xf32, 20x20xf32]) <- ([20xf32, 20xf32])
        meshgrid_5 = paddle._C_ops.meshgrid(combine_16)

        # builtin.split: (20x20xf32, 20x20xf32) <- ([20x20xf32, 20x20xf32])
        (
            split_21,
            split_22,
        ) = meshgrid_5

        # builtin.combine: ([20x20xf32, 20x20xf32]) <- (20x20xf32, 20x20xf32)
        combine_17 = [split_22, split_21]

        # pd_op.stack: (20x20x2xf32) <- ([20x20xf32, 20x20xf32])
        stack_5 = paddle._C_ops.stack(combine_17, -1)

        # pd_op.reshape: (400x2xf32) <- (20x20x2xf32, 2xi64)
        reshape_8 = paddle._C_ops.reshape(stack_5, full_int_array_2)

        # builtin.combine: ([6400x2xf32, 1600x2xf32, 400x2xf32]) <- (6400x2xf32, 1600x2xf32, 400x2xf32)
        combine_18 = [reshape_6, reshape_7, reshape_8]

        # pd_op.concat: (8400x2xf32) <- ([6400x2xf32, 1600x2xf32, 400x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_18, full_16)
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
            conv2d_3,
            reshape_0,
            conv2d_4,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            sigmoid_3,
            multiply_3,
            conv2d_5,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            sigmoid_4,
            multiply_4,
            conv2d_6,
            reshape_1,
            sigmoid_5,
            transpose_0,
            full_0,
            split_0,
            transpose_1,
            sigmoid_6,
            transpose_2,
            conv2d_7,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            sigmoid_7,
            multiply_5,
            conv2d_8,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            sigmoid_8,
            multiply_6,
            conv2d_9,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            sigmoid_9,
            multiply_7,
            conv2d_10,
            reshape_2,
            conv2d_11,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            sigmoid_10,
            multiply_8,
            conv2d_12,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            sigmoid_11,
            multiply_9,
            conv2d_13,
            reshape_3,
            sigmoid_12,
            transpose_3,
            assign_0,
            split_1,
            transpose_4,
            sigmoid_13,
            transpose_5,
            conv2d_14,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            sigmoid_14,
            multiply_10,
            conv2d_15,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            sigmoid_15,
            multiply_11,
            conv2d_16,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            sigmoid_16,
            multiply_12,
            conv2d_17,
            reshape_4,
            conv2d_18,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            sigmoid_17,
            multiply_13,
            conv2d_19,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            sigmoid_18,
            multiply_14,
            conv2d_20,
            reshape_5,
            sigmoid_19,
            transpose_6,
            assign_1,
            split_2,
            transpose_7,
            sigmoid_20,
            transpose_8,
            assign_2,
            assign_3,
            assign_4,
            full_1,
            split_3,
            divide_0,
            add_0,
            exp_0,
            full_2,
            scale_0,
            subtract_0,
            add_1,
            full_3,
            concat_0,
            concat_1,
            concat_2,
            concat_3,
            assign_5,
            reshape_6,
            reshape_7,
            reshape_8,
        )
