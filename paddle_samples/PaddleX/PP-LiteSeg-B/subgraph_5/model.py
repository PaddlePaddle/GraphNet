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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
    ):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (2x1024x1x1xf32) <- (2x1024x16x32xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            data_5,
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

        # pd_op.conv2d: (2x128x1x1xf32) <- (2x1024x1x1xf32, 128x1024x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            pool2d_0, parameter_83, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_83

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_82, full_int_array_1)
        del full_int_array_1, parameter_82

        # pd_op.add: (2x128x1x1xf32) <- (2x128x1x1xf32, 1x128x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.batch_norm_: (2x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_81,
                parameter_80,
                parameter_79,
                parameter_78,
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
        del parameter_78, parameter_79, parameter_80, parameter_81

        # pd_op.relu: (2x128x1x1xf32) <- (2x128x1x1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.bilinear_interp: (2x128x16x32xf32) <- (2x128x1x1xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            relu_1, None, None, None, "NCHW", -1, 16, 32, [], "bilinear", False, 0
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (2x1024x2x2xf32) <- (2x1024x16x32xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            data_5,
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

        # pd_op.conv2d: (2x128x2x2xf32) <- (2x1024x2x2xf32, 128x1024x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            pool2d_1, parameter_77, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_77

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_76, full_int_array_3)
        del full_int_array_3, parameter_76

        # pd_op.add: (2x128x2x2xf32) <- (2x128x2x2xf32, 1x128x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_1)

        # pd_op.batch_norm_: (2x128x2x2xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x2x2xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
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
        del parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.relu: (2x128x2x2xf32) <- (2x128x2x2xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.bilinear_interp: (2x128x16x32xf32) <- (2x128x2x2xf32, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(
            relu_2, None, None, None, "NCHW", -1, 16, 32, [], "bilinear", False, 0
        )

        # pd_op.add: (2x128x16x32xf32) <- (2x128x16x32xf32, 2x128x16x32xf32)
        add_2 = paddle._C_ops.add(bilinear_interp_0, bilinear_interp_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [4, 4]

        # pd_op.pool2d: (2x1024x4x4xf32) <- (2x1024x16x32xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            data_5,
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
        del full_int_array_4

        # pd_op.conv2d: (2x128x4x4xf32) <- (2x1024x4x4xf32, 128x1024x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            pool2d_2, parameter_71, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_71

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_70, full_int_array_5)
        del full_int_array_5, parameter_70

        # pd_op.add: (2x128x4x4xf32) <- (2x128x4x4xf32, 1x128x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_2, reshape_2)

        # pd_op.batch_norm_: (2x128x4x4xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x4x4xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_69,
                parameter_68,
                parameter_67,
                parameter_66,
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
        del parameter_66, parameter_67, parameter_68, parameter_69

        # pd_op.relu: (2x128x4x4xf32) <- (2x128x4x4xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.bilinear_interp: (2x128x16x32xf32) <- (2x128x4x4xf32, None, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(
            relu_3, None, None, None, "NCHW", -1, 16, 32, [], "bilinear", False, 0
        )

        # pd_op.add: (2x128x16x32xf32) <- (2x128x16x32xf32, 2x128x16x32xf32)
        add_4 = paddle._C_ops.add(add_2, bilinear_interp_2)

        # pd_op.conv2d: (2x128x16x32xf32) <- (2x128x16x32xf32, 128x128x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            add_4, parameter_65, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_65

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_64, full_int_array_6)
        del full_int_array_6, parameter_64

        # pd_op.add: (2x128x16x32xf32) <- (2x128x16x32xf32, 1x128x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_3, reshape_3)

        # pd_op.batch_norm_: (2x128x16x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x16x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_63,
                parameter_62,
                parameter_61,
                parameter_60,
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
        del parameter_60, parameter_61, parameter_62, parameter_63

        # pd_op.relu: (2x128x16x32xf32) <- (2x128x16x32xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (2x128x16x32xf32) <- (2x1024x16x32xf32, 128x1024x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            data_5, parameter_59, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_5, parameter_59

        # pd_op.batch_norm_: (2x128x16x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x16x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_58,
                parameter_57,
                parameter_56,
                parameter_55,
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
        del parameter_55, parameter_56, parameter_57, parameter_58

        # pd_op.relu: (2x128x16x32xf32) <- (2x128x16x32xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.bilinear_interp: (2x128x16x32xf32) <- (2x128x16x32xf32, None, None, None)
        bilinear_interp_3 = paddle._C_ops.bilinear_interp(
            relu_4, None, None, None, "NCHW", -1, 16, 32, [], "bilinear", False, 0
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.mean: (2x1x16x32xf32) <- (2x128x16x32xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(relu_5, full_int_array_7, True)
        del full_int_array_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [1]

        # pd_op.max: (2x1x16x32xf32) <- (2x128x16x32xf32, 1xi64)
        max_0 = paddle._C_ops.max(relu_5, full_int_array_8, True)
        del full_int_array_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.mean: (2x1x16x32xf32) <- (2x128x16x32xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(bilinear_interp_3, full_int_array_9, True)
        del full_int_array_9

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [1]

        # pd_op.max: (2x1x16x32xf32) <- (2x128x16x32xf32, 1xi64)
        max_1 = paddle._C_ops.max(bilinear_interp_3, full_int_array_10, True)
        del full_int_array_10

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x1x16x32xf32, 2x1x16x32xf32, 2x1x16x32xf32, 2x1x16x32xf32]) <- (2x1x16x32xf32, 2x1x16x32xf32, 2x1x16x32xf32, 2x1x16x32xf32)
        combine_0 = [mean_0, max_0, mean_1, max_1]

        # pd_op.concat: (2x4x16x32xf32) <- ([2x1x16x32xf32, 2x1x16x32xf32, 2x1x16x32xf32, 2x1x16x32xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, full_0

        # pd_op.conv2d: (2x2x16x32xf32) <- (2x4x16x32xf32, 2x4x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            concat_0, parameter_54, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_54

        # pd_op.batch_norm_: (2x2x16x32xf32, 2xf32, 2xf32, 2xf32, 2xf32, -1xui8) <- (2x2x16x32xf32, 2xf32, 2xf32, 2xf32, 2xf32)
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
                parameter_53,
                parameter_52,
                parameter_51,
                parameter_50,
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
        del parameter_50, parameter_51, parameter_52, parameter_53

        # pd_op.relu: (2x2x16x32xf32) <- (2x2x16x32xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (2x1x16x32xf32) <- (2x2x16x32xf32, 1x2x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_6, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # pd_op.batch_norm_: (2x1x16x32xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (2x1x16x32xf32, 1xf32, 1xf32, 1xf32, 1xf32)
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
                parameter_48,
                parameter_47,
                parameter_46,
                parameter_45,
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
        del parameter_45, parameter_46, parameter_47, parameter_48

        # pd_op.sigmoid: (2x1x16x32xf32) <- (2x1x16x32xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(batch_norm__36)
        del batch_norm__36

        # pd_op.multiply: (2x128x16x32xf32) <- (2x128x16x32xf32, 2x1x16x32xf32)
        multiply_0 = paddle._C_ops.multiply(relu_5, sigmoid_0)

        # pd_op.subtract: (2x1x16x32xf32) <- (1xf32, 2x1x16x32xf32)
        subtract_0 = paddle._C_ops.subtract(data_0, sigmoid_0)
        del data_0

        # pd_op.multiply: (2x128x16x32xf32) <- (2x128x16x32xf32, 2x1x16x32xf32)
        multiply_1 = paddle._C_ops.multiply(bilinear_interp_3, subtract_0)

        # pd_op.add: (2x128x16x32xf32) <- (2x128x16x32xf32, 2x128x16x32xf32)
        add_6 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.conv2d: (2x128x16x32xf32) <- (2x128x16x32xf32, 128x128x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            add_6, parameter_44, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_44

        # pd_op.batch_norm_: (2x128x16x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x16x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_43,
                parameter_42,
                parameter_41,
                parameter_40,
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
        del parameter_40, parameter_41, parameter_42, parameter_43

        # pd_op.relu: (2x128x16x32xf32) <- (2x128x16x32xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (2x128x32x64xf32) <- (2x512x32x64xf32, 128x512x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            data_4, parameter_39, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_4, parameter_39

        # pd_op.batch_norm_: (2x128x32x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x32x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_38,
                parameter_37,
                parameter_36,
                parameter_35,
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
        del parameter_35, parameter_36, parameter_37, parameter_38

        # pd_op.relu: (2x128x32x64xf32) <- (2x128x32x64xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.bilinear_interp: (2x128x32x64xf32) <- (2x128x16x32xf32, None, None, None)
        bilinear_interp_4 = paddle._C_ops.bilinear_interp(
            relu_7, None, None, None, "NCHW", -1, 32, 64, [], "bilinear", False, 0
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [1]

        # pd_op.mean: (2x1x32x64xf32) <- (2x128x32x64xf32, 1xi64)
        mean_2 = paddle._C_ops.mean(relu_8, full_int_array_11, True)
        del full_int_array_11

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [1]

        # pd_op.max: (2x1x32x64xf32) <- (2x128x32x64xf32, 1xi64)
        max_2 = paddle._C_ops.max(relu_8, full_int_array_12, True)
        del full_int_array_12

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.mean: (2x1x32x64xf32) <- (2x128x32x64xf32, 1xi64)
        mean_3 = paddle._C_ops.mean(bilinear_interp_4, full_int_array_13, True)
        del full_int_array_13

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [1]

        # pd_op.max: (2x1x32x64xf32) <- (2x128x32x64xf32, 1xi64)
        max_3 = paddle._C_ops.max(bilinear_interp_4, full_int_array_14, True)
        del full_int_array_14

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x1x32x64xf32, 2x1x32x64xf32, 2x1x32x64xf32, 2x1x32x64xf32]) <- (2x1x32x64xf32, 2x1x32x64xf32, 2x1x32x64xf32, 2x1x32x64xf32)
        combine_1 = [mean_2, max_2, mean_3, max_3]

        # pd_op.concat: (2x4x32x64xf32) <- ([2x1x32x64xf32, 2x1x32x64xf32, 2x1x32x64xf32, 2x1x32x64xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)
        del combine_1, full_1

        # pd_op.conv2d: (2x2x32x64xf32) <- (2x4x32x64xf32, 2x4x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            concat_1, parameter_34, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_34

        # pd_op.batch_norm_: (2x2x32x64xf32, 2xf32, 2xf32, 2xf32, 2xf32, -1xui8) <- (2x2x32x64xf32, 2xf32, 2xf32, 2xf32, 2xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
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
        del parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.relu: (2x2x32x64xf32) <- (2x2x32x64xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__54)
        del batch_norm__54

        # pd_op.conv2d: (2x1x32x64xf32) <- (2x2x32x64xf32, 1x2x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_9, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.batch_norm_: (2x1x32x64xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (2x1x32x64xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_28,
                parameter_27,
                parameter_26,
                parameter_25,
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
        del parameter_25, parameter_26, parameter_27, parameter_28

        # pd_op.sigmoid: (2x1x32x64xf32) <- (2x1x32x64xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(batch_norm__60)
        del batch_norm__60

        # pd_op.multiply: (2x128x32x64xf32) <- (2x128x32x64xf32, 2x1x32x64xf32)
        multiply_2 = paddle._C_ops.multiply(relu_8, sigmoid_1)

        # pd_op.subtract: (2x1x32x64xf32) <- (1xf32, 2x1x32x64xf32)
        subtract_1 = paddle._C_ops.subtract(data_1, sigmoid_1)
        del data_1

        # pd_op.multiply: (2x128x32x64xf32) <- (2x128x32x64xf32, 2x1x32x64xf32)
        multiply_3 = paddle._C_ops.multiply(bilinear_interp_4, subtract_1)

        # pd_op.add: (2x128x32x64xf32) <- (2x128x32x64xf32, 2x128x32x64xf32)
        add_7 = paddle._C_ops.add(multiply_2, multiply_3)

        # pd_op.conv2d: (2x96x32x64xf32) <- (2x128x32x64xf32, 96x128x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_7, parameter_24, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_24

        # pd_op.batch_norm_: (2x96x32x64xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x32x64xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
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
        del parameter_20, parameter_21, parameter_22, parameter_23

        # pd_op.relu: (2x96x32x64xf32) <- (2x96x32x64xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (2x96x64x128xf32) <- (2x256x64x128xf32, 96x256x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            data_3, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_3, parameter_19

        # pd_op.batch_norm_: (2x96x64x128xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x64x128xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_18,
                parameter_17,
                parameter_16,
                parameter_15,
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
        del parameter_15, parameter_16, parameter_17, parameter_18

        # pd_op.relu: (2x96x64x128xf32) <- (2x96x64x128xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.bilinear_interp: (2x96x64x128xf32) <- (2x96x32x64xf32, None, None, None)
        bilinear_interp_5 = paddle._C_ops.bilinear_interp(
            relu_10, None, None, None, "NCHW", -1, 64, 128, [], "bilinear", False, 0
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.mean: (2x1x64x128xf32) <- (2x96x64x128xf32, 1xi64)
        mean_4 = paddle._C_ops.mean(relu_11, full_int_array_15, True)
        del full_int_array_15

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [1]

        # pd_op.max: (2x1x64x128xf32) <- (2x96x64x128xf32, 1xi64)
        max_4 = paddle._C_ops.max(relu_11, full_int_array_16, True)
        del full_int_array_16

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.mean: (2x1x64x128xf32) <- (2x96x64x128xf32, 1xi64)
        mean_5 = paddle._C_ops.mean(bilinear_interp_5, full_int_array_17, True)
        del full_int_array_17

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [1]

        # pd_op.max: (2x1x64x128xf32) <- (2x96x64x128xf32, 1xi64)
        max_5 = paddle._C_ops.max(bilinear_interp_5, full_int_array_18, True)
        del full_int_array_18

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x1x64x128xf32, 2x1x64x128xf32, 2x1x64x128xf32, 2x1x64x128xf32]) <- (2x1x64x128xf32, 2x1x64x128xf32, 2x1x64x128xf32, 2x1x64x128xf32)
        combine_2 = [mean_4, max_4, mean_5, max_5]

        # pd_op.concat: (2x4x64x128xf32) <- ([2x1x64x128xf32, 2x1x64x128xf32, 2x1x64x128xf32, 2x1x64x128xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_2)
        del combine_2, full_2

        # pd_op.conv2d: (2x2x64x128xf32) <- (2x4x64x128xf32, 2x4x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_2, parameter_14, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_14

        # pd_op.batch_norm_: (2x2x64x128xf32, 2xf32, 2xf32, 2xf32, 2xf32, -1xui8) <- (2x2x64x128xf32, 2xf32, 2xf32, 2xf32, 2xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
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
        del parameter_10, parameter_11, parameter_12, parameter_13

        # pd_op.relu: (2x2x64x128xf32) <- (2x2x64x128xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (2x1x64x128xf32) <- (2x2x64x128xf32, 1x2x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_12, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9

        # pd_op.batch_norm_: (2x1x64x128xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (2x1x64x128xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
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
        del parameter_5, parameter_6, parameter_7, parameter_8

        # pd_op.sigmoid: (2x1x64x128xf32) <- (2x1x64x128xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(batch_norm__84)
        del batch_norm__84

        # pd_op.multiply: (2x96x64x128xf32) <- (2x96x64x128xf32, 2x1x64x128xf32)
        multiply_4 = paddle._C_ops.multiply(relu_11, sigmoid_2)

        # pd_op.subtract: (2x1x64x128xf32) <- (1xf32, 2x1x64x128xf32)
        subtract_2 = paddle._C_ops.subtract(data_2, sigmoid_2)
        del data_2

        # pd_op.multiply: (2x96x64x128xf32) <- (2x96x64x128xf32, 2x1x64x128xf32)
        multiply_5 = paddle._C_ops.multiply(bilinear_interp_5, subtract_2)

        # pd_op.add: (2x96x64x128xf32) <- (2x96x64x128xf32, 2x96x64x128xf32)
        add_8 = paddle._C_ops.add(multiply_4, multiply_5)

        # pd_op.conv2d: (2x64x64x128xf32) <- (2x96x64x128xf32, 64x96x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            add_8, parameter_4, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_4

        # pd_op.batch_norm_: (2x64x64x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (2x64x64x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
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
        del parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.relu: (2x64x64x128xf32) <- (2x64x64x128xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__90)
        del (
            add_0,
            add_1,
            add_2,
            add_3,
            add_4,
            add_5,
            add_6,
            add_7,
            add_8,
            batch_norm__1,
            batch_norm__10,
            batch_norm__11,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__19,
            batch_norm__2,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__3,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__49,
            batch_norm__5,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__7,
            batch_norm__70,
            batch_norm__71,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__79,
            batch_norm__8,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            batch_norm__9,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            bilinear_interp_0,
            bilinear_interp_1,
            bilinear_interp_2,
            bilinear_interp_3,
            bilinear_interp_4,
            bilinear_interp_5,
            concat_0,
            concat_1,
            concat_2,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_12,
            conv2d_13,
            conv2d_14,
            conv2d_15,
            conv2d_2,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            max_0,
            max_1,
            max_2,
            max_3,
            max_4,
            max_5,
            mean_0,
            mean_1,
            mean_2,
            mean_3,
            mean_4,
            mean_5,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            pool2d_0,
            pool2d_1,
            pool2d_2,
            relu_1,
            relu_10,
            relu_11,
            relu_12,
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
            sigmoid_0,
            sigmoid_1,
            sigmoid_2,
            subtract_0,
            subtract_1,
            subtract_2,
        )

        return relu_0
