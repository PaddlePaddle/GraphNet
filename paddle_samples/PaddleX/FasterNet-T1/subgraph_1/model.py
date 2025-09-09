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
        parameter_87,
        parameter_88,
        parameter_89,
        parameter_90,
        parameter_91,
        parameter_92,
        parameter_93,
        parameter_94,
        parameter_95,
        parameter_96,
        parameter_97,
        parameter_98,
        parameter_99,
        parameter_100,
        parameter_101,
        parameter_102,
        parameter_103,
        parameter_104,
        parameter_105,
        parameter_106,
        parameter_107,
        parameter_108,
        parameter_109,
        parameter_110,
        parameter_111,
        parameter_112,
        parameter_113,
        data_0,
    ):
        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x3x224x224xf32, 64x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_113, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_113

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_112,
                parameter_111,
                parameter_110,
                parameter_109,
                False,
                float("0.1"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_109, parameter_110, parameter_111, parameter_112

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [16, 48]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_18 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_20 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_0

        # pd_op.split: ([-1x16x56x56xf32, -1x48x56x56xf32]) <- (-1x64x56x56xf32, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(batch_norm__0, full_int_array_0, full_0)
        del full_int_array_0

        # builtin.split: (-1x16x56x56xf32, -1x48x56x56xf32) <- ([-1x16x56x56xf32, -1x48x56x56xf32])
        (
            split_1,
            split_2,
        ) = split_0
        del split_0

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 16x16x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            split_1, parameter_108, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_108

        # builtin.combine: ([-1x16x56x56xf32, -1x48x56x56xf32]) <- (-1x16x56x56xf32, -1x48x56x56xf32)
        combine_0 = [conv2d_1, split_2]

        # pd_op.concat: (-1x64x56x56xf32) <- ([-1x16x56x56xf32, -1x48x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x64x56x56xf32, 128x64x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            concat_0, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_107

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_106,
                parameter_105,
                parameter_104,
                parameter_103,
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
        del parameter_103, parameter_104, parameter_105, parameter_106

        # pd_op.gelu: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        gelu_0 = paddle._C_ops.gelu(batch_norm__6, False)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x128x56x56xf32, 64x128x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            gelu_0, parameter_102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_102

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add_1 = paddle._C_ops.add(batch_norm__0, conv2d_3)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x56x56xf32, 128x64x2x2xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            add_1, parameter_101, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_101

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_100,
                parameter_99,
                parameter_98,
                parameter_97,
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
        del parameter_100, parameter_97, parameter_98, parameter_99

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [32, 96]

        # pd_op.split: ([-1x32x28x28xf32, -1x96x28x28xf32]) <- (-1x128x28x28xf32, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(batch_norm__12, full_int_array_1, full_0)

        # builtin.split: (-1x32x28x28xf32, -1x96x28x28xf32) <- ([-1x32x28x28xf32, -1x96x28x28xf32])
        (
            split_4,
            split_5,
        ) = split_3
        del split_3

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x32x28x28xf32, 32x32x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            split_4, parameter_96, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_96

        # builtin.combine: ([-1x32x28x28xf32, -1x96x28x28xf32]) <- (-1x32x28x28xf32, -1x96x28x28xf32)
        combine_1 = [conv2d_5, split_5]

        # pd_op.concat: (-1x128x28x28xf32) <- ([-1x32x28x28xf32, -1x96x28x28xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x128x28x28xf32, 256x128x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            concat_1, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_95

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_94,
                parameter_93,
                parameter_92,
                parameter_91,
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
        del parameter_91, parameter_92, parameter_93, parameter_94

        # pd_op.gelu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        gelu_1 = paddle._C_ops.gelu(batch_norm__18, False)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x256x28x28xf32, 128x256x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            gelu_1, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_90

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full(
            [],
            float("0.998333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x28x28xf32)
        shape64_0 = paddle._C_ops.shape64(conv2d_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [slice_0, full_2, full_2, full_2]
        del slice_0

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            stack_0,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_0

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_2 = paddle._C_ops.add(full_1, uniform_0)
        del uniform_0

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_2)
        del add_2

        # pd_op.divide: (-1x128x28x28xf32) <- (-1x128x28x28xf32, xf32)
        divide_0 = paddle._C_ops.divide(conv2d_7, full_1)

        # pd_op.multiply: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add_3 = paddle._C_ops.add(batch_norm__12, multiply_0)

        # pd_op.split: ([-1x32x28x28xf32, -1x96x28x28xf32]) <- (-1x128x28x28xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(add_3, full_int_array_1, full_0)
        del full_int_array_1

        # builtin.split: (-1x32x28x28xf32, -1x96x28x28xf32) <- ([-1x32x28x28xf32, -1x96x28x28xf32])
        (
            split_7,
            split_8,
        ) = split_6
        del split_6

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x32x28x28xf32, 32x32x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            split_7, parameter_89, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_89

        # builtin.combine: ([-1x32x28x28xf32, -1x96x28x28xf32]) <- (-1x32x28x28xf32, -1x96x28x28xf32)
        combine_3 = [conv2d_8, split_8]

        # pd_op.concat: (-1x128x28x28xf32) <- ([-1x32x28x28xf32, -1x96x28x28xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x128x28x28xf32, 256x128x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            concat_2, parameter_88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_88

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_87,
                parameter_86,
                parameter_85,
                parameter_84,
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
        del parameter_84, parameter_85, parameter_86, parameter_87

        # pd_op.gelu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        gelu_2 = paddle._C_ops.gelu(batch_norm__24, False)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x256x28x28xf32, 128x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            gelu_2, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_83

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("0.996667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x28x28xf32)
        shape64_1 = paddle._C_ops.shape64(conv2d_10)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_4 = [slice_1, full_2, full_2, full_2]
        del slice_1

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            stack_1,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_1

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_4 = paddle._C_ops.add(full_5, uniform_1)
        del uniform_1

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_4)
        del add_4

        # pd_op.divide: (-1x128x28x28xf32) <- (-1x128x28x28xf32, xf32)
        divide_1 = paddle._C_ops.divide(conv2d_10, full_5)

        # pd_op.multiply: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add_5 = paddle._C_ops.add(add_3, multiply_1)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x28x28xf32, 256x128x2x2xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_5, parameter_82, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_82

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [64, 192]

        # pd_op.split: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x256x14x14xf32, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(batch_norm__30, full_int_array_4, full_0)

        # builtin.split: (-1x64x14x14xf32, -1x192x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32])
        (
            split_10,
            split_11,
        ) = split_9
        del split_9

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            split_10, parameter_77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77

        # builtin.combine: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x64x14x14xf32, -1x192x14x14xf32)
        combine_5 = [conv2d_12, split_11]

        # pd_op.concat: (-1x256x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_3, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_76

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
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

        # pd_op.gelu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        gelu_3 = paddle._C_ops.gelu(batch_norm__36, False)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x512x14x14xf32, 256x512x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            gelu_3, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [],
            float("0.995"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x14x14xf32)
        shape64_2 = paddle._C_ops.shape64(conv2d_14)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_2

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [slice_2, full_2, full_2, full_2]
        del slice_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            stack_2,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_2

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_6 = paddle._C_ops.add(full_6, uniform_2)
        del uniform_2

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_6)
        del add_6

        # pd_op.divide: (-1x256x14x14xf32) <- (-1x256x14x14xf32, xf32)
        divide_2 = paddle._C_ops.divide(conv2d_14, full_6)

        # pd_op.multiply: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_7 = paddle._C_ops.add(batch_norm__30, multiply_2)

        # pd_op.split: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x256x14x14xf32, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(add_7, full_int_array_4, full_0)

        # builtin.split: (-1x64x14x14xf32, -1x192x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32])
        (
            split_13,
            split_14,
        ) = split_12
        del split_12

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            split_13, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70

        # builtin.combine: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x64x14x14xf32, -1x192x14x14xf32)
        combine_7 = [conv2d_15, split_14]

        # pd_op.concat: (-1x256x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_7, full_0)
        del combine_7

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            concat_4, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_69

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
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
        del parameter_65, parameter_66, parameter_67, parameter_68

        # pd_op.gelu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        gelu_4 = paddle._C_ops.gelu(batch_norm__42, False)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x512x14x14xf32, 256x512x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            gelu_4, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [],
            float("0.993333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x14x14xf32)
        shape64_3 = paddle._C_ops.shape64(conv2d_17)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_3

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_8 = [slice_3, full_2, full_2, full_2]
        del slice_3

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            stack_3,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_3

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_8 = paddle._C_ops.add(full_7, uniform_3)
        del uniform_3

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_8)
        del add_8

        # pd_op.divide: (-1x256x14x14xf32) <- (-1x256x14x14xf32, xf32)
        divide_3 = paddle._C_ops.divide(conv2d_17, full_7)

        # pd_op.multiply: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_9 = paddle._C_ops.add(add_7, multiply_3)

        # pd_op.split: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x256x14x14xf32, 2xi64, 1xi32)
        split_15 = paddle._C_ops.split(add_9, full_int_array_4, full_0)

        # builtin.split: (-1x64x14x14xf32, -1x192x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32])
        (
            split_16,
            split_17,
        ) = split_15
        del split_15

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            split_16, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_63

        # builtin.combine: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x64x14x14xf32, -1x192x14x14xf32)
        combine_9 = [conv2d_18, split_17]

        # pd_op.concat: (-1x256x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_9, full_0)
        del combine_9

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            concat_5, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_62

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_61,
                parameter_60,
                parameter_59,
                parameter_58,
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
        del parameter_58, parameter_59, parameter_60, parameter_61

        # pd_op.gelu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        gelu_5 = paddle._C_ops.gelu(batch_norm__48, False)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x512x14x14xf32, 256x512x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            gelu_5, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_57

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [],
            float("0.991667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x14x14xf32)
        shape64_4 = paddle._C_ops.shape64(conv2d_20)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_4

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_10 = [slice_4, full_2, full_2, full_2]
        del slice_4

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            stack_4,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_4

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_10 = paddle._C_ops.add(full_8, uniform_4)
        del uniform_4

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_10)
        del add_10

        # pd_op.divide: (-1x256x14x14xf32) <- (-1x256x14x14xf32, xf32)
        divide_4 = paddle._C_ops.divide(conv2d_20, full_8)

        # pd_op.multiply: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_11 = paddle._C_ops.add(add_9, multiply_4)

        # pd_op.split: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x256x14x14xf32, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(add_11, full_int_array_4, full_0)

        # builtin.split: (-1x64x14x14xf32, -1x192x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32])
        (
            split_19,
            split_20,
        ) = split_18
        del split_18

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            split_19, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56

        # builtin.combine: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x64x14x14xf32, -1x192x14x14xf32)
        combine_11 = [conv2d_21, split_20]

        # pd_op.concat: (-1x256x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_11, full_0)
        del combine_11

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            concat_6, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_55

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_54,
                parameter_53,
                parameter_52,
                parameter_51,
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
        del parameter_51, parameter_52, parameter_53, parameter_54

        # pd_op.gelu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        gelu_6 = paddle._C_ops.gelu(batch_norm__54, False)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x512x14x14xf32, 256x512x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            gelu_6, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_50

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x14x14xf32)
        shape64_5 = paddle._C_ops.shape64(conv2d_23)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_5

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_12 = [slice_5, full_2, full_2, full_2]
        del slice_5

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            stack_5,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_5

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_12 = paddle._C_ops.add(full_9, uniform_5)
        del uniform_5

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_12)
        del add_12

        # pd_op.divide: (-1x256x14x14xf32) <- (-1x256x14x14xf32, xf32)
        divide_5 = paddle._C_ops.divide(conv2d_23, full_9)

        # pd_op.multiply: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_13 = paddle._C_ops.add(add_11, multiply_5)

        # pd_op.split: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x256x14x14xf32, 2xi64, 1xi32)
        split_21 = paddle._C_ops.split(add_13, full_int_array_4, full_0)

        # builtin.split: (-1x64x14x14xf32, -1x192x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32])
        (
            split_22,
            split_23,
        ) = split_21
        del split_21

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            split_22, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # builtin.combine: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x64x14x14xf32, -1x192x14x14xf32)
        combine_13 = [conv2d_24, split_23]

        # pd_op.concat: (-1x256x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_13, full_0)
        del combine_13

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_7, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_48

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_47,
                parameter_46,
                parameter_45,
                parameter_44,
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
        del parameter_44, parameter_45, parameter_46, parameter_47

        # pd_op.gelu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        gelu_7 = paddle._C_ops.gelu(batch_norm__60, False)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x512x14x14xf32, 256x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            gelu_7, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_43

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [],
            float("0.988333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x14x14xf32)
        shape64_6 = paddle._C_ops.shape64(conv2d_26)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_6

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_14 = [slice_6, full_2, full_2, full_2]
        del slice_6

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            stack_6,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_6

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_14 = paddle._C_ops.add(full_10, uniform_6)
        del uniform_6

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_14)
        del add_14

        # pd_op.divide: (-1x256x14x14xf32) <- (-1x256x14x14xf32, xf32)
        divide_6 = paddle._C_ops.divide(conv2d_26, full_10)

        # pd_op.multiply: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_15 = paddle._C_ops.add(add_13, multiply_6)

        # pd_op.split: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x256x14x14xf32, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(add_15, full_int_array_4, full_0)

        # builtin.split: (-1x64x14x14xf32, -1x192x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32])
        (
            split_25,
            split_26,
        ) = split_24
        del split_24

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            split_25, parameter_42, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42

        # builtin.combine: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x64x14x14xf32, -1x192x14x14xf32)
        combine_15 = [conv2d_27, split_26]

        # pd_op.concat: (-1x256x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_15, full_0)
        del combine_15

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_8, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_41

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_40,
                parameter_39,
                parameter_38,
                parameter_37,
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
        del parameter_37, parameter_38, parameter_39, parameter_40

        # pd_op.gelu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        gelu_8 = paddle._C_ops.gelu(batch_norm__66, False)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x512x14x14xf32, 256x512x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            gelu_8, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_36

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [],
            float("0.986667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x14x14xf32)
        shape64_7 = paddle._C_ops.shape64(conv2d_29)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_7

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_16 = [slice_7, full_2, full_2, full_2]
        del slice_7

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            stack_7,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_7

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_16 = paddle._C_ops.add(full_11, uniform_7)
        del uniform_7

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_16)
        del add_16

        # pd_op.divide: (-1x256x14x14xf32) <- (-1x256x14x14xf32, xf32)
        divide_7 = paddle._C_ops.divide(conv2d_29, full_11)

        # pd_op.multiply: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_17 = paddle._C_ops.add(add_15, multiply_7)

        # pd_op.split: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x256x14x14xf32, 2xi64, 1xi32)
        split_27 = paddle._C_ops.split(add_17, full_int_array_4, full_0)

        # builtin.split: (-1x64x14x14xf32, -1x192x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32])
        (
            split_28,
            split_29,
        ) = split_27
        del split_27

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            split_28, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # builtin.combine: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x64x14x14xf32, -1x192x14x14xf32)
        combine_17 = [conv2d_30, split_29]

        # pd_op.concat: (-1x256x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_17, full_0)
        del combine_17

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            concat_9, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_34

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
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

        # pd_op.gelu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        gelu_9 = paddle._C_ops.gelu(batch_norm__72, False)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x512x14x14xf32, 256x512x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            gelu_9, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [],
            float("0.985"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x14x14xf32)
        shape64_8 = paddle._C_ops.shape64(conv2d_32)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_8

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_18 = [slice_8, full_2, full_2, full_2]
        del slice_8

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            stack_8,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_8

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_18 = paddle._C_ops.add(full_12, uniform_8)
        del uniform_8

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_18)
        del add_18

        # pd_op.divide: (-1x256x14x14xf32) <- (-1x256x14x14xf32, xf32)
        divide_8 = paddle._C_ops.divide(conv2d_32, full_12)

        # pd_op.multiply: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_19 = paddle._C_ops.add(add_17, multiply_8)

        # pd_op.split: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x256x14x14xf32, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(add_19, full_int_array_4, full_0)
        del full_int_array_4

        # builtin.split: (-1x64x14x14xf32, -1x192x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32])
        (
            split_31,
            split_32,
        ) = split_30
        del split_30

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            split_31, parameter_28, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_28

        # builtin.combine: ([-1x64x14x14xf32, -1x192x14x14xf32]) <- (-1x64x14x14xf32, -1x192x14x14xf32)
        combine_19 = [conv2d_33, split_32]

        # pd_op.concat: (-1x256x14x14xf32) <- ([-1x64x14x14xf32, -1x192x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_19, full_0)
        del combine_19

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_27

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_26,
                parameter_25,
                parameter_24,
                parameter_23,
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
        del parameter_23, parameter_24, parameter_25, parameter_26

        # pd_op.gelu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        gelu_10 = paddle._C_ops.gelu(batch_norm__78, False)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x512x14x14xf32, 256x512x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            gelu_10, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_22

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [],
            float("0.983333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x14x14xf32)
        shape64_9 = paddle._C_ops.shape64(conv2d_35)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_9

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_20 = [slice_9, full_2, full_2, full_2]
        del slice_9

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_20, 0)
        del combine_20

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            stack_9,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_9

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_20 = paddle._C_ops.add(full_13, uniform_9)
        del uniform_9

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_20)
        del add_20

        # pd_op.divide: (-1x256x14x14xf32) <- (-1x256x14x14xf32, xf32)
        divide_9 = paddle._C_ops.divide(conv2d_35, full_13)

        # pd_op.multiply: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_21 = paddle._C_ops.add(add_19, multiply_9)

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x14x14xf32, 512x256x2x2xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            add_21, parameter_21, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_21

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_36,
                parameter_20,
                parameter_19,
                parameter_18,
                parameter_17,
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
        del parameter_17, parameter_18, parameter_19, parameter_20

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [128, 384]

        # pd_op.split: ([-1x128x7x7xf32, -1x384x7x7xf32]) <- (-1x512x7x7xf32, 2xi64, 1xi32)
        split_33 = paddle._C_ops.split(batch_norm__84, full_int_array_5, full_0)

        # builtin.split: (-1x128x7x7xf32, -1x384x7x7xf32) <- ([-1x128x7x7xf32, -1x384x7x7xf32])
        (
            split_34,
            split_35,
        ) = split_33
        del split_33

        # pd_op.conv2d: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 128x128x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            split_34, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16

        # builtin.combine: ([-1x128x7x7xf32, -1x384x7x7xf32]) <- (-1x128x7x7xf32, -1x384x7x7xf32)
        combine_21 = [conv2d_37, split_35]

        # pd_op.concat: (-1x512x7x7xf32) <- ([-1x128x7x7xf32, -1x384x7x7xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_21, full_0)
        del combine_21

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x512x7x7xf32, 1024x512x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            concat_11, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_38,
                parameter_14,
                parameter_13,
                parameter_12,
                parameter_11,
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
        del parameter_11, parameter_12, parameter_13, parameter_14

        # pd_op.gelu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        gelu_11 = paddle._C_ops.gelu(batch_norm__90, False)

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x1024x7x7xf32, 512x1024x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            gelu_11, parameter_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_10

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [],
            float("0.981667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x512x7x7xf32)
        shape64_10 = paddle._C_ops.shape64(conv2d_39)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_10

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_22 = [slice_10, full_2, full_2, full_2]
        del slice_10

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_22, 0)
        del combine_22

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            stack_10,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_10

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_22 = paddle._C_ops.add(full_14, uniform_10)
        del uniform_10

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_22)
        del add_22

        # pd_op.divide: (-1x512x7x7xf32) <- (-1x512x7x7xf32, xf32)
        divide_10 = paddle._C_ops.divide(conv2d_39, full_14)

        # pd_op.multiply: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        add_23 = paddle._C_ops.add(batch_norm__84, multiply_10)

        # pd_op.split: ([-1x128x7x7xf32, -1x384x7x7xf32]) <- (-1x512x7x7xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(add_23, full_int_array_5, full_0)
        del full_int_array_5

        # builtin.split: (-1x128x7x7xf32, -1x384x7x7xf32) <- ([-1x128x7x7xf32, -1x384x7x7xf32])
        (
            split_37,
            split_38,
        ) = split_36
        del split_36

        # pd_op.conv2d: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 128x128x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            split_37, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9

        # builtin.combine: ([-1x128x7x7xf32, -1x384x7x7xf32]) <- (-1x128x7x7xf32, -1x384x7x7xf32)
        combine_23 = [conv2d_40, split_38]

        # pd_op.concat: (-1x512x7x7xf32) <- ([-1x128x7x7xf32, -1x384x7x7xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_23, full_0)
        del combine_23

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x512x7x7xf32, 1024x512x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            concat_12, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_8

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_41,
                parameter_7,
                parameter_6,
                parameter_5,
                parameter_4,
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
        del parameter_4, parameter_5, parameter_6, parameter_7

        # pd_op.gelu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        gelu_12 = paddle._C_ops.gelu(batch_norm__96, False)

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x1024x7x7xf32, 512x1024x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            gelu_12, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [],
            float("0.98"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x512x7x7xf32)
        shape64_11 = paddle._C_ops.shape64(conv2d_42)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_11

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_24 = [slice_11, full_2, full_2, full_2]
        del full_2, slice_11

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_24, 0)
        del combine_24

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            stack_11,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_3, full_4, stack_11

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_24 = paddle._C_ops.add(full_15, uniform_11)
        del uniform_11

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_24)
        del add_24

        # pd_op.divide: (-1x512x7x7xf32) <- (-1x512x7x7xf32, xf32)
        divide_11 = paddle._C_ops.divide(conv2d_42, full_15)

        # pd_op.multiply: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x1x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        add_25 = paddle._C_ops.add(add_23, multiply_11)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_25,
            full_int_array_6,
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

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x512x1x1xf32, 1280x512x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            pool2d_0, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2

        # pd_op.gelu: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        gelu_13 = paddle._C_ops.gelu(conv2d_43, False)

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(gelu_13, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del (
            add_1,
            add_11,
            add_13,
            add_15,
            add_17,
            add_19,
            add_21,
            add_23,
            add_25,
            add_3,
            add_5,
            add_7,
            add_9,
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
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__100,
            batch_norm__101,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__2,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__3,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
            batch_norm__5,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__6,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__7,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            batch_norm__8,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
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
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            concat_0,
            concat_1,
            concat_10,
            concat_11,
            concat_12,
            concat_2,
            concat_3,
            concat_4,
            concat_5,
            concat_6,
            concat_7,
            concat_8,
            concat_9,
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
            conv2d_30,
            conv2d_31,
            conv2d_32,
            conv2d_33,
            conv2d_34,
            conv2d_35,
            conv2d_36,
            conv2d_37,
            conv2d_38,
            conv2d_39,
            conv2d_4,
            conv2d_40,
            conv2d_41,
            conv2d_42,
            conv2d_43,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            divide_0,
            divide_1,
            divide_10,
            divide_11,
            divide_2,
            divide_3,
            divide_4,
            divide_5,
            divide_6,
            divide_7,
            divide_8,
            divide_9,
            flatten_0,
            floor_0,
            floor_1,
            floor_10,
            floor_11,
            floor_2,
            floor_3,
            floor_4,
            floor_5,
            floor_6,
            floor_7,
            floor_8,
            floor_9,
            full_0,
            full_1,
            full_10,
            full_11,
            full_12,
            full_13,
            full_14,
            full_15,
            full_5,
            full_6,
            full_7,
            full_8,
            full_9,
            full_int_array_6,
            gelu_0,
            gelu_1,
            gelu_10,
            gelu_11,
            gelu_12,
            gelu_13,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            gelu_6,
            gelu_7,
            gelu_8,
            gelu_9,
            matmul_0,
            multiply_0,
            multiply_1,
            multiply_10,
            multiply_11,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            parameter_0,
            pool2d_0,
            split_1,
            split_10,
            split_11,
            split_13,
            split_14,
            split_16,
            split_17,
            split_19,
            split_2,
            split_20,
            split_22,
            split_23,
            split_25,
            split_26,
            split_28,
            split_29,
            split_31,
            split_32,
            split_34,
            split_35,
            split_37,
            split_38,
            split_4,
            split_5,
            split_7,
            split_8,
        )

        return add_0
