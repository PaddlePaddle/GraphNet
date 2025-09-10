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
        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x3x224x224xf32, 96x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_113, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_113

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                True,
                float("0.1"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_0, parameter_109, parameter_110, parameter_111, parameter_112

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [24, 72]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split: ([-1x24x56x56xf32, -1x72x56x56xf32]) <- (-1x96x56x56xf32, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(batch_norm__0, full_int_array_0, full_0)
        del full_int_array_0

        # builtin.split: (-1x24x56x56xf32, -1x72x56x56xf32) <- ([-1x24x56x56xf32, -1x72x56x56xf32])
        (
            split_1,
            split_2,
        ) = split_0
        del split_0

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x24x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            split_1, parameter_108, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_108, split_1

        # builtin.combine: ([-1x24x56x56xf32, -1x72x56x56xf32]) <- (-1x24x56x56xf32, -1x72x56x56xf32)
        combine_0 = [conv2d_1, split_2]
        del conv2d_1, split_2

        # pd_op.concat: (-1x96x56x56xf32) <- ([-1x24x56x56xf32, -1x72x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x96x56x56xf32, 192x96x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            concat_0, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_0, parameter_107

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
        del conv2d_2, parameter_103, parameter_104, parameter_105, parameter_106

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x192x56x56xf32, 96x192x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_0, parameter_102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_102, relu_0

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        add_1 = paddle._C_ops.add(batch_norm__0, conv2d_3)
        del batch_norm__0, conv2d_3

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x96x56x56xf32, 192x96x2x2xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            add_1, parameter_101, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_1, parameter_101

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
        del conv2d_4, parameter_100, parameter_97, parameter_98, parameter_99

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [48, 144]

        # pd_op.split: ([-1x48x28x28xf32, -1x144x28x28xf32]) <- (-1x192x28x28xf32, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(batch_norm__12, full_int_array_1, full_0)

        # builtin.split: (-1x48x28x28xf32, -1x144x28x28xf32) <- ([-1x48x28x28xf32, -1x144x28x28xf32])
        (
            split_4,
            split_5,
        ) = split_3
        del split_3

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x48x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            split_4, parameter_96, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_96, split_4

        # builtin.combine: ([-1x48x28x28xf32, -1x144x28x28xf32]) <- (-1x48x28x28xf32, -1x144x28x28xf32)
        combine_1 = [conv2d_5, split_5]
        del conv2d_5, split_5

        # pd_op.concat: (-1x192x28x28xf32) <- ([-1x48x28x28xf32, -1x144x28x28xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x192x28x28xf32, 384x192x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            concat_1, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_1, parameter_95

        # pd_op.batch_norm_: (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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
        del conv2d_6, parameter_91, parameter_92, parameter_93, parameter_94

        # pd_op.relu: (-1x384x28x28xf32) <- (-1x384x28x28xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x384x28x28xf32, 192x384x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_1, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_90, relu_1

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        add_2 = paddle._C_ops.add(batch_norm__12, conv2d_7)
        del batch_norm__12, conv2d_7

        # pd_op.split: ([-1x48x28x28xf32, -1x144x28x28xf32]) <- (-1x192x28x28xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(add_2, full_int_array_1, full_0)
        del full_int_array_1

        # builtin.split: (-1x48x28x28xf32, -1x144x28x28xf32) <- ([-1x48x28x28xf32, -1x144x28x28xf32])
        (
            split_7,
            split_8,
        ) = split_6
        del split_6

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x48x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            split_7, parameter_89, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_89, split_7

        # builtin.combine: ([-1x48x28x28xf32, -1x144x28x28xf32]) <- (-1x48x28x28xf32, -1x144x28x28xf32)
        combine_2 = [conv2d_8, split_8]
        del conv2d_8, split_8

        # pd_op.concat: (-1x192x28x28xf32) <- ([-1x48x28x28xf32, -1x144x28x28xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        del combine_2

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x192x28x28xf32, 384x192x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            concat_2, parameter_88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_2, parameter_88

        # pd_op.batch_norm_: (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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
        del conv2d_9, parameter_84, parameter_85, parameter_86, parameter_87

        # pd_op.relu: (-1x384x28x28xf32) <- (-1x384x28x28xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x384x28x28xf32, 192x384x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_2, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_83, relu_2

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        add_3 = paddle._C_ops.add(add_2, conv2d_10)
        del add_2, conv2d_10

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x192x28x28xf32, 384x192x2x2xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_3, parameter_82, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_3, parameter_82

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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
        del conv2d_11, parameter_78, parameter_79, parameter_80, parameter_81

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [96, 288]

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(batch_norm__30, full_int_array_2, full_0)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            split_10,
            split_11,
        ) = split_9
        del split_9

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            split_10, parameter_77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77, split_10

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        combine_3 = [conv2d_12, split_11]
        del conv2d_12, split_11

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_3, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_3, parameter_76

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
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
        del conv2d_13, parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_3, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71, relu_3

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_4 = paddle._C_ops.add(batch_norm__30, conv2d_14)
        del batch_norm__30, conv2d_14

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(add_4, full_int_array_2, full_0)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            split_13,
            split_14,
        ) = split_12
        del split_12

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            split_13, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70, split_13

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        combine_4 = [conv2d_15, split_14]
        del conv2d_15, split_14

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            concat_4, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_4, parameter_69

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
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
        del conv2d_16, parameter_65, parameter_66, parameter_67, parameter_68

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_4, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64, relu_4

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_5 = paddle._C_ops.add(add_4, conv2d_17)
        del add_4, conv2d_17

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        split_15 = paddle._C_ops.split(add_5, full_int_array_2, full_0)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            split_16,
            split_17,
        ) = split_15
        del split_15

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            split_16, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_63, split_16

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        combine_5 = [conv2d_18, split_17]
        del conv2d_18, split_17

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            concat_5, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_5, parameter_62

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
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
        del conv2d_19, parameter_58, parameter_59, parameter_60, parameter_61

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_5, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_57, relu_5

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_6 = paddle._C_ops.add(add_5, conv2d_20)
        del add_5, conv2d_20

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(add_6, full_int_array_2, full_0)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            split_19,
            split_20,
        ) = split_18
        del split_18

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            split_19, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56, split_19

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        combine_6 = [conv2d_21, split_20]
        del conv2d_21, split_20

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)
        del combine_6

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            concat_6, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_6, parameter_55

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
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
        del conv2d_22, parameter_51, parameter_52, parameter_53, parameter_54

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__54)
        del batch_norm__54

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_6, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_50, relu_6

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_7 = paddle._C_ops.add(add_6, conv2d_23)
        del add_6, conv2d_23

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        split_21 = paddle._C_ops.split(add_7, full_int_array_2, full_0)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            split_22,
            split_23,
        ) = split_21
        del split_21

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            split_22, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49, split_22

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        combine_7 = [conv2d_24, split_23]
        del conv2d_24, split_23

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)
        del combine_7

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_7, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_7, parameter_48

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
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
        del conv2d_25, parameter_44, parameter_45, parameter_46, parameter_47

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_7, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_43, relu_7

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_8 = paddle._C_ops.add(add_7, conv2d_26)
        del add_7, conv2d_26

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(add_8, full_int_array_2, full_0)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            split_25,
            split_26,
        ) = split_24
        del split_24

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            split_25, parameter_42, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42, split_25

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        combine_8 = [conv2d_27, split_26]
        del conv2d_27, split_26

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_0)
        del combine_8

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_8, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_8, parameter_41

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
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
        del conv2d_28, parameter_37, parameter_38, parameter_39, parameter_40

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_8, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_36, relu_8

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_9 = paddle._C_ops.add(add_8, conv2d_29)
        del add_8, conv2d_29

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        split_27 = paddle._C_ops.split(add_9, full_int_array_2, full_0)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            split_28,
            split_29,
        ) = split_27
        del split_27

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            split_28, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35, split_28

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        combine_9 = [conv2d_30, split_29]
        del conv2d_30, split_29

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)
        del combine_9

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            concat_9, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_9, parameter_34

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
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
        del conv2d_31, parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_9, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29, relu_9

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_10 = paddle._C_ops.add(add_9, conv2d_32)
        del add_9, conv2d_32

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(add_10, full_int_array_2, full_0)
        del full_int_array_2

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            split_31,
            split_32,
        ) = split_30
        del split_30

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            split_31, parameter_28, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_28, split_31

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        combine_10 = [conv2d_33, split_32]
        del conv2d_33, split_32

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_0)
        del combine_10

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_10, parameter_27

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
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
        del conv2d_34, parameter_23, parameter_24, parameter_25, parameter_26

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_10, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_22, relu_10

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_11 = paddle._C_ops.add(add_10, conv2d_35)
        del add_10, conv2d_35

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x384x14x14xf32, 768x384x2x2xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            add_11, parameter_21, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_11, parameter_21

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
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
        del conv2d_36, parameter_17, parameter_18, parameter_19, parameter_20

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [192, 576]

        # pd_op.split: ([-1x192x7x7xf32, -1x576x7x7xf32]) <- (-1x768x7x7xf32, 2xi64, 1xi32)
        split_33 = paddle._C_ops.split(batch_norm__84, full_int_array_3, full_0)

        # builtin.split: (-1x192x7x7xf32, -1x576x7x7xf32) <- ([-1x192x7x7xf32, -1x576x7x7xf32])
        (
            split_34,
            split_35,
        ) = split_33
        del split_33

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x192x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            split_34, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16, split_34

        # builtin.combine: ([-1x192x7x7xf32, -1x576x7x7xf32]) <- (-1x192x7x7xf32, -1x576x7x7xf32)
        combine_11 = [conv2d_37, split_35]
        del conv2d_37, split_35

        # pd_op.concat: (-1x768x7x7xf32) <- ([-1x192x7x7xf32, -1x576x7x7xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_0)
        del combine_11

        # pd_op.conv2d: (-1x1536x7x7xf32) <- (-1x768x7x7xf32, 1536x768x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            concat_11, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_11, parameter_15

        # pd_op.batch_norm_: (-1x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
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
        del conv2d_38, parameter_11, parameter_12, parameter_13, parameter_14

        # pd_op.relu: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x1536x7x7xf32, 768x1536x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            relu_11, parameter_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_10, relu_11

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        add_12 = paddle._C_ops.add(batch_norm__84, conv2d_39)
        del batch_norm__84, conv2d_39

        # pd_op.split: ([-1x192x7x7xf32, -1x576x7x7xf32]) <- (-1x768x7x7xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(add_12, full_int_array_3, full_0)
        del full_int_array_3

        # builtin.split: (-1x192x7x7xf32, -1x576x7x7xf32) <- ([-1x192x7x7xf32, -1x576x7x7xf32])
        (
            split_37,
            split_38,
        ) = split_36
        del split_36

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x192x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            split_37, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9, split_37

        # builtin.combine: ([-1x192x7x7xf32, -1x576x7x7xf32]) <- (-1x192x7x7xf32, -1x576x7x7xf32)
        combine_12 = [conv2d_40, split_38]
        del conv2d_40, split_38

        # pd_op.concat: (-1x768x7x7xf32) <- ([-1x192x7x7xf32, -1x576x7x7xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_0)
        del combine_12, full_0

        # pd_op.conv2d: (-1x1536x7x7xf32) <- (-1x768x7x7xf32, 1536x768x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            concat_12, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_12, parameter_8

        # pd_op.batch_norm_: (-1x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
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
        del conv2d_41, parameter_4, parameter_5, parameter_6, parameter_7

        # pd_op.relu: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__96)
        del batch_norm__96

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x1536x7x7xf32, 768x1536x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            relu_12, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3, relu_12

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        add_13 = paddle._C_ops.add(add_12, conv2d_42)
        del add_12, conv2d_42

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.pool2d: (-1x768x1x1xf32) <- (-1x768x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_13,
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
        del add_13, full_int_array_4

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x768x1x1xf32, 1280x768x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            pool2d_0, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2, pool2d_0

        # pd_op.relu: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        relu_13 = paddle._C_ops.relu(conv2d_43)
        del conv2d_43

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(relu_13, 1, 3)
        del relu_13

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
