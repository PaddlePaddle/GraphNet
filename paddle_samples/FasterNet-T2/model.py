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
        # pd_op.conv2d: (128x96x56x56xf32) <- (128x3x224x224xf32, 96x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_113, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [24, 72]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_20 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_18 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

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

        # pd_op.split: ([128x24x56x56xf32, 128x72x56x56xf32]) <- (128x96x56x56xf32, 2xi64, 1xi32)
        split_26 = paddle._C_ops.split(batch_norm__0, full_int_array_1, full_0)

        # builtin.split: (128x24x56x56xf32, 128x72x56x56xf32) <- ([128x24x56x56xf32, 128x72x56x56xf32])
        (
            split_0,
            split_1,
        ) = split_26

        # pd_op.conv2d: (128x24x56x56xf32) <- (128x24x56x56xf32, 24x24x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            split_0, parameter_108, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x24x56x56xf32, 128x72x56x56xf32]) <- (128x24x56x56xf32, 128x72x56x56xf32)
        combine_0 = [conv2d_1, split_1]

        # pd_op.concat: (128x96x56x56xf32) <- ([128x24x56x56xf32, 128x72x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (128x192x56x56xf32) <- (128x96x56x56xf32, 192x96x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            concat_0, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__89,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
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

        # pd_op.relu: (128x192x56x56xf32) <- (128x192x56x56xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__89)

        # pd_op.conv2d: (128x96x56x56xf32) <- (128x192x56x56xf32, 96x192x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_0, parameter_102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (128x96x56x56xf32) <- (128x96x56x56xf32, 128x96x56x56xf32)
        add_0 = paddle._C_ops.add(batch_norm__0, conv2d_3)

        # pd_op.conv2d: (128x192x28x28xf32) <- (128x96x56x56xf32, 192x96x2x2xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            add_0, parameter_101, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [48, 144]

        # pd_op.split: ([128x48x28x28xf32, 128x144x28x28xf32]) <- (128x192x28x28xf32, 2xi64, 1xi32)
        split_27 = paddle._C_ops.split(batch_norm__11, full_int_array_2, full_0)

        # builtin.split: (128x48x28x28xf32, 128x144x28x28xf32) <- ([128x48x28x28xf32, 128x144x28x28xf32])
        (
            split_2,
            split_3,
        ) = split_27

        # pd_op.conv2d: (128x48x28x28xf32) <- (128x48x28x28xf32, 48x48x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            split_2, parameter_96, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x48x28x28xf32, 128x144x28x28xf32]) <- (128x48x28x28xf32, 128x144x28x28xf32)
        combine_1 = [conv2d_5, split_3]

        # pd_op.concat: (128x192x28x28xf32) <- ([128x48x28x28xf32, 128x144x28x28xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.conv2d: (128x384x28x28xf32) <- (128x192x28x28xf32, 384x192x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            concat_1, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (128x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__90,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
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

        # pd_op.relu: (128x384x28x28xf32) <- (128x384x28x28xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__90)

        # pd_op.conv2d: (128x192x28x28xf32) <- (128x384x28x28xf32, 192x384x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_1, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full(
            [],
            float("0.995833"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [128, 1, 1, 1]

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_14 = paddle._C_ops.add(full_1, uniform_0)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_14)

        # pd_op.divide: (128x192x28x28xf32) <- (128x192x28x28xf32, xf32)
        divide_0 = paddle._C_ops.divide(conv2d_7, full_1)

        # pd_op.multiply: (128x192x28x28xf32) <- (128x192x28x28xf32, 128x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (128x192x28x28xf32) <- (128x192x28x28xf32, 128x192x28x28xf32)
        add_1 = paddle._C_ops.add(batch_norm__11, multiply_0)

        # pd_op.split: ([128x48x28x28xf32, 128x144x28x28xf32]) <- (128x192x28x28xf32, 2xi64, 1xi32)
        split_28 = paddle._C_ops.split(add_1, full_int_array_2, full_0)

        # builtin.split: (128x48x28x28xf32, 128x144x28x28xf32) <- ([128x48x28x28xf32, 128x144x28x28xf32])
        (
            split_4,
            split_5,
        ) = split_28

        # pd_op.conv2d: (128x48x28x28xf32) <- (128x48x28x28xf32, 48x48x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            split_4, parameter_89, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x48x28x28xf32, 128x144x28x28xf32]) <- (128x48x28x28xf32, 128x144x28x28xf32)
        combine_2 = [conv2d_8, split_5]

        # pd_op.concat: (128x192x28x28xf32) <- ([128x48x28x28xf32, 128x144x28x28xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)

        # pd_op.conv2d: (128x384x28x28xf32) <- (128x192x28x28xf32, 384x192x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            concat_2, parameter_88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (128x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__91,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
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

        # pd_op.relu: (128x384x28x28xf32) <- (128x384x28x28xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__91)

        # pd_op.conv2d: (128x192x28x28xf32) <- (128x384x28x28xf32, 192x384x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_2, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [],
            float("0.991667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_15 = paddle._C_ops.add(full_2, uniform_1)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_15)

        # pd_op.divide: (128x192x28x28xf32) <- (128x192x28x28xf32, xf32)
        divide_1 = paddle._C_ops.divide(conv2d_10, full_2)

        # pd_op.multiply: (128x192x28x28xf32) <- (128x192x28x28xf32, 128x1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (128x192x28x28xf32) <- (128x192x28x28xf32, 128x192x28x28xf32)
        add_2 = paddle._C_ops.add(add_1, multiply_1)

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x192x28x28xf32, 384x192x2x2xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_2, parameter_82, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (128x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [96, 288]

        # pd_op.split: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x384x14x14xf32, 2xi64, 1xi32)
        split_29 = paddle._C_ops.split(batch_norm__27, full_int_array_4, full_0)

        # builtin.split: (128x96x14x14xf32, 128x288x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32])
        (
            split_6,
            split_7,
        ) = split_29

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x96x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            split_6, parameter_77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x96x14x14xf32, 128x288x14x14xf32)
        combine_3 = [conv2d_12, split_7]

        # pd_op.concat: (128x384x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)

        # pd_op.conv2d: (128x768x14x14xf32) <- (128x384x14x14xf32, 768x384x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_3, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__92,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
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

        # pd_op.relu: (128x768x14x14xf32) <- (128x768x14x14xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__92)

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x768x14x14xf32, 384x768x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_3, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [],
            float("0.9875"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_16 = paddle._C_ops.add(full_3, uniform_2)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_16)

        # pd_op.divide: (128x384x14x14xf32) <- (128x384x14x14xf32, xf32)
        divide_2 = paddle._C_ops.divide(conv2d_14, full_3)

        # pd_op.multiply: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x384x14x14xf32)
        add_3 = paddle._C_ops.add(batch_norm__27, multiply_2)

        # pd_op.split: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x384x14x14xf32, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(add_3, full_int_array_4, full_0)

        # builtin.split: (128x96x14x14xf32, 128x288x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32])
        (
            split_8,
            split_9,
        ) = split_30

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x96x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            split_8, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x96x14x14xf32, 128x288x14x14xf32)
        combine_4 = [conv2d_15, split_9]

        # pd_op.concat: (128x384x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)

        # pd_op.conv2d: (128x768x14x14xf32) <- (128x384x14x14xf32, 768x384x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            concat_4, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__93,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
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

        # pd_op.relu: (128x768x14x14xf32) <- (128x768x14x14xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__93)

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x768x14x14xf32, 384x768x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_4, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [],
            float("0.983333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_17 = paddle._C_ops.add(full_4, uniform_3)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_17)

        # pd_op.divide: (128x384x14x14xf32) <- (128x384x14x14xf32, xf32)
        divide_3 = paddle._C_ops.divide(conv2d_17, full_4)

        # pd_op.multiply: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x384x14x14xf32)
        add_4 = paddle._C_ops.add(add_3, multiply_3)

        # pd_op.split: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x384x14x14xf32, 2xi64, 1xi32)
        split_31 = paddle._C_ops.split(add_4, full_int_array_4, full_0)

        # builtin.split: (128x96x14x14xf32, 128x288x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32])
        (
            split_10,
            split_11,
        ) = split_31

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x96x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            split_10, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x96x14x14xf32, 128x288x14x14xf32)
        combine_5 = [conv2d_18, split_11]

        # pd_op.concat: (128x384x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)

        # pd_op.conv2d: (128x768x14x14xf32) <- (128x384x14x14xf32, 768x384x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            concat_5, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__94,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
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

        # pd_op.relu: (128x768x14x14xf32) <- (128x768x14x14xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__94)

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x768x14x14xf32, 384x768x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_5, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("0.979167"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_18 = paddle._C_ops.add(full_5, uniform_4)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_18)

        # pd_op.divide: (128x384x14x14xf32) <- (128x384x14x14xf32, xf32)
        divide_4 = paddle._C_ops.divide(conv2d_20, full_5)

        # pd_op.multiply: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x384x14x14xf32)
        add_5 = paddle._C_ops.add(add_4, multiply_4)

        # pd_op.split: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x384x14x14xf32, 2xi64, 1xi32)
        split_32 = paddle._C_ops.split(add_5, full_int_array_4, full_0)

        # builtin.split: (128x96x14x14xf32, 128x288x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32])
        (
            split_12,
            split_13,
        ) = split_32

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x96x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            split_12, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x96x14x14xf32, 128x288x14x14xf32)
        combine_6 = [conv2d_21, split_13]

        # pd_op.concat: (128x384x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)

        # pd_op.conv2d: (128x768x14x14xf32) <- (128x384x14x14xf32, 768x384x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            concat_6, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__95,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
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

        # pd_op.relu: (128x768x14x14xf32) <- (128x768x14x14xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__95)

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x768x14x14xf32, 384x768x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_6, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [],
            float("0.975"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_19 = paddle._C_ops.add(full_6, uniform_5)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_19)

        # pd_op.divide: (128x384x14x14xf32) <- (128x384x14x14xf32, xf32)
        divide_5 = paddle._C_ops.divide(conv2d_23, full_6)

        # pd_op.multiply: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x384x14x14xf32)
        add_6 = paddle._C_ops.add(add_5, multiply_5)

        # pd_op.split: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x384x14x14xf32, 2xi64, 1xi32)
        split_33 = paddle._C_ops.split(add_6, full_int_array_4, full_0)

        # builtin.split: (128x96x14x14xf32, 128x288x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32])
        (
            split_14,
            split_15,
        ) = split_33

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x96x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            split_14, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x96x14x14xf32, 128x288x14x14xf32)
        combine_7 = [conv2d_24, split_15]

        # pd_op.concat: (128x384x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)

        # pd_op.conv2d: (128x768x14x14xf32) <- (128x384x14x14xf32, 768x384x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_7, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__96,
            batch_norm__53,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
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

        # pd_op.relu: (128x768x14x14xf32) <- (128x768x14x14xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__96)

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x768x14x14xf32, 384x768x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_7, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [],
            float("0.970833"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_20 = paddle._C_ops.add(full_7, uniform_6)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_20)

        # pd_op.divide: (128x384x14x14xf32) <- (128x384x14x14xf32, xf32)
        divide_6 = paddle._C_ops.divide(conv2d_26, full_7)

        # pd_op.multiply: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x384x14x14xf32)
        add_7 = paddle._C_ops.add(add_6, multiply_6)

        # pd_op.split: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x384x14x14xf32, 2xi64, 1xi32)
        split_34 = paddle._C_ops.split(add_7, full_int_array_4, full_0)

        # builtin.split: (128x96x14x14xf32, 128x288x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32])
        (
            split_16,
            split_17,
        ) = split_34

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x96x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            split_16, parameter_42, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x96x14x14xf32, 128x288x14x14xf32)
        combine_8 = [conv2d_27, split_17]

        # pd_op.concat: (128x384x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_0)

        # pd_op.conv2d: (128x768x14x14xf32) <- (128x384x14x14xf32, 768x384x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_8, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__97,
            batch_norm__58,
            batch_norm__59,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
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

        # pd_op.relu: (128x768x14x14xf32) <- (128x768x14x14xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__97)

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x768x14x14xf32, 384x768x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_8, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [],
            float("0.966667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_21 = paddle._C_ops.add(full_8, uniform_7)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_21)

        # pd_op.divide: (128x384x14x14xf32) <- (128x384x14x14xf32, xf32)
        divide_7 = paddle._C_ops.divide(conv2d_29, full_8)

        # pd_op.multiply: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x384x14x14xf32)
        add_8 = paddle._C_ops.add(add_7, multiply_7)

        # pd_op.split: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x384x14x14xf32, 2xi64, 1xi32)
        split_35 = paddle._C_ops.split(add_8, full_int_array_4, full_0)

        # builtin.split: (128x96x14x14xf32, 128x288x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32])
        (
            split_18,
            split_19,
        ) = split_35

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x96x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            split_18, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x96x14x14xf32, 128x288x14x14xf32)
        combine_9 = [conv2d_30, split_19]

        # pd_op.concat: (128x384x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)

        # pd_op.conv2d: (128x768x14x14xf32) <- (128x384x14x14xf32, 768x384x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            concat_9, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__98,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
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

        # pd_op.relu: (128x768x14x14xf32) <- (128x768x14x14xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__98)

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x768x14x14xf32, 384x768x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_9, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [],
            float("0.9625"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_22 = paddle._C_ops.add(full_9, uniform_8)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_22)

        # pd_op.divide: (128x384x14x14xf32) <- (128x384x14x14xf32, xf32)
        divide_8 = paddle._C_ops.divide(conv2d_32, full_9)

        # pd_op.multiply: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x384x14x14xf32)
        add_9 = paddle._C_ops.add(add_8, multiply_8)

        # pd_op.split: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x384x14x14xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(add_9, full_int_array_4, full_0)

        # builtin.split: (128x96x14x14xf32, 128x288x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32])
        (
            split_20,
            split_21,
        ) = split_36

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x96x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            split_20, parameter_28, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x96x14x14xf32, 128x288x14x14xf32]) <- (128x96x14x14xf32, 128x288x14x14xf32)
        combine_10 = [conv2d_33, split_21]

        # pd_op.concat: (128x384x14x14xf32) <- ([128x96x14x14xf32, 128x288x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_0)

        # pd_op.conv2d: (128x768x14x14xf32) <- (128x384x14x14xf32, 768x384x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (128x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__99,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
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

        # pd_op.relu: (128x768x14x14xf32) <- (128x768x14x14xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__99)

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x768x14x14xf32, 384x768x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_10, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [],
            float("0.958333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_23 = paddle._C_ops.add(full_10, uniform_9)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_23)

        # pd_op.divide: (128x384x14x14xf32) <- (128x384x14x14xf32, xf32)
        divide_9 = paddle._C_ops.divide(conv2d_35, full_10)

        # pd_op.multiply: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (128x384x14x14xf32) <- (128x384x14x14xf32, 128x384x14x14xf32)
        add_10 = paddle._C_ops.add(add_9, multiply_9)

        # pd_op.conv2d: (128x768x7x7xf32) <- (128x384x14x14xf32, 768x384x2x2xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            add_10, parameter_21, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (128x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [192, 576]

        # pd_op.split: ([128x192x7x7xf32, 128x576x7x7xf32]) <- (128x768x7x7xf32, 2xi64, 1xi32)
        split_37 = paddle._C_ops.split(batch_norm__73, full_int_array_5, full_0)

        # builtin.split: (128x192x7x7xf32, 128x576x7x7xf32) <- ([128x192x7x7xf32, 128x576x7x7xf32])
        (
            split_22,
            split_23,
        ) = split_37

        # pd_op.conv2d: (128x192x7x7xf32) <- (128x192x7x7xf32, 192x192x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            split_22, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x192x7x7xf32, 128x576x7x7xf32]) <- (128x192x7x7xf32, 128x576x7x7xf32)
        combine_11 = [conv2d_37, split_23]

        # pd_op.concat: (128x768x7x7xf32) <- ([128x192x7x7xf32, 128x576x7x7xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_0)

        # pd_op.conv2d: (128x1536x7x7xf32) <- (128x768x7x7xf32, 1536x768x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            concat_11, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (128x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__100,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
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

        # pd_op.relu: (128x1536x7x7xf32) <- (128x1536x7x7xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__100)

        # pd_op.conv2d: (128x768x7x7xf32) <- (128x1536x7x7xf32, 768x1536x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            relu_11, parameter_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [],
            float("0.954167"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_24 = paddle._C_ops.add(full_11, uniform_10)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_24)

        # pd_op.divide: (128x768x7x7xf32) <- (128x768x7x7xf32, xf32)
        divide_10 = paddle._C_ops.divide(conv2d_39, full_11)

        # pd_op.multiply: (128x768x7x7xf32) <- (128x768x7x7xf32, 128x1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (128x768x7x7xf32) <- (128x768x7x7xf32, 128x768x7x7xf32)
        add_11 = paddle._C_ops.add(batch_norm__73, multiply_10)

        # pd_op.split: ([128x192x7x7xf32, 128x576x7x7xf32]) <- (128x768x7x7xf32, 2xi64, 1xi32)
        split_38 = paddle._C_ops.split(add_11, full_int_array_5, full_0)

        # builtin.split: (128x192x7x7xf32, 128x576x7x7xf32) <- ([128x192x7x7xf32, 128x576x7x7xf32])
        (
            split_24,
            split_25,
        ) = split_38

        # pd_op.conv2d: (128x192x7x7xf32) <- (128x192x7x7xf32, 192x192x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            split_24, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x192x7x7xf32, 128x576x7x7xf32]) <- (128x192x7x7xf32, 128x576x7x7xf32)
        combine_12 = [conv2d_40, split_25]

        # pd_op.concat: (128x768x7x7xf32) <- ([128x192x7x7xf32, 128x576x7x7xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_0)

        # pd_op.conv2d: (128x1536x7x7xf32) <- (128x768x7x7xf32, 1536x768x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            concat_12, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (128x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__101,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
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

        # pd_op.relu: (128x1536x7x7xf32) <- (128x1536x7x7xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__101)

        # pd_op.conv2d: (128x768x7x7xf32) <- (128x1536x7x7xf32, 768x1536x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            relu_12, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [],
            float("0.95"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_13,
            full_14,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_25 = paddle._C_ops.add(full_12, uniform_11)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_25)

        # pd_op.divide: (128x768x7x7xf32) <- (128x768x7x7xf32, xf32)
        divide_11 = paddle._C_ops.divide(conv2d_42, full_12)

        # pd_op.multiply: (128x768x7x7xf32) <- (128x768x7x7xf32, 128x1x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (128x768x7x7xf32) <- (128x768x7x7xf32, 128x768x7x7xf32)
        add_12 = paddle._C_ops.add(add_11, multiply_11)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (128x768x1x1xf32) <- (128x768x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_12,
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

        # pd_op.conv2d: (128x1280x1x1xf32) <- (128x768x1x1xf32, 1280x768x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            pool2d_0, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.relu: (128x1280x1x1xf32) <- (128x1280x1x1xf32)
        relu_13 = paddle._C_ops.relu(conv2d_43)

        # pd_op.flatten: (128x1280xf32) <- (128x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(relu_13, 1, 3)

        # pd_op.matmul: (128x102xf32) <- (128x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)

        # pd_op.add: (128x102xf32) <- (128x102xf32, 102xf32)
        add_13 = paddle._C_ops.add(matmul_0, parameter_0)
        return (
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            full_0,
            split_0,
            split_1,
            conv2d_1,
            assign_0,
            concat_0,
            conv2d_2,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            relu_0,
            conv2d_3,
            add_0,
            conv2d_4,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            assign_1,
            split_2,
            split_3,
            conv2d_5,
            assign_2,
            concat_1,
            conv2d_6,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            relu_1,
            conv2d_7,
            full_1,
            floor_0,
            divide_0,
            multiply_0,
            add_1,
            assign_3,
            split_4,
            split_5,
            conv2d_8,
            assign_4,
            concat_2,
            conv2d_9,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            relu_2,
            conv2d_10,
            full_2,
            floor_1,
            divide_1,
            multiply_1,
            add_2,
            conv2d_11,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            assign_5,
            split_6,
            split_7,
            conv2d_12,
            assign_6,
            concat_3,
            conv2d_13,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            relu_3,
            conv2d_14,
            full_3,
            floor_2,
            divide_2,
            multiply_2,
            add_3,
            assign_7,
            split_8,
            split_9,
            conv2d_15,
            assign_8,
            concat_4,
            conv2d_16,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            relu_4,
            conv2d_17,
            full_4,
            floor_3,
            divide_3,
            multiply_3,
            add_4,
            assign_9,
            split_10,
            split_11,
            conv2d_18,
            assign_10,
            concat_5,
            conv2d_19,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            relu_5,
            conv2d_20,
            full_5,
            floor_4,
            divide_4,
            multiply_4,
            add_5,
            assign_11,
            split_12,
            split_13,
            conv2d_21,
            assign_12,
            concat_6,
            conv2d_22,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            relu_6,
            conv2d_23,
            full_6,
            floor_5,
            divide_5,
            multiply_5,
            add_6,
            assign_13,
            split_14,
            split_15,
            conv2d_24,
            assign_14,
            concat_7,
            conv2d_25,
            batch_norm__53,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            relu_7,
            conv2d_26,
            full_7,
            floor_6,
            divide_6,
            multiply_6,
            add_7,
            assign_15,
            split_16,
            split_17,
            conv2d_27,
            assign_16,
            concat_8,
            conv2d_28,
            batch_norm__58,
            batch_norm__59,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            relu_8,
            conv2d_29,
            full_8,
            floor_7,
            divide_7,
            multiply_7,
            add_8,
            assign_17,
            split_18,
            split_19,
            conv2d_30,
            assign_18,
            concat_9,
            conv2d_31,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            relu_9,
            conv2d_32,
            full_9,
            floor_8,
            divide_8,
            multiply_8,
            add_9,
            assign_19,
            split_20,
            split_21,
            conv2d_33,
            assign_20,
            concat_10,
            conv2d_34,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            relu_10,
            conv2d_35,
            full_10,
            floor_9,
            divide_9,
            multiply_9,
            add_10,
            conv2d_36,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            assign_21,
            split_22,
            split_23,
            conv2d_37,
            assign_22,
            concat_11,
            conv2d_38,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            relu_11,
            conv2d_39,
            full_11,
            floor_10,
            divide_10,
            multiply_10,
            add_11,
            assign_23,
            split_24,
            split_25,
            conv2d_40,
            assign_24,
            concat_12,
            conv2d_41,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            relu_12,
            conv2d_42,
            full_12,
            floor_11,
            divide_11,
            multiply_11,
            add_12,
            full_int_array_0,
            pool2d_0,
            relu_13,
            flatten_0,
            matmul_0,
            add_13,
        )
