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
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([512x4xf32, 512x4xf32]) <- (512x4xf32, 512x4xf32)
        combine_0 = [data_0, data_1]
        del data_0, data_1

        # pd_op.concat: (1024x4xf32) <- ([512x4xf32, 512x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, full_0

        # pd_op.cast: (2xi32) <- (2xi64)
        cast_0 = paddle._C_ops.cast(data_2, paddle.int32)
        del data_2

        # pd_op.roi_align: (1024x1024x14x14xf32) <- (2x1024x56x42xf32, 1024x4xf32, 2xi32)
        roi_align_0 = paddle._C_ops.roi_align(
            data_3, concat_0, cast_0, 14, 14, float("0.0625"), -1, True
        )
        del data_3

        # pd_op.conv2d: (1024x512x14x14xf32) <- (1024x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            roi_align_0, parameter_53, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_53

        # pd_op.batch_norm_: (1024x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1024x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_52,
                parameter_51,
                parameter_50,
                parameter_49,
                False,
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
        del parameter_49, parameter_50, parameter_51, parameter_52

        # pd_op.relu: (1024x512x14x14xf32) <- (1024x512x14x14xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (1024x512x7x7xf32) <- (1024x512x14x14xf32, 512x512x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_48, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_48

        # pd_op.batch_norm_: (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_47,
                parameter_46,
                parameter_45,
                parameter_44,
                False,
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
        del parameter_44, parameter_45, parameter_46, parameter_47

        # pd_op.relu: (1024x512x7x7xf32) <- (1024x512x7x7xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (1024x2048x7x7xf32) <- (1024x512x7x7xf32, 2048x512x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_43

        # pd_op.batch_norm_: (1024x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (1024x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_42,
                parameter_41,
                parameter_40,
                parameter_39,
                False,
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
        del parameter_39, parameter_40, parameter_41, parameter_42

        # pd_op.conv2d: (1024x2048x7x7xf32) <- (1024x1024x14x14xf32, 2048x1024x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            roi_align_0, parameter_38, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_38

        # pd_op.batch_norm_: (1024x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (1024x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_37,
                parameter_36,
                parameter_35,
                parameter_34,
                False,
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
        del parameter_34, parameter_35, parameter_36, parameter_37

        # pd_op.add: (1024x2048x7x7xf32) <- (1024x2048x7x7xf32, 1024x2048x7x7xf32)
        add_2 = paddle._C_ops.add(batch_norm__12, batch_norm__18)

        # pd_op.relu: (1024x2048x7x7xf32) <- (1024x2048x7x7xf32)
        relu_2 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.conv2d: (1024x512x7x7xf32) <- (1024x2048x7x7xf32, 512x2048x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_2, parameter_33, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_33

        # pd_op.batch_norm_: (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_32,
                parameter_31,
                parameter_30,
                parameter_29,
                False,
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
        del parameter_29, parameter_30, parameter_31, parameter_32

        # pd_op.relu: (1024x512x7x7xf32) <- (1024x512x7x7xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.conv2d: (1024x512x7x7xf32) <- (1024x512x7x7xf32, 512x512x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_3, parameter_28, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_28

        # pd_op.batch_norm_: (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_27,
                parameter_26,
                parameter_25,
                parameter_24,
                False,
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
        del parameter_24, parameter_25, parameter_26, parameter_27

        # pd_op.relu: (1024x512x7x7xf32) <- (1024x512x7x7xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (1024x2048x7x7xf32) <- (1024x512x7x7xf32, 2048x512x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_4, parameter_23, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_23

        # pd_op.batch_norm_: (1024x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (1024x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_22,
                parameter_21,
                parameter_20,
                parameter_19,
                False,
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
        del parameter_19, parameter_20, parameter_21, parameter_22

        # pd_op.add: (1024x2048x7x7xf32) <- (1024x2048x7x7xf32, 1024x2048x7x7xf32)
        add_3 = paddle._C_ops.add(batch_norm__36, relu_2)

        # pd_op.relu: (1024x2048x7x7xf32) <- (1024x2048x7x7xf32)
        relu_5 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.conv2d: (1024x512x7x7xf32) <- (1024x2048x7x7xf32, 512x2048x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_5, parameter_18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_18

        # pd_op.batch_norm_: (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_17,
                parameter_16,
                parameter_15,
                parameter_14,
                False,
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
        del parameter_14, parameter_15, parameter_16, parameter_17

        # pd_op.relu: (1024x512x7x7xf32) <- (1024x512x7x7xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (1024x512x7x7xf32) <- (1024x512x7x7xf32, 512x512x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_6, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_13

        # pd_op.batch_norm_: (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1024x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_12,
                parameter_11,
                parameter_10,
                parameter_9,
                False,
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
        del parameter_10, parameter_11, parameter_12, parameter_9

        # pd_op.relu: (1024x512x7x7xf32) <- (1024x512x7x7xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (1024x2048x7x7xf32) <- (1024x512x7x7xf32, 2048x512x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_7, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_8

        # pd_op.batch_norm_: (1024x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (1024x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_7,
                parameter_6,
                parameter_5,
                parameter_4,
                False,
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
        del parameter_4, parameter_5, parameter_6, parameter_7

        # pd_op.add: (1024x2048x7x7xf32) <- (1024x2048x7x7xf32, 1024x2048x7x7xf32)
        add_4 = paddle._C_ops.add(batch_norm__54, relu_5)

        # pd_op.relu: (1024x2048x7x7xf32) <- (1024x2048x7x7xf32)
        relu_8 = paddle._C_ops.relu(add_4)
        del add_4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (1024x2048x1x1xf32) <- (1024x2048x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_8,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 3]

        # pd_op.squeeze: (1024x2048xf32) <- (1024x2048x1x1xf32, 2xi64)
        squeeze_0 = paddle._C_ops.squeeze(pool2d_0, full_int_array_1)

        # pd_op.matmul: (1024x5xf32) <- (1024x2048xf32, 2048x5xf32)
        matmul_0 = paddle._C_ops.matmul(squeeze_0, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (1024x5xf32) <- (1024x5xf32, 5xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_2)
        del parameter_2

        # pd_op.matmul: (1024x16xf32) <- (1024x2048xf32, 2048x16xf32)
        matmul_1 = paddle._C_ops.matmul(squeeze_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (1024x16xf32) <- (1024x16xf32, 16xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_0)
        del (
            batch_norm__1,
            batch_norm__10,
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
            batch_norm__36,
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
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            cast_0,
            concat_0,
            conv2d_0,
            conv2d_1,
            conv2d_2,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            full_int_array_0,
            full_int_array_1,
            matmul_0,
            matmul_1,
            parameter_0,
            pool2d_0,
            relu_0,
            relu_1,
            relu_2,
            relu_3,
            relu_4,
            relu_5,
            relu_6,
            relu_7,
            relu_8,
            roi_align_0,
            squeeze_0,
        )

        return add_0, add_1
