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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.conv2d: (4x128x64x64xf32) <- (4x128x64x64xf32, 128x128x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_2, parameter_24, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__27,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
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

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__27)

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x128x64x64xf32, 128x128x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
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

        # pd_op.add: (4x128x64x64xf32) <- (4x128x64x64xf32, 4x128x64x64xf32)
        add_0 = paddle._C_ops.add(batch_norm__5, data_2)

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        relu_1 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x128x64x64xf32, 128x128x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_14, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__28,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
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

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__28)

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x128x64x64xf32, 128x128x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_2, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
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

        # pd_op.add: (4x128x64x64xf32) <- (4x128x64x64xf32, 4x128x64x64xf32)
        add_1 = paddle._C_ops.add(batch_norm__16, relu_1)

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        relu_3 = paddle._C_ops.relu(add_1)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([4x128x64x64xf32, 4x128x64x64xf32, 4x64x64x64xf32, 4x128x64x64xf32]) <- (4x128x64x64xf32, 4x128x64x64xf32, 4x64x64x64xf32, 4x128x64x64xf32)
        combine_0 = [relu_3, relu_1, data_0, data_1]

        # pd_op.concat: (4x448x64x64xf32) <- ([4x128x64x64xf32, 4x128x64x64xf32, 4x64x64x64xf32, 4x128x64x64xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x448x64x64xf32, 128x448x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            concat_0, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__29,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
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

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__29)
        return (
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            relu_0,
            conv2d_1,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            relu_1,
            conv2d_2,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            relu_2,
            conv2d_3,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            relu_3,
            full_0,
            concat_0,
            conv2d_4,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            relu_4,
        )
