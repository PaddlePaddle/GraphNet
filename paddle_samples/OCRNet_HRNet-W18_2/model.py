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
        data_0,
    ):
        # pd_op.conv2d: (2x512x-1x-1xf32) <- (2x-1x-1x-1xf32, 512x1024x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_5, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_4, full_int_array_0)

        # pd_op.add: (2x512x-1x-1xf32) <- (2x512x-1x-1xf32, 1x512x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.batch_norm_: (2x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__5,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_0,
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

        # pd_op.relu: (2x512x-1x-1xf32) <- (2x512x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__5)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1.11111"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x512x-1x-1xf32) <- (2x512x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(relu_0, full_0, float("0"), True)

        # pd_op.shape64: (4xi64) <- (2x512x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(relu_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [slice_0, slice_1, full_1, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            stack_0,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1],
            float("0.1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(uniform_0, full_4)

        # pd_op.cast: (2x512x-1x-1xf32) <- (2x512x-1x-1xf32)
        cast_0 = paddle._C_ops.cast(scale_0, paddle.float32)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_1 = paddle._C_ops.cast(greater_equal_0, paddle.float32)

        # pd_op.multiply: (2x512x-1x-1xf32) <- (2x512x-1x-1xf32, -1x-1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, cast_1)
        return (
            conv2d_0,
            reshape_0,
            add_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            relu_0,
            full_0,
            cast_0,
            cast_1,
            multiply_0,
        )
