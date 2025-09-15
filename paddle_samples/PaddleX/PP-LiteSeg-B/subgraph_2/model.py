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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
    ):
        # pd_op.greater_equal: (xb) <- (xi64, xi64)
        greater_equal_0 = paddle._C_ops.greater_equal(data_1, data_2)
        del data_1, data_2

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(greater_equal_0, paddle.int64)
        del greater_equal_0

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_0)
        del cast_0, full_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_1, full_1)
        del cast_1, full_1

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.less_equal: (xb) <- (xi64, xi64)
        less_equal_0 = paddle._C_ops.less_equal(data_3, full_2)
        del data_3, full_2

        # pd_op.cast: (xi64) <- (xb)
        cast_2 = paddle._C_ops.cast(less_equal_0, paddle.int64)
        del less_equal_0

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_1 = paddle._C_ops.not_equal(cast_2, full_3)
        del cast_2, full_3

        # pd_op.conv2d: (1x96x-1x32xf32) <- (1x256x-1x32xf32, 96x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_4, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_4, parameter_19

        # pd_op.batch_norm_: (1x96x-1x32xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x32xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
        del conv2d_0, parameter_15, parameter_16, parameter_17, parameter_18

        # pd_op.relu: (1x96x-1x32xf32) <- (1x96x-1x32xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.shape64: (4xi64) <- (1x96x-1x32xf32)
        shape64_0 = paddle._C_ops.shape64(relu_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del full_int_array_0, full_int_array_1, shape64_0

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("32"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_0 = [slice_0, full_4]
        del full_4, slice_0

        # pd_op.bilinear_interp: (1x-1x-1x32xf32) <- (1x-1x-1x-1xf32, None, [xi64, 1xi32], None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            data_5, None, combine_0, None, "NCHW", -1, -1, 32, [], "bilinear", False, 0
        )
        del combine_0, data_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.mean: (1x1x-1x32xf32) <- (1x96x-1x32xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(relu_1, full_int_array_2, True)
        del full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.max: (1x1x-1x32xf32) <- (1x96x-1x32xf32, 1xi64)
        max_0 = paddle._C_ops.max(relu_1, full_int_array_3, True)
        del full_int_array_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.mean: (1x1x-1x32xf32) <- (1x-1x-1x32xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(bilinear_interp_0, full_int_array_4, True)
        del full_int_array_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.max: (1x1x-1x32xf32) <- (1x-1x-1x32xf32, 1xi64)
        max_1 = paddle._C_ops.max(bilinear_interp_0, full_int_array_5, True)
        del full_int_array_5

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x1x-1x32xf32, 1x1x-1x32xf32, 1x1x-1x32xf32, 1x1x-1x32xf32]) <- (1x1x-1x32xf32, 1x1x-1x32xf32, 1x1x-1x32xf32, 1x1x-1x32xf32)
        combine_1 = [mean_0, max_0, mean_1, max_1]
        del max_0, max_1, mean_0, mean_1

        # pd_op.concat: (1x4x-1x32xf32) <- ([1x1x-1x32xf32, 1x1x-1x32xf32, 1x1x-1x32xf32, 1x1x-1x32xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_5)
        del combine_1, full_5

        # pd_op.conv2d: (1x2x-1x32xf32) <- (1x4x-1x32xf32, 2x4x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            concat_0, parameter_14, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_0, parameter_14

        # pd_op.batch_norm_: (1x2x-1x32xf32, 2xf32, 2xf32, 2xf32, 2xf32, -1xui8) <- (1x2x-1x32xf32, 2xf32, 2xf32, 2xf32, 2xf32)
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
        del conv2d_1, parameter_10, parameter_11, parameter_12, parameter_13

        # pd_op.relu: (1x2x-1x32xf32) <- (1x2x-1x32xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (1x1x-1x32xf32) <- (1x2x-1x32xf32, 1x2x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_2, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9, relu_2

        # pd_op.batch_norm_: (1x1x-1x32xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (1x1x-1x32xf32, 1xf32, 1xf32, 1xf32, 1xf32)
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
        del conv2d_2, parameter_5, parameter_6, parameter_7, parameter_8

        # pd_op.sigmoid: (1x1x-1x32xf32) <- (1x1x-1x32xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(batch_norm__12)
        del batch_norm__12

        # pd_op.multiply: (1x96x-1x32xf32) <- (1x96x-1x32xf32, 1x1x-1x32xf32)
        multiply_0 = paddle._C_ops.multiply(relu_1, sigmoid_0)
        del relu_1

        # pd_op.subtract: (1x1x-1x32xf32) <- (1xf32, 1x1x-1x32xf32)
        subtract_0 = paddle._C_ops.subtract(data_0, sigmoid_0)
        del data_0, sigmoid_0

        # pd_op.multiply: (1x-1x-1x32xf32) <- (1x-1x-1x32xf32, 1x1x-1x32xf32)
        multiply_1 = paddle._C_ops.multiply(bilinear_interp_0, subtract_0)
        del bilinear_interp_0, subtract_0

        # pd_op.add: (1x96x-1x32xf32) <- (1x96x-1x32xf32, 1x-1x-1x32xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)
        del multiply_0, multiply_1

        # pd_op.conv2d: (1x64x-1x32xf32) <- (1x96x-1x32xf32, 64x96x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            add_0, parameter_4, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_0, parameter_4

        # pd_op.batch_norm_: (1x64x-1x32xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x-1x32xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
        del conv2d_3, parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.relu: (1x64x-1x32xf32) <- (1x64x-1x32xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        return relu_0
