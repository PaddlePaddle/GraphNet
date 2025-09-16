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
        data_0,
        data_1,
    ):
        # pd_op.full: (2x8x512xf32) <- ()
        full_0 = paddle._C_ops.full(
            [2, 8, 512],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (2x8x512xf32) <- (2x8x512xf32)
        assign_0 = full_0

        # pd_op.transpose: (26x8x512xf32) <- (8x26x512xf32)
        transpose_0 = paddle._C_ops.transpose(data_0, [1, 0, 2])
        del data_0

        # builtin.combine: ([2x8x512xf32, 2x8x512xf32]) <- (2x8x512xf32, 2x8x512xf32)
        combine_0 = [full_0, full_0]

        # builtin.combine: ([2048x512xf32, 2048x512xf32, 2048x512xf32, 2048x512xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32]) <- (2048x512xf32, 2048x512xf32, 2048x512xf32, 2048x512xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        combine_1 = [
            parameter_13,
            parameter_12,
            parameter_11,
            parameter_10,
            parameter_9,
            parameter_8,
            parameter_7,
            parameter_6,
        ]
        del (
            parameter_10,
            parameter_11,
            parameter_12,
            parameter_13,
            parameter_6,
            parameter_7,
            parameter_8,
            parameter_9,
        )

        # pd_op.rnn: (26x8x512xf32, 0xf32, [2x8x512xf32, 2x8x512xf32], 0xf32) <- (26x8x512xf32, [2x8x512xf32, 2x8x512xf32], [2048x512xf32, 2048x512xf32, 2048x512xf32, 2048x512xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32], None, xui8)
        rnn_1, rnn_0, rnn_2, rnn_3 = (lambda x, f: f(x))(
            paddle._C_ops.rnn(
                transpose_0,
                combine_0,
                combine_1,
                None,
                parameter_14,
                float("0"),
                False,
                512,
                512,
                2,
                "LSTM",
                0,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None),
        ) + (None,)
        del combine_0, combine_1, parameter_14

        # builtin.split: (2x8x512xf32, 2x8x512xf32) <- ([2x8x512xf32, 2x8x512xf32])
        (
            split_0,
            split_1,
        ) = rnn_2
        del rnn_2

        # pd_op.transpose: (8x26x512xf32) <- (26x8x512xf32)
        transpose_1 = paddle._C_ops.transpose(rnn_1, [1, 0, 2])

        # pd_op.matmul: (8x26x512xf32) <- (8x26x512xf32, 512x512xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_1, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (8x26x512xf32) <- (8x26x512xf32, 512xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_4)
        del parameter_4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 4]

        # pd_op.unsqueeze: (8x26x512x1x1xf32) <- (8x26x512xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(add_0, full_int_array_0)

        # pd_op.conv2d: (8x512x1x40xf32) <- (8x512x1x40xf32, 512x512x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_1, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_1, parameter_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_2, full_int_array_1)
        del full_int_array_1, parameter_2

        # pd_op.add: (8x512x1x40xf32) <- (8x512x1x40xf32, 1x512x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.unsqueeze: (8x1x512x1x40xf32) <- (8x512x1x40xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(add_1, full_int_array_2)

        # pd_op.add: (8x26x512x1x40xf32) <- (8x1x512x1x40xf32, 8x26x512x1x1xf32)
        add_2 = paddle._C_ops.add(unsqueeze_1, unsqueeze_0)

        # pd_op.tanh: (8x26x512x1x40xf32) <- (8x26x512x1x40xf32)
        tanh_0 = paddle._C_ops.tanh(add_2)
        del add_2

        # pd_op.transpose: (8x26x1x40x512xf32) <- (8x26x512x1x40xf32)
        transpose_2 = paddle._C_ops.transpose(tanh_0, [0, 1, 3, 4, 2])

        # pd_op.matmul: (8x26x1x40x1xf32) <- (8x26x1x40x512xf32, 512x1xf32)
        matmul_1 = paddle._C_ops.matmul(transpose_2, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (8x26x1x40x1xf32) <- (8x26x1x40x1xf32, 1xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_0)
        del parameter_0

        # pd_op.shape64: (5xi64) <- (8x26x1x40x1xf32)
        shape64_0 = paddle._C_ops.shape64(add_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.slice: (xi64) <- (5xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_3, full_int_array_2, [1], [0]
        )
        del full_int_array_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.slice: (xi64) <- (5xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (xi64) <- (5xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del full_int_array_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (xi64) <- (5xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del full_int_array_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [5]

        # pd_op.slice: (xi64) <- (5xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_6, full_int_array_7, [1], [0]
        )
        del full_int_array_6, full_int_array_7, shape64_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(slice_4, full_1)
        del (
            add_0,
            add_1,
            add_3,
            assign_0,
            conv2d_0,
            full_0,
            full_1,
            full_int_array_0,
            full_int_array_2,
            matmul_0,
            matmul_1,
            reshape_0,
            rnn_1,
            slice_4,
            tanh_0,
            transpose_0,
            transpose_1,
            transpose_2,
            unsqueeze_0,
            unsqueeze_1,
        )

        return rnn_0, split_0, split_1, equal_0, slice_0, slice_1, slice_2, slice_3
