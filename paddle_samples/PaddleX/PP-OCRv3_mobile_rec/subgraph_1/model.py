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
        data_0,
        data_1,
    ):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (8x512x1x40xf32) <- (8x512x1x40xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            data_0,
            full_int_array_0,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )
        del data_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.squeeze: (8x512x40xf32) <- (8x512x1x40xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(pool2d_0, full_int_array_1)

        # pd_op.transpose: (8x40x512xf32) <- (8x512x40xf32)
        transpose_0 = paddle._C_ops.transpose(squeeze_0, [0, 2, 1])
        del squeeze_0

        # pd_op.full: (2x8x512xf32) <- ()
        full_0 = paddle._C_ops.full(
            [2, 8, 512],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (2x8x512xf32) <- (2x8x512xf32)
        assign_0 = full_0

        # pd_op.transpose: (40x8x512xf32) <- (8x40x512xf32)
        transpose_1 = paddle._C_ops.transpose(transpose_0, [1, 0, 2])
        del transpose_0

        # builtin.combine: ([2x8x512xf32, 2x8x512xf32]) <- (2x8x512xf32, 2x8x512xf32)
        combine_0 = [full_0, full_0]

        # builtin.combine: ([2048x512xf32, 2048x512xf32, 2048x512xf32, 2048x512xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32]) <- (2048x512xf32, 2048x512xf32, 2048x512xf32, 2048x512xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        combine_1 = [
            parameter_9,
            parameter_8,
            parameter_7,
            parameter_6,
            parameter_5,
            parameter_4,
            parameter_3,
            parameter_2,
        ]
        del (
            parameter_2,
            parameter_3,
            parameter_4,
            parameter_5,
            parameter_6,
            parameter_7,
            parameter_8,
            parameter_9,
        )

        # pd_op.rnn: (40x8x512xf32, 0xf32, [2x8x512xf32, 2x8x512xf32], 0xf32) <- (40x8x512xf32, [2x8x512xf32, 2x8x512xf32], [2048x512xf32, 2048x512xf32, 2048x512xf32, 2048x512xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32], None, xui8)
        rnn_1, rnn_0, rnn_2, rnn_3 = (lambda x, f: f(x))(
            paddle._C_ops.rnn(
                transpose_1,
                combine_0,
                combine_1,
                None,
                parameter_10,
                float("0.1"),
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
        del combine_0, combine_1, parameter_10

        # builtin.split: (2x8x512xf32, 2x8x512xf32) <- ([2x8x512xf32, 2x8x512xf32])
        (
            split_0,
            split_1,
        ) = rnn_2
        del rnn_2

        # pd_op.transpose: (8x40x512xf32) <- (40x8x512xf32)
        transpose_2 = paddle._C_ops.transpose(rnn_1, [1, 0, 2])

        # pd_op.shape64: (3xi64) <- (8x40x512xf32)
        shape64_0 = paddle._C_ops.shape64(transpose_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_1, [1], [0]
        )
        del shape64_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.slice: (xf64) <- (8xf64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_3, full_int_array_2, [1], [0]
        )
        del full_int_array_3

        # pd_op.cast: (xf64) <- (xi64)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.float64)

        # pd_op.multiply: (xf64) <- (xf64, xf64)
        multiply_0 = paddle._C_ops.multiply(slice_1, cast_0)
        del slice_1

        # pd_op.ceil: (xf64) <- (xf64)
        ceil_0 = paddle._C_ops.ceil(multiply_0)
        del multiply_0

        # pd_op.cast: (xi64) <- (xf64)
        cast_1 = paddle._C_ops.cast(ceil_0, paddle.int64)
        del ceil_0

        # pd_op.minimum: (xi64) <- (xi64, xi64)
        minimum_0 = paddle._C_ops.minimum(slice_0, cast_1)
        del cast_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(minimum_0, full_1, float("-1"), True)
        del minimum_0

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_1, float("1"), True)

        # pd_op.full: (1xi64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_4 = []

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_0 = paddle._C_ops.reshape(full_2, full_int_array_4)
        del full_2

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_1 = paddle._C_ops.reshape(full_3, full_int_array_4)
        del full_3

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_2 = [reshape_0, scale_0]
        del reshape_0, scale_0

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_3 = [reshape_1, scale_1]
        del scale_1

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.slice: (512xf32) <- (8x40x512xf32, 2xi64, 2xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_2, [0, 1], stack_0, stack_1, [1, -1], [0, 1]
        )

        # pd_op.slice: (xf64) <- (8xf64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_1, [0], full_int_array_2, full_int_array_1, [1], [0]
        )
        del full_int_array_2

        # pd_op.multiply: (xf64) <- (xf64, xf64)
        multiply_1 = paddle._C_ops.multiply(slice_3, cast_0)
        del slice_3

        # pd_op.ceil: (xf64) <- (xf64)
        ceil_1 = paddle._C_ops.ceil(multiply_1)
        del multiply_1

        # pd_op.cast: (xi64) <- (xf64)
        cast_2 = paddle._C_ops.cast(ceil_1, paddle.int64)
        del ceil_1

        # pd_op.minimum: (xi64) <- (xi64, xi64)
        minimum_1 = paddle._C_ops.minimum(slice_0, cast_2)
        del cast_2

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(minimum_1, full_1, float("-1"), True)
        del minimum_1

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_1, float("1"), True)

        # pd_op.full: (1xi64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_2 = paddle._C_ops.reshape(full_4, full_int_array_4)
        del full_4

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_4 = [reshape_1, scale_2]
        del reshape_1, scale_2

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_5 = [reshape_2, scale_3]
        del scale_3

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.slice: (512xf32) <- (8x40x512xf32, 2xi64, 2xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_2, [0, 1], stack_2, stack_3, [1, -1], [0, 1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (xf64) <- (8xf64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.multiply: (xf64) <- (xf64, xf64)
        multiply_2 = paddle._C_ops.multiply(slice_5, cast_0)
        del slice_5

        # pd_op.ceil: (xf64) <- (xf64)
        ceil_2 = paddle._C_ops.ceil(multiply_2)
        del multiply_2

        # pd_op.cast: (xi64) <- (xf64)
        cast_3 = paddle._C_ops.cast(ceil_2, paddle.int64)
        del ceil_2

        # pd_op.minimum: (xi64) <- (xi64, xi64)
        minimum_2 = paddle._C_ops.minimum(slice_0, cast_3)
        del cast_3

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_4 = paddle._C_ops.scale(minimum_2, full_1, float("-1"), True)
        del minimum_2

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_1, float("1"), True)

        # pd_op.full: (1xi64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_3 = paddle._C_ops.reshape(full_5, full_int_array_4)
        del full_5

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_6 = [reshape_2, scale_4]
        del reshape_2, scale_4

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_7 = [reshape_3, scale_5]
        del scale_5

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.slice: (512xf32) <- (8x40x512xf32, 2xi64, 2xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_2, [0, 1], stack_4, stack_5, [1, -1], [0, 1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (xf64) <- (8xf64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del full_int_array_5

        # pd_op.multiply: (xf64) <- (xf64, xf64)
        multiply_3 = paddle._C_ops.multiply(slice_7, cast_0)
        del slice_7

        # pd_op.ceil: (xf64) <- (xf64)
        ceil_3 = paddle._C_ops.ceil(multiply_3)
        del multiply_3

        # pd_op.cast: (xi64) <- (xf64)
        cast_4 = paddle._C_ops.cast(ceil_3, paddle.int64)
        del ceil_3

        # pd_op.minimum: (xi64) <- (xi64, xi64)
        minimum_3 = paddle._C_ops.minimum(slice_0, cast_4)
        del cast_4

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_6 = paddle._C_ops.scale(minimum_3, full_1, float("-1"), True)
        del minimum_3

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_1, float("1"), True)

        # pd_op.full: (1xi64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_4 = paddle._C_ops.reshape(full_6, full_int_array_4)
        del full_6

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_8 = [reshape_3, scale_6]
        del reshape_3, scale_6

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_9 = [reshape_4, scale_7]
        del scale_7

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.slice: (512xf32) <- (8x40x512xf32, 2xi64, 2xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_2, [0, 1], stack_6, stack_7, [1, -1], [0, 1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [5]

        # pd_op.slice: (xf64) <- (8xf64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_1, [0], full_int_array_6, full_int_array_7, [1], [0]
        )
        del full_int_array_6

        # pd_op.multiply: (xf64) <- (xf64, xf64)
        multiply_4 = paddle._C_ops.multiply(slice_9, cast_0)
        del slice_9

        # pd_op.ceil: (xf64) <- (xf64)
        ceil_4 = paddle._C_ops.ceil(multiply_4)
        del multiply_4

        # pd_op.cast: (xi64) <- (xf64)
        cast_5 = paddle._C_ops.cast(ceil_4, paddle.int64)
        del ceil_4

        # pd_op.minimum: (xi64) <- (xi64, xi64)
        minimum_4 = paddle._C_ops.minimum(slice_0, cast_5)
        del cast_5

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_8 = paddle._C_ops.scale(minimum_4, full_1, float("-1"), True)
        del minimum_4

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_1, float("1"), True)

        # pd_op.full: (1xi64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("5"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_5 = paddle._C_ops.reshape(full_7, full_int_array_4)
        del full_7

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_10 = [reshape_4, scale_8]
        del reshape_4, scale_8

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_11 = [reshape_5, scale_9]
        del scale_9

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.slice: (512xf32) <- (8x40x512xf32, 2xi64, 2xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_2, [0, 1], stack_8, stack_9, [1, -1], [0, 1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [6]

        # pd_op.slice: (xf64) <- (8xf64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_1, [0], full_int_array_7, full_int_array_8, [1], [0]
        )
        del full_int_array_7

        # pd_op.multiply: (xf64) <- (xf64, xf64)
        multiply_5 = paddle._C_ops.multiply(slice_11, cast_0)
        del slice_11

        # pd_op.ceil: (xf64) <- (xf64)
        ceil_5 = paddle._C_ops.ceil(multiply_5)
        del multiply_5

        # pd_op.cast: (xi64) <- (xf64)
        cast_6 = paddle._C_ops.cast(ceil_5, paddle.int64)
        del ceil_5

        # pd_op.minimum: (xi64) <- (xi64, xi64)
        minimum_5 = paddle._C_ops.minimum(slice_0, cast_6)
        del cast_6

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_10 = paddle._C_ops.scale(minimum_5, full_1, float("-1"), True)
        del minimum_5

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_11 = paddle._C_ops.scale(scale_10, full_1, float("1"), True)

        # pd_op.full: (1xi64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("6"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_6 = paddle._C_ops.reshape(full_8, full_int_array_4)
        del full_8

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_12 = [reshape_5, scale_10]
        del reshape_5, scale_10

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_13 = [reshape_6, scale_11]
        del scale_11

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.slice: (512xf32) <- (8x40x512xf32, 2xi64, 2xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_2, [0, 1], stack_10, stack_11, [1, -1], [0, 1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [7]

        # pd_op.slice: (xf64) <- (8xf64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            data_1, [0], full_int_array_8, full_int_array_9, [1], [0]
        )
        del full_int_array_8

        # pd_op.multiply: (xf64) <- (xf64, xf64)
        multiply_6 = paddle._C_ops.multiply(slice_13, cast_0)
        del slice_13

        # pd_op.ceil: (xf64) <- (xf64)
        ceil_6 = paddle._C_ops.ceil(multiply_6)
        del multiply_6

        # pd_op.cast: (xi64) <- (xf64)
        cast_7 = paddle._C_ops.cast(ceil_6, paddle.int64)
        del ceil_6

        # pd_op.minimum: (xi64) <- (xi64, xi64)
        minimum_6 = paddle._C_ops.minimum(slice_0, cast_7)
        del cast_7

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_12 = paddle._C_ops.scale(minimum_6, full_1, float("-1"), True)
        del minimum_6

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_13 = paddle._C_ops.scale(scale_12, full_1, float("1"), True)

        # pd_op.full: (1xi64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("7"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_7 = paddle._C_ops.reshape(full_9, full_int_array_4)
        del full_9

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_14 = [reshape_6, scale_12]
        del reshape_6, scale_12

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_15 = [reshape_7, scale_13]
        del scale_13

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.slice: (512xf32) <- (8x40x512xf32, 2xi64, 2xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_2, [0, 1], stack_12, stack_13, [1, -1], [0, 1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [8]

        # pd_op.slice: (xf64) <- (8xf64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            data_1, [0], full_int_array_9, full_int_array_10, [1], [0]
        )
        del data_1, full_int_array_10, full_int_array_9

        # pd_op.multiply: (xf64) <- (xf64, xf64)
        multiply_7 = paddle._C_ops.multiply(slice_15, cast_0)
        del cast_0, slice_15

        # pd_op.ceil: (xf64) <- (xf64)
        ceil_7 = paddle._C_ops.ceil(multiply_7)
        del multiply_7

        # pd_op.cast: (xi64) <- (xf64)
        cast_8 = paddle._C_ops.cast(ceil_7, paddle.int64)
        del ceil_7

        # pd_op.minimum: (xi64) <- (xi64, xi64)
        minimum_7 = paddle._C_ops.minimum(slice_0, cast_8)
        del cast_8, slice_0

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_14 = paddle._C_ops.scale(minimum_7, full_1, float("-1"), True)
        del minimum_7

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_15 = paddle._C_ops.scale(scale_14, full_1, float("1"), True)
        del full_1

        # pd_op.full: (1xi64) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_8 = paddle._C_ops.reshape(full_10, full_int_array_4)
        del full_10, full_int_array_4

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_16 = [reshape_7, scale_14]
        del reshape_7, scale_14

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_17 = [reshape_8, scale_15]
        del reshape_8, scale_15

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_17, 0)
        del combine_17

        # pd_op.slice: (512xf32) <- (8x40x512xf32, 2xi64, 2xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_2, [0, 1], stack_14, stack_15, [1, -1], [0, 1]
        )

        # builtin.combine: ([512xf32, 512xf32, 512xf32, 512xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512xf32, 512xf32, 512xf32, 512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_18 = [
            slice_2,
            slice_4,
            slice_6,
            slice_8,
            slice_10,
            slice_12,
            slice_14,
            slice_16,
        ]

        # pd_op.stack: (8x512xf32) <- ([512xf32, 512xf32, 512xf32, 512xf32, 512xf32, 512xf32, 512xf32, 512xf32])
        stack_16 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.matmul: (8x512xf32) <- (8x512xf32, 512x512xf32)
        matmul_0 = paddle._C_ops.matmul(stack_16, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (8x512xf32) <- (8x512xf32, 512xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del (
            assign_0,
            full_0,
            full_int_array_0,
            full_int_array_1,
            matmul_0,
            parameter_0,
            pool2d_0,
            rnn_1,
            slice_10,
            slice_12,
            slice_14,
            slice_16,
            slice_2,
            slice_4,
            slice_6,
            slice_8,
            stack_0,
            stack_1,
            stack_10,
            stack_11,
            stack_12,
            stack_13,
            stack_14,
            stack_15,
            stack_16,
            stack_2,
            stack_3,
            stack_4,
            stack_5,
            stack_6,
            stack_7,
            stack_8,
            stack_9,
            transpose_1,
            transpose_2,
        )

        return rnn_0, split_0, split_1, add_0
