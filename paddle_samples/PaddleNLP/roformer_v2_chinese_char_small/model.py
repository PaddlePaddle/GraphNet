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
        data_0,
        data_1,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (1x11xb) <- (1x11xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_0)
        del full_0

        # pd_op.cast: (1x11xf32) <- (1x11xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x11xf32) <- (1x11xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_1, float("0"), True)
        del cast_0, full_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2]

        # pd_op.unsqueeze: (1x1x1x11xf32) <- (1x11xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(scale_0, full_int_array_0)
        del full_int_array_0, scale_0

        # pd_op.embedding: (1x11x384xf32) <- (1x11xi64, 12000x384xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_49, -1, False)
        del data_0, parameter_49

        # pd_op.embedding: (1x11x384xf32) <- (1x11xi64, 2x384xf32)
        embedding_1 = paddle._C_ops.embedding(data_1, parameter_48, -1, False)
        del data_1, parameter_48

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)
        del embedding_0, embedding_1

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_0 = paddle._C_ops.square(add_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(square_0, full_int_array_1, True)
        del square_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(mean_0, full_2, float("1e-12"), True)
        del mean_0

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(scale_1)
        del scale_1

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_1 = paddle._C_ops.divide(add_0, sqrt_0)
        del add_0, sqrt_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                divide_1, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del divide_1

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_47, False, False)
        del parameter_47

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_46, False, False)
        del parameter_46

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_45, False, False)
        del parameter_45

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [0, 0, 6, 64]

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(matmul_0, full_int_array_2)
        del matmul_0

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(matmul_1, full_int_array_2)
        del matmul_1

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(matmul_2, full_int_array_2)
        del matmul_2

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [11]

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            parameter_11, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_11

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            parameter_10, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_10

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_0

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_0 = paddle._C_ops.multiply(strided_slice_0, slice_1)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_1 = paddle._C_ops.multiply(strided_slice_1, slice_0)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_0 = paddle._C_ops.subtract(multiply_0, multiply_1)
        del multiply_0, multiply_1

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_2 = paddle._C_ops.multiply(strided_slice_0, slice_0)
        del strided_slice_0

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_3 = paddle._C_ops.multiply(strided_slice_1, slice_1)
        del strided_slice_1

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_1 = paddle._C_ops.add(multiply_2, multiply_3)
        del multiply_2, multiply_3

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_0 = [subtract_0, add_1]
        del add_1, subtract_0

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_0 = paddle._C_ops.flatten(stack_0, 3, 4)
        del stack_0

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_1

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_4 = paddle._C_ops.multiply(strided_slice_2, slice_1)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_5 = paddle._C_ops.multiply(strided_slice_3, slice_0)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_1 = paddle._C_ops.subtract(multiply_4, multiply_5)
        del multiply_4, multiply_5

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_6 = paddle._C_ops.multiply(strided_slice_2, slice_0)
        del slice_0, strided_slice_2

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_7 = paddle._C_ops.multiply(strided_slice_3, slice_1)
        del slice_1, strided_slice_3

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_2 = paddle._C_ops.add(multiply_6, multiply_7)
        del multiply_6, multiply_7

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_1 = [subtract_1, add_2]
        del add_2, subtract_1

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_1 = paddle._C_ops.flatten(stack_1, 3, 4)
        del stack_1

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_3 = paddle._C_ops.matmul(flatten_0, flatten_1, False, True)
        del flatten_0, flatten_1

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x6x11x11xf32) <- (1x6x11x11xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_3, full_4, float("0"), True)
        del matmul_3

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_3 = paddle._C_ops.add(scale_2, unsqueeze_0)
        del scale_2

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_0 = paddle._C_ops.softmax(add_3, -1)
        del add_3

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [0, 0, 384]

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_8)
        del transpose_3

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_44, False, False)
        del parameter_44, reshape_3

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_5, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_5

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_4 = paddle._C_ops.add(dropout_0, dropout_4)
        del dropout_0, dropout_4

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_1 = paddle._C_ops.square(add_4)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(square_1, full_int_array_1, True)
        del square_1

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(mean_1, full_2, float("1e-12"), True)
        del mean_1

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_1 = paddle._C_ops.sqrt(scale_3)
        del scale_3

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_2 = paddle._C_ops.divide(add_4, sqrt_1)
        del add_4, sqrt_1

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_6 = paddle._C_ops.matmul(divide_2, parameter_43, False, False)
        del parameter_43

        # pd_op.relu: (1x11x1536xf32) <- (1x11x1536xf32)
        relu_0 = paddle._C_ops.relu(matmul_6)
        del matmul_6

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_7 = paddle._C_ops.matmul(relu_0, parameter_42, False, False)
        del parameter_42, relu_0

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_7, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_7

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_5 = paddle._C_ops.add(divide_2, dropout_6)
        del divide_2, dropout_6

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_2 = paddle._C_ops.square(add_5)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_2 = paddle._C_ops.mean(square_2, full_int_array_1, True)
        del square_2

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(mean_2, full_2, float("1e-12"), True)
        del mean_2

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_2 = paddle._C_ops.sqrt(scale_4)
        del scale_4

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_3 = paddle._C_ops.divide(add_5, sqrt_2)
        del add_5, sqrt_2

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_8 = paddle._C_ops.matmul(divide_3, parameter_41, False, False)
        del parameter_41

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_9 = paddle._C_ops.matmul(divide_3, parameter_40, False, False)
        del parameter_40

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_10 = paddle._C_ops.matmul(divide_3, parameter_39, False, False)
        del parameter_39

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(matmul_8, full_int_array_2)
        del matmul_8

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(matmul_9, full_int_array_2)
        del matmul_9

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(matmul_10, full_int_array_2)
        del matmul_10

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            parameter_9, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_9

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            parameter_8, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_8

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_4

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_8 = paddle._C_ops.multiply(strided_slice_4, slice_3)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_9 = paddle._C_ops.multiply(strided_slice_5, slice_2)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_2 = paddle._C_ops.subtract(multiply_8, multiply_9)
        del multiply_8, multiply_9

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_10 = paddle._C_ops.multiply(strided_slice_4, slice_2)
        del strided_slice_4

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_11 = paddle._C_ops.multiply(strided_slice_5, slice_3)
        del strided_slice_5

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_6 = paddle._C_ops.add(multiply_10, multiply_11)
        del multiply_10, multiply_11

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_2 = [subtract_2, add_6]
        del add_6, subtract_2

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_2 = paddle._C_ops.flatten(stack_2, 3, 4)
        del stack_2

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_5

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_12 = paddle._C_ops.multiply(strided_slice_6, slice_3)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_13 = paddle._C_ops.multiply(strided_slice_7, slice_2)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_3 = paddle._C_ops.subtract(multiply_12, multiply_13)
        del multiply_12, multiply_13

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_14 = paddle._C_ops.multiply(strided_slice_6, slice_2)
        del slice_2, strided_slice_6

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_15 = paddle._C_ops.multiply(strided_slice_7, slice_3)
        del slice_3, strided_slice_7

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_7 = paddle._C_ops.add(multiply_14, multiply_15)
        del multiply_14, multiply_15

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_3 = [subtract_3, add_7]
        del add_7, subtract_3

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_3 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_3 = paddle._C_ops.flatten(stack_3, 3, 4)
        del stack_3

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_11 = paddle._C_ops.matmul(flatten_2, flatten_3, False, True)
        del flatten_2, flatten_3

        # pd_op.scale: (1x6x11x11xf32) <- (1x6x11x11xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_11, full_4, float("0"), True)
        del matmul_11

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_8 = paddle._C_ops.add(scale_5, unsqueeze_0)
        del scale_5

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_1 = paddle._C_ops.softmax(add_8, -1)
        del add_8

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)
        del dropout_8, transpose_6

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_8)
        del transpose_7

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_38, False, False)
        del parameter_38, reshape_7

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_13, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_13

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_9 = paddle._C_ops.add(divide_3, dropout_10)
        del divide_3, dropout_10

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_3 = paddle._C_ops.square(add_9)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_3 = paddle._C_ops.mean(square_3, full_int_array_1, True)
        del square_3

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(mean_3, full_2, float("1e-12"), True)
        del mean_3

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_3 = paddle._C_ops.sqrt(scale_6)
        del scale_6

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_4 = paddle._C_ops.divide(add_9, sqrt_3)
        del add_9, sqrt_3

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_14 = paddle._C_ops.matmul(divide_4, parameter_37, False, False)
        del parameter_37

        # pd_op.relu: (1x11x1536xf32) <- (1x11x1536xf32)
        relu_1 = paddle._C_ops.relu(matmul_14)
        del matmul_14

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_15 = paddle._C_ops.matmul(relu_1, parameter_36, False, False)
        del parameter_36, relu_1

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_15, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_15

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_10 = paddle._C_ops.add(divide_4, dropout_12)
        del divide_4, dropout_12

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_4 = paddle._C_ops.square(add_10)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_4 = paddle._C_ops.mean(square_4, full_int_array_1, True)
        del square_4

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(mean_4, full_2, float("1e-12"), True)
        del mean_4

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_4 = paddle._C_ops.sqrt(scale_7)
        del scale_7

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_5 = paddle._C_ops.divide(add_10, sqrt_4)
        del add_10, sqrt_4

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_16 = paddle._C_ops.matmul(divide_5, parameter_35, False, False)
        del parameter_35

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_17 = paddle._C_ops.matmul(divide_5, parameter_34, False, False)
        del parameter_34

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_18 = paddle._C_ops.matmul(divide_5, parameter_33, False, False)
        del parameter_33

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(matmul_16, full_int_array_2)
        del matmul_16

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(matmul_17, full_int_array_2)
        del matmul_17

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(matmul_18, full_int_array_2)
        del matmul_18

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            parameter_7, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_7

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            parameter_6, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_6

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_8

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_16 = paddle._C_ops.multiply(strided_slice_8, slice_5)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_17 = paddle._C_ops.multiply(strided_slice_9, slice_4)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_4 = paddle._C_ops.subtract(multiply_16, multiply_17)
        del multiply_16, multiply_17

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_18 = paddle._C_ops.multiply(strided_slice_8, slice_4)
        del strided_slice_8

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_19 = paddle._C_ops.multiply(strided_slice_9, slice_5)
        del strided_slice_9

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_11 = paddle._C_ops.add(multiply_18, multiply_19)
        del multiply_18, multiply_19

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_4 = [subtract_4, add_11]
        del add_11, subtract_4

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_4 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_4 = paddle._C_ops.flatten(stack_4, 3, 4)
        del stack_4

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_9

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_20 = paddle._C_ops.multiply(strided_slice_10, slice_5)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_21 = paddle._C_ops.multiply(strided_slice_11, slice_4)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_5 = paddle._C_ops.subtract(multiply_20, multiply_21)
        del multiply_20, multiply_21

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_22 = paddle._C_ops.multiply(strided_slice_10, slice_4)
        del slice_4, strided_slice_10

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_23 = paddle._C_ops.multiply(strided_slice_11, slice_5)
        del slice_5, strided_slice_11

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_12 = paddle._C_ops.add(multiply_22, multiply_23)
        del multiply_22, multiply_23

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_5 = [subtract_5, add_12]
        del add_12, subtract_5

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_5 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_5 = paddle._C_ops.flatten(stack_5, 3, 4)
        del stack_5

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_19 = paddle._C_ops.matmul(flatten_4, flatten_5, False, True)
        del flatten_4, flatten_5

        # pd_op.scale: (1x6x11x11xf32) <- (1x6x11x11xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_19, full_4, float("0"), True)
        del matmul_19

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_13 = paddle._C_ops.add(scale_8, unsqueeze_0)
        del scale_8

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_2 = paddle._C_ops.softmax(add_13, -1)
        del add_13

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)
        del dropout_14, transpose_10

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_8)
        del transpose_11

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_32, False, False)
        del parameter_32, reshape_11

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_21, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_21

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_14 = paddle._C_ops.add(divide_5, dropout_16)
        del divide_5, dropout_16

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_5 = paddle._C_ops.square(add_14)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_5 = paddle._C_ops.mean(square_5, full_int_array_1, True)
        del square_5

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(mean_5, full_2, float("1e-12"), True)
        del mean_5

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_5 = paddle._C_ops.sqrt(scale_9)
        del scale_9

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_6 = paddle._C_ops.divide(add_14, sqrt_5)
        del add_14, sqrt_5

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_22 = paddle._C_ops.matmul(divide_6, parameter_31, False, False)
        del parameter_31

        # pd_op.relu: (1x11x1536xf32) <- (1x11x1536xf32)
        relu_2 = paddle._C_ops.relu(matmul_22)
        del matmul_22

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_23 = paddle._C_ops.matmul(relu_2, parameter_30, False, False)
        del parameter_30, relu_2

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_23, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_23

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_15 = paddle._C_ops.add(divide_6, dropout_18)
        del divide_6, dropout_18

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_6 = paddle._C_ops.square(add_15)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_6 = paddle._C_ops.mean(square_6, full_int_array_1, True)
        del square_6

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(mean_6, full_2, float("1e-12"), True)
        del mean_6

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_6 = paddle._C_ops.sqrt(scale_10)
        del scale_10

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_7 = paddle._C_ops.divide(add_15, sqrt_6)
        del add_15, sqrt_6

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_24 = paddle._C_ops.matmul(divide_7, parameter_29, False, False)
        del parameter_29

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_25 = paddle._C_ops.matmul(divide_7, parameter_28, False, False)
        del parameter_28

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_26 = paddle._C_ops.matmul(divide_7, parameter_27, False, False)
        del parameter_27

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(matmul_24, full_int_array_2)
        del matmul_24

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(matmul_25, full_int_array_2)
        del matmul_25

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(matmul_26, full_int_array_2)
        del matmul_26

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            parameter_5, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_5

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            parameter_4, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_4

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_12 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_13 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_12

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_24 = paddle._C_ops.multiply(strided_slice_12, slice_7)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_25 = paddle._C_ops.multiply(strided_slice_13, slice_6)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_6 = paddle._C_ops.subtract(multiply_24, multiply_25)
        del multiply_24, multiply_25

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_26 = paddle._C_ops.multiply(strided_slice_12, slice_6)
        del strided_slice_12

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_27 = paddle._C_ops.multiply(strided_slice_13, slice_7)
        del strided_slice_13

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_16 = paddle._C_ops.add(multiply_26, multiply_27)
        del multiply_26, multiply_27

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_6 = [subtract_6, add_16]
        del add_16, subtract_6

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_6 = paddle._C_ops.stack(combine_6, -1)
        del combine_6

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_6 = paddle._C_ops.flatten(stack_6, 3, 4)
        del stack_6

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_14 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_15 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_13

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_28 = paddle._C_ops.multiply(strided_slice_14, slice_7)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_29 = paddle._C_ops.multiply(strided_slice_15, slice_6)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_7 = paddle._C_ops.subtract(multiply_28, multiply_29)
        del multiply_28, multiply_29

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_30 = paddle._C_ops.multiply(strided_slice_14, slice_6)
        del slice_6, strided_slice_14

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_31 = paddle._C_ops.multiply(strided_slice_15, slice_7)
        del slice_7, strided_slice_15

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_17 = paddle._C_ops.add(multiply_30, multiply_31)
        del multiply_30, multiply_31

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_7 = [subtract_7, add_17]
        del add_17, subtract_7

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_7 = paddle._C_ops.stack(combine_7, -1)
        del combine_7

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_7 = paddle._C_ops.flatten(stack_7, 3, 4)
        del stack_7

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_27 = paddle._C_ops.matmul(flatten_6, flatten_7, False, True)
        del flatten_6, flatten_7

        # pd_op.scale: (1x6x11x11xf32) <- (1x6x11x11xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_27, full_4, float("0"), True)
        del matmul_27

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_18 = paddle._C_ops.add(scale_11, unsqueeze_0)
        del scale_11

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_3 = paddle._C_ops.softmax(add_18, -1)
        del add_18

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)
        del dropout_20, transpose_14

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_8)
        del transpose_15

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_15, parameter_26, False, False)
        del parameter_26, reshape_15

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_29, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_29

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_19 = paddle._C_ops.add(divide_7, dropout_22)
        del divide_7, dropout_22

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_7 = paddle._C_ops.square(add_19)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_7 = paddle._C_ops.mean(square_7, full_int_array_1, True)
        del square_7

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(mean_7, full_2, float("1e-12"), True)
        del mean_7

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_7 = paddle._C_ops.sqrt(scale_12)
        del scale_12

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_8 = paddle._C_ops.divide(add_19, sqrt_7)
        del add_19, sqrt_7

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_30 = paddle._C_ops.matmul(divide_8, parameter_25, False, False)
        del parameter_25

        # pd_op.relu: (1x11x1536xf32) <- (1x11x1536xf32)
        relu_3 = paddle._C_ops.relu(matmul_30)
        del matmul_30

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_31 = paddle._C_ops.matmul(relu_3, parameter_24, False, False)
        del parameter_24, relu_3

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_31, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_31

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_20 = paddle._C_ops.add(divide_8, dropout_24)
        del divide_8, dropout_24

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_8 = paddle._C_ops.square(add_20)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_8 = paddle._C_ops.mean(square_8, full_int_array_1, True)
        del square_8

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(mean_8, full_2, float("1e-12"), True)
        del mean_8

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_8 = paddle._C_ops.sqrt(scale_13)
        del scale_13

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_9 = paddle._C_ops.divide(add_20, sqrt_8)
        del add_20, sqrt_8

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_32 = paddle._C_ops.matmul(divide_9, parameter_23, False, False)
        del parameter_23

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_33 = paddle._C_ops.matmul(divide_9, parameter_22, False, False)
        del parameter_22

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_34 = paddle._C_ops.matmul(divide_9, parameter_21, False, False)
        del parameter_21

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(matmul_32, full_int_array_2)
        del matmul_32

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(matmul_33, full_int_array_2)
        del matmul_33

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(matmul_34, full_int_array_2)
        del matmul_34

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            parameter_3, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_3

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            parameter_2, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_2

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_16 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_17 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_16

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_32 = paddle._C_ops.multiply(strided_slice_16, slice_9)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_33 = paddle._C_ops.multiply(strided_slice_17, slice_8)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_8 = paddle._C_ops.subtract(multiply_32, multiply_33)
        del multiply_32, multiply_33

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_34 = paddle._C_ops.multiply(strided_slice_16, slice_8)
        del strided_slice_16

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_35 = paddle._C_ops.multiply(strided_slice_17, slice_9)
        del strided_slice_17

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_21 = paddle._C_ops.add(multiply_34, multiply_35)
        del multiply_34, multiply_35

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_8 = [subtract_8, add_21]
        del add_21, subtract_8

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_8 = paddle._C_ops.stack(combine_8, -1)
        del combine_8

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_8 = paddle._C_ops.flatten(stack_8, 3, 4)
        del stack_8

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_18 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_19 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_17

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_36 = paddle._C_ops.multiply(strided_slice_18, slice_9)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_37 = paddle._C_ops.multiply(strided_slice_19, slice_8)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_9 = paddle._C_ops.subtract(multiply_36, multiply_37)
        del multiply_36, multiply_37

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_38 = paddle._C_ops.multiply(strided_slice_18, slice_8)
        del slice_8, strided_slice_18

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_39 = paddle._C_ops.multiply(strided_slice_19, slice_9)
        del slice_9, strided_slice_19

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_22 = paddle._C_ops.add(multiply_38, multiply_39)
        del multiply_38, multiply_39

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_9 = [subtract_9, add_22]
        del add_22, subtract_9

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_9 = paddle._C_ops.stack(combine_9, -1)
        del combine_9

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_9 = paddle._C_ops.flatten(stack_9, 3, 4)
        del stack_9

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_35 = paddle._C_ops.matmul(flatten_8, flatten_9, False, True)
        del flatten_8, flatten_9

        # pd_op.scale: (1x6x11x11xf32) <- (1x6x11x11xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(matmul_35, full_4, float("0"), True)
        del matmul_35

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_23 = paddle._C_ops.add(scale_14, unsqueeze_0)
        del scale_14

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_4 = paddle._C_ops.softmax(add_23, -1)
        del add_23

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_4

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_36 = paddle._C_ops.matmul(dropout_26, transpose_18, False, False)
        del dropout_26, transpose_18

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_8)
        del transpose_19

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_19, parameter_20, False, False)
        del parameter_20, reshape_19

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_37, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_37

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_24 = paddle._C_ops.add(divide_9, dropout_28)
        del divide_9, dropout_28

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_9 = paddle._C_ops.square(add_24)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_9 = paddle._C_ops.mean(square_9, full_int_array_1, True)
        del square_9

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(mean_9, full_2, float("1e-12"), True)
        del mean_9

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_9 = paddle._C_ops.sqrt(scale_15)
        del scale_15

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_10 = paddle._C_ops.divide(add_24, sqrt_9)
        del add_24, sqrt_9

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_38 = paddle._C_ops.matmul(divide_10, parameter_19, False, False)
        del parameter_19

        # pd_op.relu: (1x11x1536xf32) <- (1x11x1536xf32)
        relu_4 = paddle._C_ops.relu(matmul_38)
        del matmul_38

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_39 = paddle._C_ops.matmul(relu_4, parameter_18, False, False)
        del parameter_18, relu_4

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_39, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_39

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_25 = paddle._C_ops.add(divide_10, dropout_30)
        del divide_10, dropout_30

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_10 = paddle._C_ops.square(add_25)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_10 = paddle._C_ops.mean(square_10, full_int_array_1, True)
        del square_10

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(mean_10, full_2, float("1e-12"), True)
        del mean_10

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_10 = paddle._C_ops.sqrt(scale_16)
        del scale_16

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_11 = paddle._C_ops.divide(add_25, sqrt_10)
        del add_25, sqrt_10

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_40 = paddle._C_ops.matmul(divide_11, parameter_17, False, False)
        del parameter_17

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_41 = paddle._C_ops.matmul(divide_11, parameter_16, False, False)
        del parameter_16

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_42 = paddle._C_ops.matmul(divide_11, parameter_15, False, False)
        del parameter_15

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(matmul_40, full_int_array_2)
        del matmul_40

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(matmul_41, full_int_array_2)
        del matmul_41

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(matmul_42, full_int_array_2)
        del full_int_array_2, matmul_42

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            parameter_1, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_1

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            parameter_0, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del full_int_array_4, parameter_0

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_20 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_21 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_20

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_40 = paddle._C_ops.multiply(strided_slice_20, slice_11)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_41 = paddle._C_ops.multiply(strided_slice_21, slice_10)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_10 = paddle._C_ops.subtract(multiply_40, multiply_41)
        del multiply_40, multiply_41

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_42 = paddle._C_ops.multiply(strided_slice_20, slice_10)
        del strided_slice_20

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_43 = paddle._C_ops.multiply(strided_slice_21, slice_11)
        del strided_slice_21

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_26 = paddle._C_ops.add(multiply_42, multiply_43)
        del multiply_42, multiply_43

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_10 = [subtract_10, add_26]
        del add_26, subtract_10

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_10 = paddle._C_ops.stack(combine_10, -1)
        del combine_10

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_10 = paddle._C_ops.flatten(stack_10, 3, 4)
        del stack_10

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_22 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )
        del full_int_array_3

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_23 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del full_int_array_5, full_int_array_6, full_int_array_7, transpose_21

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_44 = paddle._C_ops.multiply(strided_slice_22, slice_11)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_45 = paddle._C_ops.multiply(strided_slice_23, slice_10)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_11 = paddle._C_ops.subtract(multiply_44, multiply_45)
        del multiply_44, multiply_45

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_46 = paddle._C_ops.multiply(strided_slice_22, slice_10)
        del slice_10, strided_slice_22

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_47 = paddle._C_ops.multiply(strided_slice_23, slice_11)
        del slice_11, strided_slice_23

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_27 = paddle._C_ops.add(multiply_46, multiply_47)
        del multiply_46, multiply_47

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_11 = [subtract_11, add_27]
        del add_27, subtract_11

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_11 = paddle._C_ops.stack(combine_11, -1)
        del combine_11

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_11 = paddle._C_ops.flatten(stack_11, 3, 4)
        del stack_11

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_43 = paddle._C_ops.matmul(flatten_10, flatten_11, False, True)
        del flatten_10, flatten_11

        # pd_op.scale: (1x6x11x11xf32) <- (1x6x11x11xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(matmul_43, full_4, float("0"), True)
        del full_4, matmul_43

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_28 = paddle._C_ops.add(scale_17, unsqueeze_0)
        del scale_17, unsqueeze_0

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_5 = paddle._C_ops.softmax(add_28, -1)
        del add_28

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_5

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_44 = paddle._C_ops.matmul(dropout_32, transpose_22, False, False)
        del dropout_32, transpose_22

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_8)
        del full_int_array_8, transpose_23

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_23, parameter_14, False, False)
        del parameter_14, reshape_23

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_45, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_45

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_29 = paddle._C_ops.add(divide_11, dropout_34)
        del divide_11, dropout_34

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_11 = paddle._C_ops.square(add_29)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_11 = paddle._C_ops.mean(square_11, full_int_array_1, True)
        del square_11

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(mean_11, full_2, float("1e-12"), True)
        del mean_11

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_11 = paddle._C_ops.sqrt(scale_18)
        del scale_18

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_12 = paddle._C_ops.divide(add_29, sqrt_11)
        del add_29, sqrt_11

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_46 = paddle._C_ops.matmul(divide_12, parameter_13, False, False)
        del parameter_13

        # pd_op.relu: (1x11x1536xf32) <- (1x11x1536xf32)
        relu_5 = paddle._C_ops.relu(matmul_46)
        del matmul_46

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_47 = paddle._C_ops.matmul(relu_5, parameter_12, False, False)
        del parameter_12, relu_5

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_47, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_3, matmul_47

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_30 = paddle._C_ops.add(divide_12, dropout_36)
        del divide_12, dropout_36

        # pd_op.square: (1x11x384xf32) <- (1x11x384xf32)
        square_12 = paddle._C_ops.square(add_30)

        # pd_op.mean: (1x11x1xf32) <- (1x11x384xf32, 1xi64)
        mean_12 = paddle._C_ops.mean(square_12, full_int_array_1, True)
        del full_int_array_1, square_12

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(mean_12, full_2, float("1e-12"), True)
        del full_2, mean_12

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_12 = paddle._C_ops.sqrt(scale_19)
        del scale_19

        # pd_op.divide: (1x11x384xf32) <- (1x11x384xf32, 1x11x1xf32)
        divide_0 = paddle._C_ops.divide(add_30, sqrt_12)
        del add_30, sqrt_12

        return divide_0
