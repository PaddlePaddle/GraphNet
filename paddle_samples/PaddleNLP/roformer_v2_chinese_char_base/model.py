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

        # pd_op.embedding: (1x11x768xf32) <- (1x11xi64, 12000x768xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_97, -1, False)
        del data_0, parameter_97

        # pd_op.embedding: (1x11x768xf32) <- (1x11xi64, 2x768xf32)
        embedding_1 = paddle._C_ops.embedding(data_1, parameter_96, -1, False)
        del data_1, parameter_96

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)
        del embedding_0, embedding_1

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_0 = paddle._C_ops.square(add_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
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

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_1 = paddle._C_ops.divide(add_0, sqrt_0)
        del add_0, sqrt_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                divide_1, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del divide_1

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_95, False, False)
        del parameter_95

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_94, False, False)
        del parameter_94

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_93, False, False)
        del parameter_93

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [0, 0, 12, 64]

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(matmul_0, full_int_array_2)
        del matmul_0

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(matmul_1, full_int_array_2)
        del matmul_1

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(matmul_2, full_int_array_2)
        del matmul_2

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [11]

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            parameter_23, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_23

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            parameter_22, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_22

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_0

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_0 = paddle._C_ops.multiply(strided_slice_0, slice_1)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_1 = paddle._C_ops.multiply(strided_slice_1, slice_0)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_0 = paddle._C_ops.subtract(multiply_0, multiply_1)
        del multiply_0, multiply_1

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_2 = paddle._C_ops.multiply(strided_slice_0, slice_0)
        del strided_slice_0

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_3 = paddle._C_ops.multiply(strided_slice_1, slice_1)
        del strided_slice_1

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_1 = paddle._C_ops.add(multiply_2, multiply_3)
        del multiply_2, multiply_3

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_0 = [subtract_0, add_1]
        del add_1, subtract_0

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_0 = paddle._C_ops.flatten(stack_0, 3, 4)
        del stack_0

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_1

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_4 = paddle._C_ops.multiply(strided_slice_2, slice_1)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_5 = paddle._C_ops.multiply(strided_slice_3, slice_0)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_1 = paddle._C_ops.subtract(multiply_4, multiply_5)
        del multiply_4, multiply_5

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_6 = paddle._C_ops.multiply(strided_slice_2, slice_0)
        del slice_0, strided_slice_2

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_7 = paddle._C_ops.multiply(strided_slice_3, slice_1)
        del slice_1, strided_slice_3

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_2 = paddle._C_ops.add(multiply_6, multiply_7)
        del multiply_6, multiply_7

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_1 = [subtract_1, add_2]
        del add_2, subtract_1

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_1 = paddle._C_ops.flatten(stack_1, 3, 4)
        del stack_1

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_3 = paddle._C_ops.matmul(flatten_0, flatten_1, False, True)
        del flatten_0, flatten_1

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_3, full_4, float("0"), True)
        del matmul_3

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_3 = paddle._C_ops.add(scale_2, unsqueeze_0)
        del scale_2

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_0 = paddle._C_ops.softmax(add_3, -1)
        del add_3

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [0, 0, 768]

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_8)
        del transpose_3

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_92, False, False)
        del parameter_92, reshape_3

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_5, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_5

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_4 = paddle._C_ops.add(dropout_0, dropout_4)
        del dropout_0, dropout_4

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_1 = paddle._C_ops.square(add_4)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(square_1, full_int_array_1, True)
        del square_1

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(mean_1, full_2, float("1e-12"), True)
        del mean_1

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_1 = paddle._C_ops.sqrt(scale_3)
        del scale_3

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_2 = paddle._C_ops.divide(add_4, sqrt_1)
        del add_4, sqrt_1

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_6 = paddle._C_ops.matmul(divide_2, parameter_91, False, False)
        del parameter_91

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_0 = paddle._C_ops.relu(matmul_6)
        del matmul_6

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_7 = paddle._C_ops.matmul(relu_0, parameter_90, False, False)
        del parameter_90, relu_0

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_7, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_7

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_5 = paddle._C_ops.add(divide_2, dropout_6)
        del divide_2, dropout_6

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_2 = paddle._C_ops.square(add_5)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_2 = paddle._C_ops.mean(square_2, full_int_array_1, True)
        del square_2

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(mean_2, full_2, float("1e-12"), True)
        del mean_2

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_2 = paddle._C_ops.sqrt(scale_4)
        del scale_4

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_3 = paddle._C_ops.divide(add_5, sqrt_2)
        del add_5, sqrt_2

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_8 = paddle._C_ops.matmul(divide_3, parameter_89, False, False)
        del parameter_89

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_9 = paddle._C_ops.matmul(divide_3, parameter_88, False, False)
        del parameter_88

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_10 = paddle._C_ops.matmul(divide_3, parameter_87, False, False)
        del parameter_87

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(matmul_8, full_int_array_2)
        del matmul_8

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(matmul_9, full_int_array_2)
        del matmul_9

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(matmul_10, full_int_array_2)
        del matmul_10

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            parameter_21, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_21

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            parameter_20, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_20

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_4

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_8 = paddle._C_ops.multiply(strided_slice_4, slice_3)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_9 = paddle._C_ops.multiply(strided_slice_5, slice_2)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_2 = paddle._C_ops.subtract(multiply_8, multiply_9)
        del multiply_8, multiply_9

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_10 = paddle._C_ops.multiply(strided_slice_4, slice_2)
        del strided_slice_4

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_11 = paddle._C_ops.multiply(strided_slice_5, slice_3)
        del strided_slice_5

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_6 = paddle._C_ops.add(multiply_10, multiply_11)
        del multiply_10, multiply_11

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_2 = [subtract_2, add_6]
        del add_6, subtract_2

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_2 = paddle._C_ops.flatten(stack_2, 3, 4)
        del stack_2

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_5

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_12 = paddle._C_ops.multiply(strided_slice_6, slice_3)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_13 = paddle._C_ops.multiply(strided_slice_7, slice_2)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_3 = paddle._C_ops.subtract(multiply_12, multiply_13)
        del multiply_12, multiply_13

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_14 = paddle._C_ops.multiply(strided_slice_6, slice_2)
        del slice_2, strided_slice_6

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_15 = paddle._C_ops.multiply(strided_slice_7, slice_3)
        del slice_3, strided_slice_7

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_7 = paddle._C_ops.add(multiply_14, multiply_15)
        del multiply_14, multiply_15

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_3 = [subtract_3, add_7]
        del add_7, subtract_3

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_3 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_3 = paddle._C_ops.flatten(stack_3, 3, 4)
        del stack_3

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_11 = paddle._C_ops.matmul(flatten_2, flatten_3, False, True)
        del flatten_2, flatten_3

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_11, full_4, float("0"), True)
        del matmul_11

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_8 = paddle._C_ops.add(scale_5, unsqueeze_0)
        del scale_5

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_1 = paddle._C_ops.softmax(add_8, -1)
        del add_8

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)
        del dropout_8, transpose_6

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_8)
        del transpose_7

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_86, False, False)
        del parameter_86, reshape_7

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_13, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_13

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_9 = paddle._C_ops.add(divide_3, dropout_10)
        del divide_3, dropout_10

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_3 = paddle._C_ops.square(add_9)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_3 = paddle._C_ops.mean(square_3, full_int_array_1, True)
        del square_3

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(mean_3, full_2, float("1e-12"), True)
        del mean_3

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_3 = paddle._C_ops.sqrt(scale_6)
        del scale_6

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_4 = paddle._C_ops.divide(add_9, sqrt_3)
        del add_9, sqrt_3

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_14 = paddle._C_ops.matmul(divide_4, parameter_85, False, False)
        del parameter_85

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_1 = paddle._C_ops.relu(matmul_14)
        del matmul_14

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_15 = paddle._C_ops.matmul(relu_1, parameter_84, False, False)
        del parameter_84, relu_1

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_15, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_15

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_10 = paddle._C_ops.add(divide_4, dropout_12)
        del divide_4, dropout_12

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_4 = paddle._C_ops.square(add_10)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_4 = paddle._C_ops.mean(square_4, full_int_array_1, True)
        del square_4

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(mean_4, full_2, float("1e-12"), True)
        del mean_4

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_4 = paddle._C_ops.sqrt(scale_7)
        del scale_7

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_5 = paddle._C_ops.divide(add_10, sqrt_4)
        del add_10, sqrt_4

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_16 = paddle._C_ops.matmul(divide_5, parameter_83, False, False)
        del parameter_83

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_17 = paddle._C_ops.matmul(divide_5, parameter_82, False, False)
        del parameter_82

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_18 = paddle._C_ops.matmul(divide_5, parameter_81, False, False)
        del parameter_81

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(matmul_16, full_int_array_2)
        del matmul_16

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(matmul_17, full_int_array_2)
        del matmul_17

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(matmul_18, full_int_array_2)
        del matmul_18

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            parameter_19, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_19

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            parameter_18, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_18

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_8

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_16 = paddle._C_ops.multiply(strided_slice_8, slice_5)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_17 = paddle._C_ops.multiply(strided_slice_9, slice_4)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_4 = paddle._C_ops.subtract(multiply_16, multiply_17)
        del multiply_16, multiply_17

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_18 = paddle._C_ops.multiply(strided_slice_8, slice_4)
        del strided_slice_8

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_19 = paddle._C_ops.multiply(strided_slice_9, slice_5)
        del strided_slice_9

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_11 = paddle._C_ops.add(multiply_18, multiply_19)
        del multiply_18, multiply_19

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_4 = [subtract_4, add_11]
        del add_11, subtract_4

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_4 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_4 = paddle._C_ops.flatten(stack_4, 3, 4)
        del stack_4

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_9

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_20 = paddle._C_ops.multiply(strided_slice_10, slice_5)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_21 = paddle._C_ops.multiply(strided_slice_11, slice_4)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_5 = paddle._C_ops.subtract(multiply_20, multiply_21)
        del multiply_20, multiply_21

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_22 = paddle._C_ops.multiply(strided_slice_10, slice_4)
        del slice_4, strided_slice_10

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_23 = paddle._C_ops.multiply(strided_slice_11, slice_5)
        del slice_5, strided_slice_11

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_12 = paddle._C_ops.add(multiply_22, multiply_23)
        del multiply_22, multiply_23

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_5 = [subtract_5, add_12]
        del add_12, subtract_5

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_5 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_5 = paddle._C_ops.flatten(stack_5, 3, 4)
        del stack_5

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_19 = paddle._C_ops.matmul(flatten_4, flatten_5, False, True)
        del flatten_4, flatten_5

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_19, full_4, float("0"), True)
        del matmul_19

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_13 = paddle._C_ops.add(scale_8, unsqueeze_0)
        del scale_8

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_2 = paddle._C_ops.softmax(add_13, -1)
        del add_13

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)
        del dropout_14, transpose_10

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_8)
        del transpose_11

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_80, False, False)
        del parameter_80, reshape_11

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_21, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_21

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_14 = paddle._C_ops.add(divide_5, dropout_16)
        del divide_5, dropout_16

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_5 = paddle._C_ops.square(add_14)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_5 = paddle._C_ops.mean(square_5, full_int_array_1, True)
        del square_5

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(mean_5, full_2, float("1e-12"), True)
        del mean_5

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_5 = paddle._C_ops.sqrt(scale_9)
        del scale_9

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_6 = paddle._C_ops.divide(add_14, sqrt_5)
        del add_14, sqrt_5

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_22 = paddle._C_ops.matmul(divide_6, parameter_79, False, False)
        del parameter_79

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_2 = paddle._C_ops.relu(matmul_22)
        del matmul_22

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_23 = paddle._C_ops.matmul(relu_2, parameter_78, False, False)
        del parameter_78, relu_2

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_23, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_23

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_15 = paddle._C_ops.add(divide_6, dropout_18)
        del divide_6, dropout_18

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_6 = paddle._C_ops.square(add_15)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_6 = paddle._C_ops.mean(square_6, full_int_array_1, True)
        del square_6

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(mean_6, full_2, float("1e-12"), True)
        del mean_6

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_6 = paddle._C_ops.sqrt(scale_10)
        del scale_10

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_7 = paddle._C_ops.divide(add_15, sqrt_6)
        del add_15, sqrt_6

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_24 = paddle._C_ops.matmul(divide_7, parameter_77, False, False)
        del parameter_77

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_25 = paddle._C_ops.matmul(divide_7, parameter_76, False, False)
        del parameter_76

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_26 = paddle._C_ops.matmul(divide_7, parameter_75, False, False)
        del parameter_75

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(matmul_24, full_int_array_2)
        del matmul_24

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(matmul_25, full_int_array_2)
        del matmul_25

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(matmul_26, full_int_array_2)
        del matmul_26

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            parameter_17, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_17

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            parameter_16, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_16

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_12 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_13 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_12

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_24 = paddle._C_ops.multiply(strided_slice_12, slice_7)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_25 = paddle._C_ops.multiply(strided_slice_13, slice_6)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_6 = paddle._C_ops.subtract(multiply_24, multiply_25)
        del multiply_24, multiply_25

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_26 = paddle._C_ops.multiply(strided_slice_12, slice_6)
        del strided_slice_12

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_27 = paddle._C_ops.multiply(strided_slice_13, slice_7)
        del strided_slice_13

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_16 = paddle._C_ops.add(multiply_26, multiply_27)
        del multiply_26, multiply_27

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_6 = [subtract_6, add_16]
        del add_16, subtract_6

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_6 = paddle._C_ops.stack(combine_6, -1)
        del combine_6

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_6 = paddle._C_ops.flatten(stack_6, 3, 4)
        del stack_6

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_14 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_15 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_13

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_28 = paddle._C_ops.multiply(strided_slice_14, slice_7)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_29 = paddle._C_ops.multiply(strided_slice_15, slice_6)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_7 = paddle._C_ops.subtract(multiply_28, multiply_29)
        del multiply_28, multiply_29

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_30 = paddle._C_ops.multiply(strided_slice_14, slice_6)
        del slice_6, strided_slice_14

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_31 = paddle._C_ops.multiply(strided_slice_15, slice_7)
        del slice_7, strided_slice_15

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_17 = paddle._C_ops.add(multiply_30, multiply_31)
        del multiply_30, multiply_31

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_7 = [subtract_7, add_17]
        del add_17, subtract_7

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_7 = paddle._C_ops.stack(combine_7, -1)
        del combine_7

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_7 = paddle._C_ops.flatten(stack_7, 3, 4)
        del stack_7

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_27 = paddle._C_ops.matmul(flatten_6, flatten_7, False, True)
        del flatten_6, flatten_7

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_27, full_4, float("0"), True)
        del matmul_27

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_18 = paddle._C_ops.add(scale_11, unsqueeze_0)
        del scale_11

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_3 = paddle._C_ops.softmax(add_18, -1)
        del add_18

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)
        del dropout_20, transpose_14

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_8)
        del transpose_15

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_15, parameter_74, False, False)
        del parameter_74, reshape_15

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_29, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_29

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_19 = paddle._C_ops.add(divide_7, dropout_22)
        del divide_7, dropout_22

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_7 = paddle._C_ops.square(add_19)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_7 = paddle._C_ops.mean(square_7, full_int_array_1, True)
        del square_7

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(mean_7, full_2, float("1e-12"), True)
        del mean_7

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_7 = paddle._C_ops.sqrt(scale_12)
        del scale_12

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_8 = paddle._C_ops.divide(add_19, sqrt_7)
        del add_19, sqrt_7

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_30 = paddle._C_ops.matmul(divide_8, parameter_73, False, False)
        del parameter_73

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_3 = paddle._C_ops.relu(matmul_30)
        del matmul_30

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_31 = paddle._C_ops.matmul(relu_3, parameter_72, False, False)
        del parameter_72, relu_3

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_31, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_31

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_20 = paddle._C_ops.add(divide_8, dropout_24)
        del divide_8, dropout_24

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_8 = paddle._C_ops.square(add_20)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_8 = paddle._C_ops.mean(square_8, full_int_array_1, True)
        del square_8

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(mean_8, full_2, float("1e-12"), True)
        del mean_8

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_8 = paddle._C_ops.sqrt(scale_13)
        del scale_13

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_9 = paddle._C_ops.divide(add_20, sqrt_8)
        del add_20, sqrt_8

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_32 = paddle._C_ops.matmul(divide_9, parameter_71, False, False)
        del parameter_71

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_33 = paddle._C_ops.matmul(divide_9, parameter_70, False, False)
        del parameter_70

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_34 = paddle._C_ops.matmul(divide_9, parameter_69, False, False)
        del parameter_69

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(matmul_32, full_int_array_2)
        del matmul_32

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(matmul_33, full_int_array_2)
        del matmul_33

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(matmul_34, full_int_array_2)
        del matmul_34

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            parameter_15, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_15

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            parameter_14, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_14

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_16 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_17 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_16

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_32 = paddle._C_ops.multiply(strided_slice_16, slice_9)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_33 = paddle._C_ops.multiply(strided_slice_17, slice_8)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_8 = paddle._C_ops.subtract(multiply_32, multiply_33)
        del multiply_32, multiply_33

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_34 = paddle._C_ops.multiply(strided_slice_16, slice_8)
        del strided_slice_16

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_35 = paddle._C_ops.multiply(strided_slice_17, slice_9)
        del strided_slice_17

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_21 = paddle._C_ops.add(multiply_34, multiply_35)
        del multiply_34, multiply_35

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_8 = [subtract_8, add_21]
        del add_21, subtract_8

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_8 = paddle._C_ops.stack(combine_8, -1)
        del combine_8

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_8 = paddle._C_ops.flatten(stack_8, 3, 4)
        del stack_8

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_18 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_19 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_17

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_36 = paddle._C_ops.multiply(strided_slice_18, slice_9)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_37 = paddle._C_ops.multiply(strided_slice_19, slice_8)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_9 = paddle._C_ops.subtract(multiply_36, multiply_37)
        del multiply_36, multiply_37

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_38 = paddle._C_ops.multiply(strided_slice_18, slice_8)
        del slice_8, strided_slice_18

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_39 = paddle._C_ops.multiply(strided_slice_19, slice_9)
        del slice_9, strided_slice_19

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_22 = paddle._C_ops.add(multiply_38, multiply_39)
        del multiply_38, multiply_39

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_9 = [subtract_9, add_22]
        del add_22, subtract_9

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_9 = paddle._C_ops.stack(combine_9, -1)
        del combine_9

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_9 = paddle._C_ops.flatten(stack_9, 3, 4)
        del stack_9

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_35 = paddle._C_ops.matmul(flatten_8, flatten_9, False, True)
        del flatten_8, flatten_9

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(matmul_35, full_4, float("0"), True)
        del matmul_35

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_23 = paddle._C_ops.add(scale_14, unsqueeze_0)
        del scale_14

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_4 = paddle._C_ops.softmax(add_23, -1)
        del add_23

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_4

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_36 = paddle._C_ops.matmul(dropout_26, transpose_18, False, False)
        del dropout_26, transpose_18

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_8)
        del transpose_19

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_19, parameter_68, False, False)
        del parameter_68, reshape_19

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_37, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_37

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_24 = paddle._C_ops.add(divide_9, dropout_28)
        del divide_9, dropout_28

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_9 = paddle._C_ops.square(add_24)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_9 = paddle._C_ops.mean(square_9, full_int_array_1, True)
        del square_9

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(mean_9, full_2, float("1e-12"), True)
        del mean_9

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_9 = paddle._C_ops.sqrt(scale_15)
        del scale_15

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_10 = paddle._C_ops.divide(add_24, sqrt_9)
        del add_24, sqrt_9

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_38 = paddle._C_ops.matmul(divide_10, parameter_67, False, False)
        del parameter_67

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_4 = paddle._C_ops.relu(matmul_38)
        del matmul_38

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_39 = paddle._C_ops.matmul(relu_4, parameter_66, False, False)
        del parameter_66, relu_4

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_39, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_39

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_25 = paddle._C_ops.add(divide_10, dropout_30)
        del divide_10, dropout_30

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_10 = paddle._C_ops.square(add_25)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_10 = paddle._C_ops.mean(square_10, full_int_array_1, True)
        del square_10

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(mean_10, full_2, float("1e-12"), True)
        del mean_10

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_10 = paddle._C_ops.sqrt(scale_16)
        del scale_16

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_11 = paddle._C_ops.divide(add_25, sqrt_10)
        del add_25, sqrt_10

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_40 = paddle._C_ops.matmul(divide_11, parameter_65, False, False)
        del parameter_65

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_41 = paddle._C_ops.matmul(divide_11, parameter_64, False, False)
        del parameter_64

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_42 = paddle._C_ops.matmul(divide_11, parameter_63, False, False)
        del parameter_63

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(matmul_40, full_int_array_2)
        del matmul_40

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(matmul_41, full_int_array_2)
        del matmul_41

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(matmul_42, full_int_array_2)
        del matmul_42

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            parameter_13, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_13

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            parameter_12, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_12

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_20 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_21 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_20

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_40 = paddle._C_ops.multiply(strided_slice_20, slice_11)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_41 = paddle._C_ops.multiply(strided_slice_21, slice_10)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_10 = paddle._C_ops.subtract(multiply_40, multiply_41)
        del multiply_40, multiply_41

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_42 = paddle._C_ops.multiply(strided_slice_20, slice_10)
        del strided_slice_20

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_43 = paddle._C_ops.multiply(strided_slice_21, slice_11)
        del strided_slice_21

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_26 = paddle._C_ops.add(multiply_42, multiply_43)
        del multiply_42, multiply_43

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_10 = [subtract_10, add_26]
        del add_26, subtract_10

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_10 = paddle._C_ops.stack(combine_10, -1)
        del combine_10

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_10 = paddle._C_ops.flatten(stack_10, 3, 4)
        del stack_10

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_22 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_23 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_21

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_44 = paddle._C_ops.multiply(strided_slice_22, slice_11)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_45 = paddle._C_ops.multiply(strided_slice_23, slice_10)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_11 = paddle._C_ops.subtract(multiply_44, multiply_45)
        del multiply_44, multiply_45

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_46 = paddle._C_ops.multiply(strided_slice_22, slice_10)
        del slice_10, strided_slice_22

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_47 = paddle._C_ops.multiply(strided_slice_23, slice_11)
        del slice_11, strided_slice_23

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_27 = paddle._C_ops.add(multiply_46, multiply_47)
        del multiply_46, multiply_47

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_11 = [subtract_11, add_27]
        del add_27, subtract_11

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_11 = paddle._C_ops.stack(combine_11, -1)
        del combine_11

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_11 = paddle._C_ops.flatten(stack_11, 3, 4)
        del stack_11

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_43 = paddle._C_ops.matmul(flatten_10, flatten_11, False, True)
        del flatten_10, flatten_11

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(matmul_43, full_4, float("0"), True)
        del matmul_43

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_28 = paddle._C_ops.add(scale_17, unsqueeze_0)
        del scale_17

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_5 = paddle._C_ops.softmax(add_28, -1)
        del add_28

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_5

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_44 = paddle._C_ops.matmul(dropout_32, transpose_22, False, False)
        del dropout_32, transpose_22

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_8)
        del transpose_23

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_23, parameter_62, False, False)
        del parameter_62, reshape_23

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_45, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_45

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_29 = paddle._C_ops.add(divide_11, dropout_34)
        del divide_11, dropout_34

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_11 = paddle._C_ops.square(add_29)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_11 = paddle._C_ops.mean(square_11, full_int_array_1, True)
        del square_11

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(mean_11, full_2, float("1e-12"), True)
        del mean_11

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_11 = paddle._C_ops.sqrt(scale_18)
        del scale_18

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_12 = paddle._C_ops.divide(add_29, sqrt_11)
        del add_29, sqrt_11

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_46 = paddle._C_ops.matmul(divide_12, parameter_61, False, False)
        del parameter_61

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_5 = paddle._C_ops.relu(matmul_46)
        del matmul_46

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_47 = paddle._C_ops.matmul(relu_5, parameter_60, False, False)
        del parameter_60, relu_5

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_47, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_47

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_30 = paddle._C_ops.add(divide_12, dropout_36)
        del divide_12, dropout_36

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_12 = paddle._C_ops.square(add_30)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_12 = paddle._C_ops.mean(square_12, full_int_array_1, True)
        del square_12

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(mean_12, full_2, float("1e-12"), True)
        del mean_12

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_12 = paddle._C_ops.sqrt(scale_19)
        del scale_19

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_13 = paddle._C_ops.divide(add_30, sqrt_12)
        del add_30, sqrt_12

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_48 = paddle._C_ops.matmul(divide_13, parameter_59, False, False)
        del parameter_59

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_49 = paddle._C_ops.matmul(divide_13, parameter_58, False, False)
        del parameter_58

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_50 = paddle._C_ops.matmul(divide_13, parameter_57, False, False)
        del parameter_57

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(matmul_48, full_int_array_2)
        del matmul_48

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(matmul_49, full_int_array_2)
        del matmul_49

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(matmul_50, full_int_array_2)
        del matmul_50

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            parameter_11, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_11

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            parameter_10, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_10

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_24 = paddle._C_ops.strided_slice(
            transpose_24, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_25 = paddle._C_ops.strided_slice(
            transpose_24, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_24

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_48 = paddle._C_ops.multiply(strided_slice_24, slice_13)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_49 = paddle._C_ops.multiply(strided_slice_25, slice_12)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_12 = paddle._C_ops.subtract(multiply_48, multiply_49)
        del multiply_48, multiply_49

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_50 = paddle._C_ops.multiply(strided_slice_24, slice_12)
        del strided_slice_24

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_51 = paddle._C_ops.multiply(strided_slice_25, slice_13)
        del strided_slice_25

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_31 = paddle._C_ops.add(multiply_50, multiply_51)
        del multiply_50, multiply_51

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_12 = [subtract_12, add_31]
        del add_31, subtract_12

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_12 = paddle._C_ops.stack(combine_12, -1)
        del combine_12

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_12 = paddle._C_ops.flatten(stack_12, 3, 4)
        del stack_12

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_26 = paddle._C_ops.strided_slice(
            transpose_25, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_27 = paddle._C_ops.strided_slice(
            transpose_25, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_25

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_52 = paddle._C_ops.multiply(strided_slice_26, slice_13)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_53 = paddle._C_ops.multiply(strided_slice_27, slice_12)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_13 = paddle._C_ops.subtract(multiply_52, multiply_53)
        del multiply_52, multiply_53

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_54 = paddle._C_ops.multiply(strided_slice_26, slice_12)
        del slice_12, strided_slice_26

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_55 = paddle._C_ops.multiply(strided_slice_27, slice_13)
        del slice_13, strided_slice_27

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_32 = paddle._C_ops.add(multiply_54, multiply_55)
        del multiply_54, multiply_55

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_13 = [subtract_13, add_32]
        del add_32, subtract_13

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_13 = paddle._C_ops.stack(combine_13, -1)
        del combine_13

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_13 = paddle._C_ops.flatten(stack_13, 3, 4)
        del stack_13

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_51 = paddle._C_ops.matmul(flatten_12, flatten_13, False, True)
        del flatten_12, flatten_13

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(matmul_51, full_4, float("0"), True)
        del matmul_51

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_33 = paddle._C_ops.add(scale_20, unsqueeze_0)
        del scale_20

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_6 = paddle._C_ops.softmax(add_33, -1)
        del add_33

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_6

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_52 = paddle._C_ops.matmul(dropout_38, transpose_26, False, False)
        del dropout_38, transpose_26

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_27, full_int_array_8)
        del transpose_27

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_27, parameter_56, False, False)
        del parameter_56, reshape_27

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_53, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_53

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_34 = paddle._C_ops.add(divide_13, dropout_40)
        del divide_13, dropout_40

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_13 = paddle._C_ops.square(add_34)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_13 = paddle._C_ops.mean(square_13, full_int_array_1, True)
        del square_13

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(mean_13, full_2, float("1e-12"), True)
        del mean_13

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_13 = paddle._C_ops.sqrt(scale_21)
        del scale_21

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_14 = paddle._C_ops.divide(add_34, sqrt_13)
        del add_34, sqrt_13

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_54 = paddle._C_ops.matmul(divide_14, parameter_55, False, False)
        del parameter_55

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_6 = paddle._C_ops.relu(matmul_54)
        del matmul_54

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_55 = paddle._C_ops.matmul(relu_6, parameter_54, False, False)
        del parameter_54, relu_6

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_55, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_55

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_35 = paddle._C_ops.add(divide_14, dropout_42)
        del divide_14, dropout_42

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_14 = paddle._C_ops.square(add_35)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_14 = paddle._C_ops.mean(square_14, full_int_array_1, True)
        del square_14

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(mean_14, full_2, float("1e-12"), True)
        del mean_14

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_14 = paddle._C_ops.sqrt(scale_22)
        del scale_22

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_15 = paddle._C_ops.divide(add_35, sqrt_14)
        del add_35, sqrt_14

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_56 = paddle._C_ops.matmul(divide_15, parameter_53, False, False)
        del parameter_53

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_57 = paddle._C_ops.matmul(divide_15, parameter_52, False, False)
        del parameter_52

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_58 = paddle._C_ops.matmul(divide_15, parameter_51, False, False)
        del parameter_51

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(matmul_56, full_int_array_2)
        del matmul_56

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(matmul_57, full_int_array_2)
        del matmul_57

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(matmul_58, full_int_array_2)
        del matmul_58

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            parameter_9, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_9

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            parameter_8, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_8

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_28 = paddle._C_ops.strided_slice(
            transpose_28, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_29 = paddle._C_ops.strided_slice(
            transpose_28, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_28

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_56 = paddle._C_ops.multiply(strided_slice_28, slice_15)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_57 = paddle._C_ops.multiply(strided_slice_29, slice_14)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_14 = paddle._C_ops.subtract(multiply_56, multiply_57)
        del multiply_56, multiply_57

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_58 = paddle._C_ops.multiply(strided_slice_28, slice_14)
        del strided_slice_28

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_59 = paddle._C_ops.multiply(strided_slice_29, slice_15)
        del strided_slice_29

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_36 = paddle._C_ops.add(multiply_58, multiply_59)
        del multiply_58, multiply_59

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_14 = [subtract_14, add_36]
        del add_36, subtract_14

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_14 = paddle._C_ops.stack(combine_14, -1)
        del combine_14

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_14 = paddle._C_ops.flatten(stack_14, 3, 4)
        del stack_14

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_30 = paddle._C_ops.strided_slice(
            transpose_29, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_31 = paddle._C_ops.strided_slice(
            transpose_29, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_29

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_60 = paddle._C_ops.multiply(strided_slice_30, slice_15)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_61 = paddle._C_ops.multiply(strided_slice_31, slice_14)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_15 = paddle._C_ops.subtract(multiply_60, multiply_61)
        del multiply_60, multiply_61

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_62 = paddle._C_ops.multiply(strided_slice_30, slice_14)
        del slice_14, strided_slice_30

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_63 = paddle._C_ops.multiply(strided_slice_31, slice_15)
        del slice_15, strided_slice_31

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_37 = paddle._C_ops.add(multiply_62, multiply_63)
        del multiply_62, multiply_63

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_15 = [subtract_15, add_37]
        del add_37, subtract_15

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_15 = paddle._C_ops.stack(combine_15, -1)
        del combine_15

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_15 = paddle._C_ops.flatten(stack_15, 3, 4)
        del stack_15

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_59 = paddle._C_ops.matmul(flatten_14, flatten_15, False, True)
        del flatten_14, flatten_15

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(matmul_59, full_4, float("0"), True)
        del matmul_59

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_38 = paddle._C_ops.add(scale_23, unsqueeze_0)
        del scale_23

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_7 = paddle._C_ops.softmax(add_38, -1)
        del add_38

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_7

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_60 = paddle._C_ops.matmul(dropout_44, transpose_30, False, False)
        del dropout_44, transpose_30

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_60, [0, 2, 1, 3])
        del matmul_60

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_31, full_int_array_8)
        del transpose_31

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_61 = paddle._C_ops.matmul(reshape_31, parameter_50, False, False)
        del parameter_50, reshape_31

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_61, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_61

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_39 = paddle._C_ops.add(divide_15, dropout_46)
        del divide_15, dropout_46

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_15 = paddle._C_ops.square(add_39)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_15 = paddle._C_ops.mean(square_15, full_int_array_1, True)
        del square_15

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(mean_15, full_2, float("1e-12"), True)
        del mean_15

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_15 = paddle._C_ops.sqrt(scale_24)
        del scale_24

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_16 = paddle._C_ops.divide(add_39, sqrt_15)
        del add_39, sqrt_15

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_62 = paddle._C_ops.matmul(divide_16, parameter_49, False, False)
        del parameter_49

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_7 = paddle._C_ops.relu(matmul_62)
        del matmul_62

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_63 = paddle._C_ops.matmul(relu_7, parameter_48, False, False)
        del parameter_48, relu_7

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_63, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_63

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_40 = paddle._C_ops.add(divide_16, dropout_48)
        del divide_16, dropout_48

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_16 = paddle._C_ops.square(add_40)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_16 = paddle._C_ops.mean(square_16, full_int_array_1, True)
        del square_16

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(mean_16, full_2, float("1e-12"), True)
        del mean_16

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_16 = paddle._C_ops.sqrt(scale_25)
        del scale_25

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_17 = paddle._C_ops.divide(add_40, sqrt_16)
        del add_40, sqrt_16

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_64 = paddle._C_ops.matmul(divide_17, parameter_47, False, False)
        del parameter_47

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_65 = paddle._C_ops.matmul(divide_17, parameter_46, False, False)
        del parameter_46

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_66 = paddle._C_ops.matmul(divide_17, parameter_45, False, False)
        del parameter_45

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(matmul_64, full_int_array_2)
        del matmul_64

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(matmul_65, full_int_array_2)
        del matmul_65

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(matmul_66, full_int_array_2)
        del matmul_66

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            parameter_7, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_7

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            parameter_6, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_6

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_32 = paddle._C_ops.strided_slice(
            transpose_32, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_33 = paddle._C_ops.strided_slice(
            transpose_32, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_32

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_64 = paddle._C_ops.multiply(strided_slice_32, slice_17)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_65 = paddle._C_ops.multiply(strided_slice_33, slice_16)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_16 = paddle._C_ops.subtract(multiply_64, multiply_65)
        del multiply_64, multiply_65

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_66 = paddle._C_ops.multiply(strided_slice_32, slice_16)
        del strided_slice_32

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_67 = paddle._C_ops.multiply(strided_slice_33, slice_17)
        del strided_slice_33

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_41 = paddle._C_ops.add(multiply_66, multiply_67)
        del multiply_66, multiply_67

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_16 = [subtract_16, add_41]
        del add_41, subtract_16

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_16 = paddle._C_ops.stack(combine_16, -1)
        del combine_16

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_16 = paddle._C_ops.flatten(stack_16, 3, 4)
        del stack_16

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_34 = paddle._C_ops.strided_slice(
            transpose_33, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_35 = paddle._C_ops.strided_slice(
            transpose_33, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_33

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_68 = paddle._C_ops.multiply(strided_slice_34, slice_17)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_69 = paddle._C_ops.multiply(strided_slice_35, slice_16)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_17 = paddle._C_ops.subtract(multiply_68, multiply_69)
        del multiply_68, multiply_69

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_70 = paddle._C_ops.multiply(strided_slice_34, slice_16)
        del slice_16, strided_slice_34

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_71 = paddle._C_ops.multiply(strided_slice_35, slice_17)
        del slice_17, strided_slice_35

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_42 = paddle._C_ops.add(multiply_70, multiply_71)
        del multiply_70, multiply_71

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_17 = [subtract_17, add_42]
        del add_42, subtract_17

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_17 = paddle._C_ops.stack(combine_17, -1)
        del combine_17

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_17 = paddle._C_ops.flatten(stack_17, 3, 4)
        del stack_17

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_67 = paddle._C_ops.matmul(flatten_16, flatten_17, False, True)
        del flatten_16, flatten_17

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(matmul_67, full_4, float("0"), True)
        del matmul_67

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_43 = paddle._C_ops.add(scale_26, unsqueeze_0)
        del scale_26

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_8 = paddle._C_ops.softmax(add_43, -1)
        del add_43

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_8

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_68 = paddle._C_ops.matmul(dropout_50, transpose_34, False, False)
        del dropout_50, transpose_34

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])
        del matmul_68

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_35, full_int_array_8)
        del transpose_35

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_35, parameter_44, False, False)
        del parameter_44, reshape_35

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_69, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_69

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_44 = paddle._C_ops.add(divide_17, dropout_52)
        del divide_17, dropout_52

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_17 = paddle._C_ops.square(add_44)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_17 = paddle._C_ops.mean(square_17, full_int_array_1, True)
        del square_17

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(mean_17, full_2, float("1e-12"), True)
        del mean_17

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_17 = paddle._C_ops.sqrt(scale_27)
        del scale_27

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_18 = paddle._C_ops.divide(add_44, sqrt_17)
        del add_44, sqrt_17

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_70 = paddle._C_ops.matmul(divide_18, parameter_43, False, False)
        del parameter_43

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_8 = paddle._C_ops.relu(matmul_70)
        del matmul_70

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_71 = paddle._C_ops.matmul(relu_8, parameter_42, False, False)
        del parameter_42, relu_8

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_71, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_71

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_45 = paddle._C_ops.add(divide_18, dropout_54)
        del divide_18, dropout_54

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_18 = paddle._C_ops.square(add_45)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_18 = paddle._C_ops.mean(square_18, full_int_array_1, True)
        del square_18

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(mean_18, full_2, float("1e-12"), True)
        del mean_18

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_18 = paddle._C_ops.sqrt(scale_28)
        del scale_28

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_19 = paddle._C_ops.divide(add_45, sqrt_18)
        del add_45, sqrt_18

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(divide_19, parameter_41, False, False)
        del parameter_41

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_73 = paddle._C_ops.matmul(divide_19, parameter_40, False, False)
        del parameter_40

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_74 = paddle._C_ops.matmul(divide_19, parameter_39, False, False)
        del parameter_39

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(matmul_72, full_int_array_2)
        del matmul_72

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(matmul_73, full_int_array_2)
        del matmul_73

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(matmul_74, full_int_array_2)
        del matmul_74

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            parameter_5, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_5

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            parameter_4, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_4

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_36 = paddle._C_ops.strided_slice(
            transpose_36, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_37 = paddle._C_ops.strided_slice(
            transpose_36, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_36

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_72 = paddle._C_ops.multiply(strided_slice_36, slice_19)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_73 = paddle._C_ops.multiply(strided_slice_37, slice_18)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_18 = paddle._C_ops.subtract(multiply_72, multiply_73)
        del multiply_72, multiply_73

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_74 = paddle._C_ops.multiply(strided_slice_36, slice_18)
        del strided_slice_36

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_75 = paddle._C_ops.multiply(strided_slice_37, slice_19)
        del strided_slice_37

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_46 = paddle._C_ops.add(multiply_74, multiply_75)
        del multiply_74, multiply_75

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_18 = [subtract_18, add_46]
        del add_46, subtract_18

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_18 = paddle._C_ops.stack(combine_18, -1)
        del combine_18

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_18 = paddle._C_ops.flatten(stack_18, 3, 4)
        del stack_18

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_38 = paddle._C_ops.strided_slice(
            transpose_37, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_39 = paddle._C_ops.strided_slice(
            transpose_37, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_37

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_76 = paddle._C_ops.multiply(strided_slice_38, slice_19)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_77 = paddle._C_ops.multiply(strided_slice_39, slice_18)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_19 = paddle._C_ops.subtract(multiply_76, multiply_77)
        del multiply_76, multiply_77

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_78 = paddle._C_ops.multiply(strided_slice_38, slice_18)
        del slice_18, strided_slice_38

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_79 = paddle._C_ops.multiply(strided_slice_39, slice_19)
        del slice_19, strided_slice_39

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_47 = paddle._C_ops.add(multiply_78, multiply_79)
        del multiply_78, multiply_79

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_19 = [subtract_19, add_47]
        del add_47, subtract_19

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_19 = paddle._C_ops.stack(combine_19, -1)
        del combine_19

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_19 = paddle._C_ops.flatten(stack_19, 3, 4)
        del stack_19

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_75 = paddle._C_ops.matmul(flatten_18, flatten_19, False, True)
        del flatten_18, flatten_19

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(matmul_75, full_4, float("0"), True)
        del matmul_75

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_48 = paddle._C_ops.add(scale_29, unsqueeze_0)
        del scale_29

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_9 = paddle._C_ops.softmax(add_48, -1)
        del add_48

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_9

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_76 = paddle._C_ops.matmul(dropout_56, transpose_38, False, False)
        del dropout_56, transpose_38

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_76, [0, 2, 1, 3])
        del matmul_76

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_39, full_int_array_8)
        del transpose_39

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_77 = paddle._C_ops.matmul(reshape_39, parameter_38, False, False)
        del parameter_38, reshape_39

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_77, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_77

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_49 = paddle._C_ops.add(divide_19, dropout_58)
        del divide_19, dropout_58

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_19 = paddle._C_ops.square(add_49)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_19 = paddle._C_ops.mean(square_19, full_int_array_1, True)
        del square_19

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(mean_19, full_2, float("1e-12"), True)
        del mean_19

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_19 = paddle._C_ops.sqrt(scale_30)
        del scale_30

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_20 = paddle._C_ops.divide(add_49, sqrt_19)
        del add_49, sqrt_19

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_78 = paddle._C_ops.matmul(divide_20, parameter_37, False, False)
        del parameter_37

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_9 = paddle._C_ops.relu(matmul_78)
        del matmul_78

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_79 = paddle._C_ops.matmul(relu_9, parameter_36, False, False)
        del parameter_36, relu_9

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_79, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_79

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_50 = paddle._C_ops.add(divide_20, dropout_60)
        del divide_20, dropout_60

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_20 = paddle._C_ops.square(add_50)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_20 = paddle._C_ops.mean(square_20, full_int_array_1, True)
        del square_20

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(mean_20, full_2, float("1e-12"), True)
        del mean_20

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_20 = paddle._C_ops.sqrt(scale_31)
        del scale_31

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_21 = paddle._C_ops.divide(add_50, sqrt_20)
        del add_50, sqrt_20

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_80 = paddle._C_ops.matmul(divide_21, parameter_35, False, False)
        del parameter_35

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_81 = paddle._C_ops.matmul(divide_21, parameter_34, False, False)
        del parameter_34

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_82 = paddle._C_ops.matmul(divide_21, parameter_33, False, False)
        del parameter_33

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(matmul_80, full_int_array_2)
        del matmul_80

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(matmul_81, full_int_array_2)
        del matmul_81

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(matmul_82, full_int_array_2)
        del matmul_82

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            parameter_3, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_3

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            parameter_2, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_2

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_40 = paddle._C_ops.strided_slice(
            transpose_40, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_41 = paddle._C_ops.strided_slice(
            transpose_40, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_40

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_80 = paddle._C_ops.multiply(strided_slice_40, slice_21)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_81 = paddle._C_ops.multiply(strided_slice_41, slice_20)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_20 = paddle._C_ops.subtract(multiply_80, multiply_81)
        del multiply_80, multiply_81

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_82 = paddle._C_ops.multiply(strided_slice_40, slice_20)
        del strided_slice_40

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_83 = paddle._C_ops.multiply(strided_slice_41, slice_21)
        del strided_slice_41

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_51 = paddle._C_ops.add(multiply_82, multiply_83)
        del multiply_82, multiply_83

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_20 = [subtract_20, add_51]
        del add_51, subtract_20

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_20 = paddle._C_ops.stack(combine_20, -1)
        del combine_20

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_20 = paddle._C_ops.flatten(stack_20, 3, 4)
        del stack_20

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_42 = paddle._C_ops.strided_slice(
            transpose_41, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_43 = paddle._C_ops.strided_slice(
            transpose_41, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_41

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_84 = paddle._C_ops.multiply(strided_slice_42, slice_21)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_85 = paddle._C_ops.multiply(strided_slice_43, slice_20)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_21 = paddle._C_ops.subtract(multiply_84, multiply_85)
        del multiply_84, multiply_85

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_86 = paddle._C_ops.multiply(strided_slice_42, slice_20)
        del slice_20, strided_slice_42

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_87 = paddle._C_ops.multiply(strided_slice_43, slice_21)
        del slice_21, strided_slice_43

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_52 = paddle._C_ops.add(multiply_86, multiply_87)
        del multiply_86, multiply_87

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_21 = [subtract_21, add_52]
        del add_52, subtract_21

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_21 = paddle._C_ops.stack(combine_21, -1)
        del combine_21

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_21 = paddle._C_ops.flatten(stack_21, 3, 4)
        del stack_21

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_83 = paddle._C_ops.matmul(flatten_20, flatten_21, False, True)
        del flatten_20, flatten_21

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(matmul_83, full_4, float("0"), True)
        del matmul_83

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_53 = paddle._C_ops.add(scale_32, unsqueeze_0)
        del scale_32

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_10 = paddle._C_ops.softmax(add_53, -1)
        del add_53

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_10

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_84 = paddle._C_ops.matmul(dropout_62, transpose_42, False, False)
        del dropout_62, transpose_42

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_84, [0, 2, 1, 3])
        del matmul_84

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_43, full_int_array_8)
        del transpose_43

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_85 = paddle._C_ops.matmul(reshape_43, parameter_32, False, False)
        del parameter_32, reshape_43

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_85, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_85

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_54 = paddle._C_ops.add(divide_21, dropout_64)
        del divide_21, dropout_64

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_21 = paddle._C_ops.square(add_54)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_21 = paddle._C_ops.mean(square_21, full_int_array_1, True)
        del square_21

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(mean_21, full_2, float("1e-12"), True)
        del mean_21

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_21 = paddle._C_ops.sqrt(scale_33)
        del scale_33

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_22 = paddle._C_ops.divide(add_54, sqrt_21)
        del add_54, sqrt_21

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_86 = paddle._C_ops.matmul(divide_22, parameter_31, False, False)
        del parameter_31

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_10 = paddle._C_ops.relu(matmul_86)
        del matmul_86

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_87 = paddle._C_ops.matmul(relu_10, parameter_30, False, False)
        del parameter_30, relu_10

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_87, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_87

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_55 = paddle._C_ops.add(divide_22, dropout_66)
        del divide_22, dropout_66

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_22 = paddle._C_ops.square(add_55)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_22 = paddle._C_ops.mean(square_22, full_int_array_1, True)
        del square_22

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(mean_22, full_2, float("1e-12"), True)
        del mean_22

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_22 = paddle._C_ops.sqrt(scale_34)
        del scale_34

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_23 = paddle._C_ops.divide(add_55, sqrt_22)
        del add_55, sqrt_22

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_88 = paddle._C_ops.matmul(divide_23, parameter_29, False, False)
        del parameter_29

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_89 = paddle._C_ops.matmul(divide_23, parameter_28, False, False)
        del parameter_28

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_90 = paddle._C_ops.matmul(divide_23, parameter_27, False, False)
        del parameter_27

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(matmul_88, full_int_array_2)
        del matmul_88

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(matmul_89, full_int_array_2)
        del matmul_89

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(matmul_90, full_int_array_2)
        del full_int_array_2, matmul_90

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            parameter_1, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del parameter_1

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            parameter_0, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del full_int_array_4, parameter_0

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_44 = paddle._C_ops.strided_slice(
            transpose_44, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_45 = paddle._C_ops.strided_slice(
            transpose_44, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del transpose_44

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_88 = paddle._C_ops.multiply(strided_slice_44, slice_23)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_89 = paddle._C_ops.multiply(strided_slice_45, slice_22)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_22 = paddle._C_ops.subtract(multiply_88, multiply_89)
        del multiply_88, multiply_89

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_90 = paddle._C_ops.multiply(strided_slice_44, slice_22)
        del strided_slice_44

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_91 = paddle._C_ops.multiply(strided_slice_45, slice_23)
        del strided_slice_45

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_56 = paddle._C_ops.add(multiply_90, multiply_91)
        del multiply_90, multiply_91

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_22 = [subtract_22, add_56]
        del add_56, subtract_22

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_22 = paddle._C_ops.stack(combine_22, -1)
        del combine_22

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_22 = paddle._C_ops.flatten(stack_22, 3, 4)
        del stack_22

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_46 = paddle._C_ops.strided_slice(
            transpose_45, [3], full_int_array_3, full_int_array_5, full_int_array_6
        )
        del full_int_array_3

        # pd_op.strided_slice: (1x12x11x32xf32) <- (1x12x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_47 = paddle._C_ops.strided_slice(
            transpose_45, [3], full_int_array_7, full_int_array_5, full_int_array_6
        )
        del full_int_array_5, full_int_array_6, full_int_array_7, transpose_45

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_92 = paddle._C_ops.multiply(strided_slice_46, slice_23)

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_93 = paddle._C_ops.multiply(strided_slice_47, slice_22)

        # pd_op.subtract: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        subtract_23 = paddle._C_ops.subtract(multiply_92, multiply_93)
        del multiply_92, multiply_93

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_94 = paddle._C_ops.multiply(strided_slice_46, slice_22)
        del slice_22, strided_slice_46

        # pd_op.multiply: (1x12x11x32xf32) <- (1x12x11x32xf32, 11x32xf32)
        multiply_95 = paddle._C_ops.multiply(strided_slice_47, slice_23)
        del slice_23, strided_slice_47

        # pd_op.add: (1x12x11x32xf32) <- (1x12x11x32xf32, 1x12x11x32xf32)
        add_57 = paddle._C_ops.add(multiply_94, multiply_95)
        del multiply_94, multiply_95

        # builtin.combine: ([1x12x11x32xf32, 1x12x11x32xf32]) <- (1x12x11x32xf32, 1x12x11x32xf32)
        combine_23 = [subtract_23, add_57]
        del add_57, subtract_23

        # pd_op.stack: (1x12x11x32x2xf32) <- ([1x12x11x32xf32, 1x12x11x32xf32])
        stack_23 = paddle._C_ops.stack(combine_23, -1)
        del combine_23

        # pd_op.flatten: (1x12x11x64xf32) <- (1x12x11x32x2xf32)
        flatten_23 = paddle._C_ops.flatten(stack_23, 3, 4)
        del stack_23

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        matmul_91 = paddle._C_ops.matmul(flatten_22, flatten_23, False, True)
        del flatten_22, flatten_23

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(matmul_91, full_4, float("0"), True)
        del full_4, matmul_91

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_58 = paddle._C_ops.add(scale_35, unsqueeze_0)
        del scale_35, unsqueeze_0

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_11 = paddle._C_ops.softmax(add_58, -1)
        del add_58

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_11

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_92 = paddle._C_ops.matmul(dropout_68, transpose_46, False, False)
        del dropout_68, transpose_46

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_92, [0, 2, 1, 3])
        del matmul_92

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_47, full_int_array_8)
        del full_int_array_8, transpose_47

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_93 = paddle._C_ops.matmul(reshape_47, parameter_26, False, False)
        del parameter_26, reshape_47

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_93, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_93

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_59 = paddle._C_ops.add(divide_23, dropout_70)
        del divide_23, dropout_70

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_23 = paddle._C_ops.square(add_59)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_23 = paddle._C_ops.mean(square_23, full_int_array_1, True)
        del square_23

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(mean_23, full_2, float("1e-12"), True)
        del mean_23

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_23 = paddle._C_ops.sqrt(scale_36)
        del scale_36

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_24 = paddle._C_ops.divide(add_59, sqrt_23)
        del add_59, sqrt_23

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_94 = paddle._C_ops.matmul(divide_24, parameter_25, False, False)
        del parameter_25

        # pd_op.relu: (1x11x3072xf32) <- (1x11x3072xf32)
        relu_11 = paddle._C_ops.relu(matmul_94)
        del matmul_94

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_95 = paddle._C_ops.matmul(relu_11, parameter_24, False, False)
        del parameter_24, relu_11

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_95, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_3, matmul_95

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_60 = paddle._C_ops.add(divide_24, dropout_72)
        del divide_24, dropout_72

        # pd_op.square: (1x11x768xf32) <- (1x11x768xf32)
        square_24 = paddle._C_ops.square(add_60)

        # pd_op.mean: (1x11x1xf32) <- (1x11x768xf32, 1xi64)
        mean_24 = paddle._C_ops.mean(square_24, full_int_array_1, True)
        del full_int_array_1, square_24

        # pd_op.scale: (1x11x1xf32) <- (1x11x1xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(mean_24, full_2, float("1e-12"), True)
        del full_2, mean_24

        # pd_op.sqrt: (1x11x1xf32) <- (1x11x1xf32)
        sqrt_24 = paddle._C_ops.sqrt(scale_37)
        del scale_37

        # pd_op.divide: (1x11x768xf32) <- (1x11x768xf32, 1x11x1xf32)
        divide_0 = paddle._C_ops.divide(add_60, sqrt_24)
        del add_60, sqrt_24

        return divide_0
