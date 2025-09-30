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
        parameter_114,
        parameter_115,
        parameter_116,
        parameter_117,
        parameter_118,
        parameter_119,
        parameter_120,
        parameter_121,
        parameter_122,
        parameter_123,
        parameter_124,
        parameter_125,
        parameter_126,
        parameter_127,
        parameter_128,
        parameter_129,
        parameter_130,
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.embedding: (1x20x512xf32) <- (1x20xi64, 32128x512xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_130, -1, False)
        del data_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2]

        # pd_op.unsqueeze: (1x1x1x20xi64) <- (1x20xi64, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_0)
        del data_1

        # pd_op.cast: (1x1x1x20xf32) <- (1x1x1x20xi64)
        cast_0 = paddle._C_ops.cast(unsqueeze_0, paddle.float32)
        del unsqueeze_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x20xf32) <- (1x1x1x20xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_0, float("1"), True)
        del cast_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x20xf32) <- (1x1x1x20xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_1, float("0"), True)
        del scale_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                embedding_0, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del embedding_0

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_0 = paddle._C_ops.pow(dropout_0, float("2"))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(pow_0, full_int_array_1, True)
        del pow_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(mean_0, full_3, float("1e-06"), True)
        del mean_0

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_0 = paddle._C_ops.rsqrt(scale_2)
        del scale_2

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_0 = paddle._C_ops.multiply(dropout_0, rsqrt_0)
        del rsqrt_0

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_1 = paddle._C_ops.multiply(parameter_124, multiply_0)
        del multiply_0, parameter_124

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_1 = paddle._C_ops.matmul(multiply_1, parameter_129, False, False)
        del parameter_129

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, -1, 8, 64]

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(matmul_1, full_int_array_2)
        del matmul_1

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_2 = paddle._C_ops.matmul(multiply_1, parameter_128, False, False)
        del parameter_128

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(matmul_2, full_int_array_2)
        del matmul_2

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_3 = paddle._C_ops.matmul(multiply_1, parameter_127, False, False)
        del multiply_1, parameter_127

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(matmul_3, full_int_array_2)
        del matmul_3

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (1x8x20x20xf32) <- (1x8x20x64xf32, 1x8x20x64xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_0, transpose_1, False, True)
        del transpose_0, transpose_1

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("20"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (20xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_4, full_5, full_6, dtype="int64")
        del full_5

        # pd_op.unsqueeze: (20x1xi64) <- (20xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(arange_0, full_int_array_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.unsqueeze: (1x20xi64) <- (20xi64, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(arange_0, full_int_array_3)
        del arange_0

        # pd_op.subtract: (20x20xi64) <- (1x20xi64, 20x1xi64)
        subtract_0 = paddle._C_ops.subtract(unsqueeze_2, unsqueeze_1)
        del unsqueeze_1, unsqueeze_2

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (20x20xb) <- (20x20xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(subtract_0, full_7)
        del full_7

        # pd_op.cast: (20x20xi64) <- (20x20xb)
        cast_1 = paddle._C_ops.cast(greater_than_0, paddle.int64)
        del greater_than_0

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (20x20xi64) <- (20x20xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(cast_1, full_8, float("0"), True)
        del cast_1

        # pd_op.scale: (20x20xi64) <- (20x20xi64, 1xf32)
        scale_4 = paddle._C_ops.scale(scale_3, full_3, float("0"), True)
        del scale_3

        # pd_op.abs: (20x20xi64) <- (20x20xi64)
        abs_0 = paddle._C_ops.abs(subtract_0)
        del subtract_0

        # pd_op.full: (xi64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.less_than: (20x20xb) <- (20x20xi64, xi64)
        less_than_0 = paddle._C_ops.less_than(abs_0, full_9)
        del full_9

        # pd_op.cast: (20x20xf32) <- (20x20xi64)
        cast_2 = paddle._C_ops.cast(abs_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (20x20xf32) <- (20x20xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(cast_2, full_10, float("0"), True)
        del cast_2, full_10

        # pd_op.log: (20x20xf32) <- (20x20xf32)
        log_0 = paddle._C_ops.log(scale_5)
        del scale_5

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("0.360674"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (20x20xf32) <- (20x20xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(log_0, full_11, float("0"), True)
        del full_11, log_0

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (20x20xf32) <- (20x20xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_12, float("0"), True)
        del full_12, scale_6

        # pd_op.cast: (20x20xi64) <- (20x20xf32)
        cast_3 = paddle._C_ops.cast(scale_7, paddle.int64)
        del scale_7

        # pd_op.scale: (20x20xi64) <- (20x20xi64, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_3, full_3, float("8"), True)
        del cast_3

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("15"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (20x20xi64) <- (20x20xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            scale_8, full_13, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_13

        # pd_op.minimum: (20x20xi64) <- (20x20xi64, 20x20xi64)
        minimum_0 = paddle._C_ops.minimum(scale_8, full_like_0)
        del full_like_0, scale_8

        # pd_op.where: (20x20xi64) <- (20x20xb, 20x20xi64, 20x20xi64)
        where_0 = paddle._C_ops.where(less_than_0, abs_0, minimum_0)
        del abs_0, less_than_0, minimum_0

        # pd_op.add: (20x20xi64) <- (20x20xi64, 20x20xi64)
        add_0 = paddle._C_ops.add(scale_4, where_0)
        del scale_4, where_0

        # pd_op.embedding: (20x20x8xf32) <- (20x20xi64, 32x8xf32)
        embedding_1 = paddle._C_ops.embedding(add_0, parameter_125, -1, False)
        del add_0, parameter_125

        # pd_op.transpose: (8x20x20xf32) <- (20x20x8xf32)
        transpose_3 = paddle._C_ops.transpose(embedding_1, [2, 0, 1])
        del embedding_1

        # pd_op.unsqueeze: (1x8x20x20xf32) <- (8x20x20xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(transpose_3, full_int_array_3)
        del transpose_3

        # pd_op.add: (1x8x20x20xf32) <- (1x8x20x20xf32, 1x1x1x20xf32)
        add_1 = paddle._C_ops.add(unsqueeze_3, scale_1)
        del unsqueeze_3

        # pd_op.add: (1x8x20x20xf32) <- (1x8x20x20xf32, 1x8x20x20xf32)
        add_2 = paddle._C_ops.add(matmul_4, add_1)
        del matmul_4

        # pd_op.softmax: (1x8x20x20xf32) <- (1x8x20x20xf32)
        softmax_0 = paddle._C_ops.softmax(add_2, -1)
        del add_2

        # pd_op.dropout: (1x8x20x20xf32, 1x8x20x20xui8) <- (1x8x20x20xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x8x20x64xf32) <- (1x8x20x20xf32, 1x8x20x64xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x20x8x64xf32) <- (1x8x20x64xf32)
        transpose_4 = paddle._C_ops.transpose(matmul_5, [0, 2, 1, 3])
        del matmul_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [1, -1, 512]

        # pd_op.reshape: (1x20x512xf32) <- (1x20x8x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_4, full_int_array_4)
        del transpose_4

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_3, parameter_126, False, False)
        del parameter_126, reshape_3

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_6, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_6

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_3 = paddle._C_ops.add(dropout_0, dropout_4)
        del dropout_0, dropout_4

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_1 = paddle._C_ops.pow(add_3, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(pow_1, full_int_array_1, True)
        del pow_1

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(mean_1, full_3, float("1e-06"), True)
        del mean_1

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_1 = paddle._C_ops.rsqrt(scale_9)
        del scale_9

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_2 = paddle._C_ops.multiply(add_3, rsqrt_1)
        del rsqrt_1

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_3 = paddle._C_ops.multiply(parameter_121, multiply_2)
        del multiply_2, parameter_121

        # pd_op.matmul: (1x20x2048xf32) <- (1x20x512xf32, 512x2048xf32)
        matmul_7 = paddle._C_ops.matmul(multiply_3, parameter_123, False, False)
        del multiply_3, parameter_123

        # pd_op.relu: (1x20x2048xf32) <- (1x20x2048xf32)
        relu_0 = paddle._C_ops.relu(matmul_7)
        del matmul_7

        # pd_op.dropout: (1x20x2048xf32, 1x20x2048xui8) <- (1x20x2048xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_0, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_0

        # pd_op.matmul: (1x20x512xf32) <- (1x20x2048xf32, 2048x512xf32)
        matmul_8 = paddle._C_ops.matmul(dropout_6, parameter_122, False, False)
        del dropout_6, parameter_122

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_8, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_8

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_4 = paddle._C_ops.add(dropout_8, add_3)
        del add_3, dropout_8

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_2 = paddle._C_ops.pow(add_4, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_2 = paddle._C_ops.mean(pow_2, full_int_array_1, True)
        del pow_2

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(mean_2, full_3, float("1e-06"), True)
        del mean_2

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_2 = paddle._C_ops.rsqrt(scale_10)
        del scale_10

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_4 = paddle._C_ops.multiply(add_4, rsqrt_2)
        del rsqrt_2

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_5 = paddle._C_ops.multiply(parameter_116, multiply_4)
        del multiply_4, parameter_116

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_9 = paddle._C_ops.matmul(multiply_5, parameter_120, False, False)
        del parameter_120

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(matmul_9, full_int_array_2)
        del matmul_9

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_10 = paddle._C_ops.matmul(multiply_5, parameter_119, False, False)
        del parameter_119

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(matmul_10, full_int_array_2)
        del matmul_10

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_11 = paddle._C_ops.matmul(multiply_5, parameter_118, False, False)
        del multiply_5, parameter_118

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(matmul_11, full_int_array_2)
        del matmul_11

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.matmul: (1x8x20x20xf32) <- (1x8x20x64xf32, 1x8x20x64xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_5, transpose_6, False, True)
        del transpose_5, transpose_6

        # pd_op.add: (1x8x20x20xf32) <- (1x8x20x20xf32, 1x8x20x20xf32)
        add_5 = paddle._C_ops.add(matmul_12, add_1)
        del matmul_12

        # pd_op.softmax: (1x8x20x20xf32) <- (1x8x20x20xf32)
        softmax_1 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.dropout: (1x8x20x20xf32, 1x8x20x20xui8) <- (1x8x20x20xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x8x20x64xf32) <- (1x8x20x20xf32, 1x8x20x64xf32)
        matmul_13 = paddle._C_ops.matmul(dropout_10, transpose_7, False, False)
        del dropout_10, transpose_7

        # pd_op.transpose: (1x20x8x64xf32) <- (1x8x20x64xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_13, [0, 2, 1, 3])
        del matmul_13

        # pd_op.reshape: (1x20x512xf32) <- (1x20x8x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_8, full_int_array_4)
        del transpose_8

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_14 = paddle._C_ops.matmul(reshape_7, parameter_117, False, False)
        del parameter_117, reshape_7

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_14, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_14

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_6 = paddle._C_ops.add(add_4, dropout_12)
        del add_4, dropout_12

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_3 = paddle._C_ops.pow(add_6, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_3 = paddle._C_ops.mean(pow_3, full_int_array_1, True)
        del pow_3

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(mean_3, full_3, float("1e-06"), True)
        del mean_3

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_3 = paddle._C_ops.rsqrt(scale_11)
        del scale_11

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_6 = paddle._C_ops.multiply(add_6, rsqrt_3)
        del rsqrt_3

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_7 = paddle._C_ops.multiply(parameter_113, multiply_6)
        del multiply_6, parameter_113

        # pd_op.matmul: (1x20x2048xf32) <- (1x20x512xf32, 512x2048xf32)
        matmul_15 = paddle._C_ops.matmul(multiply_7, parameter_115, False, False)
        del multiply_7, parameter_115

        # pd_op.relu: (1x20x2048xf32) <- (1x20x2048xf32)
        relu_1 = paddle._C_ops.relu(matmul_15)
        del matmul_15

        # pd_op.dropout: (1x20x2048xf32, 1x20x2048xui8) <- (1x20x2048xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_1, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_1

        # pd_op.matmul: (1x20x512xf32) <- (1x20x2048xf32, 2048x512xf32)
        matmul_16 = paddle._C_ops.matmul(dropout_14, parameter_114, False, False)
        del dropout_14, parameter_114

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_16, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_16

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_7 = paddle._C_ops.add(dropout_16, add_6)
        del add_6, dropout_16

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_4 = paddle._C_ops.pow(add_7, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_4 = paddle._C_ops.mean(pow_4, full_int_array_1, True)
        del pow_4

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(mean_4, full_3, float("1e-06"), True)
        del mean_4

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_4 = paddle._C_ops.rsqrt(scale_12)
        del scale_12

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_8 = paddle._C_ops.multiply(add_7, rsqrt_4)
        del rsqrt_4

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_9 = paddle._C_ops.multiply(parameter_108, multiply_8)
        del multiply_8, parameter_108

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_17 = paddle._C_ops.matmul(multiply_9, parameter_112, False, False)
        del parameter_112

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(matmul_17, full_int_array_2)
        del matmul_17

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_18 = paddle._C_ops.matmul(multiply_9, parameter_111, False, False)
        del parameter_111

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(matmul_18, full_int_array_2)
        del matmul_18

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_19 = paddle._C_ops.matmul(multiply_9, parameter_110, False, False)
        del multiply_9, parameter_110

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(matmul_19, full_int_array_2)
        del matmul_19

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.matmul: (1x8x20x20xf32) <- (1x8x20x64xf32, 1x8x20x64xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_9, transpose_10, False, True)
        del transpose_10, transpose_9

        # pd_op.add: (1x8x20x20xf32) <- (1x8x20x20xf32, 1x8x20x20xf32)
        add_8 = paddle._C_ops.add(matmul_20, add_1)
        del matmul_20

        # pd_op.softmax: (1x8x20x20xf32) <- (1x8x20x20xf32)
        softmax_2 = paddle._C_ops.softmax(add_8, -1)
        del add_8

        # pd_op.dropout: (1x8x20x20xf32, 1x8x20x20xui8) <- (1x8x20x20xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x8x20x64xf32) <- (1x8x20x20xf32, 1x8x20x64xf32)
        matmul_21 = paddle._C_ops.matmul(dropout_18, transpose_11, False, False)
        del dropout_18, transpose_11

        # pd_op.transpose: (1x20x8x64xf32) <- (1x8x20x64xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (1x20x512xf32) <- (1x20x8x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_12, full_int_array_4)
        del transpose_12

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_11, parameter_109, False, False)
        del parameter_109, reshape_11

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_22, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_22

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_9 = paddle._C_ops.add(add_7, dropout_20)
        del add_7, dropout_20

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_5 = paddle._C_ops.pow(add_9, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_5 = paddle._C_ops.mean(pow_5, full_int_array_1, True)
        del pow_5

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(mean_5, full_3, float("1e-06"), True)
        del mean_5

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_5 = paddle._C_ops.rsqrt(scale_13)
        del scale_13

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_10 = paddle._C_ops.multiply(add_9, rsqrt_5)
        del rsqrt_5

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_11 = paddle._C_ops.multiply(parameter_105, multiply_10)
        del multiply_10, parameter_105

        # pd_op.matmul: (1x20x2048xf32) <- (1x20x512xf32, 512x2048xf32)
        matmul_23 = paddle._C_ops.matmul(multiply_11, parameter_107, False, False)
        del multiply_11, parameter_107

        # pd_op.relu: (1x20x2048xf32) <- (1x20x2048xf32)
        relu_2 = paddle._C_ops.relu(matmul_23)
        del matmul_23

        # pd_op.dropout: (1x20x2048xf32, 1x20x2048xui8) <- (1x20x2048xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_2, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_2

        # pd_op.matmul: (1x20x512xf32) <- (1x20x2048xf32, 2048x512xf32)
        matmul_24 = paddle._C_ops.matmul(dropout_22, parameter_106, False, False)
        del dropout_22, parameter_106

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_24, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_24

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_10 = paddle._C_ops.add(dropout_24, add_9)
        del add_9, dropout_24

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_6 = paddle._C_ops.pow(add_10, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_6 = paddle._C_ops.mean(pow_6, full_int_array_1, True)
        del pow_6

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(mean_6, full_3, float("1e-06"), True)
        del mean_6

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_6 = paddle._C_ops.rsqrt(scale_14)
        del scale_14

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_12 = paddle._C_ops.multiply(add_10, rsqrt_6)
        del rsqrt_6

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_13 = paddle._C_ops.multiply(parameter_100, multiply_12)
        del multiply_12, parameter_100

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_25 = paddle._C_ops.matmul(multiply_13, parameter_104, False, False)
        del parameter_104

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(matmul_25, full_int_array_2)
        del matmul_25

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_26 = paddle._C_ops.matmul(multiply_13, parameter_103, False, False)
        del parameter_103

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(matmul_26, full_int_array_2)
        del matmul_26

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_27 = paddle._C_ops.matmul(multiply_13, parameter_102, False, False)
        del multiply_13, parameter_102

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(matmul_27, full_int_array_2)
        del matmul_27

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.matmul: (1x8x20x20xf32) <- (1x8x20x64xf32, 1x8x20x64xf32)
        matmul_28 = paddle._C_ops.matmul(transpose_13, transpose_14, False, True)
        del transpose_13, transpose_14

        # pd_op.add: (1x8x20x20xf32) <- (1x8x20x20xf32, 1x8x20x20xf32)
        add_11 = paddle._C_ops.add(matmul_28, add_1)
        del matmul_28

        # pd_op.softmax: (1x8x20x20xf32) <- (1x8x20x20xf32)
        softmax_3 = paddle._C_ops.softmax(add_11, -1)
        del add_11

        # pd_op.dropout: (1x8x20x20xf32, 1x8x20x20xui8) <- (1x8x20x20xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x8x20x64xf32) <- (1x8x20x20xf32, 1x8x20x64xf32)
        matmul_29 = paddle._C_ops.matmul(dropout_26, transpose_15, False, False)
        del dropout_26, transpose_15

        # pd_op.transpose: (1x20x8x64xf32) <- (1x8x20x64xf32)
        transpose_16 = paddle._C_ops.transpose(matmul_29, [0, 2, 1, 3])
        del matmul_29

        # pd_op.reshape: (1x20x512xf32) <- (1x20x8x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_16, full_int_array_4)
        del transpose_16

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_30 = paddle._C_ops.matmul(reshape_15, parameter_101, False, False)
        del parameter_101, reshape_15

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_30, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_30

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_12 = paddle._C_ops.add(add_10, dropout_28)
        del add_10, dropout_28

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_7 = paddle._C_ops.pow(add_12, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_7 = paddle._C_ops.mean(pow_7, full_int_array_1, True)
        del pow_7

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(mean_7, full_3, float("1e-06"), True)
        del mean_7

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_7 = paddle._C_ops.rsqrt(scale_15)
        del scale_15

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_14 = paddle._C_ops.multiply(add_12, rsqrt_7)
        del rsqrt_7

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_15 = paddle._C_ops.multiply(parameter_97, multiply_14)
        del multiply_14, parameter_97

        # pd_op.matmul: (1x20x2048xf32) <- (1x20x512xf32, 512x2048xf32)
        matmul_31 = paddle._C_ops.matmul(multiply_15, parameter_99, False, False)
        del multiply_15, parameter_99

        # pd_op.relu: (1x20x2048xf32) <- (1x20x2048xf32)
        relu_3 = paddle._C_ops.relu(matmul_31)
        del matmul_31

        # pd_op.dropout: (1x20x2048xf32, 1x20x2048xui8) <- (1x20x2048xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_3, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_3

        # pd_op.matmul: (1x20x512xf32) <- (1x20x2048xf32, 2048x512xf32)
        matmul_32 = paddle._C_ops.matmul(dropout_30, parameter_98, False, False)
        del dropout_30, parameter_98

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_32, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_32

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_13 = paddle._C_ops.add(dropout_32, add_12)
        del add_12, dropout_32

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_8 = paddle._C_ops.pow(add_13, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_8 = paddle._C_ops.mean(pow_8, full_int_array_1, True)
        del pow_8

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(mean_8, full_3, float("1e-06"), True)
        del mean_8

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_8 = paddle._C_ops.rsqrt(scale_16)
        del scale_16

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_16 = paddle._C_ops.multiply(add_13, rsqrt_8)
        del rsqrt_8

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_17 = paddle._C_ops.multiply(parameter_92, multiply_16)
        del multiply_16, parameter_92

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_33 = paddle._C_ops.matmul(multiply_17, parameter_96, False, False)
        del parameter_96

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(matmul_33, full_int_array_2)
        del matmul_33

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_34 = paddle._C_ops.matmul(multiply_17, parameter_95, False, False)
        del parameter_95

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(matmul_34, full_int_array_2)
        del matmul_34

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_35 = paddle._C_ops.matmul(multiply_17, parameter_94, False, False)
        del multiply_17, parameter_94

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(matmul_35, full_int_array_2)
        del matmul_35

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.matmul: (1x8x20x20xf32) <- (1x8x20x64xf32, 1x8x20x64xf32)
        matmul_36 = paddle._C_ops.matmul(transpose_17, transpose_18, False, True)
        del transpose_17, transpose_18

        # pd_op.add: (1x8x20x20xf32) <- (1x8x20x20xf32, 1x8x20x20xf32)
        add_14 = paddle._C_ops.add(matmul_36, add_1)
        del matmul_36

        # pd_op.softmax: (1x8x20x20xf32) <- (1x8x20x20xf32)
        softmax_4 = paddle._C_ops.softmax(add_14, -1)
        del add_14

        # pd_op.dropout: (1x8x20x20xf32, 1x8x20x20xui8) <- (1x8x20x20xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_4

        # pd_op.matmul: (1x8x20x64xf32) <- (1x8x20x20xf32, 1x8x20x64xf32)
        matmul_37 = paddle._C_ops.matmul(dropout_34, transpose_19, False, False)
        del dropout_34, transpose_19

        # pd_op.transpose: (1x20x8x64xf32) <- (1x8x20x64xf32)
        transpose_20 = paddle._C_ops.transpose(matmul_37, [0, 2, 1, 3])
        del matmul_37

        # pd_op.reshape: (1x20x512xf32) <- (1x20x8x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_20, full_int_array_4)
        del transpose_20

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_19, parameter_93, False, False)
        del parameter_93, reshape_19

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_38, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_38

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_15 = paddle._C_ops.add(add_13, dropout_36)
        del add_13, dropout_36

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_9 = paddle._C_ops.pow(add_15, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_9 = paddle._C_ops.mean(pow_9, full_int_array_1, True)
        del pow_9

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(mean_9, full_3, float("1e-06"), True)
        del mean_9

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_9 = paddle._C_ops.rsqrt(scale_17)
        del scale_17

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_18 = paddle._C_ops.multiply(add_15, rsqrt_9)
        del rsqrt_9

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_19 = paddle._C_ops.multiply(parameter_89, multiply_18)
        del multiply_18, parameter_89

        # pd_op.matmul: (1x20x2048xf32) <- (1x20x512xf32, 512x2048xf32)
        matmul_39 = paddle._C_ops.matmul(multiply_19, parameter_91, False, False)
        del multiply_19, parameter_91

        # pd_op.relu: (1x20x2048xf32) <- (1x20x2048xf32)
        relu_4 = paddle._C_ops.relu(matmul_39)
        del matmul_39

        # pd_op.dropout: (1x20x2048xf32, 1x20x2048xui8) <- (1x20x2048xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_4, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_4

        # pd_op.matmul: (1x20x512xf32) <- (1x20x2048xf32, 2048x512xf32)
        matmul_40 = paddle._C_ops.matmul(dropout_38, parameter_90, False, False)
        del dropout_38, parameter_90

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_40, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_40

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_16 = paddle._C_ops.add(dropout_40, add_15)
        del add_15, dropout_40

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_10 = paddle._C_ops.pow(add_16, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_10 = paddle._C_ops.mean(pow_10, full_int_array_1, True)
        del pow_10

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(mean_10, full_3, float("1e-06"), True)
        del mean_10

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_10 = paddle._C_ops.rsqrt(scale_18)
        del scale_18

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_20 = paddle._C_ops.multiply(add_16, rsqrt_10)
        del rsqrt_10

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_21 = paddle._C_ops.multiply(parameter_84, multiply_20)
        del multiply_20, parameter_84

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_41 = paddle._C_ops.matmul(multiply_21, parameter_88, False, False)
        del parameter_88

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(matmul_41, full_int_array_2)
        del matmul_41

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_42 = paddle._C_ops.matmul(multiply_21, parameter_87, False, False)
        del parameter_87

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(matmul_42, full_int_array_2)
        del matmul_42

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_43 = paddle._C_ops.matmul(multiply_21, parameter_86, False, False)
        del multiply_21, parameter_86

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(matmul_43, full_int_array_2)
        del matmul_43

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_23 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.matmul: (1x8x20x20xf32) <- (1x8x20x64xf32, 1x8x20x64xf32)
        matmul_44 = paddle._C_ops.matmul(transpose_21, transpose_22, False, True)
        del transpose_21, transpose_22

        # pd_op.add: (1x8x20x20xf32) <- (1x8x20x20xf32, 1x8x20x20xf32)
        add_17 = paddle._C_ops.add(matmul_44, add_1)
        del add_1, matmul_44

        # pd_op.softmax: (1x8x20x20xf32) <- (1x8x20x20xf32)
        softmax_5 = paddle._C_ops.softmax(add_17, -1)
        del add_17

        # pd_op.dropout: (1x8x20x20xf32, 1x8x20x20xui8) <- (1x8x20x20xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_5

        # pd_op.matmul: (1x8x20x64xf32) <- (1x8x20x20xf32, 1x8x20x64xf32)
        matmul_45 = paddle._C_ops.matmul(dropout_42, transpose_23, False, False)
        del dropout_42, transpose_23

        # pd_op.transpose: (1x20x8x64xf32) <- (1x8x20x64xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])
        del matmul_45

        # pd_op.reshape: (1x20x512xf32) <- (1x20x8x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_24, full_int_array_4)
        del transpose_24

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_23, parameter_85, False, False)
        del parameter_85, reshape_23

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_46, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_46

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_18 = paddle._C_ops.add(add_16, dropout_44)
        del add_16, dropout_44

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_11 = paddle._C_ops.pow(add_18, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_11 = paddle._C_ops.mean(pow_11, full_int_array_1, True)
        del pow_11

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(mean_11, full_3, float("1e-06"), True)
        del mean_11

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_11 = paddle._C_ops.rsqrt(scale_19)
        del scale_19

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_22 = paddle._C_ops.multiply(add_18, rsqrt_11)
        del rsqrt_11

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_23 = paddle._C_ops.multiply(parameter_81, multiply_22)
        del multiply_22, parameter_81

        # pd_op.matmul: (1x20x2048xf32) <- (1x20x512xf32, 512x2048xf32)
        matmul_47 = paddle._C_ops.matmul(multiply_23, parameter_83, False, False)
        del multiply_23, parameter_83

        # pd_op.relu: (1x20x2048xf32) <- (1x20x2048xf32)
        relu_5 = paddle._C_ops.relu(matmul_47)
        del matmul_47

        # pd_op.dropout: (1x20x2048xf32, 1x20x2048xui8) <- (1x20x2048xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_5, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_5

        # pd_op.matmul: (1x20x512xf32) <- (1x20x2048xf32, 2048x512xf32)
        matmul_48 = paddle._C_ops.matmul(dropout_46, parameter_82, False, False)
        del dropout_46, parameter_82

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_48, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_48

        # pd_op.add: (1x20x512xf32) <- (1x20x512xf32, 1x20x512xf32)
        add_19 = paddle._C_ops.add(dropout_48, add_18)
        del add_18, dropout_48

        # pd_op.pow: (1x20x512xf32) <- (1x20x512xf32)
        pow_12 = paddle._C_ops.pow(add_19, float("2"))

        # pd_op.mean: (1x20x1xf32) <- (1x20x512xf32, 1xi64)
        mean_12 = paddle._C_ops.mean(pow_12, full_int_array_1, True)
        del pow_12

        # pd_op.scale: (1x20x1xf32) <- (1x20x1xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(mean_12, full_3, float("1e-06"), True)
        del mean_12

        # pd_op.rsqrt: (1x20x1xf32) <- (1x20x1xf32)
        rsqrt_12 = paddle._C_ops.rsqrt(scale_20)
        del scale_20

        # pd_op.multiply: (1x20x512xf32) <- (1x20x512xf32, 1x20x1xf32)
        multiply_24 = paddle._C_ops.multiply(add_19, rsqrt_12)
        del add_19, rsqrt_12

        # pd_op.multiply: (1x20x512xf32) <- (512xf32, 1x20x512xf32)
        multiply_25 = paddle._C_ops.multiply(parameter_80, multiply_24)
        del multiply_24, parameter_80

        # pd_op.dropout: (1x20x512xf32, 1x20x512xui8) <- (1x20x512xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                multiply_25, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del multiply_25

        # pd_op.embedding: (1x1x512xf32) <- (1x1xi64, 32128x512xf32)
        embedding_2 = paddle._C_ops.embedding(data_2, parameter_130, -1, False)
        del data_2

        # pd_op.full: (1x1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1, 1],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.arange: (1xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_4, full_6, full_6, dtype="int64")
        del full_4, full_6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [0, 1]

        # pd_op.unsqueeze: (1x1x1xi64) <- (1xi64, 2xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(arange_1, full_int_array_5)
        del full_int_array_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [1, 1, 1]

        # pd_op.tile: (1x1x1xi64) <- (1x1x1xi64, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_4, full_int_array_6)
        del full_int_array_6, unsqueeze_4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [0, 2]

        # pd_op.unsqueeze: (1x1x1xi64) <- (1xi64, 2xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(arange_1, full_int_array_7)
        del full_int_array_7

        # pd_op.less_equal: (1x1x1xb) <- (1x1x1xi64, 1x1x1xi64)
        less_equal_0 = paddle._C_ops.less_equal(tile_0, unsqueeze_5)
        del tile_0, unsqueeze_5

        # pd_op.cast: (1x1x1xf32) <- (1x1x1xb)
        cast_4 = paddle._C_ops.cast(less_equal_0, paddle.float32)
        del less_equal_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [1]

        # pd_op.unsqueeze: (1x1x1x1xf32) <- (1x1x1xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(cast_4, full_int_array_8)
        del cast_4, full_int_array_8

        # pd_op.unsqueeze: (1x1x1x1xf32) <- (1x1xf32, 2xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(full_14, full_int_array_0)
        del full_14, full_int_array_0

        # pd_op.multiply: (1x1x1x1xf32) <- (1x1x1x1xf32, 1x1x1x1xf32)
        multiply_26 = paddle._C_ops.multiply(unsqueeze_6, unsqueeze_7)
        del unsqueeze_6, unsqueeze_7

        # pd_op.scale: (1x1x1x1xf32) <- (1x1x1x1xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(multiply_26, full_0, float("1"), True)
        del multiply_26

        # pd_op.scale: (1x1x1x1xf32) <- (1x1x1x1xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(scale_21, full_1, float("0"), True)
        del full_1, scale_21

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                embedding_2, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del embedding_2

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_13 = paddle._C_ops.pow(dropout_52, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_13 = paddle._C_ops.mean(pow_13, full_int_array_1, True)
        del pow_13

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(mean_13, full_3, float("1e-06"), True)
        del mean_13

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_13 = paddle._C_ops.rsqrt(scale_23)
        del scale_23

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_27 = paddle._C_ops.multiply(dropout_52, rsqrt_13)
        del rsqrt_13

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_28 = paddle._C_ops.multiply(parameter_74, multiply_27)
        del multiply_27, parameter_74

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_49 = paddle._C_ops.matmul(multiply_28, parameter_79, False, False)
        del parameter_79

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(matmul_49, full_int_array_2)
        del matmul_49

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_50 = paddle._C_ops.matmul(multiply_28, parameter_78, False, False)
        del parameter_78

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(matmul_50, full_int_array_2)
        del matmul_50

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_51 = paddle._C_ops.matmul(multiply_28, parameter_77, False, False)
        del multiply_28, parameter_77

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(matmul_51, full_int_array_2)
        del matmul_51

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.matmul: (1x8x1x1xf32) <- (1x8x1x64xf32, 1x8x1x64xf32)
        matmul_52 = paddle._C_ops.matmul(transpose_25, transpose_26, False, True)
        del transpose_25

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(arange_1, full_int_array_1)

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(arange_1, full_int_array_3)
        del arange_1

        # pd_op.subtract: (1x1xi64) <- (1x1xi64, 1x1xi64)
        subtract_1 = paddle._C_ops.subtract(unsqueeze_9, unsqueeze_8)
        del unsqueeze_8, unsqueeze_9

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x1xi64) <- (1x1xi64, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            subtract_1,
            full_15,
            paddle.int64,
            paddle.framework._current_expected_place(),
        )
        del full_15

        # pd_op.minimum: (1x1xi64) <- (1x1xi64, 1x1xi64)
        minimum_1 = paddle._C_ops.minimum(subtract_1, full_like_1)
        del full_like_1, subtract_1

        # pd_op.scale: (1x1xi64) <- (1x1xi64, 1xf32)
        scale_24 = paddle._C_ops.scale(minimum_1, full_0, float("0"), True)
        del full_0, minimum_1

        # pd_op.full: (xi64) <- ()
        full_16 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.less_than: (1x1xb) <- (1x1xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_24, full_16)
        del full_16

        # pd_op.cast: (1x1xf32) <- (1x1xi64)
        cast_5 = paddle._C_ops.cast(scale_24, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full(
            [1], float("0.0625"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(cast_5, full_17, float("0"), True)
        del cast_5, full_17

        # pd_op.log: (1x1xf32) <- (1x1xf32)
        log_1 = paddle._C_ops.log(scale_25)
        del scale_25

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("0.480898"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(log_1, full_18, float("0"), True)
        del full_18, log_1

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(scale_26, full_8, float("0"), True)
        del full_8, scale_26

        # pd_op.cast: (1x1xi64) <- (1x1xf32)
        cast_6 = paddle._C_ops.cast(scale_27, paddle.int64)
        del scale_27

        # pd_op.scale: (1x1xi64) <- (1x1xi64, 1xf32)
        scale_28 = paddle._C_ops.scale(cast_6, full_3, float("16"), True)
        del cast_6

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full(
            [1], float("31"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x1xi64) <- (1x1xi64, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            scale_28, full_19, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_19

        # pd_op.minimum: (1x1xi64) <- (1x1xi64, 1x1xi64)
        minimum_2 = paddle._C_ops.minimum(scale_28, full_like_2)
        del full_like_2, scale_28

        # pd_op.where: (1x1xi64) <- (1x1xb, 1x1xi64, 1x1xi64)
        where_1 = paddle._C_ops.where(less_than_1, scale_24, minimum_2)
        del less_than_1, minimum_2, scale_24

        # pd_op.scale: (1x1xi64) <- (1x1xi64, 1xf32)
        scale_29 = paddle._C_ops.scale(where_1, full_3, float("0"), True)
        del where_1

        # pd_op.embedding: (1x1x8xf32) <- (1x1xi64, 32x8xf32)
        embedding_3 = paddle._C_ops.embedding(scale_29, parameter_75, -1, False)
        del parameter_75, scale_29

        # pd_op.transpose: (8x1x1xf32) <- (1x1x8xf32)
        transpose_28 = paddle._C_ops.transpose(embedding_3, [2, 0, 1])
        del embedding_3

        # pd_op.unsqueeze: (1x8x1x1xf32) <- (8x1x1xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(transpose_28, full_int_array_3)
        del full_int_array_3, transpose_28

        # pd_op.add: (1x8x1x1xf32) <- (1x8x1x1xf32, 1x1x1x1xf32)
        add_20 = paddle._C_ops.add(unsqueeze_10, scale_22)
        del scale_22, unsqueeze_10

        # pd_op.add: (1x8x1x1xf32) <- (1x8x1x1xf32, 1x8x1x1xf32)
        add_21 = paddle._C_ops.add(matmul_52, add_20)
        del matmul_52

        # pd_op.softmax: (1x8x1x1xf32) <- (1x8x1x1xf32)
        softmax_6 = paddle._C_ops.softmax(add_21, -1)
        del add_21

        # pd_op.dropout: (1x8x1x1xf32, 1x8x1x1xui8) <- (1x8x1x1xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_6

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x1xf32, 1x8x1x64xf32)
        matmul_53 = paddle._C_ops.matmul(dropout_54, transpose_27, False, False)
        del dropout_54

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_29 = paddle._C_ops.transpose(matmul_53, [0, 2, 1, 3])
        del matmul_53

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_29, full_int_array_4)
        del transpose_29

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_54 = paddle._C_ops.matmul(reshape_27, parameter_76, False, False)
        del parameter_76, reshape_27

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_54, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_54

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_22 = paddle._C_ops.add(dropout_52, dropout_56)
        del dropout_52, dropout_56

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_14 = paddle._C_ops.pow(add_22, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_14 = paddle._C_ops.mean(pow_14, full_int_array_1, True)
        del pow_14

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(mean_14, full_3, float("1e-06"), True)
        del mean_14

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_14 = paddle._C_ops.rsqrt(scale_30)
        del scale_30

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_29 = paddle._C_ops.multiply(add_22, rsqrt_14)
        del rsqrt_14

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_30 = paddle._C_ops.multiply(parameter_69, multiply_29)
        del multiply_29, parameter_69

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_55 = paddle._C_ops.matmul(multiply_30, parameter_73, False, False)
        del multiply_30, parameter_73

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(matmul_55, full_int_array_2)
        del matmul_55

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_56 = paddle._C_ops.matmul(dropout_50, parameter_72, False, False)
        del parameter_72

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(matmul_56, full_int_array_2)
        del matmul_56

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_57 = paddle._C_ops.matmul(dropout_50, parameter_71, False, False)
        del parameter_71

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(matmul_57, full_int_array_2)
        del matmul_57

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.matmul: (1x8x1x20xf32) <- (1x8x1x64xf32, 1x8x20x64xf32)
        matmul_58 = paddle._C_ops.matmul(transpose_30, transpose_31, False, True)
        del transpose_30

        # pd_op.full: (1x8x1x20xf32) <- ()
        full_20 = paddle._C_ops.full(
            [1, 8, 1, 20],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (1x8x1x20xf32) <- (1x8x1x20xf32, 1x1x1x20xf32)
        add_23 = paddle._C_ops.add(full_20, scale_1)
        del full_20, scale_1

        # pd_op.add: (1x8x1x20xf32) <- (1x8x1x20xf32, 1x8x1x20xf32)
        add_24 = paddle._C_ops.add(matmul_58, add_23)
        del matmul_58

        # pd_op.softmax: (1x8x1x20xf32) <- (1x8x1x20xf32)
        softmax_7 = paddle._C_ops.softmax(add_24, -1)
        del add_24

        # pd_op.dropout: (1x8x1x20xf32, 1x8x1x20xui8) <- (1x8x1x20xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_7

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x20xf32, 1x8x20x64xf32)
        matmul_59 = paddle._C_ops.matmul(dropout_58, transpose_32, False, False)
        del dropout_58

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_33 = paddle._C_ops.transpose(matmul_59, [0, 2, 1, 3])
        del matmul_59

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_33, full_int_array_4)
        del transpose_33

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_60 = paddle._C_ops.matmul(reshape_31, parameter_70, False, False)
        del parameter_70, reshape_31

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_60, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_60

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_25 = paddle._C_ops.add(add_22, dropout_60)
        del add_22, dropout_60

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_15 = paddle._C_ops.pow(add_25, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_15 = paddle._C_ops.mean(pow_15, full_int_array_1, True)
        del pow_15

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(mean_15, full_3, float("1e-06"), True)
        del mean_15

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_15 = paddle._C_ops.rsqrt(scale_31)
        del scale_31

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_31 = paddle._C_ops.multiply(add_25, rsqrt_15)
        del rsqrt_15

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_32 = paddle._C_ops.multiply(parameter_66, multiply_31)
        del multiply_31, parameter_66

        # pd_op.matmul: (1x1x2048xf32) <- (1x1x512xf32, 512x2048xf32)
        matmul_61 = paddle._C_ops.matmul(multiply_32, parameter_68, False, False)
        del multiply_32, parameter_68

        # pd_op.relu: (1x1x2048xf32) <- (1x1x2048xf32)
        relu_6 = paddle._C_ops.relu(matmul_61)
        del matmul_61

        # pd_op.dropout: (1x1x2048xf32, 1x1x2048xui8) <- (1x1x2048xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_6, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_6

        # pd_op.matmul: (1x1x512xf32) <- (1x1x2048xf32, 2048x512xf32)
        matmul_62 = paddle._C_ops.matmul(dropout_62, parameter_67, False, False)
        del dropout_62, parameter_67

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_62, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_62

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_26 = paddle._C_ops.add(dropout_64, add_25)
        del add_25, dropout_64

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_16 = paddle._C_ops.pow(add_26, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_16 = paddle._C_ops.mean(pow_16, full_int_array_1, True)
        del pow_16

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(mean_16, full_3, float("1e-06"), True)
        del mean_16

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_16 = paddle._C_ops.rsqrt(scale_32)
        del scale_32

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_33 = paddle._C_ops.multiply(add_26, rsqrt_16)
        del rsqrt_16

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_34 = paddle._C_ops.multiply(parameter_61, multiply_33)
        del multiply_33, parameter_61

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_63 = paddle._C_ops.matmul(multiply_34, parameter_65, False, False)
        del parameter_65

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(matmul_63, full_int_array_2)
        del matmul_63

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_64 = paddle._C_ops.matmul(multiply_34, parameter_64, False, False)
        del parameter_64

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(matmul_64, full_int_array_2)
        del matmul_64

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_35 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_65 = paddle._C_ops.matmul(multiply_34, parameter_63, False, False)
        del multiply_34, parameter_63

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(matmul_65, full_int_array_2)
        del matmul_65

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.matmul: (1x8x1x1xf32) <- (1x8x1x64xf32, 1x8x1x64xf32)
        matmul_66 = paddle._C_ops.matmul(transpose_34, transpose_35, False, True)
        del transpose_34

        # pd_op.add: (1x8x1x1xf32) <- (1x8x1x1xf32, 1x8x1x1xf32)
        add_27 = paddle._C_ops.add(matmul_66, add_20)
        del matmul_66

        # pd_op.softmax: (1x8x1x1xf32) <- (1x8x1x1xf32)
        softmax_8 = paddle._C_ops.softmax(add_27, -1)
        del add_27

        # pd_op.dropout: (1x8x1x1xf32, 1x8x1x1xui8) <- (1x8x1x1xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_8

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x1xf32, 1x8x1x64xf32)
        matmul_67 = paddle._C_ops.matmul(dropout_66, transpose_36, False, False)
        del dropout_66

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_37 = paddle._C_ops.transpose(matmul_67, [0, 2, 1, 3])
        del matmul_67

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_37, full_int_array_4)
        del transpose_37

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_68 = paddle._C_ops.matmul(reshape_35, parameter_62, False, False)
        del parameter_62, reshape_35

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_68, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_68

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_28 = paddle._C_ops.add(add_26, dropout_68)
        del add_26, dropout_68

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_17 = paddle._C_ops.pow(add_28, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_17 = paddle._C_ops.mean(pow_17, full_int_array_1, True)
        del pow_17

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(mean_17, full_3, float("1e-06"), True)
        del mean_17

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_17 = paddle._C_ops.rsqrt(scale_33)
        del scale_33

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_35 = paddle._C_ops.multiply(add_28, rsqrt_17)
        del rsqrt_17

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_36 = paddle._C_ops.multiply(parameter_56, multiply_35)
        del multiply_35, parameter_56

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_69 = paddle._C_ops.matmul(multiply_36, parameter_60, False, False)
        del multiply_36, parameter_60

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(matmul_69, full_int_array_2)
        del matmul_69

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_70 = paddle._C_ops.matmul(dropout_50, parameter_59, False, False)
        del parameter_59

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(matmul_70, full_int_array_2)
        del matmul_70

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_39 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_71 = paddle._C_ops.matmul(dropout_50, parameter_58, False, False)
        del parameter_58

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(matmul_71, full_int_array_2)
        del matmul_71

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.matmul: (1x8x1x20xf32) <- (1x8x1x64xf32, 1x8x20x64xf32)
        matmul_72 = paddle._C_ops.matmul(transpose_38, transpose_39, False, True)
        del transpose_38

        # pd_op.add: (1x8x1x20xf32) <- (1x8x1x20xf32, 1x8x1x20xf32)
        add_29 = paddle._C_ops.add(matmul_72, add_23)
        del matmul_72

        # pd_op.softmax: (1x8x1x20xf32) <- (1x8x1x20xf32)
        softmax_9 = paddle._C_ops.softmax(add_29, -1)
        del add_29

        # pd_op.dropout: (1x8x1x20xf32, 1x8x1x20xui8) <- (1x8x1x20xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_9

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x20xf32, 1x8x20x64xf32)
        matmul_73 = paddle._C_ops.matmul(dropout_70, transpose_40, False, False)
        del dropout_70

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_41 = paddle._C_ops.transpose(matmul_73, [0, 2, 1, 3])
        del matmul_73

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_41, full_int_array_4)
        del transpose_41

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_74 = paddle._C_ops.matmul(reshape_39, parameter_57, False, False)
        del parameter_57, reshape_39

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_74, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_74

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_30 = paddle._C_ops.add(add_28, dropout_72)
        del add_28, dropout_72

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_18 = paddle._C_ops.pow(add_30, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_18 = paddle._C_ops.mean(pow_18, full_int_array_1, True)
        del pow_18

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(mean_18, full_3, float("1e-06"), True)
        del mean_18

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_18 = paddle._C_ops.rsqrt(scale_34)
        del scale_34

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_37 = paddle._C_ops.multiply(add_30, rsqrt_18)
        del rsqrt_18

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_38 = paddle._C_ops.multiply(parameter_53, multiply_37)
        del multiply_37, parameter_53

        # pd_op.matmul: (1x1x2048xf32) <- (1x1x512xf32, 512x2048xf32)
        matmul_75 = paddle._C_ops.matmul(multiply_38, parameter_55, False, False)
        del multiply_38, parameter_55

        # pd_op.relu: (1x1x2048xf32) <- (1x1x2048xf32)
        relu_7 = paddle._C_ops.relu(matmul_75)
        del matmul_75

        # pd_op.dropout: (1x1x2048xf32, 1x1x2048xui8) <- (1x1x2048xf32, None, 1xf32)
        dropout_74, dropout_75 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_7, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_7

        # pd_op.matmul: (1x1x512xf32) <- (1x1x2048xf32, 2048x512xf32)
        matmul_76 = paddle._C_ops.matmul(dropout_74, parameter_54, False, False)
        del dropout_74, parameter_54

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_76, dropout_77 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_76, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_76

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_31 = paddle._C_ops.add(dropout_76, add_30)
        del add_30, dropout_76

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_19 = paddle._C_ops.pow(add_31, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_19 = paddle._C_ops.mean(pow_19, full_int_array_1, True)
        del pow_19

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(mean_19, full_3, float("1e-06"), True)
        del mean_19

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_19 = paddle._C_ops.rsqrt(scale_35)
        del scale_35

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_39 = paddle._C_ops.multiply(add_31, rsqrt_19)
        del rsqrt_19

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_40 = paddle._C_ops.multiply(parameter_48, multiply_39)
        del multiply_39, parameter_48

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_77 = paddle._C_ops.matmul(multiply_40, parameter_52, False, False)
        del parameter_52

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(matmul_77, full_int_array_2)
        del matmul_77

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_78 = paddle._C_ops.matmul(multiply_40, parameter_51, False, False)
        del parameter_51

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(matmul_78, full_int_array_2)
        del matmul_78

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_79 = paddle._C_ops.matmul(multiply_40, parameter_50, False, False)
        del multiply_40, parameter_50

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(matmul_79, full_int_array_2)
        del matmul_79

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.matmul: (1x8x1x1xf32) <- (1x8x1x64xf32, 1x8x1x64xf32)
        matmul_80 = paddle._C_ops.matmul(transpose_42, transpose_43, False, True)
        del transpose_42

        # pd_op.add: (1x8x1x1xf32) <- (1x8x1x1xf32, 1x8x1x1xf32)
        add_32 = paddle._C_ops.add(matmul_80, add_20)
        del matmul_80

        # pd_op.softmax: (1x8x1x1xf32) <- (1x8x1x1xf32)
        softmax_10 = paddle._C_ops.softmax(add_32, -1)
        del add_32

        # pd_op.dropout: (1x8x1x1xf32, 1x8x1x1xui8) <- (1x8x1x1xf32, None, 1xf32)
        dropout_78, dropout_79 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_10

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x1xf32, 1x8x1x64xf32)
        matmul_81 = paddle._C_ops.matmul(dropout_78, transpose_44, False, False)
        del dropout_78

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_45 = paddle._C_ops.transpose(matmul_81, [0, 2, 1, 3])
        del matmul_81

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_45, full_int_array_4)
        del transpose_45

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_82 = paddle._C_ops.matmul(reshape_43, parameter_49, False, False)
        del parameter_49, reshape_43

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_80, dropout_81 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_82, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_82

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_33 = paddle._C_ops.add(add_31, dropout_80)
        del add_31, dropout_80

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_20 = paddle._C_ops.pow(add_33, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_20 = paddle._C_ops.mean(pow_20, full_int_array_1, True)
        del pow_20

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(mean_20, full_3, float("1e-06"), True)
        del mean_20

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_20 = paddle._C_ops.rsqrt(scale_36)
        del scale_36

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_41 = paddle._C_ops.multiply(add_33, rsqrt_20)
        del rsqrt_20

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_42 = paddle._C_ops.multiply(parameter_43, multiply_41)
        del multiply_41, parameter_43

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_83 = paddle._C_ops.matmul(multiply_42, parameter_47, False, False)
        del multiply_42, parameter_47

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(matmul_83, full_int_array_2)
        del matmul_83

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_84 = paddle._C_ops.matmul(dropout_50, parameter_46, False, False)
        del parameter_46

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(matmul_84, full_int_array_2)
        del matmul_84

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_47 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_85 = paddle._C_ops.matmul(dropout_50, parameter_45, False, False)
        del parameter_45

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(matmul_85, full_int_array_2)
        del matmul_85

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.matmul: (1x8x1x20xf32) <- (1x8x1x64xf32, 1x8x20x64xf32)
        matmul_86 = paddle._C_ops.matmul(transpose_46, transpose_47, False, True)
        del transpose_46

        # pd_op.add: (1x8x1x20xf32) <- (1x8x1x20xf32, 1x8x1x20xf32)
        add_34 = paddle._C_ops.add(matmul_86, add_23)
        del matmul_86

        # pd_op.softmax: (1x8x1x20xf32) <- (1x8x1x20xf32)
        softmax_11 = paddle._C_ops.softmax(add_34, -1)
        del add_34

        # pd_op.dropout: (1x8x1x20xf32, 1x8x1x20xui8) <- (1x8x1x20xf32, None, 1xf32)
        dropout_82, dropout_83 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_11

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x20xf32, 1x8x20x64xf32)
        matmul_87 = paddle._C_ops.matmul(dropout_82, transpose_48, False, False)
        del dropout_82

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_49 = paddle._C_ops.transpose(matmul_87, [0, 2, 1, 3])
        del matmul_87

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_49, full_int_array_4)
        del transpose_49

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_88 = paddle._C_ops.matmul(reshape_47, parameter_44, False, False)
        del parameter_44, reshape_47

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_84, dropout_85 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_88, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_88

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_35 = paddle._C_ops.add(add_33, dropout_84)
        del add_33, dropout_84

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_21 = paddle._C_ops.pow(add_35, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_21 = paddle._C_ops.mean(pow_21, full_int_array_1, True)
        del pow_21

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(mean_21, full_3, float("1e-06"), True)
        del mean_21

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_21 = paddle._C_ops.rsqrt(scale_37)
        del scale_37

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_43 = paddle._C_ops.multiply(add_35, rsqrt_21)
        del rsqrt_21

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_44 = paddle._C_ops.multiply(parameter_40, multiply_43)
        del multiply_43, parameter_40

        # pd_op.matmul: (1x1x2048xf32) <- (1x1x512xf32, 512x2048xf32)
        matmul_89 = paddle._C_ops.matmul(multiply_44, parameter_42, False, False)
        del multiply_44, parameter_42

        # pd_op.relu: (1x1x2048xf32) <- (1x1x2048xf32)
        relu_8 = paddle._C_ops.relu(matmul_89)
        del matmul_89

        # pd_op.dropout: (1x1x2048xf32, 1x1x2048xui8) <- (1x1x2048xf32, None, 1xf32)
        dropout_86, dropout_87 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_8, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_8

        # pd_op.matmul: (1x1x512xf32) <- (1x1x2048xf32, 2048x512xf32)
        matmul_90 = paddle._C_ops.matmul(dropout_86, parameter_41, False, False)
        del dropout_86, parameter_41

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_88, dropout_89 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_90, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_90

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_36 = paddle._C_ops.add(dropout_88, add_35)
        del add_35, dropout_88

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_22 = paddle._C_ops.pow(add_36, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_22 = paddle._C_ops.mean(pow_22, full_int_array_1, True)
        del pow_22

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(mean_22, full_3, float("1e-06"), True)
        del mean_22

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_22 = paddle._C_ops.rsqrt(scale_38)
        del scale_38

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_45 = paddle._C_ops.multiply(add_36, rsqrt_22)
        del rsqrt_22

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_46 = paddle._C_ops.multiply(parameter_35, multiply_45)
        del multiply_45, parameter_35

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_91 = paddle._C_ops.matmul(multiply_46, parameter_39, False, False)
        del parameter_39

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(matmul_91, full_int_array_2)
        del matmul_91

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_48, [0, 2, 1, 3])
        del reshape_48

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_92 = paddle._C_ops.matmul(multiply_46, parameter_38, False, False)
        del parameter_38

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(matmul_92, full_int_array_2)
        del matmul_92

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_51 = paddle._C_ops.transpose(reshape_49, [0, 2, 1, 3])
        del reshape_49

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_93 = paddle._C_ops.matmul(multiply_46, parameter_37, False, False)
        del multiply_46, parameter_37

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(matmul_93, full_int_array_2)
        del matmul_93

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_50, [0, 2, 1, 3])
        del reshape_50

        # pd_op.matmul: (1x8x1x1xf32) <- (1x8x1x64xf32, 1x8x1x64xf32)
        matmul_94 = paddle._C_ops.matmul(transpose_50, transpose_51, False, True)
        del transpose_50

        # pd_op.add: (1x8x1x1xf32) <- (1x8x1x1xf32, 1x8x1x1xf32)
        add_37 = paddle._C_ops.add(matmul_94, add_20)
        del matmul_94

        # pd_op.softmax: (1x8x1x1xf32) <- (1x8x1x1xf32)
        softmax_12 = paddle._C_ops.softmax(add_37, -1)
        del add_37

        # pd_op.dropout: (1x8x1x1xf32, 1x8x1x1xui8) <- (1x8x1x1xf32, None, 1xf32)
        dropout_90, dropout_91 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_12, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_12

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x1xf32, 1x8x1x64xf32)
        matmul_95 = paddle._C_ops.matmul(dropout_90, transpose_52, False, False)
        del dropout_90

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_53 = paddle._C_ops.transpose(matmul_95, [0, 2, 1, 3])
        del matmul_95

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_51 = paddle._C_ops.reshape(transpose_53, full_int_array_4)
        del transpose_53

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_96 = paddle._C_ops.matmul(reshape_51, parameter_36, False, False)
        del parameter_36, reshape_51

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_92, dropout_93 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_96, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_96

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_38 = paddle._C_ops.add(add_36, dropout_92)
        del add_36, dropout_92

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_23 = paddle._C_ops.pow(add_38, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_23 = paddle._C_ops.mean(pow_23, full_int_array_1, True)
        del pow_23

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(mean_23, full_3, float("1e-06"), True)
        del mean_23

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_23 = paddle._C_ops.rsqrt(scale_39)
        del scale_39

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_47 = paddle._C_ops.multiply(add_38, rsqrt_23)
        del rsqrt_23

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_48 = paddle._C_ops.multiply(parameter_30, multiply_47)
        del multiply_47, parameter_30

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_97 = paddle._C_ops.matmul(multiply_48, parameter_34, False, False)
        del multiply_48, parameter_34

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(matmul_97, full_int_array_2)
        del matmul_97

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_54 = paddle._C_ops.transpose(reshape_52, [0, 2, 1, 3])
        del reshape_52

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_98 = paddle._C_ops.matmul(dropout_50, parameter_33, False, False)
        del parameter_33

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(matmul_98, full_int_array_2)
        del matmul_98

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_53, [0, 2, 1, 3])
        del reshape_53

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_99 = paddle._C_ops.matmul(dropout_50, parameter_32, False, False)
        del parameter_32

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(matmul_99, full_int_array_2)
        del matmul_99

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_54, [0, 2, 1, 3])
        del reshape_54

        # pd_op.matmul: (1x8x1x20xf32) <- (1x8x1x64xf32, 1x8x20x64xf32)
        matmul_100 = paddle._C_ops.matmul(transpose_54, transpose_55, False, True)
        del transpose_54

        # pd_op.add: (1x8x1x20xf32) <- (1x8x1x20xf32, 1x8x1x20xf32)
        add_39 = paddle._C_ops.add(matmul_100, add_23)
        del matmul_100

        # pd_op.softmax: (1x8x1x20xf32) <- (1x8x1x20xf32)
        softmax_13 = paddle._C_ops.softmax(add_39, -1)
        del add_39

        # pd_op.dropout: (1x8x1x20xf32, 1x8x1x20xui8) <- (1x8x1x20xf32, None, 1xf32)
        dropout_94, dropout_95 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_13, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_13

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x20xf32, 1x8x20x64xf32)
        matmul_101 = paddle._C_ops.matmul(dropout_94, transpose_56, False, False)
        del dropout_94

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_57 = paddle._C_ops.transpose(matmul_101, [0, 2, 1, 3])
        del matmul_101

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_55 = paddle._C_ops.reshape(transpose_57, full_int_array_4)
        del transpose_57

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_102 = paddle._C_ops.matmul(reshape_55, parameter_31, False, False)
        del parameter_31, reshape_55

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_96, dropout_97 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_102, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_102

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_40 = paddle._C_ops.add(add_38, dropout_96)
        del add_38, dropout_96

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_24 = paddle._C_ops.pow(add_40, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_24 = paddle._C_ops.mean(pow_24, full_int_array_1, True)
        del pow_24

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_40 = paddle._C_ops.scale(mean_24, full_3, float("1e-06"), True)
        del mean_24

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_24 = paddle._C_ops.rsqrt(scale_40)
        del scale_40

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_49 = paddle._C_ops.multiply(add_40, rsqrt_24)
        del rsqrt_24

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_50 = paddle._C_ops.multiply(parameter_27, multiply_49)
        del multiply_49, parameter_27

        # pd_op.matmul: (1x1x2048xf32) <- (1x1x512xf32, 512x2048xf32)
        matmul_103 = paddle._C_ops.matmul(multiply_50, parameter_29, False, False)
        del multiply_50, parameter_29

        # pd_op.relu: (1x1x2048xf32) <- (1x1x2048xf32)
        relu_9 = paddle._C_ops.relu(matmul_103)
        del matmul_103

        # pd_op.dropout: (1x1x2048xf32, 1x1x2048xui8) <- (1x1x2048xf32, None, 1xf32)
        dropout_98, dropout_99 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_9, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_9

        # pd_op.matmul: (1x1x512xf32) <- (1x1x2048xf32, 2048x512xf32)
        matmul_104 = paddle._C_ops.matmul(dropout_98, parameter_28, False, False)
        del dropout_98, parameter_28

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_100, dropout_101 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_104, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_104

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_41 = paddle._C_ops.add(dropout_100, add_40)
        del add_40, dropout_100

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_25 = paddle._C_ops.pow(add_41, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_25 = paddle._C_ops.mean(pow_25, full_int_array_1, True)
        del pow_25

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_41 = paddle._C_ops.scale(mean_25, full_3, float("1e-06"), True)
        del mean_25

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_25 = paddle._C_ops.rsqrt(scale_41)
        del scale_41

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_51 = paddle._C_ops.multiply(add_41, rsqrt_25)
        del rsqrt_25

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_52 = paddle._C_ops.multiply(parameter_22, multiply_51)
        del multiply_51, parameter_22

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_105 = paddle._C_ops.matmul(multiply_52, parameter_26, False, False)
        del parameter_26

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(matmul_105, full_int_array_2)
        del matmul_105

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_56, [0, 2, 1, 3])
        del reshape_56

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_106 = paddle._C_ops.matmul(multiply_52, parameter_25, False, False)
        del parameter_25

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(matmul_106, full_int_array_2)
        del matmul_106

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_59 = paddle._C_ops.transpose(reshape_57, [0, 2, 1, 3])
        del reshape_57

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_107 = paddle._C_ops.matmul(multiply_52, parameter_24, False, False)
        del multiply_52, parameter_24

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(matmul_107, full_int_array_2)
        del matmul_107

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_60 = paddle._C_ops.transpose(reshape_58, [0, 2, 1, 3])
        del reshape_58

        # pd_op.matmul: (1x8x1x1xf32) <- (1x8x1x64xf32, 1x8x1x64xf32)
        matmul_108 = paddle._C_ops.matmul(transpose_58, transpose_59, False, True)
        del transpose_58

        # pd_op.add: (1x8x1x1xf32) <- (1x8x1x1xf32, 1x8x1x1xf32)
        add_42 = paddle._C_ops.add(matmul_108, add_20)
        del matmul_108

        # pd_op.softmax: (1x8x1x1xf32) <- (1x8x1x1xf32)
        softmax_14 = paddle._C_ops.softmax(add_42, -1)
        del add_42

        # pd_op.dropout: (1x8x1x1xf32, 1x8x1x1xui8) <- (1x8x1x1xf32, None, 1xf32)
        dropout_102, dropout_103 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_14, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_14

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x1xf32, 1x8x1x64xf32)
        matmul_109 = paddle._C_ops.matmul(dropout_102, transpose_60, False, False)
        del dropout_102

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_61 = paddle._C_ops.transpose(matmul_109, [0, 2, 1, 3])
        del matmul_109

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_59 = paddle._C_ops.reshape(transpose_61, full_int_array_4)
        del transpose_61

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_110 = paddle._C_ops.matmul(reshape_59, parameter_23, False, False)
        del parameter_23, reshape_59

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_104, dropout_105 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_110, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_110

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_43 = paddle._C_ops.add(add_41, dropout_104)
        del add_41, dropout_104

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_26 = paddle._C_ops.pow(add_43, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_26 = paddle._C_ops.mean(pow_26, full_int_array_1, True)
        del pow_26

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_42 = paddle._C_ops.scale(mean_26, full_3, float("1e-06"), True)
        del mean_26

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_26 = paddle._C_ops.rsqrt(scale_42)
        del scale_42

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_53 = paddle._C_ops.multiply(add_43, rsqrt_26)
        del rsqrt_26

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_54 = paddle._C_ops.multiply(parameter_17, multiply_53)
        del multiply_53, parameter_17

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_111 = paddle._C_ops.matmul(multiply_54, parameter_21, False, False)
        del multiply_54, parameter_21

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(matmul_111, full_int_array_2)
        del matmul_111

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_60, [0, 2, 1, 3])
        del reshape_60

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_112 = paddle._C_ops.matmul(dropout_50, parameter_20, False, False)
        del parameter_20

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(matmul_112, full_int_array_2)
        del matmul_112

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_63 = paddle._C_ops.transpose(reshape_61, [0, 2, 1, 3])
        del reshape_61

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_113 = paddle._C_ops.matmul(dropout_50, parameter_19, False, False)
        del parameter_19

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(matmul_113, full_int_array_2)
        del matmul_113

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_62, [0, 2, 1, 3])
        del reshape_62

        # pd_op.matmul: (1x8x1x20xf32) <- (1x8x1x64xf32, 1x8x20x64xf32)
        matmul_114 = paddle._C_ops.matmul(transpose_62, transpose_63, False, True)
        del transpose_62

        # pd_op.add: (1x8x1x20xf32) <- (1x8x1x20xf32, 1x8x1x20xf32)
        add_44 = paddle._C_ops.add(matmul_114, add_23)
        del matmul_114

        # pd_op.softmax: (1x8x1x20xf32) <- (1x8x1x20xf32)
        softmax_15 = paddle._C_ops.softmax(add_44, -1)
        del add_44

        # pd_op.dropout: (1x8x1x20xf32, 1x8x1x20xui8) <- (1x8x1x20xf32, None, 1xf32)
        dropout_106, dropout_107 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_15, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_15

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x20xf32, 1x8x20x64xf32)
        matmul_115 = paddle._C_ops.matmul(dropout_106, transpose_64, False, False)
        del dropout_106

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_65 = paddle._C_ops.transpose(matmul_115, [0, 2, 1, 3])
        del matmul_115

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_63 = paddle._C_ops.reshape(transpose_65, full_int_array_4)
        del transpose_65

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_116 = paddle._C_ops.matmul(reshape_63, parameter_18, False, False)
        del parameter_18, reshape_63

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_108, dropout_109 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_116, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_116

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_45 = paddle._C_ops.add(add_43, dropout_108)
        del add_43, dropout_108

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_27 = paddle._C_ops.pow(add_45, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_27 = paddle._C_ops.mean(pow_27, full_int_array_1, True)
        del pow_27

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_43 = paddle._C_ops.scale(mean_27, full_3, float("1e-06"), True)
        del mean_27

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_27 = paddle._C_ops.rsqrt(scale_43)
        del scale_43

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_55 = paddle._C_ops.multiply(add_45, rsqrt_27)
        del rsqrt_27

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_56 = paddle._C_ops.multiply(parameter_14, multiply_55)
        del multiply_55, parameter_14

        # pd_op.matmul: (1x1x2048xf32) <- (1x1x512xf32, 512x2048xf32)
        matmul_117 = paddle._C_ops.matmul(multiply_56, parameter_16, False, False)
        del multiply_56, parameter_16

        # pd_op.relu: (1x1x2048xf32) <- (1x1x2048xf32)
        relu_10 = paddle._C_ops.relu(matmul_117)
        del matmul_117

        # pd_op.dropout: (1x1x2048xf32, 1x1x2048xui8) <- (1x1x2048xf32, None, 1xf32)
        dropout_110, dropout_111 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_10, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_10

        # pd_op.matmul: (1x1x512xf32) <- (1x1x2048xf32, 2048x512xf32)
        matmul_118 = paddle._C_ops.matmul(dropout_110, parameter_15, False, False)
        del dropout_110, parameter_15

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_112, dropout_113 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_118, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_118

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_46 = paddle._C_ops.add(dropout_112, add_45)
        del add_45, dropout_112

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_28 = paddle._C_ops.pow(add_46, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_28 = paddle._C_ops.mean(pow_28, full_int_array_1, True)
        del pow_28

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_44 = paddle._C_ops.scale(mean_28, full_3, float("1e-06"), True)
        del mean_28

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_28 = paddle._C_ops.rsqrt(scale_44)
        del scale_44

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_57 = paddle._C_ops.multiply(add_46, rsqrt_28)
        del rsqrt_28

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_58 = paddle._C_ops.multiply(parameter_9, multiply_57)
        del multiply_57, parameter_9

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_119 = paddle._C_ops.matmul(multiply_58, parameter_13, False, False)
        del parameter_13

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(matmul_119, full_int_array_2)
        del matmul_119

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_66 = paddle._C_ops.transpose(reshape_64, [0, 2, 1, 3])
        del reshape_64

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_120 = paddle._C_ops.matmul(multiply_58, parameter_12, False, False)
        del parameter_12

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(matmul_120, full_int_array_2)
        del matmul_120

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_65, [0, 2, 1, 3])
        del reshape_65

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_121 = paddle._C_ops.matmul(multiply_58, parameter_11, False, False)
        del multiply_58, parameter_11

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(matmul_121, full_int_array_2)
        del matmul_121

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_68 = paddle._C_ops.transpose(reshape_66, [0, 2, 1, 3])
        del reshape_66

        # pd_op.matmul: (1x8x1x1xf32) <- (1x8x1x64xf32, 1x8x1x64xf32)
        matmul_122 = paddle._C_ops.matmul(transpose_66, transpose_67, False, True)
        del transpose_66

        # pd_op.add: (1x8x1x1xf32) <- (1x8x1x1xf32, 1x8x1x1xf32)
        add_47 = paddle._C_ops.add(matmul_122, add_20)
        del add_20, matmul_122

        # pd_op.softmax: (1x8x1x1xf32) <- (1x8x1x1xf32)
        softmax_16 = paddle._C_ops.softmax(add_47, -1)
        del add_47

        # pd_op.dropout: (1x8x1x1xf32, 1x8x1x1xui8) <- (1x8x1x1xf32, None, 1xf32)
        dropout_114, dropout_115 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_16, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_16

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x1xf32, 1x8x1x64xf32)
        matmul_123 = paddle._C_ops.matmul(dropout_114, transpose_68, False, False)
        del dropout_114

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_69 = paddle._C_ops.transpose(matmul_123, [0, 2, 1, 3])
        del matmul_123

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_69, full_int_array_4)
        del transpose_69

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_124 = paddle._C_ops.matmul(reshape_67, parameter_10, False, False)
        del parameter_10, reshape_67

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_116, dropout_117 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_124, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_124

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_48 = paddle._C_ops.add(add_46, dropout_116)
        del add_46, dropout_116

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_29 = paddle._C_ops.pow(add_48, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_29 = paddle._C_ops.mean(pow_29, full_int_array_1, True)
        del pow_29

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_45 = paddle._C_ops.scale(mean_29, full_3, float("1e-06"), True)
        del mean_29

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_29 = paddle._C_ops.rsqrt(scale_45)
        del scale_45

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_59 = paddle._C_ops.multiply(add_48, rsqrt_29)
        del rsqrt_29

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_60 = paddle._C_ops.multiply(parameter_4, multiply_59)
        del multiply_59, parameter_4

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_125 = paddle._C_ops.matmul(multiply_60, parameter_8, False, False)
        del multiply_60, parameter_8

        # pd_op.reshape: (1x1x8x64xf32) <- (1x1x512xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(matmul_125, full_int_array_2)
        del matmul_125

        # pd_op.transpose: (1x8x1x64xf32) <- (1x1x8x64xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_68, [0, 2, 1, 3])
        del reshape_68

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_126 = paddle._C_ops.matmul(dropout_50, parameter_7, False, False)
        del parameter_7

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(matmul_126, full_int_array_2)
        del matmul_126

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_71 = paddle._C_ops.transpose(reshape_69, [0, 2, 1, 3])
        del reshape_69

        # pd_op.matmul: (1x20x512xf32) <- (1x20x512xf32, 512x512xf32)
        matmul_127 = paddle._C_ops.matmul(dropout_50, parameter_6, False, False)
        del parameter_6

        # pd_op.reshape: (1x20x8x64xf32) <- (1x20x512xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(matmul_127, full_int_array_2)
        del full_int_array_2, matmul_127

        # pd_op.transpose: (1x8x20x64xf32) <- (1x20x8x64xf32)
        transpose_72 = paddle._C_ops.transpose(reshape_70, [0, 2, 1, 3])
        del reshape_70

        # pd_op.matmul: (1x8x1x20xf32) <- (1x8x1x64xf32, 1x8x20x64xf32)
        matmul_128 = paddle._C_ops.matmul(transpose_70, transpose_71, False, True)
        del transpose_70

        # pd_op.add: (1x8x1x20xf32) <- (1x8x1x20xf32, 1x8x1x20xf32)
        add_49 = paddle._C_ops.add(matmul_128, add_23)
        del add_23, matmul_128

        # pd_op.softmax: (1x8x1x20xf32) <- (1x8x1x20xf32)
        softmax_17 = paddle._C_ops.softmax(add_49, -1)
        del add_49

        # pd_op.dropout: (1x8x1x20xf32, 1x8x1x20xui8) <- (1x8x1x20xf32, None, 1xf32)
        dropout_118, dropout_119 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_17, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_17

        # pd_op.matmul: (1x8x1x64xf32) <- (1x8x1x20xf32, 1x8x20x64xf32)
        matmul_129 = paddle._C_ops.matmul(dropout_118, transpose_72, False, False)
        del dropout_118

        # pd_op.transpose: (1x1x8x64xf32) <- (1x8x1x64xf32)
        transpose_73 = paddle._C_ops.transpose(matmul_129, [0, 2, 1, 3])
        del matmul_129

        # pd_op.reshape: (1x1x512xf32) <- (1x1x8x64xf32, 3xi64)
        reshape_71 = paddle._C_ops.reshape(transpose_73, full_int_array_4)
        del full_int_array_4, transpose_73

        # pd_op.matmul: (1x1x512xf32) <- (1x1x512xf32, 512x512xf32)
        matmul_130 = paddle._C_ops.matmul(reshape_71, parameter_5, False, False)
        del parameter_5, reshape_71

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_120, dropout_121 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_130, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_130

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_50 = paddle._C_ops.add(add_48, dropout_120)
        del add_48, dropout_120

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_30 = paddle._C_ops.pow(add_50, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_30 = paddle._C_ops.mean(pow_30, full_int_array_1, True)
        del pow_30

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_46 = paddle._C_ops.scale(mean_30, full_3, float("1e-06"), True)
        del mean_30

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_30 = paddle._C_ops.rsqrt(scale_46)
        del scale_46

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_61 = paddle._C_ops.multiply(add_50, rsqrt_30)
        del rsqrt_30

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_62 = paddle._C_ops.multiply(parameter_1, multiply_61)
        del multiply_61, parameter_1

        # pd_op.matmul: (1x1x2048xf32) <- (1x1x512xf32, 512x2048xf32)
        matmul_131 = paddle._C_ops.matmul(multiply_62, parameter_3, False, False)
        del multiply_62, parameter_3

        # pd_op.relu: (1x1x2048xf32) <- (1x1x2048xf32)
        relu_11 = paddle._C_ops.relu(matmul_131)
        del matmul_131

        # pd_op.dropout: (1x1x2048xf32, 1x1x2048xui8) <- (1x1x2048xf32, None, 1xf32)
        dropout_122, dropout_123 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_11, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_11

        # pd_op.matmul: (1x1x512xf32) <- (1x1x2048xf32, 2048x512xf32)
        matmul_132 = paddle._C_ops.matmul(dropout_122, parameter_2, False, False)
        del dropout_122, parameter_2

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_124, dropout_125 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                matmul_132, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del matmul_132

        # pd_op.add: (1x1x512xf32) <- (1x1x512xf32, 1x1x512xf32)
        add_51 = paddle._C_ops.add(dropout_124, add_50)
        del add_50, dropout_124

        # pd_op.pow: (1x1x512xf32) <- (1x1x512xf32)
        pow_31 = paddle._C_ops.pow(add_51, float("2"))

        # pd_op.mean: (1x1x1xf32) <- (1x1x512xf32, 1xi64)
        mean_31 = paddle._C_ops.mean(pow_31, full_int_array_1, True)
        del full_int_array_1, pow_31

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_47 = paddle._C_ops.scale(mean_31, full_3, float("1e-06"), True)
        del full_3, mean_31

        # pd_op.rsqrt: (1x1x1xf32) <- (1x1x1xf32)
        rsqrt_31 = paddle._C_ops.rsqrt(scale_47)
        del scale_47

        # pd_op.multiply: (1x1x512xf32) <- (1x1x512xf32, 1x1x1xf32)
        multiply_63 = paddle._C_ops.multiply(add_51, rsqrt_31)
        del add_51, rsqrt_31

        # pd_op.multiply: (1x1x512xf32) <- (512xf32, 1x1x512xf32)
        multiply_64 = paddle._C_ops.multiply(parameter_0, multiply_63)
        del multiply_63, parameter_0

        # pd_op.dropout: (1x1x512xf32, 1x1x512xui8) <- (1x1x512xf32, None, 1xf32)
        dropout_126, dropout_127 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                multiply_64, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_2, multiply_64

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full(
            [1], float("0.0441942"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x512xf32) <- (1x1x512xf32, 1xf32)
        scale_48 = paddle._C_ops.scale(dropout_126, full_21, float("0"), True)
        del dropout_126, full_21

        # pd_op.matmul: (1x1x32128xf32) <- (1x1x512xf32, 32128x512xf32)
        matmul_0 = paddle._C_ops.matmul(scale_48, parameter_130, False, True)
        del (
            dropout_50,
            parameter_130,
            scale_48,
            transpose_26,
            transpose_27,
            transpose_31,
            transpose_32,
            transpose_35,
            transpose_36,
            transpose_39,
            transpose_40,
            transpose_43,
            transpose_44,
            transpose_47,
            transpose_48,
            transpose_51,
            transpose_52,
            transpose_55,
            transpose_56,
            transpose_59,
            transpose_60,
            transpose_63,
            transpose_64,
            transpose_67,
            transpose_68,
            transpose_71,
            transpose_72,
        )

        return matmul_0
