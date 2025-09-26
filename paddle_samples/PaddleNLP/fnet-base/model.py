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
        data_0,
        data_1,
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [21]

        # pd_op.slice: (1x21xi64) <- (1x512xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            parameter_0, [1], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_1, parameter_0

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 32000x768xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_105, 3, False)
        del data_0, parameter_105

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 4x768xf32)
        embedding_1 = paddle._C_ops.embedding(data_1, parameter_103, -1, False)
        del data_1, parameter_103

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)
        del embedding_0, embedding_1

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 512x768xf32)
        embedding_2 = paddle._C_ops.embedding(slice_0, parameter_104, -1, False)
        del parameter_104, slice_0

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)
        del add_0, embedding_2

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_102, parameter_101, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_1, parameter_101, parameter_102

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_100, False, False)
        del layer_norm_0, parameter_100

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_99)
        del matmul_0, parameter_99

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_2, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_2

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_0 = paddle._C_ops.fft_r2c(dropout_0, [0, 1, 2], "backward", True, False)

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_0 = paddle._C_ops.real(fft_r2c_0)
        del fft_r2c_0

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_3 = paddle._C_ops.add(dropout_0, real_0)
        del dropout_0, real_0

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_98, parameter_97, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_3, parameter_97, parameter_98

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_3, parameter_96, False, False)
        del parameter_96

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_4 = paddle._C_ops.add(matmul_1, parameter_95)
        del matmul_1, parameter_95

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_4, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_0 = paddle._C_ops.pow(add_4, float("3"))

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.044715"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(pow_0, full_2, float("0"), True)
        del pow_0

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_5 = paddle._C_ops.add(add_4, scale_1)
        del add_4, scale_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.797885"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(add_5, full_3, float("0"), True)
        del add_5

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_1 = paddle._C_ops.tanh(scale_2)
        del scale_2

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(tanh_1, full_4, float("1"), True)
        del tanh_1

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, scale_3)
        del scale_0, scale_3

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_2 = paddle._C_ops.matmul(multiply_0, parameter_94, False, False)
        del multiply_0, parameter_94

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_6 = paddle._C_ops.add(matmul_2, parameter_93)
        del matmul_2, parameter_93

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_6, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_6

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_7 = paddle._C_ops.add(layer_norm_3, dropout_2)
        del dropout_2, layer_norm_3

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_92, parameter_91, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_7, parameter_91, parameter_92

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_1 = paddle._C_ops.fft_r2c(
            layer_norm_6, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_1 = paddle._C_ops.real(fft_r2c_1)
        del fft_r2c_1

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_8 = paddle._C_ops.add(layer_norm_6, real_1)
        del layer_norm_6, real_1

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_90, parameter_89, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_8, parameter_89, parameter_90

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_3 = paddle._C_ops.matmul(layer_norm_9, parameter_88, False, False)
        del parameter_88

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_9 = paddle._C_ops.add(matmul_3, parameter_87)
        del matmul_3, parameter_87

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(add_9, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_1 = paddle._C_ops.pow(add_9, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(pow_1, full_2, float("0"), True)
        del pow_1

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_10 = paddle._C_ops.add(add_9, scale_5)
        del add_9, scale_5

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(add_10, full_3, float("0"), True)
        del add_10

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_2 = paddle._C_ops.tanh(scale_6)
        del scale_6

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(tanh_2, full_4, float("1"), True)
        del tanh_2

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_1 = paddle._C_ops.multiply(scale_4, scale_7)
        del scale_4, scale_7

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_4 = paddle._C_ops.matmul(multiply_1, parameter_86, False, False)
        del multiply_1, parameter_86

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_11 = paddle._C_ops.add(matmul_4, parameter_85)
        del matmul_4, parameter_85

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_11, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_11

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_12 = paddle._C_ops.add(layer_norm_9, dropout_4)
        del dropout_4, layer_norm_9

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_84, parameter_83, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_12, parameter_83, parameter_84

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_2 = paddle._C_ops.fft_r2c(
            layer_norm_12, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_2 = paddle._C_ops.real(fft_r2c_2)
        del fft_r2c_2

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_13 = paddle._C_ops.add(layer_norm_12, real_2)
        del layer_norm_12, real_2

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_82, parameter_81, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_13, parameter_81, parameter_82

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_5 = paddle._C_ops.matmul(layer_norm_15, parameter_80, False, False)
        del parameter_80

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_14 = paddle._C_ops.add(matmul_5, parameter_79)
        del matmul_5, parameter_79

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(add_14, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_2 = paddle._C_ops.pow(add_14, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(pow_2, full_2, float("0"), True)
        del pow_2

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_15 = paddle._C_ops.add(add_14, scale_9)
        del add_14, scale_9

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(add_15, full_3, float("0"), True)
        del add_15

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_3 = paddle._C_ops.tanh(scale_10)
        del scale_10

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(tanh_3, full_4, float("1"), True)
        del tanh_3

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_2 = paddle._C_ops.multiply(scale_8, scale_11)
        del scale_11, scale_8

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_6 = paddle._C_ops.matmul(multiply_2, parameter_78, False, False)
        del multiply_2, parameter_78

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_16 = paddle._C_ops.add(matmul_6, parameter_77)
        del matmul_6, parameter_77

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_16, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_16

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_17 = paddle._C_ops.add(layer_norm_15, dropout_6)
        del dropout_6, layer_norm_15

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_76, parameter_75, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_17, parameter_75, parameter_76

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_3 = paddle._C_ops.fft_r2c(
            layer_norm_18, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_3 = paddle._C_ops.real(fft_r2c_3)
        del fft_r2c_3

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_18 = paddle._C_ops.add(layer_norm_18, real_3)
        del layer_norm_18, real_3

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_74, parameter_73, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_18, parameter_73, parameter_74

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_21, parameter_72, False, False)
        del parameter_72

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_19 = paddle._C_ops.add(matmul_7, parameter_71)
        del matmul_7, parameter_71

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(add_19, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_3 = paddle._C_ops.pow(add_19, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(pow_3, full_2, float("0"), True)
        del pow_3

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_20 = paddle._C_ops.add(add_19, scale_13)
        del add_19, scale_13

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(add_20, full_3, float("0"), True)
        del add_20

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_4 = paddle._C_ops.tanh(scale_14)
        del scale_14

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(tanh_4, full_4, float("1"), True)
        del tanh_4

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_3 = paddle._C_ops.multiply(scale_12, scale_15)
        del scale_12, scale_15

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_8 = paddle._C_ops.matmul(multiply_3, parameter_70, False, False)
        del multiply_3, parameter_70

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_8, parameter_69)
        del matmul_8, parameter_69

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_21, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_21

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_22 = paddle._C_ops.add(layer_norm_21, dropout_8)
        del dropout_8, layer_norm_21

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_68, parameter_67, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_22, parameter_67, parameter_68

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_4 = paddle._C_ops.fft_r2c(
            layer_norm_24, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_4 = paddle._C_ops.real(fft_r2c_4)
        del fft_r2c_4

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_23 = paddle._C_ops.add(layer_norm_24, real_4)
        del layer_norm_24, real_4

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_66, parameter_65, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_23, parameter_65, parameter_66

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_27, parameter_64, False, False)
        del parameter_64

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_24 = paddle._C_ops.add(matmul_9, parameter_63)
        del matmul_9, parameter_63

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(add_24, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_4 = paddle._C_ops.pow(add_24, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(pow_4, full_2, float("0"), True)
        del pow_4

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_25 = paddle._C_ops.add(add_24, scale_17)
        del add_24, scale_17

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(add_25, full_3, float("0"), True)
        del add_25

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_5 = paddle._C_ops.tanh(scale_18)
        del scale_18

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(tanh_5, full_4, float("1"), True)
        del tanh_5

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_4 = paddle._C_ops.multiply(scale_16, scale_19)
        del scale_16, scale_19

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_10 = paddle._C_ops.matmul(multiply_4, parameter_62, False, False)
        del multiply_4, parameter_62

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_26 = paddle._C_ops.add(matmul_10, parameter_61)
        del matmul_10, parameter_61

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_26, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_26

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_27 = paddle._C_ops.add(layer_norm_27, dropout_10)
        del dropout_10, layer_norm_27

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_60, parameter_59, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_27, parameter_59, parameter_60

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_5 = paddle._C_ops.fft_r2c(
            layer_norm_30, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_5 = paddle._C_ops.real(fft_r2c_5)
        del fft_r2c_5

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_28 = paddle._C_ops.add(layer_norm_30, real_5)
        del layer_norm_30, real_5

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_58, parameter_57, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_28, parameter_57, parameter_58

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_33, parameter_56, False, False)
        del parameter_56

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_29 = paddle._C_ops.add(matmul_11, parameter_55)
        del matmul_11, parameter_55

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(add_29, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_5 = paddle._C_ops.pow(add_29, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(pow_5, full_2, float("0"), True)
        del pow_5

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_30 = paddle._C_ops.add(add_29, scale_21)
        del add_29, scale_21

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(add_30, full_3, float("0"), True)
        del add_30

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_6 = paddle._C_ops.tanh(scale_22)
        del scale_22

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(tanh_6, full_4, float("1"), True)
        del tanh_6

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_5 = paddle._C_ops.multiply(scale_20, scale_23)
        del scale_20, scale_23

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_12 = paddle._C_ops.matmul(multiply_5, parameter_54, False, False)
        del multiply_5, parameter_54

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_31 = paddle._C_ops.add(matmul_12, parameter_53)
        del matmul_12, parameter_53

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_31, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_31

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_32 = paddle._C_ops.add(layer_norm_33, dropout_12)
        del dropout_12, layer_norm_33

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_32, parameter_52, parameter_51, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_32, parameter_51, parameter_52

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_6 = paddle._C_ops.fft_r2c(
            layer_norm_36, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_6 = paddle._C_ops.real(fft_r2c_6)
        del fft_r2c_6

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_33 = paddle._C_ops.add(layer_norm_36, real_6)
        del layer_norm_36, real_6

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_50, parameter_49, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_33, parameter_49, parameter_50

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_13 = paddle._C_ops.matmul(layer_norm_39, parameter_48, False, False)
        del parameter_48

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_34 = paddle._C_ops.add(matmul_13, parameter_47)
        del matmul_13, parameter_47

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(add_34, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_6 = paddle._C_ops.pow(add_34, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(pow_6, full_2, float("0"), True)
        del pow_6

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_35 = paddle._C_ops.add(add_34, scale_25)
        del add_34, scale_25

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(add_35, full_3, float("0"), True)
        del add_35

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_7 = paddle._C_ops.tanh(scale_26)
        del scale_26

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(tanh_7, full_4, float("1"), True)
        del tanh_7

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_6 = paddle._C_ops.multiply(scale_24, scale_27)
        del scale_24, scale_27

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_14 = paddle._C_ops.matmul(multiply_6, parameter_46, False, False)
        del multiply_6, parameter_46

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_36 = paddle._C_ops.add(matmul_14, parameter_45)
        del matmul_14, parameter_45

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_36, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_36

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_37 = paddle._C_ops.add(layer_norm_39, dropout_14)
        del dropout_14, layer_norm_39

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_44, parameter_43, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_37, parameter_43, parameter_44

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_7 = paddle._C_ops.fft_r2c(
            layer_norm_42, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_7 = paddle._C_ops.real(fft_r2c_7)
        del fft_r2c_7

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_38 = paddle._C_ops.add(layer_norm_42, real_7)
        del layer_norm_42, real_7

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_42, parameter_41, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_38, parameter_41, parameter_42

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_45, parameter_40, False, False)
        del parameter_40

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_39 = paddle._C_ops.add(matmul_15, parameter_39)
        del matmul_15, parameter_39

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(add_39, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_7 = paddle._C_ops.pow(add_39, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(pow_7, full_2, float("0"), True)
        del pow_7

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_40 = paddle._C_ops.add(add_39, scale_29)
        del add_39, scale_29

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(add_40, full_3, float("0"), True)
        del add_40

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_8 = paddle._C_ops.tanh(scale_30)
        del scale_30

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(tanh_8, full_4, float("1"), True)
        del tanh_8

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_7 = paddle._C_ops.multiply(scale_28, scale_31)
        del scale_28, scale_31

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_16 = paddle._C_ops.matmul(multiply_7, parameter_38, False, False)
        del multiply_7, parameter_38

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_41 = paddle._C_ops.add(matmul_16, parameter_37)
        del matmul_16, parameter_37

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_41, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_41

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_42 = paddle._C_ops.add(layer_norm_45, dropout_16)
        del dropout_16, layer_norm_45

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_36, parameter_35, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_42, parameter_35, parameter_36

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_8 = paddle._C_ops.fft_r2c(
            layer_norm_48, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_8 = paddle._C_ops.real(fft_r2c_8)
        del fft_r2c_8

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_43 = paddle._C_ops.add(layer_norm_48, real_8)
        del layer_norm_48, real_8

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_34, parameter_33, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_43, parameter_33, parameter_34

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_51, parameter_32, False, False)
        del parameter_32

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_44 = paddle._C_ops.add(matmul_17, parameter_31)
        del matmul_17, parameter_31

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(add_44, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_8 = paddle._C_ops.pow(add_44, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(pow_8, full_2, float("0"), True)
        del pow_8

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_45 = paddle._C_ops.add(add_44, scale_33)
        del add_44, scale_33

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(add_45, full_3, float("0"), True)
        del add_45

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_9 = paddle._C_ops.tanh(scale_34)
        del scale_34

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(tanh_9, full_4, float("1"), True)
        del tanh_9

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_8 = paddle._C_ops.multiply(scale_32, scale_35)
        del scale_32, scale_35

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_18 = paddle._C_ops.matmul(multiply_8, parameter_30, False, False)
        del multiply_8, parameter_30

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_46 = paddle._C_ops.add(matmul_18, parameter_29)
        del matmul_18, parameter_29

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_46, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_46

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_47 = paddle._C_ops.add(layer_norm_51, dropout_18)
        del dropout_18, layer_norm_51

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_47, parameter_28, parameter_27, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_47, parameter_27, parameter_28

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_9 = paddle._C_ops.fft_r2c(
            layer_norm_54, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_9 = paddle._C_ops.real(fft_r2c_9)
        del fft_r2c_9

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_48 = paddle._C_ops.add(layer_norm_54, real_9)
        del layer_norm_54, real_9

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_48, parameter_26, parameter_25, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_48, parameter_25, parameter_26

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_57, parameter_24, False, False)
        del parameter_24

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_49 = paddle._C_ops.add(matmul_19, parameter_23)
        del matmul_19, parameter_23

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(add_49, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_9 = paddle._C_ops.pow(add_49, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(pow_9, full_2, float("0"), True)
        del pow_9

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_50 = paddle._C_ops.add(add_49, scale_37)
        del add_49, scale_37

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(add_50, full_3, float("0"), True)
        del add_50

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_10 = paddle._C_ops.tanh(scale_38)
        del scale_38

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(tanh_10, full_4, float("1"), True)
        del tanh_10

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_9 = paddle._C_ops.multiply(scale_36, scale_39)
        del scale_36, scale_39

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_20 = paddle._C_ops.matmul(multiply_9, parameter_22, False, False)
        del multiply_9, parameter_22

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_51 = paddle._C_ops.add(matmul_20, parameter_21)
        del matmul_20, parameter_21

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_51, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_51

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_52 = paddle._C_ops.add(layer_norm_57, dropout_20)
        del dropout_20, layer_norm_57

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_20, parameter_19, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_52, parameter_19, parameter_20

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_10 = paddle._C_ops.fft_r2c(
            layer_norm_60, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_10 = paddle._C_ops.real(fft_r2c_10)
        del fft_r2c_10

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_53 = paddle._C_ops.add(layer_norm_60, real_10)
        del layer_norm_60, real_10

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_53, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_53, parameter_17, parameter_18

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_21 = paddle._C_ops.matmul(layer_norm_63, parameter_16, False, False)
        del parameter_16

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_54 = paddle._C_ops.add(matmul_21, parameter_15)
        del matmul_21, parameter_15

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_40 = paddle._C_ops.scale(add_54, full_1, float("0"), True)

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_10 = paddle._C_ops.pow(add_54, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_41 = paddle._C_ops.scale(pow_10, full_2, float("0"), True)
        del pow_10

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_55 = paddle._C_ops.add(add_54, scale_41)
        del add_54, scale_41

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_42 = paddle._C_ops.scale(add_55, full_3, float("0"), True)
        del add_55

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_11 = paddle._C_ops.tanh(scale_42)
        del scale_42

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_43 = paddle._C_ops.scale(tanh_11, full_4, float("1"), True)
        del tanh_11

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_10 = paddle._C_ops.multiply(scale_40, scale_43)
        del scale_40, scale_43

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_22 = paddle._C_ops.matmul(multiply_10, parameter_14, False, False)
        del multiply_10, parameter_14

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_56 = paddle._C_ops.add(matmul_22, parameter_13)
        del matmul_22, parameter_13

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_56, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_56

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_57 = paddle._C_ops.add(layer_norm_63, dropout_22)
        del dropout_22, layer_norm_63

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_57, parameter_12, parameter_11, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_57, parameter_11, parameter_12

        # pd_op.fft_r2c: (1x21x768xc64) <- (1x21x768xf32)
        fft_r2c_11 = paddle._C_ops.fft_r2c(
            layer_norm_66, [0, 1, 2], "backward", True, False
        )

        # pd_op.real: (1x21x768xf32) <- (1x21x768xc64)
        real_11 = paddle._C_ops.real(fft_r2c_11)
        del fft_r2c_11

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_58 = paddle._C_ops.add(layer_norm_66, real_11)
        del layer_norm_66, real_11

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_10, parameter_9, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_58, parameter_10, parameter_9

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_69, parameter_8, False, False)
        del parameter_8

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_59 = paddle._C_ops.add(matmul_23, parameter_7)
        del matmul_23, parameter_7

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_44 = paddle._C_ops.scale(add_59, full_1, float("0"), True)
        del full_1

        # pd_op.pow: (1x21x3072xf32) <- (1x21x3072xf32)
        pow_11 = paddle._C_ops.pow(add_59, float("3"))

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_45 = paddle._C_ops.scale(pow_11, full_2, float("0"), True)
        del full_2, pow_11

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        add_60 = paddle._C_ops.add(add_59, scale_45)
        del add_59, scale_45

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_46 = paddle._C_ops.scale(add_60, full_3, float("0"), True)
        del add_60, full_3

        # pd_op.tanh: (1x21x3072xf32) <- (1x21x3072xf32)
        tanh_12 = paddle._C_ops.tanh(scale_46)
        del scale_46

        # pd_op.scale: (1x21x3072xf32) <- (1x21x3072xf32, 1xf32)
        scale_47 = paddle._C_ops.scale(tanh_12, full_4, float("1"), True)
        del full_4, tanh_12

        # pd_op.multiply: (1x21x3072xf32) <- (1x21x3072xf32, 1x21x3072xf32)
        multiply_11 = paddle._C_ops.multiply(scale_44, scale_47)
        del scale_44, scale_47

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_24 = paddle._C_ops.matmul(multiply_11, parameter_6, False, False)
        del multiply_11, parameter_6

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_61 = paddle._C_ops.add(matmul_24, parameter_5)
        del matmul_24, parameter_5

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_61, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_61, full_0

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_62 = paddle._C_ops.add(layer_norm_69, dropout_24)
        del dropout_24, layer_norm_69

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_62, parameter_4, parameter_3, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_62, parameter_3, parameter_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (1x768xf32) <- (1x21x768xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            layer_norm_72, [1], full_int_array_0, full_int_array_2, [1], [1]
        )
        del full_int_array_0, full_int_array_2

        # pd_op.matmul: (1x768xf32) <- (1x768xf32, 768x768xf32)
        matmul_25 = paddle._C_ops.matmul(slice_1, parameter_2, False, False)
        del parameter_2, slice_1

        # pd_op.add: (1x768xf32) <- (1x768xf32, 768xf32)
        add_63 = paddle._C_ops.add(matmul_25, parameter_1)
        del matmul_25, parameter_1

        # pd_op.tanh: (1x768xf32) <- (1x768xf32)
        tanh_0 = paddle._C_ops.tanh(add_63)
        del add_63, layer_norm_72

        return tanh_0
