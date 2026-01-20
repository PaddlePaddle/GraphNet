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
        data_0,
        data_1,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (1x21xb) <- (1x21xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_0)
        del full_0

        # pd_op.cast: (1x21xf32) <- (1x21xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21xf32) <- (1x21xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_1, float("0"), True)
        del cast_0, full_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2]

        # pd_op.unsqueeze: (1x1x1x21xf32) <- (1x21xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(scale_0, full_int_array_0)
        del full_int_array_0, scale_0

        # pd_op.embedding: (1x21x384xf32) <- (1x21xi64, 40000x384xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_103, 0, False)
        del data_0, parameter_103

        # pd_op.full: (1x21xi64) <- ()
        full_2 = paddle._C_ops.full(
            [1, 21],
            float("1"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x21xi64) <- (1x21xi64, 1xi32)
        cumsum_0 = paddle._C_ops.cumsum(full_2, full_3, False, False, False)
        del full_3

        # pd_op.subtract: (1x21xi64) <- (1x21xi64, 1x21xi64)
        subtract_0 = paddle._C_ops.subtract(cumsum_0, full_2)
        del cumsum_0

        # pd_op.embedding: (1x21x384xf32) <- (1x21xi64, 2048x384xf32)
        embedding_1 = paddle._C_ops.embedding(subtract_0, parameter_102, -1, False)
        del parameter_102, subtract_0

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)
        del embedding_0, embedding_1

        # pd_op.embedding: (1x21x384xf32) <- (1x21xi64, 4x384xf32)
        embedding_2 = paddle._C_ops.embedding(data_1, parameter_101, -1, False)
        del data_1, parameter_101

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)
        del add_0, embedding_2

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21xi64) <- (1x21xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_2, full_4, float("0"), True)
        del full_2, full_4

        # pd_op.embedding: (1x21x384xf32) <- (1x21xi64, 16x384xf32)
        embedding_3 = paddle._C_ops.embedding(scale_1, parameter_100, -1, False)
        del parameter_100, scale_1

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_2 = paddle._C_ops.add(add_1, embedding_3)
        del add_1, embedding_3

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_2, parameter_99, parameter_98, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_2, parameter_98, parameter_99

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_96)
        del matmul_0, parameter_96

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 12, 32]

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_3, full_int_array_1)
        del add_3

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_4 = paddle._C_ops.add(matmul_1, parameter_94)
        del matmul_1, parameter_94

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_5 = paddle._C_ops.add(matmul_2, parameter_92)
        del matmul_2, parameter_92

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_4, full_int_array_1)
        del add_4

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_5, full_int_array_1)
        del add_5

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x12x21x32xf32) <- (1x12x21x32xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(transpose_0, full_6, float("0"), True)
        del transpose_0

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x32xf32, 1x12x21x32xf32)
        matmul_3 = paddle._C_ops.matmul(scale_2, transpose_1, False, True)
        del scale_2, transpose_1

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_6 = paddle._C_ops.add(matmul_3, unsqueeze_0)
        del matmul_3

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_6, -1)
        del add_6

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x12x21x32xf32) <- (1x12x21x21xf32, 1x12x21x32xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x21x12x32xf32) <- (1x12x21x32xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, 0, 384]

        # pd_op.reshape: (1x21x384xf32) <- (1x21x12x32xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_2)
        del transpose_3

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_91, False, False)
        del parameter_91, reshape_3

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_7 = paddle._C_ops.add(matmul_5, parameter_90)
        del matmul_5, parameter_90

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_7, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_7

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_8 = paddle._C_ops.add(dropout_0, dropout_4)
        del dropout_0, dropout_4

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_85, parameter_84, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_8, parameter_84, parameter_85

        # pd_op.matmul: (1x21x1536xf32) <- (1x21x384xf32, 384x1536xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (1x21x1536xf32) <- (1x21x1536xf32, 1536xf32)
        add_9 = paddle._C_ops.add(matmul_6, parameter_88)
        del matmul_6, parameter_88

        # pd_op.gelu: (1x21x1536xf32) <- (1x21x1536xf32)
        gelu_0 = paddle._C_ops.gelu(add_9, False)
        del add_9

        # pd_op.matmul: (1x21x384xf32) <- (1x21x1536xf32, 1536x384xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_87, False, False)
        del gelu_0, parameter_87

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_10 = paddle._C_ops.add(matmul_7, parameter_86)
        del matmul_7, parameter_86

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_10, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_10

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_11 = paddle._C_ops.add(layer_norm_3, dropout_6)
        del dropout_6, layer_norm_3

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_83, parameter_82, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_11, parameter_82, parameter_83

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_12 = paddle._C_ops.add(matmul_8, parameter_80)
        del matmul_8, parameter_80

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_12, full_int_array_1)
        del add_12

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_13 = paddle._C_ops.add(matmul_9, parameter_78)
        del matmul_9, parameter_78

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_77, False, False)
        del parameter_77

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_14 = paddle._C_ops.add(matmul_10, parameter_76)
        del matmul_10, parameter_76

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_13, full_int_array_1)
        del add_13

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_14, full_int_array_1)
        del add_14

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.scale: (1x12x21x32xf32) <- (1x12x21x32xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(transpose_4, full_6, float("0"), True)
        del transpose_4

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x32xf32, 1x12x21x32xf32)
        matmul_11 = paddle._C_ops.matmul(scale_3, transpose_5, False, True)
        del scale_3, transpose_5

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_15 = paddle._C_ops.add(matmul_11, unsqueeze_0)
        del matmul_11

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_15, -1)
        del add_15

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x12x21x32xf32) <- (1x12x21x21xf32, 1x12x21x32xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)
        del dropout_8, transpose_6

        # pd_op.transpose: (1x21x12x32xf32) <- (1x12x21x32xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x21x384xf32) <- (1x21x12x32xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_2)
        del transpose_7

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_75, False, False)
        del parameter_75, reshape_7

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_16 = paddle._C_ops.add(matmul_13, parameter_74)
        del matmul_13, parameter_74

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_16, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_16

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_17 = paddle._C_ops.add(layer_norm_6, dropout_10)
        del dropout_10, layer_norm_6

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_69, parameter_68, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_17, parameter_68, parameter_69

        # pd_op.matmul: (1x21x1536xf32) <- (1x21x384xf32, 384x1536xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x21x1536xf32) <- (1x21x1536xf32, 1536xf32)
        add_18 = paddle._C_ops.add(matmul_14, parameter_72)
        del matmul_14, parameter_72

        # pd_op.gelu: (1x21x1536xf32) <- (1x21x1536xf32)
        gelu_1 = paddle._C_ops.gelu(add_18, False)
        del add_18

        # pd_op.matmul: (1x21x384xf32) <- (1x21x1536xf32, 1536x384xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_71, False, False)
        del gelu_1, parameter_71

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_19 = paddle._C_ops.add(matmul_15, parameter_70)
        del matmul_15, parameter_70

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_19, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_19

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_20 = paddle._C_ops.add(layer_norm_9, dropout_12)
        del dropout_12, layer_norm_9

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_67, parameter_66, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_20, parameter_66, parameter_67

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_12, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_21 = paddle._C_ops.add(matmul_16, parameter_64)
        del matmul_16, parameter_64

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_21, full_int_array_1)
        del add_21

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_22 = paddle._C_ops.add(matmul_17, parameter_62)
        del matmul_17, parameter_62

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_23 = paddle._C_ops.add(matmul_18, parameter_60)
        del matmul_18, parameter_60

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_22, full_int_array_1)
        del add_22

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_23, full_int_array_1)
        del add_23

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.scale: (1x12x21x32xf32) <- (1x12x21x32xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(transpose_8, full_6, float("0"), True)
        del transpose_8

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x32xf32, 1x12x21x32xf32)
        matmul_19 = paddle._C_ops.matmul(scale_4, transpose_9, False, True)
        del scale_4, transpose_9

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_24 = paddle._C_ops.add(matmul_19, unsqueeze_0)
        del matmul_19

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_24, -1)
        del add_24

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x12x21x32xf32) <- (1x12x21x21xf32, 1x12x21x32xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)
        del dropout_14, transpose_10

        # pd_op.transpose: (1x21x12x32xf32) <- (1x12x21x32xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x21x384xf32) <- (1x21x12x32xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_2)
        del transpose_11

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_59, False, False)
        del parameter_59, reshape_11

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_25 = paddle._C_ops.add(matmul_21, parameter_58)
        del matmul_21, parameter_58

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_25, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_25

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_26 = paddle._C_ops.add(layer_norm_12, dropout_16)
        del dropout_16, layer_norm_12

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_26, parameter_53, parameter_52, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_26, parameter_52, parameter_53

        # pd_op.matmul: (1x21x1536xf32) <- (1x21x384xf32, 384x1536xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_15, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x21x1536xf32) <- (1x21x1536xf32, 1536xf32)
        add_27 = paddle._C_ops.add(matmul_22, parameter_56)
        del matmul_22, parameter_56

        # pd_op.gelu: (1x21x1536xf32) <- (1x21x1536xf32)
        gelu_2 = paddle._C_ops.gelu(add_27, False)
        del add_27

        # pd_op.matmul: (1x21x384xf32) <- (1x21x1536xf32, 1536x384xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_2, parameter_55, False, False)
        del gelu_2, parameter_55

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_28 = paddle._C_ops.add(matmul_23, parameter_54)
        del matmul_23, parameter_54

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_28, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_28

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_29 = paddle._C_ops.add(layer_norm_15, dropout_18)
        del dropout_18, layer_norm_15

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_29, parameter_51, parameter_50, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_29, parameter_50, parameter_51

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_18, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_30 = paddle._C_ops.add(matmul_24, parameter_48)
        del matmul_24, parameter_48

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_30, full_int_array_1)
        del add_30

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_31 = paddle._C_ops.add(matmul_25, parameter_46)
        del matmul_25, parameter_46

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_32 = paddle._C_ops.add(matmul_26, parameter_44)
        del matmul_26, parameter_44

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_31, full_int_array_1)
        del add_31

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_32, full_int_array_1)
        del add_32

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.scale: (1x12x21x32xf32) <- (1x12x21x32xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(transpose_12, full_6, float("0"), True)
        del transpose_12

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x32xf32, 1x12x21x32xf32)
        matmul_27 = paddle._C_ops.matmul(scale_5, transpose_13, False, True)
        del scale_5, transpose_13

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_33 = paddle._C_ops.add(matmul_27, unsqueeze_0)
        del matmul_27

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_33, -1)
        del add_33

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x12x21x32xf32) <- (1x12x21x21xf32, 1x12x21x32xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)
        del dropout_20, transpose_14

        # pd_op.transpose: (1x21x12x32xf32) <- (1x12x21x32xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x21x384xf32) <- (1x21x12x32xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_2)
        del transpose_15

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_15, parameter_43, False, False)
        del parameter_43, reshape_15

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_34 = paddle._C_ops.add(matmul_29, parameter_42)
        del matmul_29, parameter_42

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_34, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_34

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_35 = paddle._C_ops.add(layer_norm_18, dropout_22)
        del dropout_22, layer_norm_18

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_37, parameter_36, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_35, parameter_36, parameter_37

        # pd_op.matmul: (1x21x1536xf32) <- (1x21x384xf32, 384x1536xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_21, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x21x1536xf32) <- (1x21x1536xf32, 1536xf32)
        add_36 = paddle._C_ops.add(matmul_30, parameter_40)
        del matmul_30, parameter_40

        # pd_op.gelu: (1x21x1536xf32) <- (1x21x1536xf32)
        gelu_3 = paddle._C_ops.gelu(add_36, False)
        del add_36

        # pd_op.matmul: (1x21x384xf32) <- (1x21x1536xf32, 1536x384xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_3, parameter_39, False, False)
        del gelu_3, parameter_39

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_37 = paddle._C_ops.add(matmul_31, parameter_38)
        del matmul_31, parameter_38

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_37, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_37

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_38 = paddle._C_ops.add(layer_norm_21, dropout_24)
        del dropout_24, layer_norm_21

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_35, parameter_34, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_38, parameter_34, parameter_35

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_24, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_39 = paddle._C_ops.add(matmul_32, parameter_32)
        del matmul_32, parameter_32

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_39, full_int_array_1)
        del add_39

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_40 = paddle._C_ops.add(matmul_33, parameter_30)
        del matmul_33, parameter_30

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_41 = paddle._C_ops.add(matmul_34, parameter_28)
        del matmul_34, parameter_28

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_40, full_int_array_1)
        del add_40

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_41, full_int_array_1)
        del add_41

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.scale: (1x12x21x32xf32) <- (1x12x21x32xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(transpose_16, full_6, float("0"), True)
        del transpose_16

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x32xf32, 1x12x21x32xf32)
        matmul_35 = paddle._C_ops.matmul(scale_6, transpose_17, False, True)
        del scale_6, transpose_17

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_42 = paddle._C_ops.add(matmul_35, unsqueeze_0)
        del matmul_35

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(add_42, -1)
        del add_42

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_4

        # pd_op.matmul: (1x12x21x32xf32) <- (1x12x21x21xf32, 1x12x21x32xf32)
        matmul_36 = paddle._C_ops.matmul(dropout_26, transpose_18, False, False)
        del dropout_26, transpose_18

        # pd_op.transpose: (1x21x12x32xf32) <- (1x12x21x32xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (1x21x384xf32) <- (1x21x12x32xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_2)
        del transpose_19

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_19, parameter_27, False, False)
        del parameter_27, reshape_19

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_43 = paddle._C_ops.add(matmul_37, parameter_26)
        del matmul_37, parameter_26

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_43, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_43

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_44 = paddle._C_ops.add(layer_norm_24, dropout_28)
        del dropout_28, layer_norm_24

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_44, parameter_21, parameter_20, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_44, parameter_20, parameter_21

        # pd_op.matmul: (1x21x1536xf32) <- (1x21x384xf32, 384x1536xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_27, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x21x1536xf32) <- (1x21x1536xf32, 1536xf32)
        add_45 = paddle._C_ops.add(matmul_38, parameter_24)
        del matmul_38, parameter_24

        # pd_op.gelu: (1x21x1536xf32) <- (1x21x1536xf32)
        gelu_4 = paddle._C_ops.gelu(add_45, False)
        del add_45

        # pd_op.matmul: (1x21x384xf32) <- (1x21x1536xf32, 1536x384xf32)
        matmul_39 = paddle._C_ops.matmul(gelu_4, parameter_23, False, False)
        del gelu_4, parameter_23

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_46 = paddle._C_ops.add(matmul_39, parameter_22)
        del matmul_39, parameter_22

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_46, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_46

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_47 = paddle._C_ops.add(layer_norm_27, dropout_30)
        del dropout_30, layer_norm_27

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_47, parameter_19, parameter_18, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_47, parameter_18, parameter_19

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_30, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_48 = paddle._C_ops.add(matmul_40, parameter_16)
        del matmul_40, parameter_16

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_48, full_int_array_1)
        del add_48

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_49 = paddle._C_ops.add(matmul_41, parameter_14)
        del matmul_41, parameter_14

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_50 = paddle._C_ops.add(matmul_42, parameter_12)
        del matmul_42, parameter_12

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_49, full_int_array_1)
        del add_49

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x21x12x32xf32) <- (1x21x384xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_50, full_int_array_1)
        del add_50, full_int_array_1

        # pd_op.transpose: (1x12x21x32xf32) <- (1x21x12x32xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.scale: (1x12x21x32xf32) <- (1x12x21x32xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(transpose_20, full_6, float("0"), True)
        del full_6, transpose_20

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x32xf32, 1x12x21x32xf32)
        matmul_43 = paddle._C_ops.matmul(scale_7, transpose_21, False, True)
        del scale_7, transpose_21

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_51 = paddle._C_ops.add(matmul_43, unsqueeze_0)
        del matmul_43, unsqueeze_0

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_51, -1)
        del add_51

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_5

        # pd_op.matmul: (1x12x21x32xf32) <- (1x12x21x21xf32, 1x12x21x32xf32)
        matmul_44 = paddle._C_ops.matmul(dropout_32, transpose_22, False, False)
        del dropout_32, transpose_22

        # pd_op.transpose: (1x21x12x32xf32) <- (1x12x21x32xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (1x21x384xf32) <- (1x21x12x32xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_2)
        del full_int_array_2, transpose_23

        # pd_op.matmul: (1x21x384xf32) <- (1x21x384xf32, 384x384xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_23, parameter_11, False, False)
        del parameter_11, reshape_23

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_52 = paddle._C_ops.add(matmul_45, parameter_10)
        del matmul_45, parameter_10

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_52, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_52

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_53 = paddle._C_ops.add(layer_norm_30, dropout_34)
        del dropout_34, layer_norm_30

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_53, parameter_5, parameter_4, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_53, parameter_4, parameter_5

        # pd_op.matmul: (1x21x1536xf32) <- (1x21x384xf32, 384x1536xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_33, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x21x1536xf32) <- (1x21x1536xf32, 1536xf32)
        add_54 = paddle._C_ops.add(matmul_46, parameter_8)
        del matmul_46, parameter_8

        # pd_op.gelu: (1x21x1536xf32) <- (1x21x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add_54, False)
        del add_54

        # pd_op.matmul: (1x21x384xf32) <- (1x21x1536xf32, 1536x384xf32)
        matmul_47 = paddle._C_ops.matmul(gelu_5, parameter_7, False, False)
        del gelu_5, parameter_7

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_55 = paddle._C_ops.add(matmul_47, parameter_6)
        del matmul_47, parameter_6

        # pd_op.dropout: (1x21x384xf32, 1x21x384xui8) <- (1x21x384xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_55, None, full_5, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_55, full_5

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        add_56 = paddle._C_ops.add(layer_norm_33, dropout_36)
        del dropout_36, layer_norm_33

        # pd_op.layer_norm: (1x21x384xf32, 1x21xf32, 1x21xf32) <- (1x21x384xf32, 384xf32, 384xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_56, parameter_3, parameter_2, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_56, parameter_2, parameter_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1x384xf32) <- (1x21x384xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            layer_norm_36, [1], full_int_array_3, full_int_array_4, [1], [1]
        )
        del full_int_array_3, full_int_array_4, layer_norm_36

        # pd_op.matmul: (1x384xf32) <- (1x384xf32, 384x384xf32)
        matmul_48 = paddle._C_ops.matmul(slice_0, parameter_1, False, False)
        del parameter_1, slice_0

        # pd_op.add: (1x384xf32) <- (1x384xf32, 384xf32)
        add_57 = paddle._C_ops.add(matmul_48, parameter_0)
        del matmul_48, parameter_0

        # pd_op.tanh: (1x384xf32) <- (1x384xf32)
        tanh_0 = paddle._C_ops.tanh(add_57)
        del add_57

        return tanh_0
