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

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 39981x768xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_102, 0, False)
        del data_0, parameter_102

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
        del cumsum_0, full_2

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 2048x768xf32)
        embedding_1 = paddle._C_ops.embedding(subtract_0, parameter_101, -1, False)
        del parameter_101

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 4x768xf32)
        embedding_2 = paddle._C_ops.embedding(data_1, parameter_100, -1, False)
        del data_1, parameter_100

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_99, parameter_98, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_98, parameter_99

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_4

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_96)
        del parameter_96

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 12, 64]

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_94)
        del parameter_94

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_92)
        del parameter_92

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_3, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_4, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_19 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_20 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_21 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_22 = full_5

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(transpose_0, full_5, float("0"), True)
        del transpose_0

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_3 = paddle._C_ops.matmul(scale_1, transpose_1, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_5 = paddle._C_ops.add(matmul_3, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, 0, 768]

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_90)
        del parameter_90

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_6, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_6

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_7 = paddle._C_ops.add(dropout_0, dropout_4)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_85, parameter_84, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_84, parameter_85

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_88)
        del parameter_88

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_8, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_86)
        del parameter_86

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_9, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_9

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_10 = paddle._C_ops.add(layer_norm_3, dropout_6)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_83, parameter_82, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_82, parameter_83

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_11 = paddle._C_ops.add(matmul_8, parameter_80)
        del parameter_80

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_11, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_78)
        del parameter_78

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_77, False, False)
        del parameter_77

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_76)
        del parameter_76

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_12, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_13, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(transpose_4, full_5, float("0"), True)
        del transpose_4

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_11 = paddle._C_ops.matmul(scale_2, transpose_5, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_14 = paddle._C_ops.add(matmul_11, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_14, -1)
        del add_14

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_15 = paddle._C_ops.add(matmul_13, parameter_74)
        del parameter_74

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_15, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_15

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_16 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_69, parameter_68, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_68, parameter_69

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_17 = paddle._C_ops.add(matmul_14, parameter_72)
        del parameter_72

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_17, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(matmul_15, parameter_70)
        del parameter_70

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_18, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_18

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_19 = paddle._C_ops.add(layer_norm_9, dropout_12)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_67, parameter_66, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_66, parameter_67

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_12, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_20 = paddle._C_ops.add(matmul_16, parameter_64)
        del parameter_64

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_20, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_17, parameter_62)
        del parameter_62

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_22 = paddle._C_ops.add(matmul_18, parameter_60)
        del parameter_60

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_21, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_22, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(transpose_8, full_5, float("0"), True)
        del transpose_8

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_19 = paddle._C_ops.matmul(scale_3, transpose_9, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_23 = paddle._C_ops.add(matmul_19, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_23, -1)
        del add_23

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_24 = paddle._C_ops.add(matmul_21, parameter_58)
        del parameter_58

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_24, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_24

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_25 = paddle._C_ops.add(layer_norm_12, dropout_16)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_53, parameter_52, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_52, parameter_53

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_15, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_26 = paddle._C_ops.add(matmul_22, parameter_56)
        del parameter_56

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_26, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_2, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_27 = paddle._C_ops.add(matmul_23, parameter_54)
        del parameter_54

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_27, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_27

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_28 = paddle._C_ops.add(layer_norm_15, dropout_18)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_51, parameter_50, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_18, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_24, parameter_48)
        del parameter_48

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_29, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_30 = paddle._C_ops.add(matmul_25, parameter_46)
        del parameter_46

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_31 = paddle._C_ops.add(matmul_26, parameter_44)
        del parameter_44

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_30, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_31, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(transpose_12, full_5, float("0"), True)
        del transpose_12

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_27 = paddle._C_ops.matmul(scale_4, transpose_13, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_32 = paddle._C_ops.add(matmul_27, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_32, -1)
        del add_32

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_15, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_33 = paddle._C_ops.add(matmul_29, parameter_42)
        del parameter_42

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_33, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_33

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_34 = paddle._C_ops.add(layer_norm_18, dropout_22)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_37, parameter_36, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_36, parameter_37

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_21, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_35 = paddle._C_ops.add(matmul_30, parameter_40)
        del parameter_40

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_35, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_3, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_36 = paddle._C_ops.add(matmul_31, parameter_38)
        del parameter_38

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_36, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_36

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_37 = paddle._C_ops.add(layer_norm_21, dropout_24)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_35, parameter_34, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_24, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_38 = paddle._C_ops.add(matmul_32, parameter_32)
        del parameter_32

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_38, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_39 = paddle._C_ops.add(matmul_33, parameter_30)
        del parameter_30

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_40 = paddle._C_ops.add(matmul_34, parameter_28)
        del parameter_28

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_39, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_40, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(transpose_16, full_5, float("0"), True)
        del transpose_16

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_35 = paddle._C_ops.matmul(scale_5, transpose_17, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_41 = paddle._C_ops.add(matmul_35, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(add_41, -1)
        del add_41

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_36 = paddle._C_ops.matmul(dropout_26, transpose_18, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_19, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_42 = paddle._C_ops.add(matmul_37, parameter_26)
        del parameter_26

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_42, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_42

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_43 = paddle._C_ops.add(layer_norm_24, dropout_28)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_21, parameter_20, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_27, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_44 = paddle._C_ops.add(matmul_38, parameter_24)
        del parameter_24

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_44, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_39 = paddle._C_ops.matmul(gelu_4, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_45 = paddle._C_ops.add(matmul_39, parameter_22)
        del parameter_22

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_45, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_45

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_46 = paddle._C_ops.add(layer_norm_27, dropout_30)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_19, parameter_18, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_18, parameter_19

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_30, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_47 = paddle._C_ops.add(matmul_40, parameter_16)
        del parameter_16

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_47, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_48 = paddle._C_ops.add(matmul_41, parameter_14)
        del parameter_14

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_49 = paddle._C_ops.add(matmul_42, parameter_12)
        del parameter_12

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_48, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_49, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(transpose_20, full_5, float("0"), True)
        del transpose_20

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_43 = paddle._C_ops.matmul(scale_6, transpose_21, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_50 = paddle._C_ops.add(matmul_43, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_50, -1)
        del add_50

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_44 = paddle._C_ops.matmul(dropout_32, transpose_22, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_2)
        del full_int_array_2

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_23, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_51 = paddle._C_ops.add(matmul_45, parameter_10)
        del parameter_10

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_51, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_51

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_52 = paddle._C_ops.add(layer_norm_30, dropout_34)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_5, parameter_4, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_33, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_53 = paddle._C_ops.add(matmul_46, parameter_8)
        del parameter_8

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_53, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_47 = paddle._C_ops.matmul(gelu_5, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_54 = paddle._C_ops.add(matmul_47, parameter_6)
        del parameter_6

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_54, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_54

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_55 = paddle._C_ops.add(layer_norm_33, dropout_36)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_3, parameter_2, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1x768xf32) <- (1x21x768xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            layer_norm_36, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.matmul: (1x768xf32) <- (1x768xf32, 768x768xf32)
        matmul_48 = paddle._C_ops.matmul(slice_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (1x768xf32) <- (1x768xf32, 768xf32)
        add_56 = paddle._C_ops.add(matmul_48, parameter_0)
        del parameter_0

        # pd_op.tanh: (1x768xf32) <- (1x768xf32)
        tanh_0 = paddle._C_ops.tanh(add_56)
        del (
            add_0,
            add_1,
            add_10,
            add_11,
            add_12,
            add_13,
            add_16,
            add_17,
            add_19,
            add_2,
            add_20,
            add_21,
            add_22,
            add_25,
            add_26,
            add_28,
            add_29,
            add_3,
            add_30,
            add_31,
            add_34,
            add_35,
            add_37,
            add_38,
            add_39,
            add_4,
            add_40,
            add_43,
            add_44,
            add_46,
            add_47,
            add_48,
            add_49,
            add_52,
            add_53,
            add_55,
            add_56,
            add_7,
            add_8,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            dropout_0,
            dropout_1,
            dropout_10,
            dropout_11,
            dropout_12,
            dropout_13,
            dropout_14,
            dropout_15,
            dropout_16,
            dropout_17,
            dropout_18,
            dropout_19,
            dropout_2,
            dropout_20,
            dropout_21,
            dropout_22,
            dropout_23,
            dropout_24,
            dropout_25,
            dropout_26,
            dropout_27,
            dropout_28,
            dropout_29,
            dropout_3,
            dropout_30,
            dropout_31,
            dropout_32,
            dropout_33,
            dropout_34,
            dropout_35,
            dropout_36,
            dropout_37,
            dropout_4,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_8,
            dropout_9,
            embedding_0,
            embedding_1,
            embedding_2,
            full_4,
            full_5,
            full_int_array_3,
            full_int_array_4,
            gelu_0,
            gelu_1,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            layer_norm_1,
            layer_norm_10,
            layer_norm_11,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            layer_norm_15,
            layer_norm_16,
            layer_norm_17,
            layer_norm_18,
            layer_norm_19,
            layer_norm_2,
            layer_norm_20,
            layer_norm_21,
            layer_norm_22,
            layer_norm_23,
            layer_norm_24,
            layer_norm_25,
            layer_norm_26,
            layer_norm_27,
            layer_norm_28,
            layer_norm_29,
            layer_norm_3,
            layer_norm_30,
            layer_norm_31,
            layer_norm_32,
            layer_norm_33,
            layer_norm_34,
            layer_norm_35,
            layer_norm_36,
            layer_norm_37,
            layer_norm_38,
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_0,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_19,
            matmul_2,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_29,
            matmul_3,
            matmul_30,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_37,
            matmul_38,
            matmul_39,
            matmul_40,
            matmul_41,
            matmul_42,
            matmul_43,
            matmul_45,
            matmul_46,
            matmul_47,
            matmul_48,
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_8,
            matmul_9,
            reshape_11,
            reshape_15,
            reshape_19,
            reshape_23,
            reshape_3,
            reshape_7,
            scale_1,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            scale_6,
            slice_0,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            softmax_4,
            softmax_5,
            subtract_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_13,
            transpose_14,
            transpose_15,
            transpose_17,
            transpose_18,
            transpose_19,
            transpose_2,
            transpose_21,
            transpose_22,
            transpose_23,
            transpose_3,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_9,
            unsqueeze_0,
        )

        return tanh_0
