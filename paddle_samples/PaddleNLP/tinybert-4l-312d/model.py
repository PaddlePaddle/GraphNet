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

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x21xi64) <- (1x21xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_0, full_2, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x21xi64) <- (1x21xi64, 1xi32)
        cumsum_0 = paddle._C_ops.cumsum(full_like_0, full_3, False, False, False)
        del full_3

        # pd_op.subtract: (1x21xi64) <- (1x21xi64, 1x21xi64)
        subtract_0 = paddle._C_ops.subtract(cumsum_0, full_like_0)
        del cumsum_0, full_like_0

        # pd_op.scale: (1x21xi64) <- (1x21xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_0, full_2, float("0"), True)
        del full_2, subtract_0

        # pd_op.embedding: (1x21x312xf32) <- (1x21xi64, 30522x312xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_82, -1, False)
        del data_0, parameter_82

        # pd_op.embedding: (1x21x312xf32) <- (1x21xi64, 512x312xf32)
        embedding_1 = paddle._C_ops.embedding(scale_1, parameter_81, -1, False)
        del parameter_81, scale_1

        # pd_op.embedding: (1x21x312xf32) <- (1x21xi64, 2x312xf32)
        embedding_2 = paddle._C_ops.embedding(data_1, parameter_80, -1, False)
        del data_1, parameter_80

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)
        del embedding_0, embedding_1

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)
        del add_0, embedding_2

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_79, parameter_78, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_1, parameter_78, parameter_79

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_77, False, False)
        del parameter_77

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_76)
        del matmul_0, parameter_76

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 12, 26]

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_1)
        del add_2

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_74)
        del matmul_1, parameter_74

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_72)
        del matmul_2, parameter_72

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_3, full_int_array_1)
        del add_3

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_4, full_int_array_1)
        del add_4

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.196116"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x12x21x26xf32) <- (1x12x21x26xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(transpose_0, full_5, float("0"), True)
        del transpose_0

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x26xf32, 1x12x21x26xf32)
        matmul_3 = paddle._C_ops.matmul(scale_2, transpose_1, False, True)
        del scale_2, transpose_1

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_5 = paddle._C_ops.add(matmul_3, unsqueeze_0)
        del matmul_3

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x12x21x26xf32) <- (1x12x21x21xf32, 1x12x21x26xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x21x12x26xf32) <- (1x12x21x26xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, 0, 312]

        # pd_op.reshape: (1x21x312xf32) <- (1x21x12x26xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_2)
        del transpose_3

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_71, False, False)
        del parameter_71, reshape_3

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_70)
        del matmul_5, parameter_70

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_6, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_6

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_7 = paddle._C_ops.add(dropout_0, dropout_4)
        del dropout_0, dropout_4

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_65, parameter_64, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_7, parameter_64, parameter_65

        # pd_op.matmul: (1x21x1200xf32) <- (1x21x312xf32, 312x1200xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (1x21x1200xf32) <- (1x21x1200xf32, 1200xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_68)
        del matmul_6, parameter_68

        # pd_op.gelu: (1x21x1200xf32) <- (1x21x1200xf32)
        gelu_0 = paddle._C_ops.gelu(add_8, False)
        del add_8

        # pd_op.matmul: (1x21x312xf32) <- (1x21x1200xf32, 1200x312xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_67, False, False)
        del gelu_0, parameter_67

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_66)
        del matmul_7, parameter_66

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_9, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_9

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_10 = paddle._C_ops.add(layer_norm_3, dropout_6)
        del dropout_6, layer_norm_3

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_63, parameter_62, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_10, parameter_62, parameter_63

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_11 = paddle._C_ops.add(matmul_8, parameter_60)
        del matmul_8, parameter_60

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_11, full_int_array_1)
        del add_11

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_58)
        del matmul_9, parameter_58

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_56)
        del matmul_10, parameter_56

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_12, full_int_array_1)
        del add_12

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_13, full_int_array_1)
        del add_13

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.scale: (1x12x21x26xf32) <- (1x12x21x26xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(transpose_4, full_5, float("0"), True)
        del transpose_4

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x26xf32, 1x12x21x26xf32)
        matmul_11 = paddle._C_ops.matmul(scale_3, transpose_5, False, True)
        del scale_3, transpose_5

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_14 = paddle._C_ops.add(matmul_11, unsqueeze_0)
        del matmul_11

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_14, -1)
        del add_14

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x12x21x26xf32) <- (1x12x21x21xf32, 1x12x21x26xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)
        del dropout_8, transpose_6

        # pd_op.transpose: (1x21x12x26xf32) <- (1x12x21x26xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x21x312xf32) <- (1x21x12x26xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_2)
        del transpose_7

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_55, False, False)
        del parameter_55, reshape_7

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_15 = paddle._C_ops.add(matmul_13, parameter_54)
        del matmul_13, parameter_54

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_15, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_15

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_16 = paddle._C_ops.add(layer_norm_6, dropout_10)
        del dropout_10, layer_norm_6

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_49, parameter_48, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_16, parameter_48, parameter_49

        # pd_op.matmul: (1x21x1200xf32) <- (1x21x312xf32, 312x1200xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (1x21x1200xf32) <- (1x21x1200xf32, 1200xf32)
        add_17 = paddle._C_ops.add(matmul_14, parameter_52)
        del matmul_14, parameter_52

        # pd_op.gelu: (1x21x1200xf32) <- (1x21x1200xf32)
        gelu_1 = paddle._C_ops.gelu(add_17, False)
        del add_17

        # pd_op.matmul: (1x21x312xf32) <- (1x21x1200xf32, 1200x312xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_51, False, False)
        del gelu_1, parameter_51

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_18 = paddle._C_ops.add(matmul_15, parameter_50)
        del matmul_15, parameter_50

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_18, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_18

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_19 = paddle._C_ops.add(layer_norm_9, dropout_12)
        del dropout_12, layer_norm_9

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_47, parameter_46, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_19, parameter_46, parameter_47

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_12, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_20 = paddle._C_ops.add(matmul_16, parameter_44)
        del matmul_16, parameter_44

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_20, full_int_array_1)
        del add_20

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_21 = paddle._C_ops.add(matmul_17, parameter_42)
        del matmul_17, parameter_42

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_22 = paddle._C_ops.add(matmul_18, parameter_40)
        del matmul_18, parameter_40

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_21, full_int_array_1)
        del add_21

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_22, full_int_array_1)
        del add_22

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.scale: (1x12x21x26xf32) <- (1x12x21x26xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(transpose_8, full_5, float("0"), True)
        del transpose_8

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x26xf32, 1x12x21x26xf32)
        matmul_19 = paddle._C_ops.matmul(scale_4, transpose_9, False, True)
        del scale_4, transpose_9

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_23 = paddle._C_ops.add(matmul_19, unsqueeze_0)
        del matmul_19

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_23, -1)
        del add_23

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x12x21x26xf32) <- (1x12x21x21xf32, 1x12x21x26xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)
        del dropout_14, transpose_10

        # pd_op.transpose: (1x21x12x26xf32) <- (1x12x21x26xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x21x312xf32) <- (1x21x12x26xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_2)
        del transpose_11

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_39, False, False)
        del parameter_39, reshape_11

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_24 = paddle._C_ops.add(matmul_21, parameter_38)
        del matmul_21, parameter_38

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_24, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_24

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_25 = paddle._C_ops.add(layer_norm_12, dropout_16)
        del dropout_16, layer_norm_12

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_33, parameter_32, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_25, parameter_32, parameter_33

        # pd_op.matmul: (1x21x1200xf32) <- (1x21x312xf32, 312x1200xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_15, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (1x21x1200xf32) <- (1x21x1200xf32, 1200xf32)
        add_26 = paddle._C_ops.add(matmul_22, parameter_36)
        del matmul_22, parameter_36

        # pd_op.gelu: (1x21x1200xf32) <- (1x21x1200xf32)
        gelu_2 = paddle._C_ops.gelu(add_26, False)
        del add_26

        # pd_op.matmul: (1x21x312xf32) <- (1x21x1200xf32, 1200x312xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_2, parameter_35, False, False)
        del gelu_2, parameter_35

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_27 = paddle._C_ops.add(matmul_23, parameter_34)
        del matmul_23, parameter_34

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_27, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_27

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_28 = paddle._C_ops.add(layer_norm_15, dropout_18)
        del dropout_18, layer_norm_15

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_31, parameter_30, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_28, parameter_30, parameter_31

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_18, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_29 = paddle._C_ops.add(matmul_24, parameter_28)
        del matmul_24, parameter_28

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_29, full_int_array_1)
        del add_29

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_30 = paddle._C_ops.add(matmul_25, parameter_26)
        del matmul_25, parameter_26

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_31 = paddle._C_ops.add(matmul_26, parameter_24)
        del matmul_26, parameter_24

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_30, full_int_array_1)
        del add_30

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_31, full_int_array_1)
        del add_31, full_int_array_1

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.scale: (1x12x21x26xf32) <- (1x12x21x26xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(transpose_12, full_5, float("0"), True)
        del full_5, transpose_12

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x26xf32, 1x12x21x26xf32)
        matmul_27 = paddle._C_ops.matmul(scale_5, transpose_13, False, True)
        del scale_5, transpose_13

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_32 = paddle._C_ops.add(matmul_27, unsqueeze_0)
        del matmul_27, unsqueeze_0

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_32, -1)
        del add_32

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x12x21x26xf32) <- (1x12x21x21xf32, 1x12x21x26xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)
        del dropout_20, transpose_14

        # pd_op.transpose: (1x21x12x26xf32) <- (1x12x21x26xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x21x312xf32) <- (1x21x12x26xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_2)
        del full_int_array_2, transpose_15

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_15, parameter_23, False, False)
        del parameter_23, reshape_15

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_33 = paddle._C_ops.add(matmul_29, parameter_22)
        del matmul_29, parameter_22

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_33, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_33

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_34 = paddle._C_ops.add(layer_norm_18, dropout_22)
        del dropout_22, layer_norm_18

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_17, parameter_16, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_34, parameter_16, parameter_17

        # pd_op.matmul: (1x21x1200xf32) <- (1x21x312xf32, 312x1200xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_21, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (1x21x1200xf32) <- (1x21x1200xf32, 1200xf32)
        add_35 = paddle._C_ops.add(matmul_30, parameter_20)
        del matmul_30, parameter_20

        # pd_op.gelu: (1x21x1200xf32) <- (1x21x1200xf32)
        gelu_3 = paddle._C_ops.gelu(add_35, False)
        del add_35

        # pd_op.matmul: (1x21x312xf32) <- (1x21x1200xf32, 1200x312xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_3, parameter_19, False, False)
        del gelu_3, parameter_19

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_36 = paddle._C_ops.add(matmul_31, parameter_18)
        del matmul_31, parameter_18

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_36, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_36, full_4

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_37 = paddle._C_ops.add(layer_norm_21, dropout_24)
        del dropout_24, layer_norm_21

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_15, parameter_14, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_37, parameter_14, parameter_15

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1x312xf32) <- (1x21x312xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            layer_norm_24, [1], full_int_array_3, full_int_array_4, [1], [1]
        )
        del full_int_array_3, full_int_array_4

        # pd_op.matmul: (1x312xf32) <- (1x312xf32, 312x312xf32)
        matmul_32 = paddle._C_ops.matmul(slice_0, parameter_13, False, False)
        del parameter_13, slice_0

        # pd_op.add: (1x312xf32) <- (1x312xf32, 312xf32)
        add_38 = paddle._C_ops.add(matmul_32, parameter_12)
        del matmul_32, parameter_12

        # pd_op.tanh: (1x312xf32) <- (1x312xf32)
        tanh_0 = paddle._C_ops.tanh(add_38)
        del add_38, layer_norm_24

        return tanh_0
