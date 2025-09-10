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

        # pd_op.embedding: (1x21x312xf32) <- (1x21xi64, 40000x312xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_71, 0, False)
        del data_0, parameter_71

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

        # pd_op.embedding: (1x21x312xf32) <- (1x21xi64, 2048x312xf32)
        embedding_1 = paddle._C_ops.embedding(subtract_0, parameter_70, -1, False)
        del parameter_70

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)

        # pd_op.embedding: (1x21x312xf32) <- (1x21xi64, 4x312xf32)
        embedding_2 = paddle._C_ops.embedding(data_1, parameter_69, -1, False)
        del data_1, parameter_69

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21xi64) <- (1x21xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_2, full_4, float("0"), True)
        del full_2, full_4

        # pd_op.embedding: (1x21x312xf32) <- (1x21xi64, 16x312xf32)
        embedding_3 = paddle._C_ops.embedding(scale_1, parameter_68, -1, False)
        del parameter_68

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_2 = paddle._C_ops.add(add_1, embedding_3)

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_2, parameter_67, parameter_66, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_66, parameter_67

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_5

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_64)
        del parameter_64

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 12, 26]

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_3, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_4 = paddle._C_ops.add(matmul_1, parameter_62)
        del parameter_62

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_5 = paddle._C_ops.add(matmul_2, parameter_60)
        del parameter_60

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_4, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_5, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.196116"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_6

        # pd_op.scale: (1x12x21x26xf32) <- (1x12x21x26xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(transpose_0, full_6, float("0"), True)
        del transpose_0

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x26xf32, 1x12x21x26xf32)
        matmul_3 = paddle._C_ops.matmul(scale_2, transpose_1, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_6 = paddle._C_ops.add(matmul_3, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_6, -1)
        del add_6

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x26xf32) <- (1x12x21x21xf32, 1x12x21x26xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)

        # pd_op.transpose: (1x21x12x26xf32) <- (1x12x21x26xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, 0, 312]

        # pd_op.reshape: (1x21x312xf32) <- (1x21x12x26xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_2)

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_7 = paddle._C_ops.add(matmul_5, parameter_58)
        del parameter_58

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_7, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_7

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_8 = paddle._C_ops.add(dropout_0, dropout_4)

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_53, parameter_52, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_52, parameter_53

        # pd_op.matmul: (1x21x1248xf32) <- (1x21x312xf32, 312x1248xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x21x1248xf32) <- (1x21x1248xf32, 1248xf32)
        add_9 = paddle._C_ops.add(matmul_6, parameter_56)
        del parameter_56

        # pd_op.gelu: (1x21x1248xf32) <- (1x21x1248xf32)
        gelu_0 = paddle._C_ops.gelu(add_9, False)

        # pd_op.matmul: (1x21x312xf32) <- (1x21x1248xf32, 1248x312xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_10 = paddle._C_ops.add(matmul_7, parameter_54)
        del parameter_54

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_10, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_10

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_11 = paddle._C_ops.add(layer_norm_3, dropout_6)

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_51, parameter_50, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_12 = paddle._C_ops.add(matmul_8, parameter_48)
        del parameter_48

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_12, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_13 = paddle._C_ops.add(matmul_9, parameter_46)
        del parameter_46

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_14 = paddle._C_ops.add(matmul_10, parameter_44)
        del parameter_44

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_13, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_14, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.scale: (1x12x21x26xf32) <- (1x12x21x26xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(transpose_4, full_6, float("0"), True)
        del transpose_4

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x26xf32, 1x12x21x26xf32)
        matmul_11 = paddle._C_ops.matmul(scale_3, transpose_5, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_15 = paddle._C_ops.add(matmul_11, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_15, -1)
        del add_15

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x26xf32) <- (1x12x21x21xf32, 1x12x21x26xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)

        # pd_op.transpose: (1x21x12x26xf32) <- (1x12x21x26xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x21x312xf32) <- (1x21x12x26xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_2)

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_16 = paddle._C_ops.add(matmul_13, parameter_42)
        del parameter_42

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_16, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_16

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_17 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_37, parameter_36, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_36, parameter_37

        # pd_op.matmul: (1x21x1248xf32) <- (1x21x312xf32, 312x1248xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x21x1248xf32) <- (1x21x1248xf32, 1248xf32)
        add_18 = paddle._C_ops.add(matmul_14, parameter_40)
        del parameter_40

        # pd_op.gelu: (1x21x1248xf32) <- (1x21x1248xf32)
        gelu_1 = paddle._C_ops.gelu(add_18, False)

        # pd_op.matmul: (1x21x312xf32) <- (1x21x1248xf32, 1248x312xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_19 = paddle._C_ops.add(matmul_15, parameter_38)
        del parameter_38

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_19, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_19

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_20 = paddle._C_ops.add(layer_norm_9, dropout_12)

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_35, parameter_34, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_12, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_21 = paddle._C_ops.add(matmul_16, parameter_32)
        del parameter_32

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_21, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_22 = paddle._C_ops.add(matmul_17, parameter_30)
        del parameter_30

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_23 = paddle._C_ops.add(matmul_18, parameter_28)
        del parameter_28

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_22, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_23, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.scale: (1x12x21x26xf32) <- (1x12x21x26xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(transpose_8, full_6, float("0"), True)
        del transpose_8

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x26xf32, 1x12x21x26xf32)
        matmul_19 = paddle._C_ops.matmul(scale_4, transpose_9, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_24 = paddle._C_ops.add(matmul_19, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_24, -1)
        del add_24

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x26xf32) <- (1x12x21x21xf32, 1x12x21x26xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)

        # pd_op.transpose: (1x21x12x26xf32) <- (1x12x21x26xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x21x312xf32) <- (1x21x12x26xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_2)

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_25 = paddle._C_ops.add(matmul_21, parameter_26)
        del parameter_26

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_25, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_25

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_26 = paddle._C_ops.add(layer_norm_12, dropout_16)

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_26, parameter_21, parameter_20, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (1x21x1248xf32) <- (1x21x312xf32, 312x1248xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_15, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x21x1248xf32) <- (1x21x1248xf32, 1248xf32)
        add_27 = paddle._C_ops.add(matmul_22, parameter_24)
        del parameter_24

        # pd_op.gelu: (1x21x1248xf32) <- (1x21x1248xf32)
        gelu_2 = paddle._C_ops.gelu(add_27, False)

        # pd_op.matmul: (1x21x312xf32) <- (1x21x1248xf32, 1248x312xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_2, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_28 = paddle._C_ops.add(matmul_23, parameter_22)
        del parameter_22

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_28, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_28

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_29 = paddle._C_ops.add(layer_norm_15, dropout_18)

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_29, parameter_19, parameter_18, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_18, parameter_19

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_18, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_30 = paddle._C_ops.add(matmul_24, parameter_16)
        del parameter_16

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_30, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_31 = paddle._C_ops.add(matmul_25, parameter_14)
        del parameter_14

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_32 = paddle._C_ops.add(matmul_26, parameter_12)
        del parameter_12

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_31, full_int_array_1)

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x21x12x26xf32) <- (1x21x312xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_32, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (1x12x21x26xf32) <- (1x21x12x26xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.scale: (1x12x21x26xf32) <- (1x12x21x26xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(transpose_12, full_6, float("0"), True)
        del transpose_12

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x26xf32, 1x12x21x26xf32)
        matmul_27 = paddle._C_ops.matmul(scale_5, transpose_13, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_33 = paddle._C_ops.add(matmul_27, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_33, -1)
        del add_33

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x26xf32) <- (1x12x21x21xf32, 1x12x21x26xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)

        # pd_op.transpose: (1x21x12x26xf32) <- (1x12x21x26xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x21x312xf32) <- (1x21x12x26xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_2)
        del full_int_array_2

        # pd_op.matmul: (1x21x312xf32) <- (1x21x312xf32, 312x312xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_15, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_34 = paddle._C_ops.add(matmul_29, parameter_10)
        del parameter_10

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_34, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_34

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_35 = paddle._C_ops.add(layer_norm_18, dropout_22)

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_5, parameter_4, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.matmul: (1x21x1248xf32) <- (1x21x312xf32, 312x1248xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_21, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x21x1248xf32) <- (1x21x1248xf32, 1248xf32)
        add_36 = paddle._C_ops.add(matmul_30, parameter_8)
        del parameter_8

        # pd_op.gelu: (1x21x1248xf32) <- (1x21x1248xf32)
        gelu_3 = paddle._C_ops.gelu(add_36, False)

        # pd_op.matmul: (1x21x312xf32) <- (1x21x1248xf32, 1248x312xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_3, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 312xf32)
        add_37 = paddle._C_ops.add(matmul_31, parameter_6)
        del parameter_6

        # pd_op.dropout: (1x21x312xf32, 1x21x312xui8) <- (1x21x312xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_37, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_37

        # pd_op.add: (1x21x312xf32) <- (1x21x312xf32, 1x21x312xf32)
        add_38 = paddle._C_ops.add(layer_norm_21, dropout_24)

        # pd_op.layer_norm: (1x21x312xf32, 1x21xf32, 1x21xf32) <- (1x21x312xf32, 312xf32, 312xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_3, parameter_2, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1x312xf32) <- (1x21x312xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            layer_norm_24, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.matmul: (1x312xf32) <- (1x312xf32, 312x312xf32)
        matmul_32 = paddle._C_ops.matmul(slice_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (1x312xf32) <- (1x312xf32, 312xf32)
        add_39 = paddle._C_ops.add(matmul_32, parameter_0)
        del parameter_0

        # pd_op.tanh: (1x312xf32) <- (1x312xf32)
        tanh_0 = paddle._C_ops.tanh(add_39)
        del (
            add_0,
            add_1,
            add_11,
            add_12,
            add_13,
            add_14,
            add_17,
            add_18,
            add_2,
            add_20,
            add_21,
            add_22,
            add_23,
            add_26,
            add_27,
            add_29,
            add_3,
            add_30,
            add_31,
            add_32,
            add_35,
            add_36,
            add_38,
            add_39,
            add_4,
            add_5,
            add_8,
            add_9,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_2,
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
            dropout_3,
            dropout_4,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_8,
            dropout_9,
            embedding_0,
            embedding_1,
            embedding_2,
            embedding_3,
            full_5,
            full_6,
            full_int_array_3,
            full_int_array_4,
            gelu_0,
            gelu_1,
            gelu_2,
            gelu_3,
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
            layer_norm_3,
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
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_8,
            matmul_9,
            reshape_11,
            reshape_15,
            reshape_3,
            reshape_7,
            scale_1,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            slice_0,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            subtract_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_13,
            transpose_14,
            transpose_15,
            transpose_2,
            transpose_3,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_9,
            unsqueeze_0,
        )

        return tanh_0
