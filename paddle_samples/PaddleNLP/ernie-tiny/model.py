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
        data_0,
        data_1,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (1x20xb) <- (1x20xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_0)
        del full_0

        # pd_op.cast: (1x20xf32) <- (1x20xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x20xf32) <- (1x20xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_1, float("0"), True)
        del cast_0, full_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2]

        # pd_op.unsqueeze: (1x1x1x20xf32) <- (1x20xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(scale_0, full_int_array_0)
        del full_int_array_0, scale_0

        # pd_op.embedding: (1x20x1024xf32) <- (1x20xi64, 50006x1024xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_54, 0, False)
        del data_0, parameter_54

        # pd_op.full: (1x20xi64) <- ()
        full_2 = paddle._C_ops.full(
            [1, 20],
            float("1"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x20xi64) <- (1x20xi64, 1xi32)
        cumsum_0 = paddle._C_ops.cumsum(full_2, full_3, False, False, False)
        del full_3

        # pd_op.subtract: (1x20xi64) <- (1x20xi64, 1x20xi64)
        subtract_0 = paddle._C_ops.subtract(cumsum_0, full_2)
        del cumsum_0, full_2

        # pd_op.embedding: (1x20x1024xf32) <- (1x20xi64, 600x1024xf32)
        embedding_1 = paddle._C_ops.embedding(subtract_0, parameter_53, -1, False)
        del parameter_53, subtract_0

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1x20x1024xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)
        del embedding_0, embedding_1

        # pd_op.embedding: (1x20x1024xf32) <- (1x20xi64, 2x1024xf32)
        embedding_2 = paddle._C_ops.embedding(data_1, parameter_52, -1, False)
        del data_1, parameter_52

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1x20x1024xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)
        del add_0, embedding_2

        # pd_op.layer_norm: (1x20x1024xf32, 1x20xf32, 1x20xf32) <- (1x20x1024xf32, 1024xf32, 1024xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_51, parameter_50, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_1, parameter_50, parameter_51

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x20x1024xf32, 1x20x1024xui8) <- (1x20x1024xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_48)
        del matmul_0, parameter_48

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 16, 64]

        # pd_op.reshape: (1x20x16x64xf32) <- (1x20x1024xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_1)
        del add_2

        # pd_op.transpose: (1x16x20x64xf32) <- (1x20x16x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_46)
        del matmul_1, parameter_46

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_44)
        del matmul_2, parameter_44

        # pd_op.reshape: (1x20x16x64xf32) <- (1x20x1024xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_3, full_int_array_1)
        del add_3

        # pd_op.transpose: (1x16x20x64xf32) <- (1x20x16x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x20x16x64xf32) <- (1x20x1024xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_4, full_int_array_1)
        del add_4

        # pd_op.transpose: (1x16x20x64xf32) <- (1x20x16x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x16x20x64xf32) <- (1x16x20x64xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(transpose_0, full_5, float("0"), True)
        del transpose_0

        # pd_op.matmul: (1x16x20x20xf32) <- (1x16x20x64xf32, 1x16x20x64xf32)
        matmul_3 = paddle._C_ops.matmul(scale_1, transpose_1, False, True)
        del scale_1, transpose_1

        # pd_op.add: (1x16x20x20xf32) <- (1x16x20x20xf32, 1x1x1x20xf32)
        add_5 = paddle._C_ops.add(matmul_3, unsqueeze_0)
        del matmul_3

        # pd_op.softmax: (1x16x20x20xf32) <- (1x16x20x20xf32)
        softmax_0 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.dropout: (1x16x20x20xf32, 1x16x20x20xui8) <- (1x16x20x20xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x16x20x64xf32) <- (1x16x20x20xf32, 1x16x20x64xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x20x16x64xf32) <- (1x16x20x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, 0, 1024]

        # pd_op.reshape: (1x20x1024xf32) <- (1x20x16x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_2)
        del transpose_3

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_43, False, False)
        del parameter_43, reshape_3

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_42)
        del matmul_5, parameter_42

        # pd_op.dropout: (1x20x1024xf32, 1x20x1024xui8) <- (1x20x1024xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_6, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_6

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1x20x1024xf32)
        add_7 = paddle._C_ops.add(dropout_0, dropout_4)
        del dropout_0, dropout_4

        # pd_op.layer_norm: (1x20x1024xf32, 1x20xf32, 1x20xf32) <- (1x20x1024xf32, 1024xf32, 1024xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_37, parameter_36, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_7, parameter_36, parameter_37

        # pd_op.matmul: (1x20x4096xf32) <- (1x20x1024xf32, 1024x4096xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x20x4096xf32) <- (1x20x4096xf32, 4096xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_40)
        del matmul_6, parameter_40

        # pd_op.relu: (1x20x4096xf32) <- (1x20x4096xf32)
        relu_0 = paddle._C_ops.relu(add_8)
        del add_8

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x4096xf32, 4096x1024xf32)
        matmul_7 = paddle._C_ops.matmul(relu_0, parameter_39, False, False)
        del parameter_39, relu_0

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_38)
        del matmul_7, parameter_38

        # pd_op.dropout: (1x20x1024xf32, 1x20x1024xui8) <- (1x20x1024xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_9, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_9

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1x20x1024xf32)
        add_10 = paddle._C_ops.add(layer_norm_3, dropout_6)
        del dropout_6, layer_norm_3

        # pd_op.layer_norm: (1x20x1024xf32, 1x20xf32, 1x20xf32) <- (1x20x1024xf32, 1024xf32, 1024xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_35, parameter_34, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_10, parameter_34, parameter_35

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_11 = paddle._C_ops.add(matmul_8, parameter_32)
        del matmul_8, parameter_32

        # pd_op.reshape: (1x20x16x64xf32) <- (1x20x1024xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_11, full_int_array_1)
        del add_11

        # pd_op.transpose: (1x16x20x64xf32) <- (1x20x16x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_30)
        del matmul_9, parameter_30

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_28)
        del matmul_10, parameter_28

        # pd_op.reshape: (1x20x16x64xf32) <- (1x20x1024xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_12, full_int_array_1)
        del add_12

        # pd_op.transpose: (1x16x20x64xf32) <- (1x20x16x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x20x16x64xf32) <- (1x20x1024xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_13, full_int_array_1)
        del add_13

        # pd_op.transpose: (1x16x20x64xf32) <- (1x20x16x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.scale: (1x16x20x64xf32) <- (1x16x20x64xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(transpose_4, full_5, float("0"), True)
        del transpose_4

        # pd_op.matmul: (1x16x20x20xf32) <- (1x16x20x64xf32, 1x16x20x64xf32)
        matmul_11 = paddle._C_ops.matmul(scale_2, transpose_5, False, True)
        del scale_2, transpose_5

        # pd_op.add: (1x16x20x20xf32) <- (1x16x20x20xf32, 1x1x1x20xf32)
        add_14 = paddle._C_ops.add(matmul_11, unsqueeze_0)
        del matmul_11

        # pd_op.softmax: (1x16x20x20xf32) <- (1x16x20x20xf32)
        softmax_1 = paddle._C_ops.softmax(add_14, -1)
        del add_14

        # pd_op.dropout: (1x16x20x20xf32, 1x16x20x20xui8) <- (1x16x20x20xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x16x20x64xf32) <- (1x16x20x20xf32, 1x16x20x64xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)
        del dropout_8, transpose_6

        # pd_op.transpose: (1x20x16x64xf32) <- (1x16x20x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x20x1024xf32) <- (1x20x16x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_2)
        del transpose_7

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_27, False, False)
        del parameter_27, reshape_7

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_15 = paddle._C_ops.add(matmul_13, parameter_26)
        del matmul_13, parameter_26

        # pd_op.dropout: (1x20x1024xf32, 1x20x1024xui8) <- (1x20x1024xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_15, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_15

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1x20x1024xf32)
        add_16 = paddle._C_ops.add(layer_norm_6, dropout_10)
        del dropout_10, layer_norm_6

        # pd_op.layer_norm: (1x20x1024xf32, 1x20xf32, 1x20xf32) <- (1x20x1024xf32, 1024xf32, 1024xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_21, parameter_20, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_16, parameter_20, parameter_21

        # pd_op.matmul: (1x20x4096xf32) <- (1x20x1024xf32, 1024x4096xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x20x4096xf32) <- (1x20x4096xf32, 4096xf32)
        add_17 = paddle._C_ops.add(matmul_14, parameter_24)
        del matmul_14, parameter_24

        # pd_op.relu: (1x20x4096xf32) <- (1x20x4096xf32)
        relu_1 = paddle._C_ops.relu(add_17)
        del add_17

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x4096xf32, 4096x1024xf32)
        matmul_15 = paddle._C_ops.matmul(relu_1, parameter_23, False, False)
        del parameter_23, relu_1

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_18 = paddle._C_ops.add(matmul_15, parameter_22)
        del matmul_15, parameter_22

        # pd_op.dropout: (1x20x1024xf32, 1x20x1024xui8) <- (1x20x1024xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_18, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_18

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1x20x1024xf32)
        add_19 = paddle._C_ops.add(layer_norm_9, dropout_12)
        del dropout_12, layer_norm_9

        # pd_op.layer_norm: (1x20x1024xf32, 1x20xf32, 1x20xf32) <- (1x20x1024xf32, 1024xf32, 1024xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_19, parameter_18, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_19, parameter_18, parameter_19

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_12, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_20 = paddle._C_ops.add(matmul_16, parameter_16)
        del matmul_16, parameter_16

        # pd_op.reshape: (1x20x16x64xf32) <- (1x20x1024xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_20, full_int_array_1)
        del add_20

        # pd_op.transpose: (1x16x20x64xf32) <- (1x20x16x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_21 = paddle._C_ops.add(matmul_17, parameter_14)
        del matmul_17, parameter_14

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_22 = paddle._C_ops.add(matmul_18, parameter_12)
        del matmul_18, parameter_12

        # pd_op.reshape: (1x20x16x64xf32) <- (1x20x1024xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_21, full_int_array_1)
        del add_21

        # pd_op.transpose: (1x16x20x64xf32) <- (1x20x16x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x20x16x64xf32) <- (1x20x1024xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_22, full_int_array_1)
        del add_22, full_int_array_1

        # pd_op.transpose: (1x16x20x64xf32) <- (1x20x16x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.scale: (1x16x20x64xf32) <- (1x16x20x64xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(transpose_8, full_5, float("0"), True)
        del full_5, transpose_8

        # pd_op.matmul: (1x16x20x20xf32) <- (1x16x20x64xf32, 1x16x20x64xf32)
        matmul_19 = paddle._C_ops.matmul(scale_3, transpose_9, False, True)
        del scale_3, transpose_9

        # pd_op.add: (1x16x20x20xf32) <- (1x16x20x20xf32, 1x1x1x20xf32)
        add_23 = paddle._C_ops.add(matmul_19, unsqueeze_0)
        del matmul_19, unsqueeze_0

        # pd_op.softmax: (1x16x20x20xf32) <- (1x16x20x20xf32)
        softmax_2 = paddle._C_ops.softmax(add_23, -1)
        del add_23

        # pd_op.dropout: (1x16x20x20xf32, 1x16x20x20xui8) <- (1x16x20x20xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x16x20x64xf32) <- (1x16x20x20xf32, 1x16x20x64xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)
        del dropout_14, transpose_10

        # pd_op.transpose: (1x20x16x64xf32) <- (1x16x20x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x20x1024xf32) <- (1x20x16x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_2)
        del full_int_array_2, transpose_11

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x1024xf32, 1024x1024xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_11, False, False)
        del parameter_11, reshape_11

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_24 = paddle._C_ops.add(matmul_21, parameter_10)
        del matmul_21, parameter_10

        # pd_op.dropout: (1x20x1024xf32, 1x20x1024xui8) <- (1x20x1024xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_24, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_24

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1x20x1024xf32)
        add_25 = paddle._C_ops.add(layer_norm_12, dropout_16)
        del dropout_16, layer_norm_12

        # pd_op.layer_norm: (1x20x1024xf32, 1x20xf32, 1x20xf32) <- (1x20x1024xf32, 1024xf32, 1024xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_5, parameter_4, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_25, parameter_4, parameter_5

        # pd_op.matmul: (1x20x4096xf32) <- (1x20x1024xf32, 1024x4096xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_15, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x20x4096xf32) <- (1x20x4096xf32, 4096xf32)
        add_26 = paddle._C_ops.add(matmul_22, parameter_8)
        del matmul_22, parameter_8

        # pd_op.relu: (1x20x4096xf32) <- (1x20x4096xf32)
        relu_2 = paddle._C_ops.relu(add_26)
        del add_26

        # pd_op.matmul: (1x20x1024xf32) <- (1x20x4096xf32, 4096x1024xf32)
        matmul_23 = paddle._C_ops.matmul(relu_2, parameter_7, False, False)
        del parameter_7, relu_2

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1024xf32)
        add_27 = paddle._C_ops.add(matmul_23, parameter_6)
        del matmul_23, parameter_6

        # pd_op.dropout: (1x20x1024xf32, 1x20x1024xui8) <- (1x20x1024xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_27, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_27, full_4

        # pd_op.add: (1x20x1024xf32) <- (1x20x1024xf32, 1x20x1024xf32)
        add_28 = paddle._C_ops.add(layer_norm_15, dropout_18)
        del dropout_18, layer_norm_15

        # pd_op.layer_norm: (1x20x1024xf32, 1x20xf32, 1x20xf32) <- (1x20x1024xf32, 1024xf32, 1024xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_3, parameter_2, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_28, parameter_2, parameter_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1x1024xf32) <- (1x20x1024xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            layer_norm_18, [1], full_int_array_3, full_int_array_4, [1], [1]
        )
        del full_int_array_3, full_int_array_4, layer_norm_18

        # pd_op.matmul: (1x1024xf32) <- (1x1024xf32, 1024x1024xf32)
        matmul_24 = paddle._C_ops.matmul(slice_0, parameter_1, False, False)
        del parameter_1, slice_0

        # pd_op.add: (1x1024xf32) <- (1x1024xf32, 1024xf32)
        add_29 = paddle._C_ops.add(matmul_24, parameter_0)
        del matmul_24, parameter_0

        # pd_op.tanh: (1x1024xf32) <- (1x1024xf32)
        tanh_0 = paddle._C_ops.tanh(add_29)
        del add_29

        return tanh_0
