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

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x11xi64) <- (1x11xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_0, full_2, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_2

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x11xi64) <- (1x11xi64, 1xi32)
        cumsum_0 = paddle._C_ops.cumsum(full_like_0, full_3, False, False, False)
        del full_3

        # pd_op.subtract: (1x11xi64) <- (1x11xi64, 1x11xi64)
        subtract_0 = paddle._C_ops.subtract(cumsum_0, full_like_0)
        del cumsum_0, full_like_0

        # pd_op.embedding: (1x11x128xf32) <- (1x11xi64, 21128x128xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_38, -1, False)
        del data_0, parameter_38

        # pd_op.embedding: (1x11x128xf32) <- (1x11xi64, 512x128xf32)
        embedding_1 = paddle._C_ops.embedding(subtract_0, parameter_37, -1, False)
        del parameter_37

        # pd_op.embedding: (1x11x128xf32) <- (1x11xi64, 2x128xf32)
        embedding_2 = paddle._C_ops.embedding(data_1, parameter_36, -1, False)
        del data_1, parameter_36

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 1x11x128xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 1x11x128xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)

        # pd_op.layer_norm: (1x11x128xf32, 1x11xf32, 1x11xf32) <- (1x11x128xf32, 128xf32, 128xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_35, parameter_34, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

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

        # pd_op.dropout: (1x11x128xf32, 1x11x128xui8) <- (1x11x128xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x11x128xf32) <- (1x11x128xf32, 128x128xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_32)
        del parameter_32

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 2, 64]

        # pd_op.reshape: (1x11x2x64xf32) <- (1x11x128xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_1)

        # pd_op.transpose: (1x2x11x64xf32) <- (1x11x2x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x11x128xf32) <- (1x11x128xf32, 128x128xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_30)
        del parameter_30

        # pd_op.matmul: (1x11x128xf32) <- (1x11x128xf32, 128x128xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_28)
        del parameter_28

        # pd_op.reshape: (1x11x2x64xf32) <- (1x11x128xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_3, full_int_array_1)

        # pd_op.transpose: (1x2x11x64xf32) <- (1x11x2x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x11x2x64xf32) <- (1x11x128xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_4, full_int_array_1)

        # pd_op.transpose: (1x2x11x64xf32) <- (1x11x2x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_5

        # pd_op.scale: (1x2x11x64xf32) <- (1x2x11x64xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(transpose_0, full_5, float("0"), True)
        del transpose_0

        # pd_op.matmul: (1x2x11x11xf32) <- (1x2x11x64xf32, 1x2x11x64xf32)
        matmul_3 = paddle._C_ops.matmul(scale_1, transpose_1, False, True)

        # pd_op.add: (1x2x11x11xf32) <- (1x2x11x11xf32, 1x1x1x11xf32)
        add_5 = paddle._C_ops.add(matmul_3, unsqueeze_0)

        # pd_op.softmax: (1x2x11x11xf32) <- (1x2x11x11xf32)
        softmax_0 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.dropout: (1x2x11x11xf32, 1x2x11x11xui8) <- (1x2x11x11xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x2x11x64xf32) <- (1x2x11x11xf32, 1x2x11x64xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)

        # pd_op.transpose: (1x11x2x64xf32) <- (1x2x11x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, 0, 128]

        # pd_op.reshape: (1x11x128xf32) <- (1x11x2x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_2)

        # pd_op.matmul: (1x11x128xf32) <- (1x11x128xf32, 128x128xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_26)
        del parameter_26

        # pd_op.dropout: (1x11x128xf32, 1x11x128xui8) <- (1x11x128xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_6, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_6

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 1x11x128xf32)
        add_7 = paddle._C_ops.add(dropout_0, dropout_4)

        # pd_op.layer_norm: (1x11x128xf32, 1x11xf32, 1x11xf32) <- (1x11x128xf32, 128xf32, 128xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_21, parameter_20, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (1x11x512xf32) <- (1x11x128xf32, 128x512xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x11x512xf32) <- (1x11x512xf32, 512xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_24)
        del parameter_24

        # pd_op.gelu: (1x11x512xf32) <- (1x11x512xf32)
        gelu_0 = paddle._C_ops.gelu(add_8, False)

        # pd_op.matmul: (1x11x128xf32) <- (1x11x512xf32, 512x128xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_22)
        del parameter_22

        # pd_op.dropout: (1x11x128xf32, 1x11x128xui8) <- (1x11x128xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_9, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_9

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 1x11x128xf32)
        add_10 = paddle._C_ops.add(layer_norm_3, dropout_6)

        # pd_op.layer_norm: (1x11x128xf32, 1x11xf32, 1x11xf32) <- (1x11x128xf32, 128xf32, 128xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_19, parameter_18, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_18, parameter_19

        # pd_op.matmul: (1x11x128xf32) <- (1x11x128xf32, 128x128xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_11 = paddle._C_ops.add(matmul_8, parameter_16)
        del parameter_16

        # pd_op.reshape: (1x11x2x64xf32) <- (1x11x128xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_11, full_int_array_1)

        # pd_op.transpose: (1x2x11x64xf32) <- (1x11x2x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x11x128xf32) <- (1x11x128xf32, 128x128xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_14)
        del parameter_14

        # pd_op.matmul: (1x11x128xf32) <- (1x11x128xf32, 128x128xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_12)
        del parameter_12

        # pd_op.reshape: (1x11x2x64xf32) <- (1x11x128xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_12, full_int_array_1)

        # pd_op.transpose: (1x2x11x64xf32) <- (1x11x2x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x11x2x64xf32) <- (1x11x128xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_13, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (1x2x11x64xf32) <- (1x11x2x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.scale: (1x2x11x64xf32) <- (1x2x11x64xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(transpose_4, full_5, float("0"), True)
        del transpose_4

        # pd_op.matmul: (1x2x11x11xf32) <- (1x2x11x64xf32, 1x2x11x64xf32)
        matmul_11 = paddle._C_ops.matmul(scale_2, transpose_5, False, True)

        # pd_op.add: (1x2x11x11xf32) <- (1x2x11x11xf32, 1x1x1x11xf32)
        add_14 = paddle._C_ops.add(matmul_11, unsqueeze_0)

        # pd_op.softmax: (1x2x11x11xf32) <- (1x2x11x11xf32)
        softmax_1 = paddle._C_ops.softmax(add_14, -1)
        del add_14

        # pd_op.dropout: (1x2x11x11xf32, 1x2x11x11xui8) <- (1x2x11x11xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x2x11x64xf32) <- (1x2x11x11xf32, 1x2x11x64xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)

        # pd_op.transpose: (1x11x2x64xf32) <- (1x2x11x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x11x128xf32) <- (1x11x2x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_2)
        del full_int_array_2

        # pd_op.matmul: (1x11x128xf32) <- (1x11x128xf32, 128x128xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_15 = paddle._C_ops.add(matmul_13, parameter_10)
        del parameter_10

        # pd_op.dropout: (1x11x128xf32, 1x11x128xui8) <- (1x11x128xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_15, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_15

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 1x11x128xf32)
        add_16 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (1x11x128xf32, 1x11xf32, 1x11xf32) <- (1x11x128xf32, 128xf32, 128xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_5, parameter_4, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.matmul: (1x11x512xf32) <- (1x11x128xf32, 128x512xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x11x512xf32) <- (1x11x512xf32, 512xf32)
        add_17 = paddle._C_ops.add(matmul_14, parameter_8)
        del parameter_8

        # pd_op.gelu: (1x11x512xf32) <- (1x11x512xf32)
        gelu_1 = paddle._C_ops.gelu(add_17, False)

        # pd_op.matmul: (1x11x128xf32) <- (1x11x512xf32, 512x128xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 128xf32)
        add_18 = paddle._C_ops.add(matmul_15, parameter_6)
        del parameter_6

        # pd_op.dropout: (1x11x128xf32, 1x11x128xui8) <- (1x11x128xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_18, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_18

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 1x11x128xf32)
        add_19 = paddle._C_ops.add(layer_norm_9, dropout_12)

        # pd_op.layer_norm: (1x11x128xf32, 1x11xf32, 1x11xf32) <- (1x11x128xf32, 128xf32, 128xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_3, parameter_2, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1x128xf32) <- (1x11x128xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            layer_norm_12, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.matmul: (1x128xf32) <- (1x128xf32, 128x128xf32)
        matmul_16 = paddle._C_ops.matmul(slice_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (1x128xf32) <- (1x128xf32, 128xf32)
        add_20 = paddle._C_ops.add(matmul_16, parameter_0)
        del parameter_0

        # pd_op.tanh: (1x128xf32) <- (1x128xf32)
        tanh_0 = paddle._C_ops.tanh(add_20)
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
            add_3,
            add_4,
            add_7,
            add_8,
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            dropout_0,
            dropout_1,
            dropout_10,
            dropout_11,
            dropout_12,
            dropout_13,
            dropout_2,
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
            full_4,
            full_5,
            full_int_array_3,
            full_int_array_4,
            gelu_0,
            gelu_1,
            layer_norm_1,
            layer_norm_10,
            layer_norm_11,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            layer_norm_2,
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
            matmul_2,
            matmul_3,
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_8,
            matmul_9,
            reshape_3,
            reshape_7,
            scale_1,
            scale_2,
            slice_0,
            softmax_0,
            softmax_1,
            subtract_0,
            transpose_1,
            transpose_2,
            transpose_3,
            transpose_5,
            transpose_6,
            transpose_7,
            unsqueeze_0,
        )

        return tanh_0
