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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (1x1x21xi64) <- (1x21xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_0)
        del data_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.unsqueeze: (1x1x1x21xi64) <- (1x1x21xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(unsqueeze_0, full_int_array_1)
        del full_int_array_1, unsqueeze_0

        # pd_op.cast: (1x1x1x21xf32) <- (1x1x1x21xi64)
        cast_0 = paddle._C_ops.cast(unsqueeze_1, paddle.float32)
        del unsqueeze_1

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x21xf32) <- (1x1x1x21xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_0, float("1"), True)
        del cast_0, full_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x21xf32) <- (1x1x1x21xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_1, float("0"), True)
        del full_1, scale_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [21]

        # pd_op.slice: (1x21xi64) <- (1x512xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            parameter_0, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del full_int_array_3, parameter_0

        # pd_op.embedding: (1x21x128xf32) <- (1x21xi64, 30000x128xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_25, 0, False)
        del data_0, parameter_25

        # pd_op.embedding: (1x21x128xf32) <- (1x21xi64, 2x128xf32)
        embedding_1 = paddle._C_ops.embedding(data_2, parameter_23, -1, False)
        del data_2, parameter_23

        # pd_op.add: (1x21x128xf32) <- (1x21x128xf32, 1x21x128xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)
        del embedding_0, embedding_1

        # pd_op.embedding: (1x21x128xf32) <- (1x21xi64, 512x128xf32)
        embedding_2 = paddle._C_ops.embedding(slice_0, parameter_24, -1, False)
        del parameter_24, slice_0

        # pd_op.add: (1x21x128xf32) <- (1x21x128xf32, 1x21x128xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)
        del add_0, embedding_2

        # pd_op.layer_norm: (1x21x128xf32, 1x21xf32, 1x21xf32) <- (1x21x128xf32, 128xf32, 128xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_22, parameter_21, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_1, parameter_21, parameter_22

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x21x128xf32, 1x21x128xui8) <- (1x21x128xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x128xf32, 128x2048xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_20, False, False)
        del dropout_0, parameter_20

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_19)
        del matmul_0, parameter_19

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_1 = paddle._C_ops.matmul(add_2, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_15)
        del matmul_1

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_2 = paddle._C_ops.matmul(add_2, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_13)
        del matmul_2

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_3 = paddle._C_ops.matmul(add_2, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_5 = paddle._C_ops.add(matmul_3, parameter_11)
        del matmul_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 21, 16, 128]

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_3, full_int_array_4)
        del add_3

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_4, full_int_array_4)
        del add_4

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_5, full_int_array_4)
        del add_5

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_0, transpose_1, False, True)
        del transpose_0, transpose_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.0883883"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_4, full_3, float("0"), True)
        del matmul_4

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_6 = paddle._C_ops.add(scale_2, scale_1)
        del scale_2

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_6, -1)
        del add_6

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_5, [0, 2, 1, 3])
        del matmul_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [0, 0, -1]

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_5)
        del transpose_3

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_3, parameter_10, False, False)
        del reshape_3

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_7 = paddle._C_ops.add(matmul_6, parameter_9)
        del matmul_6

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_7, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_7

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_8 = paddle._C_ops.add(add_2, dropout_4)
        del add_2, dropout_4

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_8

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_3, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_5)
        del matmul_7

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_0 = paddle._C_ops.gelu(add_9, False)
        del add_9

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_8 = paddle._C_ops.matmul(gelu_0, parameter_4, False, False)
        del gelu_0

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_10 = paddle._C_ops.add(matmul_8, parameter_3)
        del matmul_8

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_11 = paddle._C_ops.add(add_10, layer_norm_3)
        del add_10, layer_norm_3

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_11

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_15)
        del matmul_9

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_13)
        del matmul_10

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_6, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_14 = paddle._C_ops.add(matmul_11, parameter_11)
        del matmul_11

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_12, full_int_array_4)
        del add_12

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_13, full_int_array_4)
        del add_13

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_14, full_int_array_4)
        del add_14

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_4, transpose_5, False, True)
        del transpose_4, transpose_5

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_12, full_3, float("0"), True)
        del matmul_12

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_15 = paddle._C_ops.add(scale_3, scale_1)
        del scale_3

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_15, -1)
        del add_15

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_13 = paddle._C_ops.matmul(dropout_6, transpose_6, False, False)
        del dropout_6, transpose_6

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_13, [0, 2, 1, 3])
        del matmul_13

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_5)
        del transpose_7

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_14 = paddle._C_ops.matmul(reshape_7, parameter_10, False, False)
        del reshape_7

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_16 = paddle._C_ops.add(matmul_14, parameter_9)
        del matmul_14

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_16, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_16

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_17 = paddle._C_ops.add(layer_norm_6, dropout_8)
        del dropout_8, layer_norm_6

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_17

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_9, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_18 = paddle._C_ops.add(matmul_15, parameter_5)
        del matmul_15

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_1 = paddle._C_ops.gelu(add_18, False)
        del add_18

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_16 = paddle._C_ops.matmul(gelu_1, parameter_4, False, False)
        del gelu_1

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_19 = paddle._C_ops.add(matmul_16, parameter_3)
        del matmul_16

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_20 = paddle._C_ops.add(add_19, layer_norm_9)
        del add_19, layer_norm_9

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_20

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_21 = paddle._C_ops.add(matmul_17, parameter_15)
        del matmul_17

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_22 = paddle._C_ops.add(matmul_18, parameter_13)
        del matmul_18

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_12, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_23 = paddle._C_ops.add(matmul_19, parameter_11)
        del matmul_19

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_21, full_int_array_4)
        del add_21

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_22, full_int_array_4)
        del add_22

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_23, full_int_array_4)
        del add_23

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_8, transpose_9, False, True)
        del transpose_8, transpose_9

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_20, full_3, float("0"), True)
        del matmul_20

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_24 = paddle._C_ops.add(scale_4, scale_1)
        del scale_4

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_24, -1)
        del add_24

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_21 = paddle._C_ops.matmul(dropout_10, transpose_10, False, False)
        del dropout_10, transpose_10

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_5)
        del transpose_11

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_11, parameter_10, False, False)
        del reshape_11

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_25 = paddle._C_ops.add(matmul_22, parameter_9)
        del matmul_22

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_25, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_25

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_26 = paddle._C_ops.add(layer_norm_12, dropout_12)
        del dropout_12, layer_norm_12

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_26, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_26

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_15, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_27 = paddle._C_ops.add(matmul_23, parameter_5)
        del matmul_23

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_2 = paddle._C_ops.gelu(add_27, False)
        del add_27

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_2, parameter_4, False, False)
        del gelu_2

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_28 = paddle._C_ops.add(matmul_24, parameter_3)
        del matmul_24

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_29 = paddle._C_ops.add(add_28, layer_norm_15)
        del add_28, layer_norm_15

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_29, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_29

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_30 = paddle._C_ops.add(matmul_25, parameter_15)
        del matmul_25

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_31 = paddle._C_ops.add(matmul_26, parameter_13)
        del matmul_26

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_27 = paddle._C_ops.matmul(layer_norm_18, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_32 = paddle._C_ops.add(matmul_27, parameter_11)
        del matmul_27

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_30, full_int_array_4)
        del add_30

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_31, full_int_array_4)
        del add_31

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_32, full_int_array_4)
        del add_32

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_28 = paddle._C_ops.matmul(transpose_12, transpose_13, False, True)
        del transpose_12, transpose_13

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_28, full_3, float("0"), True)
        del matmul_28

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_33 = paddle._C_ops.add(scale_5, scale_1)
        del scale_5

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_33, -1)
        del add_33

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_29 = paddle._C_ops.matmul(dropout_14, transpose_14, False, False)
        del dropout_14, transpose_14

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_29, [0, 2, 1, 3])
        del matmul_29

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_5)
        del transpose_15

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_30 = paddle._C_ops.matmul(reshape_15, parameter_10, False, False)
        del reshape_15

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_34 = paddle._C_ops.add(matmul_30, parameter_9)
        del matmul_30

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_34, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_34

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_35 = paddle._C_ops.add(layer_norm_18, dropout_16)
        del dropout_16, layer_norm_18

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_35

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_21, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_36 = paddle._C_ops.add(matmul_31, parameter_5)
        del matmul_31

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_3 = paddle._C_ops.gelu(add_36, False)
        del add_36

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_32 = paddle._C_ops.matmul(gelu_3, parameter_4, False, False)
        del gelu_3

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_37 = paddle._C_ops.add(matmul_32, parameter_3)
        del matmul_32

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_38 = paddle._C_ops.add(add_37, layer_norm_21)
        del add_37, layer_norm_21

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_38

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_39 = paddle._C_ops.add(matmul_33, parameter_15)
        del matmul_33

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_40 = paddle._C_ops.add(matmul_34, parameter_13)
        del matmul_34

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_24, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_41 = paddle._C_ops.add(matmul_35, parameter_11)
        del matmul_35

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_39, full_int_array_4)
        del add_39

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_40, full_int_array_4)
        del add_40

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_41, full_int_array_4)
        del add_41

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_36 = paddle._C_ops.matmul(transpose_16, transpose_17, False, True)
        del transpose_16, transpose_17

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_36, full_3, float("0"), True)
        del matmul_36

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_42 = paddle._C_ops.add(scale_6, scale_1)
        del scale_6

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(add_42, -1)
        del add_42

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_4

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_37 = paddle._C_ops.matmul(dropout_18, transpose_18, False, False)
        del dropout_18, transpose_18

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_37, [0, 2, 1, 3])
        del matmul_37

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_5)
        del transpose_19

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_19, parameter_10, False, False)
        del reshape_19

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_43 = paddle._C_ops.add(matmul_38, parameter_9)
        del matmul_38

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_43, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_43

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_44 = paddle._C_ops.add(layer_norm_24, dropout_20)
        del dropout_20, layer_norm_24

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_44, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_44

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_39 = paddle._C_ops.matmul(layer_norm_27, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_45 = paddle._C_ops.add(matmul_39, parameter_5)
        del matmul_39

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_4 = paddle._C_ops.gelu(add_45, False)
        del add_45

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_40 = paddle._C_ops.matmul(gelu_4, parameter_4, False, False)
        del gelu_4

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_46 = paddle._C_ops.add(matmul_40, parameter_3)
        del matmul_40

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_47 = paddle._C_ops.add(add_46, layer_norm_27)
        del add_46, layer_norm_27

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_47, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_47

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_48 = paddle._C_ops.add(matmul_41, parameter_15)
        del matmul_41

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_49 = paddle._C_ops.add(matmul_42, parameter_13)
        del matmul_42

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_43 = paddle._C_ops.matmul(layer_norm_30, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_50 = paddle._C_ops.add(matmul_43, parameter_11)
        del matmul_43

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_48, full_int_array_4)
        del add_48

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_49, full_int_array_4)
        del add_49

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_50, full_int_array_4)
        del add_50

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_44 = paddle._C_ops.matmul(transpose_20, transpose_21, False, True)
        del transpose_20, transpose_21

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_44, full_3, float("0"), True)
        del matmul_44

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_51 = paddle._C_ops.add(scale_7, scale_1)
        del scale_7

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_51, -1)
        del add_51

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_5

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_45 = paddle._C_ops.matmul(dropout_22, transpose_22, False, False)
        del dropout_22, transpose_22

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])
        del matmul_45

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_5)
        del transpose_23

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_23, parameter_10, False, False)
        del reshape_23

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_52 = paddle._C_ops.add(matmul_46, parameter_9)
        del matmul_46

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_52, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_52

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_53 = paddle._C_ops.add(layer_norm_30, dropout_24)
        del dropout_24, layer_norm_30

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_53, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_53

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_33, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_54 = paddle._C_ops.add(matmul_47, parameter_5)
        del matmul_47

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_5 = paddle._C_ops.gelu(add_54, False)
        del add_54

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_48 = paddle._C_ops.matmul(gelu_5, parameter_4, False, False)
        del gelu_5

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_55 = paddle._C_ops.add(matmul_48, parameter_3)
        del matmul_48

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_56 = paddle._C_ops.add(add_55, layer_norm_33)
        del add_55, layer_norm_33

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_56, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_56

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_36, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_57 = paddle._C_ops.add(matmul_49, parameter_15)
        del matmul_49

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_36, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_58 = paddle._C_ops.add(matmul_50, parameter_13)
        del matmul_50

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_51 = paddle._C_ops.matmul(layer_norm_36, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_59 = paddle._C_ops.add(matmul_51, parameter_11)
        del matmul_51

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_57, full_int_array_4)
        del add_57

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_58, full_int_array_4)
        del add_58

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_59, full_int_array_4)
        del add_59

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_52 = paddle._C_ops.matmul(transpose_24, transpose_25, False, True)
        del transpose_24, transpose_25

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_52, full_3, float("0"), True)
        del matmul_52

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_60 = paddle._C_ops.add(scale_8, scale_1)
        del scale_8

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_6 = paddle._C_ops.softmax(add_60, -1)
        del add_60

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_6

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_53 = paddle._C_ops.matmul(dropout_26, transpose_26, False, False)
        del dropout_26, transpose_26

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_53, [0, 2, 1, 3])
        del matmul_53

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_27, full_int_array_5)
        del transpose_27

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_54 = paddle._C_ops.matmul(reshape_27, parameter_10, False, False)
        del reshape_27

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_61 = paddle._C_ops.add(matmul_54, parameter_9)
        del matmul_54

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_61, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_61

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_62 = paddle._C_ops.add(layer_norm_36, dropout_28)
        del dropout_28, layer_norm_36

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_62, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_62

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_39, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_63 = paddle._C_ops.add(matmul_55, parameter_5)
        del matmul_55

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_6 = paddle._C_ops.gelu(add_63, False)
        del add_63

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_56 = paddle._C_ops.matmul(gelu_6, parameter_4, False, False)
        del gelu_6

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_64 = paddle._C_ops.add(matmul_56, parameter_3)
        del matmul_56

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_65 = paddle._C_ops.add(add_64, layer_norm_39)
        del add_64, layer_norm_39

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_65, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_65

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_57 = paddle._C_ops.matmul(layer_norm_42, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_66 = paddle._C_ops.add(matmul_57, parameter_15)
        del matmul_57

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_42, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_67 = paddle._C_ops.add(matmul_58, parameter_13)
        del matmul_58

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_59 = paddle._C_ops.matmul(layer_norm_42, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_68 = paddle._C_ops.add(matmul_59, parameter_11)
        del matmul_59

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_66, full_int_array_4)
        del add_66

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_67, full_int_array_4)
        del add_67

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_68, full_int_array_4)
        del add_68

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_60 = paddle._C_ops.matmul(transpose_28, transpose_29, False, True)
        del transpose_28, transpose_29

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_60, full_3, float("0"), True)
        del matmul_60

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_69 = paddle._C_ops.add(scale_9, scale_1)
        del scale_9

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_7 = paddle._C_ops.softmax(add_69, -1)
        del add_69

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_7

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_61 = paddle._C_ops.matmul(dropout_30, transpose_30, False, False)
        del dropout_30, transpose_30

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_61, [0, 2, 1, 3])
        del matmul_61

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_31, full_int_array_5)
        del transpose_31

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_62 = paddle._C_ops.matmul(reshape_31, parameter_10, False, False)
        del reshape_31

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_70 = paddle._C_ops.add(matmul_62, parameter_9)
        del matmul_62

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_70, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_70

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_71 = paddle._C_ops.add(layer_norm_42, dropout_32)
        del dropout_32, layer_norm_42

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_71, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_71

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_63 = paddle._C_ops.matmul(layer_norm_45, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_72 = paddle._C_ops.add(matmul_63, parameter_5)
        del matmul_63

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_7 = paddle._C_ops.gelu(add_72, False)
        del add_72

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_64 = paddle._C_ops.matmul(gelu_7, parameter_4, False, False)
        del gelu_7

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_73 = paddle._C_ops.add(matmul_64, parameter_3)
        del matmul_64

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_74 = paddle._C_ops.add(add_73, layer_norm_45)
        del add_73, layer_norm_45

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_74, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_74

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_65 = paddle._C_ops.matmul(layer_norm_48, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_75 = paddle._C_ops.add(matmul_65, parameter_15)
        del matmul_65

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_48, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_76 = paddle._C_ops.add(matmul_66, parameter_13)
        del matmul_66

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_48, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_77 = paddle._C_ops.add(matmul_67, parameter_11)
        del matmul_67

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_75, full_int_array_4)
        del add_75

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(add_76, full_int_array_4)
        del add_76

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_77, full_int_array_4)
        del add_77

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_68 = paddle._C_ops.matmul(transpose_32, transpose_33, False, True)
        del transpose_32, transpose_33

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_68, full_3, float("0"), True)
        del matmul_68

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_78 = paddle._C_ops.add(scale_10, scale_1)
        del scale_10

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_8 = paddle._C_ops.softmax(add_78, -1)
        del add_78

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_8

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_69 = paddle._C_ops.matmul(dropout_34, transpose_34, False, False)
        del dropout_34, transpose_34

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_69, [0, 2, 1, 3])
        del matmul_69

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_35, full_int_array_5)
        del transpose_35

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_70 = paddle._C_ops.matmul(reshape_35, parameter_10, False, False)
        del reshape_35

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_79 = paddle._C_ops.add(matmul_70, parameter_9)
        del matmul_70

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_79, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_79

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_80 = paddle._C_ops.add(layer_norm_48, dropout_36)
        del dropout_36, layer_norm_48

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_80, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_80

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_71 = paddle._C_ops.matmul(layer_norm_51, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_81 = paddle._C_ops.add(matmul_71, parameter_5)
        del matmul_71

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_8 = paddle._C_ops.gelu(add_81, False)
        del add_81

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_72 = paddle._C_ops.matmul(gelu_8, parameter_4, False, False)
        del gelu_8

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_82 = paddle._C_ops.add(matmul_72, parameter_3)
        del matmul_72

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_83 = paddle._C_ops.add(add_82, layer_norm_51)
        del add_82, layer_norm_51

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_83, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_83

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_54, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_84 = paddle._C_ops.add(matmul_73, parameter_15)
        del matmul_73

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_54, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_85 = paddle._C_ops.add(matmul_74, parameter_13)
        del matmul_74

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_75 = paddle._C_ops.matmul(layer_norm_54, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_86 = paddle._C_ops.add(matmul_75, parameter_11)
        del matmul_75

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_84, full_int_array_4)
        del add_84

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_85, full_int_array_4)
        del add_85

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_86, full_int_array_4)
        del add_86

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_76 = paddle._C_ops.matmul(transpose_36, transpose_37, False, True)
        del transpose_36, transpose_37

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_76, full_3, float("0"), True)
        del matmul_76

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_87 = paddle._C_ops.add(scale_11, scale_1)
        del scale_11

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_9 = paddle._C_ops.softmax(add_87, -1)
        del add_87

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_9

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_77 = paddle._C_ops.matmul(dropout_38, transpose_38, False, False)
        del dropout_38, transpose_38

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_77, [0, 2, 1, 3])
        del matmul_77

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_39, full_int_array_5)
        del transpose_39

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_78 = paddle._C_ops.matmul(reshape_39, parameter_10, False, False)
        del reshape_39

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_88 = paddle._C_ops.add(matmul_78, parameter_9)
        del matmul_78

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_88, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_88

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_89 = paddle._C_ops.add(layer_norm_54, dropout_40)
        del dropout_40, layer_norm_54

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_89, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_89

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_79 = paddle._C_ops.matmul(layer_norm_57, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_90 = paddle._C_ops.add(matmul_79, parameter_5)
        del matmul_79

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_9 = paddle._C_ops.gelu(add_90, False)
        del add_90

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_80 = paddle._C_ops.matmul(gelu_9, parameter_4, False, False)
        del gelu_9

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_91 = paddle._C_ops.add(matmul_80, parameter_3)
        del matmul_80

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_92 = paddle._C_ops.add(add_91, layer_norm_57)
        del add_91, layer_norm_57

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_92, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_92

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_60, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_93 = paddle._C_ops.add(matmul_81, parameter_15)
        del matmul_81

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_60, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_94 = paddle._C_ops.add(matmul_82, parameter_13)
        del matmul_82

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_83 = paddle._C_ops.matmul(layer_norm_60, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_95 = paddle._C_ops.add(matmul_83, parameter_11)
        del matmul_83

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_93, full_int_array_4)
        del add_93

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_94, full_int_array_4)
        del add_94

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_95, full_int_array_4)
        del add_95

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_84 = paddle._C_ops.matmul(transpose_40, transpose_41, False, True)
        del transpose_40, transpose_41

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(matmul_84, full_3, float("0"), True)
        del matmul_84

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_96 = paddle._C_ops.add(scale_12, scale_1)
        del scale_12

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_10 = paddle._C_ops.softmax(add_96, -1)
        del add_96

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_10

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_85 = paddle._C_ops.matmul(dropout_42, transpose_42, False, False)
        del dropout_42, transpose_42

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_85, [0, 2, 1, 3])
        del matmul_85

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_43, full_int_array_5)
        del transpose_43

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_86 = paddle._C_ops.matmul(reshape_43, parameter_10, False, False)
        del reshape_43

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_97 = paddle._C_ops.add(matmul_86, parameter_9)
        del matmul_86

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_97, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_97

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_98 = paddle._C_ops.add(layer_norm_60, dropout_44)
        del dropout_44, layer_norm_60

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_98, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_98

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_87 = paddle._C_ops.matmul(layer_norm_63, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_99 = paddle._C_ops.add(matmul_87, parameter_5)
        del matmul_87

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_10 = paddle._C_ops.gelu(add_99, False)
        del add_99

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_88 = paddle._C_ops.matmul(gelu_10, parameter_4, False, False)
        del gelu_10

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_100 = paddle._C_ops.add(matmul_88, parameter_3)
        del matmul_88

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_101 = paddle._C_ops.add(add_100, layer_norm_63)
        del add_100, layer_norm_63

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_101, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_101

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_66, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_102 = paddle._C_ops.add(matmul_89, parameter_15)
        del matmul_89

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_66, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_103 = paddle._C_ops.add(matmul_90, parameter_13)
        del matmul_90

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_66, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_104 = paddle._C_ops.add(matmul_91, parameter_11)
        del matmul_91

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(add_102, full_int_array_4)
        del add_102

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_103, full_int_array_4)
        del add_103

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_104, full_int_array_4)
        del add_104

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_92 = paddle._C_ops.matmul(transpose_44, transpose_45, False, True)
        del transpose_44, transpose_45

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(matmul_92, full_3, float("0"), True)
        del matmul_92

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_105 = paddle._C_ops.add(scale_13, scale_1)
        del scale_13

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_11 = paddle._C_ops.softmax(add_105, -1)
        del add_105

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_11

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_93 = paddle._C_ops.matmul(dropout_46, transpose_46, False, False)
        del dropout_46, transpose_46

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_93, [0, 2, 1, 3])
        del matmul_93

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_47, full_int_array_5)
        del transpose_47

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_94 = paddle._C_ops.matmul(reshape_47, parameter_10, False, False)
        del reshape_47

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_106 = paddle._C_ops.add(matmul_94, parameter_9)
        del matmul_94

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_106, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_106

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_107 = paddle._C_ops.add(layer_norm_66, dropout_48)
        del dropout_48, layer_norm_66

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_107, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_107

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_95 = paddle._C_ops.matmul(layer_norm_69, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_108 = paddle._C_ops.add(matmul_95, parameter_5)
        del matmul_95

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_11 = paddle._C_ops.gelu(add_108, False)
        del add_108

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_96 = paddle._C_ops.matmul(gelu_11, parameter_4, False, False)
        del gelu_11

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_109 = paddle._C_ops.add(matmul_96, parameter_3)
        del matmul_96

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_110 = paddle._C_ops.add(add_109, layer_norm_69)
        del add_109, layer_norm_69

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_110, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_110

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_97 = paddle._C_ops.matmul(layer_norm_72, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_111 = paddle._C_ops.add(matmul_97, parameter_15)
        del matmul_97

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_98 = paddle._C_ops.matmul(layer_norm_72, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_112 = paddle._C_ops.add(matmul_98, parameter_13)
        del matmul_98

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_99 = paddle._C_ops.matmul(layer_norm_72, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_113 = paddle._C_ops.add(matmul_99, parameter_11)
        del matmul_99

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(add_111, full_int_array_4)
        del add_111

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_48, [0, 2, 1, 3])
        del reshape_48

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(add_112, full_int_array_4)
        del add_112

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_49, [0, 2, 1, 3])
        del reshape_49

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(add_113, full_int_array_4)
        del add_113

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_50, [0, 2, 1, 3])
        del reshape_50

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_100 = paddle._C_ops.matmul(transpose_48, transpose_49, False, True)
        del transpose_48, transpose_49

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(matmul_100, full_3, float("0"), True)
        del matmul_100

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_114 = paddle._C_ops.add(scale_14, scale_1)
        del scale_14

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_12 = paddle._C_ops.softmax(add_114, -1)
        del add_114

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_12, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_12

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_101 = paddle._C_ops.matmul(dropout_50, transpose_50, False, False)
        del dropout_50, transpose_50

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_51 = paddle._C_ops.transpose(matmul_101, [0, 2, 1, 3])
        del matmul_101

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_51 = paddle._C_ops.reshape(transpose_51, full_int_array_5)
        del transpose_51

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_102 = paddle._C_ops.matmul(reshape_51, parameter_10, False, False)
        del reshape_51

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_115 = paddle._C_ops.add(matmul_102, parameter_9)
        del matmul_102

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_115, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_115

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_116 = paddle._C_ops.add(layer_norm_72, dropout_52)
        del dropout_52, layer_norm_72

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_116, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_116

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_103 = paddle._C_ops.matmul(layer_norm_75, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_117 = paddle._C_ops.add(matmul_103, parameter_5)
        del matmul_103

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_12 = paddle._C_ops.gelu(add_117, False)
        del add_117

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_104 = paddle._C_ops.matmul(gelu_12, parameter_4, False, False)
        del gelu_12

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_118 = paddle._C_ops.add(matmul_104, parameter_3)
        del matmul_104

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_119 = paddle._C_ops.add(add_118, layer_norm_75)
        del add_118, layer_norm_75

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_119, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_119

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_105 = paddle._C_ops.matmul(layer_norm_78, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_120 = paddle._C_ops.add(matmul_105, parameter_15)
        del matmul_105

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_106 = paddle._C_ops.matmul(layer_norm_78, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_121 = paddle._C_ops.add(matmul_106, parameter_13)
        del matmul_106

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_107 = paddle._C_ops.matmul(layer_norm_78, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_122 = paddle._C_ops.add(matmul_107, parameter_11)
        del matmul_107

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(add_120, full_int_array_4)
        del add_120

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_52, [0, 2, 1, 3])
        del reshape_52

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(add_121, full_int_array_4)
        del add_121

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_53, [0, 2, 1, 3])
        del reshape_53

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(add_122, full_int_array_4)
        del add_122

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_54 = paddle._C_ops.transpose(reshape_54, [0, 2, 1, 3])
        del reshape_54

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_108 = paddle._C_ops.matmul(transpose_52, transpose_53, False, True)
        del transpose_52, transpose_53

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(matmul_108, full_3, float("0"), True)
        del matmul_108

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_123 = paddle._C_ops.add(scale_15, scale_1)
        del scale_15

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_13 = paddle._C_ops.softmax(add_123, -1)
        del add_123

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_13, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_13

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_109 = paddle._C_ops.matmul(dropout_54, transpose_54, False, False)
        del dropout_54, transpose_54

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_55 = paddle._C_ops.transpose(matmul_109, [0, 2, 1, 3])
        del matmul_109

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_55 = paddle._C_ops.reshape(transpose_55, full_int_array_5)
        del transpose_55

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_110 = paddle._C_ops.matmul(reshape_55, parameter_10, False, False)
        del reshape_55

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_124 = paddle._C_ops.add(matmul_110, parameter_9)
        del matmul_110

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_124, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_124

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_125 = paddle._C_ops.add(layer_norm_78, dropout_56)
        del dropout_56, layer_norm_78

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_125, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_125

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_111 = paddle._C_ops.matmul(layer_norm_81, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_126 = paddle._C_ops.add(matmul_111, parameter_5)
        del matmul_111

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_13 = paddle._C_ops.gelu(add_126, False)
        del add_126

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_112 = paddle._C_ops.matmul(gelu_13, parameter_4, False, False)
        del gelu_13

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_127 = paddle._C_ops.add(matmul_112, parameter_3)
        del matmul_112

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_128 = paddle._C_ops.add(add_127, layer_norm_81)
        del add_127, layer_norm_81

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_128, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_128

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_113 = paddle._C_ops.matmul(layer_norm_84, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_129 = paddle._C_ops.add(matmul_113, parameter_15)
        del matmul_113

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_114 = paddle._C_ops.matmul(layer_norm_84, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_130 = paddle._C_ops.add(matmul_114, parameter_13)
        del matmul_114

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_115 = paddle._C_ops.matmul(layer_norm_84, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_131 = paddle._C_ops.add(matmul_115, parameter_11)
        del matmul_115

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(add_129, full_int_array_4)
        del add_129

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_56, [0, 2, 1, 3])
        del reshape_56

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(add_130, full_int_array_4)
        del add_130

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_57 = paddle._C_ops.transpose(reshape_57, [0, 2, 1, 3])
        del reshape_57

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(add_131, full_int_array_4)
        del add_131

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_58, [0, 2, 1, 3])
        del reshape_58

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_116 = paddle._C_ops.matmul(transpose_56, transpose_57, False, True)
        del transpose_56, transpose_57

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(matmul_116, full_3, float("0"), True)
        del matmul_116

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_132 = paddle._C_ops.add(scale_16, scale_1)
        del scale_16

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_14 = paddle._C_ops.softmax(add_132, -1)
        del add_132

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_14, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_14

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_117 = paddle._C_ops.matmul(dropout_58, transpose_58, False, False)
        del dropout_58, transpose_58

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_59 = paddle._C_ops.transpose(matmul_117, [0, 2, 1, 3])
        del matmul_117

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_59 = paddle._C_ops.reshape(transpose_59, full_int_array_5)
        del transpose_59

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_118 = paddle._C_ops.matmul(reshape_59, parameter_10, False, False)
        del reshape_59

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_133 = paddle._C_ops.add(matmul_118, parameter_9)
        del matmul_118

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_133, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_133

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_134 = paddle._C_ops.add(layer_norm_84, dropout_60)
        del dropout_60, layer_norm_84

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_134, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_134

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_119 = paddle._C_ops.matmul(layer_norm_87, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_135 = paddle._C_ops.add(matmul_119, parameter_5)
        del matmul_119

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_14 = paddle._C_ops.gelu(add_135, False)
        del add_135

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_120 = paddle._C_ops.matmul(gelu_14, parameter_4, False, False)
        del gelu_14

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_136 = paddle._C_ops.add(matmul_120, parameter_3)
        del matmul_120

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_137 = paddle._C_ops.add(add_136, layer_norm_87)
        del add_136, layer_norm_87

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_137, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_137

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_121 = paddle._C_ops.matmul(layer_norm_90, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_138 = paddle._C_ops.add(matmul_121, parameter_15)
        del matmul_121

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_122 = paddle._C_ops.matmul(layer_norm_90, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_139 = paddle._C_ops.add(matmul_122, parameter_13)
        del matmul_122

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_123 = paddle._C_ops.matmul(layer_norm_90, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_140 = paddle._C_ops.add(matmul_123, parameter_11)
        del matmul_123

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(add_138, full_int_array_4)
        del add_138

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_60 = paddle._C_ops.transpose(reshape_60, [0, 2, 1, 3])
        del reshape_60

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(add_139, full_int_array_4)
        del add_139

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_61, [0, 2, 1, 3])
        del reshape_61

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(add_140, full_int_array_4)
        del add_140

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_62, [0, 2, 1, 3])
        del reshape_62

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_124 = paddle._C_ops.matmul(transpose_60, transpose_61, False, True)
        del transpose_60, transpose_61

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(matmul_124, full_3, float("0"), True)
        del matmul_124

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_141 = paddle._C_ops.add(scale_17, scale_1)
        del scale_17

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_15 = paddle._C_ops.softmax(add_141, -1)
        del add_141

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_15, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_15

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_125 = paddle._C_ops.matmul(dropout_62, transpose_62, False, False)
        del dropout_62, transpose_62

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_63 = paddle._C_ops.transpose(matmul_125, [0, 2, 1, 3])
        del matmul_125

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_63 = paddle._C_ops.reshape(transpose_63, full_int_array_5)
        del transpose_63

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_126 = paddle._C_ops.matmul(reshape_63, parameter_10, False, False)
        del reshape_63

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_142 = paddle._C_ops.add(matmul_126, parameter_9)
        del matmul_126

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_142, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_142

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_143 = paddle._C_ops.add(layer_norm_90, dropout_64)
        del dropout_64, layer_norm_90

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_143, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_143

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_127 = paddle._C_ops.matmul(layer_norm_93, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_144 = paddle._C_ops.add(matmul_127, parameter_5)
        del matmul_127

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_15 = paddle._C_ops.gelu(add_144, False)
        del add_144

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_128 = paddle._C_ops.matmul(gelu_15, parameter_4, False, False)
        del gelu_15

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_145 = paddle._C_ops.add(matmul_128, parameter_3)
        del matmul_128

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_146 = paddle._C_ops.add(add_145, layer_norm_93)
        del add_145, layer_norm_93

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_146, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_146

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_129 = paddle._C_ops.matmul(layer_norm_96, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_147 = paddle._C_ops.add(matmul_129, parameter_15)
        del matmul_129

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_130 = paddle._C_ops.matmul(layer_norm_96, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_148 = paddle._C_ops.add(matmul_130, parameter_13)
        del matmul_130

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_131 = paddle._C_ops.matmul(layer_norm_96, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_149 = paddle._C_ops.add(matmul_131, parameter_11)
        del matmul_131

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(add_147, full_int_array_4)
        del add_147

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_64, [0, 2, 1, 3])
        del reshape_64

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(add_148, full_int_array_4)
        del add_148

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_65, [0, 2, 1, 3])
        del reshape_65

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(add_149, full_int_array_4)
        del add_149

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_66 = paddle._C_ops.transpose(reshape_66, [0, 2, 1, 3])
        del reshape_66

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_132 = paddle._C_ops.matmul(transpose_64, transpose_65, False, True)
        del transpose_64, transpose_65

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(matmul_132, full_3, float("0"), True)
        del matmul_132

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_150 = paddle._C_ops.add(scale_18, scale_1)
        del scale_18

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_16 = paddle._C_ops.softmax(add_150, -1)
        del add_150

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_16, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_16

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_133 = paddle._C_ops.matmul(dropout_66, transpose_66, False, False)
        del dropout_66, transpose_66

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_67 = paddle._C_ops.transpose(matmul_133, [0, 2, 1, 3])
        del matmul_133

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_67, full_int_array_5)
        del transpose_67

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_134 = paddle._C_ops.matmul(reshape_67, parameter_10, False, False)
        del reshape_67

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_151 = paddle._C_ops.add(matmul_134, parameter_9)
        del matmul_134

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_151, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_151

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_152 = paddle._C_ops.add(layer_norm_96, dropout_68)
        del dropout_68, layer_norm_96

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_152, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_152

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_135 = paddle._C_ops.matmul(layer_norm_99, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_153 = paddle._C_ops.add(matmul_135, parameter_5)
        del matmul_135

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_16 = paddle._C_ops.gelu(add_153, False)
        del add_153

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_136 = paddle._C_ops.matmul(gelu_16, parameter_4, False, False)
        del gelu_16

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_154 = paddle._C_ops.add(matmul_136, parameter_3)
        del matmul_136

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_155 = paddle._C_ops.add(add_154, layer_norm_99)
        del add_154, layer_norm_99

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_155, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_155

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_137 = paddle._C_ops.matmul(layer_norm_102, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_156 = paddle._C_ops.add(matmul_137, parameter_15)
        del matmul_137

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_138 = paddle._C_ops.matmul(layer_norm_102, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_157 = paddle._C_ops.add(matmul_138, parameter_13)
        del matmul_138

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_139 = paddle._C_ops.matmul(layer_norm_102, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_158 = paddle._C_ops.add(matmul_139, parameter_11)
        del matmul_139

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(add_156, full_int_array_4)
        del add_156

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_68 = paddle._C_ops.transpose(reshape_68, [0, 2, 1, 3])
        del reshape_68

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(add_157, full_int_array_4)
        del add_157

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_69 = paddle._C_ops.transpose(reshape_69, [0, 2, 1, 3])
        del reshape_69

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(add_158, full_int_array_4)
        del add_158

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_70, [0, 2, 1, 3])
        del reshape_70

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_140 = paddle._C_ops.matmul(transpose_68, transpose_69, False, True)
        del transpose_68, transpose_69

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(matmul_140, full_3, float("0"), True)
        del matmul_140

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_159 = paddle._C_ops.add(scale_19, scale_1)
        del scale_19

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_17 = paddle._C_ops.softmax(add_159, -1)
        del add_159

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_17, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_17

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_141 = paddle._C_ops.matmul(dropout_70, transpose_70, False, False)
        del dropout_70, transpose_70

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_71 = paddle._C_ops.transpose(matmul_141, [0, 2, 1, 3])
        del matmul_141

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_71 = paddle._C_ops.reshape(transpose_71, full_int_array_5)
        del transpose_71

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_142 = paddle._C_ops.matmul(reshape_71, parameter_10, False, False)
        del reshape_71

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_160 = paddle._C_ops.add(matmul_142, parameter_9)
        del matmul_142

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_160, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_160

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_161 = paddle._C_ops.add(layer_norm_102, dropout_72)
        del dropout_72, layer_norm_102

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_161, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_161

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_143 = paddle._C_ops.matmul(layer_norm_105, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_162 = paddle._C_ops.add(matmul_143, parameter_5)
        del matmul_143

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_17 = paddle._C_ops.gelu(add_162, False)
        del add_162

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_144 = paddle._C_ops.matmul(gelu_17, parameter_4, False, False)
        del gelu_17

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_163 = paddle._C_ops.add(matmul_144, parameter_3)
        del matmul_144

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_164 = paddle._C_ops.add(add_163, layer_norm_105)
        del add_163, layer_norm_105

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_164, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_164

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_145 = paddle._C_ops.matmul(layer_norm_108, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_165 = paddle._C_ops.add(matmul_145, parameter_15)
        del matmul_145

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_146 = paddle._C_ops.matmul(layer_norm_108, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_166 = paddle._C_ops.add(matmul_146, parameter_13)
        del matmul_146

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_147 = paddle._C_ops.matmul(layer_norm_108, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_167 = paddle._C_ops.add(matmul_147, parameter_11)
        del matmul_147

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(add_165, full_int_array_4)
        del add_165

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_72 = paddle._C_ops.transpose(reshape_72, [0, 2, 1, 3])
        del reshape_72

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(add_166, full_int_array_4)
        del add_166

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_73, [0, 2, 1, 3])
        del reshape_73

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_74 = paddle._C_ops.reshape(add_167, full_int_array_4)
        del add_167

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_74 = paddle._C_ops.transpose(reshape_74, [0, 2, 1, 3])
        del reshape_74

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_148 = paddle._C_ops.matmul(transpose_72, transpose_73, False, True)
        del transpose_72, transpose_73

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(matmul_148, full_3, float("0"), True)
        del matmul_148

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_168 = paddle._C_ops.add(scale_20, scale_1)
        del scale_20

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_18 = paddle._C_ops.softmax(add_168, -1)
        del add_168

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_74, dropout_75 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_18, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_18

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_149 = paddle._C_ops.matmul(dropout_74, transpose_74, False, False)
        del dropout_74, transpose_74

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_75 = paddle._C_ops.transpose(matmul_149, [0, 2, 1, 3])
        del matmul_149

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_75 = paddle._C_ops.reshape(transpose_75, full_int_array_5)
        del transpose_75

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_150 = paddle._C_ops.matmul(reshape_75, parameter_10, False, False)
        del reshape_75

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_169 = paddle._C_ops.add(matmul_150, parameter_9)
        del matmul_150

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_76, dropout_77 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_169, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_169

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_170 = paddle._C_ops.add(layer_norm_108, dropout_76)
        del dropout_76, layer_norm_108

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_170, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_170

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_151 = paddle._C_ops.matmul(layer_norm_111, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_171 = paddle._C_ops.add(matmul_151, parameter_5)
        del matmul_151

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_18 = paddle._C_ops.gelu(add_171, False)
        del add_171

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_152 = paddle._C_ops.matmul(gelu_18, parameter_4, False, False)
        del gelu_18

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_172 = paddle._C_ops.add(matmul_152, parameter_3)
        del matmul_152

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_173 = paddle._C_ops.add(add_172, layer_norm_111)
        del add_172, layer_norm_111

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_173, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_173

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_153 = paddle._C_ops.matmul(layer_norm_114, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_174 = paddle._C_ops.add(matmul_153, parameter_15)
        del matmul_153

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_154 = paddle._C_ops.matmul(layer_norm_114, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_175 = paddle._C_ops.add(matmul_154, parameter_13)
        del matmul_154

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_155 = paddle._C_ops.matmul(layer_norm_114, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_176 = paddle._C_ops.add(matmul_155, parameter_11)
        del matmul_155

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_76 = paddle._C_ops.reshape(add_174, full_int_array_4)
        del add_174

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_76 = paddle._C_ops.transpose(reshape_76, [0, 2, 1, 3])
        del reshape_76

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(add_175, full_int_array_4)
        del add_175

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_77 = paddle._C_ops.transpose(reshape_77, [0, 2, 1, 3])
        del reshape_77

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_78 = paddle._C_ops.reshape(add_176, full_int_array_4)
        del add_176

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_78 = paddle._C_ops.transpose(reshape_78, [0, 2, 1, 3])
        del reshape_78

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_156 = paddle._C_ops.matmul(transpose_76, transpose_77, False, True)
        del transpose_76, transpose_77

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(matmul_156, full_3, float("0"), True)
        del matmul_156

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_177 = paddle._C_ops.add(scale_21, scale_1)
        del scale_21

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_19 = paddle._C_ops.softmax(add_177, -1)
        del add_177

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_78, dropout_79 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_19, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_19

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_157 = paddle._C_ops.matmul(dropout_78, transpose_78, False, False)
        del dropout_78, transpose_78

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_79 = paddle._C_ops.transpose(matmul_157, [0, 2, 1, 3])
        del matmul_157

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_79 = paddle._C_ops.reshape(transpose_79, full_int_array_5)
        del transpose_79

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_158 = paddle._C_ops.matmul(reshape_79, parameter_10, False, False)
        del reshape_79

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_178 = paddle._C_ops.add(matmul_158, parameter_9)
        del matmul_158

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_80, dropout_81 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_178, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_178

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_179 = paddle._C_ops.add(layer_norm_114, dropout_80)
        del dropout_80, layer_norm_114

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_179, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_179

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_159 = paddle._C_ops.matmul(layer_norm_117, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_180 = paddle._C_ops.add(matmul_159, parameter_5)
        del matmul_159

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_19 = paddle._C_ops.gelu(add_180, False)
        del add_180

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_160 = paddle._C_ops.matmul(gelu_19, parameter_4, False, False)
        del gelu_19

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_181 = paddle._C_ops.add(matmul_160, parameter_3)
        del matmul_160

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_182 = paddle._C_ops.add(add_181, layer_norm_117)
        del add_181, layer_norm_117

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_182, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_182

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_161 = paddle._C_ops.matmul(layer_norm_120, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_183 = paddle._C_ops.add(matmul_161, parameter_15)
        del matmul_161

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_162 = paddle._C_ops.matmul(layer_norm_120, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_184 = paddle._C_ops.add(matmul_162, parameter_13)
        del matmul_162

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_163 = paddle._C_ops.matmul(layer_norm_120, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_185 = paddle._C_ops.add(matmul_163, parameter_11)
        del matmul_163

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_80 = paddle._C_ops.reshape(add_183, full_int_array_4)
        del add_183

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_80 = paddle._C_ops.transpose(reshape_80, [0, 2, 1, 3])
        del reshape_80

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_81 = paddle._C_ops.reshape(add_184, full_int_array_4)
        del add_184

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_81 = paddle._C_ops.transpose(reshape_81, [0, 2, 1, 3])
        del reshape_81

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_82 = paddle._C_ops.reshape(add_185, full_int_array_4)
        del add_185

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_82 = paddle._C_ops.transpose(reshape_82, [0, 2, 1, 3])
        del reshape_82

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_164 = paddle._C_ops.matmul(transpose_80, transpose_81, False, True)
        del transpose_80, transpose_81

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(matmul_164, full_3, float("0"), True)
        del matmul_164

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_186 = paddle._C_ops.add(scale_22, scale_1)
        del scale_22

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_20 = paddle._C_ops.softmax(add_186, -1)
        del add_186

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_82, dropout_83 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_20, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_20

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_165 = paddle._C_ops.matmul(dropout_82, transpose_82, False, False)
        del dropout_82, transpose_82

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_83 = paddle._C_ops.transpose(matmul_165, [0, 2, 1, 3])
        del matmul_165

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_83 = paddle._C_ops.reshape(transpose_83, full_int_array_5)
        del transpose_83

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_166 = paddle._C_ops.matmul(reshape_83, parameter_10, False, False)
        del reshape_83

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_187 = paddle._C_ops.add(matmul_166, parameter_9)
        del matmul_166

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_84, dropout_85 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_187, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_187

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_188 = paddle._C_ops.add(layer_norm_120, dropout_84)
        del dropout_84, layer_norm_120

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_188, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_188

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_167 = paddle._C_ops.matmul(layer_norm_123, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_189 = paddle._C_ops.add(matmul_167, parameter_5)
        del matmul_167

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_20 = paddle._C_ops.gelu(add_189, False)
        del add_189

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_168 = paddle._C_ops.matmul(gelu_20, parameter_4, False, False)
        del gelu_20

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_190 = paddle._C_ops.add(matmul_168, parameter_3)
        del matmul_168

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_191 = paddle._C_ops.add(add_190, layer_norm_123)
        del add_190, layer_norm_123

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_191, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_191

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_169 = paddle._C_ops.matmul(layer_norm_126, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_192 = paddle._C_ops.add(matmul_169, parameter_15)
        del matmul_169

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_170 = paddle._C_ops.matmul(layer_norm_126, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_193 = paddle._C_ops.add(matmul_170, parameter_13)
        del matmul_170

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_171 = paddle._C_ops.matmul(layer_norm_126, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_194 = paddle._C_ops.add(matmul_171, parameter_11)
        del matmul_171

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_84 = paddle._C_ops.reshape(add_192, full_int_array_4)
        del add_192

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_84 = paddle._C_ops.transpose(reshape_84, [0, 2, 1, 3])
        del reshape_84

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_85 = paddle._C_ops.reshape(add_193, full_int_array_4)
        del add_193

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_85 = paddle._C_ops.transpose(reshape_85, [0, 2, 1, 3])
        del reshape_85

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_86 = paddle._C_ops.reshape(add_194, full_int_array_4)
        del add_194

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_86 = paddle._C_ops.transpose(reshape_86, [0, 2, 1, 3])
        del reshape_86

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_172 = paddle._C_ops.matmul(transpose_84, transpose_85, False, True)
        del transpose_84, transpose_85

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(matmul_172, full_3, float("0"), True)
        del matmul_172

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_195 = paddle._C_ops.add(scale_23, scale_1)
        del scale_23

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_21 = paddle._C_ops.softmax(add_195, -1)
        del add_195

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_86, dropout_87 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_21, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_21

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_173 = paddle._C_ops.matmul(dropout_86, transpose_86, False, False)
        del dropout_86, transpose_86

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_87 = paddle._C_ops.transpose(matmul_173, [0, 2, 1, 3])
        del matmul_173

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_87 = paddle._C_ops.reshape(transpose_87, full_int_array_5)
        del transpose_87

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_174 = paddle._C_ops.matmul(reshape_87, parameter_10, False, False)
        del reshape_87

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_196 = paddle._C_ops.add(matmul_174, parameter_9)
        del matmul_174

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_88, dropout_89 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_196, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_196

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_197 = paddle._C_ops.add(layer_norm_126, dropout_88)
        del dropout_88, layer_norm_126

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_197, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_197

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_175 = paddle._C_ops.matmul(layer_norm_129, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_198 = paddle._C_ops.add(matmul_175, parameter_5)
        del matmul_175

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_21 = paddle._C_ops.gelu(add_198, False)
        del add_198

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_176 = paddle._C_ops.matmul(gelu_21, parameter_4, False, False)
        del gelu_21

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_199 = paddle._C_ops.add(matmul_176, parameter_3)
        del matmul_176

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_200 = paddle._C_ops.add(add_199, layer_norm_129)
        del add_199, layer_norm_129

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_200, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_200

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_177 = paddle._C_ops.matmul(layer_norm_132, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_201 = paddle._C_ops.add(matmul_177, parameter_15)
        del matmul_177

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_178 = paddle._C_ops.matmul(layer_norm_132, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_202 = paddle._C_ops.add(matmul_178, parameter_13)
        del matmul_178

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_179 = paddle._C_ops.matmul(layer_norm_132, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_203 = paddle._C_ops.add(matmul_179, parameter_11)
        del matmul_179

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_88 = paddle._C_ops.reshape(add_201, full_int_array_4)
        del add_201

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_88 = paddle._C_ops.transpose(reshape_88, [0, 2, 1, 3])
        del reshape_88

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_89 = paddle._C_ops.reshape(add_202, full_int_array_4)
        del add_202

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_89 = paddle._C_ops.transpose(reshape_89, [0, 2, 1, 3])
        del reshape_89

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_90 = paddle._C_ops.reshape(add_203, full_int_array_4)
        del add_203

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_90 = paddle._C_ops.transpose(reshape_90, [0, 2, 1, 3])
        del reshape_90

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_180 = paddle._C_ops.matmul(transpose_88, transpose_89, False, True)
        del transpose_88, transpose_89

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(matmul_180, full_3, float("0"), True)
        del matmul_180

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_204 = paddle._C_ops.add(scale_24, scale_1)
        del scale_24

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_22 = paddle._C_ops.softmax(add_204, -1)
        del add_204

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_90, dropout_91 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_22, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_22

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_181 = paddle._C_ops.matmul(dropout_90, transpose_90, False, False)
        del dropout_90, transpose_90

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_91 = paddle._C_ops.transpose(matmul_181, [0, 2, 1, 3])
        del matmul_181

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_91 = paddle._C_ops.reshape(transpose_91, full_int_array_5)
        del transpose_91

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_182 = paddle._C_ops.matmul(reshape_91, parameter_10, False, False)
        del reshape_91

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_205 = paddle._C_ops.add(matmul_182, parameter_9)
        del matmul_182

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_92, dropout_93 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_205, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_205

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_206 = paddle._C_ops.add(layer_norm_132, dropout_92)
        del dropout_92, layer_norm_132

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_206, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_206

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_183 = paddle._C_ops.matmul(layer_norm_135, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_207 = paddle._C_ops.add(matmul_183, parameter_5)
        del matmul_183

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_22 = paddle._C_ops.gelu(add_207, False)
        del add_207

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_184 = paddle._C_ops.matmul(gelu_22, parameter_4, False, False)
        del gelu_22

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_208 = paddle._C_ops.add(matmul_184, parameter_3)
        del matmul_184

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_209 = paddle._C_ops.add(add_208, layer_norm_135)
        del add_208, layer_norm_135

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_209, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_209

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_185 = paddle._C_ops.matmul(layer_norm_138, parameter_16, False, False)
        del parameter_16

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_210 = paddle._C_ops.add(matmul_185, parameter_15)
        del matmul_185, parameter_15

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_186 = paddle._C_ops.matmul(layer_norm_138, parameter_14, False, False)
        del parameter_14

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_211 = paddle._C_ops.add(matmul_186, parameter_13)
        del matmul_186, parameter_13

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_187 = paddle._C_ops.matmul(layer_norm_138, parameter_12, False, False)
        del parameter_12

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_212 = paddle._C_ops.add(matmul_187, parameter_11)
        del matmul_187, parameter_11

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_92 = paddle._C_ops.reshape(add_210, full_int_array_4)
        del add_210

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_92 = paddle._C_ops.transpose(reshape_92, [0, 2, 1, 3])
        del reshape_92

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_93 = paddle._C_ops.reshape(add_211, full_int_array_4)
        del add_211

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_93 = paddle._C_ops.transpose(reshape_93, [0, 2, 1, 3])
        del reshape_93

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_94 = paddle._C_ops.reshape(add_212, full_int_array_4)
        del add_212, full_int_array_4

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_94 = paddle._C_ops.transpose(reshape_94, [0, 2, 1, 3])
        del reshape_94

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_188 = paddle._C_ops.matmul(transpose_92, transpose_93, False, True)
        del transpose_92, transpose_93

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(matmul_188, full_3, float("0"), True)
        del full_3, matmul_188

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_213 = paddle._C_ops.add(scale_25, scale_1)
        del scale_1, scale_25

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_23 = paddle._C_ops.softmax(add_213, -1)
        del add_213

        # pd_op.dropout: (1x16x21x21xf32, 1x16x21x21xui8) <- (1x16x21x21xf32, None, 1xf32)
        dropout_94, dropout_95 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_23, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_23

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_189 = paddle._C_ops.matmul(dropout_94, transpose_94, False, False)
        del dropout_94, transpose_94

        # pd_op.transpose: (1x21x16x128xf32) <- (1x16x21x128xf32)
        transpose_95 = paddle._C_ops.transpose(matmul_189, [0, 2, 1, 3])
        del matmul_189

        # pd_op.reshape: (1x21x2048xf32) <- (1x21x16x128xf32, 3xi64)
        reshape_95 = paddle._C_ops.reshape(transpose_95, full_int_array_5)
        del full_int_array_5, transpose_95

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_190 = paddle._C_ops.matmul(reshape_95, parameter_10, False, False)
        del parameter_10, reshape_95

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_214 = paddle._C_ops.add(matmul_190, parameter_9)
        del matmul_190, parameter_9

        # pd_op.dropout: (1x21x2048xf32, 1x21x2048xui8) <- (1x21x2048xf32, None, 1xf32)
        dropout_96, dropout_97 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_214, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_214, full_2

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_215 = paddle._C_ops.add(layer_norm_138, dropout_96)
        del dropout_96, layer_norm_138

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_215, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_215, parameter_7, parameter_8

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_191 = paddle._C_ops.matmul(layer_norm_141, parameter_6, False, False)
        del parameter_6

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_216 = paddle._C_ops.add(matmul_191, parameter_5)
        del matmul_191, parameter_5

        # pd_op.gelu: (1x21x8192xf32) <- (1x21x8192xf32)
        gelu_23 = paddle._C_ops.gelu(add_216, False)
        del add_216

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_192 = paddle._C_ops.matmul(gelu_23, parameter_4, False, False)
        del gelu_23, parameter_4

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_217 = paddle._C_ops.add(matmul_192, parameter_3)
        del matmul_192, parameter_3

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_218 = paddle._C_ops.add(add_217, layer_norm_141)
        del add_217, layer_norm_141

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_218, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_218, parameter_17, parameter_18

        # pd_op.slice: (1x2048xf32) <- (1x21x2048xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            layer_norm_144, [1], full_int_array_2, full_int_array_0, [1], [1]
        )
        del full_int_array_0, full_int_array_2

        # pd_op.matmul: (1x2048xf32) <- (1x2048xf32, 2048x2048xf32)
        matmul_193 = paddle._C_ops.matmul(slice_1, parameter_2, False, False)
        del parameter_2, slice_1

        # pd_op.add: (1x2048xf32) <- (1x2048xf32, 2048xf32)
        add_219 = paddle._C_ops.add(matmul_193, parameter_1)
        del matmul_193, parameter_1

        # pd_op.tanh: (1x2048xf32) <- (1x2048xf32)
        tanh_0 = paddle._C_ops.tanh(add_219)
        del add_219, layer_norm_144

        return tanh_0
