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

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x128xf32, 128x2048xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_20, False, False)
        del layer_norm_0, parameter_20

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
        full_2 = paddle._C_ops.full(
            [1], float("0.0883883"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_4, full_2, float("0"), True)
        del matmul_4

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_6 = paddle._C_ops.add(scale_2, scale_1)
        del scale_2

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_6, -1)
        del add_6

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_5 = paddle._C_ops.matmul(softmax_0, transpose_2, False, False)
        del softmax_0, transpose_2

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

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_8 = paddle._C_ops.add(add_2, add_7)
        del add_2, add_7

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

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_9, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_0 = paddle._C_ops.pow(add_9, float("3"))

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.044715"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(pow_0, full_4, float("0"), True)
        del pow_0

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_10 = paddle._C_ops.add(add_9, scale_4)
        del add_9, scale_4

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.797885"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(add_10, full_5, float("0"), True)
        del add_10

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_1 = paddle._C_ops.tanh(scale_5)
        del scale_5

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(tanh_1, full_6, float("1"), True)
        del tanh_1

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_0 = paddle._C_ops.multiply(scale_3, scale_6)
        del scale_3, scale_6

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_8 = paddle._C_ops.matmul(multiply_0, parameter_4, False, False)
        del multiply_0

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_11 = paddle._C_ops.add(matmul_8, parameter_3)
        del matmul_8

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_12 = paddle._C_ops.add(add_11, layer_norm_3)
        del add_11, layer_norm_3

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_12

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_13 = paddle._C_ops.add(matmul_9, parameter_15)
        del matmul_9

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_14 = paddle._C_ops.add(matmul_10, parameter_13)
        del matmul_10

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_6, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_15 = paddle._C_ops.add(matmul_11, parameter_11)
        del matmul_11

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_13, full_int_array_4)
        del add_13

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_14, full_int_array_4)
        del add_14

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_15, full_int_array_4)
        del add_15

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_4, transpose_5, False, True)
        del transpose_4, transpose_5

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_12, full_2, float("0"), True)
        del matmul_12

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_16 = paddle._C_ops.add(scale_7, scale_1)
        del scale_7

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_16, -1)
        del add_16

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_13 = paddle._C_ops.matmul(softmax_1, transpose_6, False, False)
        del softmax_1, transpose_6

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
        add_17 = paddle._C_ops.add(matmul_14, parameter_9)
        del matmul_14

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_18 = paddle._C_ops.add(layer_norm_6, add_17)
        del add_17, layer_norm_6

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_18

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_9, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_19 = paddle._C_ops.add(matmul_15, parameter_5)
        del matmul_15

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(add_19, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_1 = paddle._C_ops.pow(add_19, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(pow_1, full_4, float("0"), True)
        del pow_1

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_20 = paddle._C_ops.add(add_19, scale_9)
        del add_19, scale_9

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(add_20, full_5, float("0"), True)
        del add_20

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_2 = paddle._C_ops.tanh(scale_10)
        del scale_10

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(tanh_2, full_6, float("1"), True)
        del tanh_2

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_1 = paddle._C_ops.multiply(scale_8, scale_11)
        del scale_11, scale_8

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_16 = paddle._C_ops.matmul(multiply_1, parameter_4, False, False)
        del multiply_1

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_21 = paddle._C_ops.add(matmul_16, parameter_3)
        del matmul_16

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_22 = paddle._C_ops.add(add_21, layer_norm_9)
        del add_21, layer_norm_9

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_22

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_23 = paddle._C_ops.add(matmul_17, parameter_15)
        del matmul_17

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_24 = paddle._C_ops.add(matmul_18, parameter_13)
        del matmul_18

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_12, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_25 = paddle._C_ops.add(matmul_19, parameter_11)
        del matmul_19

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_23, full_int_array_4)
        del add_23

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_24, full_int_array_4)
        del add_24

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_25, full_int_array_4)
        del add_25

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_8, transpose_9, False, True)
        del transpose_8, transpose_9

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(matmul_20, full_2, float("0"), True)
        del matmul_20

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_26 = paddle._C_ops.add(scale_12, scale_1)
        del scale_12

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_26, -1)
        del add_26

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_2, transpose_10, False, False)
        del softmax_2, transpose_10

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
        add_27 = paddle._C_ops.add(matmul_22, parameter_9)
        del matmul_22

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_28 = paddle._C_ops.add(layer_norm_12, add_27)
        del add_27, layer_norm_12

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_28

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_15, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_29 = paddle._C_ops.add(matmul_23, parameter_5)
        del matmul_23

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(add_29, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_2 = paddle._C_ops.pow(add_29, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(pow_2, full_4, float("0"), True)
        del pow_2

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_30 = paddle._C_ops.add(add_29, scale_14)
        del add_29, scale_14

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(add_30, full_5, float("0"), True)
        del add_30

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_3 = paddle._C_ops.tanh(scale_15)
        del scale_15

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(tanh_3, full_6, float("1"), True)
        del tanh_3

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_2 = paddle._C_ops.multiply(scale_13, scale_16)
        del scale_13, scale_16

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_24 = paddle._C_ops.matmul(multiply_2, parameter_4, False, False)
        del multiply_2

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_31 = paddle._C_ops.add(matmul_24, parameter_3)
        del matmul_24

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_32 = paddle._C_ops.add(add_31, layer_norm_15)
        del add_31, layer_norm_15

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_32, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_32

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_33 = paddle._C_ops.add(matmul_25, parameter_15)
        del matmul_25

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_34 = paddle._C_ops.add(matmul_26, parameter_13)
        del matmul_26

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_27 = paddle._C_ops.matmul(layer_norm_18, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_35 = paddle._C_ops.add(matmul_27, parameter_11)
        del matmul_27

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_33, full_int_array_4)
        del add_33

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_34, full_int_array_4)
        del add_34

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_35, full_int_array_4)
        del add_35

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_28 = paddle._C_ops.matmul(transpose_12, transpose_13, False, True)
        del transpose_12, transpose_13

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(matmul_28, full_2, float("0"), True)
        del matmul_28

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_36 = paddle._C_ops.add(scale_17, scale_1)
        del scale_17

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_36, -1)
        del add_36

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_29 = paddle._C_ops.matmul(softmax_3, transpose_14, False, False)
        del softmax_3, transpose_14

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
        add_37 = paddle._C_ops.add(matmul_30, parameter_9)
        del matmul_30

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_38 = paddle._C_ops.add(layer_norm_18, add_37)
        del add_37, layer_norm_18

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_38

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_21, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_39 = paddle._C_ops.add(matmul_31, parameter_5)
        del matmul_31

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(add_39, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_3 = paddle._C_ops.pow(add_39, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(pow_3, full_4, float("0"), True)
        del pow_3

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_40 = paddle._C_ops.add(add_39, scale_19)
        del add_39, scale_19

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(add_40, full_5, float("0"), True)
        del add_40

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_4 = paddle._C_ops.tanh(scale_20)
        del scale_20

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(tanh_4, full_6, float("1"), True)
        del tanh_4

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_3 = paddle._C_ops.multiply(scale_18, scale_21)
        del scale_18, scale_21

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_32 = paddle._C_ops.matmul(multiply_3, parameter_4, False, False)
        del multiply_3

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_41 = paddle._C_ops.add(matmul_32, parameter_3)
        del matmul_32

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_42 = paddle._C_ops.add(add_41, layer_norm_21)
        del add_41, layer_norm_21

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_42

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_43 = paddle._C_ops.add(matmul_33, parameter_15)
        del matmul_33

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_44 = paddle._C_ops.add(matmul_34, parameter_13)
        del matmul_34

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_24, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_45 = paddle._C_ops.add(matmul_35, parameter_11)
        del matmul_35

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_43, full_int_array_4)
        del add_43

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_44, full_int_array_4)
        del add_44

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_45, full_int_array_4)
        del add_45

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_36 = paddle._C_ops.matmul(transpose_16, transpose_17, False, True)
        del transpose_16, transpose_17

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(matmul_36, full_2, float("0"), True)
        del matmul_36

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_46 = paddle._C_ops.add(scale_22, scale_1)
        del scale_22

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(add_46, -1)
        del add_46

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_37 = paddle._C_ops.matmul(softmax_4, transpose_18, False, False)
        del softmax_4, transpose_18

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
        add_47 = paddle._C_ops.add(matmul_38, parameter_9)
        del matmul_38

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_48 = paddle._C_ops.add(layer_norm_24, add_47)
        del add_47, layer_norm_24

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_48, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_48

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_39 = paddle._C_ops.matmul(layer_norm_27, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_49 = paddle._C_ops.add(matmul_39, parameter_5)
        del matmul_39

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(add_49, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_4 = paddle._C_ops.pow(add_49, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(pow_4, full_4, float("0"), True)
        del pow_4

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_50 = paddle._C_ops.add(add_49, scale_24)
        del add_49, scale_24

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(add_50, full_5, float("0"), True)
        del add_50

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_5 = paddle._C_ops.tanh(scale_25)
        del scale_25

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(tanh_5, full_6, float("1"), True)
        del tanh_5

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_4 = paddle._C_ops.multiply(scale_23, scale_26)
        del scale_23, scale_26

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_40 = paddle._C_ops.matmul(multiply_4, parameter_4, False, False)
        del multiply_4

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_51 = paddle._C_ops.add(matmul_40, parameter_3)
        del matmul_40

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_52 = paddle._C_ops.add(add_51, layer_norm_27)
        del add_51, layer_norm_27

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_52

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_53 = paddle._C_ops.add(matmul_41, parameter_15)
        del matmul_41

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_54 = paddle._C_ops.add(matmul_42, parameter_13)
        del matmul_42

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_43 = paddle._C_ops.matmul(layer_norm_30, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_55 = paddle._C_ops.add(matmul_43, parameter_11)
        del matmul_43

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_53, full_int_array_4)
        del add_53

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_54, full_int_array_4)
        del add_54

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_55, full_int_array_4)
        del add_55

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_44 = paddle._C_ops.matmul(transpose_20, transpose_21, False, True)
        del transpose_20, transpose_21

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(matmul_44, full_2, float("0"), True)
        del matmul_44

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_56 = paddle._C_ops.add(scale_27, scale_1)
        del scale_27

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_56, -1)
        del add_56

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_45 = paddle._C_ops.matmul(softmax_5, transpose_22, False, False)
        del softmax_5, transpose_22

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
        add_57 = paddle._C_ops.add(matmul_46, parameter_9)
        del matmul_46

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_58 = paddle._C_ops.add(layer_norm_30, add_57)
        del add_57, layer_norm_30

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_58

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_33, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_59 = paddle._C_ops.add(matmul_47, parameter_5)
        del matmul_47

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(add_59, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_5 = paddle._C_ops.pow(add_59, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(pow_5, full_4, float("0"), True)
        del pow_5

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_60 = paddle._C_ops.add(add_59, scale_29)
        del add_59, scale_29

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(add_60, full_5, float("0"), True)
        del add_60

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_6 = paddle._C_ops.tanh(scale_30)
        del scale_30

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(tanh_6, full_6, float("1"), True)
        del tanh_6

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_5 = paddle._C_ops.multiply(scale_28, scale_31)
        del scale_28, scale_31

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_48 = paddle._C_ops.matmul(multiply_5, parameter_4, False, False)
        del multiply_5

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_61 = paddle._C_ops.add(matmul_48, parameter_3)
        del matmul_48

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_62 = paddle._C_ops.add(add_61, layer_norm_33)
        del add_61, layer_norm_33

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_62, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_62

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_36, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_63 = paddle._C_ops.add(matmul_49, parameter_15)
        del matmul_49

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_36, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_64 = paddle._C_ops.add(matmul_50, parameter_13)
        del matmul_50

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_51 = paddle._C_ops.matmul(layer_norm_36, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_65 = paddle._C_ops.add(matmul_51, parameter_11)
        del matmul_51

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_63, full_int_array_4)
        del add_63

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_64, full_int_array_4)
        del add_64

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_65, full_int_array_4)
        del add_65

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_52 = paddle._C_ops.matmul(transpose_24, transpose_25, False, True)
        del transpose_24, transpose_25

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(matmul_52, full_2, float("0"), True)
        del matmul_52

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_66 = paddle._C_ops.add(scale_32, scale_1)
        del scale_32

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_6 = paddle._C_ops.softmax(add_66, -1)
        del add_66

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_53 = paddle._C_ops.matmul(softmax_6, transpose_26, False, False)
        del softmax_6, transpose_26

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
        add_67 = paddle._C_ops.add(matmul_54, parameter_9)
        del matmul_54

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_68 = paddle._C_ops.add(layer_norm_36, add_67)
        del add_67, layer_norm_36

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_68, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_68

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_39, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_69 = paddle._C_ops.add(matmul_55, parameter_5)
        del matmul_55

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(add_69, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_6 = paddle._C_ops.pow(add_69, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(pow_6, full_4, float("0"), True)
        del pow_6

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_70 = paddle._C_ops.add(add_69, scale_34)
        del add_69, scale_34

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(add_70, full_5, float("0"), True)
        del add_70

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_7 = paddle._C_ops.tanh(scale_35)
        del scale_35

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(tanh_7, full_6, float("1"), True)
        del tanh_7

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_6 = paddle._C_ops.multiply(scale_33, scale_36)
        del scale_33, scale_36

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_56 = paddle._C_ops.matmul(multiply_6, parameter_4, False, False)
        del multiply_6

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_71 = paddle._C_ops.add(matmul_56, parameter_3)
        del matmul_56

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_72 = paddle._C_ops.add(add_71, layer_norm_39)
        del add_71, layer_norm_39

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_72, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_72

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_57 = paddle._C_ops.matmul(layer_norm_42, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_73 = paddle._C_ops.add(matmul_57, parameter_15)
        del matmul_57

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_42, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_74 = paddle._C_ops.add(matmul_58, parameter_13)
        del matmul_58

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_59 = paddle._C_ops.matmul(layer_norm_42, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_75 = paddle._C_ops.add(matmul_59, parameter_11)
        del matmul_59

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_73, full_int_array_4)
        del add_73

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_74, full_int_array_4)
        del add_74

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_75, full_int_array_4)
        del add_75

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_60 = paddle._C_ops.matmul(transpose_28, transpose_29, False, True)
        del transpose_28, transpose_29

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(matmul_60, full_2, float("0"), True)
        del matmul_60

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_76 = paddle._C_ops.add(scale_37, scale_1)
        del scale_37

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_7 = paddle._C_ops.softmax(add_76, -1)
        del add_76

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_61 = paddle._C_ops.matmul(softmax_7, transpose_30, False, False)
        del softmax_7, transpose_30

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
        add_77 = paddle._C_ops.add(matmul_62, parameter_9)
        del matmul_62

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_78 = paddle._C_ops.add(layer_norm_42, add_77)
        del add_77, layer_norm_42

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_78, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_78

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_63 = paddle._C_ops.matmul(layer_norm_45, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_79 = paddle._C_ops.add(matmul_63, parameter_5)
        del matmul_63

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(add_79, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_7 = paddle._C_ops.pow(add_79, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(pow_7, full_4, float("0"), True)
        del pow_7

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_80 = paddle._C_ops.add(add_79, scale_39)
        del add_79, scale_39

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_40 = paddle._C_ops.scale(add_80, full_5, float("0"), True)
        del add_80

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_8 = paddle._C_ops.tanh(scale_40)
        del scale_40

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_41 = paddle._C_ops.scale(tanh_8, full_6, float("1"), True)
        del tanh_8

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_7 = paddle._C_ops.multiply(scale_38, scale_41)
        del scale_38, scale_41

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_64 = paddle._C_ops.matmul(multiply_7, parameter_4, False, False)
        del multiply_7

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_81 = paddle._C_ops.add(matmul_64, parameter_3)
        del matmul_64

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_82 = paddle._C_ops.add(add_81, layer_norm_45)
        del add_81, layer_norm_45

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_82

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_65 = paddle._C_ops.matmul(layer_norm_48, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_83 = paddle._C_ops.add(matmul_65, parameter_15)
        del matmul_65

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_48, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_84 = paddle._C_ops.add(matmul_66, parameter_13)
        del matmul_66

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_48, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_85 = paddle._C_ops.add(matmul_67, parameter_11)
        del matmul_67

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_83, full_int_array_4)
        del add_83

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(add_84, full_int_array_4)
        del add_84

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_85, full_int_array_4)
        del add_85

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_68 = paddle._C_ops.matmul(transpose_32, transpose_33, False, True)
        del transpose_32, transpose_33

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_42 = paddle._C_ops.scale(matmul_68, full_2, float("0"), True)
        del matmul_68

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_86 = paddle._C_ops.add(scale_42, scale_1)
        del scale_42

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_8 = paddle._C_ops.softmax(add_86, -1)
        del add_86

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_69 = paddle._C_ops.matmul(softmax_8, transpose_34, False, False)
        del softmax_8, transpose_34

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
        add_87 = paddle._C_ops.add(matmul_70, parameter_9)
        del matmul_70

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_88 = paddle._C_ops.add(layer_norm_48, add_87)
        del add_87, layer_norm_48

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_88, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_88

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_71 = paddle._C_ops.matmul(layer_norm_51, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_89 = paddle._C_ops.add(matmul_71, parameter_5)
        del matmul_71

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_43 = paddle._C_ops.scale(add_89, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_8 = paddle._C_ops.pow(add_89, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_44 = paddle._C_ops.scale(pow_8, full_4, float("0"), True)
        del pow_8

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_90 = paddle._C_ops.add(add_89, scale_44)
        del add_89, scale_44

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_45 = paddle._C_ops.scale(add_90, full_5, float("0"), True)
        del add_90

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_9 = paddle._C_ops.tanh(scale_45)
        del scale_45

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_46 = paddle._C_ops.scale(tanh_9, full_6, float("1"), True)
        del tanh_9

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_8 = paddle._C_ops.multiply(scale_43, scale_46)
        del scale_43, scale_46

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_72 = paddle._C_ops.matmul(multiply_8, parameter_4, False, False)
        del multiply_8

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_91 = paddle._C_ops.add(matmul_72, parameter_3)
        del matmul_72

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_92 = paddle._C_ops.add(add_91, layer_norm_51)
        del add_91, layer_norm_51

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_92, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_92

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_54, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_93 = paddle._C_ops.add(matmul_73, parameter_15)
        del matmul_73

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_54, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_94 = paddle._C_ops.add(matmul_74, parameter_13)
        del matmul_74

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_75 = paddle._C_ops.matmul(layer_norm_54, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_95 = paddle._C_ops.add(matmul_75, parameter_11)
        del matmul_75

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_93, full_int_array_4)
        del add_93

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_94, full_int_array_4)
        del add_94

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_95, full_int_array_4)
        del add_95

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_76 = paddle._C_ops.matmul(transpose_36, transpose_37, False, True)
        del transpose_36, transpose_37

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_47 = paddle._C_ops.scale(matmul_76, full_2, float("0"), True)
        del matmul_76

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_96 = paddle._C_ops.add(scale_47, scale_1)
        del scale_47

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_9 = paddle._C_ops.softmax(add_96, -1)
        del add_96

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_77 = paddle._C_ops.matmul(softmax_9, transpose_38, False, False)
        del softmax_9, transpose_38

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
        add_97 = paddle._C_ops.add(matmul_78, parameter_9)
        del matmul_78

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_98 = paddle._C_ops.add(layer_norm_54, add_97)
        del add_97, layer_norm_54

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_98, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_98

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_79 = paddle._C_ops.matmul(layer_norm_57, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_99 = paddle._C_ops.add(matmul_79, parameter_5)
        del matmul_79

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_48 = paddle._C_ops.scale(add_99, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_9 = paddle._C_ops.pow(add_99, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_49 = paddle._C_ops.scale(pow_9, full_4, float("0"), True)
        del pow_9

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_100 = paddle._C_ops.add(add_99, scale_49)
        del add_99, scale_49

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_50 = paddle._C_ops.scale(add_100, full_5, float("0"), True)
        del add_100

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_10 = paddle._C_ops.tanh(scale_50)
        del scale_50

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_51 = paddle._C_ops.scale(tanh_10, full_6, float("1"), True)
        del tanh_10

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_9 = paddle._C_ops.multiply(scale_48, scale_51)
        del scale_48, scale_51

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_80 = paddle._C_ops.matmul(multiply_9, parameter_4, False, False)
        del multiply_9

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_101 = paddle._C_ops.add(matmul_80, parameter_3)
        del matmul_80

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_102 = paddle._C_ops.add(add_101, layer_norm_57)
        del add_101, layer_norm_57

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_102, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_102

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_60, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_103 = paddle._C_ops.add(matmul_81, parameter_15)
        del matmul_81

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_60, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_104 = paddle._C_ops.add(matmul_82, parameter_13)
        del matmul_82

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_83 = paddle._C_ops.matmul(layer_norm_60, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_105 = paddle._C_ops.add(matmul_83, parameter_11)
        del matmul_83

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_103, full_int_array_4)
        del add_103

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_104, full_int_array_4)
        del add_104

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_105, full_int_array_4)
        del add_105

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_84 = paddle._C_ops.matmul(transpose_40, transpose_41, False, True)
        del transpose_40, transpose_41

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_52 = paddle._C_ops.scale(matmul_84, full_2, float("0"), True)
        del matmul_84

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_106 = paddle._C_ops.add(scale_52, scale_1)
        del scale_52

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_10 = paddle._C_ops.softmax(add_106, -1)
        del add_106

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_85 = paddle._C_ops.matmul(softmax_10, transpose_42, False, False)
        del softmax_10, transpose_42

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
        add_107 = paddle._C_ops.add(matmul_86, parameter_9)
        del matmul_86

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_108 = paddle._C_ops.add(layer_norm_60, add_107)
        del add_107, layer_norm_60

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_108, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_108

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_87 = paddle._C_ops.matmul(layer_norm_63, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_109 = paddle._C_ops.add(matmul_87, parameter_5)
        del matmul_87

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_53 = paddle._C_ops.scale(add_109, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_10 = paddle._C_ops.pow(add_109, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_54 = paddle._C_ops.scale(pow_10, full_4, float("0"), True)
        del pow_10

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_110 = paddle._C_ops.add(add_109, scale_54)
        del add_109, scale_54

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_55 = paddle._C_ops.scale(add_110, full_5, float("0"), True)
        del add_110

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_11 = paddle._C_ops.tanh(scale_55)
        del scale_55

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_56 = paddle._C_ops.scale(tanh_11, full_6, float("1"), True)
        del tanh_11

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_10 = paddle._C_ops.multiply(scale_53, scale_56)
        del scale_53, scale_56

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_88 = paddle._C_ops.matmul(multiply_10, parameter_4, False, False)
        del multiply_10

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_111 = paddle._C_ops.add(matmul_88, parameter_3)
        del matmul_88

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_112 = paddle._C_ops.add(add_111, layer_norm_63)
        del add_111, layer_norm_63

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_112, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_112

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_66, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_113 = paddle._C_ops.add(matmul_89, parameter_15)
        del matmul_89

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_66, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_114 = paddle._C_ops.add(matmul_90, parameter_13)
        del matmul_90

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_66, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_115 = paddle._C_ops.add(matmul_91, parameter_11)
        del matmul_91

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(add_113, full_int_array_4)
        del add_113

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_114, full_int_array_4)
        del add_114

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_115, full_int_array_4)
        del add_115

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_92 = paddle._C_ops.matmul(transpose_44, transpose_45, False, True)
        del transpose_44, transpose_45

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_57 = paddle._C_ops.scale(matmul_92, full_2, float("0"), True)
        del matmul_92

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_116 = paddle._C_ops.add(scale_57, scale_1)
        del scale_57

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_11 = paddle._C_ops.softmax(add_116, -1)
        del add_116

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_93 = paddle._C_ops.matmul(softmax_11, transpose_46, False, False)
        del softmax_11, transpose_46

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
        add_117 = paddle._C_ops.add(matmul_94, parameter_9)
        del matmul_94

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_118 = paddle._C_ops.add(layer_norm_66, add_117)
        del add_117, layer_norm_66

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_118, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_118

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_95 = paddle._C_ops.matmul(layer_norm_69, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_119 = paddle._C_ops.add(matmul_95, parameter_5)
        del matmul_95

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_58 = paddle._C_ops.scale(add_119, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_11 = paddle._C_ops.pow(add_119, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_59 = paddle._C_ops.scale(pow_11, full_4, float("0"), True)
        del pow_11

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_120 = paddle._C_ops.add(add_119, scale_59)
        del add_119, scale_59

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_60 = paddle._C_ops.scale(add_120, full_5, float("0"), True)
        del add_120

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_12 = paddle._C_ops.tanh(scale_60)
        del scale_60

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_61 = paddle._C_ops.scale(tanh_12, full_6, float("1"), True)
        del tanh_12

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_11 = paddle._C_ops.multiply(scale_58, scale_61)
        del scale_58, scale_61

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_96 = paddle._C_ops.matmul(multiply_11, parameter_4, False, False)
        del multiply_11

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_121 = paddle._C_ops.add(matmul_96, parameter_3)
        del matmul_96

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_122 = paddle._C_ops.add(add_121, layer_norm_69)
        del add_121, layer_norm_69

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_122, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_122

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_97 = paddle._C_ops.matmul(layer_norm_72, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_123 = paddle._C_ops.add(matmul_97, parameter_15)
        del matmul_97

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_98 = paddle._C_ops.matmul(layer_norm_72, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_124 = paddle._C_ops.add(matmul_98, parameter_13)
        del matmul_98

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_99 = paddle._C_ops.matmul(layer_norm_72, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_125 = paddle._C_ops.add(matmul_99, parameter_11)
        del matmul_99

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(add_123, full_int_array_4)
        del add_123

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_48, [0, 2, 1, 3])
        del reshape_48

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(add_124, full_int_array_4)
        del add_124

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_49, [0, 2, 1, 3])
        del reshape_49

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(add_125, full_int_array_4)
        del add_125

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_50, [0, 2, 1, 3])
        del reshape_50

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_100 = paddle._C_ops.matmul(transpose_48, transpose_49, False, True)
        del transpose_48, transpose_49

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_62 = paddle._C_ops.scale(matmul_100, full_2, float("0"), True)
        del matmul_100

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_126 = paddle._C_ops.add(scale_62, scale_1)
        del scale_62

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_12 = paddle._C_ops.softmax(add_126, -1)
        del add_126

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_101 = paddle._C_ops.matmul(softmax_12, transpose_50, False, False)
        del softmax_12, transpose_50

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
        add_127 = paddle._C_ops.add(matmul_102, parameter_9)
        del matmul_102

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_128 = paddle._C_ops.add(layer_norm_72, add_127)
        del add_127, layer_norm_72

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_128, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_128

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_103 = paddle._C_ops.matmul(layer_norm_75, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_129 = paddle._C_ops.add(matmul_103, parameter_5)
        del matmul_103

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_63 = paddle._C_ops.scale(add_129, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_12 = paddle._C_ops.pow(add_129, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_64 = paddle._C_ops.scale(pow_12, full_4, float("0"), True)
        del pow_12

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_130 = paddle._C_ops.add(add_129, scale_64)
        del add_129, scale_64

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_65 = paddle._C_ops.scale(add_130, full_5, float("0"), True)
        del add_130

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_13 = paddle._C_ops.tanh(scale_65)
        del scale_65

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_66 = paddle._C_ops.scale(tanh_13, full_6, float("1"), True)
        del tanh_13

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_12 = paddle._C_ops.multiply(scale_63, scale_66)
        del scale_63, scale_66

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_104 = paddle._C_ops.matmul(multiply_12, parameter_4, False, False)
        del multiply_12

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_131 = paddle._C_ops.add(matmul_104, parameter_3)
        del matmul_104

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_132 = paddle._C_ops.add(add_131, layer_norm_75)
        del add_131, layer_norm_75

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_132, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_132

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_105 = paddle._C_ops.matmul(layer_norm_78, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_133 = paddle._C_ops.add(matmul_105, parameter_15)
        del matmul_105

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_106 = paddle._C_ops.matmul(layer_norm_78, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_134 = paddle._C_ops.add(matmul_106, parameter_13)
        del matmul_106

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_107 = paddle._C_ops.matmul(layer_norm_78, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_135 = paddle._C_ops.add(matmul_107, parameter_11)
        del matmul_107

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(add_133, full_int_array_4)
        del add_133

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_52, [0, 2, 1, 3])
        del reshape_52

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(add_134, full_int_array_4)
        del add_134

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_53, [0, 2, 1, 3])
        del reshape_53

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(add_135, full_int_array_4)
        del add_135

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_54 = paddle._C_ops.transpose(reshape_54, [0, 2, 1, 3])
        del reshape_54

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_108 = paddle._C_ops.matmul(transpose_52, transpose_53, False, True)
        del transpose_52, transpose_53

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_67 = paddle._C_ops.scale(matmul_108, full_2, float("0"), True)
        del matmul_108

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_136 = paddle._C_ops.add(scale_67, scale_1)
        del scale_67

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_13 = paddle._C_ops.softmax(add_136, -1)
        del add_136

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_109 = paddle._C_ops.matmul(softmax_13, transpose_54, False, False)
        del softmax_13, transpose_54

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
        add_137 = paddle._C_ops.add(matmul_110, parameter_9)
        del matmul_110

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_138 = paddle._C_ops.add(layer_norm_78, add_137)
        del add_137, layer_norm_78

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_138, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_138

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_111 = paddle._C_ops.matmul(layer_norm_81, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_139 = paddle._C_ops.add(matmul_111, parameter_5)
        del matmul_111

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_68 = paddle._C_ops.scale(add_139, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_13 = paddle._C_ops.pow(add_139, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_69 = paddle._C_ops.scale(pow_13, full_4, float("0"), True)
        del pow_13

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_140 = paddle._C_ops.add(add_139, scale_69)
        del add_139, scale_69

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_70 = paddle._C_ops.scale(add_140, full_5, float("0"), True)
        del add_140

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_14 = paddle._C_ops.tanh(scale_70)
        del scale_70

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_71 = paddle._C_ops.scale(tanh_14, full_6, float("1"), True)
        del tanh_14

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_13 = paddle._C_ops.multiply(scale_68, scale_71)
        del scale_68, scale_71

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_112 = paddle._C_ops.matmul(multiply_13, parameter_4, False, False)
        del multiply_13

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_141 = paddle._C_ops.add(matmul_112, parameter_3)
        del matmul_112

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_142 = paddle._C_ops.add(add_141, layer_norm_81)
        del add_141, layer_norm_81

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_142, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_142

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_113 = paddle._C_ops.matmul(layer_norm_84, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_143 = paddle._C_ops.add(matmul_113, parameter_15)
        del matmul_113

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_114 = paddle._C_ops.matmul(layer_norm_84, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_144 = paddle._C_ops.add(matmul_114, parameter_13)
        del matmul_114

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_115 = paddle._C_ops.matmul(layer_norm_84, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_145 = paddle._C_ops.add(matmul_115, parameter_11)
        del matmul_115

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(add_143, full_int_array_4)
        del add_143

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_56, [0, 2, 1, 3])
        del reshape_56

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(add_144, full_int_array_4)
        del add_144

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_57 = paddle._C_ops.transpose(reshape_57, [0, 2, 1, 3])
        del reshape_57

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(add_145, full_int_array_4)
        del add_145

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_58, [0, 2, 1, 3])
        del reshape_58

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_116 = paddle._C_ops.matmul(transpose_56, transpose_57, False, True)
        del transpose_56, transpose_57

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_72 = paddle._C_ops.scale(matmul_116, full_2, float("0"), True)
        del matmul_116

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_146 = paddle._C_ops.add(scale_72, scale_1)
        del scale_72

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_14 = paddle._C_ops.softmax(add_146, -1)
        del add_146

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_117 = paddle._C_ops.matmul(softmax_14, transpose_58, False, False)
        del softmax_14, transpose_58

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
        add_147 = paddle._C_ops.add(matmul_118, parameter_9)
        del matmul_118

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_148 = paddle._C_ops.add(layer_norm_84, add_147)
        del add_147, layer_norm_84

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_148, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_148

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_119 = paddle._C_ops.matmul(layer_norm_87, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_149 = paddle._C_ops.add(matmul_119, parameter_5)
        del matmul_119

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_73 = paddle._C_ops.scale(add_149, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_14 = paddle._C_ops.pow(add_149, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_74 = paddle._C_ops.scale(pow_14, full_4, float("0"), True)
        del pow_14

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_150 = paddle._C_ops.add(add_149, scale_74)
        del add_149, scale_74

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_75 = paddle._C_ops.scale(add_150, full_5, float("0"), True)
        del add_150

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_15 = paddle._C_ops.tanh(scale_75)
        del scale_75

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_76 = paddle._C_ops.scale(tanh_15, full_6, float("1"), True)
        del tanh_15

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_14 = paddle._C_ops.multiply(scale_73, scale_76)
        del scale_73, scale_76

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_120 = paddle._C_ops.matmul(multiply_14, parameter_4, False, False)
        del multiply_14

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_151 = paddle._C_ops.add(matmul_120, parameter_3)
        del matmul_120

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_152 = paddle._C_ops.add(add_151, layer_norm_87)
        del add_151, layer_norm_87

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_152, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_152

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_121 = paddle._C_ops.matmul(layer_norm_90, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_153 = paddle._C_ops.add(matmul_121, parameter_15)
        del matmul_121

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_122 = paddle._C_ops.matmul(layer_norm_90, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_154 = paddle._C_ops.add(matmul_122, parameter_13)
        del matmul_122

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_123 = paddle._C_ops.matmul(layer_norm_90, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_155 = paddle._C_ops.add(matmul_123, parameter_11)
        del matmul_123

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(add_153, full_int_array_4)
        del add_153

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_60 = paddle._C_ops.transpose(reshape_60, [0, 2, 1, 3])
        del reshape_60

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(add_154, full_int_array_4)
        del add_154

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_61, [0, 2, 1, 3])
        del reshape_61

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(add_155, full_int_array_4)
        del add_155

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_62, [0, 2, 1, 3])
        del reshape_62

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_124 = paddle._C_ops.matmul(transpose_60, transpose_61, False, True)
        del transpose_60, transpose_61

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_77 = paddle._C_ops.scale(matmul_124, full_2, float("0"), True)
        del matmul_124

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_156 = paddle._C_ops.add(scale_77, scale_1)
        del scale_77

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_15 = paddle._C_ops.softmax(add_156, -1)
        del add_156

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_125 = paddle._C_ops.matmul(softmax_15, transpose_62, False, False)
        del softmax_15, transpose_62

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
        add_157 = paddle._C_ops.add(matmul_126, parameter_9)
        del matmul_126

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_158 = paddle._C_ops.add(layer_norm_90, add_157)
        del add_157, layer_norm_90

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_158, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_158

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_127 = paddle._C_ops.matmul(layer_norm_93, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_159 = paddle._C_ops.add(matmul_127, parameter_5)
        del matmul_127

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_78 = paddle._C_ops.scale(add_159, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_15 = paddle._C_ops.pow(add_159, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_79 = paddle._C_ops.scale(pow_15, full_4, float("0"), True)
        del pow_15

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_160 = paddle._C_ops.add(add_159, scale_79)
        del add_159, scale_79

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_80 = paddle._C_ops.scale(add_160, full_5, float("0"), True)
        del add_160

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_16 = paddle._C_ops.tanh(scale_80)
        del scale_80

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_81 = paddle._C_ops.scale(tanh_16, full_6, float("1"), True)
        del tanh_16

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_15 = paddle._C_ops.multiply(scale_78, scale_81)
        del scale_78, scale_81

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_128 = paddle._C_ops.matmul(multiply_15, parameter_4, False, False)
        del multiply_15

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_161 = paddle._C_ops.add(matmul_128, parameter_3)
        del matmul_128

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_162 = paddle._C_ops.add(add_161, layer_norm_93)
        del add_161, layer_norm_93

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_162, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_162

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_129 = paddle._C_ops.matmul(layer_norm_96, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_163 = paddle._C_ops.add(matmul_129, parameter_15)
        del matmul_129

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_130 = paddle._C_ops.matmul(layer_norm_96, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_164 = paddle._C_ops.add(matmul_130, parameter_13)
        del matmul_130

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_131 = paddle._C_ops.matmul(layer_norm_96, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_165 = paddle._C_ops.add(matmul_131, parameter_11)
        del matmul_131

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(add_163, full_int_array_4)
        del add_163

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_64, [0, 2, 1, 3])
        del reshape_64

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(add_164, full_int_array_4)
        del add_164

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_65, [0, 2, 1, 3])
        del reshape_65

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(add_165, full_int_array_4)
        del add_165

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_66 = paddle._C_ops.transpose(reshape_66, [0, 2, 1, 3])
        del reshape_66

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_132 = paddle._C_ops.matmul(transpose_64, transpose_65, False, True)
        del transpose_64, transpose_65

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_82 = paddle._C_ops.scale(matmul_132, full_2, float("0"), True)
        del matmul_132

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_166 = paddle._C_ops.add(scale_82, scale_1)
        del scale_82

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_16 = paddle._C_ops.softmax(add_166, -1)
        del add_166

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_133 = paddle._C_ops.matmul(softmax_16, transpose_66, False, False)
        del softmax_16, transpose_66

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
        add_167 = paddle._C_ops.add(matmul_134, parameter_9)
        del matmul_134

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_168 = paddle._C_ops.add(layer_norm_96, add_167)
        del add_167, layer_norm_96

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_168, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_168

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_135 = paddle._C_ops.matmul(layer_norm_99, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_169 = paddle._C_ops.add(matmul_135, parameter_5)
        del matmul_135

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_83 = paddle._C_ops.scale(add_169, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_16 = paddle._C_ops.pow(add_169, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_84 = paddle._C_ops.scale(pow_16, full_4, float("0"), True)
        del pow_16

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_170 = paddle._C_ops.add(add_169, scale_84)
        del add_169, scale_84

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_85 = paddle._C_ops.scale(add_170, full_5, float("0"), True)
        del add_170

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_17 = paddle._C_ops.tanh(scale_85)
        del scale_85

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_86 = paddle._C_ops.scale(tanh_17, full_6, float("1"), True)
        del tanh_17

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_16 = paddle._C_ops.multiply(scale_83, scale_86)
        del scale_83, scale_86

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_136 = paddle._C_ops.matmul(multiply_16, parameter_4, False, False)
        del multiply_16

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_171 = paddle._C_ops.add(matmul_136, parameter_3)
        del matmul_136

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_172 = paddle._C_ops.add(add_171, layer_norm_99)
        del add_171, layer_norm_99

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_172, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_172

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_137 = paddle._C_ops.matmul(layer_norm_102, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_173 = paddle._C_ops.add(matmul_137, parameter_15)
        del matmul_137

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_138 = paddle._C_ops.matmul(layer_norm_102, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_174 = paddle._C_ops.add(matmul_138, parameter_13)
        del matmul_138

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_139 = paddle._C_ops.matmul(layer_norm_102, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_175 = paddle._C_ops.add(matmul_139, parameter_11)
        del matmul_139

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(add_173, full_int_array_4)
        del add_173

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_68 = paddle._C_ops.transpose(reshape_68, [0, 2, 1, 3])
        del reshape_68

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(add_174, full_int_array_4)
        del add_174

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_69 = paddle._C_ops.transpose(reshape_69, [0, 2, 1, 3])
        del reshape_69

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(add_175, full_int_array_4)
        del add_175

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_70, [0, 2, 1, 3])
        del reshape_70

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_140 = paddle._C_ops.matmul(transpose_68, transpose_69, False, True)
        del transpose_68, transpose_69

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_87 = paddle._C_ops.scale(matmul_140, full_2, float("0"), True)
        del matmul_140

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_176 = paddle._C_ops.add(scale_87, scale_1)
        del scale_87

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_17 = paddle._C_ops.softmax(add_176, -1)
        del add_176

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_141 = paddle._C_ops.matmul(softmax_17, transpose_70, False, False)
        del softmax_17, transpose_70

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
        add_177 = paddle._C_ops.add(matmul_142, parameter_9)
        del matmul_142

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_178 = paddle._C_ops.add(layer_norm_102, add_177)
        del add_177, layer_norm_102

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_178, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_178

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_143 = paddle._C_ops.matmul(layer_norm_105, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_179 = paddle._C_ops.add(matmul_143, parameter_5)
        del matmul_143

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_88 = paddle._C_ops.scale(add_179, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_17 = paddle._C_ops.pow(add_179, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_89 = paddle._C_ops.scale(pow_17, full_4, float("0"), True)
        del pow_17

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_180 = paddle._C_ops.add(add_179, scale_89)
        del add_179, scale_89

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_90 = paddle._C_ops.scale(add_180, full_5, float("0"), True)
        del add_180

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_18 = paddle._C_ops.tanh(scale_90)
        del scale_90

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_91 = paddle._C_ops.scale(tanh_18, full_6, float("1"), True)
        del tanh_18

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_17 = paddle._C_ops.multiply(scale_88, scale_91)
        del scale_88, scale_91

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_144 = paddle._C_ops.matmul(multiply_17, parameter_4, False, False)
        del multiply_17

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_181 = paddle._C_ops.add(matmul_144, parameter_3)
        del matmul_144

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_182 = paddle._C_ops.add(add_181, layer_norm_105)
        del add_181, layer_norm_105

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_182, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_182

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_145 = paddle._C_ops.matmul(layer_norm_108, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_183 = paddle._C_ops.add(matmul_145, parameter_15)
        del matmul_145

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_146 = paddle._C_ops.matmul(layer_norm_108, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_184 = paddle._C_ops.add(matmul_146, parameter_13)
        del matmul_146

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_147 = paddle._C_ops.matmul(layer_norm_108, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_185 = paddle._C_ops.add(matmul_147, parameter_11)
        del matmul_147

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(add_183, full_int_array_4)
        del add_183

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_72 = paddle._C_ops.transpose(reshape_72, [0, 2, 1, 3])
        del reshape_72

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(add_184, full_int_array_4)
        del add_184

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_73, [0, 2, 1, 3])
        del reshape_73

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_74 = paddle._C_ops.reshape(add_185, full_int_array_4)
        del add_185

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_74 = paddle._C_ops.transpose(reshape_74, [0, 2, 1, 3])
        del reshape_74

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_148 = paddle._C_ops.matmul(transpose_72, transpose_73, False, True)
        del transpose_72, transpose_73

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_92 = paddle._C_ops.scale(matmul_148, full_2, float("0"), True)
        del matmul_148

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_186 = paddle._C_ops.add(scale_92, scale_1)
        del scale_92

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_18 = paddle._C_ops.softmax(add_186, -1)
        del add_186

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_149 = paddle._C_ops.matmul(softmax_18, transpose_74, False, False)
        del softmax_18, transpose_74

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
        add_187 = paddle._C_ops.add(matmul_150, parameter_9)
        del matmul_150

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_188 = paddle._C_ops.add(layer_norm_108, add_187)
        del add_187, layer_norm_108

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_188, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_188

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_151 = paddle._C_ops.matmul(layer_norm_111, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_189 = paddle._C_ops.add(matmul_151, parameter_5)
        del matmul_151

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_93 = paddle._C_ops.scale(add_189, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_18 = paddle._C_ops.pow(add_189, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_94 = paddle._C_ops.scale(pow_18, full_4, float("0"), True)
        del pow_18

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_190 = paddle._C_ops.add(add_189, scale_94)
        del add_189, scale_94

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_95 = paddle._C_ops.scale(add_190, full_5, float("0"), True)
        del add_190

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_19 = paddle._C_ops.tanh(scale_95)
        del scale_95

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_96 = paddle._C_ops.scale(tanh_19, full_6, float("1"), True)
        del tanh_19

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_18 = paddle._C_ops.multiply(scale_93, scale_96)
        del scale_93, scale_96

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_152 = paddle._C_ops.matmul(multiply_18, parameter_4, False, False)
        del multiply_18

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_191 = paddle._C_ops.add(matmul_152, parameter_3)
        del matmul_152

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_192 = paddle._C_ops.add(add_191, layer_norm_111)
        del add_191, layer_norm_111

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_192, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_192

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_153 = paddle._C_ops.matmul(layer_norm_114, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_193 = paddle._C_ops.add(matmul_153, parameter_15)
        del matmul_153

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_154 = paddle._C_ops.matmul(layer_norm_114, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_194 = paddle._C_ops.add(matmul_154, parameter_13)
        del matmul_154

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_155 = paddle._C_ops.matmul(layer_norm_114, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_195 = paddle._C_ops.add(matmul_155, parameter_11)
        del matmul_155

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_76 = paddle._C_ops.reshape(add_193, full_int_array_4)
        del add_193

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_76 = paddle._C_ops.transpose(reshape_76, [0, 2, 1, 3])
        del reshape_76

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(add_194, full_int_array_4)
        del add_194

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_77 = paddle._C_ops.transpose(reshape_77, [0, 2, 1, 3])
        del reshape_77

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_78 = paddle._C_ops.reshape(add_195, full_int_array_4)
        del add_195

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_78 = paddle._C_ops.transpose(reshape_78, [0, 2, 1, 3])
        del reshape_78

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_156 = paddle._C_ops.matmul(transpose_76, transpose_77, False, True)
        del transpose_76, transpose_77

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_97 = paddle._C_ops.scale(matmul_156, full_2, float("0"), True)
        del matmul_156

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_196 = paddle._C_ops.add(scale_97, scale_1)
        del scale_97

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_19 = paddle._C_ops.softmax(add_196, -1)
        del add_196

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_157 = paddle._C_ops.matmul(softmax_19, transpose_78, False, False)
        del softmax_19, transpose_78

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
        add_197 = paddle._C_ops.add(matmul_158, parameter_9)
        del matmul_158

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_198 = paddle._C_ops.add(layer_norm_114, add_197)
        del add_197, layer_norm_114

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_198, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_198

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_159 = paddle._C_ops.matmul(layer_norm_117, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_199 = paddle._C_ops.add(matmul_159, parameter_5)
        del matmul_159

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_98 = paddle._C_ops.scale(add_199, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_19 = paddle._C_ops.pow(add_199, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_99 = paddle._C_ops.scale(pow_19, full_4, float("0"), True)
        del pow_19

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_200 = paddle._C_ops.add(add_199, scale_99)
        del add_199, scale_99

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_100 = paddle._C_ops.scale(add_200, full_5, float("0"), True)
        del add_200

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_20 = paddle._C_ops.tanh(scale_100)
        del scale_100

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_101 = paddle._C_ops.scale(tanh_20, full_6, float("1"), True)
        del tanh_20

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_19 = paddle._C_ops.multiply(scale_98, scale_101)
        del scale_101, scale_98

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_160 = paddle._C_ops.matmul(multiply_19, parameter_4, False, False)
        del multiply_19

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_201 = paddle._C_ops.add(matmul_160, parameter_3)
        del matmul_160

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_202 = paddle._C_ops.add(add_201, layer_norm_117)
        del add_201, layer_norm_117

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_202, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_202

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_161 = paddle._C_ops.matmul(layer_norm_120, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_203 = paddle._C_ops.add(matmul_161, parameter_15)
        del matmul_161

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_162 = paddle._C_ops.matmul(layer_norm_120, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_204 = paddle._C_ops.add(matmul_162, parameter_13)
        del matmul_162

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_163 = paddle._C_ops.matmul(layer_norm_120, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_205 = paddle._C_ops.add(matmul_163, parameter_11)
        del matmul_163

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_80 = paddle._C_ops.reshape(add_203, full_int_array_4)
        del add_203

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_80 = paddle._C_ops.transpose(reshape_80, [0, 2, 1, 3])
        del reshape_80

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_81 = paddle._C_ops.reshape(add_204, full_int_array_4)
        del add_204

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_81 = paddle._C_ops.transpose(reshape_81, [0, 2, 1, 3])
        del reshape_81

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_82 = paddle._C_ops.reshape(add_205, full_int_array_4)
        del add_205

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_82 = paddle._C_ops.transpose(reshape_82, [0, 2, 1, 3])
        del reshape_82

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_164 = paddle._C_ops.matmul(transpose_80, transpose_81, False, True)
        del transpose_80, transpose_81

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_102 = paddle._C_ops.scale(matmul_164, full_2, float("0"), True)
        del matmul_164

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_206 = paddle._C_ops.add(scale_102, scale_1)
        del scale_102

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_20 = paddle._C_ops.softmax(add_206, -1)
        del add_206

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_165 = paddle._C_ops.matmul(softmax_20, transpose_82, False, False)
        del softmax_20, transpose_82

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
        add_207 = paddle._C_ops.add(matmul_166, parameter_9)
        del matmul_166

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_208 = paddle._C_ops.add(layer_norm_120, add_207)
        del add_207, layer_norm_120

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_208, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_208

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_167 = paddle._C_ops.matmul(layer_norm_123, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_209 = paddle._C_ops.add(matmul_167, parameter_5)
        del matmul_167

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_103 = paddle._C_ops.scale(add_209, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_20 = paddle._C_ops.pow(add_209, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_104 = paddle._C_ops.scale(pow_20, full_4, float("0"), True)
        del pow_20

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_210 = paddle._C_ops.add(add_209, scale_104)
        del add_209, scale_104

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_105 = paddle._C_ops.scale(add_210, full_5, float("0"), True)
        del add_210

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_21 = paddle._C_ops.tanh(scale_105)
        del scale_105

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_106 = paddle._C_ops.scale(tanh_21, full_6, float("1"), True)
        del tanh_21

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_20 = paddle._C_ops.multiply(scale_103, scale_106)
        del scale_103, scale_106

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_168 = paddle._C_ops.matmul(multiply_20, parameter_4, False, False)
        del multiply_20

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_211 = paddle._C_ops.add(matmul_168, parameter_3)
        del matmul_168

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_212 = paddle._C_ops.add(add_211, layer_norm_123)
        del add_211, layer_norm_123

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_212, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_212

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_169 = paddle._C_ops.matmul(layer_norm_126, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_213 = paddle._C_ops.add(matmul_169, parameter_15)
        del matmul_169

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_170 = paddle._C_ops.matmul(layer_norm_126, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_214 = paddle._C_ops.add(matmul_170, parameter_13)
        del matmul_170

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_171 = paddle._C_ops.matmul(layer_norm_126, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_215 = paddle._C_ops.add(matmul_171, parameter_11)
        del matmul_171

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_84 = paddle._C_ops.reshape(add_213, full_int_array_4)
        del add_213

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_84 = paddle._C_ops.transpose(reshape_84, [0, 2, 1, 3])
        del reshape_84

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_85 = paddle._C_ops.reshape(add_214, full_int_array_4)
        del add_214

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_85 = paddle._C_ops.transpose(reshape_85, [0, 2, 1, 3])
        del reshape_85

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_86 = paddle._C_ops.reshape(add_215, full_int_array_4)
        del add_215

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_86 = paddle._C_ops.transpose(reshape_86, [0, 2, 1, 3])
        del reshape_86

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_172 = paddle._C_ops.matmul(transpose_84, transpose_85, False, True)
        del transpose_84, transpose_85

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_107 = paddle._C_ops.scale(matmul_172, full_2, float("0"), True)
        del matmul_172

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_216 = paddle._C_ops.add(scale_107, scale_1)
        del scale_107

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_21 = paddle._C_ops.softmax(add_216, -1)
        del add_216

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_173 = paddle._C_ops.matmul(softmax_21, transpose_86, False, False)
        del softmax_21, transpose_86

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
        add_217 = paddle._C_ops.add(matmul_174, parameter_9)
        del matmul_174

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_218 = paddle._C_ops.add(layer_norm_126, add_217)
        del add_217, layer_norm_126

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_218, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_218

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_175 = paddle._C_ops.matmul(layer_norm_129, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_219 = paddle._C_ops.add(matmul_175, parameter_5)
        del matmul_175

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_108 = paddle._C_ops.scale(add_219, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_21 = paddle._C_ops.pow(add_219, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_109 = paddle._C_ops.scale(pow_21, full_4, float("0"), True)
        del pow_21

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_220 = paddle._C_ops.add(add_219, scale_109)
        del add_219, scale_109

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_110 = paddle._C_ops.scale(add_220, full_5, float("0"), True)
        del add_220

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_22 = paddle._C_ops.tanh(scale_110)
        del scale_110

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_111 = paddle._C_ops.scale(tanh_22, full_6, float("1"), True)
        del tanh_22

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_21 = paddle._C_ops.multiply(scale_108, scale_111)
        del scale_108, scale_111

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_176 = paddle._C_ops.matmul(multiply_21, parameter_4, False, False)
        del multiply_21

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_221 = paddle._C_ops.add(matmul_176, parameter_3)
        del matmul_176

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_222 = paddle._C_ops.add(add_221, layer_norm_129)
        del add_221, layer_norm_129

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_222, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_222

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_177 = paddle._C_ops.matmul(layer_norm_132, parameter_16, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_223 = paddle._C_ops.add(matmul_177, parameter_15)
        del matmul_177

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_178 = paddle._C_ops.matmul(layer_norm_132, parameter_14, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_224 = paddle._C_ops.add(matmul_178, parameter_13)
        del matmul_178

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_179 = paddle._C_ops.matmul(layer_norm_132, parameter_12, False, False)

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_225 = paddle._C_ops.add(matmul_179, parameter_11)
        del matmul_179

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_88 = paddle._C_ops.reshape(add_223, full_int_array_4)
        del add_223

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_88 = paddle._C_ops.transpose(reshape_88, [0, 2, 1, 3])
        del reshape_88

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_89 = paddle._C_ops.reshape(add_224, full_int_array_4)
        del add_224

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_89 = paddle._C_ops.transpose(reshape_89, [0, 2, 1, 3])
        del reshape_89

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_90 = paddle._C_ops.reshape(add_225, full_int_array_4)
        del add_225

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_90 = paddle._C_ops.transpose(reshape_90, [0, 2, 1, 3])
        del reshape_90

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_180 = paddle._C_ops.matmul(transpose_88, transpose_89, False, True)
        del transpose_88, transpose_89

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_112 = paddle._C_ops.scale(matmul_180, full_2, float("0"), True)
        del matmul_180

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_226 = paddle._C_ops.add(scale_112, scale_1)
        del scale_112

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_22 = paddle._C_ops.softmax(add_226, -1)
        del add_226

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_181 = paddle._C_ops.matmul(softmax_22, transpose_90, False, False)
        del softmax_22, transpose_90

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
        add_227 = paddle._C_ops.add(matmul_182, parameter_9)
        del matmul_182

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_228 = paddle._C_ops.add(layer_norm_132, add_227)
        del add_227, layer_norm_132

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_228, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_228

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_183 = paddle._C_ops.matmul(layer_norm_135, parameter_6, False, False)

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_229 = paddle._C_ops.add(matmul_183, parameter_5)
        del matmul_183

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_113 = paddle._C_ops.scale(add_229, full_3, float("0"), True)

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_22 = paddle._C_ops.pow(add_229, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_114 = paddle._C_ops.scale(pow_22, full_4, float("0"), True)
        del pow_22

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_230 = paddle._C_ops.add(add_229, scale_114)
        del add_229, scale_114

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_115 = paddle._C_ops.scale(add_230, full_5, float("0"), True)
        del add_230

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_23 = paddle._C_ops.tanh(scale_115)
        del scale_115

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_116 = paddle._C_ops.scale(tanh_23, full_6, float("1"), True)
        del tanh_23

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_22 = paddle._C_ops.multiply(scale_113, scale_116)
        del scale_113, scale_116

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_184 = paddle._C_ops.matmul(multiply_22, parameter_4, False, False)
        del multiply_22

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_231 = paddle._C_ops.add(matmul_184, parameter_3)
        del matmul_184

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_232 = paddle._C_ops.add(add_231, layer_norm_135)
        del add_231, layer_norm_135

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_232, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_232

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_185 = paddle._C_ops.matmul(layer_norm_138, parameter_16, False, False)
        del parameter_16

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_233 = paddle._C_ops.add(matmul_185, parameter_15)
        del matmul_185, parameter_15

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_186 = paddle._C_ops.matmul(layer_norm_138, parameter_14, False, False)
        del parameter_14

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_234 = paddle._C_ops.add(matmul_186, parameter_13)
        del matmul_186, parameter_13

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x2048xf32, 2048x2048xf32)
        matmul_187 = paddle._C_ops.matmul(layer_norm_138, parameter_12, False, False)
        del parameter_12

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_235 = paddle._C_ops.add(matmul_187, parameter_11)
        del matmul_187, parameter_11

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_92 = paddle._C_ops.reshape(add_233, full_int_array_4)
        del add_233

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_92 = paddle._C_ops.transpose(reshape_92, [0, 2, 1, 3])
        del reshape_92

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_93 = paddle._C_ops.reshape(add_234, full_int_array_4)
        del add_234

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_93 = paddle._C_ops.transpose(reshape_93, [0, 2, 1, 3])
        del reshape_93

        # pd_op.reshape: (1x21x16x128xf32) <- (1x21x2048xf32, 4xi64)
        reshape_94 = paddle._C_ops.reshape(add_235, full_int_array_4)
        del add_235, full_int_array_4

        # pd_op.transpose: (1x16x21x128xf32) <- (1x21x16x128xf32)
        transpose_94 = paddle._C_ops.transpose(reshape_94, [0, 2, 1, 3])
        del reshape_94

        # pd_op.matmul: (1x16x21x21xf32) <- (1x16x21x128xf32, 1x16x21x128xf32)
        matmul_188 = paddle._C_ops.matmul(transpose_92, transpose_93, False, True)
        del transpose_92, transpose_93

        # pd_op.scale: (1x16x21x21xf32) <- (1x16x21x21xf32, 1xf32)
        scale_117 = paddle._C_ops.scale(matmul_188, full_2, float("0"), True)
        del full_2, matmul_188

        # pd_op.add: (1x16x21x21xf32) <- (1x16x21x21xf32, 1x1x1x21xf32)
        add_236 = paddle._C_ops.add(scale_117, scale_1)
        del scale_1, scale_117

        # pd_op.softmax: (1x16x21x21xf32) <- (1x16x21x21xf32)
        softmax_23 = paddle._C_ops.softmax(add_236, -1)
        del add_236

        # pd_op.matmul: (1x16x21x128xf32) <- (1x16x21x21xf32, 1x16x21x128xf32)
        matmul_189 = paddle._C_ops.matmul(softmax_23, transpose_94, False, False)
        del softmax_23, transpose_94

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
        add_237 = paddle._C_ops.add(matmul_190, parameter_9)
        del matmul_190, parameter_9

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_238 = paddle._C_ops.add(layer_norm_138, add_237)
        del add_237, layer_norm_138

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_238, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_238, parameter_7, parameter_8

        # pd_op.matmul: (1x21x8192xf32) <- (1x21x2048xf32, 2048x8192xf32)
        matmul_191 = paddle._C_ops.matmul(layer_norm_141, parameter_6, False, False)
        del parameter_6

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 8192xf32)
        add_239 = paddle._C_ops.add(matmul_191, parameter_5)
        del matmul_191, parameter_5

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_118 = paddle._C_ops.scale(add_239, full_3, float("0"), True)
        del full_3

        # pd_op.pow: (1x21x8192xf32) <- (1x21x8192xf32)
        pow_23 = paddle._C_ops.pow(add_239, float("3"))

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_119 = paddle._C_ops.scale(pow_23, full_4, float("0"), True)
        del full_4, pow_23

        # pd_op.add: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        add_240 = paddle._C_ops.add(add_239, scale_119)
        del add_239, scale_119

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_120 = paddle._C_ops.scale(add_240, full_5, float("0"), True)
        del add_240, full_5

        # pd_op.tanh: (1x21x8192xf32) <- (1x21x8192xf32)
        tanh_24 = paddle._C_ops.tanh(scale_120)
        del scale_120

        # pd_op.scale: (1x21x8192xf32) <- (1x21x8192xf32, 1xf32)
        scale_121 = paddle._C_ops.scale(tanh_24, full_6, float("1"), True)
        del full_6, tanh_24

        # pd_op.multiply: (1x21x8192xf32) <- (1x21x8192xf32, 1x21x8192xf32)
        multiply_23 = paddle._C_ops.multiply(scale_118, scale_121)
        del scale_118, scale_121

        # pd_op.matmul: (1x21x2048xf32) <- (1x21x8192xf32, 8192x2048xf32)
        matmul_192 = paddle._C_ops.matmul(multiply_23, parameter_4, False, False)
        del multiply_23, parameter_4

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 2048xf32)
        add_241 = paddle._C_ops.add(matmul_192, parameter_3)
        del matmul_192, parameter_3

        # pd_op.add: (1x21x2048xf32) <- (1x21x2048xf32, 1x21x2048xf32)
        add_242 = paddle._C_ops.add(add_241, layer_norm_141)
        del add_241, layer_norm_141

        # pd_op.layer_norm: (1x21x2048xf32, 1x21xf32, 1x21xf32) <- (1x21x2048xf32, 2048xf32, 2048xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_242, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_242, parameter_17, parameter_18

        # pd_op.slice: (1x2048xf32) <- (1x21x2048xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            layer_norm_144, [1], full_int_array_2, full_int_array_0, [1], [1]
        )
        del full_int_array_0, full_int_array_2

        # pd_op.matmul: (1x2048xf32) <- (1x2048xf32, 2048x2048xf32)
        matmul_193 = paddle._C_ops.matmul(slice_1, parameter_2, False, False)
        del parameter_2, slice_1

        # pd_op.add: (1x2048xf32) <- (1x2048xf32, 2048xf32)
        add_243 = paddle._C_ops.add(matmul_193, parameter_1)
        del matmul_193, parameter_1

        # pd_op.tanh: (1x2048xf32) <- (1x2048xf32)
        tanh_0 = paddle._C_ops.tanh(add_243)
        del add_243, layer_norm_144

        return tanh_0
