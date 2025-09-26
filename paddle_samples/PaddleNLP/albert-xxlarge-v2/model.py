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

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x128xf32, 128x4096xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_20, False, False)
        del layer_norm_0, parameter_20

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_19)
        del matmul_0, parameter_19

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_1 = paddle._C_ops.matmul(add_2, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_15)
        del matmul_1

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_2 = paddle._C_ops.matmul(add_2, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_13)
        del matmul_2

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_3 = paddle._C_ops.matmul(add_2, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_5 = paddle._C_ops.add(matmul_3, parameter_11)
        del matmul_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 21, 64, 64]

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_3, full_int_array_4)
        del add_3

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_4, full_int_array_4)
        del add_4

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_5, full_int_array_4)
        del add_5

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_0, transpose_1, False, True)
        del transpose_0, transpose_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_4, full_2, float("0"), True)
        del matmul_4

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_6 = paddle._C_ops.add(scale_2, scale_1)
        del scale_2

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_6, -1)
        del add_6

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_5 = paddle._C_ops.matmul(softmax_0, transpose_2, False, False)
        del softmax_0, transpose_2

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_5, [0, 2, 1, 3])
        del matmul_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [0, 0, -1]

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_5)
        del transpose_3

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_3, parameter_10, False, False)
        del reshape_3

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_7 = paddle._C_ops.add(matmul_6, parameter_9)
        del matmul_6

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_8 = paddle._C_ops.add(add_2, add_7)
        del add_2, add_7

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_8

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_3, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_5)
        del matmul_7

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_9, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_0 = paddle._C_ops.pow(add_9, float("3"))

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.044715"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(pow_0, full_4, float("0"), True)
        del pow_0

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_10 = paddle._C_ops.add(add_9, scale_4)
        del add_9, scale_4

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.797885"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(add_10, full_5, float("0"), True)
        del add_10

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_1 = paddle._C_ops.tanh(scale_5)
        del scale_5

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(tanh_1, full_6, float("1"), True)
        del tanh_1

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_0 = paddle._C_ops.multiply(scale_3, scale_6)
        del scale_3, scale_6

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_8 = paddle._C_ops.matmul(multiply_0, parameter_4, False, False)
        del multiply_0

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_11 = paddle._C_ops.add(matmul_8, parameter_3)
        del matmul_8

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_12 = paddle._C_ops.add(add_11, layer_norm_3)
        del add_11, layer_norm_3

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_12

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_13 = paddle._C_ops.add(matmul_9, parameter_15)
        del matmul_9

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_14 = paddle._C_ops.add(matmul_10, parameter_13)
        del matmul_10

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_6, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_15 = paddle._C_ops.add(matmul_11, parameter_11)
        del matmul_11

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_13, full_int_array_4)
        del add_13

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_14, full_int_array_4)
        del add_14

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_15, full_int_array_4)
        del add_15

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_4, transpose_5, False, True)
        del transpose_4, transpose_5

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_12, full_2, float("0"), True)
        del matmul_12

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_16 = paddle._C_ops.add(scale_7, scale_1)
        del scale_7

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_16, -1)
        del add_16

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_13 = paddle._C_ops.matmul(softmax_1, transpose_6, False, False)
        del softmax_1, transpose_6

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_13, [0, 2, 1, 3])
        del matmul_13

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_5)
        del transpose_7

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_14 = paddle._C_ops.matmul(reshape_7, parameter_10, False, False)
        del reshape_7

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_17 = paddle._C_ops.add(matmul_14, parameter_9)
        del matmul_14

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_18 = paddle._C_ops.add(layer_norm_6, add_17)
        del add_17, layer_norm_6

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_18

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_9, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_19 = paddle._C_ops.add(matmul_15, parameter_5)
        del matmul_15

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(add_19, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_1 = paddle._C_ops.pow(add_19, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(pow_1, full_4, float("0"), True)
        del pow_1

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_20 = paddle._C_ops.add(add_19, scale_9)
        del add_19, scale_9

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(add_20, full_5, float("0"), True)
        del add_20

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_2 = paddle._C_ops.tanh(scale_10)
        del scale_10

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(tanh_2, full_6, float("1"), True)
        del tanh_2

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_1 = paddle._C_ops.multiply(scale_8, scale_11)
        del scale_11, scale_8

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_16 = paddle._C_ops.matmul(multiply_1, parameter_4, False, False)
        del multiply_1

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_21 = paddle._C_ops.add(matmul_16, parameter_3)
        del matmul_16

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_22 = paddle._C_ops.add(add_21, layer_norm_9)
        del add_21, layer_norm_9

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_22

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_23 = paddle._C_ops.add(matmul_17, parameter_15)
        del matmul_17

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_24 = paddle._C_ops.add(matmul_18, parameter_13)
        del matmul_18

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_12, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_25 = paddle._C_ops.add(matmul_19, parameter_11)
        del matmul_19

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_23, full_int_array_4)
        del add_23

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_24, full_int_array_4)
        del add_24

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_25, full_int_array_4)
        del add_25

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_8, transpose_9, False, True)
        del transpose_8, transpose_9

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(matmul_20, full_2, float("0"), True)
        del matmul_20

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_26 = paddle._C_ops.add(scale_12, scale_1)
        del scale_12

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_26, -1)
        del add_26

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_2, transpose_10, False, False)
        del softmax_2, transpose_10

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_5)
        del transpose_11

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_11, parameter_10, False, False)
        del reshape_11

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_27 = paddle._C_ops.add(matmul_22, parameter_9)
        del matmul_22

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_28 = paddle._C_ops.add(layer_norm_12, add_27)
        del add_27, layer_norm_12

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_28

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_15, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_29 = paddle._C_ops.add(matmul_23, parameter_5)
        del matmul_23

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(add_29, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_2 = paddle._C_ops.pow(add_29, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(pow_2, full_4, float("0"), True)
        del pow_2

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_30 = paddle._C_ops.add(add_29, scale_14)
        del add_29, scale_14

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(add_30, full_5, float("0"), True)
        del add_30

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_3 = paddle._C_ops.tanh(scale_15)
        del scale_15

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(tanh_3, full_6, float("1"), True)
        del tanh_3

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_2 = paddle._C_ops.multiply(scale_13, scale_16)
        del scale_13, scale_16

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_24 = paddle._C_ops.matmul(multiply_2, parameter_4, False, False)
        del multiply_2

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_31 = paddle._C_ops.add(matmul_24, parameter_3)
        del matmul_24

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_32 = paddle._C_ops.add(add_31, layer_norm_15)
        del add_31, layer_norm_15

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_32, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_32

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_33 = paddle._C_ops.add(matmul_25, parameter_15)
        del matmul_25

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_34 = paddle._C_ops.add(matmul_26, parameter_13)
        del matmul_26

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_27 = paddle._C_ops.matmul(layer_norm_18, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_35 = paddle._C_ops.add(matmul_27, parameter_11)
        del matmul_27

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_33, full_int_array_4)
        del add_33

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_34, full_int_array_4)
        del add_34

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_35, full_int_array_4)
        del add_35

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_28 = paddle._C_ops.matmul(transpose_12, transpose_13, False, True)
        del transpose_12, transpose_13

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(matmul_28, full_2, float("0"), True)
        del matmul_28

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_36 = paddle._C_ops.add(scale_17, scale_1)
        del scale_17

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_36, -1)
        del add_36

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_29 = paddle._C_ops.matmul(softmax_3, transpose_14, False, False)
        del softmax_3, transpose_14

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_29, [0, 2, 1, 3])
        del matmul_29

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_5)
        del transpose_15

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_30 = paddle._C_ops.matmul(reshape_15, parameter_10, False, False)
        del reshape_15

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_37 = paddle._C_ops.add(matmul_30, parameter_9)
        del matmul_30

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_38 = paddle._C_ops.add(layer_norm_18, add_37)
        del add_37, layer_norm_18

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_38

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_21, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_39 = paddle._C_ops.add(matmul_31, parameter_5)
        del matmul_31

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(add_39, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_3 = paddle._C_ops.pow(add_39, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(pow_3, full_4, float("0"), True)
        del pow_3

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_40 = paddle._C_ops.add(add_39, scale_19)
        del add_39, scale_19

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(add_40, full_5, float("0"), True)
        del add_40

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_4 = paddle._C_ops.tanh(scale_20)
        del scale_20

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(tanh_4, full_6, float("1"), True)
        del tanh_4

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_3 = paddle._C_ops.multiply(scale_18, scale_21)
        del scale_18, scale_21

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_32 = paddle._C_ops.matmul(multiply_3, parameter_4, False, False)
        del multiply_3

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_41 = paddle._C_ops.add(matmul_32, parameter_3)
        del matmul_32

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_42 = paddle._C_ops.add(add_41, layer_norm_21)
        del add_41, layer_norm_21

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_42

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_43 = paddle._C_ops.add(matmul_33, parameter_15)
        del matmul_33

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_44 = paddle._C_ops.add(matmul_34, parameter_13)
        del matmul_34

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_24, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_45 = paddle._C_ops.add(matmul_35, parameter_11)
        del matmul_35

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_43, full_int_array_4)
        del add_43

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_44, full_int_array_4)
        del add_44

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_45, full_int_array_4)
        del add_45

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_36 = paddle._C_ops.matmul(transpose_16, transpose_17, False, True)
        del transpose_16, transpose_17

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(matmul_36, full_2, float("0"), True)
        del matmul_36

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_46 = paddle._C_ops.add(scale_22, scale_1)
        del scale_22

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(add_46, -1)
        del add_46

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_37 = paddle._C_ops.matmul(softmax_4, transpose_18, False, False)
        del softmax_4, transpose_18

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_37, [0, 2, 1, 3])
        del matmul_37

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_5)
        del transpose_19

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_19, parameter_10, False, False)
        del reshape_19

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_47 = paddle._C_ops.add(matmul_38, parameter_9)
        del matmul_38

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_48 = paddle._C_ops.add(layer_norm_24, add_47)
        del add_47, layer_norm_24

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_48, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_48

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_39 = paddle._C_ops.matmul(layer_norm_27, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_49 = paddle._C_ops.add(matmul_39, parameter_5)
        del matmul_39

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(add_49, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_4 = paddle._C_ops.pow(add_49, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(pow_4, full_4, float("0"), True)
        del pow_4

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_50 = paddle._C_ops.add(add_49, scale_24)
        del add_49, scale_24

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(add_50, full_5, float("0"), True)
        del add_50

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_5 = paddle._C_ops.tanh(scale_25)
        del scale_25

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(tanh_5, full_6, float("1"), True)
        del tanh_5

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_4 = paddle._C_ops.multiply(scale_23, scale_26)
        del scale_23, scale_26

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_40 = paddle._C_ops.matmul(multiply_4, parameter_4, False, False)
        del multiply_4

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_51 = paddle._C_ops.add(matmul_40, parameter_3)
        del matmul_40

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_52 = paddle._C_ops.add(add_51, layer_norm_27)
        del add_51, layer_norm_27

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_52

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_53 = paddle._C_ops.add(matmul_41, parameter_15)
        del matmul_41

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_54 = paddle._C_ops.add(matmul_42, parameter_13)
        del matmul_42

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_43 = paddle._C_ops.matmul(layer_norm_30, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_55 = paddle._C_ops.add(matmul_43, parameter_11)
        del matmul_43

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_53, full_int_array_4)
        del add_53

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_54, full_int_array_4)
        del add_54

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_55, full_int_array_4)
        del add_55

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_44 = paddle._C_ops.matmul(transpose_20, transpose_21, False, True)
        del transpose_20, transpose_21

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(matmul_44, full_2, float("0"), True)
        del matmul_44

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_56 = paddle._C_ops.add(scale_27, scale_1)
        del scale_27

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_56, -1)
        del add_56

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_45 = paddle._C_ops.matmul(softmax_5, transpose_22, False, False)
        del softmax_5, transpose_22

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])
        del matmul_45

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_5)
        del transpose_23

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_23, parameter_10, False, False)
        del reshape_23

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_57 = paddle._C_ops.add(matmul_46, parameter_9)
        del matmul_46

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_58 = paddle._C_ops.add(layer_norm_30, add_57)
        del add_57, layer_norm_30

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_58

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_33, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_59 = paddle._C_ops.add(matmul_47, parameter_5)
        del matmul_47

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(add_59, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_5 = paddle._C_ops.pow(add_59, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(pow_5, full_4, float("0"), True)
        del pow_5

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_60 = paddle._C_ops.add(add_59, scale_29)
        del add_59, scale_29

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(add_60, full_5, float("0"), True)
        del add_60

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_6 = paddle._C_ops.tanh(scale_30)
        del scale_30

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(tanh_6, full_6, float("1"), True)
        del tanh_6

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_5 = paddle._C_ops.multiply(scale_28, scale_31)
        del scale_28, scale_31

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_48 = paddle._C_ops.matmul(multiply_5, parameter_4, False, False)
        del multiply_5

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_61 = paddle._C_ops.add(matmul_48, parameter_3)
        del matmul_48

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_62 = paddle._C_ops.add(add_61, layer_norm_33)
        del add_61, layer_norm_33

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_62, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_62

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_36, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_63 = paddle._C_ops.add(matmul_49, parameter_15)
        del matmul_49

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_36, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_64 = paddle._C_ops.add(matmul_50, parameter_13)
        del matmul_50

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_51 = paddle._C_ops.matmul(layer_norm_36, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_65 = paddle._C_ops.add(matmul_51, parameter_11)
        del matmul_51

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_63, full_int_array_4)
        del add_63

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_64, full_int_array_4)
        del add_64

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_65, full_int_array_4)
        del add_65

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_52 = paddle._C_ops.matmul(transpose_24, transpose_25, False, True)
        del transpose_24, transpose_25

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(matmul_52, full_2, float("0"), True)
        del matmul_52

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_66 = paddle._C_ops.add(scale_32, scale_1)
        del scale_32

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_6 = paddle._C_ops.softmax(add_66, -1)
        del add_66

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_53 = paddle._C_ops.matmul(softmax_6, transpose_26, False, False)
        del softmax_6, transpose_26

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_53, [0, 2, 1, 3])
        del matmul_53

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_27, full_int_array_5)
        del transpose_27

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_54 = paddle._C_ops.matmul(reshape_27, parameter_10, False, False)
        del reshape_27

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_67 = paddle._C_ops.add(matmul_54, parameter_9)
        del matmul_54

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_68 = paddle._C_ops.add(layer_norm_36, add_67)
        del add_67, layer_norm_36

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_68, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_68

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_39, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_69 = paddle._C_ops.add(matmul_55, parameter_5)
        del matmul_55

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(add_69, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_6 = paddle._C_ops.pow(add_69, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(pow_6, full_4, float("0"), True)
        del pow_6

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_70 = paddle._C_ops.add(add_69, scale_34)
        del add_69, scale_34

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(add_70, full_5, float("0"), True)
        del add_70

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_7 = paddle._C_ops.tanh(scale_35)
        del scale_35

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(tanh_7, full_6, float("1"), True)
        del tanh_7

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_6 = paddle._C_ops.multiply(scale_33, scale_36)
        del scale_33, scale_36

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_56 = paddle._C_ops.matmul(multiply_6, parameter_4, False, False)
        del multiply_6

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_71 = paddle._C_ops.add(matmul_56, parameter_3)
        del matmul_56

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_72 = paddle._C_ops.add(add_71, layer_norm_39)
        del add_71, layer_norm_39

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_72, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_72

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_57 = paddle._C_ops.matmul(layer_norm_42, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_73 = paddle._C_ops.add(matmul_57, parameter_15)
        del matmul_57

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_42, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_74 = paddle._C_ops.add(matmul_58, parameter_13)
        del matmul_58

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_59 = paddle._C_ops.matmul(layer_norm_42, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_75 = paddle._C_ops.add(matmul_59, parameter_11)
        del matmul_59

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_73, full_int_array_4)
        del add_73

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_74, full_int_array_4)
        del add_74

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_75, full_int_array_4)
        del add_75

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_60 = paddle._C_ops.matmul(transpose_28, transpose_29, False, True)
        del transpose_28, transpose_29

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(matmul_60, full_2, float("0"), True)
        del matmul_60

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_76 = paddle._C_ops.add(scale_37, scale_1)
        del scale_37

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_7 = paddle._C_ops.softmax(add_76, -1)
        del add_76

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_61 = paddle._C_ops.matmul(softmax_7, transpose_30, False, False)
        del softmax_7, transpose_30

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_61, [0, 2, 1, 3])
        del matmul_61

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_31, full_int_array_5)
        del transpose_31

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_62 = paddle._C_ops.matmul(reshape_31, parameter_10, False, False)
        del reshape_31

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_77 = paddle._C_ops.add(matmul_62, parameter_9)
        del matmul_62

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_78 = paddle._C_ops.add(layer_norm_42, add_77)
        del add_77, layer_norm_42

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_78, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_78

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_63 = paddle._C_ops.matmul(layer_norm_45, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_79 = paddle._C_ops.add(matmul_63, parameter_5)
        del matmul_63

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(add_79, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_7 = paddle._C_ops.pow(add_79, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(pow_7, full_4, float("0"), True)
        del pow_7

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_80 = paddle._C_ops.add(add_79, scale_39)
        del add_79, scale_39

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_40 = paddle._C_ops.scale(add_80, full_5, float("0"), True)
        del add_80

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_8 = paddle._C_ops.tanh(scale_40)
        del scale_40

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_41 = paddle._C_ops.scale(tanh_8, full_6, float("1"), True)
        del tanh_8

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_7 = paddle._C_ops.multiply(scale_38, scale_41)
        del scale_38, scale_41

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_64 = paddle._C_ops.matmul(multiply_7, parameter_4, False, False)
        del multiply_7

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_81 = paddle._C_ops.add(matmul_64, parameter_3)
        del matmul_64

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_82 = paddle._C_ops.add(add_81, layer_norm_45)
        del add_81, layer_norm_45

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_82

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_65 = paddle._C_ops.matmul(layer_norm_48, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_83 = paddle._C_ops.add(matmul_65, parameter_15)
        del matmul_65

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_48, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_84 = paddle._C_ops.add(matmul_66, parameter_13)
        del matmul_66

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_48, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_85 = paddle._C_ops.add(matmul_67, parameter_11)
        del matmul_67

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_83, full_int_array_4)
        del add_83

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(add_84, full_int_array_4)
        del add_84

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_85, full_int_array_4)
        del add_85

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_68 = paddle._C_ops.matmul(transpose_32, transpose_33, False, True)
        del transpose_32, transpose_33

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_42 = paddle._C_ops.scale(matmul_68, full_2, float("0"), True)
        del matmul_68

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_86 = paddle._C_ops.add(scale_42, scale_1)
        del scale_42

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_8 = paddle._C_ops.softmax(add_86, -1)
        del add_86

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_69 = paddle._C_ops.matmul(softmax_8, transpose_34, False, False)
        del softmax_8, transpose_34

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_69, [0, 2, 1, 3])
        del matmul_69

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_35, full_int_array_5)
        del transpose_35

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_70 = paddle._C_ops.matmul(reshape_35, parameter_10, False, False)
        del reshape_35

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_87 = paddle._C_ops.add(matmul_70, parameter_9)
        del matmul_70

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_88 = paddle._C_ops.add(layer_norm_48, add_87)
        del add_87, layer_norm_48

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_88, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_88

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_71 = paddle._C_ops.matmul(layer_norm_51, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_89 = paddle._C_ops.add(matmul_71, parameter_5)
        del matmul_71

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_43 = paddle._C_ops.scale(add_89, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_8 = paddle._C_ops.pow(add_89, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_44 = paddle._C_ops.scale(pow_8, full_4, float("0"), True)
        del pow_8

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_90 = paddle._C_ops.add(add_89, scale_44)
        del add_89, scale_44

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_45 = paddle._C_ops.scale(add_90, full_5, float("0"), True)
        del add_90

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_9 = paddle._C_ops.tanh(scale_45)
        del scale_45

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_46 = paddle._C_ops.scale(tanh_9, full_6, float("1"), True)
        del tanh_9

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_8 = paddle._C_ops.multiply(scale_43, scale_46)
        del scale_43, scale_46

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_72 = paddle._C_ops.matmul(multiply_8, parameter_4, False, False)
        del multiply_8

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_91 = paddle._C_ops.add(matmul_72, parameter_3)
        del matmul_72

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_92 = paddle._C_ops.add(add_91, layer_norm_51)
        del add_91, layer_norm_51

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_92, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_92

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_54, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_93 = paddle._C_ops.add(matmul_73, parameter_15)
        del matmul_73

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_54, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_94 = paddle._C_ops.add(matmul_74, parameter_13)
        del matmul_74

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_75 = paddle._C_ops.matmul(layer_norm_54, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_95 = paddle._C_ops.add(matmul_75, parameter_11)
        del matmul_75

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_93, full_int_array_4)
        del add_93

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_94, full_int_array_4)
        del add_94

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_95, full_int_array_4)
        del add_95

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_76 = paddle._C_ops.matmul(transpose_36, transpose_37, False, True)
        del transpose_36, transpose_37

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_47 = paddle._C_ops.scale(matmul_76, full_2, float("0"), True)
        del matmul_76

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_96 = paddle._C_ops.add(scale_47, scale_1)
        del scale_47

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_9 = paddle._C_ops.softmax(add_96, -1)
        del add_96

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_77 = paddle._C_ops.matmul(softmax_9, transpose_38, False, False)
        del softmax_9, transpose_38

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_77, [0, 2, 1, 3])
        del matmul_77

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_39, full_int_array_5)
        del transpose_39

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_78 = paddle._C_ops.matmul(reshape_39, parameter_10, False, False)
        del reshape_39

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_97 = paddle._C_ops.add(matmul_78, parameter_9)
        del matmul_78

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_98 = paddle._C_ops.add(layer_norm_54, add_97)
        del add_97, layer_norm_54

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_98, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_98

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_79 = paddle._C_ops.matmul(layer_norm_57, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_99 = paddle._C_ops.add(matmul_79, parameter_5)
        del matmul_79

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_48 = paddle._C_ops.scale(add_99, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_9 = paddle._C_ops.pow(add_99, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_49 = paddle._C_ops.scale(pow_9, full_4, float("0"), True)
        del pow_9

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_100 = paddle._C_ops.add(add_99, scale_49)
        del add_99, scale_49

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_50 = paddle._C_ops.scale(add_100, full_5, float("0"), True)
        del add_100

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_10 = paddle._C_ops.tanh(scale_50)
        del scale_50

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_51 = paddle._C_ops.scale(tanh_10, full_6, float("1"), True)
        del tanh_10

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_9 = paddle._C_ops.multiply(scale_48, scale_51)
        del scale_48, scale_51

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_80 = paddle._C_ops.matmul(multiply_9, parameter_4, False, False)
        del multiply_9

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_101 = paddle._C_ops.add(matmul_80, parameter_3)
        del matmul_80

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_102 = paddle._C_ops.add(add_101, layer_norm_57)
        del add_101, layer_norm_57

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_102, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_102

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_60, parameter_16, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_103 = paddle._C_ops.add(matmul_81, parameter_15)
        del matmul_81

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_60, parameter_14, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_104 = paddle._C_ops.add(matmul_82, parameter_13)
        del matmul_82

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_83 = paddle._C_ops.matmul(layer_norm_60, parameter_12, False, False)

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_105 = paddle._C_ops.add(matmul_83, parameter_11)
        del matmul_83

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_103, full_int_array_4)
        del add_103

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_104, full_int_array_4)
        del add_104

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_105, full_int_array_4)
        del add_105

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_84 = paddle._C_ops.matmul(transpose_40, transpose_41, False, True)
        del transpose_40, transpose_41

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_52 = paddle._C_ops.scale(matmul_84, full_2, float("0"), True)
        del matmul_84

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_106 = paddle._C_ops.add(scale_52, scale_1)
        del scale_52

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_10 = paddle._C_ops.softmax(add_106, -1)
        del add_106

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_85 = paddle._C_ops.matmul(softmax_10, transpose_42, False, False)
        del softmax_10, transpose_42

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_85, [0, 2, 1, 3])
        del matmul_85

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_43, full_int_array_5)
        del transpose_43

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_86 = paddle._C_ops.matmul(reshape_43, parameter_10, False, False)
        del reshape_43

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_107 = paddle._C_ops.add(matmul_86, parameter_9)
        del matmul_86

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_108 = paddle._C_ops.add(layer_norm_60, add_107)
        del add_107, layer_norm_60

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_108, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_108

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_87 = paddle._C_ops.matmul(layer_norm_63, parameter_6, False, False)

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_109 = paddle._C_ops.add(matmul_87, parameter_5)
        del matmul_87

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_53 = paddle._C_ops.scale(add_109, full_3, float("0"), True)

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_10 = paddle._C_ops.pow(add_109, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_54 = paddle._C_ops.scale(pow_10, full_4, float("0"), True)
        del pow_10

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_110 = paddle._C_ops.add(add_109, scale_54)
        del add_109, scale_54

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_55 = paddle._C_ops.scale(add_110, full_5, float("0"), True)
        del add_110

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_11 = paddle._C_ops.tanh(scale_55)
        del scale_55

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_56 = paddle._C_ops.scale(tanh_11, full_6, float("1"), True)
        del tanh_11

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_10 = paddle._C_ops.multiply(scale_53, scale_56)
        del scale_53, scale_56

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_88 = paddle._C_ops.matmul(multiply_10, parameter_4, False, False)
        del multiply_10

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_111 = paddle._C_ops.add(matmul_88, parameter_3)
        del matmul_88

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_112 = paddle._C_ops.add(add_111, layer_norm_63)
        del add_111, layer_norm_63

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_112, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_112

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_66, parameter_16, False, False)
        del parameter_16

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_113 = paddle._C_ops.add(matmul_89, parameter_15)
        del matmul_89, parameter_15

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_66, parameter_14, False, False)
        del parameter_14

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_114 = paddle._C_ops.add(matmul_90, parameter_13)
        del matmul_90, parameter_13

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_66, parameter_12, False, False)
        del parameter_12

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_115 = paddle._C_ops.add(matmul_91, parameter_11)
        del matmul_91, parameter_11

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(add_113, full_int_array_4)
        del add_113

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_114, full_int_array_4)
        del add_114

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.reshape: (1x21x64x64xf32) <- (1x21x4096xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_115, full_int_array_4)
        del add_115, full_int_array_4

        # pd_op.transpose: (1x64x21x64xf32) <- (1x21x64x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.matmul: (1x64x21x21xf32) <- (1x64x21x64xf32, 1x64x21x64xf32)
        matmul_92 = paddle._C_ops.matmul(transpose_44, transpose_45, False, True)
        del transpose_44, transpose_45

        # pd_op.scale: (1x64x21x21xf32) <- (1x64x21x21xf32, 1xf32)
        scale_57 = paddle._C_ops.scale(matmul_92, full_2, float("0"), True)
        del full_2, matmul_92

        # pd_op.add: (1x64x21x21xf32) <- (1x64x21x21xf32, 1x1x1x21xf32)
        add_116 = paddle._C_ops.add(scale_57, scale_1)
        del scale_1, scale_57

        # pd_op.softmax: (1x64x21x21xf32) <- (1x64x21x21xf32)
        softmax_11 = paddle._C_ops.softmax(add_116, -1)
        del add_116

        # pd_op.matmul: (1x64x21x64xf32) <- (1x64x21x21xf32, 1x64x21x64xf32)
        matmul_93 = paddle._C_ops.matmul(softmax_11, transpose_46, False, False)
        del softmax_11, transpose_46

        # pd_op.transpose: (1x21x64x64xf32) <- (1x64x21x64xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_93, [0, 2, 1, 3])
        del matmul_93

        # pd_op.reshape: (1x21x4096xf32) <- (1x21x64x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_47, full_int_array_5)
        del full_int_array_5, transpose_47

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x4096xf32, 4096x4096xf32)
        matmul_94 = paddle._C_ops.matmul(reshape_47, parameter_10, False, False)
        del parameter_10, reshape_47

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_117 = paddle._C_ops.add(matmul_94, parameter_9)
        del matmul_94, parameter_9

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_118 = paddle._C_ops.add(layer_norm_66, add_117)
        del add_117, layer_norm_66

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_118, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_118, parameter_7, parameter_8

        # pd_op.matmul: (1x21x16384xf32) <- (1x21x4096xf32, 4096x16384xf32)
        matmul_95 = paddle._C_ops.matmul(layer_norm_69, parameter_6, False, False)
        del parameter_6

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 16384xf32)
        add_119 = paddle._C_ops.add(matmul_95, parameter_5)
        del matmul_95, parameter_5

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_58 = paddle._C_ops.scale(add_119, full_3, float("0"), True)
        del full_3

        # pd_op.pow: (1x21x16384xf32) <- (1x21x16384xf32)
        pow_11 = paddle._C_ops.pow(add_119, float("3"))

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_59 = paddle._C_ops.scale(pow_11, full_4, float("0"), True)
        del full_4, pow_11

        # pd_op.add: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        add_120 = paddle._C_ops.add(add_119, scale_59)
        del add_119, scale_59

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_60 = paddle._C_ops.scale(add_120, full_5, float("0"), True)
        del add_120, full_5

        # pd_op.tanh: (1x21x16384xf32) <- (1x21x16384xf32)
        tanh_12 = paddle._C_ops.tanh(scale_60)
        del scale_60

        # pd_op.scale: (1x21x16384xf32) <- (1x21x16384xf32, 1xf32)
        scale_61 = paddle._C_ops.scale(tanh_12, full_6, float("1"), True)
        del full_6, tanh_12

        # pd_op.multiply: (1x21x16384xf32) <- (1x21x16384xf32, 1x21x16384xf32)
        multiply_11 = paddle._C_ops.multiply(scale_58, scale_61)
        del scale_58, scale_61

        # pd_op.matmul: (1x21x4096xf32) <- (1x21x16384xf32, 16384x4096xf32)
        matmul_96 = paddle._C_ops.matmul(multiply_11, parameter_4, False, False)
        del multiply_11, parameter_4

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 4096xf32)
        add_121 = paddle._C_ops.add(matmul_96, parameter_3)
        del matmul_96, parameter_3

        # pd_op.add: (1x21x4096xf32) <- (1x21x4096xf32, 1x21x4096xf32)
        add_122 = paddle._C_ops.add(add_121, layer_norm_69)
        del add_121, layer_norm_69

        # pd_op.layer_norm: (1x21x4096xf32, 1x21xf32, 1x21xf32) <- (1x21x4096xf32, 4096xf32, 4096xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_122, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_122, parameter_17, parameter_18

        # pd_op.slice: (1x4096xf32) <- (1x21x4096xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            layer_norm_72, [1], full_int_array_2, full_int_array_0, [1], [1]
        )
        del full_int_array_0, full_int_array_2

        # pd_op.matmul: (1x4096xf32) <- (1x4096xf32, 4096x4096xf32)
        matmul_97 = paddle._C_ops.matmul(slice_1, parameter_2, False, False)
        del parameter_2, slice_1

        # pd_op.add: (1x4096xf32) <- (1x4096xf32, 4096xf32)
        add_123 = paddle._C_ops.add(matmul_97, parameter_1)
        del matmul_97, parameter_1

        # pd_op.tanh: (1x4096xf32) <- (1x4096xf32)
        tanh_0 = paddle._C_ops.tanh(add_123)
        del add_123, layer_norm_72

        return tanh_0
