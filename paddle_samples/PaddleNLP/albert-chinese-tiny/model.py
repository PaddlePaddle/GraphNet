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

        # pd_op.unsqueeze: (1x1x11xi64) <- (1x11xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_0)
        del data_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.unsqueeze: (1x1x1x11xi64) <- (1x1x11xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(unsqueeze_0, full_int_array_1)
        del full_int_array_1, unsqueeze_0

        # pd_op.cast: (1x1x1x11xf32) <- (1x1x1x11xi64)
        cast_0 = paddle._C_ops.cast(unsqueeze_1, paddle.float32)
        del unsqueeze_1

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x11xf32) <- (1x1x1x11xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_0, float("1"), True)
        del cast_0, full_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x11xf32) <- (1x1x1x11xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_1, float("0"), True)
        del full_1, scale_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [11]

        # pd_op.slice: (1x11xi64) <- (1x512xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            parameter_0, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del full_int_array_3, parameter_0

        # pd_op.embedding: (1x11x128xf32) <- (1x11xi64, 21128x128xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_25, 0, False)
        del data_0, parameter_25

        # pd_op.embedding: (1x11x128xf32) <- (1x11xi64, 2x128xf32)
        embedding_1 = paddle._C_ops.embedding(data_2, parameter_23, -1, False)
        del data_2, parameter_23

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 1x11x128xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)
        del embedding_0, embedding_1

        # pd_op.embedding: (1x11x128xf32) <- (1x11xi64, 512x128xf32)
        embedding_2 = paddle._C_ops.embedding(slice_0, parameter_24, -1, False)
        del parameter_24, slice_0

        # pd_op.add: (1x11x128xf32) <- (1x11x128xf32, 1x11x128xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)
        del add_0, embedding_2

        # pd_op.layer_norm: (1x11x128xf32, 1x11xf32, 1x11xf32) <- (1x11x128xf32, 128xf32, 128xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_22, parameter_21, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_1, parameter_21, parameter_22

        # pd_op.matmul: (1x11x312xf32) <- (1x11x128xf32, 128x312xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_20, False, False)
        del layer_norm_0, parameter_20

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_19)
        del matmul_0, parameter_19

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_1 = paddle._C_ops.matmul(add_2, parameter_16, False, False)

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_15)
        del matmul_1

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_2 = paddle._C_ops.matmul(add_2, parameter_14, False, False)

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_13)
        del matmul_2

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_3 = paddle._C_ops.matmul(add_2, parameter_12, False, False)

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_5 = paddle._C_ops.add(matmul_3, parameter_11)
        del matmul_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 11, 12, 26]

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_3, full_int_array_4)
        del add_3

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_4, full_int_array_4)
        del add_4

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_5, full_int_array_4)
        del add_5

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x26xf32, 1x12x11x26xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_0, transpose_1, False, True)
        del transpose_0, transpose_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.196116"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_4, full_2, float("0"), True)
        del matmul_4

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_6 = paddle._C_ops.add(scale_2, scale_1)
        del scale_2

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_0 = paddle._C_ops.softmax(add_6, -1)
        del add_6

        # pd_op.matmul: (1x12x11x26xf32) <- (1x12x11x11xf32, 1x12x11x26xf32)
        matmul_5 = paddle._C_ops.matmul(softmax_0, transpose_2, False, False)
        del softmax_0, transpose_2

        # pd_op.transpose: (1x11x12x26xf32) <- (1x12x11x26xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_5, [0, 2, 1, 3])
        del matmul_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [0, 0, -1]

        # pd_op.reshape: (1x11x312xf32) <- (1x11x12x26xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_5)
        del transpose_3

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_3, parameter_10, False, False)
        del reshape_3

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_7 = paddle._C_ops.add(matmul_6, parameter_9)
        del matmul_6

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 1x11x312xf32)
        add_8 = paddle._C_ops.add(add_2, add_7)
        del add_2, add_7

        # pd_op.layer_norm: (1x11x312xf32, 1x11xf32, 1x11xf32) <- (1x11x312xf32, 312xf32, 312xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_8

        # pd_op.matmul: (1x11x1248xf32) <- (1x11x312xf32, 312x1248xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_3, parameter_6, False, False)

        # pd_op.add: (1x11x1248xf32) <- (1x11x1248xf32, 1248xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_5)
        del matmul_7

        # pd_op.gelu: (1x11x1248xf32) <- (1x11x1248xf32)
        gelu_0 = paddle._C_ops.gelu(add_9, False)
        del add_9

        # pd_op.matmul: (1x11x312xf32) <- (1x11x1248xf32, 1248x312xf32)
        matmul_8 = paddle._C_ops.matmul(gelu_0, parameter_4, False, False)
        del gelu_0

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_10 = paddle._C_ops.add(matmul_8, parameter_3)
        del matmul_8

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 1x11x312xf32)
        add_11 = paddle._C_ops.add(add_10, layer_norm_3)
        del add_10, layer_norm_3

        # pd_op.layer_norm: (1x11x312xf32, 1x11xf32, 1x11xf32) <- (1x11x312xf32, 312xf32, 312xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_11

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_16, False, False)

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_15)
        del matmul_9

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_14, False, False)

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_13)
        del matmul_10

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_6, parameter_12, False, False)

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_14 = paddle._C_ops.add(matmul_11, parameter_11)
        del matmul_11

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_12, full_int_array_4)
        del add_12

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_13, full_int_array_4)
        del add_13

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_14, full_int_array_4)
        del add_14

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x26xf32, 1x12x11x26xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_4, transpose_5, False, True)
        del transpose_4, transpose_5

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_12, full_2, float("0"), True)
        del matmul_12

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_15 = paddle._C_ops.add(scale_3, scale_1)
        del scale_3

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_1 = paddle._C_ops.softmax(add_15, -1)
        del add_15

        # pd_op.matmul: (1x12x11x26xf32) <- (1x12x11x11xf32, 1x12x11x26xf32)
        matmul_13 = paddle._C_ops.matmul(softmax_1, transpose_6, False, False)
        del softmax_1, transpose_6

        # pd_op.transpose: (1x11x12x26xf32) <- (1x12x11x26xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_13, [0, 2, 1, 3])
        del matmul_13

        # pd_op.reshape: (1x11x312xf32) <- (1x11x12x26xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_5)
        del transpose_7

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_14 = paddle._C_ops.matmul(reshape_7, parameter_10, False, False)
        del reshape_7

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_16 = paddle._C_ops.add(matmul_14, parameter_9)
        del matmul_14

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 1x11x312xf32)
        add_17 = paddle._C_ops.add(layer_norm_6, add_16)
        del add_16, layer_norm_6

        # pd_op.layer_norm: (1x11x312xf32, 1x11xf32, 1x11xf32) <- (1x11x312xf32, 312xf32, 312xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_17

        # pd_op.matmul: (1x11x1248xf32) <- (1x11x312xf32, 312x1248xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_9, parameter_6, False, False)

        # pd_op.add: (1x11x1248xf32) <- (1x11x1248xf32, 1248xf32)
        add_18 = paddle._C_ops.add(matmul_15, parameter_5)
        del matmul_15

        # pd_op.gelu: (1x11x1248xf32) <- (1x11x1248xf32)
        gelu_1 = paddle._C_ops.gelu(add_18, False)
        del add_18

        # pd_op.matmul: (1x11x312xf32) <- (1x11x1248xf32, 1248x312xf32)
        matmul_16 = paddle._C_ops.matmul(gelu_1, parameter_4, False, False)
        del gelu_1

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_19 = paddle._C_ops.add(matmul_16, parameter_3)
        del matmul_16

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 1x11x312xf32)
        add_20 = paddle._C_ops.add(add_19, layer_norm_9)
        del add_19, layer_norm_9

        # pd_op.layer_norm: (1x11x312xf32, 1x11xf32, 1x11xf32) <- (1x11x312xf32, 312xf32, 312xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_20

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_16, False, False)

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_21 = paddle._C_ops.add(matmul_17, parameter_15)
        del matmul_17

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_14, False, False)

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_22 = paddle._C_ops.add(matmul_18, parameter_13)
        del matmul_18

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_12, parameter_12, False, False)

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_23 = paddle._C_ops.add(matmul_19, parameter_11)
        del matmul_19

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_21, full_int_array_4)
        del add_21

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_22, full_int_array_4)
        del add_22

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_23, full_int_array_4)
        del add_23

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x26xf32, 1x12x11x26xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_8, transpose_9, False, True)
        del transpose_8, transpose_9

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_20, full_2, float("0"), True)
        del matmul_20

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_24 = paddle._C_ops.add(scale_4, scale_1)
        del scale_4

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_2 = paddle._C_ops.softmax(add_24, -1)
        del add_24

        # pd_op.matmul: (1x12x11x26xf32) <- (1x12x11x11xf32, 1x12x11x26xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_2, transpose_10, False, False)
        del softmax_2, transpose_10

        # pd_op.transpose: (1x11x12x26xf32) <- (1x12x11x26xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (1x11x312xf32) <- (1x11x12x26xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_5)
        del transpose_11

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_11, parameter_10, False, False)
        del reshape_11

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_25 = paddle._C_ops.add(matmul_22, parameter_9)
        del matmul_22

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 1x11x312xf32)
        add_26 = paddle._C_ops.add(layer_norm_12, add_25)
        del add_25, layer_norm_12

        # pd_op.layer_norm: (1x11x312xf32, 1x11xf32, 1x11xf32) <- (1x11x312xf32, 312xf32, 312xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_26, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_26

        # pd_op.matmul: (1x11x1248xf32) <- (1x11x312xf32, 312x1248xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_15, parameter_6, False, False)

        # pd_op.add: (1x11x1248xf32) <- (1x11x1248xf32, 1248xf32)
        add_27 = paddle._C_ops.add(matmul_23, parameter_5)
        del matmul_23

        # pd_op.gelu: (1x11x1248xf32) <- (1x11x1248xf32)
        gelu_2 = paddle._C_ops.gelu(add_27, False)
        del add_27

        # pd_op.matmul: (1x11x312xf32) <- (1x11x1248xf32, 1248x312xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_2, parameter_4, False, False)
        del gelu_2

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_28 = paddle._C_ops.add(matmul_24, parameter_3)
        del matmul_24

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 1x11x312xf32)
        add_29 = paddle._C_ops.add(add_28, layer_norm_15)
        del add_28, layer_norm_15

        # pd_op.layer_norm: (1x11x312xf32, 1x11xf32, 1x11xf32) <- (1x11x312xf32, 312xf32, 312xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_29, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_29

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_16, False, False)
        del parameter_16

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_30 = paddle._C_ops.add(matmul_25, parameter_15)
        del matmul_25, parameter_15

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_14, False, False)
        del parameter_14

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_31 = paddle._C_ops.add(matmul_26, parameter_13)
        del matmul_26, parameter_13

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_27 = paddle._C_ops.matmul(layer_norm_18, parameter_12, False, False)
        del parameter_12

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_32 = paddle._C_ops.add(matmul_27, parameter_11)
        del matmul_27, parameter_11

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_30, full_int_array_4)
        del add_30

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_31, full_int_array_4)
        del add_31

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x11x12x26xf32) <- (1x11x312xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_32, full_int_array_4)
        del add_32, full_int_array_4

        # pd_op.transpose: (1x12x11x26xf32) <- (1x11x12x26xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x26xf32, 1x12x11x26xf32)
        matmul_28 = paddle._C_ops.matmul(transpose_12, transpose_13, False, True)
        del transpose_12, transpose_13

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_28, full_2, float("0"), True)
        del full_2, matmul_28

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_33 = paddle._C_ops.add(scale_5, scale_1)
        del scale_1, scale_5

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_3 = paddle._C_ops.softmax(add_33, -1)
        del add_33

        # pd_op.matmul: (1x12x11x26xf32) <- (1x12x11x11xf32, 1x12x11x26xf32)
        matmul_29 = paddle._C_ops.matmul(softmax_3, transpose_14, False, False)
        del softmax_3, transpose_14

        # pd_op.transpose: (1x11x12x26xf32) <- (1x12x11x26xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_29, [0, 2, 1, 3])
        del matmul_29

        # pd_op.reshape: (1x11x312xf32) <- (1x11x12x26xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_5)
        del full_int_array_5, transpose_15

        # pd_op.matmul: (1x11x312xf32) <- (1x11x312xf32, 312x312xf32)
        matmul_30 = paddle._C_ops.matmul(reshape_15, parameter_10, False, False)
        del parameter_10, reshape_15

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_34 = paddle._C_ops.add(matmul_30, parameter_9)
        del matmul_30, parameter_9

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 1x11x312xf32)
        add_35 = paddle._C_ops.add(layer_norm_18, add_34)
        del add_34, layer_norm_18

        # pd_op.layer_norm: (1x11x312xf32, 1x11xf32, 1x11xf32) <- (1x11x312xf32, 312xf32, 312xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_8, parameter_7, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_35, parameter_7, parameter_8

        # pd_op.matmul: (1x11x1248xf32) <- (1x11x312xf32, 312x1248xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_21, parameter_6, False, False)
        del parameter_6

        # pd_op.add: (1x11x1248xf32) <- (1x11x1248xf32, 1248xf32)
        add_36 = paddle._C_ops.add(matmul_31, parameter_5)
        del matmul_31, parameter_5

        # pd_op.gelu: (1x11x1248xf32) <- (1x11x1248xf32)
        gelu_3 = paddle._C_ops.gelu(add_36, False)
        del add_36

        # pd_op.matmul: (1x11x312xf32) <- (1x11x1248xf32, 1248x312xf32)
        matmul_32 = paddle._C_ops.matmul(gelu_3, parameter_4, False, False)
        del gelu_3, parameter_4

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 312xf32)
        add_37 = paddle._C_ops.add(matmul_32, parameter_3)
        del matmul_32, parameter_3

        # pd_op.add: (1x11x312xf32) <- (1x11x312xf32, 1x11x312xf32)
        add_38 = paddle._C_ops.add(add_37, layer_norm_21)
        del add_37, layer_norm_21

        # pd_op.layer_norm: (1x11x312xf32, 1x11xf32, 1x11xf32) <- (1x11x312xf32, 312xf32, 312xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_18, parameter_17, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_38, parameter_17, parameter_18

        # pd_op.slice: (1x312xf32) <- (1x11x312xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            layer_norm_24, [1], full_int_array_2, full_int_array_0, [1], [1]
        )
        del full_int_array_0, full_int_array_2

        # pd_op.matmul: (1x312xf32) <- (1x312xf32, 312x312xf32)
        matmul_33 = paddle._C_ops.matmul(slice_1, parameter_2, False, False)
        del parameter_2, slice_1

        # pd_op.add: (1x312xf32) <- (1x312xf32, 312xf32)
        add_39 = paddle._C_ops.add(matmul_33, parameter_1)
        del matmul_33, parameter_1

        # pd_op.tanh: (1x312xf32) <- (1x312xf32)
        tanh_0 = paddle._C_ops.tanh(add_39)
        del add_39, layer_norm_24

        return tanh_0
