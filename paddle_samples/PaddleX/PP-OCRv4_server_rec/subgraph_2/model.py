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
        data_0,
    ):
        # pd_op.assign: (-1x1024x1x40xf32) <- (-1x1024x1x40xf32)
        assign_0 = data_0
        del data_0

        # pd_op.conv2d: (-1x128x1x40xf32) <- (-1x1024x1x40xf32, 128x1024x1x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            assign_0, parameter_52, [1, 1], [0, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_52

        # pd_op.batch_norm_: (-1x128x1x40xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x1x40xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
                parameter_51,
                parameter_50,
                parameter_49,
                parameter_48,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_0, parameter_48, parameter_49, parameter_50, parameter_51

        # pd_op.swish: (-1x128x1x40xf32) <- (-1x128x1x40xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (-1x120x1x40xf32) <- (-1x128x1x40xf32, 120x128x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            swish_0, parameter_47, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_47, swish_0

        # pd_op.batch_norm_: (-1x120x1x40xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x1x40xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_46,
                parameter_45,
                parameter_44,
                parameter_43,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_1, parameter_43, parameter_44, parameter_45, parameter_46

        # pd_op.swish: (-1x120x1x40xf32) <- (-1x120x1x40xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__6)
        del batch_norm__6

        # pd_op.shape64: (4xi64) <- (-1x120x1x40xf32)
        shape64_0 = paddle._C_ops.shape64(swish_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del full_int_array_0, full_int_array_1, shape64_0

        # pd_op.flatten: (-1x120x40xf32) <- (-1x120x1x40xf32)
        flatten_0 = paddle._C_ops.flatten(swish_1, 2, 3)
        del swish_1

        # pd_op.transpose: (-1x40x120xf32) <- (-1x120x40xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (-1x40x120xf32, -1x40xf32, -1x40xf32) <- (-1x40x120xf32, 120xf32, 120xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_42, parameter_41, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_41, parameter_42

        # pd_op.matmul: (-1x40x360xf32) <- (-1x40x120xf32, 120x360xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_40, False, False)
        del layer_norm_0, parameter_40

        # pd_op.add: (-1x40x360xf32) <- (-1x40x360xf32, 360xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_39)
        del matmul_0, parameter_39

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_2 = [0, -1, 3, 8, 15]

        # pd_op.reshape: (-1x-1x3x8x15xf32) <- (-1x40x360xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_0, full_int_array_2)
        del add_0, full_int_array_2

        # pd_op.transpose: (3x-1x8x-1x15xf32) <- (-1x-1x3x8x15xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (-1x8x-1x15xf32) <- (3x-1x8x-1x15xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del full_int_array_3, full_int_array_4

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.258199"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x8x-1x15xf32) <- (-1x8x-1x15xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_1, full_0, float("0"), True)
        del full_0, slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.slice: (-1x8x-1x15xf32) <- (3x-1x8x-1x15xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del full_int_array_5, full_int_array_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [3]

        # pd_op.slice: (-1x8x-1x15xf32) <- (3x-1x8x-1x15xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_7, full_int_array_8, [1], [0]
        )
        del full_int_array_7, full_int_array_8, transpose_1

        # pd_op.transpose: (-1x8x15x-1xf32) <- (-1x8x-1x15xf32)
        transpose_2 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])
        del slice_2

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x15xf32, -1x8x15x-1xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_2, False, False)
        del scale_0, transpose_2

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(matmul_1, -1)
        del matmul_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x8x-1x-1xf32, -1x8x-1x-1xui8) <- (-1x8x-1x-1xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_1, softmax_1

        # pd_op.matmul: (-1x8x-1x15xf32) <- (-1x8x-1x-1xf32, -1x8x-1x15xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, slice_3, False, False)
        del dropout_0, slice_3

        # pd_op.transpose: (-1x-1x8x15xf32) <- (-1x8x-1x15xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_9 = [0, -1, 120]

        # pd_op.reshape: (-1x-1x120xf32) <- (-1x-1x8x15xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_3, full_int_array_9)
        del full_int_array_9, transpose_3

        # pd_op.matmul: (-1x-1x120xf32) <- (-1x-1x120xf32, 120x120xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_1, parameter_38, False, False)
        del parameter_38, reshape_1

        # pd_op.add: (-1x-1x120xf32) <- (-1x-1x120xf32, 120xf32)
        add_1 = paddle._C_ops.add(matmul_3, parameter_37)
        del matmul_3, parameter_37

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x-1x120xf32, -1x-1x120xui8) <- (-1x-1x120xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_1, None, full_2, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_1, full_2

        # pd_op.add: (-1x40x120xf32) <- (-1x40x120xf32, -1x-1x120xf32)
        add_2 = paddle._C_ops.add(transpose_0, dropout_2)
        del dropout_2, transpose_0

        # pd_op.layer_norm: (-1x40x120xf32, -1x40xf32, -1x40xf32) <- (-1x40x120xf32, 120xf32, 120xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_2, parameter_36, parameter_35, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_35, parameter_36

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x120xf32, 120x240xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_3, parameter_34, False, False)
        del layer_norm_3, parameter_34

        # pd_op.add: (-1x40x240xf32) <- (-1x40x240xf32, 240xf32)
        add_3 = paddle._C_ops.add(matmul_4, parameter_33)
        del matmul_4, parameter_33

        # pd_op.swish: (-1x40x240xf32) <- (-1x40x240xf32)
        swish_2 = paddle._C_ops.swish(add_3)
        del add_3

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x40x240xf32, -1x40x240xui8) <- (-1x40x240xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                swish_2, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_3, swish_2

        # pd_op.matmul: (-1x40x120xf32) <- (-1x40x240xf32, 240x120xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_4, parameter_32, False, False)
        del dropout_4, parameter_32

        # pd_op.add: (-1x40x120xf32) <- (-1x40x120xf32, 120xf32)
        add_4 = paddle._C_ops.add(matmul_5, parameter_31)
        del matmul_5, parameter_31

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x40x120xf32, -1x40x120xui8) <- (-1x40x120xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_4, None, full_4, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_4, full_4

        # pd_op.add: (-1x40x120xf32) <- (-1x40x120xf32, -1x40x120xf32)
        add_5 = paddle._C_ops.add(add_2, dropout_6)
        del add_2, dropout_6

        # pd_op.layer_norm: (-1x40x120xf32, -1x40xf32, -1x40xf32) <- (-1x40x120xf32, 120xf32, 120xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_30, parameter_29, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_29, parameter_30

        # pd_op.matmul: (-1x40x360xf32) <- (-1x40x120xf32, 120x360xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_6, parameter_28, False, False)
        del layer_norm_6, parameter_28

        # pd_op.add: (-1x40x360xf32) <- (-1x40x360xf32, 360xf32)
        add_6 = paddle._C_ops.add(matmul_6, parameter_27)
        del matmul_6, parameter_27

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_10 = [0, -1, 3, 8, 15]

        # pd_op.reshape: (-1x-1x3x8x15xf32) <- (-1x40x360xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_6, full_int_array_10)
        del add_6, full_int_array_10

        # pd_op.transpose: (3x-1x8x-1x15xf32) <- (-1x-1x3x8x15xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])
        del reshape_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [1]

        # pd_op.slice: (-1x8x-1x15xf32) <- (3x-1x8x-1x15xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_11, full_int_array_12, [1], [0]
        )
        del full_int_array_11, full_int_array_12

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.258199"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x8x-1x15xf32) <- (-1x8x-1x15xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_4, full_5, float("0"), True)
        del full_5, slice_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [2]

        # pd_op.slice: (-1x8x-1x15xf32) <- (3x-1x8x-1x15xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_13, full_int_array_14, [1], [0]
        )
        del full_int_array_13, full_int_array_14

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [3]

        # pd_op.slice: (-1x8x-1x15xf32) <- (3x-1x8x-1x15xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_15, full_int_array_16, [1], [0]
        )
        del full_int_array_15, full_int_array_16, transpose_4

        # pd_op.transpose: (-1x8x15x-1xf32) <- (-1x8x-1x15xf32)
        transpose_5 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])
        del slice_5

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x15xf32, -1x8x15x-1xf32)
        matmul_7 = paddle._C_ops.matmul(scale_1, transpose_5, False, False)
        del scale_1, transpose_5

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(matmul_7, -1)
        del matmul_7

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x8x-1x-1xf32, -1x8x-1x-1xui8) <- (-1x8x-1x-1xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_6, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_6, softmax_2

        # pd_op.matmul: (-1x8x-1x15xf32) <- (-1x8x-1x-1xf32, -1x8x-1x15xf32)
        matmul_8 = paddle._C_ops.matmul(dropout_8, slice_6, False, False)
        del dropout_8, slice_6

        # pd_op.transpose: (-1x-1x8x15xf32) <- (-1x8x-1x15xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_17 = [0, -1, 120]

        # pd_op.reshape: (-1x-1x120xf32) <- (-1x-1x8x15xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_6, full_int_array_17)
        del full_int_array_17, transpose_6

        # pd_op.matmul: (-1x-1x120xf32) <- (-1x-1x120xf32, 120x120xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_3, parameter_26, False, False)
        del parameter_26, reshape_3

        # pd_op.add: (-1x-1x120xf32) <- (-1x-1x120xf32, 120xf32)
        add_7 = paddle._C_ops.add(matmul_9, parameter_25)
        del matmul_9, parameter_25

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x-1x120xf32, -1x-1x120xui8) <- (-1x-1x120xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_7, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_7, full_7

        # pd_op.add: (-1x40x120xf32) <- (-1x40x120xf32, -1x-1x120xf32)
        add_8 = paddle._C_ops.add(add_5, dropout_10)
        del add_5, dropout_10

        # pd_op.layer_norm: (-1x40x120xf32, -1x40xf32, -1x40xf32) <- (-1x40x120xf32, 120xf32, 120xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_24, parameter_23, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_23, parameter_24

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x120xf32, 120x240xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_9, parameter_22, False, False)
        del layer_norm_9, parameter_22

        # pd_op.add: (-1x40x240xf32) <- (-1x40x240xf32, 240xf32)
        add_9 = paddle._C_ops.add(matmul_10, parameter_21)
        del matmul_10, parameter_21

        # pd_op.swish: (-1x40x240xf32) <- (-1x40x240xf32)
        swish_3 = paddle._C_ops.swish(add_9)
        del add_9

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x40x240xf32, -1x40x240xui8) <- (-1x40x240xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                swish_3, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_8, swish_3

        # pd_op.matmul: (-1x40x120xf32) <- (-1x40x240xf32, 240x120xf32)
        matmul_11 = paddle._C_ops.matmul(dropout_12, parameter_20, False, False)
        del dropout_12, parameter_20

        # pd_op.add: (-1x40x120xf32) <- (-1x40x120xf32, 120xf32)
        add_10 = paddle._C_ops.add(matmul_11, parameter_19)
        del matmul_11, parameter_19

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x40x120xf32, -1x40x120xui8) <- (-1x40x120xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_10, None, full_9, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_10, full_9

        # pd_op.add: (-1x40x120xf32) <- (-1x40x120xf32, -1x40x120xf32)
        add_11 = paddle._C_ops.add(add_8, dropout_14)
        del add_8, dropout_14

        # pd_op.layer_norm: (-1x40x120xf32, -1x40xf32, -1x40xf32) <- (-1x40x120xf32, 120xf32, 120xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_18, parameter_17, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_11, parameter_17, parameter_18

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [0, 1, 40, 120]

        # pd_op.reshape: (-1x1x40x120xf32) <- (-1x40x120xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(layer_norm_12, full_int_array_18)
        del full_int_array_18, layer_norm_12

        # pd_op.transpose: (-1x120x1x40xf32) <- (-1x1x40x120xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_4, [0, 3, 1, 2])
        del reshape_4

        # pd_op.conv2d: (-1x1024x1x40xf32) <- (-1x120x1x40xf32, 1024x120x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            transpose_7, parameter_16, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16, transpose_7

        # pd_op.batch_norm_: (-1x1024x1x40xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x1x40xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_15,
                parameter_14,
                parameter_13,
                parameter_12,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_2, parameter_12, parameter_13, parameter_14, parameter_15

        # pd_op.swish: (-1x1024x1x40xf32) <- (-1x1024x1x40xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__12)
        del batch_norm__12

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x1024x1x40xf32, -1x1024x1x40xf32]) <- (-1x1024x1x40xf32, -1x1024x1x40xf32)
        combine_0 = [assign_0, swish_4]
        del assign_0, swish_4

        # pd_op.concat: (-1x2048x1x40xf32) <- ([-1x1024x1x40xf32, -1x1024x1x40xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_10)
        del combine_0, full_10

        # pd_op.conv2d: (-1x128x1x40xf32) <- (-1x2048x1x40xf32, 128x2048x1x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            concat_0, parameter_11, [1, 1], [0, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_0, parameter_11

        # pd_op.batch_norm_: (-1x128x1x40xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x1x40xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_10,
                parameter_9,
                parameter_8,
                parameter_7,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_3, parameter_10, parameter_7, parameter_8, parameter_9

        # pd_op.swish: (-1x128x1x40xf32) <- (-1x128x1x40xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x120x1x40xf32) <- (-1x128x1x40xf32, 120x128x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            swish_5, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_6, swish_5

        # pd_op.batch_norm_: (-1x120x1x40xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x1x40xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_4, parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.swish: (-1x120x1x40xf32) <- (-1x120x1x40xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__24)
        del batch_norm__24

        # pd_op.shape64: (4xi64) <- (-1x120x1x40xf32)
        shape64_1 = paddle._C_ops.shape64(swish_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_19, full_int_array_20, [1], [0]
        )
        del full_int_array_19, full_int_array_20, shape64_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [2]

        # pd_op.squeeze: (-1x120x40xf32) <- (-1x120x1x40xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(swish_6, full_int_array_21)
        del full_int_array_21, swish_6

        # pd_op.transpose: (-1x40x120xf32) <- (-1x120x40xf32)
        transpose_8 = paddle._C_ops.transpose(squeeze_0, [0, 2, 1])
        del squeeze_0

        # pd_op.matmul: (-1x40x6625xf32) <- (-1x40x120xf32, 120x6625xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_8, parameter_1, False, False)
        del parameter_1, transpose_8

        # pd_op.add: (-1x40x6625xf32) <- (-1x40x6625xf32, 6625xf32)
        add_12 = paddle._C_ops.add(matmul_12, parameter_0)
        del matmul_12, parameter_0

        # pd_op.softmax: (-1x40x6625xf32) <- (-1x40x6625xf32)
        softmax_0 = paddle._C_ops.softmax(add_12, 2)
        del add_12

        return softmax_0
