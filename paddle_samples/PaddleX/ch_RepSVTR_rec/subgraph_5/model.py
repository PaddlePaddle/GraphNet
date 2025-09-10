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
        # pd_op.assign: (4x384x1x40xf32) <- (4x384x1x40xf32)
        assign_0 = data_0
        del data_0

        # pd_op.conv2d: (4x48x1x40xf32) <- (4x384x1x40xf32, 48x384x1x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            assign_0, parameter_52, [1, 1], [0, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_52

        # pd_op.batch_norm_: (4x48x1x40xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (4x48x1x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_48, parameter_49, parameter_50, parameter_51

        # pd_op.swish: (4x48x1x40xf32) <- (4x48x1x40xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.conv2d: (4x256x1x40xf32) <- (4x48x1x40xf32, 256x48x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            swish_0, parameter_47, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_47

        # pd_op.batch_norm_: (4x256x1x40xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x1x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_43, parameter_44, parameter_45, parameter_46

        # pd_op.swish: (4x256x1x40xf32) <- (4x256x1x40xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.flatten: (4x256x40xf32) <- (4x256x1x40xf32)
        flatten_0 = paddle._C_ops.flatten(swish_1, 2, 3)

        # pd_op.transpose: (4x40x256xf32) <- (4x256x40xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (4x40x256xf32, 4x40xf32, 4x40xf32) <- (4x40x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_42, parameter_41, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_41, parameter_42

        # pd_op.matmul: (4x40x768xf32) <- (4x40x256xf32, 256x768xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_40, False, False)
        del parameter_40

        # pd_op.add: (4x40x768xf32) <- (4x40x768xf32, 768xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_39)
        del parameter_39

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_0 = [0, -1, 3, 8, 32]

        # pd_op.reshape: (4x40x3x8x32xf32) <- (4x40x768xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_0)

        # pd_op.transpose: (3x4x8x40x32xf32) <- (4x40x3x8x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_2

        # pd_op.slice: (4x8x40x32xf32) <- (3x4x8x40x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_0

        # pd_op.scale: (4x8x40x32xf32) <- (4x8x40x32xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float("0"), True)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_3

        # pd_op.slice: (4x8x40x32xf32) <- (3x4x8x40x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_4

        # pd_op.slice: (4x8x40x32xf32) <- (3x4x8x40x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (4x8x32x40xf32) <- (4x8x40x32xf32)
        transpose_2 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (4x8x40x40xf32) <- (4x8x40x32xf32, 4x8x32x40xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_2, False, False)

        # pd_op.softmax: (4x8x40x40xf32) <- (4x8x40x40xf32)
        softmax_0 = paddle._C_ops.softmax(matmul_1, -1)
        del matmul_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_1

        # pd_op.dropout: (4x8x40x40xf32, 4x8x40x40xui8) <- (4x8x40x40xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (4x8x40x32xf32) <- (4x8x40x40xf32, 4x8x40x32xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, slice_2, False, False)

        # pd_op.transpose: (4x40x8x32xf32) <- (4x8x40x32xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [0, -1, 256]

        # pd_op.reshape: (4x40x256xf32) <- (4x40x8x32xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_3, full_int_array_5)

        # pd_op.matmul: (4x40x256xf32) <- (4x40x256xf32, 256x256xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_1, parameter_38, False, False)
        del parameter_38

        # pd_op.add: (4x40x256xf32) <- (4x40x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_3, parameter_37)
        del parameter_37

        # pd_op.dropout: (4x40x256xf32, 4x40x256xui8) <- (4x40x256xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_2, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_2

        # pd_op.add: (4x40x256xf32) <- (4x40x256xf32, 4x40x256xf32)
        add_3 = paddle._C_ops.add(transpose_0, dropout_2)

        # pd_op.layer_norm: (4x40x256xf32, 4x40xf32, 4x40xf32) <- (4x40x256xf32, 256xf32, 256xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_36, parameter_35, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_35, parameter_36

        # pd_op.matmul: (4x40x512xf32) <- (4x40x256xf32, 256x512xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_3, parameter_34, False, False)
        del parameter_34

        # pd_op.add: (4x40x512xf32) <- (4x40x512xf32, 512xf32)
        add_4 = paddle._C_ops.add(matmul_4, parameter_33)
        del parameter_33

        # pd_op.swish: (4x40x512xf32) <- (4x40x512xf32)
        swish_2 = paddle._C_ops.swish(add_4)

        # pd_op.dropout: (4x40x512xf32, 4x40x512xui8) <- (4x40x512xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                swish_2, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del swish_2

        # pd_op.matmul: (4x40x256xf32) <- (4x40x512xf32, 512x256xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_4, parameter_32, False, False)
        del parameter_32

        # pd_op.add: (4x40x256xf32) <- (4x40x256xf32, 256xf32)
        add_5 = paddle._C_ops.add(matmul_5, parameter_31)
        del parameter_31

        # pd_op.dropout: (4x40x256xf32, 4x40x256xui8) <- (4x40x256xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_5, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_5

        # pd_op.add: (4x40x256xf32) <- (4x40x256xf32, 4x40x256xf32)
        add_6 = paddle._C_ops.add(add_3, dropout_6)

        # pd_op.layer_norm: (4x40x256xf32, 4x40xf32, 4x40xf32) <- (4x40x256xf32, 256xf32, 256xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_30, parameter_29, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_29, parameter_30

        # pd_op.matmul: (4x40x768xf32) <- (4x40x256xf32, 256x768xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_6, parameter_28, False, False)
        del parameter_28

        # pd_op.add: (4x40x768xf32) <- (4x40x768xf32, 768xf32)
        add_7 = paddle._C_ops.add(matmul_6, parameter_27)
        del parameter_27

        # pd_op.reshape: (4x40x3x8x32xf32) <- (4x40x768xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_7, full_int_array_0)
        del full_int_array_0

        # pd_op.transpose: (3x4x8x40x32xf32) <- (4x40x3x8x32xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])
        del reshape_2

        # pd_op.slice: (4x8x40x32xf32) <- (3x4x8x40x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.scale: (4x8x40x32xf32) <- (4x8x40x32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_3, full_0, float("0"), True)
        del slice_3

        # pd_op.slice: (4x8x40x32xf32) <- (3x4x8x40x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x8x40x32xf32) <- (3x4x8x40x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (4x8x32x40xf32) <- (4x8x40x32xf32)
        transpose_5 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])
        del slice_4

        # pd_op.matmul: (4x8x40x40xf32) <- (4x8x40x32xf32, 4x8x32x40xf32)
        matmul_7 = paddle._C_ops.matmul(scale_1, transpose_5, False, False)

        # pd_op.softmax: (4x8x40x40xf32) <- (4x8x40x40xf32)
        softmax_1 = paddle._C_ops.softmax(matmul_7, -1)
        del matmul_7

        # pd_op.dropout: (4x8x40x40xf32, 4x8x40x40xui8) <- (4x8x40x40xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (4x8x40x32xf32) <- (4x8x40x40xf32, 4x8x40x32xf32)
        matmul_8 = paddle._C_ops.matmul(dropout_8, slice_5, False, False)

        # pd_op.transpose: (4x40x8x32xf32) <- (4x8x40x32xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (4x40x256xf32) <- (4x40x8x32xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_6, full_int_array_5)
        del full_int_array_5

        # pd_op.matmul: (4x40x256xf32) <- (4x40x256xf32, 256x256xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_3, parameter_26, False, False)
        del parameter_26

        # pd_op.add: (4x40x256xf32) <- (4x40x256xf32, 256xf32)
        add_8 = paddle._C_ops.add(matmul_9, parameter_25)
        del parameter_25

        # pd_op.dropout: (4x40x256xf32, 4x40x256xui8) <- (4x40x256xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_8, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_8

        # pd_op.add: (4x40x256xf32) <- (4x40x256xf32, 4x40x256xf32)
        add_9 = paddle._C_ops.add(add_6, dropout_10)

        # pd_op.layer_norm: (4x40x256xf32, 4x40xf32, 4x40xf32) <- (4x40x256xf32, 256xf32, 256xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_24, parameter_23, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_23, parameter_24

        # pd_op.matmul: (4x40x512xf32) <- (4x40x256xf32, 256x512xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_9, parameter_22, False, False)
        del parameter_22

        # pd_op.add: (4x40x512xf32) <- (4x40x512xf32, 512xf32)
        add_10 = paddle._C_ops.add(matmul_10, parameter_21)
        del parameter_21

        # pd_op.swish: (4x40x512xf32) <- (4x40x512xf32)
        swish_3 = paddle._C_ops.swish(add_10)

        # pd_op.dropout: (4x40x512xf32, 4x40x512xui8) <- (4x40x512xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                swish_3, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del swish_3

        # pd_op.matmul: (4x40x256xf32) <- (4x40x512xf32, 512x256xf32)
        matmul_11 = paddle._C_ops.matmul(dropout_12, parameter_20, False, False)
        del parameter_20

        # pd_op.add: (4x40x256xf32) <- (4x40x256xf32, 256xf32)
        add_11 = paddle._C_ops.add(matmul_11, parameter_19)
        del parameter_19

        # pd_op.dropout: (4x40x256xf32, 4x40x256xui8) <- (4x40x256xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_11, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_11

        # pd_op.add: (4x40x256xf32) <- (4x40x256xf32, 4x40x256xf32)
        add_12 = paddle._C_ops.add(add_9, dropout_14)

        # pd_op.layer_norm: (4x40x256xf32, 4x40xf32, 4x40xf32) <- (4x40x256xf32, 256xf32, 256xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_18, parameter_17, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_17, parameter_18

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [0, 1, 40, 256]

        # pd_op.reshape: (4x1x40x256xf32) <- (4x40x256xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(layer_norm_12, full_int_array_6)
        del full_int_array_6

        # pd_op.transpose: (4x256x1x40xf32) <- (4x1x40x256xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_4, [0, 3, 1, 2])
        del reshape_4

        # pd_op.conv2d: (4x384x1x40xf32) <- (4x256x1x40xf32, 384x256x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            transpose_7, parameter_16, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16

        # pd_op.batch_norm_: (4x384x1x40xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (4x384x1x40xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_12, parameter_13, parameter_14, parameter_15

        # pd_op.swish: (4x384x1x40xf32) <- (4x384x1x40xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__12)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([4x384x1x40xf32, 4x384x1x40xf32]) <- (4x384x1x40xf32, 4x384x1x40xf32)
        combine_0 = [assign_0, swish_4]

        # pd_op.concat: (4x768x1x40xf32) <- ([4x384x1x40xf32, 4x384x1x40xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0

        # pd_op.conv2d: (4x48x1x40xf32) <- (4x768x1x40xf32, 48x768x1x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            concat_0, parameter_11, [1, 1], [0, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_11

        # pd_op.batch_norm_: (4x48x1x40xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (4x48x1x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_10, parameter_7, parameter_8, parameter_9

        # pd_op.swish: (4x48x1x40xf32) <- (4x48x1x40xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__18)

        # pd_op.conv2d: (4x256x1x40xf32) <- (4x48x1x40xf32, 256x48x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            swish_5, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_6

        # pd_op.batch_norm_: (4x256x1x40xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x1x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.swish: (4x256x1x40xf32) <- (4x256x1x40xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__24)

        # pd_op.squeeze: (4x256x40xf32) <- (4x256x1x40xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(swish_6, full_int_array_3)

        # pd_op.transpose: (4x40x256xf32) <- (4x256x40xf32)
        transpose_8 = paddle._C_ops.transpose(squeeze_0, [0, 2, 1])
        del squeeze_0

        # pd_op.matmul: (4x40x6625xf32) <- (4x40x256xf32, 256x6625xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_8, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (4x40x6625xf32) <- (4x40x6625xf32, 6625xf32)
        add_0 = paddle._C_ops.add(matmul_12, parameter_0)
        del (
            add_1,
            add_10,
            add_12,
            add_3,
            add_4,
            add_6,
            add_7,
            add_9,
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
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__2,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            concat_0,
            conv2d_0,
            conv2d_1,
            conv2d_2,
            conv2d_3,
            conv2d_4,
            dropout_0,
            dropout_1,
            dropout_10,
            dropout_11,
            dropout_12,
            dropout_13,
            dropout_14,
            dropout_15,
            dropout_2,
            dropout_3,
            dropout_4,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_8,
            dropout_9,
            full_0,
            full_1,
            full_2,
            full_int_array_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            layer_norm_0,
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
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_3,
            matmul_4,
            matmul_5,
            matmul_6,
            matmul_9,
            parameter_0,
            reshape_1,
            reshape_3,
            scale_0,
            scale_1,
            slice_2,
            slice_5,
            softmax_0,
            softmax_1,
            swish_0,
            swish_1,
            swish_4,
            swish_5,
            swish_6,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
        )

        return add_0
