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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
    ):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_0

        # pd_op.dropout: (16x144x512xf32, 16x144x512xui8) <- (16x144x512xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                data_1, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_1

        # pd_op.add: (16x144x512xf32) <- (16x144x512xf32, 16x144x512xf32)
        add_0 = paddle._C_ops.add(data_0, dropout_0)
        del data_0

        # pd_op.layer_norm: (16x144x512xf32, 16x144xf32, 16x144xf32) <- (16x144x512xf32, 512xf32, 512xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_17, parameter_16, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_16, parameter_17

        # pd_op.matmul: (16x144x512xf32) <- (16x144x512xf32, 512x512xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_1, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (16x144x512xf32) <- (16x144x512xf32, 512xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_14)
        del parameter_14

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [16, 144, 8, -1]

        # pd_op.reshape: (16x144x8x64xf32) <- (16x144x512xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_0)
        del full_int_array_0

        # pd_op.matmul: (16x96x512xf32) <- (16x96x512xf32, 512x512xf32)
        matmul_1 = paddle._C_ops.matmul(data_2, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (16x96x512xf32) <- (16x96x512xf32, 512xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_12)
        del parameter_12

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [16, 96, 8, -1]

        # pd_op.reshape: (16x96x8x64xf32) <- (16x96x512xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_2, full_int_array_1)

        # pd_op.matmul: (16x96x512xf32) <- (16x96x512xf32, 512x512xf32)
        matmul_2 = paddle._C_ops.matmul(data_2, parameter_11, False, False)
        del data_2, parameter_11

        # pd_op.add: (16x96x512xf32) <- (16x96x512xf32, 512xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_10)
        del parameter_10

        # pd_op.reshape: (16x96x8x64xf32) <- (16x96x512xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_3, full_int_array_1)
        del full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_2

        # pd_op.unsqueeze: (16x1x1xf32) <- (16x1xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_3, full_int_array_2)
        del data_3

        # pd_op.unsqueeze: (16x1x1x1xf32) <- (16x1x1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(unsqueeze_0, full_int_array_2)

        # pd_op.unsqueeze: (16x1x96xf32) <- (16x96xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_4, full_int_array_2)
        del data_4

        # pd_op.unsqueeze: (16x1x1x96xf32) <- (16x1x96xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(unsqueeze_2, full_int_array_2)

        # builtin.combine: ([16x144x8x64xf32, 16x96x8x64xf32]) <- (16x144x8x64xf32, 16x96x8x64xf32)
        combine_0 = [reshape_0, reshape_1]
        del reshape_0, reshape_1

        # pd_op.einsum: (16x8x144x96xf32, [0xf32, 0xf32], [16x144x8x64xf32, 16x96x8x64xf32]) <- ([16x144x8x64xf32, 16x96x8x64xf32])
        einsum_0, einsum_1, einsum_2 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_0, "blhe,bshe->bhls"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_0

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_0,
            split_1,
        ) = einsum_1
        del einsum_1

        # builtin.split: (16x144x8x64xf32, 16x96x8x64xf32) <- ([16x144x8x64xf32, 16x96x8x64xf32])
        (
            split_2,
            split_3,
        ) = einsum_2
        del einsum_2

        # pd_op.multiply: (16x8x144x96xf32) <- (16x8x144x96xf32, 16x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(einsum_0, unsqueeze_1)

        # pd_op.add: (16x8x144x96xf32) <- (16x8x144x96xf32, 16x1x1x96xf32)
        add_4 = paddle._C_ops.add(multiply_0, unsqueeze_3)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x8x144x96xf32) <- (16x8x144x96xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_4, full_1, float("0"), True)
        del add_4

        # pd_op.softmax: (16x8x144x96xf32) <- (16x8x144x96xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.dropout: (16x8x144x96xf32, 16x8x144x96xui8) <- (16x8x144x96xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # builtin.combine: ([16x8x144x96xf32, 16x96x8x64xf32]) <- (16x8x144x96xf32, 16x96x8x64xf32)
        combine_1 = [dropout_2, reshape_2]
        del dropout_2, reshape_2

        # pd_op.einsum: (16x144x8x64xf32, [0xf32, 0xf32], [16x8x144x96xf32, 16x96x8x64xf32]) <- ([16x8x144x96xf32, 16x96x8x64xf32])
        einsum_3, einsum_4, einsum_5 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_1, "bhls,bshd->blhd"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_1

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_4,
            split_5,
        ) = einsum_4
        del einsum_4

        # builtin.split: (16x8x144x96xf32, 16x96x8x64xf32) <- ([16x8x144x96xf32, 16x96x8x64xf32])
        (
            split_6,
            split_7,
        ) = einsum_5
        del einsum_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [16, 144, -1]

        # pd_op.reshape: (16x144x512xf32) <- (16x144x8x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(einsum_3, full_int_array_3)
        del full_int_array_3

        # pd_op.matmul: (16x144x512xf32) <- (16x144x512xf32, 512x512xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_3, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (16x144x512xf32) <- (16x144x512xf32, 512xf32)
        add_5 = paddle._C_ops.add(matmul_3, parameter_8)
        del parameter_8

        # pd_op.dropout: (16x144x512xf32, 16x144x512xui8) <- (16x144x512xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_5, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_5

        # pd_op.add: (16x144x512xf32) <- (16x144x512xf32, 16x144x512xf32)
        add_6 = paddle._C_ops.add(layer_norm_1, dropout_4)

        # pd_op.layer_norm: (16x144x512xf32, 16x144xf32, 16x144xf32) <- (16x144x512xf32, 512xf32, 512xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_6, parameter_7

        # pd_op.transpose: (16x512x144xf32) <- (16x144x512xf32)
        transpose_0 = paddle._C_ops.transpose(layer_norm_4, [0, 2, 1])

        # pd_op.assign: (2048x512x1xf32) <- (2048x512x1xf32)
        assign_7 = parameter_5
        del parameter_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_4

        # pd_op.unsqueeze: (2048x512x1x1xf32) <- (2048x512x1xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(assign_7, full_int_array_4)

        # pd_op.unsqueeze: (16x512x1x144xf32) <- (16x512x144xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(transpose_0, full_int_array_4)

        # pd_op.conv2d: (16x2048x1x144xf32) <- (16x512x1x144xf32, 2048x512x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            unsqueeze_5, unsqueeze_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 2048, 1, 1]

        # pd_op.reshape: (1x2048x1x1xf32) <- (2048xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_4, full_int_array_5)
        del full_int_array_5, parameter_4

        # pd_op.add: (16x2048x1x144xf32) <- (16x2048x1x144xf32, 1x2048x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_0, reshape_4)

        # pd_op.squeeze: (16x2048x144xf32) <- (16x2048x1x144xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(add_7, full_int_array_4)

        # pd_op.gelu: (16x2048x144xf32) <- (16x2048x144xf32)
        gelu_0 = paddle._C_ops.gelu(squeeze_0, False)

        # pd_op.dropout: (16x2048x144xf32, 16x2048x144xui8) <- (16x2048x144xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_0, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_0

        # pd_op.assign: (512x2048x1xf32) <- (512x2048x1xf32)
        assign_13 = parameter_3
        del parameter_3

        # pd_op.unsqueeze: (512x2048x1x1xf32) <- (512x2048x1xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(assign_13, full_int_array_4)

        # pd_op.unsqueeze: (16x2048x1x144xf32) <- (16x2048x144xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(dropout_6, full_int_array_4)

        # pd_op.conv2d: (16x512x1x144xf32) <- (16x2048x1x144xf32, 512x2048x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            unsqueeze_7, unsqueeze_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_2, full_int_array_6)
        del full_int_array_6, parameter_2

        # pd_op.add: (16x512x1x144xf32) <- (16x512x1x144xf32, 1x512x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_1, reshape_5)

        # pd_op.squeeze: (16x512x144xf32) <- (16x512x1x144xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(add_8, full_int_array_4)

        # pd_op.transpose: (16x144x512xf32) <- (16x512x144xf32)
        transpose_1 = paddle._C_ops.transpose(squeeze_1, [0, 2, 1])
        del squeeze_1

        # pd_op.dropout: (16x144x512xf32, 16x144x512xui8) <- (16x144x512xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_1, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_1

        # pd_op.add: (16x144x512xf32) <- (16x144x512xf32, 16x144x512xf32)
        add_9 = paddle._C_ops.add(layer_norm_4, dropout_8)

        # pd_op.layer_norm: (16x144x512xf32, 16x144xf32, 16x144xf32) <- (16x144x512xf32, 512xf32, 512xf32)
        layer_norm_0, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del (
            add_0,
            add_1,
            add_2,
            add_3,
            add_6,
            add_7,
            add_8,
            add_9,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            conv2d_0,
            conv2d_1,
            dropout_0,
            dropout_1,
            dropout_3,
            dropout_4,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_8,
            dropout_9,
            einsum_0,
            einsum_3,
            full_0,
            full_1,
            full_int_array_2,
            full_int_array_4,
            layer_norm_1,
            layer_norm_2,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            matmul_0,
            matmul_1,
            matmul_2,
            matmul_3,
            multiply_0,
            parameter_0,
            parameter_1,
            reshape_3,
            reshape_4,
            reshape_5,
            softmax_0,
            squeeze_0,
            transpose_0,
            unsqueeze_0,
            unsqueeze_1,
            unsqueeze_2,
            unsqueeze_3,
            unsqueeze_4,
            unsqueeze_5,
            unsqueeze_6,
            unsqueeze_7,
        )

        return layer_norm_0
