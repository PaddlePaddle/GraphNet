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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
    ):
        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 1x-1x512xf32)
        add_0 = paddle._C_ops.add(data_4, data_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [512]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_1

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x512xf32, 512x512xf32)
        matmul_0 = paddle._C_ops.matmul(add_0, slice_0, False, False)

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_1 = paddle._C_ops.add(matmul_0, slice_1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [0, 0, 8, 64]

        # pd_op.reshape: (1x-1x8x64xf32) <- (1x-1x512xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_2)

        # pd_op.transpose: (1x8x-1x64xf32) <- (1x-1x8x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1024]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_3

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_1, [0], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x512xf32, 512x512xf32)
        matmul_1 = paddle._C_ops.matmul(add_0, slice_2, False, False)

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_2 = paddle._C_ops.add(matmul_1, slice_3)

        # pd_op.reshape: (1x-1x8x64xf32) <- (1x-1x512xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_2, full_int_array_2)

        # pd_op.transpose: (1x8x-1x64xf32) <- (1x-1x8x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_4

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_4, [1], []
        )
        del data_0

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del data_1

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x512xf32, 512x512xf32)
        matmul_2 = paddle._C_ops.matmul(data_4, slice_4, False, False)

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_3 = paddle._C_ops.add(matmul_2, slice_5)

        # pd_op.reshape: (1x-1x8x64xf32) <- (1x-1x512xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_3, full_int_array_2)

        # pd_op.transpose: (1x8x-1x64xf32) <- (1x-1x8x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (1x8x-1x-1xf32) <- (1x8x-1x64xf32, 1x8x-1x64xf32)
        matmul_3 = paddle._C_ops.matmul(transpose_0, transpose_1, False, True)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_20 = full_0

        # pd_op.scale: (1x8x-1x-1xf32) <- (1x8x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_3, full_0, float("0"), True)
        del matmul_3

        # pd_op.softmax: (1x8x-1x-1xf32) <- (1x8x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (1x8x-1x64xf32) <- (1x8x-1x-1xf32, 1x8x-1x64xf32)
        matmul_4 = paddle._C_ops.matmul(softmax_0, transpose_2, False, False)

        # pd_op.transpose: (1x-1x8x64xf32) <- (1x8x-1x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.shape64: (4xi64) <- (1x-1x8x64xf32)
        shape64_0 = paddle._C_ops.shape64(transpose_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del shape64_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [0, 0, 512]

        # pd_op.reshape: (1x-1x512xf32) <- (1x-1x8x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_7)

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x512xf32, 512x512xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_4 = paddle._C_ops.add(matmul_5, parameter_18)
        del parameter_18

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 1x-1x512xf32)
        add_5 = paddle._C_ops.add(data_4, add_4)
        del data_4

        # pd_op.layer_norm: (1x-1x512xf32, 1x-1xf32, 1x-1xf32) <- (1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_17, parameter_16, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_16, parameter_17

        # pd_op.matmul: (1x-1x2048xf32) <- (1x-1x512xf32, 512x2048xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_1, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x-1x2048xf32) <- (1x-1x2048xf32, 2048xf32)
        add_6 = paddle._C_ops.add(matmul_6, parameter_14)
        del parameter_14

        # pd_op.gelu: (1x-1x2048xf32) <- (1x-1x2048xf32)
        gelu_0 = paddle._C_ops.gelu(add_6, False)

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x2048xf32, 2048x512xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_7 = paddle._C_ops.add(matmul_7, parameter_12)
        del parameter_12

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 1x-1x512xf32)
        add_8 = paddle._C_ops.add(layer_norm_1, add_7)

        # pd_op.layer_norm: (1x-1x512xf32, 1x-1xf32, 1x-1xf32) <- (1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_11, parameter_10, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_10, parameter_11

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 1x-1x512xf32)
        add_9 = paddle._C_ops.add(layer_norm_4, data_5)
        del data_5

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_3, [0], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x512xf32, 512x512xf32)
        matmul_8 = paddle._C_ops.matmul(add_9, slice_7, False, False)

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_10 = paddle._C_ops.add(matmul_8, slice_8)

        # pd_op.reshape: (1x-1x8x64xf32) <- (1x-1x512xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_10, full_int_array_2)

        # pd_op.transpose: (1x8x-1x64xf32) <- (1x-1x8x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_2, [1], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_3, [0], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x512xf32, 512x512xf32)
        matmul_9 = paddle._C_ops.matmul(add_9, slice_9, False, False)

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_11 = paddle._C_ops.add(matmul_9, slice_10)

        # pd_op.reshape: (1x-1x8x64xf32) <- (1x-1x512xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_11, full_int_array_2)

        # pd_op.transpose: (1x8x-1x64xf32) <- (1x-1x8x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_2, [1], full_int_array_3, full_int_array_4, [1], []
        )
        del data_2

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            data_3, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del data_3

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x512xf32, 512x512xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_4, slice_11, False, False)

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_12 = paddle._C_ops.add(matmul_10, slice_12)

        # pd_op.reshape: (1x-1x8x64xf32) <- (1x-1x512xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_12, full_int_array_2)
        del full_int_array_2

        # pd_op.transpose: (1x8x-1x64xf32) <- (1x-1x8x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.matmul: (1x8x-1x-1xf32) <- (1x8x-1x64xf32, 1x8x-1x64xf32)
        matmul_11 = paddle._C_ops.matmul(transpose_4, transpose_5, False, True)

        # pd_op.scale: (1x8x-1x-1xf32) <- (1x8x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_11, full_0, float("0"), True)
        del matmul_11

        # pd_op.softmax: (1x8x-1x-1xf32) <- (1x8x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (1x8x-1x64xf32) <- (1x8x-1x-1xf32, 1x8x-1x64xf32)
        matmul_12 = paddle._C_ops.matmul(softmax_1, transpose_6, False, False)

        # pd_op.transpose: (1x-1x8x64xf32) <- (1x8x-1x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.shape64: (4xi64) <- (1x-1x8x64xf32)
        shape64_1 = paddle._C_ops.shape64(transpose_7)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del full_int_array_5, full_int_array_6, shape64_1

        # pd_op.reshape: (1x-1x512xf32) <- (1x-1x8x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_7)
        del full_int_array_7

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x512xf32, 512x512xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_13 = paddle._C_ops.add(matmul_13, parameter_8)
        del parameter_8

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 1x-1x512xf32)
        add_14 = paddle._C_ops.add(layer_norm_4, add_13)

        # pd_op.layer_norm: (1x-1x512xf32, 1x-1xf32, 1x-1xf32) <- (1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_7, layer_norm_8, layer_norm_9 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_14, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_6, parameter_7

        # pd_op.matmul: (1x-1x2048xf32) <- (1x-1x512xf32, 512x2048xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_7, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (1x-1x2048xf32) <- (1x-1x2048xf32, 2048xf32)
        add_15 = paddle._C_ops.add(matmul_14, parameter_4)
        del parameter_4

        # pd_op.gelu: (1x-1x2048xf32) <- (1x-1x2048xf32)
        gelu_1 = paddle._C_ops.gelu(add_15, False)

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x2048xf32, 2048x512xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_16 = paddle._C_ops.add(matmul_15, parameter_2)
        del parameter_2

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 1x-1x512xf32)
        add_17 = paddle._C_ops.add(layer_norm_7, add_16)

        # pd_op.layer_norm: (1x-1x512xf32, 1x-1xf32, 1x-1xf32) <- (1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_0, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del (
            add_0,
            add_1,
            add_10,
            add_11,
            add_12,
            add_13,
            add_14,
            add_15,
            add_16,
            add_17,
            add_2,
            add_3,
            add_4,
            add_5,
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
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            full_0,
            full_int_array_0,
            full_int_array_1,
            full_int_array_3,
            full_int_array_4,
            gelu_0,
            gelu_1,
            layer_norm_1,
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
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_2,
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_8,
            matmul_9,
            parameter_0,
            parameter_1,
            reshape_3,
            reshape_7,
            slice_0,
            slice_1,
            slice_10,
            slice_11,
            slice_12,
            slice_2,
            slice_3,
            slice_4,
            slice_5,
            slice_7,
            slice_8,
            slice_9,
            softmax_0,
            softmax_1,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
        )

        return layer_norm_0
