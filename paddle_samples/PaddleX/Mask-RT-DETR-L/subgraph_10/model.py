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
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 1x-1x256xf32)
        add_0 = paddle._C_ops.add(data_2, data_3)
        del data_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [256]

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(add_0, slice_0, False, False)
        del slice_0

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_1 = paddle._C_ops.add(matmul_0, slice_1)
        del matmul_0, slice_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [0, 0, 8, 32]

        # pd_op.reshape: (-1x-1x8x32xf32) <- (-1x-1x256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_2)
        del add_1

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [512]

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_1, [0], full_int_array_1, full_int_array_3, [1], []
        )
        del full_int_array_1

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(add_0, slice_2, False, False)
        del add_0, slice_2

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_1, slice_3)
        del matmul_1, slice_3

        # pd_op.reshape: (-1x-1x8x32xf32) <- (-1x-1x256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_2, full_int_array_2)
        del add_2

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_4, [1], []
        )
        del data_0

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del data_1, full_int_array_3, full_int_array_4

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(data_2, slice_4, False, False)
        del slice_4

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_2, slice_5)
        del matmul_2, slice_5

        # pd_op.reshape: (-1x-1x8x32xf32) <- (-1x-1x256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_3, full_int_array_2)
        del add_3, full_int_array_2

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x-1x32xf32)
        matmul_3 = paddle._C_ops.matmul(transpose_0, transpose_1, False, True)
        del transpose_0, transpose_1

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_3, full_0, float("0"), True)
        del full_0, matmul_3

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_4 = paddle._C_ops.matmul(softmax_0, transpose_2, False, False)
        del softmax_0, transpose_2

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.shape64: (4xi64) <- (-1x-1x8x32xf32)
        shape64_0 = paddle._C_ops.shape64(transpose_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_5, [1], [0]
        )
        del full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del full_int_array_5, full_int_array_6, shape64_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [0, 0, 256]

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_7)
        del full_int_array_7, transpose_3

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_9, False, False)
        del parameter_9, reshape_3

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_4 = paddle._C_ops.add(matmul_5, parameter_8)
        del matmul_5, parameter_8

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_5 = paddle._C_ops.add(data_2, add_4)
        del add_4, data_2

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_5, parameter_6, parameter_7

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_1, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_6 = paddle._C_ops.add(matmul_6, parameter_4)
        del matmul_6, parameter_4

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_0 = paddle._C_ops.gelu(add_6, False)
        del add_6

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_3, False, False)
        del gelu_0, parameter_3

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_7 = paddle._C_ops.add(matmul_7, parameter_2)
        del matmul_7, parameter_2

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_8 = paddle._C_ops.add(layer_norm_1, add_7)
        del add_7, layer_norm_1

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_8, parameter_0, parameter_1

        return layer_norm_0
