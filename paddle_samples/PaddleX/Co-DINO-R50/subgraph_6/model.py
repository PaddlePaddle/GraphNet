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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
    ):
        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 1x1100x256xf32)
        add_1 = paddle._C_ops.add(data_2, data_4)

        # pd_op.full: (1100x1100xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1100, 1100],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1100x1100xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1100, 1100],
            float("-inf"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1100x1100xf32) <- (1100x1100xb, 1100x1100xf32, 1100x1100xf32)
        where_0 = paddle._C_ops.where(data_3, full_0, full_1)
        del data_3, full_0, full_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [256]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_1

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(add_1, slice_0, False, False)

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_0, slice_1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [0, 0, 8, 32]

        # pd_op.reshape: (1x1100x8x32xf32) <- (1x1100x256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_2)

        # pd_op.transpose: (1x8x1100x32xf32) <- (1x1100x8x32xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [512]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_3

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_1, [0], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(add_1, slice_2, False, False)

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_1, slice_3)

        # pd_op.reshape: (1x1100x8x32xf32) <- (1x1100x256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_3, full_int_array_2)

        # pd_op.transpose: (1x8x1100x32xf32) <- (1x1100x8x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_4

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_4, [1], []
        )
        del data_0

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del data_1

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(data_2, slice_4, False, False)

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_4 = paddle._C_ops.add(matmul_2, slice_5)

        # pd_op.reshape: (1x1100x8x32xf32) <- (1x1100x256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_4, full_int_array_2)
        del full_int_array_2

        # pd_op.transpose: (1x8x1100x32xf32) <- (1x1100x8x32xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (1x8x1100x1100xf32) <- (1x8x1100x32xf32, 1x8x1100x32xf32)
        matmul_3 = paddle._C_ops.matmul(transpose_0, transpose_1, False, True)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x8x1100x1100xf32) <- (1x8x1100x1100xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_3, full_2, float("0"), True)
        del matmul_3

        # pd_op.add: (1x8x1100x1100xf32) <- (1x8x1100x1100xf32, 1100x1100xf32)
        add_5 = paddle._C_ops.add(scale_0, where_0)

        # pd_op.softmax: (1x8x1100x1100xf32) <- (1x8x1100x1100xf32)
        softmax_0 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.matmul: (1x8x1100x32xf32) <- (1x8x1100x1100xf32, 1x8x1100x32xf32)
        matmul_4 = paddle._C_ops.matmul(softmax_0, transpose_2, False, False)

        # pd_op.transpose: (1x1100x8x32xf32) <- (1x8x1100x32xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [0, 0, 256]

        # pd_op.reshape: (1x1100x256xf32) <- (1x1100x8x32xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_5)
        del full_int_array_5

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_2)
        del parameter_2

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 1x1100x256xf32)
        add_7 = paddle._C_ops.add(data_2, add_6)
        del data_2

        # pd_op.layer_norm: (1x1100x256xf32, 1x1100xf32, 1x1100xf32) <- (1x1100x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_0, parameter_1

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 1x1100x256xf32)
        add_0 = paddle._C_ops.add(layer_norm_0, data_4)
        del (
            add_1,
            add_2,
            add_3,
            add_4,
            add_6,
            add_7,
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            data_4,
            full_2,
            full_int_array_0,
            full_int_array_1,
            full_int_array_3,
            full_int_array_4,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
            matmul_1,
            matmul_2,
            matmul_5,
            reshape_3,
            scale_0,
            slice_0,
            slice_1,
            slice_2,
            slice_3,
            slice_4,
            slice_5,
            softmax_0,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            where_0,
        )

        return add_0
