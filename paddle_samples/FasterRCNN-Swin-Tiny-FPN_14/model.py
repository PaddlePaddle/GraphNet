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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.layer_norm: (2x42240x96xf32, 2x42240xf32, 2x42240xf32) <- (2x42240x96xf32, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                data_2, parameter_11, parameter_10, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [-1, 240, 176, 96]

        # pd_op.reshape: (2x240x176x96xf32) <- (2x42240x96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(layer_norm_0, full_int_array_6)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.pad: (2x245x182x96xf32) <- (2x240x176x96xf32, 1xf32)
        pad_0 = paddle._C_ops.pad(reshape_0, [0, 0, 0, 5, 0, 6, 0, 0], full_0)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_7 = [-1, 35, 7, 26, 7, 96]

        # pd_op.reshape: (2x35x7x26x7x96xf32) <- (2x245x182x96xf32, 6xi64)
        reshape_7 = paddle._C_ops.reshape(pad_0, full_int_array_7)

        # pd_op.transpose: (2x35x26x7x7x96xf32) <- (2x35x7x26x7x96xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_7, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [-1, 7, 7, 96]

        # pd_op.reshape: (1820x7x7x96xf32) <- (2x35x26x7x7x96xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_0, full_int_array_8)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_9 = [1820, 49, 96]

        # pd_op.reshape: (1820x49x96xf32) <- (1820x7x7x96xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(reshape_1, full_int_array_9)

        # pd_op.matmul: (1820x49x288xf32) <- (1820x49x96xf32, 96x288xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_2, parameter_9, False, False)

        # pd_op.add: (1820x49x288xf32) <- (1820x49x288xf32, 288xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_8)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_10 = [-1, 49, 3, 3, 32]

        # pd_op.reshape: (1820x49x3x3x32xf32) <- (1820x49x288xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_0, full_int_array_10)

        # pd_op.transpose: (3x1820x3x49x32xf32) <- (1820x49x3x3x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.slice: (1820x3x49x32xf32) <- (3x1820x3x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_2

        # pd_op.slice: (1820x3x49x32xf32) <- (3x1820x3x49x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (1820x3x49x32xf32) <- (3x1820x3x49x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1820x3x49x32xf32) <- (1820x3x49x32xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_2, full_1, float("0"), True)

        # pd_op.transpose: (1820x3x32x49xf32) <- (1820x3x49x32xf32)
        transpose_2 = paddle._C_ops.transpose(slice_3, [0, 1, 3, 2])

        # pd_op.matmul: (1820x3x49x49xf32) <- (1820x3x49x32xf32, 1820x3x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_2, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_0 = paddle._C_ops.flatten(data_0, 0, 1)

        # pd_op.index_select: (2401x3xf32) <- (169x3xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_1, flatten_0, 0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [49, 49, -1]

        # pd_op.reshape: (49x49x3xf32) <- (2401x3xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(index_select_0, full_int_array_11)

        # pd_op.transpose: (3x49x49xf32) <- (49x49x3xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_9, [2, 0, 1])

        # pd_op.unsqueeze: (1x3x49x49xf32) <- (3x49x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_3, full_int_array_0)

        # pd_op.add: (1820x3x49x49xf32) <- (1820x3x49x49xf32, 1x3x49x49xf32)
        add_6 = paddle._C_ops.add(matmul_1, unsqueeze_0)

        # pd_op.softmax: (1820x3x49x49xf32) <- (1820x3x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(add_6, -1)

        # pd_op.matmul: (1820x3x49x32xf32) <- (1820x3x49x49xf32, 1820x3x49x32xf32)
        matmul_5 = paddle._C_ops.matmul(softmax_0, slice_0, False, False)

        # pd_op.transpose: (1820x49x3x32xf32) <- (1820x3x49x32xf32)
        transpose_4 = paddle._C_ops.transpose(matmul_5, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_12 = [-1, 49, 96]

        # pd_op.reshape: (1820x49x96xf32) <- (1820x49x3x32xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_4, full_int_array_12)

        # pd_op.matmul: (1820x49x96xf32) <- (1820x49x96xf32, 96x96xf32)
        matmul_2 = paddle._C_ops.matmul(reshape_3, parameter_7, False, False)

        # pd_op.add: (1820x49x96xf32) <- (1820x49x96xf32, 96xf32)
        add_1 = paddle._C_ops.add(matmul_2, parameter_6)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1820, 7, 7, 96]

        # pd_op.reshape: (1820x7x7x96xf32) <- (1820x49x96xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_1, full_int_array_13)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_14 = [-1, 35, 26, 7, 7, 96]

        # pd_op.reshape: (2x35x26x7x7x96xf32) <- (1820x7x7x96xf32, 6xi64)
        reshape_10 = paddle._C_ops.reshape(reshape_4, full_int_array_14)

        # pd_op.transpose: (2x35x7x26x7x96xf32) <- (2x35x26x7x7x96xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_10, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [-1, 245, 182, 96]

        # pd_op.reshape: (2x245x182x96xf32) <- (2x35x7x26x7x96xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_5, full_int_array_15)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [240, 176]

        # pd_op.slice: (2x240x176x96xf32) <- (2x245x182x96xf32, 2xi64, 2xi64)
        slice_1 = paddle._C_ops.slice(
            reshape_5, [1, 2], full_int_array_4, full_int_array_5, [1, 1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_16 = [-1, 42240, 96]

        # pd_op.reshape: (2x42240x96xf32) <- (2x240x176x96xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(slice_1, full_int_array_16)

        # pd_op.add: (2x42240x96xf32) <- (2x42240x96xf32, 2x42240x96xf32)
        add_2 = paddle._C_ops.add(data_2, reshape_6)

        # pd_op.layer_norm: (2x42240x96xf32, 2x42240xf32, 2x42240xf32) <- (2x42240x96xf32, 96xf32, 96xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_2, parameter_5, parameter_4, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x42240x384xf32) <- (2x42240x96xf32, 96x384xf32)
        matmul_3 = paddle._C_ops.matmul(layer_norm_3, parameter_3, False, False)

        # pd_op.add: (2x42240x384xf32) <- (2x42240x384xf32, 384xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_2)

        # pd_op.gelu: (2x42240x384xf32) <- (2x42240x384xf32)
        gelu_0 = paddle._C_ops.gelu(add_3, False)

        # pd_op.matmul: (2x42240x96xf32) <- (2x42240x384xf32, 384x96xf32)
        matmul_4 = paddle._C_ops.matmul(gelu_0, parameter_1, False, False)

        # pd_op.add: (2x42240x96xf32) <- (2x42240x96xf32, 96xf32)
        add_4 = paddle._C_ops.add(matmul_4, parameter_0)

        # pd_op.add: (2x42240x96xf32) <- (2x42240x96xf32, 2x42240x96xf32)
        add_5 = paddle._C_ops.add(add_2, add_4)
        return (
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            reshape_0,
            full_0,
            pad_0,
            transpose_0,
            reshape_1,
            reshape_2,
            matmul_0,
            add_0,
            transpose_1,
            full_int_array_0,
            full_int_array_1,
            assign_0,
            full_int_array_2,
            assign_1,
            full_int_array_3,
            slice_0,
            full_1,
            scale_0,
            transpose_2,
            matmul_1,
            flatten_0,
            index_select_0,
            transpose_3,
            assign_2,
            unsqueeze_0,
            softmax_0,
            transpose_4,
            reshape_3,
            matmul_2,
            add_1,
            reshape_4,
            transpose_5,
            reshape_5,
            full_int_array_4,
            full_int_array_5,
            slice_1,
            reshape_6,
            add_2,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            matmul_3,
            add_3,
            gelu_0,
            matmul_4,
            add_4,
            add_5,
        )
