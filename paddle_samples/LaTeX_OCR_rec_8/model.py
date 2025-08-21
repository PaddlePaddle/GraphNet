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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
    ):
        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [20, -1, -1]

        # pd_op.expand: (20x1x256xf32) <- (1x1x256xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_1, full_int_array_0)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([20x1x256xf32, 20x56x256xf32]) <- (20x1x256xf32, 20x56x256xf32)
        combine_0 = [expand_0, data_0]

        # pd_op.concat: (20x57x256xf32) <- ([20x1x256xf32, 20x56x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("4"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (4xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_2, full_3, full_4, dtype="int64")

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("28"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4xi64) <- (4xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(arange_0, full_5, float("0"), True)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 1]

        # pd_op.reshape: (4x1xi64) <- (4xi64, 2xi64)
        reshape_4 = paddle._C_ops.reshape(scale_0, full_int_array_1)

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.int64,
            [float("14")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.reshape: (1xi64) <- (xi64, 1xi64)
        reshape_5 = paddle._C_ops.reshape(assign_value__0, full_int_array_2)

        # pd_op.repeat_interleave_with_tensor_index: (4x-1xi64) <- (4x1xi64, 1xi64)
        repeat_interleave_with_tensor_index_0 = (
            paddle._C_ops.repeat_interleave_with_tensor_index(reshape_4, reshape_5, 1)
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.reshape: (-1xi64) <- (4x-1xi64, 1xi64)
        reshape_6 = paddle._C_ops.reshape(
            repeat_interleave_with_tensor_index_0, full_int_array_3
        )

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("56"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (56xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_2, full_7, full_4, dtype="int64")

        # pd_op.add: (56xi64) <- (-1xi64, 56xi64)
        add_25 = paddle._C_ops.add(reshape_6, arange_1)

        # pd_op.full: (1xi64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (56xi64) <- (56xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(add_25, full_9, float("1"), True)

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1xi64, 56xi64]) <- (1xi64, 56xi64)
        combine_1 = [full_8, scale_1]

        # pd_op.concat: (57xi64) <- ([1xi64, 56xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_10)

        # pd_op.cast: (57xi64) <- (57xi64)
        cast_0 = paddle._C_ops.cast(concat_1, paddle.int64)

        # pd_op.transpose: (505x1x256xf32) <- (1x505x256xf32)
        transpose_0 = paddle._C_ops.transpose(data_2, [1, 0, 2])

        # pd_op.unsqueeze: (57x1xi64) <- (57xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(cast_0, full_int_array_3)

        # pd_op.gather_nd: (57x1x256xf32) <- (505x1x256xf32, 57x1xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(transpose_0, unsqueeze_0)

        # pd_op.transpose: (1x57x256xf32) <- (57x1x256xf32)
        transpose_1 = paddle._C_ops.transpose(gather_nd_0, [1, 0, 2])

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 1x57x256xf32)
        add_0 = paddle._C_ops.add(concat_0, transpose_1)

        # pd_op.layer_norm: (20x57x256xf32, 20x57xf32, 20x57xf32) <- (20x57x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_41, parameter_40, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (20x57x768xf32) <- (20x57x256xf32, 256x768xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_39, False, False)

        # pd_op.add: (20x57x768xf32) <- (20x57x768xf32, 768xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_38)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [20, 57, 3, 8, 32]

        # pd_op.reshape: (20x57x3x8x32xf32) <- (20x57x768xf32, 5xi64)
        reshape_7 = paddle._C_ops.reshape(add_1, full_int_array_4)

        # pd_op.transpose: (3x20x8x57x32xf32) <- (20x57x3x8x32xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_7, [2, 0, 3, 1, 4])

        # pd_op.unbind: ([20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32]) <- (3x20x8x57x32xf32)
        unbind_0 = paddle._C_ops.unbind(transpose_10, 0)

        # builtin.split: (20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32) <- ([20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32])
        (
            split_0,
            split_8,
            split_1,
        ) = unbind_0

        # pd_op.transpose: (20x8x32x57xf32) <- (20x8x57x32xf32)
        transpose_2 = paddle._C_ops.transpose(split_8, [0, 1, 3, 2])

        # pd_op.matmul: (20x8x57x57xf32) <- (20x8x57x32xf32, 20x8x32x57xf32)
        matmul_16 = paddle._C_ops.matmul(split_0, transpose_2, False, False)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_1

        # pd_op.scale: (20x8x57x57xf32) <- (20x8x57x57xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_16, full_1, float("0"), True)

        # pd_op.softmax: (20x8x57x57xf32) <- (20x8x57x57xf32)
        softmax_0 = paddle._C_ops.softmax(scale_2, -1)

        # pd_op.matmul: (20x8x57x32xf32) <- (20x8x57x57xf32, 20x8x57x32xf32)
        matmul_17 = paddle._C_ops.matmul(softmax_0, split_1, False, False)

        # pd_op.transpose: (20x57x8x32xf32) <- (20x8x57x32xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_17, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [20, 57, 256]

        # pd_op.reshape: (20x57x256xf32) <- (20x57x8x32xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_3, full_int_array_5)

        # pd_op.matmul: (20x57x256xf32) <- (20x57x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(reshape_0, data_3, False, True)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_1, data_4)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 20x57x256xf32)
        add_3 = paddle._C_ops.add(add_0, add_2)

        # pd_op.layer_norm: (20x57x256xf32, 20x57xf32, 20x57xf32) <- (20x57x256xf32, 256xf32, 256xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_37, parameter_36, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (20x57x1024xf32) <- (20x57x256xf32, 256x1024xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_3, parameter_35, False, False)

        # pd_op.add: (20x57x1024xf32) <- (20x57x1024xf32, 1024xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_34)

        # pd_op.gelu: (20x57x1024xf32) <- (20x57x1024xf32)
        gelu_0 = paddle._C_ops.gelu(add_4, False)

        # pd_op.matmul: (20x57x256xf32) <- (20x57x1024xf32, 1024x256xf32)
        matmul_3 = paddle._C_ops.matmul(gelu_0, parameter_33, False, False)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 256xf32)
        add_5 = paddle._C_ops.add(matmul_3, parameter_32)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 20x57x256xf32)
        add_6 = paddle._C_ops.add(add_3, add_5)

        # pd_op.layer_norm: (20x57x256xf32, 20x57xf32, 20x57xf32) <- (20x57x256xf32, 256xf32, 256xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_31, parameter_30, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (20x57x768xf32) <- (20x57x256xf32, 256x768xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_29, False, False)

        # pd_op.add: (20x57x768xf32) <- (20x57x768xf32, 768xf32)
        add_7 = paddle._C_ops.add(matmul_4, parameter_28)

        # pd_op.reshape: (20x57x3x8x32xf32) <- (20x57x768xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_7, full_int_array_4)

        # pd_op.transpose: (3x20x8x57x32xf32) <- (20x57x3x8x32xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])

        # pd_op.unbind: ([20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32]) <- (3x20x8x57x32xf32)
        unbind_1 = paddle._C_ops.unbind(transpose_11, 0)

        # builtin.split: (20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32) <- ([20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32])
        (
            split_2,
            split_9,
            split_3,
        ) = unbind_1

        # pd_op.transpose: (20x8x32x57xf32) <- (20x8x57x32xf32)
        transpose_4 = paddle._C_ops.transpose(split_9, [0, 1, 3, 2])

        # pd_op.matmul: (20x8x57x57xf32) <- (20x8x57x32xf32, 20x8x32x57xf32)
        matmul_18 = paddle._C_ops.matmul(split_2, transpose_4, False, False)

        # pd_op.scale: (20x8x57x57xf32) <- (20x8x57x57xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_18, full_1, float("0"), True)

        # pd_op.softmax: (20x8x57x57xf32) <- (20x8x57x57xf32)
        softmax_1 = paddle._C_ops.softmax(scale_3, -1)

        # pd_op.matmul: (20x8x57x32xf32) <- (20x8x57x57xf32, 20x8x57x32xf32)
        matmul_19 = paddle._C_ops.matmul(softmax_1, split_3, False, False)

        # pd_op.transpose: (20x57x8x32xf32) <- (20x8x57x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_19, [0, 2, 1, 3])

        # pd_op.reshape: (20x57x256xf32) <- (20x57x8x32xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_5, full_int_array_5)

        # pd_op.matmul: (20x57x256xf32) <- (20x57x256xf32, 256x256xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_1, data_5, False, True)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 256xf32)
        add_8 = paddle._C_ops.add(matmul_5, data_6)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 20x57x256xf32)
        add_9 = paddle._C_ops.add(add_6, add_8)

        # pd_op.layer_norm: (20x57x256xf32, 20x57xf32, 20x57xf32) <- (20x57x256xf32, 256xf32, 256xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_27, parameter_26, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (20x57x1024xf32) <- (20x57x256xf32, 256x1024xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_9, parameter_25, False, False)

        # pd_op.add: (20x57x1024xf32) <- (20x57x1024xf32, 1024xf32)
        add_10 = paddle._C_ops.add(matmul_6, parameter_24)

        # pd_op.gelu: (20x57x1024xf32) <- (20x57x1024xf32)
        gelu_1 = paddle._C_ops.gelu(add_10, False)

        # pd_op.matmul: (20x57x256xf32) <- (20x57x1024xf32, 1024x256xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_1, parameter_23, False, False)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 256xf32)
        add_11 = paddle._C_ops.add(matmul_7, parameter_22)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 20x57x256xf32)
        add_12 = paddle._C_ops.add(add_9, add_11)

        # pd_op.layer_norm: (20x57x256xf32, 20x57xf32, 20x57xf32) <- (20x57x256xf32, 256xf32, 256xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_21, parameter_20, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (20x57x768xf32) <- (20x57x256xf32, 256x768xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_12, parameter_19, False, False)

        # pd_op.add: (20x57x768xf32) <- (20x57x768xf32, 768xf32)
        add_13 = paddle._C_ops.add(matmul_8, parameter_18)

        # pd_op.reshape: (20x57x3x8x32xf32) <- (20x57x768xf32, 5xi64)
        reshape_9 = paddle._C_ops.reshape(add_13, full_int_array_4)

        # pd_op.transpose: (3x20x8x57x32xf32) <- (20x57x3x8x32xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_9, [2, 0, 3, 1, 4])

        # pd_op.unbind: ([20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32]) <- (3x20x8x57x32xf32)
        unbind_2 = paddle._C_ops.unbind(transpose_12, 0)

        # builtin.split: (20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32) <- ([20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32])
        (
            split_4,
            split_10,
            split_5,
        ) = unbind_2

        # pd_op.transpose: (20x8x32x57xf32) <- (20x8x57x32xf32)
        transpose_6 = paddle._C_ops.transpose(split_10, [0, 1, 3, 2])

        # pd_op.matmul: (20x8x57x57xf32) <- (20x8x57x32xf32, 20x8x32x57xf32)
        matmul_20 = paddle._C_ops.matmul(split_4, transpose_6, False, False)

        # pd_op.scale: (20x8x57x57xf32) <- (20x8x57x57xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_20, full_1, float("0"), True)

        # pd_op.softmax: (20x8x57x57xf32) <- (20x8x57x57xf32)
        softmax_2 = paddle._C_ops.softmax(scale_4, -1)

        # pd_op.matmul: (20x8x57x32xf32) <- (20x8x57x57xf32, 20x8x57x32xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_2, split_5, False, False)

        # pd_op.transpose: (20x57x8x32xf32) <- (20x8x57x32xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])

        # pd_op.reshape: (20x57x256xf32) <- (20x57x8x32xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_7, full_int_array_5)

        # pd_op.matmul: (20x57x256xf32) <- (20x57x256xf32, 256x256xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_2, data_7, False, True)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 256xf32)
        add_14 = paddle._C_ops.add(matmul_9, data_8)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 20x57x256xf32)
        add_15 = paddle._C_ops.add(add_12, add_14)

        # pd_op.layer_norm: (20x57x256xf32, 20x57xf32, 20x57xf32) <- (20x57x256xf32, 256xf32, 256xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_17, parameter_16, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (20x57x1024xf32) <- (20x57x256xf32, 256x1024xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_15, parameter_15, False, False)

        # pd_op.add: (20x57x1024xf32) <- (20x57x1024xf32, 1024xf32)
        add_16 = paddle._C_ops.add(matmul_10, parameter_14)

        # pd_op.gelu: (20x57x1024xf32) <- (20x57x1024xf32)
        gelu_2 = paddle._C_ops.gelu(add_16, False)

        # pd_op.matmul: (20x57x256xf32) <- (20x57x1024xf32, 1024x256xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_2, parameter_13, False, False)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 256xf32)
        add_17 = paddle._C_ops.add(matmul_11, parameter_12)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 20x57x256xf32)
        add_18 = paddle._C_ops.add(add_15, add_17)

        # pd_op.layer_norm: (20x57x256xf32, 20x57xf32, 20x57xf32) <- (20x57x256xf32, 256xf32, 256xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_11, parameter_10, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (20x57x768xf32) <- (20x57x256xf32, 256x768xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_18, parameter_9, False, False)

        # pd_op.add: (20x57x768xf32) <- (20x57x768xf32, 768xf32)
        add_19 = paddle._C_ops.add(matmul_12, parameter_8)

        # pd_op.reshape: (20x57x3x8x32xf32) <- (20x57x768xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_19, full_int_array_4)

        # pd_op.transpose: (3x20x8x57x32xf32) <- (20x57x3x8x32xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])

        # pd_op.unbind: ([20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32]) <- (3x20x8x57x32xf32)
        unbind_3 = paddle._C_ops.unbind(transpose_13, 0)

        # builtin.split: (20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32) <- ([20x8x57x32xf32, 20x8x57x32xf32, 20x8x57x32xf32])
        (
            split_6,
            split_11,
            split_7,
        ) = unbind_3

        # pd_op.transpose: (20x8x32x57xf32) <- (20x8x57x32xf32)
        transpose_8 = paddle._C_ops.transpose(split_11, [0, 1, 3, 2])

        # pd_op.matmul: (20x8x57x57xf32) <- (20x8x57x32xf32, 20x8x32x57xf32)
        matmul_22 = paddle._C_ops.matmul(split_6, transpose_8, False, False)

        # pd_op.scale: (20x8x57x57xf32) <- (20x8x57x57xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_22, full_1, float("0"), True)

        # pd_op.softmax: (20x8x57x57xf32) <- (20x8x57x57xf32)
        softmax_3 = paddle._C_ops.softmax(scale_5, -1)

        # pd_op.matmul: (20x8x57x32xf32) <- (20x8x57x57xf32, 20x8x57x32xf32)
        matmul_23 = paddle._C_ops.matmul(softmax_3, split_7, False, False)

        # pd_op.transpose: (20x57x8x32xf32) <- (20x8x57x32xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_23, [0, 2, 1, 3])

        # pd_op.reshape: (20x57x256xf32) <- (20x57x8x32xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_9, full_int_array_5)

        # pd_op.matmul: (20x57x256xf32) <- (20x57x256xf32, 256x256xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_3, data_9, False, True)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 256xf32)
        add_20 = paddle._C_ops.add(matmul_13, data_10)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 20x57x256xf32)
        add_21 = paddle._C_ops.add(add_18, add_20)

        # pd_op.layer_norm: (20x57x256xf32, 20x57xf32, 20x57xf32) <- (20x57x256xf32, 256xf32, 256xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_21, parameter_7, parameter_6, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (20x57x1024xf32) <- (20x57x256xf32, 256x1024xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_21, parameter_5, False, False)

        # pd_op.add: (20x57x1024xf32) <- (20x57x1024xf32, 1024xf32)
        add_22 = paddle._C_ops.add(matmul_14, parameter_4)

        # pd_op.gelu: (20x57x1024xf32) <- (20x57x1024xf32)
        gelu_3 = paddle._C_ops.gelu(add_22, False)

        # pd_op.matmul: (20x57x256xf32) <- (20x57x1024xf32, 1024x256xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_3, parameter_3, False, False)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 256xf32)
        add_23 = paddle._C_ops.add(matmul_15, parameter_2)

        # pd_op.add: (20x57x256xf32) <- (20x57x256xf32, 20x57x256xf32)
        add_24 = paddle._C_ops.add(add_21, add_23)

        # pd_op.layer_norm: (20x57x256xf32, 20x57xf32, 20x57xf32) <- (20x57x256xf32, 256xf32, 256xf32)
        layer_norm_26, layer_norm_24, layer_norm_25 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_24, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        return (
            full_int_array_0,
            expand_0,
            full_0,
            concat_0,
            transpose_0,
            unsqueeze_0,
            transpose_1,
            add_0,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
            add_1,
            split_0,
            split_1,
            transpose_2,
            full_1,
            softmax_0,
            transpose_3,
            reshape_0,
            matmul_1,
            add_2,
            add_3,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            matmul_2,
            add_4,
            gelu_0,
            matmul_3,
            add_5,
            add_6,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            matmul_4,
            add_7,
            split_2,
            split_3,
            transpose_4,
            assign_0,
            softmax_1,
            transpose_5,
            reshape_1,
            matmul_5,
            add_8,
            add_9,
            layer_norm_9,
            layer_norm_10,
            layer_norm_11,
            matmul_6,
            add_10,
            gelu_1,
            matmul_7,
            add_11,
            add_12,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            matmul_8,
            add_13,
            split_4,
            split_5,
            transpose_6,
            assign_1,
            softmax_2,
            transpose_7,
            reshape_2,
            matmul_9,
            add_14,
            add_15,
            layer_norm_15,
            layer_norm_16,
            layer_norm_17,
            matmul_10,
            add_16,
            gelu_2,
            matmul_11,
            add_17,
            add_18,
            layer_norm_18,
            layer_norm_19,
            layer_norm_20,
            matmul_12,
            add_19,
            split_6,
            split_7,
            transpose_8,
            assign_2,
            softmax_3,
            transpose_9,
            reshape_3,
            matmul_13,
            add_20,
            add_21,
            layer_norm_21,
            layer_norm_22,
            layer_norm_23,
            matmul_14,
            add_22,
            gelu_3,
            matmul_15,
            add_23,
            add_24,
            layer_norm_24,
            layer_norm_25,
            layer_norm_26,
        )
