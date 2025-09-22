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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("2160"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_1, full_0)
        del data_1, full_0

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.int64)
        del equal_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_1)
        del cast_0, full_1

        # pd_op.layer_norm: (2x-1x384xf32, 2x-1xf32, 2x-1xf32) <- (2x-1x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                data_2, parameter_5, parameter_4, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_2, parameter_4, parameter_5

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [-1, 54, 40, 384]

        # pd_op.reshape: (-1x54x40x384xf32) <- (2x-1x384xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(layer_norm_0, full_int_array_0)
        del full_int_array_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.pad: (-1x56x42x384xf32) <- (-1x54x40x384xf32, 1xf32)
        pad_0 = paddle._C_ops.pad(reshape_1, [0, 0, 0, 2, 0, 2, 0, 0], full_2)

        # pd_op.shape64: (4xi64) <- (-1x56x42x384xf32)
        shape64_0 = paddle._C_ops.shape64(pad_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_2

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [-3, -3]

        # pd_op.roll: (-1x56x42x384xf32) <- (-1x56x42x384xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(pad_0, full_int_array_3, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x56x42x384xf32)
        shape64_1 = paddle._C_ops.shape64(roll_0)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_1

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_4 = [-1, 8, 7, 6, 7, 384]

        # pd_op.reshape: (-1x8x7x6x7x384xf32) <- (-1x56x42x384xf32, 6xi64)
        reshape_2 = paddle._C_ops.reshape(roll_0, full_int_array_4)
        del full_int_array_4

        # pd_op.transpose: (-1x8x6x7x7x384xf32) <- (-1x8x7x6x7x384xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_2, [0, 1, 3, 2, 4, 5])
        del reshape_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [-1, 7, 7, 384]

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x8x6x7x7x384xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_0, full_int_array_5)
        del full_int_array_5

        # pd_op.shape64: (4xi64) <- (-1x7x7x384xf32)
        shape64_2 = paddle._C_ops.shape64(reshape_3)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_2

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("49"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [slice_2, full_3, full_4]
        del full_3, slice_2

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x49x384xf32) <- (-1x7x7x384xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(reshape_3, stack_0)
        del stack_0

        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        shape64_3 = paddle._C_ops.shape64(reshape_4)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_3

        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_4, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_2)
        del parameter_2

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_6 = [-1, 49, 3, 12, 32]

        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        reshape_5 = paddle._C_ops.reshape(add_0, full_int_array_6)
        del full_int_array_6

        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_5, [2, 0, 3, 1, 4])
        del reshape_5

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_7

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_2, full_int_array_7, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [3]

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_4, full_5, float("0"), True)
        del slice_4

        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        transpose_2 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])
        del slice_5

        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_2, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_0 = paddle._C_ops.flatten(data_4, 0, 1)
        del data_4

        # pd_op.index_select: (2401x-1xf32) <- (169x-1xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_0, flatten_0, 0)
        del data_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_9 = [49, 49, -1]

        # pd_op.reshape: (49x49x-1xf32) <- (2401x-1xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(index_select_0, full_int_array_9)
        del full_int_array_9

        # pd_op.transpose: (-1x49x49xf32) <- (49x49x-1xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_6, [2, 0, 1])
        del reshape_6

        # pd_op.unsqueeze: (1x-1x49x49xf32) <- (-1x49x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_3, full_int_array_1)

        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x-1x49x49xf32)
        add_1 = paddle._C_ops.add(matmul_1, unsqueeze_0)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_10 = [-1, 48, 12, 49, 49]

        # pd_op.reshape: (-1x48x12x49x49xf32) <- (-1x12x49x49xf32, 5xi64)
        reshape_7 = paddle._C_ops.reshape(add_1, full_int_array_10)
        del full_int_array_10

        # pd_op.unsqueeze: (48x1x49x49xf32) <- (48x49x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_3, full_int_array_2)
        del data_3, full_int_array_2

        # pd_op.unsqueeze: (1x48x1x49x49xf32) <- (48x1x49x49xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(unsqueeze_1, full_int_array_1)
        del full_int_array_1, unsqueeze_1

        # pd_op.add: (-1x48x12x49x49xf32) <- (-1x48x12x49x49xf32, 1x48x1x49x49xf32)
        add_2 = paddle._C_ops.add(reshape_7, unsqueeze_2)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [-1, 12, 49, 49]

        # pd_op.reshape: (-1x12x49x49xf32) <- (-1x48x12x49x49xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_2, full_int_array_11)
        del full_int_array_11

        # pd_op.softmax: (-1x12x49x49xf32) <- (-1x12x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_8, -1)
        del reshape_8

        # pd_op.matmul: (-1x12x49x32xf32) <- (-1x12x49x49xf32, -1x12x49x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_6, False, False)

        # pd_op.transpose: (-1x49x12x32xf32) <- (-1x12x49x32xf32)
        transpose_4 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_12 = [-1, 49, 384]

        # pd_op.reshape: (-1x49x384xf32) <- (-1x49x12x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_4, full_int_array_12)
        del full_int_array_12

        # pd_op.matmul: (-1x49x384xf32) <- (-1x49x384xf32, 384x384xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_9, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (-1x49x384xf32) <- (-1x49x384xf32, 384xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_0)
        del parameter_0

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("7"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [slice_3, full_6, full_6, full_4]
        del full_4, full_6, slice_3

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x49x384xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_3, stack_1)
        del (
            add_0,
            add_1,
            add_2,
            add_3,
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            flatten_0,
            full_2,
            full_5,
            full_int_array_3,
            full_int_array_7,
            full_int_array_8,
            index_select_0,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
            matmul_1,
            matmul_3,
            pad_0,
            reshape_1,
            reshape_3,
            reshape_4,
            reshape_7,
            reshape_9,
            roll_0,
            scale_0,
            slice_6,
            softmax_0,
            stack_1,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            transpose_4,
            unsqueeze_0,
            unsqueeze_2,
        )

        return reshape_0
