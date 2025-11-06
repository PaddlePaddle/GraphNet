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
        data_3,
        data_4,
    ):
        # pd_op.layer_norm: (2x7680x192xf32, 2x7680xf32, 2x7680xf32) <- (2x7680x192xf32, 192xf32, 192xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                data_2, parameter_11, parameter_10, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_10, parameter_11

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [-1, 80, 96, 192]

        # pd_op.reshape: (2x80x96x192xf32) <- (2x7680x192xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(layer_norm_0, full_int_array_0)
        del full_int_array_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.pad: (2x84x98x192xf32) <- (2x80x96x192xf32, 1xf32)
        pad_0 = paddle._C_ops.pad(reshape_0, [0, 0, 0, 4, 0, 2, 0, 0], full_0)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(data_1, full_1)

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.int64)
        del greater_than_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_1)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_1, full_1)
        del cast_1, full_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_1, full_2, float("0"), True)
        del full_2

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [scale_0, scale_0]
        del scale_0

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.roll: (2x84x98x192xf32) <- (2x84x98x192xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(pad_0, stack_0, [1, 2])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_1 = [-1, 12, 7, 14, 7, 192]

        # pd_op.reshape: (2x12x7x14x7x192xf32) <- (2x84x98x192xf32, 6xi64)
        reshape_1 = paddle._C_ops.reshape(roll_0, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (2x12x14x7x7x192xf32) <- (2x12x7x14x7x192xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_1, [0, 1, 3, 2, 4, 5])
        del reshape_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [-1, 7, 7, 192]

        # pd_op.reshape: (336x7x7x192xf32) <- (2x12x14x7x7x192xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_0, full_int_array_2)
        del full_int_array_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [336, 49, 192]

        # pd_op.reshape: (336x49x192xf32) <- (336x7x7x192xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(reshape_2, full_int_array_3)
        del full_int_array_3

        # pd_op.matmul: (336x49x576xf32) <- (336x49x192xf32, 192x576xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_3, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (336x49x576xf32) <- (336x49x576xf32, 576xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_8)
        del parameter_8

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [-1, 49, 3, 6, 32]

        # pd_op.reshape: (336x49x3x6x32xf32) <- (336x49x576xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_1, full_int_array_4)
        del full_int_array_4

        # pd_op.transpose: (3x336x6x49x32xf32) <- (336x49x3x6x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_6

        # pd_op.slice: (336x6x49x32xf32) <- (3x336x6x49x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_7

        # pd_op.slice: (336x6x49x32xf32) <- (3x336x6x49x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [3]

        # pd_op.slice: (336x6x49x32xf32) <- (3x336x6x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (336x6x49x32xf32) <- (336x6x49x32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_3, float("0"), True)
        del slice_0

        # pd_op.transpose: (336x6x32x49xf32) <- (336x6x49x32xf32)
        transpose_2 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (336x6x49x49xf32) <- (336x6x49x32xf32, 336x6x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_1, transpose_2, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_0 = paddle._C_ops.flatten(data_4, 0, 1)
        del data_4

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_0, flatten_0, 0)
        del data_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_9 = [49, 49, -1]

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(index_select_0, full_int_array_9)
        del full_int_array_9

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_5, [2, 0, 1])
        del reshape_5

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_3, full_int_array_5)

        # pd_op.add: (336x6x49x49xf32) <- (336x6x49x49xf32, 1x6x49x49xf32)
        add_2 = paddle._C_ops.add(matmul_1, unsqueeze_0)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_10 = [-1, 168, 6, 49, 49]

        # pd_op.reshape: (2x168x6x49x49xf32) <- (336x6x49x49xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_2, full_int_array_10)
        del full_int_array_10

        # pd_op.unsqueeze: (168x1x49x49xf32) <- (168x49x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_3, full_int_array_6)
        del data_3

        # pd_op.unsqueeze: (1x168x1x49x49xf32) <- (168x1x49x49xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(unsqueeze_1, full_int_array_5)
        del unsqueeze_1

        # pd_op.add: (2x168x6x49x49xf32) <- (2x168x6x49x49xf32, 1x168x1x49x49xf32)
        add_3 = paddle._C_ops.add(reshape_6, unsqueeze_2)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [-1, 6, 49, 49]

        # pd_op.reshape: (336x6x49x49xf32) <- (2x168x6x49x49xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(add_3, full_int_array_11)
        del full_int_array_11

        # pd_op.softmax: (336x6x49x49xf32) <- (336x6x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_7, -1)
        del reshape_7

        # pd_op.matmul: (336x6x49x32xf32) <- (336x6x49x49xf32, 336x6x49x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)

        # pd_op.transpose: (336x49x6x32xf32) <- (336x6x49x32xf32)
        transpose_4 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_12 = [-1, 49, 192]

        # pd_op.reshape: (336x49x192xf32) <- (336x49x6x32xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_4, full_int_array_12)
        del full_int_array_12

        # pd_op.matmul: (336x49x192xf32) <- (336x49x192xf32, 192x192xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_8, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (336x49x192xf32) <- (336x49x192xf32, 192xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_6)
        del parameter_6

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [336, 7, 7, 192]

        # pd_op.reshape: (336x7x7x192xf32) <- (336x49x192xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_4, full_int_array_13)
        del full_int_array_13

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_14 = [-1, 12, 14, 7, 7, 192]

        # pd_op.reshape: (2x12x14x7x7x192xf32) <- (336x7x7x192xf32, 6xi64)
        reshape_10 = paddle._C_ops.reshape(reshape_9, full_int_array_14)
        del full_int_array_14

        # pd_op.transpose: (2x12x7x14x7x192xf32) <- (2x12x14x7x7x192xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_10, [0, 1, 3, 2, 4, 5])
        del reshape_10

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [-1, 84, 98, 192]

        # pd_op.reshape: (2x84x98x192xf32) <- (2x12x7x14x7x192xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_5, full_int_array_15)
        del full_int_array_15

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_1 = [data_1, data_1]
        del data_1

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.roll: (2x84x98x192xf32) <- (2x84x98x192xf32, 2xi64)
        roll_1 = paddle._C_ops.roll(reshape_11, stack_1, [1, 2])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [80, 96]

        # pd_op.slice: (2x80x96x192xf32) <- (2x84x98x192xf32, 2xi64, 2xi64)
        slice_3 = paddle._C_ops.slice(
            roll_1, [1, 2], full_int_array_16, full_int_array_17, [1, 1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_18 = [-1, 7680, 192]

        # pd_op.reshape: (2x7680x192xf32) <- (2x80x96x192xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(slice_3, full_int_array_18)
        del full_int_array_18

        # pd_op.full: (xf64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_4,
            [],
            paddle.float64,
            [float("0.972727")],
            paddle.framework._current_expected_place(),
        )
        del full_4

        # pd_op.cast: (xf32) <- (xf64)
        cast_2 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_19 = [2, 1, 1]

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_19,
            paddle.float32,
            full_0,
            full_5,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_5 = paddle._C_ops.add(cast_2, uniform_0)
        del uniform_0

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_5)
        del add_5

        # pd_op.divide: (2x7680x192xf32) <- (2x7680x192xf32, xf32)
        divide_0 = paddle._C_ops.divide(reshape_12, cast_2)

        # pd_op.multiply: (2x7680x192xf32) <- (2x7680x192xf32, 2x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (2x7680x192xf32) <- (2x7680x192xf32, 2x7680x192xf32)
        add_6 = paddle._C_ops.add(data_2, multiply_0)
        del data_2

        # pd_op.layer_norm: (2x7680x192xf32, 2x7680xf32, 2x7680xf32) <- (2x7680x192xf32, 192xf32, 192xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_5, parameter_4, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.matmul: (2x7680x768xf32) <- (2x7680x192xf32, 192x768xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_3, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (2x7680x768xf32) <- (2x7680x768xf32, 768xf32)
        add_7 = paddle._C_ops.add(matmul_4, parameter_2)
        del parameter_2

        # pd_op.gelu: (2x7680x768xf32) <- (2x7680x768xf32)
        gelu_0 = paddle._C_ops.gelu(add_7, False)

        # pd_op.matmul: (2x7680x192xf32) <- (2x7680x768xf32, 768x192xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (2x7680x192xf32) <- (2x7680x192xf32, 192xf32)
        add_8 = paddle._C_ops.add(matmul_5, parameter_0)
        del parameter_0

        # pd_op.full: (xf64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.float64,
            [float("0.972727")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.cast: (xf32) <- (xf64)
        cast_3 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_19,
            paddle.float32,
            full_0,
            full_5,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_5, full_int_array_19

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_9 = paddle._C_ops.add(cast_3, uniform_1)
        del uniform_1

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_9)
        del add_9

        # pd_op.divide: (2x7680x192xf32) <- (2x7680x192xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_8, cast_3)

        # pd_op.multiply: (2x7680x192xf32) <- (2x7680x192xf32, 2x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (2x7680x192xf32) <- (2x7680x192xf32, 2x7680x192xf32)
        add_0 = paddle._C_ops.add(add_6, multiply_1)
        del (
            add_1,
            add_2,
            add_3,
            add_4,
            add_6,
            add_7,
            add_8,
            assign_0,
            assign_1,
            assign_2,
            cast_2,
            cast_3,
            divide_0,
            divide_1,
            flatten_0,
            floor_0,
            floor_1,
            full_0,
            full_3,
            full_int_array_16,
            full_int_array_17,
            full_int_array_5,
            full_int_array_6,
            full_int_array_7,
            full_int_array_8,
            gelu_0,
            index_select_0,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            matmul_0,
            matmul_1,
            matmul_3,
            matmul_4,
            matmul_5,
            multiply_0,
            multiply_1,
            pad_0,
            reshape_0,
            reshape_11,
            reshape_12,
            reshape_2,
            reshape_3,
            reshape_6,
            reshape_8,
            reshape_9,
            roll_0,
            roll_1,
            scale_1,
            slice_2,
            slice_3,
            softmax_0,
            stack_0,
            stack_1,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            unsqueeze_0,
            unsqueeze_2,
        )

        return add_0
