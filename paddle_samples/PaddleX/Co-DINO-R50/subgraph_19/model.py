import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (900xi64) <- ()
        full_0 = paddle._C_ops.full(
            [900], float("-1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign: (900xi64) <- (900xi64)
        assign_0 = full_0

        # pd_op.full: (4xi64) <- ()
        full_1 = paddle._C_ops.full(
            [4], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xi64) <- (4xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_1,
            [4],
            paddle.int64,
            [float("640"), float("853"), float("640"), float("853")],
            paddle.framework._current_expected_place(),
        )
        del full_1

        # pd_op.cast: (4xf32) <- (4xi64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.unsqueeze: (1x4xf32) <- (4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(cast_0, full_int_array_0)
        del cast_0

        # pd_op.sigmoid: (900x4xf32) <- (900x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(data_1)
        del data_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_2

        # pd_op.scale: (900x4xf32) <- (900x4xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(sigmoid_0, full_2, float("1"), True)

        # pd_op.assign: (900x4xf32) <- (900x4xf32)
        assign_4 = scale_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_3

        # pd_op.scale: (900x4xf32) <- (900x4xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_3, float("1e-12"), True)

        # pd_op.log: (900x4xf32) <- (900x4xf32)
        log_0 = paddle._C_ops.log(scale_3)

        # pd_op.scale: (900x4xf32) <- (900x4xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(log_0, full_2, float("0"), True)
        del log_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.75"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (900x4xf32) <- (900x4xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_4, float("0"), True)
        del scale_4

        # pd_op.pow: (900x4xf32) <- (900x4xf32)
        pow_0 = paddle._C_ops.pow(sigmoid_0, float("2"))

        # pd_op.multiply: (900x4xf32) <- (900x4xf32, 900x4xf32)
        multiply_1 = paddle._C_ops.multiply(scale_5, pow_0)

        # pd_op.scale: (900x4xf32) <- (900x4xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(sigmoid_0, full_3, float("1e-12"), True)

        # pd_op.log: (900x4xf32) <- (900x4xf32)
        log_1 = paddle._C_ops.log(scale_6)

        # pd_op.scale: (900x4xf32) <- (900x4xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(log_1, full_2, float("0"), True)
        del log_1

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (900x4xf32) <- (900x4xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(scale_7, full_5, float("0"), True)
        del scale_7

        # pd_op.pow: (900x4xf32) <- (900x4xf32)
        pow_1 = paddle._C_ops.pow(scale_2, float("2"))
        del scale_2

        # pd_op.multiply: (900x4xf32) <- (900x4xf32, 900x4xf32)
        multiply_2 = paddle._C_ops.multiply(scale_8, pow_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.squeeze: (1xi32) <- (1x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_3, full_int_array_1)
        del data_3

        # pd_op.transpose: (4x900xf32) <- (900x4xf32)
        transpose_0 = paddle._C_ops.transpose(multiply_2, [1, 0])
        del multiply_2

        # pd_op.unsqueeze: (1x1xi32) <- (1xi32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_1)
        del full_int_array_1, squeeze_0

        # pd_op.assign: (1x1xi32) <- (1x1xi32)
        assign_6 = unsqueeze_1

        # pd_op.gather_nd: (1x900xf32) <- (4x900xf32, 1x1xi32)
        gather_nd_0 = paddle._C_ops.gather_nd(transpose_0, unsqueeze_1)

        # pd_op.transpose: (900x1xf32) <- (1x900xf32)
        transpose_1 = paddle._C_ops.transpose(gather_nd_0, [1, 0])
        del gather_nd_0

        # pd_op.transpose: (4x900xf32) <- (900x4xf32)
        transpose_2 = paddle._C_ops.transpose(multiply_1, [1, 0])
        del multiply_1

        # pd_op.gather_nd: (1x900xf32) <- (4x900xf32, 1x1xi32)
        gather_nd_1 = paddle._C_ops.gather_nd(transpose_2, unsqueeze_1)

        # pd_op.transpose: (900x1xf32) <- (1x900xf32)
        transpose_3 = paddle._C_ops.transpose(gather_nd_1, [1, 0])
        del gather_nd_1

        # pd_op.subtract: (900x1xf32) <- (900x1xf32, 900x1xf32)
        subtract_0 = paddle._C_ops.subtract(transpose_1, transpose_3)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (900x1xf32) <- (900x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_6, float("0"), True)
        del subtract_0

        # pd_op.divide: (1x4xf32) <- (1x4xf32, 1x4xf32)
        divide_0 = paddle._C_ops.divide(data_2, unsqueeze_0)
        del data_2

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_7

        # pd_op.split_with_num: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(divide_0, 4, full_7)
        del divide_0

        # builtin.split: (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.add: (1x1xf32) <- (1x1xf32, 1x1xf32)
        add_0 = paddle._C_ops.add(split_0, split_2)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_8

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(add_0, full_8, float("0"), True)
        del add_0

        # pd_op.add: (1x1xf32) <- (1x1xf32, 1x1xf32)
        add_1 = paddle._C_ops.add(split_1, split_3)

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(add_1, full_8, float("0"), True)
        del add_1

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_2, split_0)
        del split_0, split_2

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_2 = paddle._C_ops.subtract(split_3, split_1)
        del split_1, split_3

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_9

        # builtin.combine: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32)
        combine_0 = [scale_9, scale_10, subtract_1, subtract_2]
        del scale_10, scale_9, subtract_1, subtract_2

        # pd_op.concat: (1x4xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_9)
        del combine_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.unsqueeze: (900x1x4xf32) <- (900x4xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_0, full_int_array_2)

        # pd_op.unsqueeze: (1x1x4xf32) <- (1x4xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(concat_0, full_int_array_0)
        del concat_0, full_int_array_0

        # pd_op.subtract: (900x1x4xf32) <- (900x1x4xf32, 1x1x4xf32)
        subtract_3 = paddle._C_ops.subtract(unsqueeze_2, unsqueeze_3)

        # pd_op.p_norm: (900x1xf32) <- (900x1x4xf32)
        p_norm_0 = paddle._C_ops.p_norm(
            subtract_3, float("1"), -1, float("1e-12"), False, False
        )

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (900x1xf32) <- (900x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(p_norm_0, full_10, float("0"), True)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 1, 1, 1]

        # pd_op.split: ([900x1xf32, 900x1xf32, 900x1xf32, 900x1xf32]) <- (900x4xf32, 4xi64, 1xi32)
        split_4 = paddle._C_ops.split(data_0, full_int_array_3, full_7)
        del data_0, full_7, full_int_array_3

        # builtin.split: (900x1xf32, 900x1xf32, 900x1xf32, 900x1xf32) <- ([900x1xf32, 900x1xf32, 900x1xf32, 900x1xf32])
        (
            split_5,
            split_6,
            split_7,
            split_8,
        ) = split_4
        del split_4

        # pd_op.scale: (900x1xf32) <- (900x1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(split_7, full_8, float("0"), True)
        del split_7

        # pd_op.assign: (900x1xf32) <- (900x1xf32)
        assign_13 = scale_11

        # pd_op.subtract: (900x1xf32) <- (900x1xf32, 900x1xf32)
        subtract_4 = paddle._C_ops.subtract(split_5, scale_11)

        # pd_op.scale: (900x1xf32) <- (900x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(split_8, full_8, float("0"), True)
        del full_8, split_8

        # pd_op.assign: (900x1xf32) <- (900x1xf32)
        assign_14 = scale_12

        # pd_op.subtract: (900x1xf32) <- (900x1xf32, 900x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_6, scale_12)

        # pd_op.add: (900x1xf32) <- (900x1xf32, 900x1xf32)
        add_2 = paddle._C_ops.add(split_5, scale_11)

        # pd_op.add: (900x1xf32) <- (900x1xf32, 900x1xf32)
        add_3 = paddle._C_ops.add(split_6, scale_12)

        # builtin.combine: ([900x1xf32, 900x1xf32, 900x1xf32, 900x1xf32]) <- (900x1xf32, 900x1xf32, 900x1xf32, 900x1xf32)
        combine_1 = [subtract_4, subtract_5, add_2, add_3]

        # pd_op.concat: (900x4xf32) <- ([900x1xf32, 900x1xf32, 900x1xf32, 900x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_9)
        del combine_1, full_9

        # pd_op.multiply: (900x4xf32) <- (900x4xf32, 1x4xf32)
        multiply_0 = paddle._C_ops.multiply(concat_1, unsqueeze_0)
        del (
            add_2,
            add_3,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            concat_1,
            full_10,
            full_2,
            full_3,
            full_4,
            full_5,
            full_6,
            full_int_array_2,
            p_norm_0,
            pow_0,
            pow_1,
            scale_11,
            scale_12,
            scale_3,
            scale_5,
            scale_6,
            scale_8,
            sigmoid_0,
            split_5,
            split_6,
            subtract_3,
            subtract_4,
            subtract_5,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            unsqueeze_0,
            unsqueeze_1,
            unsqueeze_2,
            unsqueeze_3,
        )

        return multiply_0, scale_0, scale_1, full_0
