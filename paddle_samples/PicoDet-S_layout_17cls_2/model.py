import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_8

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_8

        # pd_op.gather: (41x4xf32) <- (8330x4xf32, 41xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(data_2, data_3, full_8)

        # pd_op.gather: (41x4xf32) <- (8330x4xf32, 41xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(data_4, data_3, full_8)

        # pd_op.gather: (41x32xf32) <- (8330x32xf32, 41xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_5, data_3, full_8)

        # pd_op.gather: (41x1xf32) <- (8330x1xf32, 41xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(data_6, data_3, full_8)

        # pd_op.gather: (41x2xf32) <- (8330x2xf32, 41xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(data_7, data_3, full_8)

        # pd_op.divide: (41x2xf32) <- (41x2xf32, 41x1xf32)
        divide_5 = paddle._C_ops.divide(gather_5, gather_4)

        # pd_op.share_data_: (8330x11xf32) <- (8330x11xf32)
        share_data__0 = data_0.detach()

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_1

        # pd_op.max: (8330x1xf32) <- (8330x11xf32, 1xi64)
        max_0 = paddle._C_ops.max(share_data__0, full_int_array_1, True)

        # pd_op.gather: (41x1xf32) <- (8330x1xf32, 41xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(max_0, data_3, full_8)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 8]

        # pd_op.reshape: (164x8xf32) <- (41x32xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(gather_0, full_int_array_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.slice: (41xf32) <- (41x2xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            divide_5, [1], full_int_array_3, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (41xf32) <- (41x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            gather_2, [1], full_int_array_3, full_int_array_1, [1], [1]
        )

        # pd_op.subtract: (41xf32) <- (41xf32, 41xf32)
        subtract_10 = paddle._C_ops.subtract(slice_0, slice_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.slice: (41xf32) <- (41x2xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            divide_5, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (41xf32) <- (41x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            gather_2, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.subtract: (41xf32) <- (41xf32, 41xf32)
        subtract_11 = paddle._C_ops.subtract(slice_2, slice_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (41xf32) <- (41x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            gather_2, [1], full_int_array_4, full_int_array_5, [1], [1]
        )

        # pd_op.subtract: (41xf32) <- (41xf32, 41xf32)
        subtract_12 = paddle._C_ops.subtract(slice_4, slice_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (41xf32) <- (41x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            gather_2, [1], full_int_array_5, full_int_array_6, [1], [1]
        )

        # pd_op.subtract: (41xf32) <- (41xf32, 41xf32)
        subtract_13 = paddle._C_ops.subtract(slice_5, slice_2)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_9

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("6.9"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (41xf32) <- (41xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_10, full_9, full_10)

        # pd_op.clip: (41xf32) <- (41xf32, 1xf32, 1xf32)
        clip_3 = paddle._C_ops.clip(subtract_11, full_9, full_10)

        # pd_op.clip: (41xf32) <- (41xf32, 1xf32, 1xf32)
        clip_4 = paddle._C_ops.clip(subtract_12, full_9, full_10)

        # pd_op.clip: (41xf32) <- (41xf32, 1xf32, 1xf32)
        clip_5 = paddle._C_ops.clip(subtract_13, full_9, full_10)

        # builtin.combine: ([41xf32, 41xf32, 41xf32, 41xf32]) <- (41xf32, 41xf32, 41xf32, 41xf32)
        combine_0 = [clip_2, clip_3, clip_4, clip_5]

        # pd_op.stack: (41x4xf32) <- ([41xf32, 41xf32, 41xf32, 41xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_7

        # pd_op.reshape: (164xf32) <- (41x4xf32, 1xi64)
        reshape_2 = paddle._C_ops.reshape(stack_0, full_int_array_7)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([41x1xf32, 41x1xf32, 41x1xf32, 41x1xf32]) <- (41x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(gather_3, 4, full_0)

        # builtin.split: (41x1xf32, 41x1xf32, 41x1xf32, 41x1xf32) <- ([41x1xf32, 41x1xf32, 41x1xf32, 41x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_with_num_0

        # pd_op.split_with_num: ([41x1xf32, 41x1xf32, 41x1xf32, 41x1xf32]) <- (41x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(gather_2, 4, full_0)

        # builtin.split: (41x1xf32, 41x1xf32, 41x1xf32, 41x1xf32) <- ([41x1xf32, 41x1xf32, 41x1xf32, 41x1xf32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = split_with_num_1

        # pd_op.maximum: (41x1xf32) <- (41x1xf32, 41x1xf32)
        maximum_0 = paddle._C_ops.maximum(split_0, split_4)

        # pd_op.maximum: (41x1xf32) <- (41x1xf32, 41x1xf32)
        maximum_1 = paddle._C_ops.maximum(split_1, split_5)

        # pd_op.minimum: (41x1xf32) <- (41x1xf32, 41x1xf32)
        minimum_0 = paddle._C_ops.minimum(split_2, split_6)

        # pd_op.minimum: (41x1xf32) <- (41x1xf32, 41x1xf32)
        minimum_1 = paddle._C_ops.minimum(split_3, split_7)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_1

        # pd_op.clip: (41x1xf32) <- (41x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_9, full_1)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_1 = paddle._C_ops.subtract(minimum_1, maximum_1)

        # pd_op.clip: (41x1xf32) <- (41x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_9, full_1)

        # pd_op.multiply: (41x1xf32) <- (41x1xf32, 41x1xf32)
        multiply_0 = paddle._C_ops.multiply(clip_0, clip_1)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_2 = paddle._C_ops.subtract(split_2, split_0)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_3 = paddle._C_ops.subtract(split_3, split_1)

        # pd_op.multiply: (41x1xf32) <- (41x1xf32, 41x1xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_2, subtract_3)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_14 = paddle._C_ops.subtract(split_6, split_4)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_15 = paddle._C_ops.subtract(split_7, split_5)

        # pd_op.multiply: (41x1xf32) <- (41x1xf32, 41x1xf32)
        multiply_2 = paddle._C_ops.multiply(subtract_14, subtract_15)

        # pd_op.add: (41x1xf32) <- (41x1xf32, 41x1xf32)
        add_0 = paddle._C_ops.add(multiply_1, multiply_2)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_16 = paddle._C_ops.subtract(add_0, multiply_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_2

        # pd_op.scale: (41x1xf32) <- (41x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_16, full_2, float("1e-10"), True)

        # pd_op.divide: (41x1xf32) <- (41x1xf32, 41x1xf32)
        divide_0 = paddle._C_ops.divide(multiply_0, scale_0)

        # pd_op.minimum: (41x1xf32) <- (41x1xf32, 41x1xf32)
        minimum_2 = paddle._C_ops.minimum(split_0, split_4)

        # pd_op.minimum: (41x1xf32) <- (41x1xf32, 41x1xf32)
        minimum_3 = paddle._C_ops.minimum(split_1, split_5)

        # pd_op.maximum: (41x1xf32) <- (41x1xf32, 41x1xf32)
        maximum_2 = paddle._C_ops.maximum(split_2, split_6)

        # pd_op.maximum: (41x1xf32) <- (41x1xf32, 41x1xf32)
        maximum_3 = paddle._C_ops.maximum(split_3, split_7)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_4 = paddle._C_ops.subtract(maximum_2, minimum_2)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_5 = paddle._C_ops.subtract(maximum_3, minimum_3)

        # pd_op.multiply: (41x1xf32) <- (41x1xf32, 41x1xf32)
        multiply_10 = paddle._C_ops.multiply(subtract_4, subtract_5)

        # pd_op.scale: (41x1xf32) <- (41x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_10, full_2, float("1e-10"), True)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_6 = paddle._C_ops.subtract(scale_1, scale_0)

        # pd_op.divide: (41x1xf32) <- (41x1xf32, 41x1xf32)
        divide_1 = paddle._C_ops.divide(subtract_6, scale_1)

        # pd_op.subtract: (41x1xf32) <- (41x1xf32, 41x1xf32)
        subtract_17 = paddle._C_ops.subtract(divide_0, divide_1)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (41x1xf32) <- (41x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(subtract_17, full_3, float("1"), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("2.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (41x1xf32) <- (41x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_7, full_4, float("0"), True)

        # pd_op.multiply: (41x1xf32) <- (41x1xf32, 41x1xf32)
        multiply_3 = paddle._C_ops.multiply(scale_2, gather_1)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_11 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_8 = full_int_array_0

        # pd_op.sum: (xf32) <- (41x1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(multiply_3, full_int_array_0, None, False)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [-1, 4]

        # pd_op.expand: (41x4xf32) <- (41x1xf32, 2xi64)
        expand_0 = paddle._C_ops.expand(gather_1, full_int_array_8)

        # pd_op.reshape: (164xf32) <- (41x4xf32, 1xi64)
        reshape_0 = paddle._C_ops.reshape(expand_0, full_int_array_7)

        # pd_op.cast: (164xi64) <- (164xf32)
        cast_2 = paddle._C_ops.cast(reshape_2, paddle.int64)

        # pd_op.scale: (164xi64) <- (164xi64, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_2, full_2, float("1"), True)

        # pd_op.cast: (164xf32) <- (164xi64)
        cast_3 = paddle._C_ops.cast(scale_8, paddle.float32)

        # pd_op.subtract: (164xf32) <- (164xf32, 164xf32)
        subtract_7 = paddle._C_ops.subtract(cast_3, reshape_2)

        # pd_op.cast: (164xf32) <- (164xi64)
        cast_4 = paddle._C_ops.cast(cast_2, paddle.float32)

        # pd_op.subtract: (164xf32) <- (164xf32, 164xf32)
        subtract_8 = paddle._C_ops.subtract(reshape_2, cast_4)

        # pd_op.unsqueeze: (164x1xi64) <- (164xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(cast_2, full_int_array_7)

        # pd_op.cross_entropy_with_softmax: (164x8xf32, 164x1xf32) <- (164x8xf32, 164x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_1 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                reshape_1, unsqueeze_0, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.squeeze: (164xf32) <- (164x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_1, full_int_array_7
        )

        # pd_op.multiply: (164xf32) <- (164xf32, 164xf32)
        multiply_4 = paddle._C_ops.multiply(squeeze_0, subtract_7)

        # pd_op.unsqueeze: (164x1xi64) <- (164xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(scale_8, full_int_array_7)

        # pd_op.cross_entropy_with_softmax: (164x8xf32, 164x1xf32) <- (164x8xf32, 164x1xi64)
        cross_entropy_with_softmax_2, cross_entropy_with_softmax_3 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                reshape_1, unsqueeze_1, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.squeeze: (164xf32) <- (164x1xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_3, full_int_array_7
        )

        # pd_op.multiply: (164xf32) <- (164xf32, 164xf32)
        multiply_5 = paddle._C_ops.multiply(squeeze_1, subtract_8)

        # pd_op.add: (164xf32) <- (164xf32, 164xf32)
        add_2 = paddle._C_ops.add(multiply_4, multiply_5)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (164xf32) <- (164xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_2, full_5, float("0"), True)

        # pd_op.multiply: (164xf32) <- (164xf32, 164xf32)
        multiply_6 = paddle._C_ops.multiply(scale_3, reshape_0)

        # pd_op.sum: (xf32) <- (164xf32, 0xi64)
        sum_3 = paddle._C_ops.sum(multiply_6, full_int_array_0, None, False)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(sum_3, full_6, float("0"), True)

        # pd_op.sum: (xf32) <- (8330x11xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(data_0, full_int_array_0, None, False)

        # pd_op.cast: (8330x11xf32) <- (8330x11xf32)
        cast_0 = paddle._C_ops.cast(data_0, paddle.float32)

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (8330x11xb) <- (8330x11xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(cast_0, full_11)

        # pd_op.cast: (8330x11xf32) <- (8330x11xb)
        cast_5 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.multiply: (8330x11xf32) <- (8330x11xf32, 8330x11xf32)
        multiply_7 = paddle._C_ops.multiply(cast_0, cast_5)

        # pd_op.subtract: (8330x11xf32) <- (8330x11xf32, 8330x11xf32)
        subtract_9 = paddle._C_ops.subtract(data_1, cast_0)

        # pd_op.abs: (8330x11xf32) <- (8330x11xf32)
        abs_0 = paddle._C_ops.abs(subtract_9)

        # pd_op.pow: (8330x11xf32) <- (8330x11xf32)
        pow_0 = paddle._C_ops.pow(abs_0, float("2"))

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.75"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8330x11xf32) <- (8330x11xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(pow_0, full_7, float("0"), True)

        # pd_op.less_equal: (8330x11xb) <- (8330x11xf32, xf32)
        less_equal_0 = paddle._C_ops.less_equal(cast_0, full_11)

        # pd_op.cast: (8330x11xf32) <- (8330x11xb)
        cast_1 = paddle._C_ops.cast(less_equal_0, paddle.float32)

        # pd_op.multiply: (8330x11xf32) <- (8330x11xf32, 8330x11xf32)
        multiply_8 = paddle._C_ops.multiply(scale_5, cast_1)

        # pd_op.add: (8330x11xf32) <- (8330x11xf32, 8330x11xf32)
        add_1 = paddle._C_ops.add(multiply_7, multiply_8)

        # pd_op.bce_loss: (8330x11xf32) <- (8330x11xf32, 8330x11xf32)
        bce_loss_0 = paddle._C_ops.bce_loss(data_1, cast_0)

        # pd_op.multiply: (8330x11xf32) <- (8330x11xf32, 8330x11xf32)
        multiply_9 = paddle._C_ops.multiply(bce_loss_0, add_1)

        # pd_op.sum: (8330xf32) <- (8330x11xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(multiply_9, full_int_array_1, None, False)

        # pd_op.scale: (8330xf32) <- (8330xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(sum_4, full_2, float("0"), True)

        # pd_op.sum: (xf32) <- (8330xf32, 0xi64)
        sum_2 = paddle._C_ops.sum(scale_6, full_int_array_0, None, False)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_2 = paddle._C_ops.divide(sum_2, sum_1)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_3 = paddle._C_ops.divide(sum_0, sum_1)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_4 = paddle._C_ops.divide(scale_4, sum_1)
        return (
            assign_0,
            assign_1,
            gather_0,
            gather_1,
            full_0,
            split_0,
            split_1,
            split_2,
            split_3,
            split_4,
            split_5,
            split_6,
            split_7,
            maximum_0,
            maximum_1,
            minimum_0,
            minimum_1,
            subtract_0,
            assign_2,
            full_1,
            clip_0,
            subtract_1,
            assign_3,
            assign_4,
            clip_1,
            multiply_0,
            subtract_2,
            subtract_3,
            multiply_1,
            multiply_2,
            add_0,
            full_2,
            scale_0,
            divide_0,
            minimum_2,
            minimum_3,
            maximum_2,
            maximum_3,
            subtract_4,
            subtract_5,
            assign_5,
            scale_1,
            subtract_6,
            divide_1,
            full_3,
            full_4,
            scale_2,
            multiply_3,
            full_int_array_0,
            sum_0,
            reshape_0,
            subtract_7,
            subtract_8,
            unsqueeze_0,
            cross_entropy_with_softmax_0,
            cross_entropy_with_softmax_1,
            assign_6,
            squeeze_0,
            multiply_4,
            unsqueeze_1,
            cross_entropy_with_softmax_2,
            cross_entropy_with_softmax_3,
            assign_7,
            squeeze_1,
            multiply_5,
            full_5,
            scale_3,
            multiply_6,
            assign_8,
            full_6,
            scale_4,
            sum_1,
            cast_0,
            multiply_7,
            subtract_9,
            abs_0,
            full_7,
            scale_5,
            cast_1,
            multiply_8,
            add_1,
            bce_loss_0,
            multiply_9,
            assign_9,
            assign_10,
            scale_6,
            assign_11,
            sum_2,
            divide_2,
            divide_3,
            divide_4,
        )
