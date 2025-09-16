import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
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
    ):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1600xi32, 1296xi32, 576xi32, 256xi32, 144xi32]) <- (1600xi32, 1296xi32, 576xi32, 256xi32, 144xi32)
        combine_0 = [data_0, data_1, data_2, data_3, data_4]
        del data_0, data_1, data_2, data_3, data_4

        # pd_op.concat: (3872xi32) <- ([1600xi32, 1296xi32, 576xi32, 256xi32, 144xi32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # builtin.combine: ([3872xi32, 1xi32]) <- (3872xi32, 1xi32)
        combine_1 = [concat_0, full_1]
        del concat_0, full_1

        # pd_op.concat: (3873xi32) <- ([3872xi32, 1xi32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (-1xi64) <- (-1x2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_5, [1], full_int_array_0, full_int_array_1, [1], [1]
        )
        del data_5

        # pd_op.gather: (-1xi32) <- (3873xi32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(concat_1, slice_0, full_0)
        del concat_1, slice_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 3]

        # pd_op.unsqueeze: (-1x256x1x1xf32) <- (-1x256xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_6, full_int_array_2)
        del data_6, full_int_array_2

        # pd_op.conv2d: (1x-1x-1x-1xf32) <- (1x256x-1x-1xf32, -1x256x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_7, unsqueeze_0, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_7, unsqueeze_0

        # pd_op.squeeze: (-1x-1x-1xf32) <- (1x-1x-1x-1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(conv2d_0, full_int_array_0)
        del conv2d_0, full_int_array_0

        # pd_op.sigmoid: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(squeeze_0)
        del squeeze_0

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0.5"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (-1x-1x-1xb) <- (-1x-1x-1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(sigmoid_0, full_2)
        del full_2

        # pd_op.cast: (-1x-1x-1xf32) <- (-1x-1x-1xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.float32)
        del greater_than_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [1, 2]

        # pd_op.sum: (-1xf32) <- (-1x-1x-1xf32, 2xi64)
        sum_0 = paddle._C_ops.sum(cast_0, full_int_array_3, None, False)

        # pd_op.shape64: (1xi64) <- (-1xf32)
        shape64_0 = paddle._C_ops.shape64(sum_0)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1xf32) <- (1xf32, 1xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_3, shape64_0, paddle.float32
        )
        del full_3

        # pd_op.cast: (-1xf32) <- (-1xi32)
        cast_1 = paddle._C_ops.cast(gather_1, paddle.float32)
        del gather_1

        # pd_op.greater_than: (-1xb) <- (-1xf32, -1xf32)
        greater_than_1 = paddle._C_ops.greater_than(sum_0, cast_1)
        del cast_1

        # pd_op.where: (-1xf32) <- (-1xb, -1xf32, -1xf32)
        where_0 = paddle._C_ops.where(greater_than_1, sum_0, full_with_tensor_0)
        del full_with_tensor_0, greater_than_1

        # pd_op.nonzero: (-1x1xi64) <- (-1xf32)
        nonzero_0 = paddle._C_ops.nonzero(where_0)
        del where_0

        # pd_op.squeeze: (-1xi64) <- (-1x1xi64, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(nonzero_0, full_int_array_1)
        del full_int_array_1, nonzero_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xi64) <- (1xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(shape64_0, full_4, float("-1"), True)
        del full_4

        # pd_op.cast: (1xi64) <- (1xi64)
        cast_2 = paddle._C_ops.cast(scale_0, paddle.int64)
        del scale_0

        # builtin.combine: ([-1xi64, 1xi64]) <- (-1xi64, 1xi64)
        combine_2 = [squeeze_1, cast_2]
        del cast_2

        # pd_op.concat: (-1xi64) <- ([-1xi64, 1xi64], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        del combine_2

        # pd_op.cast: (1xi64) <- (1xi64)
        cast_3 = paddle._C_ops.cast(shape64_0, paddle.int64)
        del shape64_0

        # builtin.combine: ([-1xi64, 1xi64]) <- (-1xi64, 1xi64)
        combine_3 = [squeeze_1, cast_3]
        del cast_3, squeeze_1

        # pd_op.concat: (-1xi64) <- ([-1xi64, 1xi64], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # builtin.combine: ([-1xf32, 1xf32]) <- (-1xf32, 1xf32)
        combine_4 = [data_8, full_5]
        del data_8, full_5

        # pd_op.concat: (-1xf32) <- ([-1xf32, 1xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4

        # pd_op.gather: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(cast_0, concat_2, full_0)
        del cast_0

        # pd_op.gather: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(sigmoid_0, concat_2, full_0)
        del sigmoid_0

        # pd_op.gather: (-1xf32) <- (-1xf32, -1xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(sum_0, concat_2, full_0)
        del sum_0

        # pd_op.gather: (-1xi64) <- (-1xi64, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_9, concat_2, full_0)
        del concat_2, data_9

        # pd_op.gather: (-1xf32) <- (-1xf32, -1xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(concat_4, concat_3, full_0)
        del concat_3, concat_4, full_0

        # pd_op.multiply: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x-1xf32)
        multiply_1 = paddle._C_ops.multiply(gather_3, gather_2)

        # pd_op.cast: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        cast_4 = paddle._C_ops.cast(multiply_1, paddle.float32)
        del multiply_1

        # pd_op.sum: (-1xf32) <- (-1x-1x-1xf32, 2xi64)
        sum_1 = paddle._C_ops.sum(cast_4, full_int_array_3, None, False)
        del cast_4, full_int_array_3

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_0 = paddle._C_ops.divide(sum_1, gather_4)
        del sum_1

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_0 = paddle._C_ops.multiply(gather_5, divide_0)
        del divide_0, gather_2, gather_3, gather_4, gather_5

        return gather_0, multiply_0
