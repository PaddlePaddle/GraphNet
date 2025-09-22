import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("12"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(data_5, full_0, float("0"), True)
        del data_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.unsqueeze: (1x512x4xf32) <- (512x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_2, full_int_array_0)
        del data_2

        # pd_op.unsqueeze: (1x512xi32) <- (512xi32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)
        del data_0

        # pd_op.unsqueeze: (1x512x4xf32) <- (512x4xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_1, full_int_array_0)
        del data_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (1xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del data_4, full_int_array_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_1, float("0"), True)
        del full_1, slice_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [scale_1]
        del scale_1

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.slice: (-1x256x7x7xf32) <- (512x256x7x7xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(data_3, [0], full_int_array_0, stack_0, [-1], [])
        del data_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_2

        # pd_op.mean: (-1x256x7xf32) <- (-1x256x7x7xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(slice_1, full_int_array_2, False)

        # pd_op.mean: (-1x256xf32) <- (-1x256x7xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(mean_0, full_int_array_2, False)

        # pd_op.unsqueeze: (1x-1x256xf32) <- (-1x256xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(mean_1, full_int_array_0)
        del full_int_array_0

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_2

        # builtin.combine: ([1x512x4xf32]) <- (1x512x4xf32)
        combine_1 = [unsqueeze_0]
        del unsqueeze_0

        # pd_op.concat: (1x512x4xf32) <- ([1x512x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_2)
        del combine_1

        # builtin.combine: ([1x512xi32]) <- (1x512xi32)
        combine_2 = [unsqueeze_1]
        del unsqueeze_1

        # pd_op.concat: (1x512xi32) <- ([1x512xi32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_2)
        del combine_2

        # builtin.combine: ([1x512x4xf32]) <- (1x512x4xf32)
        combine_3 = [unsqueeze_2]
        del unsqueeze_2

        # pd_op.concat: (1x512x4xf32) <- ([1x512x4xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_3, full_2)
        del combine_3

        # builtin.combine: ([1x-1x256xf32]) <- (1x-1x256xf32)
        combine_4 = [unsqueeze_3]

        # pd_op.concat: (1x-1x256xf32) <- ([1x-1x256xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_4, full_2)
        del (
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            combine_4,
            full_0,
            full_2,
            full_int_array_2,
            mean_0,
            mean_1,
            slice_1,
            stack_0,
            unsqueeze_3,
        )

        return scale_0, concat_0, concat_1, concat_2, concat_3
