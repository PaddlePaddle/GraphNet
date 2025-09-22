import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, float("2"), True)
        del data_0

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_0, float("1"), True)
        del scale_0

        # pd_op.cast: (xf32) <- (xi64)
        cast_0 = paddle._C_ops.cast(scale_1, paddle.float32)
        del scale_1

        # builtin.combine: ([xf32]) <- (xf32)
        combine_0 = [cast_0]
        del cast_0

        # pd_op.stack: (1xf32) <- ([xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = stack_0
        del stack_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1xf32) <- (1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(assign_0, full_0, full_1)
        del assign_0, full_0, full_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.slice: (4x100x4xf32) <- (6x4x100x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del data_1

        # pd_op.slice: (4x100x5xf32) <- (6x4x100x5xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_2, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del assign_1, assign_2, data_2, full_int_array_0, full_int_array_1

        return slice_0, slice_1, clip_0
