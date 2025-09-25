import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.slice: (xf32) <- (2xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_2

        # pd_op.slice: (xf32) <- (2xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del data_1

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )
        del full_int_array_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_0

        # pd_op.clip: (512xf32) <- (512xf32, 1xf32, xf32)
        clip_0 = paddle._C_ops.clip(slice_2, full_0, slice_1)

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del full_int_array_1

        # pd_op.clip: (512xf32) <- (512xf32, 1xf32, xf32)
        clip_1 = paddle._C_ops.clip(slice_3, full_0, slice_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_3

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del full_int_array_2

        # pd_op.clip: (512xf32) <- (512xf32, 1xf32, xf32)
        clip_2 = paddle._C_ops.clip(slice_4, full_0, slice_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_4, [1], [1]
        )
        del data_0

        # pd_op.clip: (512xf32) <- (512xf32, 1xf32, xf32)
        clip_3 = paddle._C_ops.clip(slice_5, full_0, slice_0)

        # builtin.combine: ([512xf32, 512xf32, 512xf32, 512xf32]) <- (512xf32, 512xf32, 512xf32, 512xf32)
        combine_0 = [clip_0, clip_1, clip_2, clip_3]

        # pd_op.stack: (512x4xf32) <- ([512xf32, 512xf32, 512xf32, 512xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)
        del (
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            clip_0,
            clip_1,
            clip_2,
            clip_3,
            combine_0,
            full_0,
            full_int_array_3,
            full_int_array_4,
            slice_0,
            slice_1,
            slice_2,
            slice_3,
            slice_4,
            slice_5,
        )

        return stack_0
