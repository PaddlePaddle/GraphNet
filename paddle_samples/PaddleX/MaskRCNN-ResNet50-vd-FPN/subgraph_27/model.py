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
        data_10,
        data_11,
        data_12,
        data_13,
        data_14,
        data_15,
        data_16,
        data_17,
        data_18,
    ):
        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (16xi32) <- (512xi32, 16x1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_0, data_1, full_1)
        del data_0, data_1, full_1

        # pd_op.full: (16xf32) <- ()
        full_0 = paddle._C_ops.full(
            [16], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # builtin.combine: ([28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32]) <- (28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32)
        combine_0 = [
            data_3,
            data_4,
            data_5,
            data_6,
            data_7,
            data_8,
            data_9,
            data_10,
            data_11,
            data_12,
            data_13,
            data_14,
            data_15,
            data_16,
            data_17,
            data_18,
        ]
        del (
            data_10,
            data_11,
            data_12,
            data_13,
            data_14,
            data_15,
            data_16,
            data_17,
            data_18,
            data_3,
            data_4,
            data_5,
            data_6,
            data_7,
            data_8,
            data_9,
        )

        # pd_op.stack: (16x28x28xi32) <- ([28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32, 28x28xi32])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.shape64: (2xi64) <- (16x4xf32)
        shape64_0 = paddle._C_ops.shape64(data_2)
        del data_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_0, full_int_array_1, shape64_0

        return gather_0, full_0, stack_0, slice_0
