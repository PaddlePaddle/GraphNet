import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2147483647]

        # pd_op.slice: (-1x1x1xf32) <- (-1x96x1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_0, full_int_array_1

        # pd_op.assign: (-1x1x1xf32) <- (-1x1x1xf32)
        assign_0 = slice_0
        del slice_0

        # pd_op.share_data_: (-1x1x1xf32) <- (-1x1x1xf32)
        share_data__0 = assign_0.detach()
        del assign_0

        # pd_op.subtract: (-1x96x1xf32) <- (-1x96x1xf32, -1x1x1xf32)
        subtract_0 = paddle._C_ops.subtract(data_0, share_data__0)
        del data_0

        # pd_op.shape64: (3xi64) <- (-1x96x1xf32)
        shape64_0 = paddle._C_ops.shape64(subtract_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("96"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [slice_1, full_0, full_1]
        del full_0, full_1, slice_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1x96x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_2, stack_0, paddle.float32
        )
        del full_2, stack_0

        # pd_op.slice: (-1x96xf32) <- (-1x96x1xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            subtract_0, [2], full_int_array_2, full_int_array_3, [1], [2]
        )
        del subtract_0

        # pd_op.matmul: (-1x96xf32) <- (-1x96xf32, 96x96xf32)
        matmul_0 = paddle._C_ops.matmul(slice_2, parameter_1, False, False)
        del parameter_1, slice_2

        # pd_op.add: (-1x96xf32) <- (-1x96xf32, 96xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        # pd_op.set_value_with_tensor_: (-1x96x1xf32) <- (-1x96x1xf32, -1x96xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            full_with_tensor_0,
            add_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_3,
            [2],
            [2],
            [],
        )
        del add_1, full_int_array_2, full_int_array_3, full_with_tensor_0

        # pd_op.add: (-1x96x1xf32) <- (-1x96x1xf32, -1x1x1xf32)
        add_0 = paddle._C_ops.add(set_value_with_tensor__0, share_data__0)
        del set_value_with_tensor__0, share_data__0

        return add_0
