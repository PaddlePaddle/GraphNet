import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2147483647]

        # pd_op.slice: (16x1x1xf32) <- (16x96x1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_0, full_int_array_1

        # pd_op.assign: (16x1x1xf32) <- (16x1x1xf32)
        assign_0 = slice_0
        del slice_0

        # pd_op.share_data_: (16x1x1xf32) <- (16x1x1xf32)
        share_data__0 = assign_0.detach()
        del assign_0

        # pd_op.subtract: (16x96x1xf32) <- (16x96x1xf32, 16x1x1xf32)
        subtract_0 = paddle._C_ops.subtract(data_0, share_data__0)
        del data_0

        # pd_op.full: (16x96x1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [16, 96, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_3

        # pd_op.slice: (16x96xf32) <- (16x96x1xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            subtract_0, [2], full_int_array_2, full_int_array_3, [1], [2]
        )
        del subtract_0

        # pd_op.matmul: (16x96xf32) <- (16x96xf32, 96x96xf32)
        matmul_0 = paddle._C_ops.matmul(slice_1, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (16x96xf32) <- (16x96xf32, 96xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_0)
        del parameter_0

        # pd_op.set_value_with_tensor_: (16x96x1xf32) <- (16x96x1xf32, 16x96xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            full_0,
            add_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_3,
            [2],
            [2],
            [],
        )
        del full_0, full_int_array_2, full_int_array_3

        # pd_op.add: (16x96x1xf32) <- (16x96x1xf32, 16x1x1xf32)
        add_0 = paddle._C_ops.add(set_value_with_tensor__0, share_data__0)
        del (
            add_1,
            assign_1,
            assign_2,
            assign_3,
            matmul_0,
            set_value_with_tensor__0,
            share_data__0,
            slice_1,
        )

        return add_0
