import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("5"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (34240x5xf32) <- (34240xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            data_1 % paddle.cast(full_0, data_1.dtype), full_0
        )
        del data_1, full_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.slice: (34240x4xf32) <- (34240x5xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            one_hot_0, [1], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_0, full_int_array_1, one_hot_0

        # pd_op.share_data_: (34240x4xf32) <- (34240x4xf32)
        share_data__0 = slice_0.detach()
        del slice_0

        # pd_op.shape64: (2xi64) <- (34240x4xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (34240x4xf32) <- (1xf32, 2xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_1, shape64_0, paddle.float32
        )
        del full_1, shape64_0

        # pd_op.sigmoid_cross_entropy_with_logits: (34240x4xf32) <- (34240x4xf32, 34240x4xf32, None)
        sigmoid_cross_entropy_with_logits_0 = (
            paddle._C_ops.sigmoid_cross_entropy_with_logits(
                data_0, share_data__0, None, False, -100
            )
        )

        # pd_op.sigmoid: (34240x4xf32) <- (34240x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(data_0)
        del data_0

        # pd_op.multiply: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        multiply_0 = paddle._C_ops.multiply(sigmoid_0, share_data__0)

        # pd_op.subtract: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        subtract_0 = paddle._C_ops.subtract(full_with_tensor_0, sigmoid_0)

        # pd_op.subtract: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        subtract_1 = paddle._C_ops.subtract(full_with_tensor_0, share_data__0)

        # pd_op.multiply: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_0, subtract_1)

        # pd_op.add: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_2,
            [],
            paddle.float32,
            [float("0.25")],
            paddle.framework._current_expected_place(),
        )
        del full_2

        # pd_op.multiply: (34240x4xf32) <- (xf32, 34240x4xf32)
        multiply_2 = paddle._C_ops.multiply(assign_value__0, share_data__0)

        # pd_op.subtract: (34240x4xf32) <- (34240x4xf32, xf32)
        subtract_2 = paddle._C_ops.subtract(full_with_tensor_0, assign_value__0)
        del assign_value__0

        # pd_op.subtract: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        subtract_3 = paddle._C_ops.subtract(full_with_tensor_0, share_data__0)

        # pd_op.multiply: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        multiply_3 = paddle._C_ops.multiply(subtract_2, subtract_3)
        del subtract_2, subtract_3

        # pd_op.add: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        add_1 = paddle._C_ops.add(multiply_2, multiply_3)
        del multiply_2, multiply_3

        # pd_op.multiply: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        multiply_4 = paddle._C_ops.multiply(add_1, sigmoid_cross_entropy_with_logits_0)

        # pd_op.subtract: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        subtract_4 = paddle._C_ops.subtract(full_with_tensor_0, add_0)

        # pd_op.pow: (34240x4xf32) <- (34240x4xf32)
        pow_0 = paddle._C_ops.pow(subtract_4, float("2"))

        # pd_op.multiply: (34240x4xf32) <- (34240x4xf32, 34240x4xf32)
        multiply_5 = paddle._C_ops.multiply(pow_0, multiply_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 1]

        # pd_op.reshape: (34240x1xf32) <- (34240xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(data_2, full_int_array_2)
        del data_2, full_int_array_2

        # pd_op.multiply: (34240x4xf32) <- (34240x4xf32, 34240x1xf32)
        multiply_6 = paddle._C_ops.multiply(multiply_5, reshape_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_3 = []

        # pd_op.sum: (xf32) <- (34240x4xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(multiply_6, full_int_array_3, None, False)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.0555556"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(sum_0, full_3, float("0"), True)
        del sum_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("12"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(scale_1, full_4, float("0"), True)
        del (
            add_0,
            add_1,
            full_3,
            full_4,
            full_int_array_3,
            full_with_tensor_0,
            multiply_0,
            multiply_1,
            multiply_4,
            multiply_5,
            multiply_6,
            pow_0,
            reshape_0,
            scale_1,
            share_data__0,
            sigmoid_0,
            sigmoid_cross_entropy_with_logits_0,
            subtract_0,
            subtract_1,
            subtract_4,
        )

        return scale_0
