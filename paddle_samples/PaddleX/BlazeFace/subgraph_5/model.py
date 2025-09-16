import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.masked_select: (-1xf32) <- (4x22400x4xf32, 4x22400x4xb)
        masked_select_0 = paddle._C_ops.masked_select(data_0, data_1)
        del data_0

        # pd_op.masked_select: (-1xf32) <- (4x22400x4xf32, 4x22400x4xb)
        masked_select_1 = paddle._C_ops.masked_select(data_2, data_1)
        del data_1, data_2

        # pd_op.huber_loss: (-1xf32, -1xf32) <- (-1xf32, -1xf32)
        huber_loss_1, huber_loss_0 = (lambda x, f: f(x))(
            paddle._C_ops.huber_loss(masked_select_0, masked_select_1, float("1")),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del masked_select_0, masked_select_1

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(huber_loss_1, full_int_array_0, None, False)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(sum_0, full_0, float("0"), True)
        del sum_0

        # pd_op.cross_entropy_with_softmax: (4x22400x2xf32, 4x22400x1xf32) <- (4x22400x2xf32, 4x22400x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_1 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                data_3, data_4, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.squeeze: (4x22400xf32) <- (4x22400x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_1, full_int_array_1
        )

        # pd_op.squeeze: (4x22400xi64) <- (4x22400x1xi64, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(data_4, full_int_array_1)
        del (
            cross_entropy_with_softmax_1,
            data_4,
            full_0,
            full_int_array_0,
            full_int_array_1,
            huber_loss_1,
        )

        return huber_loss_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, scale_0
