import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (525xf32) <- (525x1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (xi64) <- (1xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.topk: (-1xf32, -1xi64) <- (525xf32, xi64)
        topk_1, topk_0 = (lambda x, f: f(x))(
            paddle._C_ops.topk(slice_0, slice_1, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        return topk_0, topk_1
