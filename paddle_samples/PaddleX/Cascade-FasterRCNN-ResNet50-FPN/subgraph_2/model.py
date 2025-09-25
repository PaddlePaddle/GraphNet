import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2000"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (2000xf32, 2000xi64) <- (-1xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_0, full_0, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_0, full_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (2000x4xf32) <- (-1x4xf32, 2000xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_1, topk_1, full_1)
        del data_1, full_1

        # pd_op.shape64: (2xi64) <- (2000x4xf32)
        shape64_0 = paddle._C_ops.shape64(gather_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_0, full_int_array_1, gather_0, shape64_0, topk_1

        return topk_0, slice_0
