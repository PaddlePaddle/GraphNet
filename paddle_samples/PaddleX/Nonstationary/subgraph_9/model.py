import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.unsqueeze: (-1x1x1xf32) <- (-1x1xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_2, full_int_array_0)
        del data_2

        # pd_op.unsqueeze: (-1x1x1x1xf32) <- (-1x1x1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(unsqueeze_0, full_int_array_0)

        # builtin.combine: ([-1x144x8x64xf32, -1x144x8x64xf32]) <- (-1x144x8x64xf32, -1x144x8x64xf32)
        combine_0 = [data_0, data_1]
        del data_0, data_1

        # pd_op.einsum: (-1x8x144x144xf32, [0xf32, 0xf32], [-1x144x8x64xf32, -1x144x8x64xf32]) <- ([-1x144x8x64xf32, -1x144x8x64xf32])
        einsum_0, einsum_1, einsum_2 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_0, "blhe,bshe->bhls"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_0

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_0,
            split_1,
        ) = einsum_1
        del einsum_1

        # builtin.split: (-1x144x8x64xf32, -1x144x8x64xf32) <- ([-1x144x8x64xf32, -1x144x8x64xf32])
        (
            split_2,
            split_3,
        ) = einsum_2
        del einsum_2

        # pd_op.multiply: (-1x8x144x144xf32) <- (-1x8x144x144xf32, -1x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(einsum_0, unsqueeze_1)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x8x144x144xf32) <- (-1x8x144x144xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(multiply_0, full_0, float("0"), True)
        del (
            assign_0,
            einsum_0,
            full_0,
            full_int_array_0,
            multiply_0,
            unsqueeze_0,
            unsqueeze_1,
        )

        return scale_0
