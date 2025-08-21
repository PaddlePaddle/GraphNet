import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, parameter_0, parameter_1, parameter_2, parameter_3, data_0, data_1
    ):
        # pd_op.shape64: (3xi64) <- (-1x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(data_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(data_1, [0, 2, 1])

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("80"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, slice_3, data_0, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, stack_0)

        # pd_op.conv2d: (-1x384x-1x80xf32) <- (-1x-1x-1x80xf32, 384x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            reshape_0, parameter_3, [2, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, -1, 1, 1]

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_2, full_int_array_4)

        # pd_op.add: (-1x384x-1x80xf32) <- (-1x384x-1x80xf32, 1x384x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_1)

        # pd_op.shape64: (4xi64) <- (-1x384x-1x80xf32)
        shape64_1 = paddle._C_ops.shape64(add_0)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.flatten: (-1x384x-1xf32) <- (-1x384x-1x80xf32)
        flatten_0 = paddle._C_ops.flatten(add_0, 2, 3)

        # pd_op.transpose: (-1x-1x384xf32) <- (-1x384x-1xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_1, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        return layer_norm_0, slice_0
