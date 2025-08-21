import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0):
        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_0, full_int_array_0)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.flatten: (-1x256x-1xf32) <- (-1x256x-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(add_0, 2, 3)

        # pd_op.transpose: (-1x-1x256xf32) <- (-1x256x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        return transpose_0
