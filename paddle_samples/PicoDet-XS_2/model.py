import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, parameter_0, parameter_1, parameter_2, parameter_3, data_0, data_1
    ):
        # pd_op.conv2d: (4x44x1x1xf32) <- (4x-1x1x1xf32, 44x176x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x44x1x1xf32) <- (44xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_2, full_int_array_0)

        # pd_op.add: (4x44x1x1xf32) <- (4x44x1x1xf32, 1x44x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.relu: (4x44x1x1xf32) <- (4x44x1x1xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (4x176x1x1xf32) <- (4x44x1x1xf32, 176x44x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x176x1x1xf32) <- (176xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_0, full_int_array_0)

        # pd_op.add: (4x176x1x1xf32) <- (4x176x1x1xf32, 1x176x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_1)

        # pd_op.hardsigmoid: (4x176x1x1xf32) <- (4x176x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(
            add_1, float("0.166667"), float("0.5")
        )

        # pd_op.multiply: (4x176x-1x-1xf32) <- (4x-1x-1x-1xf32, 4x176x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(data_1, hardsigmoid_0)
        return (
            conv2d_0,
            reshape_0,
            relu_0,
            conv2d_1,
            reshape_1,
            hardsigmoid_0,
            multiply_0,
        )
