import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        parameter_4,
        parameter_5,
        parameter_6,
        parameter_7,
        parameter_8,
        parameter_9,
        data_0,
        data_1,
    ):
        # pd_op.layer_norm: (1x400x256xf32, 1x400xf32, 1x400xf32) <- (1x400x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                data_0, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (1x400x2xf32) <- (1x400x256xf32, 256x2xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_7, False, False)

        # pd_op.add: (1x400x2xf32) <- (1x400x2xf32, 2xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_6)

        # pd_op.matmul: (1x400x256xf32) <- (1x400x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_0, parameter_5, False, False)

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_4)

        # pd_op.relu: (1x400x256xf32) <- (1x400x256xf32)
        relu_0 = paddle._C_ops.relu(add_2)

        # pd_op.matmul: (1x400x256xf32) <- (1x400x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(relu_0, parameter_3, False, False)

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_2)

        # pd_op.relu: (1x400x256xf32) <- (1x400x256xf32)
        relu_1 = paddle._C_ops.relu(add_3)

        # pd_op.matmul: (1x400x128xf32) <- (1x400x256xf32, 256x128xf32)
        matmul_3 = paddle._C_ops.matmul(relu_1, parameter_1, False, False)

        # pd_op.add: (1x400x128xf32) <- (1x400x128xf32, 128xf32)
        add_0 = paddle._C_ops.add(matmul_3, parameter_0)

        # pd_op.flatten: (1x128x25600xf32) <- (1x128x160x160xf32)
        flatten_0 = paddle._C_ops.flatten(data_1, 2, 3)

        # pd_op.bmm: (1x400x25600xf32) <- (1x400x128xf32, 1x128x25600xf32)
        bmm_0 = paddle._C_ops.bmm(add_0, flatten_0)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 400, 160, 160]

        # pd_op.reshape: (1x400x160x160xf32) <- (1x400x25600xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(bmm_0, full_int_array_0)
        return (
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
            matmul_1,
            relu_0,
            matmul_2,
            relu_1,
            matmul_3,
            add_0,
            flatten_0,
            bmm_0,
            add_1,
            reshape_0,
        )
