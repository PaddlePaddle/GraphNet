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
        data_0,
        data_1,
    ):
        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_6)
        del parameter_6

        # pd_op.relu: (1x-1x256xf32) <- (1x-1x256xf32)
        relu_0 = paddle._C_ops.relu(add_0)
        del add_0

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_4)
        del parameter_4

        # pd_op.relu: (1x-1x256xf32) <- (1x-1x256xf32)
        relu_1 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.matmul: (1x-1x4xf32) <- (1x-1x256xf32, 256x4xf32)
        matmul_2 = paddle._C_ops.matmul(relu_1, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (1x-1x4xf32) <- (1x-1x4xf32, 4xf32)
        add_2 = paddle._C_ops.add(matmul_2, parameter_2)
        del parameter_2

        # pd_op.share_data_: (1x-1x4xf32) <- (1x-1x4xf32)
        share_data__0 = data_1.detach()
        del data_1

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x-1x4xf32) <- (1x-1x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(share_data__0, full_0, full_1)
        del full_0, full_1, share_data__0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1e-05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x-1x4xf32) <- (1x-1x4xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(clip_0, full_2, full_3)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x4xf32) <- (1x-1x4xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(clip_0, full_4, float("1"), True)
        del clip_0, full_4

        # pd_op.clip: (1x-1x4xf32) <- (1x-1x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(scale_0, full_2, full_3)
        del full_2, full_3, scale_0

        # pd_op.divide: (1x-1x4xf32) <- (1x-1x4xf32, 1x-1x4xf32)
        divide_0 = paddle._C_ops.divide(clip_1, clip_2)
        del clip_1, clip_2

        # pd_op.log: (1x-1x4xf32) <- (1x-1x4xf32)
        log_0 = paddle._C_ops.log(divide_0)
        del divide_0

        # pd_op.add: (1x-1x4xf32) <- (1x-1x4xf32, 1x-1x4xf32)
        add_3 = paddle._C_ops.add(add_2, log_0)

        # pd_op.sigmoid: (1x-1x4xf32) <- (1x-1x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_3)
        del add_3

        # pd_op.layer_norm: (1x-1x256xf32, 1x-1xf32, 1x-1xf32) <- (1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                data_0, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del (
            add_2,
            data_0,
            log_0,
            matmul_0,
            matmul_1,
            matmul_2,
            parameter_0,
            parameter_1,
            relu_0,
            relu_1,
        )

        return sigmoid_0, layer_norm_0
