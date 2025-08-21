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
        # pd_op.matmul: (1x500x256xf32) <- (1x500x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_7, False, False)

        # pd_op.add: (1x500x256xf32) <- (1x500x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_6)

        # pd_op.relu: (1x500x256xf32) <- (1x500x256xf32)
        relu_0 = paddle._C_ops.relu(add_2)

        # pd_op.matmul: (1x500x256xf32) <- (1x500x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_5, False, False)

        # pd_op.add: (1x500x256xf32) <- (1x500x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_4)

        # pd_op.relu: (1x500x256xf32) <- (1x500x256xf32)
        relu_1 = paddle._C_ops.relu(add_3)

        # pd_op.matmul: (1x500x4xf32) <- (1x500x256xf32, 256x4xf32)
        matmul_2 = paddle._C_ops.matmul(relu_1, parameter_3, False, False)

        # pd_op.add: (1x500x4xf32) <- (1x500x4xf32, 4xf32)
        add_0 = paddle._C_ops.add(matmul_2, parameter_2)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x500x4xf32) <- (1x500x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(data_1, full_0, full_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1e-05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x500x4xf32) <- (1x500x4xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(clip_0, full_2, full_3)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x500x4xf32) <- (1x500x4xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(clip_0, full_4, float("1"), True)

        # pd_op.clip: (1x500x4xf32) <- (1x500x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(scale_0, full_2, full_3)

        # pd_op.divide: (1x500x4xf32) <- (1x500x4xf32, 1x500x4xf32)
        divide_0 = paddle._C_ops.divide(clip_1, clip_2)

        # pd_op.log: (1x500x4xf32) <- (1x500x4xf32)
        log_0 = paddle._C_ops.log(divide_0)

        # pd_op.add: (1x500x4xf32) <- (1x500x4xf32, 1x500x4xf32)
        add_4 = paddle._C_ops.add(add_0, log_0)

        # pd_op.sigmoid: (1x500x4xf32) <- (1x500x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_4)

        # pd_op.matmul: (1x500x4xf32) <- (1x500x256xf32, 256x4xf32)
        matmul_3 = paddle._C_ops.matmul(data_0, parameter_1, False, False)

        # pd_op.add: (1x500x4xf32) <- (1x500x4xf32, 4xf32)
        add_1 = paddle._C_ops.add(matmul_3, parameter_0)

        # pd_op.share_data_: (1x500x4xf32) <- (1x500x4xf32)
        share_data__0 = sigmoid_0.detach()
        return (
            matmul_0,
            relu_0,
            matmul_1,
            relu_1,
            matmul_2,
            add_0,
            log_0,
            matmul_3,
            sigmoid_0,
            share_data__0,
            add_1,
        )
