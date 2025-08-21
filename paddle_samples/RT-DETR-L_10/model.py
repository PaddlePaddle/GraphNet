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
        data_2,
    ):
        # pd_op.matmul: (8x498x256xf32) <- (8x498x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_7, False, False)

        # pd_op.assign: (8x498x256xf32) <- (8x498x256xf32)
        assign_0 = matmul_0

        # pd_op.add: (8x498x256xf32) <- (8x498x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_6)

        # pd_op.relu: (8x498x256xf32) <- (8x498x256xf32)
        relu_0 = paddle._C_ops.relu(add_2)

        # pd_op.assign: (8x498x256xf32) <- (8x498x256xf32)
        assign_1 = relu_0

        # pd_op.matmul: (8x498x256xf32) <- (8x498x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_5, False, False)

        # pd_op.assign: (8x498x256xf32) <- (8x498x256xf32)
        assign_2 = matmul_1

        # pd_op.add: (8x498x256xf32) <- (8x498x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_4)

        # pd_op.relu: (8x498x256xf32) <- (8x498x256xf32)
        relu_1 = paddle._C_ops.relu(add_3)

        # pd_op.assign: (8x498x256xf32) <- (8x498x256xf32)
        assign_3 = relu_1

        # pd_op.matmul: (8x498x4xf32) <- (8x498x256xf32, 256x4xf32)
        matmul_2 = paddle._C_ops.matmul(relu_1, parameter_3, False, False)

        # pd_op.assign: (8x498x4xf32) <- (8x498x4xf32)
        assign_4 = matmul_2

        # pd_op.add: (8x498x4xf32) <- (8x498x4xf32, 4xf32)
        add_0 = paddle._C_ops.add(matmul_2, parameter_2)

        # pd_op.assign: (8x498x4xf32) <- (8x498x4xf32)
        assign_5 = add_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_1

        # pd_op.clip: (8x498x4xf32) <- (8x498x4xf32, 1xf32, 1xf32)
        clip_3 = paddle._C_ops.clip(data_1, full_0, full_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1e-05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_3

        # pd_op.clip: (8x498x4xf32) <- (8x498x4xf32, 1xf32, 1xf32)
        clip_4 = paddle._C_ops.clip(clip_3, full_2, full_3)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_4

        # pd_op.scale: (8x498x4xf32) <- (8x498x4xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(clip_3, full_4, float("1"), True)

        # pd_op.clip: (8x498x4xf32) <- (8x498x4xf32, 1xf32, 1xf32)
        clip_5 = paddle._C_ops.clip(scale_1, full_2, full_3)

        # pd_op.divide: (8x498x4xf32) <- (8x498x4xf32, 8x498x4xf32)
        divide_1 = paddle._C_ops.divide(clip_4, clip_5)

        # pd_op.log: (8x498x4xf32) <- (8x498x4xf32)
        log_0 = paddle._C_ops.log(divide_1)

        # pd_op.add: (8x498x4xf32) <- (8x498x4xf32, 8x498x4xf32)
        add_4 = paddle._C_ops.add(add_0, log_0)

        # pd_op.sigmoid: (8x498x4xf32) <- (8x498x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_4)

        # pd_op.matmul: (8x498x4xf32) <- (8x498x256xf32, 256x4xf32)
        matmul_3 = paddle._C_ops.matmul(data_0, parameter_1, False, False)

        # pd_op.add: (8x498x4xf32) <- (8x498x4xf32, 4xf32)
        add_1 = paddle._C_ops.add(matmul_3, parameter_0)

        # pd_op.clip: (8x-1x4xf32) <- (8x-1x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(data_2, full_0, full_1)

        # pd_op.clip: (8x-1x4xf32) <- (8x-1x4xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(clip_0, full_2, full_3)

        # pd_op.scale: (8x-1x4xf32) <- (8x-1x4xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(clip_0, full_4, float("1"), True)

        # pd_op.clip: (8x-1x4xf32) <- (8x-1x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(scale_0, full_2, full_3)

        # pd_op.divide: (8x-1x4xf32) <- (8x-1x4xf32, 8x-1x4xf32)
        divide_0 = paddle._C_ops.divide(clip_1, clip_2)

        # pd_op.log: (8x-1x4xf32) <- (8x-1x4xf32)
        log_1 = paddle._C_ops.log(divide_0)

        # pd_op.add: (8x498x4xf32) <- (8x498x4xf32, 8x-1x4xf32)
        add_5 = paddle._C_ops.add(add_0, log_1)

        # pd_op.sigmoid: (8x498x4xf32) <- (8x498x4xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_5)

        # pd_op.share_data_: (8x498x4xf32) <- (8x498x4xf32)
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
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            clip_0,
            assign_8,
            assign_9,
            clip_1,
            assign_10,
            scale_0,
            assign_11,
            assign_12,
            clip_2,
            divide_0,
            log_1,
            sigmoid_0,
            share_data__0,
            add_1,
            sigmoid_1,
        )
