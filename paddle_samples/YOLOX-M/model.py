import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, input0: paddle.Tensor, input1: paddle.Tensor, input2: paddle.Tensor
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        t0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        t1 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        t2 = [2]

        # pd_op.strided_slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        t3 = paddle._C_ops.strided_slice(input0, [1], t0, t1, t2)

        # pd_op.multiply: (-1x2xf32) <- (-1x2xf32, xf32)
        t4 = paddle._C_ops.multiply(t3, input1)
        del input1, t3

        # pd_op.set_value_with_tensor_: (-1x4xf32) <- (-1x4xf32, -1x2xf32, 1xi64, 1xi64, 1xi64)
        t5 = paddle._C_ops.set_value_with_tensor_(input0, t4, t0, t1, t2, [1], [], [])
        del input0, t0, t4

        # pd_op.full_int_array: (1xi64) <- ()
        t6 = [1]

        # pd_op.strided_slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        t7 = paddle._C_ops.strided_slice(t5, [1], t6, t1, t2)

        # pd_op.multiply: (-1x2xf32) <- (-1x2xf32, xf32)
        t8 = paddle._C_ops.multiply(t7, input2)
        del input2, t7

        return t1, t2, t5, t6, t8
