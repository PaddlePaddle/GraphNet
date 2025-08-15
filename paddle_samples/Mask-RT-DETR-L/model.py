import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input0: paddle.Tensor,
        input1: paddle.Tensor,
        input2: paddle.Tensor,
        input3: paddle.Tensor,
        input4: paddle.Tensor,
        input5: paddle.Tensor,
        input6: paddle.Tensor,
        input7: paddle.Tensor,
        input8: paddle.Tensor,
        input9: paddle.Tensor,
        input10: paddle.Tensor,
        input11: paddle.Tensor,
        input12: paddle.Tensor,
        input13: paddle.Tensor,
        input14: paddle.Tensor,
        input15: paddle.Tensor,
        input16: paddle.Tensor,
        input17: paddle.Tensor,
    ):
        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        t0 = [input0, input1, input2, input3, input4, input5]
        del input0, input1, input2, input3, input4, input5

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        t1 = paddle._C_ops.add_n(t0)
        del t0

        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        t2 = [input6, input7, input8, input9, input10, input11]
        del input10, input11, input6, input7, input8, input9

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        t3 = paddle._C_ops.add_n(t2)
        del t2

        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        t4 = [input12, input13, input14, input15, input16, input17]
        del input12, input13, input14, input15, input16, input17

        return t1, t3, t4
