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
    ):
        # pd_op.full: (242991xi32) <- ()
        t0 = paddle._C_ops.full(
            [242991],
            float("-1"),
            paddle.int32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf32) <- ()
        t1 = paddle._C_ops.full([1], float("0"), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full_like: (250xi32) <- (250xi32, 1xf32)
        t2 = paddle._C_ops.full_like(
            input0, t1, paddle.int32, paddle.framework._current_expected_place()
        )
        del t1

        # pd_op.scatter: (242991xi32) <- (242991xi32, 250xi32, 250xi32)
        t3 = paddle._C_ops.scatter(t0, input0, t2, True)
        del input0, t0, t2

        # pd_op.full: (1xf32) <- ()
        t4 = paddle._C_ops.full([1], float("1"), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full_like: (6xi32) <- (6xi32, 1xf32)
        t5 = paddle._C_ops.full_like(
            input1, t4, paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.scatter: (242991xi32) <- (242991xi32, 6xi32, 6xi32)
        t6 = paddle._C_ops.scatter(t3, input1, t5, True)
        del input1, t5, t3

        # pd_op.full: (1xi32) <- ()
        t7 = paddle._C_ops.full([1], float("0"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (242991x4xf32) <- (1x4xf32, 242991xi64, 1xi32)
        t8 = paddle._C_ops.gather(input2, input3, t7)
        del input2, input3, t7

        # pd_op.full_int_array: (1xi64) <- ()
        t9 = [2]

        return t4, t6, t8, t9
