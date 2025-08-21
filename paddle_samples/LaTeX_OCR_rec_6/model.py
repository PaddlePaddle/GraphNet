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
        data_0,
        data_1,
    ):
        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("11"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_1)

        # pd_op.layer_norm: (20x90x256xf32, 20x90xf32, 20x90xf32) <- (20x90x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                data_1, parameter_5, parameter_4, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (20x90x2048xf32) <- (20x90x256xf32, 256x2048xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_3, False, False)

        # pd_op.add: (20x90x2048xf32) <- (20x90x2048xf32, 2048xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([20x90x1024xf32, 20x90x1024xf32]) <- (20x90x2048xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add_2, 2, full_0)

        # builtin.split: (20x90x1024xf32, 20x90x1024xf32) <- ([20x90x1024xf32, 20x90x1024xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0

        # pd_op.gelu: (20x90x1024xf32) <- (20x90x1024xf32)
        gelu_0 = paddle._C_ops.gelu(split_1, False)

        # pd_op.multiply: (20x90x1024xf32) <- (20x90x1024xf32, 20x90x1024xf32)
        multiply_0 = paddle._C_ops.multiply(split_0, gelu_0)

        # pd_op.matmul: (20x90x256xf32) <- (20x90x1024xf32, 1024x256xf32)
        matmul_1 = paddle._C_ops.matmul(multiply_0, parameter_1, False, False)

        # pd_op.add: (20x90x256xf32) <- (20x90x256xf32, 256xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_0)

        # pd_op.add: (20x90x256xf32) <- (20x90x256xf32, 20x90x256xf32)
        add_0 = paddle._C_ops.add(add_1, data_1)
        return (
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
            full_0,
            split_0,
            split_1,
            gelu_0,
            multiply_0,
            matmul_1,
            equal_0,
            add_0,
            add_1,
        )
