import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.unsqueeze: (2x60x1x4xf32) <- (2x60x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.unsqueeze: (2x1x2541x4xf32) <- (2x2541x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_1, full_int_array_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.slice: (2x60x1x2xf32) <- (2x60x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            unsqueeze_0, [3], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2147483647]

        # pd_op.slice: (2x60x1x2xf32) <- (2x60x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_0, [3], full_int_array_0, full_int_array_3, [1], []
        )

        # pd_op.slice: (2x1x2541x2xf32) <- (2x1x2541x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.slice: (2x1x2541x2xf32) <- (2x1x2541x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_0, full_int_array_3, [1], []
        )

        # pd_op.maximum: (2x60x2541x2xf32) <- (2x60x1x2xf32, 2x1x2541x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_0, slice_2)

        # pd_op.minimum: (2x60x2541x2xf32) <- (2x60x1x2xf32, 2x1x2541x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_1, slice_3)

        # pd_op.subtract: (2x60x2541x2xf32) <- (2x60x2541x2xf32, 2x60x2541x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (2x60x2541x2xf32) <- (2x60x2541x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_0, full_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.prod: (2x60x2541xf32) <- (2x60x2541x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_4, False, False)

        # pd_op.subtract: (2x60x1x2xf32) <- (2x60x1x2xf32, 2x60x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_1, slice_0)

        # pd_op.clip: (2x60x1x2xf32) <- (2x60x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_0, full_1)

        # pd_op.prod: (2x60x1xf32) <- (2x60x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_4, False, False)

        # pd_op.subtract: (2x1x2541x2xf32) <- (2x1x2541x2xf32, 2x1x2541x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_3, slice_2)

        # pd_op.clip: (2x1x2541x2xf32) <- (2x1x2541x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_0, full_1)

        # pd_op.prod: (2x1x2541xf32) <- (2x1x2541x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_4, False, False)

        # pd_op.add: (2x60x2541xf32) <- (2x60x1xf32, 2x1x2541xf32)
        add_0 = paddle._C_ops.add(prod_1, prod_2)

        # pd_op.subtract: (2x60x2541xf32) <- (2x60x2541xf32, 2x60x2541xf32)
        subtract_3 = paddle._C_ops.subtract(add_0, prod_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x60x2541xf32) <- (2x60x2541xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_3, full_2, float("1e-09"), True)

        # pd_op.divide: (2x60x2541xf32) <- (2x60x2541xf32, 2x60x2541xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_0)
        return divide_0
