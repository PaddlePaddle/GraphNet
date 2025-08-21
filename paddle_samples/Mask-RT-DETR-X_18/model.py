import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.share_data_: (-1x144x144xf32) <- (-1x144x144xf32)
        share_data__0 = data_0.detach()

        # pd_op.shape64: (3xi64) <- (-1x144x144xf32)
        shape64_0 = paddle._C_ops.shape64(share_data__0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("37632"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [slice_0, full_0, full_1, full_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x37632x2xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            stack_0,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.unsqueeze: (-1x1x144x144xf32) <- (-1x144x144xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(share_data__0, full_int_array_1)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1x37632x2xf32) <- (-1x1x37632x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(uniform_0, full_5, float("0"), True)

        # pd_op.scale: (-1x1x37632x2xf32) <- (-1x1x37632x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, float("-1"), True)

        # pd_op.grid_sample: (-1x1x1x37632xf32) <- (-1x1x144x144xf32, -1x1x37632x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(
            unsqueeze_0, scale_1, "bilinear", "zeros", False
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 2]

        # pd_op.squeeze: (-1x37632xf32) <- (-1x1x1x37632xf32, 2xi64)
        squeeze_0 = paddle._C_ops.squeeze(grid_sample_0, full_int_array_2)

        # pd_op.abs: (-1x37632xf32) <- (-1x37632xf32)
        abs_0 = paddle._C_ops.abs(squeeze_0)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x37632xf32) <- (-1x37632xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(abs_0, full_6, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("9408"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (-1x9408xf32, -1x9408xi64) <- (-1x37632xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(scale_2, full_7, 1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.full: (1xi64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_8, slice_0, full_9, dtype="int64")

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.unsqueeze: (-1x1xi64) <- (-1xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(arange_0, full_int_array_3)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 9408]

        # pd_op.tile: (-1x9408xi64) <- (-1x1xi64, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_1, full_int_array_4)

        # builtin.combine: ([-1x9408xi64, -1x9408xi64]) <- (-1x9408xi64, -1x9408xi64)
        combine_1 = [tile_0, topk_1]

        # pd_op.stack: (-1x9408x2xi64) <- ([-1x9408xi64, -1x9408xi64])
        stack_1 = paddle._C_ops.stack(combine_1, -1)

        # pd_op.squeeze: (-1x37632x2xf32) <- (-1x1x37632x2xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(uniform_0, full_int_array_1)

        # pd_op.gather_nd: (-1x9408x2xf32) <- (-1x37632x2xf32, -1x9408x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(squeeze_1, stack_1)

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("3136"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_2 = [slice_0, full_10, full_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.uniform: (-1x3136x2xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            stack_2,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x9408x2xf32, -1x3136x2xf32]) <- (-1x9408x2xf32, -1x3136x2xf32)
        combine_3 = [gather_nd_0, uniform_1]

        # pd_op.concat: (-1x12544x2xf32) <- ([-1x9408x2xf32, -1x3136x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_3, full_11)
        return concat_0
