import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.fft_r2c: (-1x97x32xc64) <- (-1x192x32xf32)
        fft_r2c_0 = paddle._C_ops.fft_r2c(data_0, [1], "backward", True, True)

        # pd_op.abs: (-1x97x32xf32) <- (-1x97x32xc64)
        abs_0 = paddle._C_ops.abs(fft_r2c_0)

        # pd_op.assign: (-1x97x32xf32) <- (-1x97x32xf32)
        assign_0 = abs_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.mean: (97x32xf32) <- (-1x97x32xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(abs_0, full_int_array_0, False)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.mean: (97xf32) <- (97x32xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(mean_0, full_int_array_1, False)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.set_value_: (97xf32) <- (97xf32, 1xi64, 1xi64, 1xi64)
        set_value__0 = paddle._C_ops.set_value_(
            mean_1,
            full_int_array_0,
            full_int_array_2,
            full_int_array_2,
            [0],
            [0],
            [],
            [1],
            [float("0")],
        )
        del mean_1

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("5"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (5xf32, 5xi64) <- (97xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(set_value__0, full_0, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_0

        # pd_op.share_data_: (5xi64) <- (5xi64)
        share_data__0 = topk_1.detach()
        del topk_1

        # pd_op.cast: (5xi32) <- (5xi64)
        cast_0 = paddle._C_ops.cast(share_data__0, paddle.int32)
        del share_data__0

        # pd_op.shape64: (3xi64) <- (-1x192x32xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)
        del data_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_0

        # pd_op.cast: (xi32) <- (xi64)
        cast_1 = paddle._C_ops.cast(slice_0, paddle.int32)
        del slice_0

        # pd_op.floor_divide: (5xi32) <- (xi32, 5xi32)
        floor_divide_0 = paddle._C_ops.floor_divide(cast_1, cast_0)
        del cast_1

        # pd_op.mean: (-1x97xf32) <- (-1x97x32xf32, 1xi64)
        mean_2 = paddle._C_ops.mean(abs_0, full_int_array_1, False)

        # pd_op.index_select: (-1x5xf32) <- (-1x97xf32, 5xi32)
        index_select_0 = paddle._C_ops.index_select(mean_2, cast_0, 1)
        del (
            abs_0,
            assign_0,
            assign_1,
            cast_0,
            fft_r2c_0,
            full_int_array_0,
            full_int_array_1,
            mean_0,
            mean_2,
            set_value__0,
        )

        return floor_divide_0, index_select_0
