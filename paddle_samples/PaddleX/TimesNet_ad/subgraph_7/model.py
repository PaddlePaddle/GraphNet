import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.mean: (-1x1x2xf32) <- (-1x96x2xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(data_0, full_int_array_0, True)

        # pd_op.share_data_: (-1x1x2xf32) <- (-1x1x2xf32)
        share_data__0 = mean_0.detach()
        del mean_0

        # pd_op.subtract: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        subtract_0 = paddle._C_ops.subtract(data_0, share_data__0)
        del data_0

        # pd_op.mean: (-1x1x2xf32) <- (-1x96x2xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(subtract_0, full_int_array_0, True)

        # pd_op.subtract: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(subtract_0, mean_1)
        del mean_1

        # pd_op.pow: (-1x96x2xf32) <- (-1x96x2xf32)
        pow_0 = paddle._C_ops.pow(subtract_1, float("2"))
        del subtract_1

        # pd_op.sum: (-1x1x2xf32) <- (-1x96x2xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(pow_0, full_int_array_0, paddle.float32, True)
        del pow_0

        # pd_op.numel: (xi64) <- (-1x96x2xf32)
        numel_0 = paddle._C_ops.numel(subtract_0)

        # pd_op.cast: (xi64) <- (xi64)
        cast_0 = paddle._C_ops.cast(numel_0, paddle.int64)
        del numel_0

        # pd_op.numel: (xi64) <- (-1x1x2xf32)
        numel_1 = paddle._C_ops.numel(sum_0)

        # pd_op.cast: (xi64) <- (xi64)
        cast_1 = paddle._C_ops.cast(numel_1, paddle.int64)
        del numel_1

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(cast_0, paddle.float32)
        del cast_0

        # pd_op.cast: (xf32) <- (xi64)
        cast_3 = paddle._C_ops.cast(cast_1, paddle.float32)
        del cast_1

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(cast_2, cast_3)
        del cast_2, cast_3

        # pd_op.divide: (-1x1x2xf32) <- (-1x1x2xf32, xf32)
        divide_1 = paddle._C_ops.divide(sum_0, divide_0)
        del divide_0, sum_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1x2xf32) <- (-1x1x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_1, full_0, float("1e-05"), True)
        del divide_1, full_0

        # pd_op.sqrt: (-1x1x2xf32) <- (-1x1x2xf32)
        sqrt_0 = paddle._C_ops.sqrt(scale_0)
        del scale_0

        # pd_op.divide: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        divide_2 = paddle._C_ops.divide(subtract_0, sqrt_0)
        del subtract_0

        # pd_op.transpose: (-1x2x96xf32) <- (-1x96x2xf32)
        transpose_0 = paddle._C_ops.transpose(divide_2, [0, 2, 1])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [3, 4]

        # pd_op.unsqueeze: (-1x2x96x1x1xf32) <- (-1x2x96xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_0, full_int_array_1)
        del transpose_0

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_2 = [0, 0, 0, 0, 1, 1]

        # pd_op.pad3d: (-1x2x98x1x1xf32) <- (-1x2x96x1x1xf32, 6xi64)
        pad3d_0 = paddle._C_ops.pad3d(
            unsqueeze_0, full_int_array_2, "circular", float("0"), "NCDHW"
        )
        del full_int_array_2, unsqueeze_0

        # pd_op.squeeze: (-1x2x98xf32) <- (-1x2x98x1x1xf32, 2xi64)
        squeeze_0 = paddle._C_ops.squeeze(pad3d_0, full_int_array_1)
        del full_int_array_1, pad3d_0

        # pd_op.assign: (32x2x3xf32) <- (32x2x3xf32)
        assign_0 = parameter_0
        del parameter_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-2]

        # pd_op.unsqueeze: (32x2x1x3xf32) <- (32x2x3xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(assign_0, full_int_array_3)
        del assign_0

        # pd_op.unsqueeze: (-1x2x1x98xf32) <- (-1x2x98xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_3)
        del squeeze_0

        # pd_op.conv2d: (-1x32x1x96xf32) <- (-1x2x1x98xf32, 32x2x1x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            unsqueeze_2, unsqueeze_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_1, unsqueeze_2

        # pd_op.squeeze: (-1x32x96xf32) <- (-1x32x1x96xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(conv2d_0, full_int_array_3)
        del conv2d_0, full_int_array_3

        # pd_op.transpose: (-1x96x32xf32) <- (-1x32x96xf32)
        transpose_1 = paddle._C_ops.transpose(squeeze_1, [0, 2, 1])
        del squeeze_1

        # pd_op.shape64: (3xi64) <- (-1x96x2xf32)
        shape64_0 = paddle._C_ops.shape64(divide_2)
        del divide_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_4, full_int_array_0, [1], [0]
        )
        del full_int_array_0, shape64_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [96]

        # pd_op.slice: (1x96x32xf32) <- (1x5000x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [1], full_int_array_4, full_int_array_5, [1], []
        )
        del data_1, full_int_array_4, full_int_array_5

        # pd_op.add: (-1x96x32xf32) <- (-1x96x32xf32, 1x96x32xf32)
        add_0 = paddle._C_ops.add(transpose_1, slice_1)
        del slice_1, transpose_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x96x32xf32, -1x96x32xui8) <- (-1x96x32xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_0, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_0, full_1, share_data__0, sqrt_0

        return dropout_0
