import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, data_0, data_1):
        # pd_op.transpose: (11x3x405xf32) <- (11x405x3xf32)
        transpose_0 = paddle._C_ops.transpose(data_0, [0, 2, 1])
        del data_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 4]

        # pd_op.unsqueeze: (11x3x405x1x1xf32) <- (11x3x405xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_0, full_int_array_0)
        del transpose_0

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_1 = [0, 0, 0, 0, 1, 1]

        # pd_op.pad3d: (11x3x407x1x1xf32) <- (11x3x405x1x1xf32, 6xi64)
        pad3d_0 = paddle._C_ops.pad3d(
            unsqueeze_0, full_int_array_1, "circular", float("0"), "NCDHW"
        )
        del full_int_array_1, unsqueeze_0

        # pd_op.squeeze: (11x3x407xf32) <- (11x3x407x1x1xf32, 2xi64)
        squeeze_0 = paddle._C_ops.squeeze(pad3d_0, full_int_array_0)
        del full_int_array_0, pad3d_0

        # pd_op.assign: (32x3x3xf32) <- (32x3x3xf32)
        assign_0 = parameter_0
        del parameter_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_2

        # pd_op.unsqueeze: (32x3x1x3xf32) <- (32x3x3xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(assign_0, full_int_array_2)

        # pd_op.unsqueeze: (11x3x1x407xf32) <- (11x3x407xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_2)
        del squeeze_0

        # pd_op.conv2d: (11x32x1x405xf32) <- (11x3x1x407xf32, 32x3x1x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            unsqueeze_2, unsqueeze_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.squeeze: (11x32x405xf32) <- (11x32x1x405xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(conv2d_0, full_int_array_2)

        # pd_op.transpose: (11x405x32xf32) <- (11x32x405xf32)
        transpose_1 = paddle._C_ops.transpose(squeeze_1, [0, 2, 1])
        del squeeze_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [405]

        # pd_op.slice: (1x405x32xf32) <- (1x5000x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_1, [1], full_int_array_3, full_int_array_4, [1], []
        )
        del data_1, full_int_array_3, full_int_array_4

        # pd_op.add: (11x405x32xf32) <- (11x405x32xf32, 1x405x32xf32)
        add_0 = paddle._C_ops.add(transpose_1, slice_0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (11x405x32xf32, 11x405x32xui8) <- (11x405x32xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_0, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del (
            add_0,
            assign_0,
            assign_1,
            conv2d_0,
            full_0,
            full_int_array_2,
            slice_0,
            transpose_1,
            unsqueeze_1,
            unsqueeze_2,
        )

        return dropout_0
