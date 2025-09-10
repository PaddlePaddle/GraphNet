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
        parameter_8,
        parameter_9,
        parameter_10,
        parameter_11,
        parameter_12,
        parameter_13,
        parameter_14,
        parameter_15,
        parameter_16,
        parameter_17,
        parameter_18,
        parameter_19,
        parameter_20,
        parameter_21,
        parameter_22,
        parameter_23,
        parameter_24,
        parameter_25,
        parameter_26,
        parameter_27,
        parameter_28,
        parameter_29,
        parameter_30,
        parameter_31,
        parameter_32,
        parameter_33,
        parameter_34,
        parameter_35,
        parameter_36,
        parameter_37,
        parameter_38,
        parameter_39,
        parameter_40,
        parameter_41,
        parameter_42,
        parameter_43,
        parameter_44,
        parameter_45,
        parameter_46,
        parameter_47,
        parameter_48,
        data_0,
        data_1,
    ):
        # pd_op.cast: (-1x96x2xf32) <- (-1x96x2xf32)
        cast_0 = paddle._C_ops.cast(data_0, paddle.float32)
        del data_0

        # pd_op.assign: (-1x96x2xf32) <- (-1x96x2xf32)
        assign_0 = cast_0

        # pd_op.assign: (-1x96x2xf32) <- (-1x96x2xf32)
        assign_1 = cast_0

        # pd_op.share_data_: (-1x96x2xf32) <- (-1x96x2xf32)
        share_data__0 = assign_1.detach()
        del assign_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.mean: (-1x1x2xf32) <- (-1x96x2xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(cast_0, full_int_array_0, True)

        # pd_op.share_data_: (-1x1x2xf32) <- (-1x1x2xf32)
        share_data__1 = mean_0.detach()
        del mean_0

        # pd_op.subtract: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        subtract_0 = paddle._C_ops.subtract(cast_0, share_data__1)
        del cast_0

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
        cast_1 = paddle._C_ops.cast(numel_0, paddle.int64)
        del numel_0

        # pd_op.numel: (xi64) <- (-1x1x2xf32)
        numel_1 = paddle._C_ops.numel(sum_0)

        # pd_op.cast: (xi64) <- (xi64)
        cast_2 = paddle._C_ops.cast(numel_1, paddle.int64)
        del numel_1

        # pd_op.cast: (xf32) <- (xi64)
        cast_3 = paddle._C_ops.cast(cast_1, paddle.float32)
        del cast_1

        # pd_op.cast: (xf32) <- (xi64)
        cast_4 = paddle._C_ops.cast(cast_2, paddle.float32)
        del cast_2

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(cast_3, cast_4)
        del cast_3, cast_4

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

        # pd_op.share_data_: (-1x1x2xf32) <- (-1x1x2xf32)
        share_data__2 = sqrt_0.detach()
        del sqrt_0

        # pd_op.divide: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        divide_2 = paddle._C_ops.divide(subtract_0, share_data__2)
        del subtract_0

        # pd_op.shape64: (3xi64) <- (-1x96x2xf32)
        shape64_0 = paddle._C_ops.shape64(share_data__0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_0, [1], [0]
        )
        del shape64_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [3, 4]

        # pd_op.unsqueeze: (-1x96x2x1x1xf32) <- (-1x96x2xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(share_data__0, full_int_array_2)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_3 = [0, 0, 0, 0, 1, 1]

        # pd_op.pad3d: (-1x96x4x1x1xf32) <- (-1x96x2x1x1xf32, 6xi64)
        pad3d_0 = paddle._C_ops.pad3d(
            unsqueeze_0, full_int_array_3, "circular", float("0"), "NCDHW"
        )
        del unsqueeze_0

        # pd_op.squeeze: (-1x96x4xf32) <- (-1x96x4x1x1xf32, 2xi64)
        squeeze_0 = paddle._C_ops.squeeze(pad3d_0, full_int_array_2)
        del pad3d_0

        # pd_op.assign: (1x96x3xf32) <- (1x96x3xf32)
        assign_2 = parameter_48
        del parameter_48

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-2]

        # pd_op.unsqueeze: (1x96x1x3xf32) <- (1x96x3xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(assign_2, full_int_array_4)
        del assign_2

        # pd_op.unsqueeze: (-1x96x1x4xf32) <- (-1x96x4xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_4)
        del squeeze_0

        # pd_op.conv2d: (-1x1x1x2xf32) <- (-1x96x1x4xf32, 1x96x1x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            unsqueeze_2, unsqueeze_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_1, unsqueeze_2

        # pd_op.squeeze: (-1x1x2xf32) <- (-1x1x1x2xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(conv2d_0, full_int_array_4)
        del conv2d_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x1x2xf32, -1x1x2xf32]) <- (-1x1x2xf32, -1x1x2xf32)
        combine_0 = [squeeze_1, share_data__2]
        del squeeze_1

        # pd_op.concat: (-1x2x2xf32) <- ([-1x1x2xf32, -1x1x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)
        del combine_0

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_1 = [slice_0, full_2]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x-1xf32) <- (-1x2x2xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(concat_0, stack_0)
        del concat_0, stack_0

        # pd_op.matmul: (-1x32xf32) <- (-1x-1xf32, 4x32xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_0, parameter_47, False, False)
        del parameter_47, reshape_0

        # pd_op.add: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_46)
        del matmul_0, parameter_46

        # pd_op.relu: (-1x32xf32) <- (-1x32xf32)
        relu_0 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.matmul: (-1x32xf32) <- (-1x32xf32, 32x32xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_45, False, False)
        del parameter_45, relu_0

        # pd_op.add: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_44)
        del matmul_1, parameter_44

        # pd_op.relu: (-1x32xf32) <- (-1x32xf32)
        relu_1 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.matmul: (-1x1xf32) <- (-1x32xf32, 32x1xf32)
        matmul_2 = paddle._C_ops.matmul(relu_1, parameter_43, False, False)
        del parameter_43, relu_1

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_0 = paddle._C_ops.exp(matmul_2)
        del matmul_2

        # pd_op.unsqueeze: (-1x96x2x1x1xf32) <- (-1x96x2xf32, 2xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(share_data__0, full_int_array_2)
        del share_data__0

        # pd_op.pad3d: (-1x96x4x1x1xf32) <- (-1x96x2x1x1xf32, 6xi64)
        pad3d_1 = paddle._C_ops.pad3d(
            unsqueeze_3, full_int_array_3, "circular", float("0"), "NCDHW"
        )
        del unsqueeze_3

        # pd_op.squeeze: (-1x96x4xf32) <- (-1x96x4x1x1xf32, 2xi64)
        squeeze_2 = paddle._C_ops.squeeze(pad3d_1, full_int_array_2)
        del pad3d_1

        # pd_op.assign: (1x96x3xf32) <- (1x96x3xf32)
        assign_3 = parameter_42
        del parameter_42

        # pd_op.unsqueeze: (1x96x1x3xf32) <- (1x96x3xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(assign_3, full_int_array_4)
        del assign_3

        # pd_op.unsqueeze: (-1x96x1x4xf32) <- (-1x96x4xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(squeeze_2, full_int_array_4)
        del squeeze_2

        # pd_op.conv2d: (-1x1x1x2xf32) <- (-1x96x1x4xf32, 1x96x1x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            unsqueeze_5, unsqueeze_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_4, unsqueeze_5

        # pd_op.squeeze: (-1x1x2xf32) <- (-1x1x1x2xf32, 1xi64)
        squeeze_3 = paddle._C_ops.squeeze(conv2d_1, full_int_array_4)
        del conv2d_1

        # builtin.combine: ([-1x1x2xf32, -1x1x2xf32]) <- (-1x1x2xf32, -1x1x2xf32)
        combine_2 = [squeeze_3, share_data__1]
        del squeeze_3

        # pd_op.concat: (-1x2x2xf32) <- ([-1x1x2xf32, -1x1x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_1)
        del combine_2, full_1

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_3 = [slice_0, full_2]
        del slice_0

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.reshape: (-1x-1xf32) <- (-1x2x2xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(concat_1, stack_1)
        del concat_1, stack_1

        # pd_op.matmul: (-1x32xf32) <- (-1x-1xf32, 4x32xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_1, parameter_41, False, False)
        del parameter_41, reshape_1

        # pd_op.add: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_40)
        del matmul_3, parameter_40

        # pd_op.relu: (-1x32xf32) <- (-1x32xf32)
        relu_2 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.matmul: (-1x32xf32) <- (-1x32xf32, 32x32xf32)
        matmul_4 = paddle._C_ops.matmul(relu_2, parameter_39, False, False)
        del parameter_39, relu_2

        # pd_op.add: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add_4 = paddle._C_ops.add(matmul_4, parameter_38)
        del matmul_4, parameter_38

        # pd_op.relu: (-1x32xf32) <- (-1x32xf32)
        relu_3 = paddle._C_ops.relu(add_4)
        del add_4

        # pd_op.matmul: (-1x96xf32) <- (-1x32xf32, 32x96xf32)
        matmul_5 = paddle._C_ops.matmul(relu_3, parameter_37, False, False)
        del parameter_37, relu_3

        # pd_op.transpose: (-1x2x96xf32) <- (-1x96x2xf32)
        transpose_0 = paddle._C_ops.transpose(divide_2, [0, 2, 1])

        # pd_op.unsqueeze: (-1x2x96x1x1xf32) <- (-1x2x96xf32, 2xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(transpose_0, full_int_array_2)
        del transpose_0

        # pd_op.pad3d: (-1x2x98x1x1xf32) <- (-1x2x96x1x1xf32, 6xi64)
        pad3d_2 = paddle._C_ops.pad3d(
            unsqueeze_6, full_int_array_3, "circular", float("0"), "NCDHW"
        )
        del full_int_array_3, unsqueeze_6

        # pd_op.squeeze: (-1x2x98xf32) <- (-1x2x98x1x1xf32, 2xi64)
        squeeze_4 = paddle._C_ops.squeeze(pad3d_2, full_int_array_2)
        del full_int_array_2, pad3d_2

        # pd_op.assign: (64x2x3xf32) <- (64x2x3xf32)
        assign_4 = parameter_36
        del parameter_36

        # pd_op.unsqueeze: (64x2x1x3xf32) <- (64x2x3xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(assign_4, full_int_array_4)
        del assign_4

        # pd_op.unsqueeze: (-1x2x1x98xf32) <- (-1x2x98xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(squeeze_4, full_int_array_4)
        del squeeze_4

        # pd_op.conv2d: (-1x64x1x96xf32) <- (-1x2x1x98xf32, 64x2x1x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            unsqueeze_8, unsqueeze_7, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_7, unsqueeze_8

        # pd_op.squeeze: (-1x64x96xf32) <- (-1x64x1x96xf32, 1xi64)
        squeeze_5 = paddle._C_ops.squeeze(conv2d_2, full_int_array_4)
        del conv2d_2

        # pd_op.transpose: (-1x96x64xf32) <- (-1x64x96xf32)
        transpose_1 = paddle._C_ops.transpose(squeeze_5, [0, 2, 1])
        del squeeze_5

        # pd_op.shape64: (3xi64) <- (-1x96x2xf32)
        shape64_1 = paddle._C_ops.shape64(divide_2)
        del divide_2

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_0, [1], [0]
        )
        del shape64_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [96]

        # pd_op.slice: (1x96x64xf32) <- (1x5000x64xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_5, [1], []
        )
        del data_1, full_int_array_5

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, 1x96x64xf32)
        add_5 = paddle._C_ops.add(transpose_1, slice_2)
        del slice_2, transpose_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x96x64xf32, -1x96x64xui8) <- (-1x96x64xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_5, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_5

        # pd_op.shape64: (3xi64) <- (-1x96x64xf32)
        shape64_2 = paddle._C_ops.shape64(dropout_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_0, [1], [0]
        )
        del shape64_2

        # pd_op.matmul: (-1x96x64xf32) <- (-1x96x64xf32, 64x64xf32)
        matmul_6 = paddle._C_ops.matmul(dropout_0, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, 64xf32)
        add_6 = paddle._C_ops.add(matmul_6, parameter_34)
        del matmul_6, parameter_34

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("96"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_4 = [slice_3, full_4, full_5, full_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.reshape: (-1x96x8x-1xf32) <- (-1x96x64xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_6, stack_2)
        del add_6, stack_2

        # pd_op.matmul: (-1x96x64xf32) <- (-1x96x64xf32, 64x64xf32)
        matmul_7 = paddle._C_ops.matmul(dropout_0, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, 64xf32)
        add_7 = paddle._C_ops.add(matmul_7, parameter_32)
        del matmul_7, parameter_32

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_5 = [slice_3, full_4, full_5, full_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.reshape: (-1x96x8x-1xf32) <- (-1x96x64xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(add_7, stack_3)
        del add_7, stack_3

        # pd_op.matmul: (-1x96x64xf32) <- (-1x96x64xf32, 64x64xf32)
        matmul_8 = paddle._C_ops.matmul(dropout_0, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, 64xf32)
        add_8 = paddle._C_ops.add(matmul_8, parameter_30)
        del matmul_8, parameter_30

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [slice_3, full_4, full_5, full_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.reshape: (-1x96x8x-1xf32) <- (-1x96x64xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_8, stack_4)
        del add_8, stack_4

        # pd_op.shape64: (4xi64) <- (-1x96x8x-1xf32)
        shape64_3 = paddle._C_ops.shape64(reshape_2)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_0, [1], [0]
        )
        del shape64_3

        # pd_op.shape64: (4xi64) <- (-1x96x8x-1xf32)
        shape64_4 = paddle._C_ops.shape64(reshape_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_6, full_int_array_7, [1], [0]
        )
        del shape64_4

        # pd_op.shape64: (4xi64) <- (-1x96x8x-1xf32)
        shape64_5 = paddle._C_ops.shape64(reshape_4)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_0, [1], [0]
        )
        del shape64_5

        # pd_op.shape64: (4xi64) <- (-1x96x8x-1xf32)
        shape64_6 = paddle._C_ops.shape64(reshape_4)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_6, full_int_array_7, [1], [0]
        )
        del shape64_6

        # pd_op.unsqueeze: (-1x1x1xf32) <- (-1x1xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(exp_0, full_int_array_0)

        # pd_op.unsqueeze: (-1x1x1x1xf32) <- (-1x1x1xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(unsqueeze_9, full_int_array_0)
        del unsqueeze_9

        # pd_op.unsqueeze: (-1x1x96xf32) <- (-1x96xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(matmul_5, full_int_array_0)

        # pd_op.unsqueeze: (-1x1x1x96xf32) <- (-1x1x96xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(unsqueeze_11, full_int_array_0)
        del unsqueeze_11

        # builtin.combine: ([-1x96x8x-1xf32, -1x96x8x-1xf32]) <- (-1x96x8x-1xf32, -1x96x8x-1xf32)
        combine_7 = [reshape_2, reshape_3]
        del reshape_2, reshape_3

        # pd_op.einsum: (-1x8x96x96xf32, [0xf32, 0xf32], [-1x96x8x-1xf32, -1x96x8x-1xf32]) <- ([-1x96x8x-1xf32, -1x96x8x-1xf32])
        einsum_0, einsum_1, einsum_2 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_7, "blhe,bshe->bhls"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_7

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_0,
            split_1,
        ) = einsum_1
        del einsum_1

        # builtin.split: (-1x96x8x-1xf32, -1x96x8x-1xf32) <- ([-1x96x8x-1xf32, -1x96x8x-1xf32])
        (
            split_2,
            split_3,
        ) = einsum_2
        del einsum_2

        # pd_op.multiply: (-1x8x96x96xf32) <- (-1x8x96x96xf32, -1x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(einsum_0, unsqueeze_10)
        del einsum_0, unsqueeze_10

        # pd_op.add: (-1x8x96x96xf32) <- (-1x8x96x96xf32, -1x1x1x96xf32)
        add_9 = paddle._C_ops.add(multiply_0, unsqueeze_12)
        del multiply_0, unsqueeze_12

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x8x96x96xf32) <- (-1x8x96x96xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_9, full_6, float("0"), True)
        del add_9

        # pd_op.softmax: (-1x8x96x96xf32) <- (-1x8x96x96xf32)
        softmax_0 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.dropout: (-1x8x96x96xf32, -1x8x96x96xui8) <- (-1x8x96x96xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # builtin.combine: ([-1x8x96x96xf32, -1x96x8x-1xf32]) <- (-1x8x96x96xf32, -1x96x8x-1xf32)
        combine_8 = [dropout_2, reshape_4]
        del dropout_2, reshape_4

        # pd_op.einsum: (-1x96x8x-1xf32, [0xf32, 0xf32], [-1x8x96x96xf32, -1x96x8x-1xf32]) <- ([-1x8x96x96xf32, -1x96x8x-1xf32])
        einsum_3, einsum_4, einsum_5 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_8, "bhls,bshd->blhd"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_8

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_4,
            split_5,
        ) = einsum_4
        del einsum_4

        # builtin.split: (-1x8x96x96xf32, -1x96x8x-1xf32) <- ([-1x8x96x96xf32, -1x96x8x-1xf32])
        (
            split_6,
            split_7,
        ) = einsum_5
        del einsum_5

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [slice_3, full_4, full_2]
        del slice_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.reshape: (-1x96x-1xf32) <- (-1x96x8x-1xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(einsum_3, stack_5)
        del einsum_3, stack_5

        # pd_op.matmul: (-1x96x64xf32) <- (-1x96x-1xf32, 64x64xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_5, parameter_29, False, False)
        del parameter_29, reshape_5

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, 64xf32)
        add_10 = paddle._C_ops.add(matmul_9, parameter_28)
        del matmul_9, parameter_28

        # pd_op.dropout: (-1x96x64xf32, -1x96x64xui8) <- (-1x96x64xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_10, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_10

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, -1x96x64xf32)
        add_11 = paddle._C_ops.add(dropout_0, dropout_4)
        del dropout_0, dropout_4

        # pd_op.layer_norm: (-1x96x64xf32, -1x96xf32, -1x96xf32) <- (-1x96x64xf32, 64xf32, 64xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_11, parameter_26, parameter_27

        # pd_op.transpose: (-1x64x96xf32) <- (-1x96x64xf32)
        transpose_2 = paddle._C_ops.transpose(layer_norm_0, [0, 2, 1])

        # pd_op.assign: (64x64x1xf32) <- (64x64x1xf32)
        assign_5 = parameter_25
        del parameter_25

        # pd_op.unsqueeze: (64x64x1x1xf32) <- (64x64x1xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(assign_5, full_int_array_4)
        del assign_5

        # pd_op.unsqueeze: (-1x64x1x96xf32) <- (-1x64x96xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(transpose_2, full_int_array_4)
        del transpose_2

        # pd_op.conv2d: (-1x64x1x96xf32) <- (-1x64x1x96xf32, 64x64x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            unsqueeze_14, unsqueeze_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_13, unsqueeze_14

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_24, full_int_array_8)
        del parameter_24

        # pd_op.add: (-1x64x1x96xf32) <- (-1x64x1x96xf32, 1x64x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_3, reshape_6)
        del conv2d_3, reshape_6

        # pd_op.squeeze: (-1x64x96xf32) <- (-1x64x1x96xf32, 1xi64)
        squeeze_6 = paddle._C_ops.squeeze(add_12, full_int_array_4)
        del add_12

        # pd_op.gelu: (-1x64x96xf32) <- (-1x64x96xf32)
        gelu_0 = paddle._C_ops.gelu(squeeze_6, False)
        del squeeze_6

        # pd_op.dropout: (-1x64x96xf32, -1x64x96xui8) <- (-1x64x96xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_0, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_0

        # pd_op.assign: (64x64x1xf32) <- (64x64x1xf32)
        assign_6 = parameter_23
        del parameter_23

        # pd_op.unsqueeze: (64x64x1x1xf32) <- (64x64x1xf32, 1xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(assign_6, full_int_array_4)
        del assign_6

        # pd_op.unsqueeze: (-1x64x1x96xf32) <- (-1x64x96xf32, 1xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(dropout_6, full_int_array_4)
        del dropout_6

        # pd_op.conv2d: (-1x64x1x96xf32) <- (-1x64x1x96xf32, 64x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            unsqueeze_16, unsqueeze_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_15, unsqueeze_16

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_22, full_int_array_8)
        del parameter_22

        # pd_op.add: (-1x64x1x96xf32) <- (-1x64x1x96xf32, 1x64x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_4, reshape_7)
        del conv2d_4, reshape_7

        # pd_op.squeeze: (-1x64x96xf32) <- (-1x64x1x96xf32, 1xi64)
        squeeze_7 = paddle._C_ops.squeeze(add_13, full_int_array_4)
        del add_13

        # pd_op.transpose: (-1x96x64xf32) <- (-1x64x96xf32)
        transpose_3 = paddle._C_ops.transpose(squeeze_7, [0, 2, 1])
        del squeeze_7

        # pd_op.dropout: (-1x96x64xf32, -1x96x64xui8) <- (-1x96x64xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_3, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_3

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, -1x96x64xf32)
        add_14 = paddle._C_ops.add(layer_norm_0, dropout_8)
        del dropout_8, layer_norm_0

        # pd_op.layer_norm: (-1x96x64xf32, -1x96xf32, -1x96xf32) <- (-1x96x64xf32, 64xf32, 64xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_14, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_14, parameter_20, parameter_21

        # pd_op.shape64: (3xi64) <- (-1x96x64xf32)
        shape64_7 = paddle._C_ops.shape64(layer_norm_3)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_1, full_int_array_0, [1], [0]
        )
        del shape64_7

        # pd_op.matmul: (-1x96x64xf32) <- (-1x96x64xf32, 64x64xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_3, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, 64xf32)
        add_15 = paddle._C_ops.add(matmul_10, parameter_18)
        del matmul_10, parameter_18

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_10 = [slice_8, full_4, full_5, full_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.reshape: (-1x96x8x-1xf32) <- (-1x96x64xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_15, stack_6)
        del add_15, stack_6

        # pd_op.matmul: (-1x96x64xf32) <- (-1x96x64xf32, 64x64xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_3, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, 64xf32)
        add_16 = paddle._C_ops.add(matmul_11, parameter_16)
        del matmul_11, parameter_16

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_11 = [slice_8, full_4, full_5, full_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.reshape: (-1x96x8x-1xf32) <- (-1x96x64xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_16, stack_7)
        del add_16, stack_7

        # pd_op.matmul: (-1x96x64xf32) <- (-1x96x64xf32, 64x64xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_3, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, 64xf32)
        add_17 = paddle._C_ops.add(matmul_12, parameter_14)
        del matmul_12, parameter_14

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_12 = [slice_8, full_4, full_5, full_2]
        del full_5

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.reshape: (-1x96x8x-1xf32) <- (-1x96x64xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_17, stack_8)
        del add_17, stack_8

        # pd_op.shape64: (4xi64) <- (-1x96x8x-1xf32)
        shape64_8 = paddle._C_ops.shape64(reshape_8)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_1, full_int_array_0, [1], [0]
        )
        del shape64_8

        # pd_op.shape64: (4xi64) <- (-1x96x8x-1xf32)
        shape64_9 = paddle._C_ops.shape64(reshape_8)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_6, full_int_array_7, [1], [0]
        )
        del shape64_9

        # pd_op.shape64: (4xi64) <- (-1x96x8x-1xf32)
        shape64_10 = paddle._C_ops.shape64(reshape_10)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_1, full_int_array_0, [1], [0]
        )
        del full_int_array_1, shape64_10

        # pd_op.shape64: (4xi64) <- (-1x96x8x-1xf32)
        shape64_11 = paddle._C_ops.shape64(reshape_10)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_6, full_int_array_7, [1], [0]
        )
        del full_int_array_6, full_int_array_7, shape64_11

        # pd_op.unsqueeze: (-1x1x1xf32) <- (-1x1xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(exp_0, full_int_array_0)
        del exp_0

        # pd_op.unsqueeze: (-1x1x1x1xf32) <- (-1x1x1xf32, 1xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(unsqueeze_17, full_int_array_0)
        del unsqueeze_17

        # pd_op.unsqueeze: (-1x1x96xf32) <- (-1x96xf32, 1xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(matmul_5, full_int_array_0)
        del matmul_5

        # pd_op.unsqueeze: (-1x1x1x96xf32) <- (-1x1x96xf32, 1xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(unsqueeze_19, full_int_array_0)
        del full_int_array_0, unsqueeze_19

        # builtin.combine: ([-1x96x8x-1xf32, -1x96x8x-1xf32]) <- (-1x96x8x-1xf32, -1x96x8x-1xf32)
        combine_13 = [reshape_8, reshape_9]
        del reshape_8, reshape_9

        # pd_op.einsum: (-1x8x96x96xf32, [0xf32, 0xf32], [-1x96x8x-1xf32, -1x96x8x-1xf32]) <- ([-1x96x8x-1xf32, -1x96x8x-1xf32])
        einsum_6, einsum_7, einsum_8 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_13, "blhe,bshe->bhls"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_13

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_8,
            split_9,
        ) = einsum_7
        del einsum_7

        # builtin.split: (-1x96x8x-1xf32, -1x96x8x-1xf32) <- ([-1x96x8x-1xf32, -1x96x8x-1xf32])
        (
            split_10,
            split_11,
        ) = einsum_8
        del einsum_8

        # pd_op.multiply: (-1x8x96x96xf32) <- (-1x8x96x96xf32, -1x1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(einsum_6, unsqueeze_18)
        del einsum_6, unsqueeze_18

        # pd_op.add: (-1x8x96x96xf32) <- (-1x8x96x96xf32, -1x1x1x96xf32)
        add_18 = paddle._C_ops.add(multiply_1, unsqueeze_20)
        del multiply_1, unsqueeze_20

        # pd_op.scale: (-1x8x96x96xf32) <- (-1x8x96x96xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(add_18, full_6, float("0"), True)
        del add_18, full_6

        # pd_op.softmax: (-1x8x96x96xf32) <- (-1x8x96x96xf32)
        softmax_1 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.dropout: (-1x8x96x96xf32, -1x8x96x96xui8) <- (-1x8x96x96xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # builtin.combine: ([-1x8x96x96xf32, -1x96x8x-1xf32]) <- (-1x8x96x96xf32, -1x96x8x-1xf32)
        combine_14 = [dropout_10, reshape_10]
        del dropout_10, reshape_10

        # pd_op.einsum: (-1x96x8x-1xf32, [0xf32, 0xf32], [-1x8x96x96xf32, -1x96x8x-1xf32]) <- ([-1x8x96x96xf32, -1x96x8x-1xf32])
        einsum_9, einsum_10, einsum_11 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_14, "bhls,bshd->blhd"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_14

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_12,
            split_13,
        ) = einsum_10
        del einsum_10

        # builtin.split: (-1x8x96x96xf32, -1x96x8x-1xf32) <- ([-1x8x96x96xf32, -1x96x8x-1xf32])
        (
            split_14,
            split_15,
        ) = einsum_11
        del einsum_11

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_15 = [slice_8, full_4, full_2]
        del full_2, full_4, slice_8

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.reshape: (-1x96x-1xf32) <- (-1x96x8x-1xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(einsum_9, stack_9)
        del einsum_9, stack_9

        # pd_op.matmul: (-1x96x64xf32) <- (-1x96x-1xf32, 64x64xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_11, parameter_13, False, False)
        del parameter_13, reshape_11

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, 64xf32)
        add_19 = paddle._C_ops.add(matmul_13, parameter_12)
        del matmul_13, parameter_12

        # pd_op.dropout: (-1x96x64xf32, -1x96x64xui8) <- (-1x96x64xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_19, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_19

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, -1x96x64xf32)
        add_20 = paddle._C_ops.add(layer_norm_3, dropout_12)
        del dropout_12, layer_norm_3

        # pd_op.layer_norm: (-1x96x64xf32, -1x96xf32, -1x96xf32) <- (-1x96x64xf32, 64xf32, 64xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_11, parameter_10, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_20, parameter_10, parameter_11

        # pd_op.transpose: (-1x64x96xf32) <- (-1x96x64xf32)
        transpose_4 = paddle._C_ops.transpose(layer_norm_6, [0, 2, 1])

        # pd_op.assign: (64x64x1xf32) <- (64x64x1xf32)
        assign_7 = parameter_9
        del parameter_9

        # pd_op.unsqueeze: (64x64x1x1xf32) <- (64x64x1xf32, 1xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(assign_7, full_int_array_4)
        del assign_7

        # pd_op.unsqueeze: (-1x64x1x96xf32) <- (-1x64x96xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(transpose_4, full_int_array_4)
        del transpose_4

        # pd_op.conv2d: (-1x64x1x96xf32) <- (-1x64x1x96xf32, 64x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            unsqueeze_22, unsqueeze_21, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_21, unsqueeze_22

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_8, full_int_array_8)
        del parameter_8

        # pd_op.add: (-1x64x1x96xf32) <- (-1x64x1x96xf32, 1x64x1x1xf32)
        add_21 = paddle._C_ops.add(conv2d_5, reshape_12)
        del conv2d_5, reshape_12

        # pd_op.squeeze: (-1x64x96xf32) <- (-1x64x1x96xf32, 1xi64)
        squeeze_8 = paddle._C_ops.squeeze(add_21, full_int_array_4)
        del add_21

        # pd_op.gelu: (-1x64x96xf32) <- (-1x64x96xf32)
        gelu_1 = paddle._C_ops.gelu(squeeze_8, False)
        del squeeze_8

        # pd_op.dropout: (-1x64x96xf32, -1x64x96xui8) <- (-1x64x96xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_1, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_1

        # pd_op.assign: (64x64x1xf32) <- (64x64x1xf32)
        assign_8 = parameter_7
        del parameter_7

        # pd_op.unsqueeze: (64x64x1x1xf32) <- (64x64x1xf32, 1xi64)
        unsqueeze_23 = paddle._C_ops.unsqueeze(assign_8, full_int_array_4)
        del assign_8

        # pd_op.unsqueeze: (-1x64x1x96xf32) <- (-1x64x96xf32, 1xi64)
        unsqueeze_24 = paddle._C_ops.unsqueeze(dropout_14, full_int_array_4)
        del dropout_14

        # pd_op.conv2d: (-1x64x1x96xf32) <- (-1x64x1x96xf32, 64x64x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            unsqueeze_24, unsqueeze_23, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_23, unsqueeze_24

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_6, full_int_array_8)
        del full_int_array_8, parameter_6

        # pd_op.add: (-1x64x1x96xf32) <- (-1x64x1x96xf32, 1x64x1x1xf32)
        add_22 = paddle._C_ops.add(conv2d_6, reshape_13)
        del conv2d_6, reshape_13

        # pd_op.squeeze: (-1x64x96xf32) <- (-1x64x1x96xf32, 1xi64)
        squeeze_9 = paddle._C_ops.squeeze(add_22, full_int_array_4)
        del add_22, full_int_array_4

        # pd_op.transpose: (-1x96x64xf32) <- (-1x64x96xf32)
        transpose_5 = paddle._C_ops.transpose(squeeze_9, [0, 2, 1])
        del squeeze_9

        # pd_op.dropout: (-1x96x64xf32, -1x96x64xui8) <- (-1x96x64xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_5, None, full_3, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_3, transpose_5

        # pd_op.add: (-1x96x64xf32) <- (-1x96x64xf32, -1x96x64xf32)
        add_23 = paddle._C_ops.add(layer_norm_6, dropout_16)
        del dropout_16, layer_norm_6

        # pd_op.layer_norm: (-1x96x64xf32, -1x96xf32, -1x96xf32) <- (-1x96x64xf32, 64xf32, 64xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_5, parameter_4, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_23, parameter_4, parameter_5

        # pd_op.layer_norm: (-1x96x64xf32, -1x96xf32, -1x96xf32) <- (-1x96x64xf32, 64xf32, 64xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_9, parameter_3, parameter_2, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del layer_norm_9, parameter_2, parameter_3

        # pd_op.matmul: (-1x96x2xf32) <- (-1x96x64xf32, 64x2xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_12, parameter_1, False, False)
        del layer_norm_12, parameter_1

        # pd_op.add: (-1x96x2xf32) <- (-1x96x2xf32, 2xf32)
        add_24 = paddle._C_ops.add(matmul_14, parameter_0)
        del matmul_14, parameter_0

        # pd_op.multiply: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        multiply_2 = paddle._C_ops.multiply(add_24, share_data__2)
        del add_24, share_data__2

        # pd_op.add: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        add_0 = paddle._C_ops.add(multiply_2, share_data__1)
        del assign_0, multiply_2, share_data__1

        return add_0
