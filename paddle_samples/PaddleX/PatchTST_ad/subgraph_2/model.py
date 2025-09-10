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
        parameter_49,
        parameter_50,
        parameter_51,
        parameter_52,
        parameter_53,
        parameter_54,
        parameter_55,
        parameter_56,
        parameter_57,
        parameter_58,
        parameter_59,
        parameter_60,
        parameter_61,
        parameter_62,
        parameter_63,
        parameter_64,
        parameter_65,
        parameter_66,
        parameter_67,
        parameter_68,
        parameter_69,
        parameter_70,
        parameter_71,
        parameter_72,
        parameter_73,
        parameter_74,
        parameter_75,
        parameter_76,
        parameter_77,
        parameter_78,
        parameter_79,
        parameter_80,
        parameter_81,
        parameter_82,
        parameter_83,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
    ):
        # pd_op.cast: (-1x96x2xf32) <- (-1x96x2xf32)
        cast_0 = paddle._C_ops.cast(data_5, paddle.float32)
        del data_5

        # pd_op.assign: (-1x96x2xf32) <- (-1x96x2xf32)
        assign_0 = cast_0

        # pd_op.transpose: (-1x2x96xf32) <- (-1x96x2xf32)
        transpose_1 = paddle._C_ops.transpose(cast_0, [0, 2, 1])
        del cast_0

        # pd_op.transpose: (-1x96x2xf32) <- (-1x2x96xf32)
        transpose_2 = paddle._C_ops.transpose(transpose_1, [0, 2, 1])
        del transpose_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.mean: (-1x1x2xf32) <- (-1x96x2xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(transpose_2, full_int_array_0, True)

        # pd_op.share_data_: (-1x1x2xf32) <- (-1x1x2xf32)
        share_data__0 = mean_0.detach()
        del mean_0

        # pd_op.mean: (-1x1x2xf32) <- (-1x96x2xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(transpose_2, full_int_array_0, True)

        # pd_op.subtract: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        subtract_0 = paddle._C_ops.subtract(transpose_2, mean_1)
        del mean_1

        # pd_op.pow: (-1x96x2xf32) <- (-1x96x2xf32)
        pow_0 = paddle._C_ops.pow(subtract_0, float("2"))
        del subtract_0

        # pd_op.sum: (-1x1x2xf32) <- (-1x96x2xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(pow_0, full_int_array_0, paddle.float32, True)
        del pow_0

        # pd_op.numel: (xi64) <- (-1x96x2xf32)
        numel_0 = paddle._C_ops.numel(transpose_2)

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
        share_data__1 = sqrt_0.detach()
        del sqrt_0

        # pd_op.subtract: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(transpose_2, share_data__0)
        del transpose_2

        # pd_op.divide: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        divide_2 = paddle._C_ops.divide(subtract_1, share_data__1)
        del subtract_1

        # pd_op.transpose: (-1x2x96xf32) <- (-1x96x2xf32)
        transpose_3 = paddle._C_ops.transpose(divide_2, [0, 2, 1])
        del divide_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [3, 4]

        # pd_op.unsqueeze: (-1x2x96x1x1xf32) <- (-1x2x96xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_3, full_int_array_1)
        del transpose_3

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_2 = [0, 0, 0, 0, 0, 8]

        # pd_op.pad3d: (-1x2x104x1x1xf32) <- (-1x2x96x1x1xf32, 6xi64)
        pad3d_0 = paddle._C_ops.pad3d(
            unsqueeze_0, full_int_array_2, "replicate", float("0"), "NCDHW"
        )
        del full_int_array_2, unsqueeze_0

        # pd_op.squeeze: (-1x2x104xf32) <- (-1x2x104x1x1xf32, 2xi64)
        squeeze_0 = paddle._C_ops.squeeze(pad3d_0, full_int_array_1)
        del full_int_array_1, pad3d_0

        # pd_op.shape64: (3xi64) <- (-1x2x104xf32)
        shape64_0 = paddle._C_ops.shape64(squeeze_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [0, 0, 0]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [slice_0, full_1, full_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_1 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_4, stack_0, [-1, 1, 1], []
        )
        del full_int_array_4, stack_0

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("24"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [0, 0, 8]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_1 = [slice_0, full_1, full_3]
        del full_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_2 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_5, stack_1, [-1, 1, 1], []
        )
        del full_int_array_5, stack_1

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [0, 0, 16]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_2 = [slice_0, full_1, full_4]
        del full_4

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_3 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_6, stack_2, [-1, 1, 1], []
        )
        del full_int_array_6, stack_2

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("40"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [0, 0, 24]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [slice_0, full_1, full_5]
        del full_5

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_4 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_7, stack_3, [-1, 1, 1], []
        )
        del full_int_array_7, stack_3

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("48"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [0, 0, 32]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_4 = [slice_0, full_1, full_6]
        del full_6

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_5 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_8, stack_4, [-1, 1, 1], []
        )
        del full_int_array_8, stack_4

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("56"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_9 = [0, 0, 40]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [slice_0, full_1, full_7]
        del full_7

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_6 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_9, stack_5, [-1, 1, 1], []
        )
        del full_int_array_9, stack_5

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("64"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_10 = [0, 0, 48]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_6 = [slice_0, full_1, full_8]
        del full_8

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_7 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_10, stack_6, [-1, 1, 1], []
        )
        del full_int_array_10, stack_6

        # pd_op.full: (xi64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("72"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [0, 0, 56]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_7 = [slice_0, full_1, full_9]
        del full_9

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_8 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_11, stack_7, [-1, 1, 1], []
        )
        del full_int_array_11, stack_7

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("80"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_12 = [0, 0, 64]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_8 = [slice_0, full_1, full_10]
        del full_10

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_9 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_12, stack_8, [-1, 1, 1], []
        )
        del full_int_array_12, stack_8

        # pd_op.full: (xi64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("88"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_13 = [0, 0, 72]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [slice_0, full_1, full_11]
        del full_11

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_10 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_13, stack_9, [-1, 1, 1], []
        )
        del full_int_array_13, stack_9

        # pd_op.full: (xi64) <- ()
        full_12 = paddle._C_ops.full(
            [], float("96"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_14 = [0, 0, 80]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_10 = [slice_0, full_1, full_12]
        del full_12

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_11 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_14, stack_10, [-1, 1, 1], []
        )
        del full_int_array_14, stack_10

        # pd_op.full: (xi64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("104"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_15 = [0, 0, 88]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [slice_0, full_1, full_13]
        del full_1, full_13, slice_0

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.slice: (-1x2x16xf32) <- (-1x2x104xf32, 3xi64, 3xi64)
        slice_12 = paddle._C_ops.slice(
            squeeze_0, [0, 1, 2], full_int_array_15, stack_11, [-1, 1, 1], []
        )
        del full_int_array_15, squeeze_0, stack_11

        # builtin.combine: ([-1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32]) <- (-1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32)
        combine_12 = [
            slice_1,
            slice_2,
            slice_3,
            slice_4,
            slice_5,
            slice_6,
            slice_7,
            slice_8,
            slice_9,
            slice_10,
            slice_11,
            slice_12,
        ]
        del (
            slice_1,
            slice_10,
            slice_11,
            slice_12,
            slice_2,
            slice_3,
            slice_4,
            slice_5,
            slice_6,
            slice_7,
            slice_8,
            slice_9,
        )

        # pd_op.stack: (-1x2x16x12xf32) <- ([-1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32, -1x2x16xf32])
        stack_12 = paddle._C_ops.stack(combine_12, -1)
        del combine_12

        # pd_op.shape64: (4xi64) <- (-1x2x16x12xf32)
        shape64_1 = paddle._C_ops.shape64(stack_12)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_1

        # pd_op.transpose: (-1x2x12x16xf32) <- (-1x2x16x12xf32)
        transpose_4 = paddle._C_ops.transpose(stack_12, [0, 1, 3, 2])
        del stack_12

        # pd_op.matmul: (-1x2x12x16xf32) <- (-1x2x12x16xf32, 16x16xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_4, parameter_83, False, False)
        del parameter_83

        # pd_op.add: (-1x2x12x16xf32) <- (-1x2x12x16xf32, 16xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_82)
        del parameter_82

        # pd_op.shape64: (4xi64) <- (-1x2x12x16xf32)
        shape64_2 = paddle._C_ops.shape64(add_0)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_2

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_14, full_14, float("0"), True)
        del full_14, slice_14

        # pd_op.full: (xi64) <- ()
        full_15 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_13 = [scale_1, full_15, full_2]
        del full_15, scale_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.reshape: (-1x12x16xf32) <- (-1x2x12x16xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(add_0, stack_13)
        del stack_13

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 12x16xf32)
        add_1 = paddle._C_ops.add(reshape_0, data_0)
        del data_0

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_16

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_16

        # pd_op.dropout: (-1x12x16xf32, -1x12x16xui8) <- (-1x12x16xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_1, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_1

        # pd_op.shape64: (3xi64) <- (-1x12x16xf32)
        shape64_3 = paddle._C_ops.shape64(dropout_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_3

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_80)
        del parameter_80

        # pd_op.full: (xi64) <- ()
        full_17 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_18 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_14 = [slice_15, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_2, stack_14)
        del stack_14

        # pd_op.transpose: (-1x4x-1x4xf32) <- (-1x-1x4x4xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_78)
        del parameter_78

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_15 = [slice_15, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_3, stack_15)
        del stack_15

        # pd_op.transpose: (-1x4x4x-1xf32) <- (-1x-1x4x4xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_2, [0, 2, 3, 1])
        del reshape_2

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_3 = paddle._C_ops.matmul(dropout_0, parameter_77, False, False)
        del parameter_77

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_76)
        del parameter_76

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_16 = [slice_15, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(add_4, stack_16)
        del stack_16

        # pd_op.transpose: (-1x4x-1x4xf32) <- (-1x-1x4x4xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_3, [0, 2, 1, 3])
        del reshape_3

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x4xf32, -1x4x4x-1xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_5, transpose_6, False, False)

        # pd_op.multiply: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32, xf32)
        multiply_0 = paddle._C_ops.multiply(matmul_4, data_1)
        del data_1

        # pd_op.softmax: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(multiply_0, -1)

        # pd_op.matmul: (-1x4x-1x4xf32) <- (-1x4x-1x-1xf32, -1x4x-1x4xf32)
        matmul_5 = paddle._C_ops.matmul(softmax_0, transpose_7, False, False)

        # pd_op.transpose: (-1x-1x4x4xf32) <- (-1x4x-1x4xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_5, [0, 2, 1, 3])
        del matmul_5

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_17 = [slice_15, full_17, full_2]
        del slice_15

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_17, 0)
        del combine_17

        # pd_op.reshape: (-1x-1x16xf32) <- (-1x-1x4x4xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_8, stack_17)
        del stack_17

        # pd_op.matmul: (-1x-1x16xf32) <- (-1x-1x16xf32, 16x16xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_4, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (-1x-1x16xf32) <- (-1x-1x16xf32, 16xf32)
        add_5 = paddle._C_ops.add(matmul_6, parameter_74)
        del parameter_74

        # pd_op.dropout: (-1x-1x16xf32, -1x-1x16xui8) <- (-1x-1x16xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_5, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_5

        # pd_op.dropout: (-1x-1x16xf32, -1x-1x16xui8) <- (-1x-1x16xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_2, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_2

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, -1x-1x16xf32)
        add_6 = paddle._C_ops.add(dropout_0, dropout_4)

        # pd_op.transpose: (-1x16x12xf32) <- (-1x12x16xf32)
        transpose_9 = paddle._C_ops.transpose(add_6, [0, 2, 1])
        del add_6

        # pd_op.batch_norm_: (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                transpose_9,
                parameter_73,
                parameter_72,
                parameter_71,
                parameter_70,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_70, parameter_71, parameter_72, parameter_73

        # pd_op.transpose: (-1x12x16xf32) <- (-1x16x12xf32)
        transpose_10 = paddle._C_ops.transpose(batch_norm__0, [0, 2, 1])
        del batch_norm__0

        # pd_op.matmul: (-1x12x128xf32) <- (-1x12x16xf32, 16x128xf32)
        matmul_7 = paddle._C_ops.matmul(transpose_10, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (-1x12x128xf32) <- (-1x12x128xf32, 128xf32)
        add_7 = paddle._C_ops.add(matmul_7, parameter_68)
        del parameter_68

        # pd_op.gelu: (-1x12x128xf32) <- (-1x12x128xf32)
        gelu_0 = paddle._C_ops.gelu(add_7, False)

        # pd_op.dropout: (-1x12x128xf32, -1x12x128xui8) <- (-1x12x128xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_0, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_0

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x128xf32, 128x16xf32)
        matmul_8 = paddle._C_ops.matmul(dropout_6, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_8 = paddle._C_ops.add(matmul_8, parameter_66)
        del parameter_66

        # pd_op.dropout: (-1x12x16xf32, -1x12x16xui8) <- (-1x12x16xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_8, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_8

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, -1x12x16xf32)
        add_9 = paddle._C_ops.add(transpose_10, dropout_8)

        # pd_op.transpose: (-1x16x12xf32) <- (-1x12x16xf32)
        transpose_11 = paddle._C_ops.transpose(add_9, [0, 2, 1])
        del add_9

        # pd_op.batch_norm_: (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                transpose_11,
                parameter_65,
                parameter_64,
                parameter_63,
                parameter_62,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_62, parameter_63, parameter_64, parameter_65

        # pd_op.transpose: (-1x12x16xf32) <- (-1x16x12xf32)
        transpose_12 = paddle._C_ops.transpose(batch_norm__6, [0, 2, 1])
        del batch_norm__6

        # pd_op.shape64: (3xi64) <- (-1x12x16xf32)
        shape64_4 = paddle._C_ops.shape64(transpose_12)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_4

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_9 = paddle._C_ops.matmul(transpose_12, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_10 = paddle._C_ops.add(matmul_9, parameter_60)
        del parameter_60

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_18 = [slice_16, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_18 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_10, stack_18)
        del stack_18

        # pd_op.transpose: (-1x4x-1x4xf32) <- (-1x-1x4x4xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_10 = paddle._C_ops.matmul(transpose_12, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_11 = paddle._C_ops.add(matmul_10, parameter_58)
        del parameter_58

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_19 = [slice_16, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_19 = paddle._C_ops.stack(combine_19, 0)
        del combine_19

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_11, stack_19)
        del stack_19

        # pd_op.transpose: (-1x4x4x-1xf32) <- (-1x-1x4x4xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_6, [0, 2, 3, 1])
        del reshape_6

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_11 = paddle._C_ops.matmul(transpose_12, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_12 = paddle._C_ops.add(matmul_11, parameter_56)
        del parameter_56

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_20 = [slice_16, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_20 = paddle._C_ops.stack(combine_20, 0)
        del combine_20

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(add_12, stack_20)
        del stack_20

        # pd_op.transpose: (-1x4x-1x4xf32) <- (-1x-1x4x4xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_7, [0, 2, 1, 3])
        del reshape_7

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x4xf32, -1x4x4x-1xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_13, transpose_14, False, False)

        # pd_op.multiply: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32, xf32)
        multiply_1 = paddle._C_ops.multiply(matmul_12, data_2)
        del data_2

        # pd_op.add: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32, -1x4x-1x-1xf32)
        add_13 = paddle._C_ops.add(multiply_1, multiply_0)

        # pd_op.softmax: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(add_13, -1)

        # pd_op.matmul: (-1x4x-1x4xf32) <- (-1x4x-1x-1xf32, -1x4x-1x4xf32)
        matmul_13 = paddle._C_ops.matmul(softmax_1, transpose_15, False, False)

        # pd_op.transpose: (-1x-1x4x4xf32) <- (-1x4x-1x4xf32)
        transpose_16 = paddle._C_ops.transpose(matmul_13, [0, 2, 1, 3])
        del matmul_13

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_21 = [slice_16, full_17, full_2]
        del slice_16

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_21 = paddle._C_ops.stack(combine_21, 0)
        del combine_21

        # pd_op.reshape: (-1x-1x16xf32) <- (-1x-1x4x4xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_16, stack_21)
        del stack_21

        # pd_op.matmul: (-1x-1x16xf32) <- (-1x-1x16xf32, 16x16xf32)
        matmul_14 = paddle._C_ops.matmul(reshape_8, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (-1x-1x16xf32) <- (-1x-1x16xf32, 16xf32)
        add_14 = paddle._C_ops.add(matmul_14, parameter_54)
        del parameter_54

        # pd_op.dropout: (-1x-1x16xf32, -1x-1x16xui8) <- (-1x-1x16xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_14, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_14

        # pd_op.dropout: (-1x-1x16xf32, -1x-1x16xui8) <- (-1x-1x16xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_10, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_10

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, -1x-1x16xf32)
        add_15 = paddle._C_ops.add(transpose_12, dropout_12)

        # pd_op.transpose: (-1x16x12xf32) <- (-1x12x16xf32)
        transpose_17 = paddle._C_ops.transpose(add_15, [0, 2, 1])
        del add_15

        # pd_op.batch_norm_: (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                transpose_17,
                parameter_53,
                parameter_52,
                parameter_51,
                parameter_50,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_50, parameter_51, parameter_52, parameter_53

        # pd_op.transpose: (-1x12x16xf32) <- (-1x16x12xf32)
        transpose_18 = paddle._C_ops.transpose(batch_norm__12, [0, 2, 1])
        del batch_norm__12

        # pd_op.matmul: (-1x12x128xf32) <- (-1x12x16xf32, 16x128xf32)
        matmul_15 = paddle._C_ops.matmul(transpose_18, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (-1x12x128xf32) <- (-1x12x128xf32, 128xf32)
        add_16 = paddle._C_ops.add(matmul_15, parameter_48)
        del parameter_48

        # pd_op.gelu: (-1x12x128xf32) <- (-1x12x128xf32)
        gelu_1 = paddle._C_ops.gelu(add_16, False)

        # pd_op.dropout: (-1x12x128xf32, -1x12x128xui8) <- (-1x12x128xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_1, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_1

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x128xf32, 128x16xf32)
        matmul_16 = paddle._C_ops.matmul(dropout_14, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_17 = paddle._C_ops.add(matmul_16, parameter_46)
        del parameter_46

        # pd_op.dropout: (-1x12x16xf32, -1x12x16xui8) <- (-1x12x16xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_17, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_17

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, -1x12x16xf32)
        add_18 = paddle._C_ops.add(transpose_18, dropout_16)

        # pd_op.transpose: (-1x16x12xf32) <- (-1x12x16xf32)
        transpose_19 = paddle._C_ops.transpose(add_18, [0, 2, 1])
        del add_18

        # pd_op.batch_norm_: (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                transpose_19,
                parameter_45,
                parameter_44,
                parameter_43,
                parameter_42,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_42, parameter_43, parameter_44, parameter_45

        # pd_op.transpose: (-1x12x16xf32) <- (-1x16x12xf32)
        transpose_20 = paddle._C_ops.transpose(batch_norm__18, [0, 2, 1])
        del batch_norm__18

        # pd_op.shape64: (3xi64) <- (-1x12x16xf32)
        shape64_5 = paddle._C_ops.shape64(transpose_20)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_5

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_17 = paddle._C_ops.matmul(transpose_20, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_19 = paddle._C_ops.add(matmul_17, parameter_40)
        del parameter_40

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_22 = [slice_17, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_22 = paddle._C_ops.stack(combine_22, 0)
        del combine_22

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_19, stack_22)
        del stack_22

        # pd_op.transpose: (-1x4x-1x4xf32) <- (-1x-1x4x4xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_18 = paddle._C_ops.matmul(transpose_20, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_20 = paddle._C_ops.add(matmul_18, parameter_38)
        del parameter_38

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_23 = [slice_17, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_23 = paddle._C_ops.stack(combine_23, 0)
        del combine_23

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_20, stack_23)
        del stack_23

        # pd_op.transpose: (-1x4x4x-1xf32) <- (-1x-1x4x4xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_10, [0, 2, 3, 1])
        del reshape_10

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_19 = paddle._C_ops.matmul(transpose_20, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_21 = paddle._C_ops.add(matmul_19, parameter_36)
        del parameter_36

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_24 = [slice_17, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_24 = paddle._C_ops.stack(combine_24, 0)
        del combine_24

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(add_21, stack_24)
        del stack_24

        # pd_op.transpose: (-1x4x-1x4xf32) <- (-1x-1x4x4xf32)
        transpose_23 = paddle._C_ops.transpose(reshape_11, [0, 2, 1, 3])
        del reshape_11

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x4xf32, -1x4x4x-1xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_21, transpose_22, False, False)

        # pd_op.multiply: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32, xf32)
        multiply_2 = paddle._C_ops.multiply(matmul_20, data_3)
        del data_3

        # pd_op.add: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32, -1x4x-1x-1xf32)
        add_22 = paddle._C_ops.add(multiply_2, add_13)

        # pd_op.softmax: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(add_22, -1)

        # pd_op.matmul: (-1x4x-1x4xf32) <- (-1x4x-1x-1xf32, -1x4x-1x4xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_2, transpose_23, False, False)

        # pd_op.transpose: (-1x-1x4x4xf32) <- (-1x4x-1x4xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_25 = [slice_17, full_17, full_2]
        del slice_17

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_25 = paddle._C_ops.stack(combine_25, 0)
        del combine_25

        # pd_op.reshape: (-1x-1x16xf32) <- (-1x-1x4x4xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_24, stack_25)
        del stack_25

        # pd_op.matmul: (-1x-1x16xf32) <- (-1x-1x16xf32, 16x16xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_12, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (-1x-1x16xf32) <- (-1x-1x16xf32, 16xf32)
        add_23 = paddle._C_ops.add(matmul_22, parameter_34)
        del parameter_34

        # pd_op.dropout: (-1x-1x16xf32, -1x-1x16xui8) <- (-1x-1x16xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_23, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_23

        # pd_op.dropout: (-1x-1x16xf32, -1x-1x16xui8) <- (-1x-1x16xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_18, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_18

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, -1x-1x16xf32)
        add_24 = paddle._C_ops.add(transpose_20, dropout_20)

        # pd_op.transpose: (-1x16x12xf32) <- (-1x12x16xf32)
        transpose_25 = paddle._C_ops.transpose(add_24, [0, 2, 1])
        del add_24

        # pd_op.batch_norm_: (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                transpose_25,
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.transpose: (-1x12x16xf32) <- (-1x16x12xf32)
        transpose_26 = paddle._C_ops.transpose(batch_norm__24, [0, 2, 1])
        del batch_norm__24

        # pd_op.matmul: (-1x12x128xf32) <- (-1x12x16xf32, 16x128xf32)
        matmul_23 = paddle._C_ops.matmul(transpose_26, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (-1x12x128xf32) <- (-1x12x128xf32, 128xf32)
        add_25 = paddle._C_ops.add(matmul_23, parameter_28)
        del parameter_28

        # pd_op.gelu: (-1x12x128xf32) <- (-1x12x128xf32)
        gelu_2 = paddle._C_ops.gelu(add_25, False)

        # pd_op.dropout: (-1x12x128xf32, -1x12x128xui8) <- (-1x12x128xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_2, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_2

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x128xf32, 128x16xf32)
        matmul_24 = paddle._C_ops.matmul(dropout_22, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_26 = paddle._C_ops.add(matmul_24, parameter_26)
        del parameter_26

        # pd_op.dropout: (-1x12x16xf32, -1x12x16xui8) <- (-1x12x16xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_26, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_26

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, -1x12x16xf32)
        add_27 = paddle._C_ops.add(transpose_26, dropout_24)

        # pd_op.transpose: (-1x16x12xf32) <- (-1x12x16xf32)
        transpose_27 = paddle._C_ops.transpose(add_27, [0, 2, 1])
        del add_27

        # pd_op.batch_norm_: (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                transpose_27,
                parameter_25,
                parameter_24,
                parameter_23,
                parameter_22,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_22, parameter_23, parameter_24, parameter_25

        # pd_op.transpose: (-1x12x16xf32) <- (-1x16x12xf32)
        transpose_28 = paddle._C_ops.transpose(batch_norm__30, [0, 2, 1])
        del batch_norm__30

        # pd_op.shape64: (3xi64) <- (-1x12x16xf32)
        shape64_6 = paddle._C_ops.shape64(transpose_28)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_6

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_25 = paddle._C_ops.matmul(transpose_28, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_28 = paddle._C_ops.add(matmul_25, parameter_20)
        del parameter_20

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_26 = [slice_18, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_26 = paddle._C_ops.stack(combine_26, 0)
        del combine_26

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_28, stack_26)
        del stack_26

        # pd_op.transpose: (-1x4x-1x4xf32) <- (-1x-1x4x4xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_26 = paddle._C_ops.matmul(transpose_28, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_29 = paddle._C_ops.add(matmul_26, parameter_18)
        del parameter_18

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_27 = [slice_18, full_17, full_18, full_18]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_27 = paddle._C_ops.stack(combine_27, 0)
        del combine_27

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_29, stack_27)
        del stack_27

        # pd_op.transpose: (-1x4x4x-1xf32) <- (-1x-1x4x4xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_14, [0, 2, 3, 1])
        del reshape_14

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x16xf32, 16x16xf32)
        matmul_27 = paddle._C_ops.matmul(transpose_28, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_30 = paddle._C_ops.add(matmul_27, parameter_16)
        del parameter_16

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_28 = [slice_18, full_17, full_18, full_18]
        del full_18

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_28 = paddle._C_ops.stack(combine_28, 0)
        del combine_28

        # pd_op.reshape: (-1x-1x4x4xf32) <- (-1x12x16xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(add_30, stack_28)
        del stack_28

        # pd_op.transpose: (-1x4x-1x4xf32) <- (-1x-1x4x4xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_15, [0, 2, 1, 3])
        del reshape_15

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x4xf32, -1x4x4x-1xf32)
        matmul_28 = paddle._C_ops.matmul(transpose_29, transpose_30, False, False)

        # pd_op.multiply: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32, xf32)
        multiply_3 = paddle._C_ops.multiply(matmul_28, data_4)
        del data_4

        # pd_op.add: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32, -1x4x-1x-1xf32)
        add_31 = paddle._C_ops.add(multiply_3, add_22)

        # pd_op.softmax: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(add_31, -1)
        del add_31

        # pd_op.matmul: (-1x4x-1x4xf32) <- (-1x4x-1x-1xf32, -1x4x-1x4xf32)
        matmul_29 = paddle._C_ops.matmul(softmax_3, transpose_31, False, False)

        # pd_op.transpose: (-1x-1x4x4xf32) <- (-1x4x-1x4xf32)
        transpose_32 = paddle._C_ops.transpose(matmul_29, [0, 2, 1, 3])
        del matmul_29

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_29 = [slice_18, full_17, full_2]
        del full_17, full_2, slice_18

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_29 = paddle._C_ops.stack(combine_29, 0)
        del combine_29

        # pd_op.reshape: (-1x-1x16xf32) <- (-1x-1x4x4xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_32, stack_29)
        del stack_29

        # pd_op.matmul: (-1x-1x16xf32) <- (-1x-1x16xf32, 16x16xf32)
        matmul_30 = paddle._C_ops.matmul(reshape_16, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (-1x-1x16xf32) <- (-1x-1x16xf32, 16xf32)
        add_32 = paddle._C_ops.add(matmul_30, parameter_14)
        del parameter_14

        # pd_op.dropout: (-1x-1x16xf32, -1x-1x16xui8) <- (-1x-1x16xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_32, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_32

        # pd_op.dropout: (-1x-1x16xf32, -1x-1x16xui8) <- (-1x-1x16xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_26, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_26

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, -1x-1x16xf32)
        add_33 = paddle._C_ops.add(transpose_28, dropout_28)

        # pd_op.transpose: (-1x16x12xf32) <- (-1x12x16xf32)
        transpose_33 = paddle._C_ops.transpose(add_33, [0, 2, 1])
        del add_33

        # pd_op.batch_norm_: (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                transpose_33,
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_10, parameter_11, parameter_12, parameter_13

        # pd_op.transpose: (-1x12x16xf32) <- (-1x16x12xf32)
        transpose_34 = paddle._C_ops.transpose(batch_norm__36, [0, 2, 1])
        del batch_norm__36

        # pd_op.matmul: (-1x12x128xf32) <- (-1x12x16xf32, 16x128xf32)
        matmul_31 = paddle._C_ops.matmul(transpose_34, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (-1x12x128xf32) <- (-1x12x128xf32, 128xf32)
        add_34 = paddle._C_ops.add(matmul_31, parameter_8)
        del parameter_8

        # pd_op.gelu: (-1x12x128xf32) <- (-1x12x128xf32)
        gelu_3 = paddle._C_ops.gelu(add_34, False)

        # pd_op.dropout: (-1x12x128xf32, -1x12x128xui8) <- (-1x12x128xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_3, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_3

        # pd_op.matmul: (-1x12x16xf32) <- (-1x12x128xf32, 128x16xf32)
        matmul_32 = paddle._C_ops.matmul(dropout_30, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, 16xf32)
        add_35 = paddle._C_ops.add(matmul_32, parameter_6)
        del parameter_6

        # pd_op.dropout: (-1x12x16xf32, -1x12x16xui8) <- (-1x12x16xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_35, None, full_16, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_35

        # pd_op.add: (-1x12x16xf32) <- (-1x12x16xf32, -1x12x16xf32)
        add_36 = paddle._C_ops.add(transpose_34, dropout_32)

        # pd_op.transpose: (-1x16x12xf32) <- (-1x12x16xf32)
        transpose_35 = paddle._C_ops.transpose(add_36, [0, 2, 1])
        del add_36

        # pd_op.batch_norm_: (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x12xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                transpose_35,
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.transpose: (-1x12x16xf32) <- (-1x16x12xf32)
        transpose_36 = paddle._C_ops.transpose(batch_norm__42, [0, 2, 1])
        del batch_norm__42

        # pd_op.shape64: (3xi64) <- (-1x12x16xf32)
        shape64_7 = paddle._C_ops.shape64(transpose_36)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del full_int_array_0, full_int_array_3, shape64_7

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [-1, 2, 12, 16]

        # pd_op.reshape: (-1x2x12x16xf32) <- (-1x12x16xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_36, full_int_array_16)
        del full_int_array_16

        # pd_op.transpose: (-1x2x16x12xf32) <- (-1x2x12x16xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_17, [0, 1, 3, 2])
        del reshape_17

        # pd_op.flatten: (-1x2x192xf32) <- (-1x2x16x12xf32)
        flatten_0 = paddle._C_ops.flatten(transpose_37, 2, 3)

        # pd_op.matmul: (-1x2x96xf32) <- (-1x2x192xf32, 192x96xf32)
        matmul_33 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (-1x2x96xf32) <- (-1x2x96xf32, 96xf32)
        add_37 = paddle._C_ops.add(matmul_33, parameter_0)
        del parameter_0

        # pd_op.transpose: (-1x96x2xf32) <- (-1x2x96xf32)
        transpose_38 = paddle._C_ops.transpose(add_37, [0, 2, 1])
        del add_37

        # pd_op.multiply: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        multiply_4 = paddle._C_ops.multiply(transpose_38, share_data__1)

        # pd_op.add: (-1x96x2xf32) <- (-1x96x2xf32, -1x1x2xf32)
        add_38 = paddle._C_ops.add(multiply_4, share_data__0)

        # pd_op.transpose: (-1x2x96xf32) <- (-1x96x2xf32)
        transpose_39 = paddle._C_ops.transpose(add_38, [0, 2, 1])
        del add_38

        # pd_op.transpose: (-1x96x2xf32) <- (-1x2x96xf32)
        transpose_0 = paddle._C_ops.transpose(transpose_39, [0, 2, 1])
        del (
            add_0,
            add_10,
            add_11,
            add_12,
            add_13,
            add_16,
            add_19,
            add_2,
            add_20,
            add_21,
            add_22,
            add_25,
            add_28,
            add_29,
            add_3,
            add_30,
            add_34,
            add_4,
            add_7,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            batch_norm__1,
            batch_norm__10,
            batch_norm__11,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__19,
            batch_norm__2,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__3,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__5,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            dropout_0,
            dropout_1,
            dropout_11,
            dropout_12,
            dropout_13,
            dropout_14,
            dropout_15,
            dropout_16,
            dropout_17,
            dropout_19,
            dropout_20,
            dropout_21,
            dropout_22,
            dropout_23,
            dropout_24,
            dropout_25,
            dropout_27,
            dropout_28,
            dropout_29,
            dropout_3,
            dropout_30,
            dropout_31,
            dropout_32,
            dropout_33,
            dropout_4,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_8,
            dropout_9,
            flatten_0,
            full_16,
            matmul_0,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_19,
            matmul_2,
            matmul_20,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_28,
            matmul_3,
            matmul_30,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_4,
            matmul_6,
            matmul_7,
            matmul_8,
            matmul_9,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            multiply_4,
            reshape_0,
            reshape_12,
            reshape_16,
            reshape_4,
            reshape_8,
            share_data__0,
            share_data__1,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            transpose_10,
            transpose_11,
            transpose_12,
            transpose_13,
            transpose_14,
            transpose_15,
            transpose_16,
            transpose_17,
            transpose_18,
            transpose_19,
            transpose_20,
            transpose_21,
            transpose_22,
            transpose_23,
            transpose_24,
            transpose_25,
            transpose_26,
            transpose_27,
            transpose_28,
            transpose_29,
            transpose_30,
            transpose_31,
            transpose_32,
            transpose_33,
            transpose_34,
            transpose_35,
            transpose_36,
            transpose_37,
            transpose_38,
            transpose_39,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
        )

        return transpose_0
