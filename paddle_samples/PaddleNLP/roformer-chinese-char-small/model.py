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
        parameter_84,
        parameter_85,
        parameter_86,
        parameter_87,
        parameter_88,
        parameter_89,
        parameter_90,
        parameter_91,
        parameter_92,
        parameter_93,
        parameter_94,
        parameter_95,
        parameter_96,
        parameter_97,
        parameter_98,
        parameter_99,
        parameter_100,
        parameter_101,
        parameter_102,
        parameter_103,
        parameter_104,
        parameter_105,
        parameter_106,
        parameter_107,
        parameter_108,
        parameter_109,
        parameter_110,
        parameter_111,
        parameter_112,
        parameter_113,
        data_0,
        data_1,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (1x11xb) <- (1x11xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_0)
        del full_0

        # pd_op.cast: (1x11xf32) <- (1x11xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x11xf32) <- (1x11xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_1, float("0"), True)
        del cast_0, full_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2]

        # pd_op.unsqueeze: (1x1x1x11xf32) <- (1x11xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(scale_0, full_int_array_0)
        del full_int_array_0, scale_0

        # pd_op.embedding: (1x11x384xf32) <- (1x11xi64, 12000x384xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_113, -1, False)
        del data_0, parameter_113

        # pd_op.embedding: (1x11x384xf32) <- (1x11xi64, 2x384xf32)
        embedding_1 = paddle._C_ops.embedding(data_1, parameter_112, -1, False)
        del data_1, parameter_112

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_111, parameter_110, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_110, parameter_111

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_2

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_109, False, False)
        del parameter_109

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_108)
        del parameter_108

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 6, 64]

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_107, False, False)
        del parameter_107

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_106)
        del parameter_106

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_104)
        del parameter_104

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_2, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_3, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_29 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [11]

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            parameter_11, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_11

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_31 = slice_0

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            parameter_10, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_10

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_32 = slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_55 = full_int_array_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_60 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_61 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_62 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_63 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_64 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_65 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_66 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_67 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_68 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_69 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_70 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_71 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_72 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_73 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_74 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_75 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_76 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_77 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_78 = full_int_array_5

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_79 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_80 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_81 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_82 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_83 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_84 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_85 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_86 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_87 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_88 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_89 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_90 = full_int_array_6

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_0 = paddle._C_ops.multiply(strided_slice_0, slice_1)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_1 = paddle._C_ops.multiply(strided_slice_1, slice_0)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_0 = paddle._C_ops.subtract(multiply_0, multiply_1)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_2 = paddle._C_ops.multiply(strided_slice_0, slice_0)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_3 = paddle._C_ops.multiply(strided_slice_1, slice_1)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_4 = paddle._C_ops.add(multiply_2, multiply_3)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_0 = [subtract_0, add_4]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_0 = paddle._C_ops.flatten(stack_0, 3, 4)

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_4 = paddle._C_ops.multiply(strided_slice_2, slice_1)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_5 = paddle._C_ops.multiply(strided_slice_3, slice_0)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_1 = paddle._C_ops.subtract(multiply_4, multiply_5)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_6 = paddle._C_ops.multiply(strided_slice_2, slice_0)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_7 = paddle._C_ops.multiply(strided_slice_3, slice_1)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_5 = paddle._C_ops.add(multiply_6, multiply_7)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_1 = [subtract_1, add_5]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_1 = paddle._C_ops.flatten(stack_1, 3, 4)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_91 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_92 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_93 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_94 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_95 = full_3

        # pd_op.scale: (1x6x11x64xf32) <- (1x6x11x64xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(flatten_0, full_3, float("0"), True)
        del flatten_0

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_3 = paddle._C_ops.matmul(scale_1, flatten_1, False, True)

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_6 = paddle._C_ops.add(matmul_3, unsqueeze_0)

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_0 = paddle._C_ops.softmax(add_6, -1)
        del add_6

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [0, 0, 384]

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_7)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_7 = paddle._C_ops.add(matmul_5, parameter_102)
        del parameter_102

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_7, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_7

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_8 = paddle._C_ops.add(dropout_0, dropout_4)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_97, parameter_96, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_96, parameter_97

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_101, False, False)
        del parameter_101

        # pd_op.add: (1x11x1536xf32) <- (1x11x1536xf32, 1536xf32)
        add_9 = paddle._C_ops.add(matmul_6, parameter_100)
        del parameter_100

        # pd_op.gelu: (1x11x1536xf32) <- (1x11x1536xf32)
        gelu_0 = paddle._C_ops.gelu(add_9, False)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_10 = paddle._C_ops.add(matmul_7, parameter_98)
        del parameter_98

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_10, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_10

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_11 = paddle._C_ops.add(layer_norm_3, dropout_6)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_95, parameter_94, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_94, parameter_95

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_12 = paddle._C_ops.add(matmul_8, parameter_92)
        del parameter_92

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_12, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_13 = paddle._C_ops.add(matmul_9, parameter_90)
        del parameter_90

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_14 = paddle._C_ops.add(matmul_10, parameter_88)
        del parameter_88

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_13, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_14, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            parameter_9, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_9

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_96 = slice_2

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            parameter_8, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_8

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_97 = slice_3

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_8 = paddle._C_ops.multiply(strided_slice_4, slice_3)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_9 = paddle._C_ops.multiply(strided_slice_5, slice_2)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_2 = paddle._C_ops.subtract(multiply_8, multiply_9)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_10 = paddle._C_ops.multiply(strided_slice_4, slice_2)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_11 = paddle._C_ops.multiply(strided_slice_5, slice_3)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_15 = paddle._C_ops.add(multiply_10, multiply_11)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_2 = [subtract_2, add_15]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_2 = paddle._C_ops.flatten(stack_2, 3, 4)

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_12 = paddle._C_ops.multiply(strided_slice_6, slice_3)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_13 = paddle._C_ops.multiply(strided_slice_7, slice_2)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_3 = paddle._C_ops.subtract(multiply_12, multiply_13)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_14 = paddle._C_ops.multiply(strided_slice_6, slice_2)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_15 = paddle._C_ops.multiply(strided_slice_7, slice_3)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_16 = paddle._C_ops.add(multiply_14, multiply_15)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_3 = [subtract_3, add_16]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_3 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_3 = paddle._C_ops.flatten(stack_3, 3, 4)

        # pd_op.scale: (1x6x11x64xf32) <- (1x6x11x64xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(flatten_2, full_3, float("0"), True)
        del flatten_2

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_11 = paddle._C_ops.matmul(scale_2, flatten_3, False, True)

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_17 = paddle._C_ops.add(matmul_11, unsqueeze_0)

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_1 = paddle._C_ops.softmax(add_17, -1)
        del add_17

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_7)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_18 = paddle._C_ops.add(matmul_13, parameter_86)
        del parameter_86

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_18, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_18

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_19 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_81, parameter_80, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_80, parameter_81

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (1x11x1536xf32) <- (1x11x1536xf32, 1536xf32)
        add_20 = paddle._C_ops.add(matmul_14, parameter_84)
        del parameter_84

        # pd_op.gelu: (1x11x1536xf32) <- (1x11x1536xf32)
        gelu_1 = paddle._C_ops.gelu(add_20, False)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_83, False, False)
        del parameter_83

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_21 = paddle._C_ops.add(matmul_15, parameter_82)
        del parameter_82

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_21, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_21

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_22 = paddle._C_ops.add(layer_norm_9, dropout_12)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_79, parameter_78, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_78, parameter_79

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_12, parameter_77, False, False)
        del parameter_77

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_23 = paddle._C_ops.add(matmul_16, parameter_76)
        del parameter_76

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_23, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_24 = paddle._C_ops.add(matmul_17, parameter_74)
        del parameter_74

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_25 = paddle._C_ops.add(matmul_18, parameter_72)
        del parameter_72

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_24, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_25, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            parameter_7, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_7

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_98 = slice_4

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            parameter_6, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_6

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_99 = slice_5

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_16 = paddle._C_ops.multiply(strided_slice_8, slice_5)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_17 = paddle._C_ops.multiply(strided_slice_9, slice_4)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_4 = paddle._C_ops.subtract(multiply_16, multiply_17)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_18 = paddle._C_ops.multiply(strided_slice_8, slice_4)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_19 = paddle._C_ops.multiply(strided_slice_9, slice_5)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_26 = paddle._C_ops.add(multiply_18, multiply_19)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_4 = [subtract_4, add_26]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_4 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_4 = paddle._C_ops.flatten(stack_4, 3, 4)

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_20 = paddle._C_ops.multiply(strided_slice_10, slice_5)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_21 = paddle._C_ops.multiply(strided_slice_11, slice_4)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_5 = paddle._C_ops.subtract(multiply_20, multiply_21)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_22 = paddle._C_ops.multiply(strided_slice_10, slice_4)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_23 = paddle._C_ops.multiply(strided_slice_11, slice_5)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_27 = paddle._C_ops.add(multiply_22, multiply_23)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_5 = [subtract_5, add_27]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_5 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_5 = paddle._C_ops.flatten(stack_5, 3, 4)

        # pd_op.scale: (1x6x11x64xf32) <- (1x6x11x64xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(flatten_4, full_3, float("0"), True)
        del flatten_4

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_19 = paddle._C_ops.matmul(scale_3, flatten_5, False, True)

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_28 = paddle._C_ops.add(matmul_19, unsqueeze_0)

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_2 = paddle._C_ops.softmax(add_28, -1)
        del add_28

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_7)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_29 = paddle._C_ops.add(matmul_21, parameter_70)
        del parameter_70

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_29, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_29

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_30 = paddle._C_ops.add(layer_norm_12, dropout_16)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_65, parameter_64, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_64, parameter_65

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_15, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (1x11x1536xf32) <- (1x11x1536xf32, 1536xf32)
        add_31 = paddle._C_ops.add(matmul_22, parameter_68)
        del parameter_68

        # pd_op.gelu: (1x11x1536xf32) <- (1x11x1536xf32)
        gelu_2 = paddle._C_ops.gelu(add_31, False)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_2, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_32 = paddle._C_ops.add(matmul_23, parameter_66)
        del parameter_66

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_32, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_32

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_33 = paddle._C_ops.add(layer_norm_15, dropout_18)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_63, parameter_62, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_62, parameter_63

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_18, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_34 = paddle._C_ops.add(matmul_24, parameter_60)
        del parameter_60

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_34, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_35 = paddle._C_ops.add(matmul_25, parameter_58)
        del parameter_58

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_36 = paddle._C_ops.add(matmul_26, parameter_56)
        del parameter_56

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_35, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_36, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            parameter_5, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_5

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_100 = slice_6

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            parameter_4, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_4

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_101 = slice_7

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_12 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_13 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_24 = paddle._C_ops.multiply(strided_slice_12, slice_7)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_25 = paddle._C_ops.multiply(strided_slice_13, slice_6)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_6 = paddle._C_ops.subtract(multiply_24, multiply_25)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_26 = paddle._C_ops.multiply(strided_slice_12, slice_6)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_27 = paddle._C_ops.multiply(strided_slice_13, slice_7)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_37 = paddle._C_ops.add(multiply_26, multiply_27)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_6 = [subtract_6, add_37]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_6 = paddle._C_ops.stack(combine_6, -1)
        del combine_6

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_6 = paddle._C_ops.flatten(stack_6, 3, 4)

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_14 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_15 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_28 = paddle._C_ops.multiply(strided_slice_14, slice_7)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_29 = paddle._C_ops.multiply(strided_slice_15, slice_6)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_7 = paddle._C_ops.subtract(multiply_28, multiply_29)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_30 = paddle._C_ops.multiply(strided_slice_14, slice_6)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_31 = paddle._C_ops.multiply(strided_slice_15, slice_7)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_38 = paddle._C_ops.add(multiply_30, multiply_31)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_7 = [subtract_7, add_38]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_7 = paddle._C_ops.stack(combine_7, -1)
        del combine_7

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_7 = paddle._C_ops.flatten(stack_7, 3, 4)

        # pd_op.scale: (1x6x11x64xf32) <- (1x6x11x64xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(flatten_6, full_3, float("0"), True)
        del flatten_6

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_27 = paddle._C_ops.matmul(scale_4, flatten_7, False, True)

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_39 = paddle._C_ops.add(matmul_27, unsqueeze_0)

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_3 = paddle._C_ops.softmax(add_39, -1)
        del add_39

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_7)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_15, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_40 = paddle._C_ops.add(matmul_29, parameter_54)
        del parameter_54

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_40, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_40

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_41 = paddle._C_ops.add(layer_norm_18, dropout_22)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_41, parameter_49, parameter_48, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_48, parameter_49

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_21, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (1x11x1536xf32) <- (1x11x1536xf32, 1536xf32)
        add_42 = paddle._C_ops.add(matmul_30, parameter_52)
        del parameter_52

        # pd_op.gelu: (1x11x1536xf32) <- (1x11x1536xf32)
        gelu_3 = paddle._C_ops.gelu(add_42, False)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_3, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_43 = paddle._C_ops.add(matmul_31, parameter_50)
        del parameter_50

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_43, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_43

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_44 = paddle._C_ops.add(layer_norm_21, dropout_24)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_44, parameter_47, parameter_46, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_46, parameter_47

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_24, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_45 = paddle._C_ops.add(matmul_32, parameter_44)
        del parameter_44

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_45, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_46 = paddle._C_ops.add(matmul_33, parameter_42)
        del parameter_42

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_47 = paddle._C_ops.add(matmul_34, parameter_40)
        del parameter_40

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_46, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_47, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            parameter_3, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_3

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_102 = slice_8

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            parameter_2, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_2

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_103 = slice_9

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_16 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_17 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_32 = paddle._C_ops.multiply(strided_slice_16, slice_9)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_33 = paddle._C_ops.multiply(strided_slice_17, slice_8)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_8 = paddle._C_ops.subtract(multiply_32, multiply_33)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_34 = paddle._C_ops.multiply(strided_slice_16, slice_8)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_35 = paddle._C_ops.multiply(strided_slice_17, slice_9)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_48 = paddle._C_ops.add(multiply_34, multiply_35)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_8 = [subtract_8, add_48]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_8 = paddle._C_ops.stack(combine_8, -1)
        del combine_8

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_8 = paddle._C_ops.flatten(stack_8, 3, 4)

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_18 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_19 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_36 = paddle._C_ops.multiply(strided_slice_18, slice_9)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_37 = paddle._C_ops.multiply(strided_slice_19, slice_8)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_9 = paddle._C_ops.subtract(multiply_36, multiply_37)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_38 = paddle._C_ops.multiply(strided_slice_18, slice_8)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_39 = paddle._C_ops.multiply(strided_slice_19, slice_9)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_49 = paddle._C_ops.add(multiply_38, multiply_39)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_9 = [subtract_9, add_49]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_9 = paddle._C_ops.stack(combine_9, -1)
        del combine_9

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_9 = paddle._C_ops.flatten(stack_9, 3, 4)

        # pd_op.scale: (1x6x11x64xf32) <- (1x6x11x64xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(flatten_8, full_3, float("0"), True)
        del flatten_8

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_35 = paddle._C_ops.matmul(scale_5, flatten_9, False, True)

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_50 = paddle._C_ops.add(matmul_35, unsqueeze_0)

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_4 = paddle._C_ops.softmax(add_50, -1)
        del add_50

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_36 = paddle._C_ops.matmul(dropout_26, transpose_18, False, False)

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_7)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_19, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_51 = paddle._C_ops.add(matmul_37, parameter_38)
        del parameter_38

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_51, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_51

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_52 = paddle._C_ops.add(layer_norm_24, dropout_28)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_33, parameter_32, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_32, parameter_33

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_27, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (1x11x1536xf32) <- (1x11x1536xf32, 1536xf32)
        add_53 = paddle._C_ops.add(matmul_38, parameter_36)
        del parameter_36

        # pd_op.gelu: (1x11x1536xf32) <- (1x11x1536xf32)
        gelu_4 = paddle._C_ops.gelu(add_53, False)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_39 = paddle._C_ops.matmul(gelu_4, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_54 = paddle._C_ops.add(matmul_39, parameter_34)
        del parameter_34

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_54, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_54

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_55 = paddle._C_ops.add(layer_norm_27, dropout_30)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_31, parameter_30, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_30, parameter_31

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_30, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_56 = paddle._C_ops.add(matmul_40, parameter_28)
        del parameter_28

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_56, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_57 = paddle._C_ops.add(matmul_41, parameter_26)
        del parameter_26

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_58 = paddle._C_ops.add(matmul_42, parameter_24)
        del parameter_24

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_57, full_int_array_1)

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x11x6x64xf32) <- (1x11x384xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_58, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (1x6x11x64xf32) <- (1x11x6x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            parameter_1, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_1

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_104 = slice_10

        # pd_op.slice: (11x32xf32) <- (512x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            parameter_0, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del full_int_array_3, parameter_0

        # pd_op.assign: (11x32xf32) <- (11x32xf32)
        assign_105 = slice_11

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_20 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_21 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_40 = paddle._C_ops.multiply(strided_slice_20, slice_11)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_41 = paddle._C_ops.multiply(strided_slice_21, slice_10)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_10 = paddle._C_ops.subtract(multiply_40, multiply_41)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_42 = paddle._C_ops.multiply(strided_slice_20, slice_10)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_43 = paddle._C_ops.multiply(strided_slice_21, slice_11)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_59 = paddle._C_ops.add(multiply_42, multiply_43)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_10 = [subtract_10, add_59]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_10 = paddle._C_ops.stack(combine_10, -1)
        del combine_10

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_10 = paddle._C_ops.flatten(stack_10, 3, 4)

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_22 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x6x11x32xf32) <- (1x6x11x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_23 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_44 = paddle._C_ops.multiply(strided_slice_22, slice_11)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_45 = paddle._C_ops.multiply(strided_slice_23, slice_10)

        # pd_op.subtract: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        subtract_11 = paddle._C_ops.subtract(multiply_44, multiply_45)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_46 = paddle._C_ops.multiply(strided_slice_22, slice_10)

        # pd_op.multiply: (1x6x11x32xf32) <- (1x6x11x32xf32, 11x32xf32)
        multiply_47 = paddle._C_ops.multiply(strided_slice_23, slice_11)

        # pd_op.add: (1x6x11x32xf32) <- (1x6x11x32xf32, 1x6x11x32xf32)
        add_60 = paddle._C_ops.add(multiply_46, multiply_47)

        # builtin.combine: ([1x6x11x32xf32, 1x6x11x32xf32]) <- (1x6x11x32xf32, 1x6x11x32xf32)
        combine_11 = [subtract_11, add_60]

        # pd_op.stack: (1x6x11x32x2xf32) <- ([1x6x11x32xf32, 1x6x11x32xf32])
        stack_11 = paddle._C_ops.stack(combine_11, -1)
        del combine_11

        # pd_op.flatten: (1x6x11x64xf32) <- (1x6x11x32x2xf32)
        flatten_11 = paddle._C_ops.flatten(stack_11, 3, 4)

        # pd_op.scale: (1x6x11x64xf32) <- (1x6x11x64xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(flatten_10, full_3, float("0"), True)
        del flatten_10

        # pd_op.matmul: (1x6x11x11xf32) <- (1x6x11x64xf32, 1x6x11x64xf32)
        matmul_43 = paddle._C_ops.matmul(scale_6, flatten_11, False, True)

        # pd_op.add: (1x6x11x11xf32) <- (1x6x11x11xf32, 1x1x1x11xf32)
        add_61 = paddle._C_ops.add(matmul_43, unsqueeze_0)

        # pd_op.softmax: (1x6x11x11xf32) <- (1x6x11x11xf32)
        softmax_5 = paddle._C_ops.softmax(add_61, -1)
        del add_61

        # pd_op.dropout: (1x6x11x11xf32, 1x6x11x11xui8) <- (1x6x11x11xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x11x64xf32) <- (1x6x11x11xf32, 1x6x11x64xf32)
        matmul_44 = paddle._C_ops.matmul(dropout_32, transpose_22, False, False)

        # pd_op.transpose: (1x11x6x64xf32) <- (1x6x11x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (1x11x384xf32) <- (1x11x6x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_7)
        del full_int_array_7

        # pd_op.matmul: (1x11x384xf32) <- (1x11x384xf32, 384x384xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_23, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_62 = paddle._C_ops.add(matmul_45, parameter_22)
        del parameter_22

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_62, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_62

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_63 = paddle._C_ops.add(layer_norm_30, dropout_34)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_63, parameter_17, parameter_16, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_16, parameter_17

        # pd_op.matmul: (1x11x1536xf32) <- (1x11x384xf32, 384x1536xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_33, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (1x11x1536xf32) <- (1x11x1536xf32, 1536xf32)
        add_64 = paddle._C_ops.add(matmul_46, parameter_20)
        del parameter_20

        # pd_op.gelu: (1x11x1536xf32) <- (1x11x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add_64, False)

        # pd_op.matmul: (1x11x384xf32) <- (1x11x1536xf32, 1536x384xf32)
        matmul_47 = paddle._C_ops.matmul(gelu_5, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 384xf32)
        add_65 = paddle._C_ops.add(matmul_47, parameter_18)
        del parameter_18

        # pd_op.dropout: (1x11x384xf32, 1x11x384xui8) <- (1x11x384xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_65, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_65

        # pd_op.add: (1x11x384xf32) <- (1x11x384xf32, 1x11x384xf32)
        add_66 = paddle._C_ops.add(layer_norm_33, dropout_36)

        # pd_op.layer_norm: (1x11x384xf32, 1x11xf32, 1x11xf32) <- (1x11x384xf32, 384xf32, 384xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_66, parameter_15, parameter_14, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_14, parameter_15

        # pd_op.slice: (1x384xf32) <- (1x11x384xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            layer_norm_36, [1], full_int_array_2, full_int_array_6, [1], [1]
        )
        del full_int_array_2

        # pd_op.matmul: (1x384xf32) <- (1x384xf32, 384x384xf32)
        matmul_48 = paddle._C_ops.matmul(slice_12, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x384xf32) <- (1x384xf32, 384xf32)
        add_67 = paddle._C_ops.add(matmul_48, parameter_12)
        del parameter_12

        # pd_op.tanh: (1x384xf32) <- (1x384xf32)
        tanh_0 = paddle._C_ops.tanh(add_67)
        del (
            add_0,
            add_1,
            add_11,
            add_12,
            add_13,
            add_14,
            add_15,
            add_16,
            add_19,
            add_2,
            add_20,
            add_22,
            add_23,
            add_24,
            add_25,
            add_26,
            add_27,
            add_3,
            add_30,
            add_31,
            add_33,
            add_34,
            add_35,
            add_36,
            add_37,
            add_38,
            add_4,
            add_41,
            add_42,
            add_44,
            add_45,
            add_46,
            add_47,
            add_48,
            add_49,
            add_5,
            add_52,
            add_53,
            add_55,
            add_56,
            add_57,
            add_58,
            add_59,
            add_60,
            add_63,
            add_64,
            add_66,
            add_67,
            add_8,
            add_9,
            assign_0,
            assign_1,
            assign_10,
            assign_100,
            assign_101,
            assign_102,
            assign_103,
            assign_104,
            assign_105,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
            assign_23,
            assign_24,
            assign_25,
            assign_26,
            assign_27,
            assign_28,
            assign_29,
            assign_3,
            assign_30,
            assign_31,
            assign_32,
            assign_33,
            assign_34,
            assign_35,
            assign_36,
            assign_37,
            assign_38,
            assign_39,
            assign_4,
            assign_40,
            assign_41,
            assign_42,
            assign_43,
            assign_44,
            assign_45,
            assign_46,
            assign_47,
            assign_48,
            assign_49,
            assign_5,
            assign_50,
            assign_51,
            assign_52,
            assign_53,
            assign_54,
            assign_55,
            assign_56,
            assign_57,
            assign_58,
            assign_59,
            assign_6,
            assign_60,
            assign_61,
            assign_62,
            assign_63,
            assign_64,
            assign_65,
            assign_66,
            assign_67,
            assign_68,
            assign_69,
            assign_7,
            assign_70,
            assign_71,
            assign_72,
            assign_73,
            assign_74,
            assign_75,
            assign_76,
            assign_77,
            assign_78,
            assign_79,
            assign_8,
            assign_80,
            assign_81,
            assign_82,
            assign_83,
            assign_84,
            assign_85,
            assign_86,
            assign_87,
            assign_88,
            assign_89,
            assign_9,
            assign_90,
            assign_91,
            assign_92,
            assign_93,
            assign_94,
            assign_95,
            assign_96,
            assign_97,
            assign_98,
            assign_99,
            dropout_0,
            dropout_1,
            dropout_10,
            dropout_11,
            dropout_12,
            dropout_13,
            dropout_14,
            dropout_15,
            dropout_16,
            dropout_17,
            dropout_18,
            dropout_19,
            dropout_2,
            dropout_20,
            dropout_21,
            dropout_22,
            dropout_23,
            dropout_24,
            dropout_25,
            dropout_26,
            dropout_27,
            dropout_28,
            dropout_29,
            dropout_3,
            dropout_30,
            dropout_31,
            dropout_32,
            dropout_33,
            dropout_34,
            dropout_35,
            dropout_36,
            dropout_37,
            dropout_4,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_8,
            dropout_9,
            embedding_0,
            embedding_1,
            flatten_1,
            flatten_11,
            flatten_3,
            flatten_5,
            flatten_7,
            flatten_9,
            full_2,
            full_3,
            full_int_array_4,
            full_int_array_5,
            full_int_array_6,
            gelu_0,
            gelu_1,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            layer_norm_1,
            layer_norm_10,
            layer_norm_11,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            layer_norm_15,
            layer_norm_16,
            layer_norm_17,
            layer_norm_18,
            layer_norm_19,
            layer_norm_2,
            layer_norm_20,
            layer_norm_21,
            layer_norm_22,
            layer_norm_23,
            layer_norm_24,
            layer_norm_25,
            layer_norm_26,
            layer_norm_27,
            layer_norm_28,
            layer_norm_29,
            layer_norm_3,
            layer_norm_30,
            layer_norm_31,
            layer_norm_32,
            layer_norm_33,
            layer_norm_34,
            layer_norm_35,
            layer_norm_36,
            layer_norm_37,
            layer_norm_38,
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_0,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_19,
            matmul_2,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_29,
            matmul_3,
            matmul_30,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_37,
            matmul_38,
            matmul_39,
            matmul_40,
            matmul_41,
            matmul_42,
            matmul_43,
            matmul_45,
            matmul_46,
            matmul_47,
            matmul_48,
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_8,
            matmul_9,
            multiply_0,
            multiply_1,
            multiply_10,
            multiply_11,
            multiply_12,
            multiply_13,
            multiply_14,
            multiply_15,
            multiply_16,
            multiply_17,
            multiply_18,
            multiply_19,
            multiply_2,
            multiply_20,
            multiply_21,
            multiply_22,
            multiply_23,
            multiply_24,
            multiply_25,
            multiply_26,
            multiply_27,
            multiply_28,
            multiply_29,
            multiply_3,
            multiply_30,
            multiply_31,
            multiply_32,
            multiply_33,
            multiply_34,
            multiply_35,
            multiply_36,
            multiply_37,
            multiply_38,
            multiply_39,
            multiply_4,
            multiply_40,
            multiply_41,
            multiply_42,
            multiply_43,
            multiply_44,
            multiply_45,
            multiply_46,
            multiply_47,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            reshape_11,
            reshape_15,
            reshape_19,
            reshape_23,
            reshape_3,
            reshape_7,
            scale_1,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            scale_6,
            slice_0,
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
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            softmax_4,
            softmax_5,
            stack_0,
            stack_1,
            stack_10,
            stack_11,
            stack_2,
            stack_3,
            stack_4,
            stack_5,
            stack_6,
            stack_7,
            stack_8,
            stack_9,
            strided_slice_0,
            strided_slice_1,
            strided_slice_10,
            strided_slice_11,
            strided_slice_12,
            strided_slice_13,
            strided_slice_14,
            strided_slice_15,
            strided_slice_16,
            strided_slice_17,
            strided_slice_18,
            strided_slice_19,
            strided_slice_2,
            strided_slice_20,
            strided_slice_21,
            strided_slice_22,
            strided_slice_23,
            strided_slice_3,
            strided_slice_4,
            strided_slice_5,
            strided_slice_6,
            strided_slice_7,
            strided_slice_8,
            strided_slice_9,
            subtract_0,
            subtract_1,
            subtract_10,
            subtract_11,
            subtract_2,
            subtract_3,
            subtract_4,
            subtract_5,
            subtract_6,
            subtract_7,
            subtract_8,
            subtract_9,
            transpose_0,
            transpose_1,
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
            transpose_2,
            transpose_20,
            transpose_21,
            transpose_22,
            transpose_23,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_0,
        )

        return tanh_0
