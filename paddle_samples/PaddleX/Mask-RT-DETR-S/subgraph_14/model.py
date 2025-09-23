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
        parameter_114,
        parameter_115,
        parameter_116,
        parameter_117,
        parameter_118,
        parameter_119,
        parameter_120,
        parameter_121,
        parameter_122,
        parameter_123,
        parameter_124,
        parameter_125,
        parameter_126,
        parameter_127,
        parameter_128,
        parameter_129,
        parameter_130,
        parameter_131,
        parameter_132,
        parameter_133,
        parameter_134,
        parameter_135,
        parameter_136,
        parameter_137,
        parameter_138,
        parameter_139,
        parameter_140,
        parameter_141,
        parameter_142,
        parameter_143,
        parameter_144,
        parameter_145,
        parameter_146,
        parameter_147,
        parameter_148,
        parameter_149,
        parameter_150,
        parameter_151,
        parameter_152,
        parameter_153,
        parameter_154,
        parameter_155,
        parameter_156,
        parameter_157,
        parameter_158,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
        data_11,
        data_12,
        data_13,
        data_14,
        data_15,
        data_16,
        data_17,
        data_18,
        data_19,
        data_20,
        data_21,
        data_22,
    ):
        # pd_op.conv2d: (1x256x-1x-1xf32) <- (1x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_14, parameter_158, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_14, parameter_158

        # pd_op.batch_norm_: (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
                parameter_157,
                parameter_156,
                parameter_155,
                parameter_154,
                True,
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
        del conv2d_0, parameter_154, parameter_155, parameter_156, parameter_157

        # pd_op.conv2d: (1x256x-1x-1xf32) <- (1x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_15, parameter_153, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_15, parameter_153

        # pd_op.batch_norm_: (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_152,
                parameter_151,
                parameter_150,
                parameter_149,
                True,
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
        del conv2d_1, parameter_149, parameter_150, parameter_151, parameter_152

        # pd_op.conv2d: (1x256x-1x-1xf32) <- (1x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_16, parameter_148, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_16, parameter_148

        # pd_op.batch_norm_: (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_147,
                parameter_146,
                parameter_145,
                parameter_144,
                True,
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
        del conv2d_2, parameter_144, parameter_145, parameter_146, parameter_147

        # pd_op.shape64: (4xi64) <- (1x256x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(batch_norm__0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_0

        # pd_op.shape64: (4xi64) <- (1x256x-1x-1xf32)
        shape64_1 = paddle._C_ops.shape64(batch_norm__0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_1

        # pd_op.flatten: (1x256x-1xf32) <- (1x256x-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(batch_norm__0, 2, 3)
        del batch_norm__0

        # pd_op.transpose: (1x-1x256xf32) <- (1x256x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_0 = paddle._C_ops.multiply(slice_0, slice_1)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(multiply_0, full_0, float("0"), True)
        del multiply_0

        # pd_op.shape64: (4xi64) <- (1x256x-1x-1xf32)
        shape64_2 = paddle._C_ops.shape64(batch_norm__6)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_2

        # pd_op.shape64: (4xi64) <- (1x256x-1x-1xf32)
        shape64_3 = paddle._C_ops.shape64(batch_norm__6)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_3

        # pd_op.flatten: (1x256x-1xf32) <- (1x256x-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(batch_norm__6, 2, 3)
        del batch_norm__6

        # pd_op.transpose: (1x-1x256xf32) <- (1x256x-1xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_1 = paddle._C_ops.multiply(slice_2, slice_3)

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_0 = paddle._C_ops.add(multiply_1, scale_0)
        del multiply_1

        # pd_op.shape64: (4xi64) <- (1x256x-1x-1xf32)
        shape64_4 = paddle._C_ops.shape64(batch_norm__12)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_4

        # pd_op.shape64: (4xi64) <- (1x256x-1x-1xf32)
        shape64_5 = paddle._C_ops.shape64(batch_norm__12)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_5

        # pd_op.flatten: (1x256x-1xf32) <- (1x256x-1x-1xf32)
        flatten_2 = paddle._C_ops.flatten(batch_norm__12, 2, 3)
        del batch_norm__12

        # pd_op.transpose: (1x-1x256xf32) <- (1x256x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_2 = paddle._C_ops.multiply(slice_4, slice_5)

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_1 = paddle._C_ops.add(multiply_2, add_0)
        del multiply_2

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x-1x256xf32, 1x-1x256xf32, 1x-1x256xf32]) <- (1x-1x256xf32, 1x-1x256xf32, 1x-1x256xf32)
        combine_0 = [transpose_0, transpose_1, transpose_2]
        del transpose_0, transpose_1, transpose_2

        # pd_op.concat: (1x-1x256xf32) <- ([1x-1x256xf32, 1x-1x256xf32, 1x-1x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)
        del combine_0

        # pd_op.shape64: (3xi64) <- (1x-1x256xf32)
        shape64_6 = paddle._C_ops.shape64(concat_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_6

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_2,
            [],
            paddle.float32,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x-1x256xf32) <- (1x-1x256xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            concat_0, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            assign_value__0,
            full_3,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x8400x1xb) <- (1x8400x1xb, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            data_22, full_3, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x8400x1xf32) <- (1x8400x1xb)
        cast_1 = paddle._C_ops.cast(full_like_2, paddle.float32)
        del full_like_2

        # pd_op.cast: (1x8400x1xf32) <- (1x8400x1xb)
        cast_2 = paddle._C_ops.cast(data_22, paddle.float32)
        del data_22

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, xf32)
        add_2 = paddle._C_ops.add(full_like_0, full_like_1)
        del full_like_0, full_like_1

        # pd_op.add: (1x8400x256xf32) <- (1x-1x256xf32, 1x8400x1xf32)
        add_3 = paddle._C_ops.add(add_2, cast_1)
        del add_2, cast_1

        # pd_op.add: (1x8400x256xf32) <- (1x-1x256xf32, 1x8400x256xf32)
        add_4 = paddle._C_ops.add(concat_0, add_3)

        # pd_op.add: (1x8400x256xf32) <- (xf32, 1x8400x256xf32)
        add_5 = paddle._C_ops.add(assign_value__0, add_3)
        del assign_value__0

        # pd_op.add: (1x8400x256xf32) <- (1x8400x1xf32, 1x8400x256xf32)
        add_6 = paddle._C_ops.add(cast_2, add_3)
        del add_3, cast_2

        # pd_op.cast: (1x8400x256xb) <- (1x8400x256xf32)
        cast_3 = paddle._C_ops.cast(add_6, paddle.bool)
        del add_6

        # pd_op.where: (1x8400x256xf32) <- (1x8400x256xb, 1x8400x256xf32, 1x8400x256xf32)
        where_0 = paddle._C_ops.where(cast_3, add_4, add_5)
        del add_4, add_5, cast_3

        # pd_op.matmul: (1x8400x256xf32) <- (1x8400x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(where_0, parameter_143, False, False)
        del parameter_143, where_0

        # pd_op.add: (1x8400x256xf32) <- (1x8400x256xf32, 256xf32)
        add_7 = paddle._C_ops.add(matmul_0, parameter_142)
        del matmul_0, parameter_142

        # pd_op.layer_norm: (1x8400x256xf32, 1x8400xf32, 1x8400xf32) <- (1x8400x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_141, parameter_140, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_7, parameter_140, parameter_141

        # pd_op.matmul: (1x8400x2xf32) <- (1x8400x256xf32, 256x2xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_0, parameter_139, False, False)

        # pd_op.add: (1x8400x2xf32) <- (1x8400x2xf32, 2xf32)
        add_8 = paddle._C_ops.add(matmul_1, parameter_138)
        del matmul_1

        # pd_op.matmul: (1x8400x256xf32) <- (1x8400x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_0, parameter_137, False, False)

        # pd_op.add: (1x8400x256xf32) <- (1x8400x256xf32, 256xf32)
        add_9 = paddle._C_ops.add(matmul_2, parameter_136)
        del matmul_2

        # pd_op.relu: (1x8400x256xf32) <- (1x8400x256xf32)
        relu_0 = paddle._C_ops.relu(add_9)
        del add_9

        # pd_op.matmul: (1x8400x256xf32) <- (1x8400x256xf32, 256x256xf32)
        matmul_3 = paddle._C_ops.matmul(relu_0, parameter_135, False, False)
        del relu_0

        # pd_op.add: (1x8400x256xf32) <- (1x8400x256xf32, 256xf32)
        add_10 = paddle._C_ops.add(matmul_3, parameter_134)
        del matmul_3

        # pd_op.relu: (1x8400x256xf32) <- (1x8400x256xf32)
        relu_1 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.matmul: (1x8400x4xf32) <- (1x8400x256xf32, 256x4xf32)
        matmul_4 = paddle._C_ops.matmul(relu_1, parameter_133, False, False)
        del relu_1

        # pd_op.add: (1x8400x4xf32) <- (1x8400x4xf32, 4xf32)
        add_11 = paddle._C_ops.add(matmul_4, parameter_132)
        del matmul_4

        # pd_op.add: (1x8400x4xf32) <- (1x8400x4xf32, 1x8400x4xf32)
        add_12 = paddle._C_ops.add(add_11, data_21)
        del add_11, data_21

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.max: (1x8400xf32) <- (1x8400x2xf32, 1xi64)
        max_0 = paddle._C_ops.max(add_8, full_int_array_4, False)
        del add_8

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("300"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (1x300xf32, 1x300xi64) <- (1x8400xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(max_0, full_4, 1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_4, max_0

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (1xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_5, full_6, full_6, dtype="int64")
        del full_5, full_6

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(arange_0, full_int_array_4)
        del arange_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 300]

        # pd_op.tile: (1x300xi64) <- (1x1xi64, 2xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_0, full_int_array_5)
        del full_int_array_5

        # builtin.combine: ([1x300xi64, 1x300xi64]) <- (1x300xi64, 1x300xi64)
        combine_1 = [tile_1, topk_1]
        del tile_1, topk_1

        # pd_op.stack: (1x300x2xi64) <- ([1x300xi64, 1x300xi64])
        stack_0 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.gather_nd: (1x300x256xf32) <- (1x8400x256xf32, 1x300x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(layer_norm_0, stack_0)
        del layer_norm_0

        # pd_op.gather_nd: (1x300x4xf32) <- (1x8400x4xf32, 1x300x2xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(add_12, stack_0)
        del add_12, stack_0

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                gather_nd_0, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (1x300x2xf32) <- (1x300x256xf32, 256x2xf32)
        matmul_5 = paddle._C_ops.matmul(layer_norm_3, parameter_139, False, False)

        # pd_op.add: (1x300x2xf32) <- (1x300x2xf32, 2xf32)
        add_13 = paddle._C_ops.add(matmul_5, parameter_138)
        del matmul_5

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_129, False, False)
        del layer_norm_3

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_14 = paddle._C_ops.add(matmul_6, parameter_128)
        del matmul_6

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_2 = paddle._C_ops.relu(add_14)
        del add_14

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_7 = paddle._C_ops.matmul(relu_2, parameter_127, False, False)
        del relu_2

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_15 = paddle._C_ops.add(matmul_7, parameter_126)
        del matmul_7

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_3 = paddle._C_ops.relu(add_15)
        del add_15

        # pd_op.matmul: (1x300x128xf32) <- (1x300x256xf32, 256x128xf32)
        matmul_8 = paddle._C_ops.matmul(relu_3, parameter_125, False, False)
        del relu_3

        # pd_op.add: (1x300x128xf32) <- (1x300x128xf32, 128xf32)
        add_16 = paddle._C_ops.add(matmul_8, parameter_124)
        del matmul_8

        # pd_op.flatten: (1x128x-1xf32) <- (1x128x-1x-1xf32)
        flatten_3 = paddle._C_ops.flatten(data_17, 2, 3)
        del data_17

        # pd_op.bmm: (1x300x-1xf32) <- (1x300x128xf32, 1x128x-1xf32)
        bmm_0 = paddle._C_ops.bmm(add_16, flatten_3)
        del add_16

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("300"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [full_7, full_8, data_12, data_13]
        del data_12, data_13, full_8

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.reshape: (1x300x-1x-1xf32) <- (1x300x-1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(bmm_0, stack_1)
        del bmm_0

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(gather_nd_1)
        del gather_nd_1

        # pd_op.share_data_: (1x300x256xf32) <- (1x300x256xf32)
        share_data__0 = gather_nd_0.detach()
        del gather_nd_0

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x300x-1x-1xb) <- (1x300x-1x-1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(reshape_1, full_9)
        del full_9, reshape_1

        # pd_op.shape64: (4xi64) <- (1x300x-1x-1xb)
        shape64_7 = paddle._C_ops.shape64(greater_than_0)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_7

        # pd_op.shape64: (4xi64) <- (1x300x-1x-1xb)
        shape64_8 = paddle._C_ops.shape64(greater_than_0)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_8

        # pd_op.cast: (xf32) <- (xi64)
        cast_4 = paddle._C_ops.cast(slice_7, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_1 = paddle.arange(full_3, cast_4, full_0, dtype="float32")
        del cast_4

        # pd_op.cast: (xf32) <- (xi64)
        cast_5 = paddle._C_ops.cast(slice_8, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_2 = paddle.arange(full_3, cast_5, full_0, dtype="float32")
        del cast_5

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_3 = [arange_1, arange_2]
        del arange_1, arange_2

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_3)
        del combine_3

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_6 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.multiply: (1x300x-1x-1xf32) <- (-1x-1xf32, 1x300x-1x-1xf32)
        multiply_3 = paddle._C_ops.multiply(split_1, cast_6)
        del cast_6, split_1

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        flatten_4 = paddle._C_ops.flatten(multiply_3, 2, 3)

        # pd_op.max: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        max_1 = paddle._C_ops.max(flatten_4, full_int_array_4, False)
        del flatten_4

        # pd_op.scale: (1x300xf32) <- (1x300xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(max_1, full_0, float("1"), True)
        del max_1

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.full_like: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            multiply_3,
            full_3,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_4 = paddle._C_ops.full_like(
            assign_value__1,
            full_3,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x-1x-1xb) <- (1x300x-1x-1xb, 1xf32)
        full_like_5 = paddle._C_ops.full_like(
            greater_than_0,
            full_3,
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_7 = paddle._C_ops.cast(full_like_5, paddle.float32)
        del full_like_5

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_8 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, xf32)
        add_17 = paddle._C_ops.add(full_like_3, full_like_4)
        del full_like_3, full_like_4

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_18 = paddle._C_ops.add(add_17, cast_7)
        del add_17, cast_7

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_19 = paddle._C_ops.add(multiply_3, add_18)
        del multiply_3

        # pd_op.add: (1x300x-1x-1xf32) <- (xf32, 1x300x-1x-1xf32)
        add_20 = paddle._C_ops.add(assign_value__1, add_18)
        del assign_value__1

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_21 = paddle._C_ops.add(cast_8, add_18)
        del add_18, cast_8

        # pd_op.cast: (1x300x-1x-1xb) <- (1x300x-1x-1xf32)
        cast_9 = paddle._C_ops.cast(add_21, paddle.bool)
        del add_21

        # pd_op.where: (1x300x-1x-1xf32) <- (1x300x-1x-1xb, 1x300x-1x-1xf32, 1x300x-1x-1xf32)
        where_1 = paddle._C_ops.where(cast_9, add_19, add_20)
        del add_19, add_20, cast_9

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        flatten_5 = paddle._C_ops.flatten(where_1, 2, 3)
        del where_1

        # pd_op.min: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        min_0 = paddle._C_ops.min(flatten_5, full_int_array_4, False)
        del flatten_5

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_10 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.multiply: (1x300x-1x-1xf32) <- (-1x-1xf32, 1x300x-1x-1xf32)
        multiply_4 = paddle._C_ops.multiply(split_0, cast_10)
        del cast_10, split_0

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        flatten_6 = paddle._C_ops.flatten(multiply_4, 2, 3)

        # pd_op.max: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        max_2 = paddle._C_ops.max(flatten_6, full_int_array_4, False)
        del flatten_6

        # pd_op.scale: (1x300xf32) <- (1x300xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(max_2, full_0, float("1"), True)
        del max_2

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_11,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )
        del full_11

        # pd_op.full_like: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1xf32)
        full_like_6 = paddle._C_ops.full_like(
            multiply_4,
            full_3,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_7 = paddle._C_ops.full_like(
            assign_value__2,
            full_3,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x-1x-1xb) <- (1x300x-1x-1xb, 1xf32)
        full_like_8 = paddle._C_ops.full_like(
            greater_than_0,
            full_3,
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_11 = paddle._C_ops.cast(full_like_8, paddle.float32)
        del full_like_8

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_12 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, xf32)
        add_22 = paddle._C_ops.add(full_like_6, full_like_7)
        del full_like_6, full_like_7

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_23 = paddle._C_ops.add(add_22, cast_11)
        del add_22, cast_11

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_24 = paddle._C_ops.add(multiply_4, add_23)
        del multiply_4

        # pd_op.add: (1x300x-1x-1xf32) <- (xf32, 1x300x-1x-1xf32)
        add_25 = paddle._C_ops.add(assign_value__2, add_23)
        del assign_value__2

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_26 = paddle._C_ops.add(cast_12, add_23)
        del add_23, cast_12

        # pd_op.cast: (1x300x-1x-1xb) <- (1x300x-1x-1xf32)
        cast_13 = paddle._C_ops.cast(add_26, paddle.bool)
        del add_26

        # pd_op.where: (1x300x-1x-1xf32) <- (1x300x-1x-1xb, 1x300x-1x-1xf32, 1x300x-1x-1xf32)
        where_2 = paddle._C_ops.where(cast_13, add_24, add_25)
        del add_24, add_25, cast_13

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        flatten_7 = paddle._C_ops.flatten(where_2, 2, 3)
        del where_2

        # pd_op.min: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        min_1 = paddle._C_ops.min(flatten_7, full_int_array_4, False)
        del flatten_7

        # builtin.combine: ([1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32]) <- (1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32)
        combine_4 = [min_0, min_1, scale_1, scale_2]
        del min_0, min_1, scale_1, scale_2

        # pd_op.stack: (1x300x4xf32) <- ([1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        # pd_op.any: (1x300xb) <- (1x300x-1x-1xb)
        any_0 = paddle._C_ops.any(greater_than_0, [2, 3], False)
        del greater_than_0

        # pd_op.unsqueeze: (1x300x1xb) <- (1x300xb, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(any_0, full_int_array_0)
        del any_0

        # pd_op.cast: (1x300x1xf32) <- (1x300x1xb)
        cast_14 = paddle._C_ops.cast(unsqueeze_1, paddle.float32)
        del unsqueeze_1

        # pd_op.multiply: (1x300x4xf32) <- (1x300x4xf32, 1x300x1xf32)
        multiply_5 = paddle._C_ops.multiply(stack_2, cast_14)
        del cast_14, stack_2

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_5 = [slice_8, slice_7, slice_8, slice_7]
        del slice_7, slice_8

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.assign: (4xi64) <- (4xi64)
        assign_0 = stack_3
        del stack_3

        # pd_op.cast: (4xf32) <- (4xi64)
        cast_15 = paddle._C_ops.cast(assign_0, paddle.float32)
        del assign_0

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        divide_0 = paddle._C_ops.divide(multiply_5, cast_15)
        del cast_15, multiply_5

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32]) <- (1x300x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(divide_0, 4, full_12)
        del divide_0

        # builtin.split: (1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32) <- ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32])
        (
            split_2,
            split_3,
            split_4,
            split_5,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.add: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        add_27 = paddle._C_ops.add(split_2, split_4)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x300x1xf32) <- (1x300x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_27, full_13, float("0"), True)
        del add_27

        # pd_op.add: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        add_28 = paddle._C_ops.add(split_3, split_5)

        # pd_op.scale: (1x300x1xf32) <- (1x300x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(add_28, full_13, float("0"), True)
        del add_28

        # pd_op.subtract: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        subtract_0 = paddle._C_ops.subtract(split_4, split_2)
        del split_2, split_4

        # pd_op.subtract: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_5, split_3)
        del split_3, split_5

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32]) <- (1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32)
        combine_6 = [scale_3, scale_4, subtract_0, subtract_1]
        del scale_3, scale_4, subtract_0, subtract_1

        # pd_op.concat: (1x300x4xf32) <- ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_6, full_14)
        del combine_6

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(concat_1, full_3, full_0)
        del concat_1

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("1e-05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(clip_0, full_15, full_16)

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(clip_0, full_17, float("1"), True)
        del clip_0

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(scale_5, full_15, full_16)
        del scale_5

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        divide_1 = paddle._C_ops.divide(clip_1, clip_2)
        del clip_1, clip_2

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        log_0 = paddle._C_ops.log(divide_1)
        del divide_1

        # pd_op.share_data_: (1x300x4xf32) <- (1x300x4xf32)
        share_data__1 = log_0.detach()
        del log_0

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(share_data__1)
        del share_data__1

        # pd_op.unsqueeze: (1x300x1x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(sigmoid_1, full_int_array_0)

        # pd_op.matmul: (1x300x512xf32) <- (1x300x4xf32, 4x512xf32)
        matmul_9 = paddle._C_ops.matmul(sigmoid_1, parameter_123, False, False)

        # pd_op.add: (1x300x512xf32) <- (1x300x512xf32, 512xf32)
        add_29 = paddle._C_ops.add(matmul_9, parameter_122)
        del matmul_9

        # pd_op.relu: (1x300x512xf32) <- (1x300x512xf32)
        relu_4 = paddle._C_ops.relu(add_29)
        del add_29

        # pd_op.matmul: (1x300x256xf32) <- (1x300x512xf32, 512x256xf32)
        matmul_10 = paddle._C_ops.matmul(relu_4, parameter_121, False, False)
        del relu_4

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_30 = paddle._C_ops.add(matmul_10, parameter_120)
        del matmul_10

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_31 = paddle._C_ops.add(share_data__0, add_30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [256]

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_0, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_1, [0], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_11 = paddle._C_ops.matmul(add_31, slice_9, False, False)
        del slice_9

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_32 = paddle._C_ops.add(matmul_11, slice_10)
        del matmul_11, slice_10

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [0, 0, 8, 32]

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_32, full_int_array_8)
        del add_32

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [512]

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_0, [1], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            data_1, [0], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_12 = paddle._C_ops.matmul(add_31, slice_11, False, False)
        del add_31, slice_11

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_33 = paddle._C_ops.add(matmul_12, slice_12)
        del matmul_12, slice_12

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(add_33, full_int_array_8)
        del add_33

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_3, [0, 2, 1, 3])
        del reshape_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [2147483647]

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            data_0, [1], full_int_array_9, full_int_array_10, [1], []
        )
        del data_0

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            data_1, [0], full_int_array_9, full_int_array_10, [1], []
        )
        del data_1

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_13 = paddle._C_ops.matmul(share_data__0, slice_13, False, False)
        del slice_13

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_34 = paddle._C_ops.add(matmul_13, slice_14)
        del matmul_13, slice_14

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_34, full_int_array_8)
        del add_34

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x8x300x300xf32) <- (1x8x300x32xf32, 1x8x300x32xf32)
        matmul_14 = paddle._C_ops.matmul(transpose_3, transpose_4, False, True)
        del transpose_3, transpose_4

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x8x300x300xf32) <- (1x8x300x300xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_14, full_18, float("0"), True)
        del matmul_14

        # pd_op.softmax: (1x8x300x300xf32) <- (1x8x300x300xf32)
        softmax_0 = paddle._C_ops.softmax(scale_6, -1)
        del scale_6

        # pd_op.matmul: (1x8x300x32xf32) <- (1x8x300x300xf32, 1x8x300x32xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_0, transpose_5, False, False)
        del softmax_0, transpose_5

        # pd_op.transpose: (1x300x8x32xf32) <- (1x8x300x32xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])
        del matmul_15

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [0, 0, 256]

        # pd_op.reshape: (1x300x256xf32) <- (1x300x8x32xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_6, full_int_array_11)
        del transpose_6

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_5, parameter_119, False, False)
        del parameter_119, reshape_5

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_35 = paddle._C_ops.add(matmul_16, parameter_118)
        del matmul_16, parameter_118

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_36 = paddle._C_ops.add(share_data__0, add_35)
        del add_35, share_data__0

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_36, parameter_117, parameter_116, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_36, parameter_116, parameter_117

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_37 = paddle._C_ops.add(layer_norm_6, add_30)
        del add_30

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x256xf32, 256x256xf32)
        matmul_17 = paddle._C_ops.matmul(concat_0, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_38 = paddle._C_ops.add(matmul_17, parameter_114)
        del matmul_17, parameter_114

        # pd_op.full: (xi64) <- ()
        full_19 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_20 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_7 = [full_7, slice_6, full_19, full_20]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.reshape: (1x-1x8x32xf32) <- (1x-1x256xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_38, stack_4)
        del add_38, stack_4

        # pd_op.matmul: (1x300x192xf32) <- (1x300x256xf32, 256x192xf32)
        matmul_18 = paddle._C_ops.matmul(add_37, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (1x300x192xf32) <- (1x300x192xf32, 192xf32)
        add_39 = paddle._C_ops.add(matmul_18, parameter_112)
        del matmul_18, parameter_112

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_12 = [1, 300, 8, 3, 4, 2]

        # pd_op.reshape: (1x300x8x3x4x2xf32) <- (1x300x192xf32, 6xi64)
        reshape_7 = paddle._C_ops.reshape(add_39, full_int_array_12)
        del add_39

        # pd_op.matmul: (1x300x96xf32) <- (1x300x256xf32, 256x96xf32)
        matmul_19 = paddle._C_ops.matmul(add_37, parameter_111, False, False)
        del add_37, parameter_111

        # pd_op.add: (1x300x96xf32) <- (1x300x96xf32, 96xf32)
        add_40 = paddle._C_ops.add(matmul_19, parameter_110)
        del matmul_19, parameter_110

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 300, 8, 12]

        # pd_op.reshape: (1x300x8x12xf32) <- (1x300x96xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_40, full_int_array_13)
        del add_40

        # pd_op.softmax: (1x300x8x12xf32) <- (1x300x8x12xf32)
        softmax_1 = paddle._C_ops.softmax(reshape_8, -1)
        del reshape_8

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_14 = [1, 300, 8, 3, 4]

        # pd_op.reshape: (1x300x8x3x4xf32) <- (1x300x8x12xf32, 5xi64)
        reshape_9 = paddle._C_ops.reshape(softmax_1, full_int_array_14)
        del softmax_1

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            unsqueeze_2, [3], full_int_array_6, full_int_array_0, [1], []
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_15 = [2, 4]

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(slice_15, full_int_array_15)
        del slice_15

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(reshape_7, full_21, float("0"), True)
        del reshape_7

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            unsqueeze_2, [3], full_int_array_0, full_int_array_10, [1], []
        )
        del unsqueeze_2

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(slice_16, full_int_array_15)
        del slice_16

        # pd_op.multiply: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1x300x1x1x1x2xf32)
        multiply_6 = paddle._C_ops.multiply(scale_7, unsqueeze_4)
        del scale_7, unsqueeze_4

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(multiply_6, full_13, float("0"), True)
        del multiply_6

        # pd_op.add: (1x300x8x3x4x2xf32) <- (1x300x1x1x1x2xf32, 1x300x8x3x4x2xf32)
        add_41 = paddle._C_ops.add(unsqueeze_3, scale_8)
        del scale_8, unsqueeze_3

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_8 = [slice_0, slice_1]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = stack_5
        del stack_5

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_9 = [slice_2, slice_3]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = stack_6
        del stack_6

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_10 = [slice_4, slice_5]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = stack_7
        del stack_7

        # builtin.combine: ([2xi64, 2xi64, 2xi64]) <- (2xi64, 2xi64, 2xi64)
        combine_11 = [assign_1, assign_2, assign_3]
        del assign_1, assign_2, assign_3

        # pd_op.stack: (3x2xi64) <- ([2xi64, 2xi64, 2xi64])
        stack_8 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.assign: (3x2xi64) <- (3x2xi64)
        assign_4 = stack_8
        del stack_8

        # pd_op.full: (xi64) <- ()
        full_22 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_22,
            [],
            paddle.int64,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del full_22

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_12 = [assign_value__3, scale_0, add_0]
        del assign_value__3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.assign: (3xi64) <- (3xi64)
        assign_5 = stack_9
        del stack_9

        # pd_op.shape64: (4xi64) <- (1x-1x8x32xf32)
        shape64_9 = paddle._C_ops.shape64(reshape_6)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_9

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            assign_4, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            slice_18, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            slice_18, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_18

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_7 = paddle._C_ops.multiply(slice_19, slice_20)
        del slice_19, slice_20

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            assign_4, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            slice_21, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            slice_21, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_21

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_8 = paddle._C_ops.multiply(slice_22, slice_23)
        del slice_22, slice_23

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            assign_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            slice_24, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            slice_24, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_24

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_9 = paddle._C_ops.multiply(slice_25, slice_26)
        del slice_25, slice_26

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_13 = [multiply_7, multiply_8, multiply_9]
        del multiply_7, multiply_8, multiply_9

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x-1x8x32xf32, 3xi64, 1xi32)
        split_6 = paddle._C_ops.split(reshape_6, stack_10, full_1)
        del reshape_6, stack_10

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_7,
            split_8,
            split_9,
        ) = split_6
        del split_6

        # pd_op.full: (1xf32) <- ()
        full_23 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(add_41, full_23, float("0"), True)
        del add_41

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(scale_9, full_0, float("-1"), True)
        del scale_9

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            assign_4, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            slice_27, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            slice_27, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_27

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_8 = paddle._C_ops.flatten(split_7, 2, 3)
        del split_7

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_7 = paddle._C_ops.transpose(flatten_8, [0, 2, 1])
        del flatten_8

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_14 = [full_19, full_20, slice_28, slice_29]
        del slice_28, slice_29

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(transpose_7, stack_11)
        del stack_11, transpose_7

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            scale_10, [3], full_int_array_6, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_8 = paddle._C_ops.transpose(slice_30, [0, 2, 1, 3, 4])
        del slice_30

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_9 = paddle._C_ops.flatten(transpose_8, 0, 1)
        del transpose_8

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(
            reshape_10, flatten_9, "bilinear", "zeros", False
        )
        del flatten_9, reshape_10

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            assign_4, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            slice_31, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            slice_31, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_31

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_10 = paddle._C_ops.flatten(split_8, 2, 3)
        del split_8

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_9 = paddle._C_ops.transpose(flatten_10, [0, 2, 1])
        del flatten_10

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_15 = [full_19, full_20, slice_32, slice_33]
        del slice_32, slice_33

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_9, stack_12)
        del stack_12, transpose_9

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            scale_10, [3], full_int_array_3, full_int_array_0, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_10 = paddle._C_ops.transpose(slice_34, [0, 2, 1, 3, 4])
        del slice_34

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_11 = paddle._C_ops.flatten(transpose_10, 0, 1)
        del transpose_10

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(
            reshape_11, flatten_11, "bilinear", "zeros", False
        )
        del flatten_11, reshape_11

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            assign_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del assign_4

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            slice_35, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            slice_35, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_35

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_12 = paddle._C_ops.flatten(split_9, 2, 3)
        del split_9

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_11 = paddle._C_ops.transpose(flatten_12, [0, 2, 1])
        del flatten_12

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_16 = [full_19, full_20, slice_36, slice_37]
        del slice_36, slice_37

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_11, stack_13)
        del stack_13, transpose_11

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            scale_10, [3], full_int_array_0, full_int_array_1, [1], [3]
        )
        del scale_10

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_12 = paddle._C_ops.transpose(slice_38, [0, 2, 1, 3, 4])
        del slice_38

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_13 = paddle._C_ops.flatten(transpose_12, 0, 1)
        del transpose_12

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(
            reshape_12, flatten_13, "bilinear", "zeros", False
        )
        del flatten_13, reshape_12

        # pd_op.transpose: (1x8x300x3x4xf32) <- (1x300x8x3x4xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3, 4])
        del reshape_9

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [8, 1, 300, 12]

        # pd_op.reshape: (8x1x300x12xf32) <- (1x8x300x3x4xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_13, full_int_array_16)
        del transpose_13

        # builtin.combine: ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32]) <- (8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32)
        combine_17 = [grid_sample_0, grid_sample_1, grid_sample_2]
        del grid_sample_0, grid_sample_1, grid_sample_2

        # pd_op.stack: (8x32x300x3x4xf32) <- ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32])
        stack_14 = paddle._C_ops.stack(combine_17, -2)
        del combine_17

        # pd_op.flatten: (8x32x300x12xf32) <- (8x32x300x3x4xf32)
        flatten_14 = paddle._C_ops.flatten(stack_14, 3, 4)
        del stack_14

        # pd_op.multiply: (8x32x300x12xf32) <- (8x32x300x12xf32, 8x1x300x12xf32)
        multiply_10 = paddle._C_ops.multiply(flatten_14, reshape_13)
        del flatten_14, reshape_13

        # pd_op.sum: (8x32x300xf32) <- (8x32x300x12xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_10, full_int_array_4, None, False)
        del multiply_10

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_17 = [1, 256, 300]

        # pd_op.reshape: (1x256x300xf32) <- (8x32x300xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(sum_0, full_int_array_17)
        del sum_0

        # pd_op.transpose: (1x300x256xf32) <- (1x256x300xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1])
        del reshape_14

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_14, parameter_109, False, False)
        del parameter_109, transpose_14

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_42 = paddle._C_ops.add(matmul_20, parameter_108)
        del matmul_20, parameter_108

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_43 = paddle._C_ops.add(layer_norm_6, add_42)
        del add_42, layer_norm_6

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_107, parameter_106, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_43, parameter_106, parameter_107

        # pd_op.matmul: (1x300x1024xf32) <- (1x300x256xf32, 256x1024xf32)
        matmul_21 = paddle._C_ops.matmul(layer_norm_9, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (1x300x1024xf32) <- (1x300x1024xf32, 1024xf32)
        add_44 = paddle._C_ops.add(matmul_21, parameter_104)
        del matmul_21, parameter_104

        # pd_op.relu: (1x300x1024xf32) <- (1x300x1024xf32)
        relu_5 = paddle._C_ops.relu(add_44)
        del add_44

        # pd_op.matmul: (1x300x256xf32) <- (1x300x1024xf32, 1024x256xf32)
        matmul_22 = paddle._C_ops.matmul(relu_5, parameter_103, False, False)
        del parameter_103, relu_5

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_45 = paddle._C_ops.add(matmul_22, parameter_102)
        del matmul_22, parameter_102

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_46 = paddle._C_ops.add(layer_norm_9, add_45)
        del add_45, layer_norm_9

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_101, parameter_100, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_46, parameter_100, parameter_101

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_12, parameter_137, False, False)

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_47 = paddle._C_ops.add(matmul_23, parameter_136)
        del matmul_23

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_6 = paddle._C_ops.relu(add_47)
        del add_47

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_24 = paddle._C_ops.matmul(relu_6, parameter_135, False, False)
        del relu_6

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_48 = paddle._C_ops.add(matmul_24, parameter_134)
        del matmul_24

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_7 = paddle._C_ops.relu(add_48)
        del add_48

        # pd_op.matmul: (1x300x4xf32) <- (1x300x256xf32, 256x4xf32)
        matmul_25 = paddle._C_ops.matmul(relu_7, parameter_133, False, False)
        del relu_7

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        add_49 = paddle._C_ops.add(matmul_25, parameter_132)
        del matmul_25

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_3 = paddle._C_ops.clip(sigmoid_1, full_3, full_0)
        del sigmoid_1

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_4 = paddle._C_ops.clip(clip_3, full_15, full_16)

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(clip_3, full_17, float("1"), True)
        del clip_3

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_5 = paddle._C_ops.clip(scale_11, full_15, full_16)
        del scale_11

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        divide_2 = paddle._C_ops.divide(clip_4, clip_5)
        del clip_4, clip_5

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        log_1 = paddle._C_ops.log(divide_2)
        del divide_2

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        add_50 = paddle._C_ops.add(add_49, log_1)
        del add_49, log_1

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_50)
        del add_50

        # pd_op.unsqueeze: (1x300x1x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(sigmoid_2, full_int_array_0)

        # pd_op.matmul: (1x300x512xf32) <- (1x300x4xf32, 4x512xf32)
        matmul_26 = paddle._C_ops.matmul(sigmoid_2, parameter_123, False, False)

        # pd_op.add: (1x300x512xf32) <- (1x300x512xf32, 512xf32)
        add_51 = paddle._C_ops.add(matmul_26, parameter_122)
        del matmul_26

        # pd_op.relu: (1x300x512xf32) <- (1x300x512xf32)
        relu_8 = paddle._C_ops.relu(add_51)
        del add_51

        # pd_op.matmul: (1x300x256xf32) <- (1x300x512xf32, 512x256xf32)
        matmul_27 = paddle._C_ops.matmul(relu_8, parameter_121, False, False)
        del relu_8

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_52 = paddle._C_ops.add(matmul_27, parameter_120)
        del matmul_27

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_53 = paddle._C_ops.add(layer_norm_12, add_52)

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            data_2, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            data_3, [0], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_28 = paddle._C_ops.matmul(add_53, slice_39, False, False)
        del slice_39

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_54 = paddle._C_ops.add(matmul_28, slice_40)
        del matmul_28, slice_40

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(add_54, full_int_array_8)
        del add_54

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_15, [0, 2, 1, 3])
        del reshape_15

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            data_2, [1], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            data_3, [0], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_29 = paddle._C_ops.matmul(add_53, slice_41, False, False)
        del add_53, slice_41

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_55 = paddle._C_ops.add(matmul_29, slice_42)
        del matmul_29, slice_42

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_55, full_int_array_8)
        del add_55

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            data_2, [1], full_int_array_9, full_int_array_10, [1], []
        )
        del data_2

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            data_3, [0], full_int_array_9, full_int_array_10, [1], []
        )
        del data_3

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_12, slice_43, False, False)
        del slice_43

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_56 = paddle._C_ops.add(matmul_30, slice_44)
        del matmul_30, slice_44

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_56, full_int_array_8)
        del add_56

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.matmul: (1x8x300x300xf32) <- (1x8x300x32xf32, 1x8x300x32xf32)
        matmul_31 = paddle._C_ops.matmul(transpose_15, transpose_16, False, True)
        del transpose_15, transpose_16

        # pd_op.scale: (1x8x300x300xf32) <- (1x8x300x300xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(matmul_31, full_18, float("0"), True)
        del matmul_31

        # pd_op.softmax: (1x8x300x300xf32) <- (1x8x300x300xf32)
        softmax_2 = paddle._C_ops.softmax(scale_12, -1)
        del scale_12

        # pd_op.matmul: (1x8x300x32xf32) <- (1x8x300x300xf32, 1x8x300x32xf32)
        matmul_32 = paddle._C_ops.matmul(softmax_2, transpose_17, False, False)
        del softmax_2, transpose_17

        # pd_op.transpose: (1x300x8x32xf32) <- (1x8x300x32xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])
        del matmul_32

        # pd_op.reshape: (1x300x256xf32) <- (1x300x8x32xf32, 3xi64)
        reshape_18 = paddle._C_ops.reshape(transpose_18, full_int_array_11)
        del transpose_18

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_33 = paddle._C_ops.matmul(reshape_18, parameter_99, False, False)
        del parameter_99, reshape_18

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_57 = paddle._C_ops.add(matmul_33, parameter_98)
        del matmul_33, parameter_98

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_58 = paddle._C_ops.add(layer_norm_12, add_57)
        del add_57, layer_norm_12

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_97, parameter_96, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_58, parameter_96, parameter_97

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_59 = paddle._C_ops.add(layer_norm_15, add_52)
        del add_52

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x256xf32, 256x256xf32)
        matmul_34 = paddle._C_ops.matmul(concat_0, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_60 = paddle._C_ops.add(matmul_34, parameter_94)
        del matmul_34, parameter_94

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_18 = [full_7, slice_6, full_19, full_20]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.reshape: (1x-1x8x32xf32) <- (1x-1x256xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(add_60, stack_15)
        del add_60, stack_15

        # pd_op.matmul: (1x300x192xf32) <- (1x300x256xf32, 256x192xf32)
        matmul_35 = paddle._C_ops.matmul(add_59, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (1x300x192xf32) <- (1x300x192xf32, 192xf32)
        add_61 = paddle._C_ops.add(matmul_35, parameter_92)
        del matmul_35, parameter_92

        # pd_op.reshape: (1x300x8x3x4x2xf32) <- (1x300x192xf32, 6xi64)
        reshape_20 = paddle._C_ops.reshape(add_61, full_int_array_12)
        del add_61

        # pd_op.matmul: (1x300x96xf32) <- (1x300x256xf32, 256x96xf32)
        matmul_36 = paddle._C_ops.matmul(add_59, parameter_91, False, False)
        del add_59, parameter_91

        # pd_op.add: (1x300x96xf32) <- (1x300x96xf32, 96xf32)
        add_62 = paddle._C_ops.add(matmul_36, parameter_90)
        del matmul_36, parameter_90

        # pd_op.reshape: (1x300x8x12xf32) <- (1x300x96xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_62, full_int_array_13)
        del add_62

        # pd_op.softmax: (1x300x8x12xf32) <- (1x300x8x12xf32)
        softmax_3 = paddle._C_ops.softmax(reshape_21, -1)
        del reshape_21

        # pd_op.reshape: (1x300x8x3x4xf32) <- (1x300x8x12xf32, 5xi64)
        reshape_22 = paddle._C_ops.reshape(softmax_3, full_int_array_14)
        del softmax_3

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            unsqueeze_5, [3], full_int_array_6, full_int_array_0, [1], []
        )

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(slice_45, full_int_array_15)
        del slice_45

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(reshape_20, full_21, float("0"), True)
        del reshape_20

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            unsqueeze_5, [3], full_int_array_0, full_int_array_10, [1], []
        )
        del unsqueeze_5

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(slice_46, full_int_array_15)
        del slice_46

        # pd_op.multiply: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1x300x1x1x1x2xf32)
        multiply_11 = paddle._C_ops.multiply(scale_13, unsqueeze_7)
        del scale_13, unsqueeze_7

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(multiply_11, full_13, float("0"), True)
        del multiply_11

        # pd_op.add: (1x300x8x3x4x2xf32) <- (1x300x1x1x1x2xf32, 1x300x8x3x4x2xf32)
        add_63 = paddle._C_ops.add(unsqueeze_6, scale_14)
        del scale_14, unsqueeze_6

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_19 = [slice_0, slice_1]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_19, 0)
        del combine_19

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_6 = stack_16
        del stack_16

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_20 = [slice_2, slice_3]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_20, 0)
        del combine_20

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_7 = stack_17
        del stack_17

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_21 = [slice_4, slice_5]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_18 = paddle._C_ops.stack(combine_21, 0)
        del combine_21

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_8 = stack_18
        del stack_18

        # builtin.combine: ([2xi64, 2xi64, 2xi64]) <- (2xi64, 2xi64, 2xi64)
        combine_22 = [assign_6, assign_7, assign_8]
        del assign_6, assign_7, assign_8

        # pd_op.stack: (3x2xi64) <- ([2xi64, 2xi64, 2xi64])
        stack_19 = paddle._C_ops.stack(combine_22, 0)
        del combine_22

        # pd_op.assign: (3x2xi64) <- (3x2xi64)
        assign_9 = stack_19
        del stack_19

        # pd_op.full: (xi64) <- ()
        full_24 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_24,
            [],
            paddle.int64,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del full_24

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_23 = [assign_value__4, scale_0, add_0]
        del assign_value__4

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_20 = paddle._C_ops.stack(combine_23, 0)
        del combine_23

        # pd_op.assign: (3xi64) <- (3xi64)
        assign_10 = stack_20
        del stack_20

        # pd_op.shape64: (4xi64) <- (1x-1x8x32xf32)
        shape64_10 = paddle._C_ops.shape64(reshape_19)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_10

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            assign_9, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            slice_48, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            slice_48, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_48

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_12 = paddle._C_ops.multiply(slice_49, slice_50)
        del slice_49, slice_50

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            assign_9, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            slice_51, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            slice_51, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_51

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_13 = paddle._C_ops.multiply(slice_52, slice_53)
        del slice_52, slice_53

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            assign_9, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            slice_54, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            slice_54, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_54

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_14 = paddle._C_ops.multiply(slice_55, slice_56)
        del slice_55, slice_56

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_24 = [multiply_12, multiply_13, multiply_14]
        del multiply_12, multiply_13, multiply_14

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_21 = paddle._C_ops.stack(combine_24, 0)
        del combine_24

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x-1x8x32xf32, 3xi64, 1xi32)
        split_10 = paddle._C_ops.split(reshape_19, stack_21, full_1)
        del reshape_19, stack_21

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_11,
            split_12,
            split_13,
        ) = split_10
        del split_10

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(add_63, full_23, float("0"), True)
        del add_63

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(scale_15, full_0, float("-1"), True)
        del scale_15

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(
            assign_9, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(
            slice_57, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(
            slice_57, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_57

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_15 = paddle._C_ops.flatten(split_11, 2, 3)
        del split_11

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_19 = paddle._C_ops.transpose(flatten_15, [0, 2, 1])
        del flatten_15

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_25 = [full_19, full_20, slice_58, slice_59]
        del slice_58, slice_59

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_22 = paddle._C_ops.stack(combine_25, 0)
        del combine_25

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_19, stack_22)
        del stack_22, transpose_19

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(
            scale_16, [3], full_int_array_6, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_20 = paddle._C_ops.transpose(slice_60, [0, 2, 1, 3, 4])
        del slice_60

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_16 = paddle._C_ops.flatten(transpose_20, 0, 1)
        del transpose_20

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_3 = paddle._C_ops.grid_sample(
            reshape_23, flatten_16, "bilinear", "zeros", False
        )
        del flatten_16, reshape_23

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(
            assign_9, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(
            slice_61, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(
            slice_61, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_61

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_17 = paddle._C_ops.flatten(split_12, 2, 3)
        del split_12

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_21 = paddle._C_ops.transpose(flatten_17, [0, 2, 1])
        del flatten_17

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_26 = [full_19, full_20, slice_62, slice_63]
        del slice_62, slice_63

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_23 = paddle._C_ops.stack(combine_26, 0)
        del combine_26

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(transpose_21, stack_23)
        del stack_23, transpose_21

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(
            scale_16, [3], full_int_array_3, full_int_array_0, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_22 = paddle._C_ops.transpose(slice_64, [0, 2, 1, 3, 4])
        del slice_64

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_18 = paddle._C_ops.flatten(transpose_22, 0, 1)
        del transpose_22

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_4 = paddle._C_ops.grid_sample(
            reshape_24, flatten_18, "bilinear", "zeros", False
        )
        del flatten_18, reshape_24

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(
            assign_9, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del assign_9

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(
            slice_65, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(
            slice_65, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_65

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_19 = paddle._C_ops.flatten(split_13, 2, 3)
        del split_13

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_23 = paddle._C_ops.transpose(flatten_19, [0, 2, 1])
        del flatten_19

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_27 = [full_19, full_20, slice_66, slice_67]
        del slice_66, slice_67

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_24 = paddle._C_ops.stack(combine_27, 0)
        del combine_27

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_23, stack_24)
        del stack_24, transpose_23

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(
            scale_16, [3], full_int_array_0, full_int_array_1, [1], [3]
        )
        del scale_16

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_24 = paddle._C_ops.transpose(slice_68, [0, 2, 1, 3, 4])
        del slice_68

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_20 = paddle._C_ops.flatten(transpose_24, 0, 1)
        del transpose_24

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_5 = paddle._C_ops.grid_sample(
            reshape_25, flatten_20, "bilinear", "zeros", False
        )
        del flatten_20, reshape_25

        # pd_op.transpose: (1x8x300x3x4xf32) <- (1x300x8x3x4xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3, 4])
        del reshape_22

        # pd_op.reshape: (8x1x300x12xf32) <- (1x8x300x3x4xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(transpose_25, full_int_array_16)
        del transpose_25

        # builtin.combine: ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32]) <- (8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32)
        combine_28 = [grid_sample_3, grid_sample_4, grid_sample_5]
        del grid_sample_3, grid_sample_4, grid_sample_5

        # pd_op.stack: (8x32x300x3x4xf32) <- ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32])
        stack_25 = paddle._C_ops.stack(combine_28, -2)
        del combine_28

        # pd_op.flatten: (8x32x300x12xf32) <- (8x32x300x3x4xf32)
        flatten_21 = paddle._C_ops.flatten(stack_25, 3, 4)
        del stack_25

        # pd_op.multiply: (8x32x300x12xf32) <- (8x32x300x12xf32, 8x1x300x12xf32)
        multiply_15 = paddle._C_ops.multiply(flatten_21, reshape_26)
        del flatten_21, reshape_26

        # pd_op.sum: (8x32x300xf32) <- (8x32x300x12xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(multiply_15, full_int_array_4, None, False)
        del multiply_15

        # pd_op.reshape: (1x256x300xf32) <- (8x32x300xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(sum_1, full_int_array_17)
        del sum_1

        # pd_op.transpose: (1x300x256xf32) <- (1x256x300xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_27, [0, 2, 1])
        del reshape_27

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_37 = paddle._C_ops.matmul(transpose_26, parameter_89, False, False)
        del parameter_89, transpose_26

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_64 = paddle._C_ops.add(matmul_37, parameter_88)
        del matmul_37, parameter_88

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_65 = paddle._C_ops.add(layer_norm_15, add_64)
        del add_64, layer_norm_15

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_65, parameter_87, parameter_86, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_65, parameter_86, parameter_87

        # pd_op.matmul: (1x300x1024xf32) <- (1x300x256xf32, 256x1024xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_18, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (1x300x1024xf32) <- (1x300x1024xf32, 1024xf32)
        add_66 = paddle._C_ops.add(matmul_38, parameter_84)
        del matmul_38, parameter_84

        # pd_op.relu: (1x300x1024xf32) <- (1x300x1024xf32)
        relu_9 = paddle._C_ops.relu(add_66)
        del add_66

        # pd_op.matmul: (1x300x256xf32) <- (1x300x1024xf32, 1024x256xf32)
        matmul_39 = paddle._C_ops.matmul(relu_9, parameter_83, False, False)
        del parameter_83, relu_9

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_67 = paddle._C_ops.add(matmul_39, parameter_82)
        del matmul_39, parameter_82

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_68 = paddle._C_ops.add(layer_norm_18, add_67)
        del add_67, layer_norm_18

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_68, parameter_81, parameter_80, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_68, parameter_80, parameter_81

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_21, parameter_137, False, False)

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_69 = paddle._C_ops.add(matmul_40, parameter_136)
        del matmul_40

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_10 = paddle._C_ops.relu(add_69)
        del add_69

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_41 = paddle._C_ops.matmul(relu_10, parameter_135, False, False)
        del relu_10

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_70 = paddle._C_ops.add(matmul_41, parameter_134)
        del matmul_41

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_11 = paddle._C_ops.relu(add_70)
        del add_70

        # pd_op.matmul: (1x300x4xf32) <- (1x300x256xf32, 256x4xf32)
        matmul_42 = paddle._C_ops.matmul(relu_11, parameter_133, False, False)
        del relu_11

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        add_71 = paddle._C_ops.add(matmul_42, parameter_132)
        del matmul_42

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_6 = paddle._C_ops.clip(sigmoid_2, full_3, full_0)
        del sigmoid_2

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_7 = paddle._C_ops.clip(clip_6, full_15, full_16)

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(clip_6, full_17, float("1"), True)
        del clip_6

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_8 = paddle._C_ops.clip(scale_17, full_15, full_16)
        del scale_17

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        divide_3 = paddle._C_ops.divide(clip_7, clip_8)
        del clip_7, clip_8

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        log_2 = paddle._C_ops.log(divide_3)
        del divide_3

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        add_72 = paddle._C_ops.add(add_71, log_2)
        del add_71, log_2

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(add_72)
        del add_72

        # pd_op.unsqueeze: (1x300x1x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(sigmoid_3, full_int_array_0)

        # pd_op.matmul: (1x300x512xf32) <- (1x300x4xf32, 4x512xf32)
        matmul_43 = paddle._C_ops.matmul(sigmoid_3, parameter_123, False, False)

        # pd_op.add: (1x300x512xf32) <- (1x300x512xf32, 512xf32)
        add_73 = paddle._C_ops.add(matmul_43, parameter_122)
        del matmul_43

        # pd_op.relu: (1x300x512xf32) <- (1x300x512xf32)
        relu_12 = paddle._C_ops.relu(add_73)
        del add_73

        # pd_op.matmul: (1x300x256xf32) <- (1x300x512xf32, 512x256xf32)
        matmul_44 = paddle._C_ops.matmul(relu_12, parameter_121, False, False)
        del relu_12

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_74 = paddle._C_ops.add(matmul_44, parameter_120)
        del matmul_44

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_75 = paddle._C_ops.add(layer_norm_21, add_74)

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(
            data_4, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(
            data_5, [0], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_45 = paddle._C_ops.matmul(add_75, slice_69, False, False)
        del slice_69

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_76 = paddle._C_ops.add(matmul_45, slice_70)
        del matmul_45, slice_70

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_76, full_int_array_8)
        del add_76

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(
            data_4, [1], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(
            data_5, [0], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_46 = paddle._C_ops.matmul(add_75, slice_71, False, False)
        del add_75, slice_71

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_77 = paddle._C_ops.add(matmul_46, slice_72)
        del matmul_46, slice_72

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_77, full_int_array_8)
        del add_77

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(
            data_4, [1], full_int_array_9, full_int_array_10, [1], []
        )
        del data_4

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(
            data_5, [0], full_int_array_9, full_int_array_10, [1], []
        )
        del data_5

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_21, slice_73, False, False)
        del slice_73

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_78 = paddle._C_ops.add(matmul_47, slice_74)
        del matmul_47, slice_74

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_78, full_int_array_8)
        del add_78

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.matmul: (1x8x300x300xf32) <- (1x8x300x32xf32, 1x8x300x32xf32)
        matmul_48 = paddle._C_ops.matmul(transpose_27, transpose_28, False, True)
        del transpose_27, transpose_28

        # pd_op.scale: (1x8x300x300xf32) <- (1x8x300x300xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(matmul_48, full_18, float("0"), True)
        del matmul_48

        # pd_op.softmax: (1x8x300x300xf32) <- (1x8x300x300xf32)
        softmax_4 = paddle._C_ops.softmax(scale_18, -1)
        del scale_18

        # pd_op.matmul: (1x8x300x32xf32) <- (1x8x300x300xf32, 1x8x300x32xf32)
        matmul_49 = paddle._C_ops.matmul(softmax_4, transpose_29, False, False)
        del softmax_4, transpose_29

        # pd_op.transpose: (1x300x8x32xf32) <- (1x8x300x32xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_49, [0, 2, 1, 3])
        del matmul_49

        # pd_op.reshape: (1x300x256xf32) <- (1x300x8x32xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_30, full_int_array_11)
        del transpose_30

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_50 = paddle._C_ops.matmul(reshape_31, parameter_79, False, False)
        del parameter_79, reshape_31

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_79 = paddle._C_ops.add(matmul_50, parameter_78)
        del matmul_50, parameter_78

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_80 = paddle._C_ops.add(layer_norm_21, add_79)
        del add_79, layer_norm_21

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_80, parameter_77, parameter_76, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_80, parameter_76, parameter_77

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_81 = paddle._C_ops.add(layer_norm_24, add_74)
        del add_74

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x256xf32, 256x256xf32)
        matmul_51 = paddle._C_ops.matmul(concat_0, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_82 = paddle._C_ops.add(matmul_51, parameter_74)
        del matmul_51, parameter_74

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_29 = [full_7, slice_6, full_19, full_20]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_26 = paddle._C_ops.stack(combine_29, 0)
        del combine_29

        # pd_op.reshape: (1x-1x8x32xf32) <- (1x-1x256xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_82, stack_26)
        del add_82, stack_26

        # pd_op.matmul: (1x300x192xf32) <- (1x300x256xf32, 256x192xf32)
        matmul_52 = paddle._C_ops.matmul(add_81, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x300x192xf32) <- (1x300x192xf32, 192xf32)
        add_83 = paddle._C_ops.add(matmul_52, parameter_72)
        del matmul_52, parameter_72

        # pd_op.reshape: (1x300x8x3x4x2xf32) <- (1x300x192xf32, 6xi64)
        reshape_33 = paddle._C_ops.reshape(add_83, full_int_array_12)
        del add_83

        # pd_op.matmul: (1x300x96xf32) <- (1x300x256xf32, 256x96xf32)
        matmul_53 = paddle._C_ops.matmul(add_81, parameter_71, False, False)
        del add_81, parameter_71

        # pd_op.add: (1x300x96xf32) <- (1x300x96xf32, 96xf32)
        add_84 = paddle._C_ops.add(matmul_53, parameter_70)
        del matmul_53, parameter_70

        # pd_op.reshape: (1x300x8x12xf32) <- (1x300x96xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_84, full_int_array_13)
        del add_84

        # pd_op.softmax: (1x300x8x12xf32) <- (1x300x8x12xf32)
        softmax_5 = paddle._C_ops.softmax(reshape_34, -1)
        del reshape_34

        # pd_op.reshape: (1x300x8x3x4xf32) <- (1x300x8x12xf32, 5xi64)
        reshape_35 = paddle._C_ops.reshape(softmax_5, full_int_array_14)
        del softmax_5

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(
            unsqueeze_8, [3], full_int_array_6, full_int_array_0, [1], []
        )

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(slice_75, full_int_array_15)
        del slice_75

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(reshape_33, full_21, float("0"), True)
        del reshape_33

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(
            unsqueeze_8, [3], full_int_array_0, full_int_array_10, [1], []
        )
        del unsqueeze_8

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(slice_76, full_int_array_15)
        del slice_76

        # pd_op.multiply: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1x300x1x1x1x2xf32)
        multiply_16 = paddle._C_ops.multiply(scale_19, unsqueeze_10)
        del scale_19, unsqueeze_10

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(multiply_16, full_13, float("0"), True)
        del multiply_16

        # pd_op.add: (1x300x8x3x4x2xf32) <- (1x300x1x1x1x2xf32, 1x300x8x3x4x2xf32)
        add_85 = paddle._C_ops.add(unsqueeze_9, scale_20)
        del scale_20, unsqueeze_9

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_30 = [slice_0, slice_1]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_27 = paddle._C_ops.stack(combine_30, 0)
        del combine_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_11 = stack_27
        del stack_27

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_31 = [slice_2, slice_3]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_28 = paddle._C_ops.stack(combine_31, 0)
        del combine_31

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_12 = stack_28
        del stack_28

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_32 = [slice_4, slice_5]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_29 = paddle._C_ops.stack(combine_32, 0)
        del combine_32

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_13 = stack_29
        del stack_29

        # builtin.combine: ([2xi64, 2xi64, 2xi64]) <- (2xi64, 2xi64, 2xi64)
        combine_33 = [assign_11, assign_12, assign_13]
        del assign_11, assign_12, assign_13

        # pd_op.stack: (3x2xi64) <- ([2xi64, 2xi64, 2xi64])
        stack_30 = paddle._C_ops.stack(combine_33, 0)
        del combine_33

        # pd_op.assign: (3x2xi64) <- (3x2xi64)
        assign_14 = stack_30
        del stack_30

        # pd_op.full: (xi64) <- ()
        full_25 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_25,
            [],
            paddle.int64,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del full_25

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_34 = [assign_value__5, scale_0, add_0]
        del assign_value__5

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_31 = paddle._C_ops.stack(combine_34, 0)
        del combine_34

        # pd_op.assign: (3xi64) <- (3xi64)
        assign_15 = stack_31
        del stack_31

        # pd_op.shape64: (4xi64) <- (1x-1x8x32xf32)
        shape64_11 = paddle._C_ops.shape64(reshape_32)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_11

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(
            assign_14, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(
            slice_78, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(
            slice_78, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_78

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_17 = paddle._C_ops.multiply(slice_79, slice_80)
        del slice_79, slice_80

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(
            assign_14, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(
            slice_81, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(
            slice_81, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_81

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_18 = paddle._C_ops.multiply(slice_82, slice_83)
        del slice_82, slice_83

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(
            assign_14, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(
            slice_84, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(
            slice_84, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_84

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_19 = paddle._C_ops.multiply(slice_85, slice_86)
        del slice_85, slice_86

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_35 = [multiply_17, multiply_18, multiply_19]
        del multiply_17, multiply_18, multiply_19

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_32 = paddle._C_ops.stack(combine_35, 0)
        del combine_35

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x-1x8x32xf32, 3xi64, 1xi32)
        split_14 = paddle._C_ops.split(reshape_32, stack_32, full_1)
        del reshape_32, stack_32

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_15,
            split_16,
            split_17,
        ) = split_14
        del split_14

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(add_85, full_23, float("0"), True)
        del add_85

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(scale_21, full_0, float("-1"), True)
        del scale_21

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(
            assign_14, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(
            slice_87, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(
            slice_87, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_87

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_22 = paddle._C_ops.flatten(split_15, 2, 3)
        del split_15

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_31 = paddle._C_ops.transpose(flatten_22, [0, 2, 1])
        del flatten_22

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_36 = [full_19, full_20, slice_88, slice_89]
        del slice_88, slice_89

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_33 = paddle._C_ops.stack(combine_36, 0)
        del combine_36

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(transpose_31, stack_33)
        del stack_33, transpose_31

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(
            scale_22, [3], full_int_array_6, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_32 = paddle._C_ops.transpose(slice_90, [0, 2, 1, 3, 4])
        del slice_90

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_23 = paddle._C_ops.flatten(transpose_32, 0, 1)
        del transpose_32

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_6 = paddle._C_ops.grid_sample(
            reshape_36, flatten_23, "bilinear", "zeros", False
        )
        del flatten_23, reshape_36

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(
            assign_14, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(
            slice_91, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(
            slice_91, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_91

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_24 = paddle._C_ops.flatten(split_16, 2, 3)
        del split_16

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_33 = paddle._C_ops.transpose(flatten_24, [0, 2, 1])
        del flatten_24

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_37 = [full_19, full_20, slice_92, slice_93]
        del slice_92, slice_93

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_34 = paddle._C_ops.stack(combine_37, 0)
        del combine_37

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(transpose_33, stack_34)
        del stack_34, transpose_33

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(
            scale_22, [3], full_int_array_3, full_int_array_0, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_34 = paddle._C_ops.transpose(slice_94, [0, 2, 1, 3, 4])
        del slice_94

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_25 = paddle._C_ops.flatten(transpose_34, 0, 1)
        del transpose_34

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_7 = paddle._C_ops.grid_sample(
            reshape_37, flatten_25, "bilinear", "zeros", False
        )
        del flatten_25, reshape_37

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(
            assign_14, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del assign_14

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(
            slice_95, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(
            slice_95, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_95

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_26 = paddle._C_ops.flatten(split_17, 2, 3)
        del split_17

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_35 = paddle._C_ops.transpose(flatten_26, [0, 2, 1])
        del flatten_26

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_38 = [full_19, full_20, slice_96, slice_97]
        del slice_96, slice_97

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_35 = paddle._C_ops.stack(combine_38, 0)
        del combine_38

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(transpose_35, stack_35)
        del stack_35, transpose_35

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(
            scale_22, [3], full_int_array_0, full_int_array_1, [1], [3]
        )
        del scale_22

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_36 = paddle._C_ops.transpose(slice_98, [0, 2, 1, 3, 4])
        del slice_98

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_27 = paddle._C_ops.flatten(transpose_36, 0, 1)
        del transpose_36

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_8 = paddle._C_ops.grid_sample(
            reshape_38, flatten_27, "bilinear", "zeros", False
        )
        del flatten_27, reshape_38

        # pd_op.transpose: (1x8x300x3x4xf32) <- (1x300x8x3x4xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_35, [0, 2, 1, 3, 4])
        del reshape_35

        # pd_op.reshape: (8x1x300x12xf32) <- (1x8x300x3x4xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_37, full_int_array_16)
        del transpose_37

        # builtin.combine: ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32]) <- (8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32)
        combine_39 = [grid_sample_6, grid_sample_7, grid_sample_8]
        del grid_sample_6, grid_sample_7, grid_sample_8

        # pd_op.stack: (8x32x300x3x4xf32) <- ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32])
        stack_36 = paddle._C_ops.stack(combine_39, -2)
        del combine_39

        # pd_op.flatten: (8x32x300x12xf32) <- (8x32x300x3x4xf32)
        flatten_28 = paddle._C_ops.flatten(stack_36, 3, 4)
        del stack_36

        # pd_op.multiply: (8x32x300x12xf32) <- (8x32x300x12xf32, 8x1x300x12xf32)
        multiply_20 = paddle._C_ops.multiply(flatten_28, reshape_39)
        del flatten_28, reshape_39

        # pd_op.sum: (8x32x300xf32) <- (8x32x300x12xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(multiply_20, full_int_array_4, None, False)
        del multiply_20

        # pd_op.reshape: (1x256x300xf32) <- (8x32x300xf32, 3xi64)
        reshape_40 = paddle._C_ops.reshape(sum_2, full_int_array_17)
        del sum_2

        # pd_op.transpose: (1x300x256xf32) <- (1x256x300xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_40, [0, 2, 1])
        del reshape_40

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_54 = paddle._C_ops.matmul(transpose_38, parameter_69, False, False)
        del parameter_69, transpose_38

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_86 = paddle._C_ops.add(matmul_54, parameter_68)
        del matmul_54, parameter_68

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_87 = paddle._C_ops.add(layer_norm_24, add_86)
        del add_86, layer_norm_24

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_87, parameter_67, parameter_66, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_87, parameter_66, parameter_67

        # pd_op.matmul: (1x300x1024xf32) <- (1x300x256xf32, 256x1024xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_27, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x300x1024xf32) <- (1x300x1024xf32, 1024xf32)
        add_88 = paddle._C_ops.add(matmul_55, parameter_64)
        del matmul_55, parameter_64

        # pd_op.relu: (1x300x1024xf32) <- (1x300x1024xf32)
        relu_13 = paddle._C_ops.relu(add_88)
        del add_88

        # pd_op.matmul: (1x300x256xf32) <- (1x300x1024xf32, 1024x256xf32)
        matmul_56 = paddle._C_ops.matmul(relu_13, parameter_63, False, False)
        del parameter_63, relu_13

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_89 = paddle._C_ops.add(matmul_56, parameter_62)
        del matmul_56, parameter_62

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_90 = paddle._C_ops.add(layer_norm_27, add_89)
        del add_89, layer_norm_27

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_90, parameter_61, parameter_60, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_90, parameter_60, parameter_61

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_57 = paddle._C_ops.matmul(layer_norm_30, parameter_137, False, False)

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_91 = paddle._C_ops.add(matmul_57, parameter_136)
        del matmul_57

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_14 = paddle._C_ops.relu(add_91)
        del add_91

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_58 = paddle._C_ops.matmul(relu_14, parameter_135, False, False)
        del relu_14

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_92 = paddle._C_ops.add(matmul_58, parameter_134)
        del matmul_58

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_15 = paddle._C_ops.relu(add_92)
        del add_92

        # pd_op.matmul: (1x300x4xf32) <- (1x300x256xf32, 256x4xf32)
        matmul_59 = paddle._C_ops.matmul(relu_15, parameter_133, False, False)
        del relu_15

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        add_93 = paddle._C_ops.add(matmul_59, parameter_132)
        del matmul_59

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_9 = paddle._C_ops.clip(sigmoid_3, full_3, full_0)
        del sigmoid_3

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_10 = paddle._C_ops.clip(clip_9, full_15, full_16)

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(clip_9, full_17, float("1"), True)
        del clip_9

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_11 = paddle._C_ops.clip(scale_23, full_15, full_16)
        del scale_23

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        divide_4 = paddle._C_ops.divide(clip_10, clip_11)
        del clip_10, clip_11

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        log_3 = paddle._C_ops.log(divide_4)
        del divide_4

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        add_94 = paddle._C_ops.add(add_93, log_3)
        del add_93, log_3

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(add_94)
        del add_94

        # pd_op.unsqueeze: (1x300x1x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(sigmoid_4, full_int_array_0)

        # pd_op.matmul: (1x300x512xf32) <- (1x300x4xf32, 4x512xf32)
        matmul_60 = paddle._C_ops.matmul(sigmoid_4, parameter_123, False, False)

        # pd_op.add: (1x300x512xf32) <- (1x300x512xf32, 512xf32)
        add_95 = paddle._C_ops.add(matmul_60, parameter_122)
        del matmul_60

        # pd_op.relu: (1x300x512xf32) <- (1x300x512xf32)
        relu_16 = paddle._C_ops.relu(add_95)
        del add_95

        # pd_op.matmul: (1x300x256xf32) <- (1x300x512xf32, 512x256xf32)
        matmul_61 = paddle._C_ops.matmul(relu_16, parameter_121, False, False)
        del relu_16

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_96 = paddle._C_ops.add(matmul_61, parameter_120)
        del matmul_61

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_97 = paddle._C_ops.add(layer_norm_30, add_96)

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(
            data_6, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_100 = paddle._C_ops.slice(
            data_7, [0], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_62 = paddle._C_ops.matmul(add_97, slice_99, False, False)
        del slice_99

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_98 = paddle._C_ops.add(matmul_62, slice_100)
        del matmul_62, slice_100

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_98, full_int_array_8)
        del add_98

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_39 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_101 = paddle._C_ops.slice(
            data_6, [1], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_102 = paddle._C_ops.slice(
            data_7, [0], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_63 = paddle._C_ops.matmul(add_97, slice_101, False, False)
        del add_97, slice_101

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_99 = paddle._C_ops.add(matmul_63, slice_102)
        del matmul_63, slice_102

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_99, full_int_array_8)
        del add_99

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_103 = paddle._C_ops.slice(
            data_6, [1], full_int_array_9, full_int_array_10, [1], []
        )
        del data_6

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(
            data_7, [0], full_int_array_9, full_int_array_10, [1], []
        )
        del data_7

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_30, slice_103, False, False)
        del slice_103

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_100 = paddle._C_ops.add(matmul_64, slice_104)
        del matmul_64, slice_104

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(add_100, full_int_array_8)
        del add_100

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_43, [0, 2, 1, 3])
        del reshape_43

        # pd_op.matmul: (1x8x300x300xf32) <- (1x8x300x32xf32, 1x8x300x32xf32)
        matmul_65 = paddle._C_ops.matmul(transpose_39, transpose_40, False, True)
        del transpose_39, transpose_40

        # pd_op.scale: (1x8x300x300xf32) <- (1x8x300x300xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(matmul_65, full_18, float("0"), True)
        del matmul_65

        # pd_op.softmax: (1x8x300x300xf32) <- (1x8x300x300xf32)
        softmax_6 = paddle._C_ops.softmax(scale_24, -1)
        del scale_24

        # pd_op.matmul: (1x8x300x32xf32) <- (1x8x300x300xf32, 1x8x300x32xf32)
        matmul_66 = paddle._C_ops.matmul(softmax_6, transpose_41, False, False)
        del softmax_6, transpose_41

        # pd_op.transpose: (1x300x8x32xf32) <- (1x8x300x32xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_66, [0, 2, 1, 3])
        del matmul_66

        # pd_op.reshape: (1x300x256xf32) <- (1x300x8x32xf32, 3xi64)
        reshape_44 = paddle._C_ops.reshape(transpose_42, full_int_array_11)
        del transpose_42

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_67 = paddle._C_ops.matmul(reshape_44, parameter_59, False, False)
        del parameter_59, reshape_44

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_101 = paddle._C_ops.add(matmul_67, parameter_58)
        del matmul_67, parameter_58

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_102 = paddle._C_ops.add(layer_norm_30, add_101)
        del add_101, layer_norm_30

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_102, parameter_57, parameter_56, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_102, parameter_56, parameter_57

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_103 = paddle._C_ops.add(layer_norm_33, add_96)
        del add_96

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x256xf32, 256x256xf32)
        matmul_68 = paddle._C_ops.matmul(concat_0, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_104 = paddle._C_ops.add(matmul_68, parameter_54)
        del matmul_68, parameter_54

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_40 = [full_7, slice_6, full_19, full_20]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_37 = paddle._C_ops.stack(combine_40, 0)
        del combine_40

        # pd_op.reshape: (1x-1x8x32xf32) <- (1x-1x256xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_104, stack_37)
        del add_104, stack_37

        # pd_op.matmul: (1x300x192xf32) <- (1x300x256xf32, 256x192xf32)
        matmul_69 = paddle._C_ops.matmul(add_103, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (1x300x192xf32) <- (1x300x192xf32, 192xf32)
        add_105 = paddle._C_ops.add(matmul_69, parameter_52)
        del matmul_69, parameter_52

        # pd_op.reshape: (1x300x8x3x4x2xf32) <- (1x300x192xf32, 6xi64)
        reshape_46 = paddle._C_ops.reshape(add_105, full_int_array_12)
        del add_105

        # pd_op.matmul: (1x300x96xf32) <- (1x300x256xf32, 256x96xf32)
        matmul_70 = paddle._C_ops.matmul(add_103, parameter_51, False, False)
        del add_103, parameter_51

        # pd_op.add: (1x300x96xf32) <- (1x300x96xf32, 96xf32)
        add_106 = paddle._C_ops.add(matmul_70, parameter_50)
        del matmul_70, parameter_50

        # pd_op.reshape: (1x300x8x12xf32) <- (1x300x96xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(add_106, full_int_array_13)
        del add_106

        # pd_op.softmax: (1x300x8x12xf32) <- (1x300x8x12xf32)
        softmax_7 = paddle._C_ops.softmax(reshape_47, -1)
        del reshape_47

        # pd_op.reshape: (1x300x8x3x4xf32) <- (1x300x8x12xf32, 5xi64)
        reshape_48 = paddle._C_ops.reshape(softmax_7, full_int_array_14)
        del softmax_7

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(
            unsqueeze_11, [3], full_int_array_6, full_int_array_0, [1], []
        )

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(slice_105, full_int_array_15)
        del slice_105

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(reshape_46, full_21, float("0"), True)
        del reshape_46

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(
            unsqueeze_11, [3], full_int_array_0, full_int_array_10, [1], []
        )
        del unsqueeze_11

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(slice_106, full_int_array_15)
        del slice_106

        # pd_op.multiply: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1x300x1x1x1x2xf32)
        multiply_21 = paddle._C_ops.multiply(scale_25, unsqueeze_13)
        del scale_25, unsqueeze_13

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(multiply_21, full_13, float("0"), True)
        del multiply_21

        # pd_op.add: (1x300x8x3x4x2xf32) <- (1x300x1x1x1x2xf32, 1x300x8x3x4x2xf32)
        add_107 = paddle._C_ops.add(unsqueeze_12, scale_26)
        del scale_26, unsqueeze_12

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_41 = [slice_0, slice_1]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_38 = paddle._C_ops.stack(combine_41, 0)
        del combine_41

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_16 = stack_38
        del stack_38

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_42 = [slice_2, slice_3]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_39 = paddle._C_ops.stack(combine_42, 0)
        del combine_42

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_17 = stack_39
        del stack_39

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_43 = [slice_4, slice_5]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_40 = paddle._C_ops.stack(combine_43, 0)
        del combine_43

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_18 = stack_40
        del stack_40

        # builtin.combine: ([2xi64, 2xi64, 2xi64]) <- (2xi64, 2xi64, 2xi64)
        combine_44 = [assign_16, assign_17, assign_18]
        del assign_16, assign_17, assign_18

        # pd_op.stack: (3x2xi64) <- ([2xi64, 2xi64, 2xi64])
        stack_41 = paddle._C_ops.stack(combine_44, 0)
        del combine_44

        # pd_op.assign: (3x2xi64) <- (3x2xi64)
        assign_19 = stack_41
        del stack_41

        # pd_op.full: (xi64) <- ()
        full_26 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_26,
            [],
            paddle.int64,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del full_26

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_45 = [assign_value__6, scale_0, add_0]
        del assign_value__6

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_42 = paddle._C_ops.stack(combine_45, 0)
        del combine_45

        # pd_op.assign: (3xi64) <- (3xi64)
        assign_20 = stack_42
        del stack_42

        # pd_op.shape64: (4xi64) <- (1x-1x8x32xf32)
        shape64_12 = paddle._C_ops.shape64(reshape_45)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_12

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_108 = paddle._C_ops.slice(
            assign_19, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_109 = paddle._C_ops.slice(
            slice_108, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_110 = paddle._C_ops.slice(
            slice_108, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_108

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_22 = paddle._C_ops.multiply(slice_109, slice_110)
        del slice_109, slice_110

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_111 = paddle._C_ops.slice(
            assign_19, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(
            slice_111, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_113 = paddle._C_ops.slice(
            slice_111, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_111

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_23 = paddle._C_ops.multiply(slice_112, slice_113)
        del slice_112, slice_113

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_114 = paddle._C_ops.slice(
            assign_19, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_115 = paddle._C_ops.slice(
            slice_114, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_116 = paddle._C_ops.slice(
            slice_114, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_114

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_24 = paddle._C_ops.multiply(slice_115, slice_116)
        del slice_115, slice_116

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_46 = [multiply_22, multiply_23, multiply_24]
        del multiply_22, multiply_23, multiply_24

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_43 = paddle._C_ops.stack(combine_46, 0)
        del combine_46

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x-1x8x32xf32, 3xi64, 1xi32)
        split_18 = paddle._C_ops.split(reshape_45, stack_43, full_1)
        del reshape_45, stack_43

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_19,
            split_20,
            split_21,
        ) = split_18
        del split_18

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(add_107, full_23, float("0"), True)
        del add_107

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(scale_27, full_0, float("-1"), True)
        del scale_27

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_117 = paddle._C_ops.slice(
            assign_19, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_118 = paddle._C_ops.slice(
            slice_117, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(
            slice_117, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_117

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_29 = paddle._C_ops.flatten(split_19, 2, 3)
        del split_19

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_43 = paddle._C_ops.transpose(flatten_29, [0, 2, 1])
        del flatten_29

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_47 = [full_19, full_20, slice_118, slice_119]
        del slice_118, slice_119

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_44 = paddle._C_ops.stack(combine_47, 0)
        del combine_47

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(transpose_43, stack_44)
        del stack_44, transpose_43

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_120 = paddle._C_ops.slice(
            scale_28, [3], full_int_array_6, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_44 = paddle._C_ops.transpose(slice_120, [0, 2, 1, 3, 4])
        del slice_120

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_30 = paddle._C_ops.flatten(transpose_44, 0, 1)
        del transpose_44

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_9 = paddle._C_ops.grid_sample(
            reshape_49, flatten_30, "bilinear", "zeros", False
        )
        del flatten_30, reshape_49

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_121 = paddle._C_ops.slice(
            assign_19, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(
            slice_121, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_123 = paddle._C_ops.slice(
            slice_121, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_121

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_31 = paddle._C_ops.flatten(split_20, 2, 3)
        del split_20

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_45 = paddle._C_ops.transpose(flatten_31, [0, 2, 1])
        del flatten_31

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_48 = [full_19, full_20, slice_122, slice_123]
        del slice_122, slice_123

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_45 = paddle._C_ops.stack(combine_48, 0)
        del combine_48

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(transpose_45, stack_45)
        del stack_45, transpose_45

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_124 = paddle._C_ops.slice(
            scale_28, [3], full_int_array_3, full_int_array_0, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_46 = paddle._C_ops.transpose(slice_124, [0, 2, 1, 3, 4])
        del slice_124

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_32 = paddle._C_ops.flatten(transpose_46, 0, 1)
        del transpose_46

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_10 = paddle._C_ops.grid_sample(
            reshape_50, flatten_32, "bilinear", "zeros", False
        )
        del flatten_32, reshape_50

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_125 = paddle._C_ops.slice(
            assign_19, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del assign_19

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_126 = paddle._C_ops.slice(
            slice_125, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_127 = paddle._C_ops.slice(
            slice_125, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_125

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_33 = paddle._C_ops.flatten(split_21, 2, 3)
        del split_21

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_47 = paddle._C_ops.transpose(flatten_33, [0, 2, 1])
        del flatten_33

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_49 = [full_19, full_20, slice_126, slice_127]
        del slice_126, slice_127

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_46 = paddle._C_ops.stack(combine_49, 0)
        del combine_49

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(transpose_47, stack_46)
        del stack_46, transpose_47

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_128 = paddle._C_ops.slice(
            scale_28, [3], full_int_array_0, full_int_array_1, [1], [3]
        )
        del scale_28

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_48 = paddle._C_ops.transpose(slice_128, [0, 2, 1, 3, 4])
        del slice_128

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_34 = paddle._C_ops.flatten(transpose_48, 0, 1)
        del transpose_48

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_11 = paddle._C_ops.grid_sample(
            reshape_51, flatten_34, "bilinear", "zeros", False
        )
        del flatten_34, reshape_51

        # pd_op.transpose: (1x8x300x3x4xf32) <- (1x300x8x3x4xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_48, [0, 2, 1, 3, 4])
        del reshape_48

        # pd_op.reshape: (8x1x300x12xf32) <- (1x8x300x3x4xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(transpose_49, full_int_array_16)
        del transpose_49

        # builtin.combine: ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32]) <- (8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32)
        combine_50 = [grid_sample_9, grid_sample_10, grid_sample_11]
        del grid_sample_10, grid_sample_11, grid_sample_9

        # pd_op.stack: (8x32x300x3x4xf32) <- ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32])
        stack_47 = paddle._C_ops.stack(combine_50, -2)
        del combine_50

        # pd_op.flatten: (8x32x300x12xf32) <- (8x32x300x3x4xf32)
        flatten_35 = paddle._C_ops.flatten(stack_47, 3, 4)
        del stack_47

        # pd_op.multiply: (8x32x300x12xf32) <- (8x32x300x12xf32, 8x1x300x12xf32)
        multiply_25 = paddle._C_ops.multiply(flatten_35, reshape_52)
        del flatten_35, reshape_52

        # pd_op.sum: (8x32x300xf32) <- (8x32x300x12xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(multiply_25, full_int_array_4, None, False)
        del multiply_25

        # pd_op.reshape: (1x256x300xf32) <- (8x32x300xf32, 3xi64)
        reshape_53 = paddle._C_ops.reshape(sum_3, full_int_array_17)
        del sum_3

        # pd_op.transpose: (1x300x256xf32) <- (1x256x300xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_53, [0, 2, 1])
        del reshape_53

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_71 = paddle._C_ops.matmul(transpose_50, parameter_49, False, False)
        del parameter_49, transpose_50

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_108 = paddle._C_ops.add(matmul_71, parameter_48)
        del matmul_71, parameter_48

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_109 = paddle._C_ops.add(layer_norm_33, add_108)
        del add_108, layer_norm_33

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_109, parameter_47, parameter_46, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_109, parameter_46, parameter_47

        # pd_op.matmul: (1x300x1024xf32) <- (1x300x256xf32, 256x1024xf32)
        matmul_72 = paddle._C_ops.matmul(layer_norm_36, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x300x1024xf32) <- (1x300x1024xf32, 1024xf32)
        add_110 = paddle._C_ops.add(matmul_72, parameter_44)
        del matmul_72, parameter_44

        # pd_op.relu: (1x300x1024xf32) <- (1x300x1024xf32)
        relu_17 = paddle._C_ops.relu(add_110)
        del add_110

        # pd_op.matmul: (1x300x256xf32) <- (1x300x1024xf32, 1024x256xf32)
        matmul_73 = paddle._C_ops.matmul(relu_17, parameter_43, False, False)
        del parameter_43, relu_17

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_111 = paddle._C_ops.add(matmul_73, parameter_42)
        del matmul_73, parameter_42

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_112 = paddle._C_ops.add(layer_norm_36, add_111)
        del add_111, layer_norm_36

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_112, parameter_41, parameter_40, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_112, parameter_40, parameter_41

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_39, parameter_137, False, False)

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_113 = paddle._C_ops.add(matmul_74, parameter_136)
        del matmul_74

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_18 = paddle._C_ops.relu(add_113)
        del add_113

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_75 = paddle._C_ops.matmul(relu_18, parameter_135, False, False)
        del relu_18

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_114 = paddle._C_ops.add(matmul_75, parameter_134)
        del matmul_75

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_19 = paddle._C_ops.relu(add_114)
        del add_114

        # pd_op.matmul: (1x300x4xf32) <- (1x300x256xf32, 256x4xf32)
        matmul_76 = paddle._C_ops.matmul(relu_19, parameter_133, False, False)
        del relu_19

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        add_115 = paddle._C_ops.add(matmul_76, parameter_132)
        del matmul_76

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_12 = paddle._C_ops.clip(sigmoid_4, full_3, full_0)
        del sigmoid_4

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_13 = paddle._C_ops.clip(clip_12, full_15, full_16)

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(clip_12, full_17, float("1"), True)
        del clip_12

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_14 = paddle._C_ops.clip(scale_29, full_15, full_16)
        del scale_29

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        divide_5 = paddle._C_ops.divide(clip_13, clip_14)
        del clip_13, clip_14

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        log_4 = paddle._C_ops.log(divide_5)
        del divide_5

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        add_116 = paddle._C_ops.add(add_115, log_4)
        del add_115, log_4

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(add_116)
        del add_116

        # pd_op.unsqueeze: (1x300x1x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(sigmoid_5, full_int_array_0)

        # pd_op.matmul: (1x300x512xf32) <- (1x300x4xf32, 4x512xf32)
        matmul_77 = paddle._C_ops.matmul(sigmoid_5, parameter_123, False, False)

        # pd_op.add: (1x300x512xf32) <- (1x300x512xf32, 512xf32)
        add_117 = paddle._C_ops.add(matmul_77, parameter_122)
        del matmul_77

        # pd_op.relu: (1x300x512xf32) <- (1x300x512xf32)
        relu_20 = paddle._C_ops.relu(add_117)
        del add_117

        # pd_op.matmul: (1x300x256xf32) <- (1x300x512xf32, 512x256xf32)
        matmul_78 = paddle._C_ops.matmul(relu_20, parameter_121, False, False)
        del relu_20

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_118 = paddle._C_ops.add(matmul_78, parameter_120)
        del matmul_78

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_119 = paddle._C_ops.add(layer_norm_39, add_118)

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_129 = paddle._C_ops.slice(
            data_8, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_130 = paddle._C_ops.slice(
            data_9, [0], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_79 = paddle._C_ops.matmul(add_119, slice_129, False, False)
        del slice_129

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_120 = paddle._C_ops.add(matmul_79, slice_130)
        del matmul_79, slice_130

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(add_120, full_int_array_8)
        del add_120

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_51 = paddle._C_ops.transpose(reshape_54, [0, 2, 1, 3])
        del reshape_54

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_131 = paddle._C_ops.slice(
            data_8, [1], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_132 = paddle._C_ops.slice(
            data_9, [0], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_80 = paddle._C_ops.matmul(add_119, slice_131, False, False)
        del add_119, slice_131

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_121 = paddle._C_ops.add(matmul_80, slice_132)
        del matmul_80, slice_132

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(add_121, full_int_array_8)
        del add_121

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_55, [0, 2, 1, 3])
        del reshape_55

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_133 = paddle._C_ops.slice(
            data_8, [1], full_int_array_9, full_int_array_10, [1], []
        )
        del data_8

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(
            data_9, [0], full_int_array_9, full_int_array_10, [1], []
        )
        del data_9

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_39, slice_133, False, False)
        del slice_133

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_122 = paddle._C_ops.add(matmul_81, slice_134)
        del matmul_81, slice_134

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(add_122, full_int_array_8)
        del add_122

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_56, [0, 2, 1, 3])
        del reshape_56

        # pd_op.matmul: (1x8x300x300xf32) <- (1x8x300x32xf32, 1x8x300x32xf32)
        matmul_82 = paddle._C_ops.matmul(transpose_51, transpose_52, False, True)
        del transpose_51, transpose_52

        # pd_op.scale: (1x8x300x300xf32) <- (1x8x300x300xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(matmul_82, full_18, float("0"), True)
        del matmul_82

        # pd_op.softmax: (1x8x300x300xf32) <- (1x8x300x300xf32)
        softmax_8 = paddle._C_ops.softmax(scale_30, -1)
        del scale_30

        # pd_op.matmul: (1x8x300x32xf32) <- (1x8x300x300xf32, 1x8x300x32xf32)
        matmul_83 = paddle._C_ops.matmul(softmax_8, transpose_53, False, False)
        del softmax_8, transpose_53

        # pd_op.transpose: (1x300x8x32xf32) <- (1x8x300x32xf32)
        transpose_54 = paddle._C_ops.transpose(matmul_83, [0, 2, 1, 3])
        del matmul_83

        # pd_op.reshape: (1x300x256xf32) <- (1x300x8x32xf32, 3xi64)
        reshape_57 = paddle._C_ops.reshape(transpose_54, full_int_array_11)
        del transpose_54

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_84 = paddle._C_ops.matmul(reshape_57, parameter_39, False, False)
        del parameter_39, reshape_57

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_123 = paddle._C_ops.add(matmul_84, parameter_38)
        del matmul_84, parameter_38

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_124 = paddle._C_ops.add(layer_norm_39, add_123)
        del add_123, layer_norm_39

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_124, parameter_37, parameter_36, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_124, parameter_36, parameter_37

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_125 = paddle._C_ops.add(layer_norm_42, add_118)
        del add_118

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x256xf32, 256x256xf32)
        matmul_85 = paddle._C_ops.matmul(concat_0, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_126 = paddle._C_ops.add(matmul_85, parameter_34)
        del matmul_85, parameter_34

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_51 = [full_7, slice_6, full_19, full_20]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_48 = paddle._C_ops.stack(combine_51, 0)
        del combine_51

        # pd_op.reshape: (1x-1x8x32xf32) <- (1x-1x256xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(add_126, stack_48)
        del add_126, stack_48

        # pd_op.matmul: (1x300x192xf32) <- (1x300x256xf32, 256x192xf32)
        matmul_86 = paddle._C_ops.matmul(add_125, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x300x192xf32) <- (1x300x192xf32, 192xf32)
        add_127 = paddle._C_ops.add(matmul_86, parameter_32)
        del matmul_86, parameter_32

        # pd_op.reshape: (1x300x8x3x4x2xf32) <- (1x300x192xf32, 6xi64)
        reshape_59 = paddle._C_ops.reshape(add_127, full_int_array_12)
        del add_127

        # pd_op.matmul: (1x300x96xf32) <- (1x300x256xf32, 256x96xf32)
        matmul_87 = paddle._C_ops.matmul(add_125, parameter_31, False, False)
        del add_125, parameter_31

        # pd_op.add: (1x300x96xf32) <- (1x300x96xf32, 96xf32)
        add_128 = paddle._C_ops.add(matmul_87, parameter_30)
        del matmul_87, parameter_30

        # pd_op.reshape: (1x300x8x12xf32) <- (1x300x96xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(add_128, full_int_array_13)
        del add_128

        # pd_op.softmax: (1x300x8x12xf32) <- (1x300x8x12xf32)
        softmax_9 = paddle._C_ops.softmax(reshape_60, -1)
        del reshape_60

        # pd_op.reshape: (1x300x8x3x4xf32) <- (1x300x8x12xf32, 5xi64)
        reshape_61 = paddle._C_ops.reshape(softmax_9, full_int_array_14)
        del softmax_9

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_135 = paddle._C_ops.slice(
            unsqueeze_14, [3], full_int_array_6, full_int_array_0, [1], []
        )

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(slice_135, full_int_array_15)
        del slice_135

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(reshape_59, full_21, float("0"), True)
        del reshape_59

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_136 = paddle._C_ops.slice(
            unsqueeze_14, [3], full_int_array_0, full_int_array_10, [1], []
        )
        del unsqueeze_14

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(slice_136, full_int_array_15)
        del slice_136

        # pd_op.multiply: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1x300x1x1x1x2xf32)
        multiply_26 = paddle._C_ops.multiply(scale_31, unsqueeze_16)
        del scale_31, unsqueeze_16

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(multiply_26, full_13, float("0"), True)
        del multiply_26

        # pd_op.add: (1x300x8x3x4x2xf32) <- (1x300x1x1x1x2xf32, 1x300x8x3x4x2xf32)
        add_129 = paddle._C_ops.add(unsqueeze_15, scale_32)
        del scale_32, unsqueeze_15

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_52 = [slice_0, slice_1]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_49 = paddle._C_ops.stack(combine_52, 0)
        del combine_52

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_21 = stack_49
        del stack_49

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_53 = [slice_2, slice_3]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_50 = paddle._C_ops.stack(combine_53, 0)
        del combine_53

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_22 = stack_50
        del stack_50

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_54 = [slice_4, slice_5]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_51 = paddle._C_ops.stack(combine_54, 0)
        del combine_54

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_23 = stack_51
        del stack_51

        # builtin.combine: ([2xi64, 2xi64, 2xi64]) <- (2xi64, 2xi64, 2xi64)
        combine_55 = [assign_21, assign_22, assign_23]
        del assign_21, assign_22, assign_23

        # pd_op.stack: (3x2xi64) <- ([2xi64, 2xi64, 2xi64])
        stack_52 = paddle._C_ops.stack(combine_55, 0)
        del combine_55

        # pd_op.assign: (3x2xi64) <- (3x2xi64)
        assign_24 = stack_52
        del stack_52

        # pd_op.full: (xi64) <- ()
        full_27 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_27,
            [],
            paddle.int64,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del full_27

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_56 = [assign_value__7, scale_0, add_0]
        del assign_value__7

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_53 = paddle._C_ops.stack(combine_56, 0)
        del combine_56

        # pd_op.assign: (3xi64) <- (3xi64)
        assign_25 = stack_53
        del stack_53

        # pd_op.shape64: (4xi64) <- (1x-1x8x32xf32)
        shape64_13 = paddle._C_ops.shape64(reshape_58)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_137 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_13

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_138 = paddle._C_ops.slice(
            assign_24, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_139 = paddle._C_ops.slice(
            slice_138, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_140 = paddle._C_ops.slice(
            slice_138, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_138

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_27 = paddle._C_ops.multiply(slice_139, slice_140)
        del slice_139, slice_140

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_141 = paddle._C_ops.slice(
            assign_24, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_142 = paddle._C_ops.slice(
            slice_141, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_143 = paddle._C_ops.slice(
            slice_141, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_141

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_28 = paddle._C_ops.multiply(slice_142, slice_143)
        del slice_142, slice_143

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_144 = paddle._C_ops.slice(
            assign_24, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_145 = paddle._C_ops.slice(
            slice_144, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_146 = paddle._C_ops.slice(
            slice_144, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_144

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_29 = paddle._C_ops.multiply(slice_145, slice_146)
        del slice_145, slice_146

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_57 = [multiply_27, multiply_28, multiply_29]
        del multiply_27, multiply_28, multiply_29

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_54 = paddle._C_ops.stack(combine_57, 0)
        del combine_57

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x-1x8x32xf32, 3xi64, 1xi32)
        split_22 = paddle._C_ops.split(reshape_58, stack_54, full_1)
        del reshape_58, stack_54

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_23,
            split_24,
            split_25,
        ) = split_22
        del split_22

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(add_129, full_23, float("0"), True)
        del add_129

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(scale_33, full_0, float("-1"), True)
        del scale_33

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_147 = paddle._C_ops.slice(
            assign_24, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_148 = paddle._C_ops.slice(
            slice_147, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_149 = paddle._C_ops.slice(
            slice_147, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_147

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_36 = paddle._C_ops.flatten(split_23, 2, 3)
        del split_23

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_55 = paddle._C_ops.transpose(flatten_36, [0, 2, 1])
        del flatten_36

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_58 = [full_19, full_20, slice_148, slice_149]
        del slice_148, slice_149

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_55 = paddle._C_ops.stack(combine_58, 0)
        del combine_58

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(transpose_55, stack_55)
        del stack_55, transpose_55

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_150 = paddle._C_ops.slice(
            scale_34, [3], full_int_array_6, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_56 = paddle._C_ops.transpose(slice_150, [0, 2, 1, 3, 4])
        del slice_150

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_37 = paddle._C_ops.flatten(transpose_56, 0, 1)
        del transpose_56

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_12 = paddle._C_ops.grid_sample(
            reshape_62, flatten_37, "bilinear", "zeros", False
        )
        del flatten_37, reshape_62

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_151 = paddle._C_ops.slice(
            assign_24, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_152 = paddle._C_ops.slice(
            slice_151, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_153 = paddle._C_ops.slice(
            slice_151, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_151

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_38 = paddle._C_ops.flatten(split_24, 2, 3)
        del split_24

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_57 = paddle._C_ops.transpose(flatten_38, [0, 2, 1])
        del flatten_38

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_59 = [full_19, full_20, slice_152, slice_153]
        del slice_152, slice_153

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_56 = paddle._C_ops.stack(combine_59, 0)
        del combine_59

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(transpose_57, stack_56)
        del stack_56, transpose_57

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_154 = paddle._C_ops.slice(
            scale_34, [3], full_int_array_3, full_int_array_0, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_58 = paddle._C_ops.transpose(slice_154, [0, 2, 1, 3, 4])
        del slice_154

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_39 = paddle._C_ops.flatten(transpose_58, 0, 1)
        del transpose_58

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_13 = paddle._C_ops.grid_sample(
            reshape_63, flatten_39, "bilinear", "zeros", False
        )
        del flatten_39, reshape_63

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_155 = paddle._C_ops.slice(
            assign_24, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del assign_24

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_156 = paddle._C_ops.slice(
            slice_155, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_157 = paddle._C_ops.slice(
            slice_155, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_155

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_40 = paddle._C_ops.flatten(split_25, 2, 3)
        del split_25

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_59 = paddle._C_ops.transpose(flatten_40, [0, 2, 1])
        del flatten_40

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_60 = [full_19, full_20, slice_156, slice_157]
        del slice_156, slice_157

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_57 = paddle._C_ops.stack(combine_60, 0)
        del combine_60

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(transpose_59, stack_57)
        del stack_57, transpose_59

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_158 = paddle._C_ops.slice(
            scale_34, [3], full_int_array_0, full_int_array_1, [1], [3]
        )
        del scale_34

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_60 = paddle._C_ops.transpose(slice_158, [0, 2, 1, 3, 4])
        del slice_158

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_41 = paddle._C_ops.flatten(transpose_60, 0, 1)
        del transpose_60

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_14 = paddle._C_ops.grid_sample(
            reshape_64, flatten_41, "bilinear", "zeros", False
        )
        del flatten_41, reshape_64

        # pd_op.transpose: (1x8x300x3x4xf32) <- (1x300x8x3x4xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_61, [0, 2, 1, 3, 4])
        del reshape_61

        # pd_op.reshape: (8x1x300x12xf32) <- (1x8x300x3x4xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(transpose_61, full_int_array_16)
        del transpose_61

        # builtin.combine: ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32]) <- (8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32)
        combine_61 = [grid_sample_12, grid_sample_13, grid_sample_14]
        del grid_sample_12, grid_sample_13, grid_sample_14

        # pd_op.stack: (8x32x300x3x4xf32) <- ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32])
        stack_58 = paddle._C_ops.stack(combine_61, -2)
        del combine_61

        # pd_op.flatten: (8x32x300x12xf32) <- (8x32x300x3x4xf32)
        flatten_42 = paddle._C_ops.flatten(stack_58, 3, 4)
        del stack_58

        # pd_op.multiply: (8x32x300x12xf32) <- (8x32x300x12xf32, 8x1x300x12xf32)
        multiply_30 = paddle._C_ops.multiply(flatten_42, reshape_65)
        del flatten_42, reshape_65

        # pd_op.sum: (8x32x300xf32) <- (8x32x300x12xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(multiply_30, full_int_array_4, None, False)
        del multiply_30

        # pd_op.reshape: (1x256x300xf32) <- (8x32x300xf32, 3xi64)
        reshape_66 = paddle._C_ops.reshape(sum_4, full_int_array_17)
        del sum_4

        # pd_op.transpose: (1x300x256xf32) <- (1x256x300xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_66, [0, 2, 1])
        del reshape_66

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_88 = paddle._C_ops.matmul(transpose_62, parameter_29, False, False)
        del parameter_29, transpose_62

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_130 = paddle._C_ops.add(matmul_88, parameter_28)
        del matmul_88, parameter_28

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_131 = paddle._C_ops.add(layer_norm_42, add_130)
        del add_130, layer_norm_42

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_131, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_131, parameter_26, parameter_27

        # pd_op.matmul: (1x300x1024xf32) <- (1x300x256xf32, 256x1024xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_45, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x300x1024xf32) <- (1x300x1024xf32, 1024xf32)
        add_132 = paddle._C_ops.add(matmul_89, parameter_24)
        del matmul_89, parameter_24

        # pd_op.relu: (1x300x1024xf32) <- (1x300x1024xf32)
        relu_21 = paddle._C_ops.relu(add_132)
        del add_132

        # pd_op.matmul: (1x300x256xf32) <- (1x300x1024xf32, 1024x256xf32)
        matmul_90 = paddle._C_ops.matmul(relu_21, parameter_23, False, False)
        del parameter_23, relu_21

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_133 = paddle._C_ops.add(matmul_90, parameter_22)
        del matmul_90, parameter_22

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_134 = paddle._C_ops.add(layer_norm_45, add_133)
        del add_133, layer_norm_45

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_134, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_134, parameter_20, parameter_21

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_48, parameter_137, False, False)

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_135 = paddle._C_ops.add(matmul_91, parameter_136)
        del matmul_91

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_22 = paddle._C_ops.relu(add_135)
        del add_135

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_92 = paddle._C_ops.matmul(relu_22, parameter_135, False, False)
        del relu_22

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_136 = paddle._C_ops.add(matmul_92, parameter_134)
        del matmul_92

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_23 = paddle._C_ops.relu(add_136)
        del add_136

        # pd_op.matmul: (1x300x4xf32) <- (1x300x256xf32, 256x4xf32)
        matmul_93 = paddle._C_ops.matmul(relu_23, parameter_133, False, False)
        del relu_23

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        add_137 = paddle._C_ops.add(matmul_93, parameter_132)
        del matmul_93

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_15 = paddle._C_ops.clip(sigmoid_5, full_3, full_0)
        del sigmoid_5

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_16 = paddle._C_ops.clip(clip_15, full_15, full_16)

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(clip_15, full_17, float("1"), True)
        del clip_15

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_17 = paddle._C_ops.clip(scale_35, full_15, full_16)
        del scale_35

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        divide_6 = paddle._C_ops.divide(clip_16, clip_17)
        del clip_16, clip_17

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        log_5 = paddle._C_ops.log(divide_6)
        del divide_6

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        add_138 = paddle._C_ops.add(add_137, log_5)
        del add_137, log_5

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_6 = paddle._C_ops.sigmoid(add_138)
        del add_138

        # pd_op.unsqueeze: (1x300x1x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(sigmoid_6, full_int_array_0)

        # pd_op.matmul: (1x300x512xf32) <- (1x300x4xf32, 4x512xf32)
        matmul_94 = paddle._C_ops.matmul(sigmoid_6, parameter_123, False, False)
        del parameter_123

        # pd_op.add: (1x300x512xf32) <- (1x300x512xf32, 512xf32)
        add_139 = paddle._C_ops.add(matmul_94, parameter_122)
        del matmul_94, parameter_122

        # pd_op.relu: (1x300x512xf32) <- (1x300x512xf32)
        relu_24 = paddle._C_ops.relu(add_139)
        del add_139

        # pd_op.matmul: (1x300x256xf32) <- (1x300x512xf32, 512x256xf32)
        matmul_95 = paddle._C_ops.matmul(relu_24, parameter_121, False, False)
        del parameter_121, relu_24

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_140 = paddle._C_ops.add(matmul_95, parameter_120)
        del matmul_95, parameter_120

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_141 = paddle._C_ops.add(layer_norm_48, add_140)

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_159 = paddle._C_ops.slice(
            data_10, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_160 = paddle._C_ops.slice(
            data_11, [0], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_96 = paddle._C_ops.matmul(add_141, slice_159, False, False)
        del slice_159

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_142 = paddle._C_ops.add(matmul_96, slice_160)
        del matmul_96, slice_160

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(add_142, full_int_array_8)
        del add_142

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_63 = paddle._C_ops.transpose(reshape_67, [0, 2, 1, 3])
        del reshape_67

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_161 = paddle._C_ops.slice(
            data_10, [1], full_int_array_7, full_int_array_9, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_162 = paddle._C_ops.slice(
            data_11, [0], full_int_array_7, full_int_array_9, [1], []
        )
        del full_int_array_7

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_97 = paddle._C_ops.matmul(add_141, slice_161, False, False)
        del add_141, slice_161

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_143 = paddle._C_ops.add(matmul_97, slice_162)
        del matmul_97, slice_162

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(add_143, full_int_array_8)
        del add_143

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_68, [0, 2, 1, 3])
        del reshape_68

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_163 = paddle._C_ops.slice(
            data_10, [1], full_int_array_9, full_int_array_10, [1], []
        )
        del data_10

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_164 = paddle._C_ops.slice(
            data_11, [0], full_int_array_9, full_int_array_10, [1], []
        )
        del data_11, full_int_array_9

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_98 = paddle._C_ops.matmul(layer_norm_48, slice_163, False, False)
        del slice_163

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_144 = paddle._C_ops.add(matmul_98, slice_164)
        del matmul_98, slice_164

        # pd_op.reshape: (1x300x8x32xf32) <- (1x300x256xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(add_144, full_int_array_8)
        del add_144, full_int_array_8

        # pd_op.transpose: (1x8x300x32xf32) <- (1x300x8x32xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_69, [0, 2, 1, 3])
        del reshape_69

        # pd_op.matmul: (1x8x300x300xf32) <- (1x8x300x32xf32, 1x8x300x32xf32)
        matmul_99 = paddle._C_ops.matmul(transpose_63, transpose_64, False, True)
        del transpose_63, transpose_64

        # pd_op.scale: (1x8x300x300xf32) <- (1x8x300x300xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(matmul_99, full_18, float("0"), True)
        del full_18, matmul_99

        # pd_op.softmax: (1x8x300x300xf32) <- (1x8x300x300xf32)
        softmax_10 = paddle._C_ops.softmax(scale_36, -1)
        del scale_36

        # pd_op.matmul: (1x8x300x32xf32) <- (1x8x300x300xf32, 1x8x300x32xf32)
        matmul_100 = paddle._C_ops.matmul(softmax_10, transpose_65, False, False)
        del softmax_10, transpose_65

        # pd_op.transpose: (1x300x8x32xf32) <- (1x8x300x32xf32)
        transpose_66 = paddle._C_ops.transpose(matmul_100, [0, 2, 1, 3])
        del matmul_100

        # pd_op.reshape: (1x300x256xf32) <- (1x300x8x32xf32, 3xi64)
        reshape_70 = paddle._C_ops.reshape(transpose_66, full_int_array_11)
        del full_int_array_11, transpose_66

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_101 = paddle._C_ops.matmul(reshape_70, parameter_19, False, False)
        del parameter_19, reshape_70

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_145 = paddle._C_ops.add(matmul_101, parameter_18)
        del matmul_101, parameter_18

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_146 = paddle._C_ops.add(layer_norm_48, add_145)
        del add_145, layer_norm_48

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_146, parameter_17, parameter_16, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_146, parameter_16, parameter_17

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_147 = paddle._C_ops.add(layer_norm_51, add_140)
        del add_140

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x256xf32, 256x256xf32)
        matmul_102 = paddle._C_ops.matmul(concat_0, parameter_15, False, False)
        del concat_0, parameter_15

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_148 = paddle._C_ops.add(matmul_102, parameter_14)
        del matmul_102, parameter_14

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_62 = [full_7, slice_6, full_19, full_20]
        del full_7, slice_6

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_59 = paddle._C_ops.stack(combine_62, 0)
        del combine_62

        # pd_op.reshape: (1x-1x8x32xf32) <- (1x-1x256xf32, 4xi64)
        reshape_71 = paddle._C_ops.reshape(add_148, stack_59)
        del add_148, stack_59

        # pd_op.matmul: (1x300x192xf32) <- (1x300x256xf32, 256x192xf32)
        matmul_103 = paddle._C_ops.matmul(add_147, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x300x192xf32) <- (1x300x192xf32, 192xf32)
        add_149 = paddle._C_ops.add(matmul_103, parameter_12)
        del matmul_103, parameter_12

        # pd_op.reshape: (1x300x8x3x4x2xf32) <- (1x300x192xf32, 6xi64)
        reshape_72 = paddle._C_ops.reshape(add_149, full_int_array_12)
        del add_149, full_int_array_12

        # pd_op.matmul: (1x300x96xf32) <- (1x300x256xf32, 256x96xf32)
        matmul_104 = paddle._C_ops.matmul(add_147, parameter_11, False, False)
        del add_147, parameter_11

        # pd_op.add: (1x300x96xf32) <- (1x300x96xf32, 96xf32)
        add_150 = paddle._C_ops.add(matmul_104, parameter_10)
        del matmul_104, parameter_10

        # pd_op.reshape: (1x300x8x12xf32) <- (1x300x96xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(add_150, full_int_array_13)
        del add_150, full_int_array_13

        # pd_op.softmax: (1x300x8x12xf32) <- (1x300x8x12xf32)
        softmax_11 = paddle._C_ops.softmax(reshape_73, -1)
        del reshape_73

        # pd_op.reshape: (1x300x8x3x4xf32) <- (1x300x8x12xf32, 5xi64)
        reshape_74 = paddle._C_ops.reshape(softmax_11, full_int_array_14)
        del full_int_array_14, softmax_11

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_165 = paddle._C_ops.slice(
            unsqueeze_17, [3], full_int_array_6, full_int_array_0, [1], []
        )

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(slice_165, full_int_array_15)
        del slice_165

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(reshape_72, full_21, float("0"), True)
        del full_21, reshape_72

        # pd_op.slice: (1x300x1x2xf32) <- (1x300x1x4xf32, 1xi64, 1xi64)
        slice_166 = paddle._C_ops.slice(
            unsqueeze_17, [3], full_int_array_0, full_int_array_10, [1], []
        )
        del unsqueeze_17

        # pd_op.unsqueeze: (1x300x1x1x1x2xf32) <- (1x300x1x2xf32, 2xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(slice_166, full_int_array_15)
        del full_int_array_15, slice_166

        # pd_op.multiply: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1x300x1x1x1x2xf32)
        multiply_31 = paddle._C_ops.multiply(scale_37, unsqueeze_19)
        del scale_37, unsqueeze_19

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(multiply_31, full_13, float("0"), True)
        del multiply_31

        # pd_op.add: (1x300x8x3x4x2xf32) <- (1x300x1x1x1x2xf32, 1x300x8x3x4x2xf32)
        add_151 = paddle._C_ops.add(unsqueeze_18, scale_38)
        del scale_38, unsqueeze_18

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_63 = [slice_0, slice_1]
        del slice_0, slice_1

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_60 = paddle._C_ops.stack(combine_63, 0)
        del combine_63

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_26 = stack_60
        del stack_60

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_64 = [slice_2, slice_3]
        del slice_2, slice_3

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_61 = paddle._C_ops.stack(combine_64, 0)
        del combine_64

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_27 = stack_61
        del stack_61

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_65 = [slice_4, slice_5]
        del slice_4, slice_5

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_62 = paddle._C_ops.stack(combine_65, 0)
        del combine_65

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_28 = stack_62
        del stack_62

        # builtin.combine: ([2xi64, 2xi64, 2xi64]) <- (2xi64, 2xi64, 2xi64)
        combine_66 = [assign_26, assign_27, assign_28]
        del assign_26, assign_27, assign_28

        # pd_op.stack: (3x2xi64) <- ([2xi64, 2xi64, 2xi64])
        stack_63 = paddle._C_ops.stack(combine_66, 0)
        del combine_66

        # pd_op.assign: (3x2xi64) <- (3x2xi64)
        assign_29 = stack_63
        del stack_63

        # pd_op.full: (xi64) <- ()
        full_28 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_28,
            [],
            paddle.int64,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del full_28

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_67 = [assign_value__8, scale_0, add_0]
        del add_0, assign_value__8, scale_0

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_64 = paddle._C_ops.stack(combine_67, 0)
        del combine_67

        # pd_op.assign: (3xi64) <- (3xi64)
        assign_30 = stack_64
        del stack_64

        # pd_op.shape64: (4xi64) <- (1x-1x8x32xf32)
        shape64_14 = paddle._C_ops.shape64(reshape_71)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_167 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del shape64_14

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_168 = paddle._C_ops.slice(
            assign_29, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_169 = paddle._C_ops.slice(
            slice_168, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_170 = paddle._C_ops.slice(
            slice_168, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_168

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_32 = paddle._C_ops.multiply(slice_169, slice_170)
        del slice_169, slice_170

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_171 = paddle._C_ops.slice(
            assign_29, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_172 = paddle._C_ops.slice(
            slice_171, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_173 = paddle._C_ops.slice(
            slice_171, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_171

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_33 = paddle._C_ops.multiply(slice_172, slice_173)
        del slice_172, slice_173

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_174 = paddle._C_ops.slice(
            assign_29, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_175 = paddle._C_ops.slice(
            slice_174, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_176 = paddle._C_ops.slice(
            slice_174, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_174

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_34 = paddle._C_ops.multiply(slice_175, slice_176)
        del slice_175, slice_176

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_68 = [multiply_32, multiply_33, multiply_34]
        del multiply_32, multiply_33, multiply_34

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_65 = paddle._C_ops.stack(combine_68, 0)
        del combine_68

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x-1x8x32xf32, 3xi64, 1xi32)
        split_26 = paddle._C_ops.split(reshape_71, stack_65, full_1)
        del reshape_71, stack_65

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_27,
            split_28,
            split_29,
        ) = split_26
        del split_26

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(add_151, full_23, float("0"), True)
        del add_151, full_23

        # pd_op.scale: (1x300x8x3x4x2xf32) <- (1x300x8x3x4x2xf32, 1xf32)
        scale_40 = paddle._C_ops.scale(scale_39, full_0, float("-1"), True)
        del scale_39

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_177 = paddle._C_ops.slice(
            assign_29, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_178 = paddle._C_ops.slice(
            slice_177, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_179 = paddle._C_ops.slice(
            slice_177, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_177

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_43 = paddle._C_ops.flatten(split_27, 2, 3)
        del split_27

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_67 = paddle._C_ops.transpose(flatten_43, [0, 2, 1])
        del flatten_43

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_69 = [full_19, full_20, slice_178, slice_179]
        del slice_178, slice_179

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_66 = paddle._C_ops.stack(combine_69, 0)
        del combine_69

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_75 = paddle._C_ops.reshape(transpose_67, stack_66)
        del stack_66, transpose_67

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_180 = paddle._C_ops.slice(
            scale_40, [3], full_int_array_6, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_68 = paddle._C_ops.transpose(slice_180, [0, 2, 1, 3, 4])
        del slice_180

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_44 = paddle._C_ops.flatten(transpose_68, 0, 1)
        del transpose_68

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_15 = paddle._C_ops.grid_sample(
            reshape_75, flatten_44, "bilinear", "zeros", False
        )
        del flatten_44, reshape_75

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_181 = paddle._C_ops.slice(
            assign_29, [0], full_int_array_3, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_182 = paddle._C_ops.slice(
            slice_181, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_183 = paddle._C_ops.slice(
            slice_181, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_181

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_45 = paddle._C_ops.flatten(split_28, 2, 3)
        del split_28

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_69 = paddle._C_ops.transpose(flatten_45, [0, 2, 1])
        del flatten_45

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_70 = [full_19, full_20, slice_182, slice_183]
        del slice_182, slice_183

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_67 = paddle._C_ops.stack(combine_70, 0)
        del combine_70

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_76 = paddle._C_ops.reshape(transpose_69, stack_67)
        del stack_67, transpose_69

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_184 = paddle._C_ops.slice(
            scale_40, [3], full_int_array_3, full_int_array_0, [1], [3]
        )

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_70 = paddle._C_ops.transpose(slice_184, [0, 2, 1, 3, 4])
        del slice_184

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_46 = paddle._C_ops.flatten(transpose_70, 0, 1)
        del transpose_70

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_16 = paddle._C_ops.grid_sample(
            reshape_76, flatten_46, "bilinear", "zeros", False
        )
        del flatten_46, reshape_76

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_185 = paddle._C_ops.slice(
            assign_29, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del assign_29

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_186 = paddle._C_ops.slice(
            slice_185, [0], full_int_array_6, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_187 = paddle._C_ops.slice(
            slice_185, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_185

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_47 = paddle._C_ops.flatten(split_29, 2, 3)
        del split_29

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_71 = paddle._C_ops.transpose(flatten_47, [0, 2, 1])
        del flatten_47

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_71 = [full_19, full_20, slice_186, slice_187]
        del full_19, full_20, slice_186, slice_187

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_68 = paddle._C_ops.stack(combine_71, 0)
        del combine_71

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(transpose_71, stack_68)
        del stack_68, transpose_71

        # pd_op.slice: (1x300x8x4x2xf32) <- (1x300x8x3x4x2xf32, 1xi64, 1xi64)
        slice_188 = paddle._C_ops.slice(
            scale_40, [3], full_int_array_0, full_int_array_1, [1], [3]
        )
        del scale_40

        # pd_op.transpose: (1x8x300x4x2xf32) <- (1x300x8x4x2xf32)
        transpose_72 = paddle._C_ops.transpose(slice_188, [0, 2, 1, 3, 4])
        del slice_188

        # pd_op.flatten: (8x300x4x2xf32) <- (1x8x300x4x2xf32)
        flatten_48 = paddle._C_ops.flatten(transpose_72, 0, 1)
        del transpose_72

        # pd_op.grid_sample: (8x32x300x4xf32) <- (8x32x-1x-1xf32, 8x300x4x2xf32)
        grid_sample_17 = paddle._C_ops.grid_sample(
            reshape_77, flatten_48, "bilinear", "zeros", False
        )
        del flatten_48, reshape_77

        # pd_op.transpose: (1x8x300x3x4xf32) <- (1x300x8x3x4xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_74, [0, 2, 1, 3, 4])
        del reshape_74

        # pd_op.reshape: (8x1x300x12xf32) <- (1x8x300x3x4xf32, 4xi64)
        reshape_78 = paddle._C_ops.reshape(transpose_73, full_int_array_16)
        del full_int_array_16, transpose_73

        # builtin.combine: ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32]) <- (8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32)
        combine_72 = [grid_sample_15, grid_sample_16, grid_sample_17]
        del grid_sample_15, grid_sample_16, grid_sample_17

        # pd_op.stack: (8x32x300x3x4xf32) <- ([8x32x300x4xf32, 8x32x300x4xf32, 8x32x300x4xf32])
        stack_69 = paddle._C_ops.stack(combine_72, -2)
        del combine_72

        # pd_op.flatten: (8x32x300x12xf32) <- (8x32x300x3x4xf32)
        flatten_49 = paddle._C_ops.flatten(stack_69, 3, 4)
        del stack_69

        # pd_op.multiply: (8x32x300x12xf32) <- (8x32x300x12xf32, 8x1x300x12xf32)
        multiply_35 = paddle._C_ops.multiply(flatten_49, reshape_78)
        del flatten_49, reshape_78

        # pd_op.sum: (8x32x300xf32) <- (8x32x300x12xf32, 1xi64)
        sum_5 = paddle._C_ops.sum(multiply_35, full_int_array_4, None, False)
        del multiply_35

        # pd_op.reshape: (1x256x300xf32) <- (8x32x300xf32, 3xi64)
        reshape_79 = paddle._C_ops.reshape(sum_5, full_int_array_17)
        del full_int_array_17, sum_5

        # pd_op.transpose: (1x300x256xf32) <- (1x256x300xf32)
        transpose_74 = paddle._C_ops.transpose(reshape_79, [0, 2, 1])
        del reshape_79

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_105 = paddle._C_ops.matmul(transpose_74, parameter_9, False, False)
        del parameter_9, transpose_74

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_152 = paddle._C_ops.add(matmul_105, parameter_8)
        del matmul_105, parameter_8

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_153 = paddle._C_ops.add(layer_norm_51, add_152)
        del add_152, layer_norm_51

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_153, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_153, parameter_6, parameter_7

        # pd_op.matmul: (1x300x1024xf32) <- (1x300x256xf32, 256x1024xf32)
        matmul_106 = paddle._C_ops.matmul(layer_norm_54, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (1x300x1024xf32) <- (1x300x1024xf32, 1024xf32)
        add_154 = paddle._C_ops.add(matmul_106, parameter_4)
        del matmul_106, parameter_4

        # pd_op.relu: (1x300x1024xf32) <- (1x300x1024xf32)
        relu_25 = paddle._C_ops.relu(add_154)
        del add_154

        # pd_op.matmul: (1x300x256xf32) <- (1x300x1024xf32, 1024x256xf32)
        matmul_107 = paddle._C_ops.matmul(relu_25, parameter_3, False, False)
        del parameter_3, relu_25

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_155 = paddle._C_ops.add(matmul_107, parameter_2)
        del matmul_107, parameter_2

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 1x300x256xf32)
        add_156 = paddle._C_ops.add(layer_norm_54, add_155)
        del add_155, layer_norm_54

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_156, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_156, parameter_0, parameter_1

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_108 = paddle._C_ops.matmul(layer_norm_57, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_157 = paddle._C_ops.add(matmul_108, parameter_136)
        del matmul_108, parameter_136

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_26 = paddle._C_ops.relu(add_157)
        del add_157

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_109 = paddle._C_ops.matmul(relu_26, parameter_135, False, False)
        del parameter_135, relu_26

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_158 = paddle._C_ops.add(matmul_109, parameter_134)
        del matmul_109, parameter_134

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_27 = paddle._C_ops.relu(add_158)
        del add_158

        # pd_op.matmul: (1x300x4xf32) <- (1x300x256xf32, 256x4xf32)
        matmul_110 = paddle._C_ops.matmul(relu_27, parameter_133, False, False)
        del parameter_133, relu_27

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        add_159 = paddle._C_ops.add(matmul_110, parameter_132)
        del matmul_110, parameter_132

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_18 = paddle._C_ops.clip(sigmoid_6, full_3, full_0)
        del full_3, sigmoid_6

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_19 = paddle._C_ops.clip(clip_18, full_15, full_16)

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        scale_41 = paddle._C_ops.scale(clip_18, full_17, float("1"), True)
        del clip_18, full_17

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_20 = paddle._C_ops.clip(scale_41, full_15, full_16)
        del full_15, full_16, scale_41

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        divide_7 = paddle._C_ops.divide(clip_19, clip_20)
        del clip_19, clip_20

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        log_6 = paddle._C_ops.log(divide_7)
        del divide_7

        # pd_op.add: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        add_160 = paddle._C_ops.add(add_159, log_6)
        del add_159, log_6

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_7 = paddle._C_ops.sigmoid(add_160)
        del add_160

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_57, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del layer_norm_57, parameter_130, parameter_131

        # pd_op.matmul: (1x300x2xf32) <- (1x300x256xf32, 256x2xf32)
        matmul_111 = paddle._C_ops.matmul(layer_norm_60, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (1x300x2xf32) <- (1x300x2xf32, 2xf32)
        add_161 = paddle._C_ops.add(matmul_111, parameter_138)
        del matmul_111, parameter_138

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_112 = paddle._C_ops.matmul(layer_norm_60, parameter_129, False, False)
        del layer_norm_60, parameter_129

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_162 = paddle._C_ops.add(matmul_112, parameter_128)
        del matmul_112, parameter_128

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_28 = paddle._C_ops.relu(add_162)
        del add_162

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_113 = paddle._C_ops.matmul(relu_28, parameter_127, False, False)
        del parameter_127, relu_28

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_163 = paddle._C_ops.add(matmul_113, parameter_126)
        del matmul_113, parameter_126

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_29 = paddle._C_ops.relu(add_163)
        del add_163

        # pd_op.matmul: (1x300x128xf32) <- (1x300x256xf32, 256x128xf32)
        matmul_114 = paddle._C_ops.matmul(relu_29, parameter_125, False, False)
        del parameter_125, relu_29

        # pd_op.add: (1x300x128xf32) <- (1x300x128xf32, 128xf32)
        add_164 = paddle._C_ops.add(matmul_114, parameter_124)
        del matmul_114, parameter_124

        # pd_op.bmm: (1x300x-1xf32) <- (1x300x128xf32, 1x128x-1xf32)
        bmm_1 = paddle._C_ops.bmm(add_164, flatten_3)
        del add_164, flatten_3

        # pd_op.reshape: (1x300x-1x-1xf32) <- (1x300x-1xf32, 4xi64)
        reshape_80 = paddle._C_ops.reshape(bmm_1, stack_1)
        del bmm_1, stack_1

        # builtin.combine: ([1x300x4xf32]) <- (1x300x4xf32)
        combine_73 = [sigmoid_7]
        del sigmoid_7

        # pd_op.stack: (1x1x300x4xf32) <- ([1x300x4xf32])
        stack_70 = paddle._C_ops.stack(combine_73, 0)
        del combine_73

        # builtin.combine: ([1x300x2xf32]) <- (1x300x2xf32)
        combine_74 = [add_161]
        del add_161

        # pd_op.stack: (1x1x300x2xf32) <- ([1x300x2xf32])
        stack_71 = paddle._C_ops.stack(combine_74, 0)
        del combine_74

        # builtin.combine: ([1x300x-1x-1xf32]) <- (1x300x-1x-1xf32)
        combine_75 = [reshape_80]
        del reshape_80

        # pd_op.stack: (1x1x300x-1x-1xf32) <- ([1x300x-1x-1xf32])
        stack_72 = paddle._C_ops.stack(combine_75, 0)
        del combine_75

        # pd_op.slice: (1x300x4xf32) <- (1x1x300x4xf32, 1xi64, 1xi64)
        slice_189 = paddle._C_ops.slice(
            stack_70, [0], full_int_array_4, full_int_array_10, [1], [0]
        )
        del stack_70

        # pd_op.slice: (1x300x2xf32) <- (1x1x300x2xf32, 1xi64, 1xi64)
        slice_190 = paddle._C_ops.slice(
            stack_71, [0], full_int_array_4, full_int_array_10, [1], [0]
        )
        del stack_71

        # pd_op.slice: (1x300x-1x-1xf32) <- (1x1x300x-1x-1xf32, 1xi64, 1xi64)
        slice_191 = paddle._C_ops.slice(
            stack_72, [0], full_int_array_4, full_int_array_10, [1], [0]
        )
        del stack_72

        # pd_op.slice: (0x3x-1x-1xf32) <- (1x3x-1x-1xf32, 1xi64, 1xi64)
        slice_192 = paddle._C_ops.slice(
            data_18, [0], full_int_array_0, full_int_array_10, [1], []
        )
        del data_18, full_int_array_10

        # pd_op.shape64: (4xi64) <- (0x3x-1x-1xf32)
        shape64_15 = paddle._C_ops.shape64(slice_192)
        del slice_192

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_193 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_194 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_15

        # pd_op.split_with_num: ([1x300x2xf32, 1x300x2xf32]) <- (1x300x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(slice_189, 2, full_12)
        del full_12, slice_189

        # builtin.split: (1x300x2xf32, 1x300x2xf32) <- ([1x300x2xf32, 1x300x2xf32])
        (
            split_30,
            split_31,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.scale: (1x300x2xf32) <- (1x300x2xf32, 1xf32)
        scale_42 = paddle._C_ops.scale(split_31, full_13, float("0"), True)

        # pd_op.subtract: (1x300x2xf32) <- (1x300x2xf32, 1x300x2xf32)
        subtract_2 = paddle._C_ops.subtract(split_30, scale_42)
        del scale_42

        # pd_op.scale: (1x300x2xf32) <- (1x300x2xf32, 1xf32)
        scale_43 = paddle._C_ops.scale(split_31, full_13, float("0"), True)
        del full_13, split_31

        # pd_op.add: (1x300x2xf32) <- (1x300x2xf32, 1x300x2xf32)
        add_165 = paddle._C_ops.add(split_30, scale_43)
        del scale_43, split_30

        # builtin.combine: ([1x300x2xf32, 1x300x2xf32]) <- (1x300x2xf32, 1x300x2xf32)
        combine_76 = [subtract_2, add_165]
        del add_165, subtract_2

        # pd_op.concat: (1x300x4xf32) <- ([1x300x2xf32, 1x300x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_76, full_14)
        del combine_76

        # pd_op.divide: (1x2xf32) <- (1x2xf32, 1x2xf32)
        divide_8 = paddle._C_ops.divide(data_19, data_20)
        del data_19, data_20

        # pd_op.scale: (1x2xf32) <- (1x2xf32, 1xf32)
        scale_44 = paddle._C_ops.scale(divide_8, full_0, float("0.5"), True)
        del divide_8, full_0

        # pd_op.floor: (1x2xf32) <- (1x2xf32)
        floor_0 = paddle._C_ops.floor(scale_44)
        del scale_44

        # pd_op.split_with_num: ([1x1xf32, 1x1xf32]) <- (1x2xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(floor_0, 2, full_1)
        del full_1

        # builtin.split: (1x1xf32, 1x1xf32) <- ([1x1xf32, 1x1xf32])
        (
            split_32,
            split_33,
        ) = split_with_num_2
        del split_with_num_2

        # pd_op.flip: (1x2xf32) <- (1x2xf32)
        flip_0 = paddle._C_ops.flip(floor_0, [1])
        del floor_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [1, 2]

        # pd_op.tile: (1x4xf32) <- (1x2xf32, 2xi64)
        tile_2 = paddle._C_ops.tile(flip_0, full_int_array_18)
        del flip_0, full_int_array_18

        # pd_op.unsqueeze: (1x1x4xf32) <- (1x4xf32, 1xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(tile_2, full_int_array_3)
        del tile_2

        # pd_op.multiply: (1x300x4xf32) <- (1x300x4xf32, 1x1x4xf32)
        multiply_36 = paddle._C_ops.multiply(concat_2, unsqueeze_20)
        del concat_2, unsqueeze_20

        # pd_op.sigmoid: (1x300x2xf32) <- (1x300x2xf32)
        sigmoid_8 = paddle._C_ops.sigmoid(slice_190)
        del slice_190

        # pd_op.flatten: (1x600xf32) <- (1x300x2xf32)
        flatten_50 = paddle._C_ops.flatten(sigmoid_8, 1, 2)
        del sigmoid_8

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full(
            [1], float("100"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (1x100xf32, 1x100xi64) <- (1x600xf32, 1xi32)
        topk_2, topk_3 = (lambda x, f: f(x))(
            paddle._C_ops.topk(flatten_50, full_29, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del flatten_50, full_29

        # pd_op.full: (xi64) <- ()
        full_30 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.remainder: (1x100xi64) <- (1x100xi64, xi64)
        remainder_0 = paddle._C_ops.remainder(topk_3, full_30)

        # pd_op.floor_divide: (1x100xi64) <- (1x100xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(topk_3, full_30)
        del full_30, topk_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [1, 100]

        # pd_op.tile: (1x100xi64) <- (1x1xi64, 2xi64)
        tile_3 = paddle._C_ops.tile(unsqueeze_0, full_int_array_19)
        del full_int_array_19, unsqueeze_0

        # builtin.combine: ([1x100xi64, 1x100xi64]) <- (1x100xi64, 1x100xi64)
        combine_77 = [tile_3, floor_divide_0]
        del floor_divide_0, tile_3

        # pd_op.stack: (1x100x2xi64) <- ([1x100xi64, 1x100xi64])
        stack_73 = paddle._C_ops.stack(combine_77, -1)
        del combine_77

        # pd_op.gather_nd: (1x100x4xf32) <- (1x300x4xf32, 1x100x2xi64)
        gather_nd_2 = paddle._C_ops.gather_nd(multiply_36, stack_73)
        del multiply_36

        # pd_op.shape64: (4xi64) <- (1x300x-1x-1xf32)
        shape64_16 = paddle._C_ops.shape64(slice_191)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_195 = paddle._C_ops.slice(
            shape64_16, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del full_int_array_0, shape64_16

        # pd_op.shape64: (4xi64) <- (1x300x-1x-1xf32)
        shape64_17 = paddle._C_ops.shape64(slice_191)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_196 = paddle._C_ops.slice(
            shape64_17, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1, full_int_array_2, shape64_17

        # pd_op.gather_nd: (1x100x-1x-1xf32) <- (1x300x-1x-1xf32, 1x100x2xi64)
        gather_nd_3 = paddle._C_ops.gather_nd(slice_191, stack_73)
        del slice_191, stack_73

        # pd_op.slice: (1xf32) <- (1x1xf32, 1xi64, 1xi64)
        slice_197 = paddle._C_ops.slice(
            split_32, [0], full_int_array_6, full_int_array_3, [1], [0]
        )
        del split_32

        # pd_op.cast: (1xi32) <- (1xf32)
        cast_16 = paddle._C_ops.cast(slice_197, paddle.int32)
        del slice_197

        # pd_op.slice: (1xf32) <- (1x1xf32, 1xi64, 1xi64)
        slice_198 = paddle._C_ops.slice(
            split_33, [0], full_int_array_6, full_int_array_3, [1], [0]
        )
        del full_int_array_6, split_33

        # pd_op.cast: (1xi32) <- (1xf32)
        cast_17 = paddle._C_ops.cast(slice_198, paddle.int32)
        del slice_198

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_78 = [cast_16, cast_17]
        del cast_16, cast_17

        # pd_op.bilinear_interp: (1x100x-1x-1xf32) <- (1x100x-1x-1xf32, None, [1xi32, 1xi32], None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            gather_nd_3,
            None,
            combine_78,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [],
            "bilinear",
            False,
            0,
        )
        del combine_78, gather_nd_3

        # pd_op.sigmoid: (1x100x-1x-1xf32) <- (1x100x-1x-1xf32)
        sigmoid_9 = paddle._C_ops.sigmoid(bilinear_interp_0)
        del bilinear_interp_0

        # pd_op.full: (xf32) <- ()
        full_31 = paddle._C_ops.full(
            [], float("0.5"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x100x-1x-1xb) <- (1x100x-1x-1xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(sigmoid_9, full_31)
        del full_31, sigmoid_9

        # pd_op.cast: (1x100x-1x-1xf32) <- (1x100x-1x-1xb)
        cast_18 = paddle._C_ops.cast(greater_than_1, paddle.float32)
        del greater_than_1

        # pd_op.flatten: (100x-1x-1xf32) <- (1x100x-1x-1xf32)
        flatten_51 = paddle._C_ops.flatten(cast_18, 0, 1)
        del cast_18

        # pd_op.cast: (100x-1x-1xi32) <- (100x-1x-1xf32)
        cast_0 = paddle._C_ops.cast(flatten_51, paddle.int32)
        del flatten_51

        # pd_op.unsqueeze: (1x100x1xi64) <- (1x100xi64, 1xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(remainder_0, full_int_array_4)
        del remainder_0

        # pd_op.cast: (1x100x1xf32) <- (1x100x1xi64)
        cast_19 = paddle._C_ops.cast(unsqueeze_21, paddle.float32)
        del unsqueeze_21

        # pd_op.unsqueeze: (1x100x1xf32) <- (1x100xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(topk_2, full_int_array_4)
        del full_int_array_4, topk_2

        # builtin.combine: ([1x100x1xf32, 1x100x1xf32, 1x100x4xf32]) <- (1x100x1xf32, 1x100x1xf32, 1x100x4xf32)
        combine_79 = [cast_19, unsqueeze_22, gather_nd_2]
        del cast_19, gather_nd_2, unsqueeze_22

        # pd_op.concat: (1x100x6xf32) <- ([1x100x1xf32, 1x100x1xf32, 1x100x4xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_79, full_14)
        del combine_79, full_14

        # pd_op.full: (xi64) <- ()
        full_32 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_32,
            [],
            paddle.int64,
            [float("100")],
            paddle.framework._current_expected_place(),
        )
        del full_32

        # pd_op.cast: (xi32) <- (xi64)
        cast_20 = paddle._C_ops.cast(assign_value__9, paddle.int32)
        del assign_value__9

        # pd_op.tile: (1xi32) <- (xi32, 1xi64)
        tile_0 = paddle._C_ops.tile(cast_20, full_int_array_3)
        del cast_20, full_int_array_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [-1, 6]

        # pd_op.reshape: (100x6xf32) <- (1x100x6xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(concat_3, full_int_array_20)
        del concat_3, full_int_array_20

        return reshape_0, tile_0, cast_0
