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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("32"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (32xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="int64")
        del full_1

        # pd_op.cast: (32xf32) <- (32xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)
        del arange_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (32xf32) <- (32xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_3, float("0.5"), True)
        del cast_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (32xf32) <- (32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, float("0"), True)
        del full_4, scale_0

        # builtin.combine: ([32xf32, 32xf32]) <- (32xf32, 32xf32)
        combine_0 = [scale_1, scale_1]
        del scale_1

        # pd_op.meshgrid: ([32x32xf32, 32x32xf32]) <- ([32xf32, 32xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (32x32xf32, 32x32xf32) <- ([32x32xf32, 32x32xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # builtin.combine: ([32x32xf32, 32x32xf32]) <- (32x32xf32, 32x32xf32)
        combine_1 = [split_1, split_0]
        del split_0, split_1

        # pd_op.stack: (32x32x2xf32) <- ([32x32xf32, 32x32xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.cast: (32x32x2xf32) <- (32x32x2xf32)
        cast_1 = paddle._C_ops.cast(stack_0, paddle.float32)
        del stack_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [1, -1, 2]

        # pd_op.reshape: (1x1024x2xf32) <- (32x32x2xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(cast_1, full_int_array_0)
        del cast_1

        # pd_op.full: (1x1024x1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1, 1024, 1],
            float("32"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("64"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (64xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_6, full_2, dtype="int64")
        del full_6

        # pd_op.cast: (64xf32) <- (64xi64)
        cast_2 = paddle._C_ops.cast(arange_1, paddle.float32)
        del arange_1

        # pd_op.scale: (64xf32) <- (64xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_2, full_3, float("0.5"), True)
        del cast_2

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (64xf32) <- (64xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_7, float("0"), True)
        del full_7, scale_2

        # builtin.combine: ([64xf32, 64xf32]) <- (64xf32, 64xf32)
        combine_2 = [scale_3, scale_3]
        del scale_3

        # pd_op.meshgrid: ([64x64xf32, 64x64xf32]) <- ([64xf32, 64xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_2)
        del combine_2

        # builtin.split: (64x64xf32, 64x64xf32) <- ([64x64xf32, 64x64xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # builtin.combine: ([64x64xf32, 64x64xf32]) <- (64x64xf32, 64x64xf32)
        combine_3 = [split_3, split_2]
        del split_2, split_3

        # pd_op.stack: (64x64x2xf32) <- ([64x64xf32, 64x64xf32])
        stack_1 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.cast: (64x64x2xf32) <- (64x64x2xf32)
        cast_3 = paddle._C_ops.cast(stack_1, paddle.float32)
        del stack_1

        # pd_op.reshape: (1x4096x2xf32) <- (64x64x2xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(cast_3, full_int_array_0)
        del cast_3

        # pd_op.full: (1x4096x1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1, 4096, 1],
            float("16"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("128"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (128xi64) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_9, full_2, dtype="int64")
        del full_0, full_2, full_9

        # pd_op.cast: (128xf32) <- (128xi64)
        cast_4 = paddle._C_ops.cast(arange_2, paddle.float32)
        del arange_2

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_4, full_3, float("0.5"), True)
        del cast_4, full_3

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_10, float("0"), True)
        del full_10, scale_4

        # builtin.combine: ([128xf32, 128xf32]) <- (128xf32, 128xf32)
        combine_4 = [scale_5, scale_5]
        del scale_5

        # pd_op.meshgrid: ([128x128xf32, 128x128xf32]) <- ([128xf32, 128xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_4)
        del combine_4

        # builtin.split: (128x128xf32, 128x128xf32) <- ([128x128xf32, 128x128xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # builtin.combine: ([128x128xf32, 128x128xf32]) <- (128x128xf32, 128x128xf32)
        combine_5 = [split_5, split_4]
        del split_4, split_5

        # pd_op.stack: (128x128x2xf32) <- ([128x128xf32, 128x128xf32])
        stack_2 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.cast: (128x128x2xf32) <- (128x128x2xf32)
        cast_5 = paddle._C_ops.cast(stack_2, paddle.float32)
        del stack_2

        # pd_op.reshape: (1x16384x2xf32) <- (128x128x2xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(cast_5, full_int_array_0)
        del cast_5, full_int_array_0

        # pd_op.full: (1x16384x1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1, 16384, 1],
            float("8"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_12

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_12

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_12

        # builtin.combine: ([1x1024x2xf32, 1x4096x2xf32, 1x16384x2xf32]) <- (1x1024x2xf32, 1x4096x2xf32, 1x16384x2xf32)
        combine_6 = [reshape_0, reshape_1, reshape_2]
        del reshape_0, reshape_1, reshape_2

        # pd_op.concat: (1x21504x2xf32) <- ([1x1024x2xf32, 1x4096x2xf32, 1x16384x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_6, full_12)
        del combine_6

        # builtin.combine: ([1x1024x1xf32, 1x4096x1xf32, 1x16384x1xf32]) <- (1x1024x1xf32, 1x4096x1xf32, 1x16384x1xf32)
        combine_7 = [full_5, full_8, full_11]
        del full_11, full_5, full_8

        # pd_op.concat: (1x21504x1xf32) <- ([1x1024x1xf32, 1x4096x1xf32, 1x16384x1xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_7, full_12)
        del combine_7

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_1

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_1

        # pd_op.pool2d: (1x768x1x1xf32) <- (1x768x32x32xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            data_0,
            full_int_array_1,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )

        # pd_op.conv2d: (1x768x1x1xf32) <- (1x768x1x1xf32, 768x768x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            pool2d_0, parameter_80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_80

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, -1, 1, 1]

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_79, full_int_array_2)
        del parameter_79

        # pd_op.add: (1x768x1x1xf32) <- (1x768x1x1xf32, 1x768x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_3)

        # pd_op.sigmoid: (1x768x1x1xf32) <- (1x768x1x1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_0)
        del add_0

        # pd_op.multiply: (1x768x32x32xf32) <- (1x768x32x32xf32, 1x768x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(data_0, sigmoid_0)

        # pd_op.conv2d: (1x768x32x32xf32) <- (1x768x32x32xf32, 768x768x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            multiply_0, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_78

        # pd_op.batch_norm_: (1x768x32x32xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (1x768x32x32xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_77,
                parameter_76,
                parameter_75,
                parameter_74,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_74, parameter_75, parameter_76, parameter_77

        # pd_op.swish: (1x768x32x32xf32) <- (1x768x32x32xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.add: (1x768x32x32xf32) <- (1x768x32x32xf32, 1x768x32x32xf32)
        add_1 = paddle._C_ops.add(swish_0, data_0)

        # pd_op.conv2d: (1x15x32x32xf32) <- (1x768x32x32xf32, 15x768x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            add_1, parameter_73, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_73

        # pd_op.reshape: (1x15x1x1xf32) <- (15xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_72, full_int_array_2)
        del parameter_72

        # pd_op.add: (1x15x32x32xf32) <- (1x15x32x32xf32, 1x15x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_2, reshape_4)

        # pd_op.conv2d: (1x768x1x1xf32) <- (1x768x1x1xf32, 768x768x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            pool2d_0, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_70, full_int_array_2)
        del parameter_70

        # pd_op.add: (1x768x1x1xf32) <- (1x768x1x1xf32, 1x768x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_3, reshape_5)

        # pd_op.sigmoid: (1x768x1x1xf32) <- (1x768x1x1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_3)
        del add_3

        # pd_op.multiply: (1x768x32x32xf32) <- (1x768x32x32xf32, 1x768x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(data_0, sigmoid_1)

        # pd_op.conv2d: (1x768x32x32xf32) <- (1x768x32x32xf32, 768x768x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            multiply_1, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_69

        # pd_op.batch_norm_: (1x768x32x32xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (1x768x32x32xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_65, parameter_66, parameter_67, parameter_68

        # pd_op.swish: (1x768x32x32xf32) <- (1x768x32x32xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.conv2d: (1x4x32x32xf32) <- (1x768x32x32xf32, 4x768x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            swish_1, parameter_64, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_63, full_int_array_2)
        del parameter_63

        # pd_op.add: (1x4x32x32xf32) <- (1x4x32x32xf32, 1x4x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_5, reshape_6)

        # pd_op.conv2d: (1x768x1x1xf32) <- (1x768x1x1xf32, 768x768x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            pool2d_0, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_62

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_61, full_int_array_2)
        del parameter_61

        # pd_op.add: (1x768x1x1xf32) <- (1x768x1x1xf32, 1x768x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_6, reshape_7)

        # pd_op.sigmoid: (1x768x1x1xf32) <- (1x768x1x1xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_5)
        del add_5

        # pd_op.multiply: (1x768x32x32xf32) <- (1x768x32x32xf32, 1x768x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(data_0, sigmoid_2)
        del data_0

        # pd_op.conv2d: (1x768x32x32xf32) <- (1x768x32x32xf32, 768x768x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            multiply_2, parameter_60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_60

        # pd_op.batch_norm_: (1x768x32x32xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (1x768x32x32xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_59,
                parameter_58,
                parameter_57,
                parameter_56,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_56, parameter_57, parameter_58, parameter_59

        # pd_op.swish: (1x768x32x32xf32) <- (1x768x32x32xf32)
        swish_2 = paddle._C_ops.swish(batch_norm__12)

        # pd_op.conv2d: (1x91x32x32xf32) <- (1x768x32x32xf32, 91x768x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            swish_2, parameter_55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_55

        # pd_op.reshape: (1x91x1x1xf32) <- (91xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_54, full_int_array_2)
        del parameter_54

        # pd_op.add: (1x91x32x32xf32) <- (1x91x32x32xf32, 1x91x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_8, reshape_8)

        # pd_op.sigmoid: (1x15x32x32xf32) <- (1x15x32x32xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(add_2)
        del add_2

        # pd_op.flatten: (1x15x1024xf32) <- (1x15x32x32xf32)
        flatten_0 = paddle._C_ops.flatten(sigmoid_3, 2, 3)

        # pd_op.transpose: (1x1024x15xf32) <- (1x15x1024xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.flatten: (1x4x1024xf32) <- (1x4x32x32xf32)
        flatten_1 = paddle._C_ops.flatten(add_4, 2, 3)

        # pd_op.transpose: (1x1024x4xf32) <- (1x4x1024xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.flatten: (1x91x1024xf32) <- (1x91x32x32xf32)
        flatten_2 = paddle._C_ops.flatten(add_6, 2, 3)

        # pd_op.transpose: (1x1024x91xf32) <- (1x91x1024xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.pool2d: (1x384x1x1xf32) <- (1x384x64x64xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            data_1,
            full_int_array_1,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )

        # pd_op.conv2d: (1x384x1x1xf32) <- (1x384x1x1xf32, 384x384x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            pool2d_1, parameter_53, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_53

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_52, full_int_array_2)
        del parameter_52

        # pd_op.add: (1x384x1x1xf32) <- (1x384x1x1xf32, 1x384x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_9, reshape_9)

        # pd_op.sigmoid: (1x384x1x1xf32) <- (1x384x1x1xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(add_7)
        del add_7

        # pd_op.multiply: (1x384x64x64xf32) <- (1x384x64x64xf32, 1x384x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(data_1, sigmoid_4)

        # pd_op.conv2d: (1x384x64x64xf32) <- (1x384x64x64xf32, 384x384x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            multiply_3, parameter_51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_51

        # pd_op.batch_norm_: (1x384x64x64xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x64x64xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_50,
                parameter_49,
                parameter_48,
                parameter_47,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_47, parameter_48, parameter_49, parameter_50

        # pd_op.swish: (1x384x64x64xf32) <- (1x384x64x64xf32)
        swish_3 = paddle._C_ops.swish(batch_norm__18)

        # pd_op.add: (1x384x64x64xf32) <- (1x384x64x64xf32, 1x384x64x64xf32)
        add_8 = paddle._C_ops.add(swish_3, data_1)

        # pd_op.conv2d: (1x15x64x64xf32) <- (1x384x64x64xf32, 15x384x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_8, parameter_46, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_46

        # pd_op.reshape: (1x15x1x1xf32) <- (15xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_45, full_int_array_2)
        del parameter_45

        # pd_op.add: (1x15x64x64xf32) <- (1x15x64x64xf32, 1x15x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_11, reshape_10)

        # pd_op.conv2d: (1x384x1x1xf32) <- (1x384x1x1xf32, 384x384x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            pool2d_1, parameter_44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_44

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_43, full_int_array_2)
        del parameter_43

        # pd_op.add: (1x384x1x1xf32) <- (1x384x1x1xf32, 1x384x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_12, reshape_11)

        # pd_op.sigmoid: (1x384x1x1xf32) <- (1x384x1x1xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(add_10)
        del add_10

        # pd_op.multiply: (1x384x64x64xf32) <- (1x384x64x64xf32, 1x384x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(data_1, sigmoid_5)

        # pd_op.conv2d: (1x384x64x64xf32) <- (1x384x64x64xf32, 384x384x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            multiply_4, parameter_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42

        # pd_op.batch_norm_: (1x384x64x64xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x64x64xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_41,
                parameter_40,
                parameter_39,
                parameter_38,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_38, parameter_39, parameter_40, parameter_41

        # pd_op.swish: (1x384x64x64xf32) <- (1x384x64x64xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__24)

        # pd_op.conv2d: (1x4x64x64xf32) <- (1x384x64x64xf32, 4x384x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            swish_4, parameter_37, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_37

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_36, full_int_array_2)
        del parameter_36

        # pd_op.add: (1x4x64x64xf32) <- (1x4x64x64xf32, 1x4x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_14, reshape_12)

        # pd_op.conv2d: (1x384x1x1xf32) <- (1x384x1x1xf32, 384x384x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            pool2d_1, parameter_35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_34, full_int_array_2)
        del parameter_34

        # pd_op.add: (1x384x1x1xf32) <- (1x384x1x1xf32, 1x384x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_15, reshape_13)

        # pd_op.sigmoid: (1x384x1x1xf32) <- (1x384x1x1xf32)
        sigmoid_6 = paddle._C_ops.sigmoid(add_12)
        del add_12

        # pd_op.multiply: (1x384x64x64xf32) <- (1x384x64x64xf32, 1x384x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(data_1, sigmoid_6)
        del data_1

        # pd_op.conv2d: (1x384x64x64xf32) <- (1x384x64x64xf32, 384x384x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            multiply_5, parameter_33, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_33

        # pd_op.batch_norm_: (1x384x64x64xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x64x64xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_32,
                parameter_31,
                parameter_30,
                parameter_29,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_29, parameter_30, parameter_31, parameter_32

        # pd_op.swish: (1x384x64x64xf32) <- (1x384x64x64xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__30)

        # pd_op.conv2d: (1x91x64x64xf32) <- (1x384x64x64xf32, 91x384x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            swish_5, parameter_28, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_28

        # pd_op.reshape: (1x91x1x1xf32) <- (91xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_27, full_int_array_2)
        del parameter_27

        # pd_op.add: (1x91x64x64xf32) <- (1x91x64x64xf32, 1x91x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_17, reshape_14)

        # pd_op.sigmoid: (1x15x64x64xf32) <- (1x15x64x64xf32)
        sigmoid_7 = paddle._C_ops.sigmoid(add_9)
        del add_9

        # pd_op.flatten: (1x15x4096xf32) <- (1x15x64x64xf32)
        flatten_3 = paddle._C_ops.flatten(sigmoid_7, 2, 3)

        # pd_op.transpose: (1x4096x15xf32) <- (1x15x4096xf32)
        transpose_3 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.flatten: (1x4x4096xf32) <- (1x4x64x64xf32)
        flatten_4 = paddle._C_ops.flatten(add_11, 2, 3)

        # pd_op.transpose: (1x4096x4xf32) <- (1x4x4096xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])
        del flatten_4

        # pd_op.flatten: (1x91x4096xf32) <- (1x91x64x64xf32)
        flatten_5 = paddle._C_ops.flatten(add_13, 2, 3)

        # pd_op.transpose: (1x4096x91xf32) <- (1x91x4096xf32)
        transpose_5 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])
        del flatten_5

        # pd_op.pool2d: (1x192x1x1xf32) <- (1x192x128x128xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            data_2,
            full_int_array_1,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )

        # pd_op.conv2d: (1x192x1x1xf32) <- (1x192x1x1xf32, 192x192x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            pool2d_2, parameter_26, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_26

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_25, full_int_array_2)
        del parameter_25

        # pd_op.add: (1x192x1x1xf32) <- (1x192x1x1xf32, 1x192x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_18, reshape_15)

        # pd_op.sigmoid: (1x192x1x1xf32) <- (1x192x1x1xf32)
        sigmoid_8 = paddle._C_ops.sigmoid(add_14)
        del add_14

        # pd_op.multiply: (1x192x128x128xf32) <- (1x192x128x128xf32, 1x192x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(data_2, sigmoid_8)

        # pd_op.conv2d: (1x192x128x128xf32) <- (1x192x128x128xf32, 192x192x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            multiply_6, parameter_24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_24

        # pd_op.batch_norm_: (1x192x128x128xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x128x128xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_20, parameter_21, parameter_22, parameter_23

        # pd_op.swish: (1x192x128x128xf32) <- (1x192x128x128xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__36)

        # pd_op.add: (1x192x128x128xf32) <- (1x192x128x128xf32, 1x192x128x128xf32)
        add_15 = paddle._C_ops.add(swish_6, data_2)

        # pd_op.conv2d: (1x15x128x128xf32) <- (1x192x128x128xf32, 15x192x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            add_15, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_19

        # pd_op.reshape: (1x15x1x1xf32) <- (15xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_18, full_int_array_2)
        del parameter_18

        # pd_op.add: (1x15x128x128xf32) <- (1x15x128x128xf32, 1x15x1x1xf32)
        add_16 = paddle._C_ops.add(conv2d_20, reshape_16)

        # pd_op.conv2d: (1x192x1x1xf32) <- (1x192x1x1xf32, 192x192x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            pool2d_2, parameter_17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_16, full_int_array_2)
        del parameter_16

        # pd_op.add: (1x192x1x1xf32) <- (1x192x1x1xf32, 1x192x1x1xf32)
        add_17 = paddle._C_ops.add(conv2d_21, reshape_17)

        # pd_op.sigmoid: (1x192x1x1xf32) <- (1x192x1x1xf32)
        sigmoid_9 = paddle._C_ops.sigmoid(add_17)
        del add_17

        # pd_op.multiply: (1x192x128x128xf32) <- (1x192x128x128xf32, 1x192x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(data_2, sigmoid_9)

        # pd_op.conv2d: (1x192x128x128xf32) <- (1x192x128x128xf32, 192x192x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            multiply_7, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15

        # pd_op.batch_norm_: (1x192x128x128xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x128x128xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_14,
                parameter_13,
                parameter_12,
                parameter_11,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_11, parameter_12, parameter_13, parameter_14

        # pd_op.swish: (1x192x128x128xf32) <- (1x192x128x128xf32)
        swish_7 = paddle._C_ops.swish(batch_norm__42)

        # pd_op.conv2d: (1x4x128x128xf32) <- (1x192x128x128xf32, 4x192x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            swish_7, parameter_10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_10

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_9, full_int_array_2)
        del parameter_9

        # pd_op.add: (1x4x128x128xf32) <- (1x4x128x128xf32, 1x4x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_23, reshape_18)

        # pd_op.conv2d: (1x192x1x1xf32) <- (1x192x1x1xf32, 192x192x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            pool2d_2, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_8

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(parameter_7, full_int_array_2)
        del parameter_7

        # pd_op.add: (1x192x1x1xf32) <- (1x192x1x1xf32, 1x192x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_24, reshape_19)

        # pd_op.sigmoid: (1x192x1x1xf32) <- (1x192x1x1xf32)
        sigmoid_10 = paddle._C_ops.sigmoid(add_19)
        del add_19

        # pd_op.multiply: (1x192x128x128xf32) <- (1x192x128x128xf32, 1x192x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(data_2, sigmoid_10)
        del data_2

        # pd_op.conv2d: (1x192x128x128xf32) <- (1x192x128x128xf32, 192x192x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            multiply_8, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_6

        # pd_op.batch_norm_: (1x192x128x128xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x128x128xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.swish: (1x192x128x128xf32) <- (1x192x128x128xf32)
        swish_8 = paddle._C_ops.swish(batch_norm__48)

        # pd_op.conv2d: (1x91x128x128xf32) <- (1x192x128x128xf32, 91x192x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            swish_8, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.reshape: (1x91x1x1xf32) <- (91xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(parameter_0, full_int_array_2)
        del full_int_array_2, parameter_0

        # pd_op.add: (1x91x128x128xf32) <- (1x91x128x128xf32, 1x91x1x1xf32)
        add_20 = paddle._C_ops.add(conv2d_26, reshape_20)

        # pd_op.sigmoid: (1x15x128x128xf32) <- (1x15x128x128xf32)
        sigmoid_11 = paddle._C_ops.sigmoid(add_16)
        del add_16

        # pd_op.flatten: (1x15x16384xf32) <- (1x15x128x128xf32)
        flatten_6 = paddle._C_ops.flatten(sigmoid_11, 2, 3)

        # pd_op.transpose: (1x16384x15xf32) <- (1x15x16384xf32)
        transpose_6 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])
        del flatten_6

        # pd_op.flatten: (1x4x16384xf32) <- (1x4x128x128xf32)
        flatten_7 = paddle._C_ops.flatten(add_18, 2, 3)

        # pd_op.transpose: (1x16384x4xf32) <- (1x4x16384xf32)
        transpose_7 = paddle._C_ops.transpose(flatten_7, [0, 2, 1])
        del flatten_7

        # pd_op.flatten: (1x91x16384xf32) <- (1x91x128x128xf32)
        flatten_8 = paddle._C_ops.flatten(add_20, 2, 3)

        # pd_op.transpose: (1x16384x91xf32) <- (1x91x16384xf32)
        transpose_8 = paddle._C_ops.transpose(flatten_8, [0, 2, 1])
        del flatten_8

        # builtin.combine: ([1x1024x15xf32, 1x4096x15xf32, 1x16384x15xf32]) <- (1x1024x15xf32, 1x4096x15xf32, 1x16384x15xf32)
        combine_8 = [transpose_0, transpose_3, transpose_6]

        # pd_op.concat: (1x21504x15xf32) <- ([1x1024x15xf32, 1x4096x15xf32, 1x16384x15xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_8, full_12)
        del combine_8

        # builtin.combine: ([1x1024x4xf32, 1x4096x4xf32, 1x16384x4xf32]) <- (1x1024x4xf32, 1x4096x4xf32, 1x16384x4xf32)
        combine_9 = [transpose_1, transpose_4, transpose_7]

        # pd_op.concat: (1x21504x4xf32) <- ([1x1024x4xf32, 1x4096x4xf32, 1x16384x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_9, full_12)
        del combine_9

        # builtin.combine: ([1x1024x91xf32, 1x4096x91xf32, 1x16384x91xf32]) <- (1x1024x91xf32, 1x4096x91xf32, 1x16384x91xf32)
        combine_10 = [transpose_2, transpose_5, transpose_8]

        # pd_op.concat: (1x21504x91xf32) <- ([1x1024x91xf32, 1x4096x91xf32, 1x16384x91xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_10, full_12)
        del (
            add_1,
            add_11,
            add_13,
            add_15,
            add_18,
            add_20,
            add_4,
            add_6,
            add_8,
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__2,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__3,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
            batch_norm__5,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            combine_10,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_12,
            conv2d_13,
            conv2d_14,
            conv2d_15,
            conv2d_16,
            conv2d_17,
            conv2d_18,
            conv2d_19,
            conv2d_2,
            conv2d_20,
            conv2d_21,
            conv2d_22,
            conv2d_23,
            conv2d_24,
            conv2d_25,
            conv2d_26,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            full_12,
            full_int_array_1,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            pool2d_0,
            pool2d_1,
            pool2d_2,
            reshape_10,
            reshape_11,
            reshape_12,
            reshape_13,
            reshape_14,
            reshape_15,
            reshape_16,
            reshape_17,
            reshape_18,
            reshape_19,
            reshape_20,
            reshape_3,
            reshape_4,
            reshape_5,
            reshape_6,
            reshape_7,
            reshape_8,
            reshape_9,
            sigmoid_0,
            sigmoid_1,
            sigmoid_10,
            sigmoid_11,
            sigmoid_2,
            sigmoid_3,
            sigmoid_4,
            sigmoid_5,
            sigmoid_6,
            sigmoid_7,
            sigmoid_8,
            sigmoid_9,
            swish_0,
            swish_1,
            swish_2,
            swish_3,
            swish_4,
            swish_5,
            swish_6,
            swish_7,
            swish_8,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
        )

        return concat_0, concat_1, concat_2, concat_3, concat_4
