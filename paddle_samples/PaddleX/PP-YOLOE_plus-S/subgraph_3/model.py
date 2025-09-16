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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
    ):
        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_0, data_1, full_1, dtype="int64")
        del data_1

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)
        del arange_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_2, float("0.5"), True)
        del cast_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_3, float("0"), True)
        del scale_0

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_0, data_0, full_1, dtype="int64")
        del data_0

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)
        del arange_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_1, full_2, float("0.5"), True)
        del cast_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_3, float("0"), True)
        del scale_2

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [scale_3, scale_1]
        del scale_1, scale_3

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_2, float("-80"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_2, float("-80"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(split_1, full_2, float("80"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(split_0, full_2, float("80"), True)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32)
        combine_1 = [scale_4, scale_5, scale_6, scale_7]
        del scale_4, scale_5, scale_6, scale_7

        # pd_op.stack: (-1x-1x4xf32) <- ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_2 = [split_1, split_0]
        del split_0, split_1

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1x-1x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_0)
        del stack_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (-1x2xf32) <- (-1x-1x2xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_1, full_int_array_1)
        del stack_1

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_0 = paddle._C_ops.shape64(reshape_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_3 = [slice_0, full_4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.full_with_tensor: (-1x1xf32) <- (1xf32, 2xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_3, stack_2, paddle.float32
        )
        del full_3, stack_2

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_2 = paddle.arange(full_0, data_3, full_1, dtype="int64")
        del data_3

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)
        del arange_2

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_2, full_2, float("0.5"), True)
        del cast_2

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_5, float("0"), True)
        del scale_8

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_3 = paddle.arange(full_0, data_2, full_1, dtype="int64")
        del data_2

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_3 = paddle._C_ops.cast(arange_3, paddle.float32)
        del arange_3

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(cast_3, full_2, float("0.5"), True)
        del cast_3

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(scale_10, full_5, float("0"), True)
        del scale_10

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_4 = [scale_11, scale_9]
        del scale_11, scale_9

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_4)
        del combine_4

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(split_3, full_2, float("-40"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(split_2, full_2, float("-40"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_3, full_2, float("40"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_2, full_2, float("40"), True)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32)
        combine_5 = [scale_12, scale_13, scale_14, scale_15]
        del scale_12, scale_13, scale_14, scale_15

        # pd_op.stack: (-1x-1x4xf32) <- ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32])
        stack_3 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_6 = [split_3, split_2]
        del split_2, split_3

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_4 = paddle._C_ops.stack(combine_6, -1)
        del combine_6

        # pd_op.reshape: (-1x4xf32) <- (-1x-1x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(stack_3, full_int_array_0)
        del stack_3

        # pd_op.reshape: (-1x2xf32) <- (-1x-1x2xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(stack_4, full_int_array_1)
        del stack_4

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_1 = paddle._C_ops.shape64(reshape_2)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_7 = [slice_1, full_4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.full_with_tensor: (-1x1xf32) <- (1xf32, 2xi64)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(
            full_5, stack_5, paddle.float32
        )
        del full_5, stack_5

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_4 = paddle.arange(full_0, data_5, full_1, dtype="int64")
        del data_5

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_4 = paddle._C_ops.cast(arange_4, paddle.float32)
        del arange_4

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(cast_4, full_2, float("0.5"), True)
        del cast_4

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(scale_16, full_6, float("0"), True)
        del scale_16

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_5 = paddle.arange(full_0, data_4, full_1, dtype="int64")
        del data_4, full_0, full_1

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_5 = paddle._C_ops.cast(arange_5, paddle.float32)
        del arange_5

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_5, full_2, float("0.5"), True)
        del cast_5

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_6, float("0"), True)
        del scale_18

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_8 = [scale_19, scale_17]
        del scale_17, scale_19

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_8)
        del combine_8

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(split_5, full_2, float("-20"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(split_4, full_2, float("-20"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(split_5, full_2, float("20"), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(split_4, full_2, float("20"), True)
        del full_2

        # builtin.combine: ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32)
        combine_9 = [scale_20, scale_21, scale_22, scale_23]
        del scale_20, scale_21, scale_22, scale_23

        # pd_op.stack: (-1x-1x4xf32) <- ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32])
        stack_6 = paddle._C_ops.stack(combine_9, -1)
        del combine_9

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_10 = [split_5, split_4]
        del split_4, split_5

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_7 = paddle._C_ops.stack(combine_10, -1)
        del combine_10

        # pd_op.reshape: (-1x4xf32) <- (-1x-1x4xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(stack_6, full_int_array_0)
        del full_int_array_0, stack_6

        # pd_op.reshape: (-1x2xf32) <- (-1x-1x2xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(stack_7, full_int_array_1)
        del full_int_array_1, stack_7

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_2 = paddle._C_ops.shape64(reshape_4)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_2

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_11 = [slice_2, full_4]
        del full_4

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.full_with_tensor: (-1x1xf32) <- (1xf32, 2xi64)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(
            full_6, stack_8, paddle.float32
        )
        del full_6, stack_8

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x4xf32, -1x4xf32, -1x4xf32]) <- (-1x4xf32, -1x4xf32, -1x4xf32)
        combine_12 = [reshape_0, reshape_2, reshape_4]
        del reshape_0, reshape_2, reshape_4

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_12, full_7)
        del combine_12

        # builtin.combine: ([-1x2xf32, -1x2xf32, -1x2xf32]) <- (-1x2xf32, -1x2xf32, -1x2xf32)
        combine_13 = [reshape_1, reshape_3, reshape_5]
        del reshape_1, reshape_3, reshape_5

        # pd_op.concat: (-1x2xf32) <- ([-1x2xf32, -1x2xf32, -1x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_13, full_7)
        del combine_13

        # builtin.combine: ([-1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x1xf32, -1x1xf32, -1x1xf32)
        combine_14 = [full_with_tensor_0, full_with_tensor_1, full_with_tensor_2]
        del full_with_tensor_0, full_with_tensor_1, full_with_tensor_2

        # pd_op.concat: (-1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_14, full_7)
        del combine_14, full_7

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_4

        # pd_op.pool2d: (8x384x1x1xf32) <- (8x384x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            data_6,
            full_int_array_4,
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

        # pd_op.conv2d: (8x384x1x1xf32) <- (8x384x1x1xf32, 384x384x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            pool2d_0, parameter_53, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_53

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, -1, 1, 1]

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_52, full_int_array_5)
        del parameter_52

        # pd_op.add: (8x384x1x1xf32) <- (8x384x1x1xf32, 1x384x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_6)

        # pd_op.sigmoid: (8x384x1x1xf32) <- (8x384x1x1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_0)
        del add_0

        # pd_op.multiply: (8x384x-1x-1xf32) <- (8x384x-1x-1xf32, 8x384x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(data_6, sigmoid_0)

        # pd_op.conv2d: (8x384x-1x-1xf32) <- (8x384x-1x-1xf32, 384x384x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            multiply_0, parameter_51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_51

        # pd_op.batch_norm_: (8x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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

        # pd_op.swish: (8x384x-1x-1xf32) <- (8x384x-1x-1xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.add: (8x384x-1x-1xf32) <- (8x384x-1x-1xf32, 8x384x-1x-1xf32)
        add_1 = paddle._C_ops.add(swish_0, data_6)

        # pd_op.conv2d: (8x4x-1x-1xf32) <- (8x384x-1x-1xf32, 4x384x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            add_1, parameter_46, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_46

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_45, full_int_array_5)
        del parameter_45

        # pd_op.add: (8x4x-1x-1xf32) <- (8x4x-1x-1xf32, 1x4x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_2, reshape_7)

        # pd_op.conv2d: (8x384x1x1xf32) <- (8x384x1x1xf32, 384x384x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            pool2d_0, parameter_44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_44

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_43, full_int_array_5)
        del parameter_43

        # pd_op.add: (8x384x1x1xf32) <- (8x384x1x1xf32, 1x384x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_3, reshape_8)

        # pd_op.sigmoid: (8x384x1x1xf32) <- (8x384x1x1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_3)
        del add_3

        # pd_op.multiply: (8x384x-1x-1xf32) <- (8x384x-1x-1xf32, 8x384x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(data_6, sigmoid_1)
        del data_6

        # pd_op.conv2d: (8x384x-1x-1xf32) <- (8x384x-1x-1xf32, 384x384x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            multiply_1, parameter_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42

        # pd_op.batch_norm_: (8x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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

        # pd_op.swish: (8x384x-1x-1xf32) <- (8x384x-1x-1xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.conv2d: (8x68x-1x-1xf32) <- (8x384x-1x-1xf32, 68x384x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            swish_1, parameter_37, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_37

        # pd_op.reshape: (1x68x1x1xf32) <- (68xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_36, full_int_array_5)
        del parameter_36

        # pd_op.add: (8x68x-1x-1xf32) <- (8x68x-1x-1xf32, 1x68x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_5, reshape_9)

        # pd_op.sigmoid: (8x4x-1x-1xf32) <- (8x4x-1x-1xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_2)
        del add_2

        # pd_op.flatten: (8x4x-1xf32) <- (8x4x-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(sigmoid_2, 2, 3)

        # pd_op.transpose: (8x-1x4xf32) <- (8x4x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.flatten: (8x68x-1xf32) <- (8x68x-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(add_4, 2, 3)

        # pd_op.transpose: (8x-1x68xf32) <- (8x68x-1xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.pool2d: (8x192x1x1xf32) <- (8x192x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            data_7,
            full_int_array_4,
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

        # pd_op.conv2d: (8x192x1x1xf32) <- (8x192x1x1xf32, 192x192x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            pool2d_1, parameter_35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_34, full_int_array_5)
        del parameter_34

        # pd_op.add: (8x192x1x1xf32) <- (8x192x1x1xf32, 1x192x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_6, reshape_10)

        # pd_op.sigmoid: (8x192x1x1xf32) <- (8x192x1x1xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(add_5)
        del add_5

        # pd_op.multiply: (8x192x-1x-1xf32) <- (8x192x-1x-1xf32, 8x192x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(data_7, sigmoid_3)

        # pd_op.conv2d: (8x192x-1x-1xf32) <- (8x192x-1x-1xf32, 192x192x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            multiply_2, parameter_33, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_33

        # pd_op.batch_norm_: (8x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (8x192x-1x-1xf32) <- (8x192x-1x-1xf32)
        swish_2 = paddle._C_ops.swish(batch_norm__12)

        # pd_op.add: (8x192x-1x-1xf32) <- (8x192x-1x-1xf32, 8x192x-1x-1xf32)
        add_6 = paddle._C_ops.add(swish_2, data_7)

        # pd_op.conv2d: (8x4x-1x-1xf32) <- (8x192x-1x-1xf32, 4x192x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            add_6, parameter_28, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_28

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_27, full_int_array_5)
        del parameter_27

        # pd_op.add: (8x4x-1x-1xf32) <- (8x4x-1x-1xf32, 1x4x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_8, reshape_11)

        # pd_op.conv2d: (8x192x1x1xf32) <- (8x192x1x1xf32, 192x192x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            pool2d_1, parameter_26, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_26

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_25, full_int_array_5)
        del parameter_25

        # pd_op.add: (8x192x1x1xf32) <- (8x192x1x1xf32, 1x192x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_9, reshape_12)

        # pd_op.sigmoid: (8x192x1x1xf32) <- (8x192x1x1xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(add_8)
        del add_8

        # pd_op.multiply: (8x192x-1x-1xf32) <- (8x192x-1x-1xf32, 8x192x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(data_7, sigmoid_4)
        del data_7

        # pd_op.conv2d: (8x192x-1x-1xf32) <- (8x192x-1x-1xf32, 192x192x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            multiply_3, parameter_24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_24

        # pd_op.batch_norm_: (8x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (8x192x-1x-1xf32) <- (8x192x-1x-1xf32)
        swish_3 = paddle._C_ops.swish(batch_norm__18)

        # pd_op.conv2d: (8x68x-1x-1xf32) <- (8x192x-1x-1xf32, 68x192x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            swish_3, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_19

        # pd_op.reshape: (1x68x1x1xf32) <- (68xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_18, full_int_array_5)
        del parameter_18

        # pd_op.add: (8x68x-1x-1xf32) <- (8x68x-1x-1xf32, 1x68x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_11, reshape_13)

        # pd_op.sigmoid: (8x4x-1x-1xf32) <- (8x4x-1x-1xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(add_7)
        del add_7

        # pd_op.flatten: (8x4x-1xf32) <- (8x4x-1x-1xf32)
        flatten_2 = paddle._C_ops.flatten(sigmoid_5, 2, 3)

        # pd_op.transpose: (8x-1x4xf32) <- (8x4x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.flatten: (8x68x-1xf32) <- (8x68x-1x-1xf32)
        flatten_3 = paddle._C_ops.flatten(add_9, 2, 3)

        # pd_op.transpose: (8x-1x68xf32) <- (8x68x-1xf32)
        transpose_3 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.pool2d: (8x96x1x1xf32) <- (8x96x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            data_8,
            full_int_array_4,
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

        # pd_op.conv2d: (8x96x1x1xf32) <- (8x96x1x1xf32, 96x96x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            pool2d_2, parameter_17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_16, full_int_array_5)
        del parameter_16

        # pd_op.add: (8x96x1x1xf32) <- (8x96x1x1xf32, 1x96x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_12, reshape_14)

        # pd_op.sigmoid: (8x96x1x1xf32) <- (8x96x1x1xf32)
        sigmoid_6 = paddle._C_ops.sigmoid(add_10)
        del add_10

        # pd_op.multiply: (8x96x-1x-1xf32) <- (8x96x-1x-1xf32, 8x96x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(data_8, sigmoid_6)

        # pd_op.conv2d: (8x96x-1x-1xf32) <- (8x96x-1x-1xf32, 96x96x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            multiply_4, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15

        # pd_op.batch_norm_: (8x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (8x96x-1x-1xf32) <- (8x96x-1x-1xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__24)

        # pd_op.add: (8x96x-1x-1xf32) <- (8x96x-1x-1xf32, 8x96x-1x-1xf32)
        add_11 = paddle._C_ops.add(swish_4, data_8)

        # pd_op.conv2d: (8x4x-1x-1xf32) <- (8x96x-1x-1xf32, 4x96x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            add_11, parameter_10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_10

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_9, full_int_array_5)
        del parameter_9

        # pd_op.add: (8x4x-1x-1xf32) <- (8x4x-1x-1xf32, 1x4x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_14, reshape_15)

        # pd_op.conv2d: (8x96x1x1xf32) <- (8x96x1x1xf32, 96x96x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            pool2d_2, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_8

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_7, full_int_array_5)
        del parameter_7

        # pd_op.add: (8x96x1x1xf32) <- (8x96x1x1xf32, 1x96x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_15, reshape_16)

        # pd_op.sigmoid: (8x96x1x1xf32) <- (8x96x1x1xf32)
        sigmoid_7 = paddle._C_ops.sigmoid(add_13)
        del add_13

        # pd_op.multiply: (8x96x-1x-1xf32) <- (8x96x-1x-1xf32, 8x96x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(data_8, sigmoid_7)
        del data_8

        # pd_op.conv2d: (8x96x-1x-1xf32) <- (8x96x-1x-1xf32, 96x96x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            multiply_5, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_6

        # pd_op.batch_norm_: (8x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (8x96x-1x-1xf32) <- (8x96x-1x-1xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__30)

        # pd_op.conv2d: (8x68x-1x-1xf32) <- (8x96x-1x-1xf32, 68x96x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            swish_5, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.reshape: (1x68x1x1xf32) <- (68xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_0, full_int_array_5)
        del full_int_array_5, parameter_0

        # pd_op.add: (8x68x-1x-1xf32) <- (8x68x-1x-1xf32, 1x68x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_17, reshape_17)

        # pd_op.sigmoid: (8x4x-1x-1xf32) <- (8x4x-1x-1xf32)
        sigmoid_8 = paddle._C_ops.sigmoid(add_12)
        del add_12

        # pd_op.flatten: (8x4x-1xf32) <- (8x4x-1x-1xf32)
        flatten_4 = paddle._C_ops.flatten(sigmoid_8, 2, 3)

        # pd_op.transpose: (8x-1x4xf32) <- (8x4x-1xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])
        del flatten_4

        # pd_op.flatten: (8x68x-1xf32) <- (8x68x-1x-1xf32)
        flatten_5 = paddle._C_ops.flatten(add_14, 2, 3)

        # pd_op.transpose: (8x-1x68xf32) <- (8x68x-1xf32)
        transpose_5 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])
        del flatten_5

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_8

        # builtin.combine: ([8x-1x4xf32, 8x-1x4xf32, 8x-1x4xf32]) <- (8x-1x4xf32, 8x-1x4xf32, 8x-1x4xf32)
        combine_15 = [transpose_0, transpose_2, transpose_4]

        # pd_op.concat: (8x-1x4xf32) <- ([8x-1x4xf32, 8x-1x4xf32, 8x-1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_15, full_8)
        del combine_15

        # builtin.combine: ([8x-1x68xf32, 8x-1x68xf32, 8x-1x68xf32]) <- (8x-1x68xf32, 8x-1x68xf32, 8x-1x68xf32)
        combine_16 = [transpose_1, transpose_3, transpose_5]

        # pd_op.concat: (8x-1x68xf32) <- ([8x-1x68xf32, 8x-1x68xf32, 8x-1x68xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_16, full_8)
        del (
            add_1,
            add_11,
            add_14,
            add_4,
            add_6,
            add_9,
            assign_0,
            assign_1,
            assign_2,
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
            batch_norm__4,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            combine_16,
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
            conv2d_2,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            full_8,
            full_int_array_4,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
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
            reshape_6,
            reshape_7,
            reshape_8,
            reshape_9,
            sigmoid_0,
            sigmoid_1,
            sigmoid_2,
            sigmoid_3,
            sigmoid_4,
            sigmoid_5,
            sigmoid_6,
            sigmoid_7,
            sigmoid_8,
            slice_0,
            slice_1,
            slice_2,
            swish_0,
            swish_1,
            swish_2,
            swish_3,
            swish_4,
            swish_5,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
        )

        return concat_0, concat_1, concat_2, concat_3, concat_4
