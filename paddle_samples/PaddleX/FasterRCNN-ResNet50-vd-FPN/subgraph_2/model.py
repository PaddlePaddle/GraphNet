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
    ):
        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_4, full_int_array_0)
        del parameter_4

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_0 = reshape_5

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_1 = reshape_5

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_2 = reshape_5

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_3 = reshape_5

        # pd_op.add: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 1x256x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_0, reshape_5)

        # pd_op.relu: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_1, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_1

        # pd_op.add: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 1x256x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_1, reshape_5)

        # pd_op.relu: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(add_11)
        del add_11

        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_2, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_2

        # pd_op.add: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 1x256x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_2, reshape_5)

        # pd_op.relu: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(add_12)
        del add_12

        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            data_3, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_3

        # pd_op.add: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 1x256x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_3, reshape_5)

        # pd_op.relu: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(add_13)
        del add_13

        # pd_op.conv2d: (4x256x16x-1xf32) <- (4x256x16x-1xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            data_4, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_4, parameter_5

        # pd_op.add: (4x256x16x-1xf32) <- (4x256x16x-1xf32, 1x256x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_4, reshape_5)

        # pd_op.relu: (4x256x16x-1xf32) <- (4x256x16x-1xf32)
        relu_4 = paddle._C_ops.relu(add_14)
        del add_14

        # pd_op.conv2d: (4x3x-1x-1xf32) <- (4x256x-1x-1xf32, 3x256x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_0, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x3x1x1xf32) <- (3xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_2, full_int_array_0)
        del parameter_2

        # pd_op.assign: (1x3x1x1xf32) <- (1x3x1x1xf32)
        assign_4 = reshape_6

        # pd_op.assign: (1x3x1x1xf32) <- (1x3x1x1xf32)
        assign_5 = reshape_6

        # pd_op.assign: (1x3x1x1xf32) <- (1x3x1x1xf32)
        assign_6 = reshape_6

        # pd_op.assign: (1x3x1x1xf32) <- (1x3x1x1xf32)
        assign_7 = reshape_6

        # pd_op.add: (4x3x-1x-1xf32) <- (4x3x-1x-1xf32, 1x3x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_5, reshape_6)

        # pd_op.conv2d: (4x12x-1x-1xf32) <- (4x256x-1x-1xf32, 12x256x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x12x1x1xf32) <- (12xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_0, full_int_array_0)
        del full_int_array_0, parameter_0

        # pd_op.assign: (1x12x1x1xf32) <- (1x12x1x1xf32)
        assign_8 = reshape_7

        # pd_op.assign: (1x12x1x1xf32) <- (1x12x1x1xf32)
        assign_9 = reshape_7

        # pd_op.assign: (1x12x1x1xf32) <- (1x12x1x1xf32)
        assign_10 = reshape_7

        # pd_op.assign: (1x12x1x1xf32) <- (1x12x1x1xf32)
        assign_11 = reshape_7

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x12x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_6, reshape_7)

        # pd_op.conv2d: (4x3x-1x-1xf32) <- (4x256x-1x-1xf32, 3x256x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_1, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (4x3x-1x-1xf32) <- (4x3x-1x-1xf32, 1x3x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_7, reshape_6)

        # pd_op.conv2d: (4x12x-1x-1xf32) <- (4x256x-1x-1xf32, 12x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_1, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x12x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_8, reshape_7)

        # pd_op.conv2d: (4x3x-1x-1xf32) <- (4x256x-1x-1xf32, 3x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_2, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (4x3x-1x-1xf32) <- (4x3x-1x-1xf32, 1x3x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_9, reshape_6)

        # pd_op.conv2d: (4x12x-1x-1xf32) <- (4x256x-1x-1xf32, 12x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_2, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x12x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_10, reshape_7)

        # pd_op.conv2d: (4x3x-1x-1xf32) <- (4x256x-1x-1xf32, 3x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_3, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (4x3x-1x-1xf32) <- (4x3x-1x-1xf32, 1x3x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_11, reshape_6)

        # pd_op.conv2d: (4x12x-1x-1xf32) <- (4x256x-1x-1xf32, 12x256x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_3, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x12x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_12, reshape_7)

        # pd_op.conv2d: (4x3x16x-1xf32) <- (4x256x16x-1xf32, 3x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_4, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.add: (4x3x16x-1xf32) <- (4x3x16x-1xf32, 1x3x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_13, reshape_6)

        # pd_op.conv2d: (4x12x16x-1xf32) <- (4x256x16x-1xf32, 12x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_4, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.add: (4x12x16x-1xf32) <- (4x12x16x-1xf32, 1x12x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_14, reshape_7)

        # pd_op.shape64: (4xi64) <- (4x256x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(relu_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.shape64: (4xi64) <- (4x256x-1x-1xf32)
        shape64_1 = paddle._C_ops.shape64(relu_1)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_1

        # pd_op.shape64: (4xi64) <- (4x256x-1x-1xf32)
        shape64_2 = paddle._C_ops.shape64(relu_2)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_2

        # pd_op.shape64: (4xi64) <- (4x256x-1x-1xf32)
        shape64_3 = paddle._C_ops.shape64(relu_3)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_3

        # pd_op.shape64: (4xi64) <- (4x256x16x-1xf32)
        shape64_4 = paddle._C_ops.shape64(relu_4)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_4

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_1, full_0, float("0"), True)
        del slice_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.cast: (xf32) <- (xi64)
        cast_0 = paddle._C_ops.cast(scale_0, paddle.float32)
        del scale_0

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_0 = paddle.arange(full_1, cast_0, full_0, dtype="float32")
        del cast_0

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_0, float("0"), True)
        del slice_0

        # pd_op.cast: (xf32) <- (xi64)
        cast_1 = paddle._C_ops.cast(scale_1, paddle.float32)
        del scale_1

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_1 = paddle.arange(full_1, cast_1, full_0, dtype="float32")
        del cast_1, full_0

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [arange_1, arange_0]
        del arange_0, arange_1

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.reshape: (-1xf32) <- (-1x-1xf32, 1xi64)
        reshape_8 = paddle._C_ops.reshape(split_1, full_int_array_4)
        del split_1

        # pd_op.reshape: (-1xf32) <- (-1x-1xf32, 1xi64)
        reshape_9 = paddle._C_ops.reshape(split_0, full_int_array_4)
        del split_0

        # builtin.combine: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, -1xf32, -1xf32, -1xf32)
        combine_1 = [reshape_8, reshape_9, reshape_8, reshape_9]
        del reshape_8, reshape_9

        # pd_op.stack: (-1x4xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, 1)
        del combine_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [-1, 1, 4]

        # pd_op.reshape: (-1x1x4xf32) <- (-1x4xf32, 3xi64)
        reshape_10 = paddle._C_ops.reshape(stack_0, full_int_array_5)
        del stack_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [1, -1, 4]

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(data_5, full_int_array_6)
        del data_5

        # pd_op.add: (-1x3x4xf32) <- (-1x1x4xf32, 1x3x4xf32)
        add_15 = paddle._C_ops.add(reshape_10, reshape_11)
        del reshape_10, reshape_11

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1x3x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(add_15, full_int_array_7)
        del add_15

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_3, full_2, float("0"), True)
        del slice_3

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(scale_2, paddle.float32)
        del scale_2

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_2 = paddle.arange(full_1, cast_2, full_2, dtype="float32")
        del cast_2

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(slice_2, full_2, float("0"), True)
        del slice_2

        # pd_op.cast: (xf32) <- (xi64)
        cast_3 = paddle._C_ops.cast(scale_3, paddle.float32)
        del scale_3

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_3 = paddle.arange(full_1, cast_3, full_2, dtype="float32")
        del cast_3, full_2

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_2 = [arange_3, arange_2]
        del arange_2, arange_3

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_2)
        del combine_2

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # pd_op.reshape: (-1xf32) <- (-1x-1xf32, 1xi64)
        reshape_12 = paddle._C_ops.reshape(split_3, full_int_array_4)
        del split_3

        # pd_op.reshape: (-1xf32) <- (-1x-1xf32, 1xi64)
        reshape_13 = paddle._C_ops.reshape(split_2, full_int_array_4)
        del split_2

        # builtin.combine: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, -1xf32, -1xf32, -1xf32)
        combine_3 = [reshape_12, reshape_13, reshape_12, reshape_13]
        del reshape_12, reshape_13

        # pd_op.stack: (-1x4xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        stack_1 = paddle._C_ops.stack(combine_3, 1)
        del combine_3

        # pd_op.reshape: (-1x1x4xf32) <- (-1x4xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(stack_1, full_int_array_5)
        del stack_1

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(data_6, full_int_array_6)
        del data_6

        # pd_op.add: (-1x3x4xf32) <- (-1x1x4xf32, 1x3x4xf32)
        add_16 = paddle._C_ops.add(reshape_14, reshape_15)
        del reshape_14, reshape_15

        # pd_op.reshape: (-1x4xf32) <- (-1x3x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(add_16, full_int_array_7)
        del add_16

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_5, full_3, float("0"), True)
        del slice_5

        # pd_op.cast: (xf32) <- (xi64)
        cast_4 = paddle._C_ops.cast(scale_4, paddle.float32)
        del scale_4

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_4 = paddle.arange(full_1, cast_4, full_3, dtype="float32")
        del cast_4

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_5 = paddle._C_ops.scale(slice_4, full_3, float("0"), True)
        del slice_4

        # pd_op.cast: (xf32) <- (xi64)
        cast_5 = paddle._C_ops.cast(scale_5, paddle.float32)
        del scale_5

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_5 = paddle.arange(full_1, cast_5, full_3, dtype="float32")
        del cast_5, full_3

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_4 = [arange_5, arange_4]
        del arange_4, arange_5

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_4)
        del combine_4

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # pd_op.reshape: (-1xf32) <- (-1x-1xf32, 1xi64)
        reshape_16 = paddle._C_ops.reshape(split_5, full_int_array_4)
        del split_5

        # pd_op.reshape: (-1xf32) <- (-1x-1xf32, 1xi64)
        reshape_17 = paddle._C_ops.reshape(split_4, full_int_array_4)
        del split_4

        # builtin.combine: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, -1xf32, -1xf32, -1xf32)
        combine_5 = [reshape_16, reshape_17, reshape_16, reshape_17]
        del reshape_16, reshape_17

        # pd_op.stack: (-1x4xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        stack_2 = paddle._C_ops.stack(combine_5, 1)
        del combine_5

        # pd_op.reshape: (-1x1x4xf32) <- (-1x4xf32, 3xi64)
        reshape_18 = paddle._C_ops.reshape(stack_2, full_int_array_5)
        del stack_2

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(data_7, full_int_array_6)
        del data_7

        # pd_op.add: (-1x3x4xf32) <- (-1x1x4xf32, 1x3x4xf32)
        add_17 = paddle._C_ops.add(reshape_18, reshape_19)
        del reshape_18, reshape_19

        # pd_op.reshape: (-1x4xf32) <- (-1x3x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(add_17, full_int_array_7)
        del add_17

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_6 = paddle._C_ops.scale(slice_7, full_4, float("0"), True)
        del slice_7

        # pd_op.cast: (xf32) <- (xi64)
        cast_6 = paddle._C_ops.cast(scale_6, paddle.float32)
        del scale_6

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_6 = paddle.arange(full_1, cast_6, full_4, dtype="float32")
        del cast_6

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_6, full_4, float("0"), True)
        del slice_6

        # pd_op.cast: (xf32) <- (xi64)
        cast_7 = paddle._C_ops.cast(scale_7, paddle.float32)
        del scale_7

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_7 = paddle.arange(full_1, cast_7, full_4, dtype="float32")
        del cast_7, full_4

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_6 = [arange_7, arange_6]
        del arange_6, arange_7

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_6)
        del combine_6

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3
        del meshgrid_3

        # pd_op.reshape: (-1xf32) <- (-1x-1xf32, 1xi64)
        reshape_20 = paddle._C_ops.reshape(split_7, full_int_array_4)
        del split_7

        # pd_op.reshape: (-1xf32) <- (-1x-1xf32, 1xi64)
        reshape_21 = paddle._C_ops.reshape(split_6, full_int_array_4)
        del split_6

        # builtin.combine: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, -1xf32, -1xf32, -1xf32)
        combine_7 = [reshape_20, reshape_21, reshape_20, reshape_21]
        del reshape_20, reshape_21

        # pd_op.stack: (-1x4xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        stack_3 = paddle._C_ops.stack(combine_7, 1)
        del combine_7

        # pd_op.reshape: (-1x1x4xf32) <- (-1x4xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(stack_3, full_int_array_5)
        del stack_3

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(data_8, full_int_array_6)
        del data_8

        # pd_op.add: (-1x3x4xf32) <- (-1x1x4xf32, 1x3x4xf32)
        add_18 = paddle._C_ops.add(reshape_22, reshape_23)
        del reshape_22, reshape_23

        # pd_op.reshape: (-1x4xf32) <- (-1x3x4xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(add_18, full_int_array_7)
        del add_18

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("64"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_8 = paddle._C_ops.scale(slice_8, full_5, float("0"), True)
        del slice_8

        # pd_op.cast: (xf32) <- (xi64)
        cast_8 = paddle._C_ops.cast(scale_8, paddle.float32)
        del scale_8

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_8 = paddle.arange(full_1, cast_8, full_5, dtype="float32")
        del cast_8, full_1, full_5

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1024"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("64"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (16xf32) <- (1xf64, 1xf64, 1xf64)
        arange_9 = paddle.arange(full_6, full_7, full_8, dtype="float32")
        del full_6, full_7, full_8

        # builtin.combine: ([16xf32, -1xf32]) <- (16xf32, -1xf32)
        combine_8 = [arange_9, arange_8]
        del arange_8, arange_9

        # pd_op.meshgrid: ([16x-1xf32, 16x-1xf32]) <- ([16xf32, -1xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_8)
        del combine_8

        # builtin.split: (16x-1xf32, 16x-1xf32) <- ([16x-1xf32, 16x-1xf32])
        (
            split_8,
            split_9,
        ) = meshgrid_4
        del meshgrid_4

        # pd_op.reshape: (-1xf32) <- (16x-1xf32, 1xi64)
        reshape_24 = paddle._C_ops.reshape(split_9, full_int_array_4)
        del split_9

        # pd_op.reshape: (-1xf32) <- (16x-1xf32, 1xi64)
        reshape_25 = paddle._C_ops.reshape(split_8, full_int_array_4)
        del full_int_array_4, split_8

        # builtin.combine: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, -1xf32, -1xf32, -1xf32)
        combine_9 = [reshape_24, reshape_25, reshape_24, reshape_25]
        del reshape_24, reshape_25

        # pd_op.stack: (-1x4xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        stack_4 = paddle._C_ops.stack(combine_9, 1)
        del combine_9

        # pd_op.reshape: (-1x1x4xf32) <- (-1x4xf32, 3xi64)
        reshape_26 = paddle._C_ops.reshape(stack_4, full_int_array_5)
        del full_int_array_5, stack_4

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(data_9, full_int_array_6)
        del data_9, full_int_array_6

        # pd_op.add: (-1x3x4xf32) <- (-1x1x4xf32, 1x3x4xf32)
        add_19 = paddle._C_ops.add(reshape_26, reshape_27)
        del reshape_26, reshape_27

        # pd_op.reshape: (-1x4xf32) <- (-1x3x4xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(add_19, full_int_array_7)
        del (
            add_19,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_12,
            conv2d_13,
            conv2d_14,
            conv2d_2,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            full_int_array_7,
            relu_0,
            relu_1,
            relu_2,
            relu_3,
            relu_4,
            reshape_5,
            reshape_6,
            reshape_7,
        )

        return (
            add_0,
            add_1,
            add_2,
            add_3,
            add_4,
            add_5,
            add_6,
            add_7,
            add_8,
            add_9,
            reshape_0,
            reshape_1,
            reshape_2,
            reshape_3,
            reshape_4,
        )
