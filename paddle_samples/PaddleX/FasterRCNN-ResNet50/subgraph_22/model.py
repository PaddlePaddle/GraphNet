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
    ):
        # pd_op.conv2d: (2x1024x-1x-1xf32) <- (2x1024x-1x-1xf32, 1024x1024x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_5

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_4, full_int_array_0)
        del parameter_4

        # pd_op.add: (2x1024x-1x-1xf32) <- (2x1024x-1x-1xf32, 1x1024x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.relu: (2x1024x-1x-1xf32) <- (2x1024x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(add_0)
        del add_0

        # pd_op.conv2d: (2x15x-1x-1xf32) <- (2x1024x-1x-1xf32, 15x1024x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.reshape: (1x15x1x1xf32) <- (15xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_2, full_int_array_0)
        del parameter_2

        # pd_op.add: (2x15x-1x-1xf32) <- (2x15x-1x-1xf32, 1x15x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_1)

        # pd_op.conv2d: (2x60x-1x-1xf32) <- (2x1024x-1x-1xf32, 60x1024x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.reshape: (1x60x1x1xf32) <- (60xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_0, full_int_array_0)
        del full_int_array_0, parameter_0

        # pd_op.add: (2x60x-1x-1xf32) <- (2x60x-1x-1xf32, 1x60x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_2, reshape_2)

        # pd_op.shape64: (4xi64) <- (2x1024x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(relu_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
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
        del cast_1, full_0, full_1

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
        reshape_3 = paddle._C_ops.reshape(split_1, full_int_array_4)
        del split_1

        # pd_op.reshape: (-1xf32) <- (-1x-1xf32, 1xi64)
        reshape_4 = paddle._C_ops.reshape(split_0, full_int_array_4)
        del full_int_array_4, split_0

        # builtin.combine: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, -1xf32, -1xf32, -1xf32)
        combine_1 = [reshape_3, reshape_4, reshape_3, reshape_4]
        del reshape_3, reshape_4

        # pd_op.stack: (-1x4xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, 1)
        del combine_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [-1, 1, 4]

        # pd_op.reshape: (-1x1x4xf32) <- (-1x4xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(stack_0, full_int_array_5)
        del full_int_array_5, stack_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [1, -1, 4]

        # pd_op.reshape: (1x15x4xf32) <- (15x4xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(data_1, full_int_array_6)
        del data_1, full_int_array_6

        # pd_op.add: (-1x15x4xf32) <- (-1x1x4xf32, 1x15x4xf32)
        add_3 = paddle._C_ops.add(reshape_5, reshape_6)
        del reshape_5, reshape_6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1x15x4xf32, 2xi64)
        reshape_7 = paddle._C_ops.reshape(add_3, full_int_array_7)
        del add_3, full_int_array_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.slice: (1x15x-1x-1xf32) <- (2x15x-1x-1xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            add_1, [0], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x60x-1x-1xf32) <- (2x60x-1x-1xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            add_2, [0], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x2xf32) <- (2x2xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_2, [0], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1x4xf32) <- (-1x4xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            reshape_7,
            full_2,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del full_2

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x15x-1x-1xf32, 1x60x-1x-1xf32, 1x2xf32, -1x4xf32, -1x4xf32)
        generate_proposals_0, generate_proposals_1, generate_proposals_2 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                slice_2,
                slice_3,
                slice_4,
                reshape_7,
                full_like_0,
                12000,
                2000,
                float("0.7"),
                float("0"),
                float("1"),
                False,
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del slice_2, slice_3, slice_4

        # pd_op.flatten: (-1xf32) <- (-1x1xf32)
        flatten_0 = paddle._C_ops.flatten(generate_proposals_1, 0, 1)
        del generate_proposals_1

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_1 = paddle._C_ops.shape64(generate_proposals_0)

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_8, full_int_array_9, [1], []
        )
        del shape64_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [2147483647]

        # pd_op.slice: (1x15x-1x-1xf32) <- (2x15x-1x-1xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            add_1, [0], full_int_array_9, full_int_array_10, [1], []
        )

        # pd_op.slice: (1x60x-1x-1xf32) <- (2x60x-1x-1xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            add_2, [0], full_int_array_9, full_int_array_10, [1], []
        )

        # pd_op.slice: (1x2xf32) <- (2x2xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_2, [0], full_int_array_9, full_int_array_10, [1], []
        )
        del data_2, full_int_array_10

        # pd_op.generate_proposals: (-1x4xf32, -1x1xf32, 1xf32) <- (1x15x-1x-1xf32, 1x60x-1x-1xf32, 1x2xf32, -1x4xf32, -1x4xf32)
        generate_proposals_3, generate_proposals_4, generate_proposals_5 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.generate_proposals(
                slice_6,
                slice_7,
                slice_8,
                reshape_7,
                full_like_0,
                12000,
                2000,
                float("0.7"),
                float("0"),
                float("1"),
                False,
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del full_like_0, slice_6, slice_7, slice_8

        # pd_op.flatten: (-1xf32) <- (-1x1xf32)
        flatten_1 = paddle._C_ops.flatten(generate_proposals_4, 0, 1)
        del generate_proposals_4

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_2 = paddle._C_ops.shape64(generate_proposals_3)

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_8, full_int_array_9, [1], []
        )
        del full_int_array_8, full_int_array_9, shape64_2

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1xi64, 1xi64]) <- (1xi64, 1xi64)
        combine_2 = [slice_5, slice_9]
        del slice_5, slice_9

        # pd_op.concat: (2xi64) <- ([1xi64, 1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_3)
        del (
            add_1,
            add_2,
            combine_2,
            conv2d_0,
            conv2d_1,
            conv2d_2,
            full_3,
            generate_proposals_0,
            generate_proposals_3,
            relu_0,
            reshape_0,
            reshape_1,
            reshape_2,
            reshape_7,
        )

        return concat_0
