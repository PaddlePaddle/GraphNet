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
        # pd_op.conv2d: (1x256x176x264xf32) <- (1x256x176x264xf32, 256x256x3x3xf32)
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

        # pd_op.add: (1x256x176x264xf32) <- (1x256x176x264xf32, 1x256x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_0, reshape_5)

        # pd_op.relu: (1x256x176x264xf32) <- (1x256x176x264xf32)
        relu_0 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.conv2d: (1x256x88x132xf32) <- (1x256x88x132xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_1, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_1

        # pd_op.add: (1x256x88x132xf32) <- (1x256x88x132xf32, 1x256x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_1, reshape_5)

        # pd_op.relu: (1x256x88x132xf32) <- (1x256x88x132xf32)
        relu_1 = paddle._C_ops.relu(add_11)
        del add_11

        # pd_op.conv2d: (1x256x44x66xf32) <- (1x256x44x66xf32, 256x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_2, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_2

        # pd_op.add: (1x256x44x66xf32) <- (1x256x44x66xf32, 1x256x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_2, reshape_5)

        # pd_op.relu: (1x256x44x66xf32) <- (1x256x44x66xf32)
        relu_2 = paddle._C_ops.relu(add_12)
        del add_12

        # pd_op.conv2d: (1x256x22x33xf32) <- (1x256x22x33xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            data_3, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_3

        # pd_op.add: (1x256x22x33xf32) <- (1x256x22x33xf32, 1x256x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_3, reshape_5)

        # pd_op.relu: (1x256x22x33xf32) <- (1x256x22x33xf32)
        relu_3 = paddle._C_ops.relu(add_13)
        del add_13

        # pd_op.conv2d: (1x256x11x17xf32) <- (1x256x11x17xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            data_4, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_4, parameter_5

        # pd_op.add: (1x256x11x17xf32) <- (1x256x11x17xf32, 1x256x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_4, reshape_5)

        # pd_op.relu: (1x256x11x17xf32) <- (1x256x11x17xf32)
        relu_4 = paddle._C_ops.relu(add_14)
        del add_14

        # pd_op.conv2d: (1x3x176x264xf32) <- (1x256x176x264xf32, 3x256x1x1xf32)
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

        # pd_op.add: (1x3x176x264xf32) <- (1x3x176x264xf32, 1x3x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_5, reshape_6)

        # pd_op.conv2d: (1x12x176x264xf32) <- (1x256x176x264xf32, 12x256x1x1xf32)
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

        # pd_op.add: (1x12x176x264xf32) <- (1x12x176x264xf32, 1x12x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_6, reshape_7)

        # pd_op.conv2d: (1x3x88x132xf32) <- (1x256x88x132xf32, 3x256x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_1, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x3x88x132xf32) <- (1x3x88x132xf32, 1x3x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_7, reshape_6)

        # pd_op.conv2d: (1x12x88x132xf32) <- (1x256x88x132xf32, 12x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_1, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x12x88x132xf32) <- (1x12x88x132xf32, 1x12x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_8, reshape_7)

        # pd_op.conv2d: (1x3x44x66xf32) <- (1x256x44x66xf32, 3x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_2, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x3x44x66xf32) <- (1x3x44x66xf32, 1x3x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_9, reshape_6)

        # pd_op.conv2d: (1x12x44x66xf32) <- (1x256x44x66xf32, 12x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_2, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x12x44x66xf32) <- (1x12x44x66xf32, 1x12x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_10, reshape_7)

        # pd_op.conv2d: (1x3x22x33xf32) <- (1x256x22x33xf32, 3x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_3, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x3x22x33xf32) <- (1x3x22x33xf32, 1x3x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_11, reshape_6)

        # pd_op.conv2d: (1x12x22x33xf32) <- (1x256x22x33xf32, 12x256x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_3, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x12x22x33xf32) <- (1x12x22x33xf32, 1x12x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_12, reshape_7)

        # pd_op.conv2d: (1x3x11x17xf32) <- (1x256x11x17xf32, 3x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_4, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.add: (1x3x11x17xf32) <- (1x3x11x17xf32, 1x3x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_13, reshape_6)

        # pd_op.conv2d: (1x12x11x17xf32) <- (1x256x11x17xf32, 12x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_4, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.add: (1x12x11x17xf32) <- (1x12x11x17xf32, 1x12x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_14, reshape_7)

        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1056"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("4"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (264xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("704"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (176xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_3, full_2, dtype="float32")
        del full_2

        # builtin.combine: ([176xf32, 264xf32]) <- (176xf32, 264xf32)
        combine_0 = [arange_1, arange_0]
        del arange_0, arange_1

        # pd_op.meshgrid: ([176x264xf32, 176x264xf32]) <- ([176xf32, 264xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (176x264xf32, 176x264xf32) <- ([176x264xf32, 176x264xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.reshape: (46464xf32) <- (176x264xf32, 1xi64)
        reshape_8 = paddle._C_ops.reshape(split_1, full_int_array_1)
        del split_1

        # pd_op.reshape: (46464xf32) <- (176x264xf32, 1xi64)
        reshape_9 = paddle._C_ops.reshape(split_0, full_int_array_1)
        del split_0

        # builtin.combine: ([46464xf32, 46464xf32, 46464xf32, 46464xf32]) <- (46464xf32, 46464xf32, 46464xf32, 46464xf32)
        combine_1 = [reshape_8, reshape_9, reshape_8, reshape_9]
        del reshape_8, reshape_9

        # pd_op.stack: (46464x4xf32) <- ([46464xf32, 46464xf32, 46464xf32, 46464xf32])
        stack_0 = paddle._C_ops.stack(combine_1, 1)
        del combine_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [-1, 1, 4]

        # pd_op.reshape: (46464x1x4xf32) <- (46464x4xf32, 3xi64)
        reshape_10 = paddle._C_ops.reshape(stack_0, full_int_array_2)
        del stack_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [1, -1, 4]

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(data_5, full_int_array_3)
        del data_5

        # pd_op.add: (46464x3x4xf32) <- (46464x1x4xf32, 1x3x4xf32)
        add_15 = paddle._C_ops.add(reshape_10, reshape_11)
        del reshape_10, reshape_11

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [-1, 4]

        # pd_op.reshape: (139392x4xf32) <- (46464x3x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(add_15, full_int_array_4)
        del add_15

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("8"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (132xf32) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_1, full_4, dtype="float32")

        # pd_op.arange: (88xf32) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_0, full_3, full_4, dtype="float32")
        del full_4

        # builtin.combine: ([88xf32, 132xf32]) <- (88xf32, 132xf32)
        combine_2 = [arange_3, arange_2]
        del arange_2, arange_3

        # pd_op.meshgrid: ([88x132xf32, 88x132xf32]) <- ([88xf32, 132xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_2)
        del combine_2

        # builtin.split: (88x132xf32, 88x132xf32) <- ([88x132xf32, 88x132xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # pd_op.reshape: (11616xf32) <- (88x132xf32, 1xi64)
        reshape_12 = paddle._C_ops.reshape(split_3, full_int_array_1)
        del split_3

        # pd_op.reshape: (11616xf32) <- (88x132xf32, 1xi64)
        reshape_13 = paddle._C_ops.reshape(split_2, full_int_array_1)
        del split_2

        # builtin.combine: ([11616xf32, 11616xf32, 11616xf32, 11616xf32]) <- (11616xf32, 11616xf32, 11616xf32, 11616xf32)
        combine_3 = [reshape_12, reshape_13, reshape_12, reshape_13]
        del reshape_12, reshape_13

        # pd_op.stack: (11616x4xf32) <- ([11616xf32, 11616xf32, 11616xf32, 11616xf32])
        stack_1 = paddle._C_ops.stack(combine_3, 1)
        del combine_3

        # pd_op.reshape: (11616x1x4xf32) <- (11616x4xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(stack_1, full_int_array_2)
        del stack_1

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(data_6, full_int_array_3)
        del data_6

        # pd_op.add: (11616x3x4xf32) <- (11616x1x4xf32, 1x3x4xf32)
        add_16 = paddle._C_ops.add(reshape_14, reshape_15)
        del reshape_14, reshape_15

        # pd_op.reshape: (34848x4xf32) <- (11616x3x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(add_16, full_int_array_4)
        del add_16

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("16"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (66xf32) <- (1xf64, 1xf64, 1xf64)
        arange_4 = paddle.arange(full_0, full_1, full_5, dtype="float32")

        # pd_op.arange: (44xf32) <- (1xf64, 1xf64, 1xf64)
        arange_5 = paddle.arange(full_0, full_3, full_5, dtype="float32")
        del full_5

        # builtin.combine: ([44xf32, 66xf32]) <- (44xf32, 66xf32)
        combine_4 = [arange_5, arange_4]
        del arange_4, arange_5

        # pd_op.meshgrid: ([44x66xf32, 44x66xf32]) <- ([44xf32, 66xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_4)
        del combine_4

        # builtin.split: (44x66xf32, 44x66xf32) <- ([44x66xf32, 44x66xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # pd_op.reshape: (2904xf32) <- (44x66xf32, 1xi64)
        reshape_16 = paddle._C_ops.reshape(split_5, full_int_array_1)
        del split_5

        # pd_op.reshape: (2904xf32) <- (44x66xf32, 1xi64)
        reshape_17 = paddle._C_ops.reshape(split_4, full_int_array_1)
        del split_4

        # builtin.combine: ([2904xf32, 2904xf32, 2904xf32, 2904xf32]) <- (2904xf32, 2904xf32, 2904xf32, 2904xf32)
        combine_5 = [reshape_16, reshape_17, reshape_16, reshape_17]
        del reshape_16, reshape_17

        # pd_op.stack: (2904x4xf32) <- ([2904xf32, 2904xf32, 2904xf32, 2904xf32])
        stack_2 = paddle._C_ops.stack(combine_5, 1)
        del combine_5

        # pd_op.reshape: (2904x1x4xf32) <- (2904x4xf32, 3xi64)
        reshape_18 = paddle._C_ops.reshape(stack_2, full_int_array_2)
        del stack_2

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(data_7, full_int_array_3)
        del data_7

        # pd_op.add: (2904x3x4xf32) <- (2904x1x4xf32, 1x3x4xf32)
        add_17 = paddle._C_ops.add(reshape_18, reshape_19)
        del reshape_18, reshape_19

        # pd_op.reshape: (8712x4xf32) <- (2904x3x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(add_17, full_int_array_4)
        del add_17

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("32"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (33xf32) <- (1xf64, 1xf64, 1xf64)
        arange_6 = paddle.arange(full_0, full_1, full_6, dtype="float32")
        del full_1

        # pd_op.arange: (22xf32) <- (1xf64, 1xf64, 1xf64)
        arange_7 = paddle.arange(full_0, full_3, full_6, dtype="float32")
        del full_6

        # builtin.combine: ([22xf32, 33xf32]) <- (22xf32, 33xf32)
        combine_6 = [arange_7, arange_6]
        del arange_6, arange_7

        # pd_op.meshgrid: ([22x33xf32, 22x33xf32]) <- ([22xf32, 33xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_6)
        del combine_6

        # builtin.split: (22x33xf32, 22x33xf32) <- ([22x33xf32, 22x33xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3
        del meshgrid_3

        # pd_op.reshape: (726xf32) <- (22x33xf32, 1xi64)
        reshape_20 = paddle._C_ops.reshape(split_7, full_int_array_1)
        del split_7

        # pd_op.reshape: (726xf32) <- (22x33xf32, 1xi64)
        reshape_21 = paddle._C_ops.reshape(split_6, full_int_array_1)
        del split_6

        # builtin.combine: ([726xf32, 726xf32, 726xf32, 726xf32]) <- (726xf32, 726xf32, 726xf32, 726xf32)
        combine_7 = [reshape_20, reshape_21, reshape_20, reshape_21]
        del reshape_20, reshape_21

        # pd_op.stack: (726x4xf32) <- ([726xf32, 726xf32, 726xf32, 726xf32])
        stack_3 = paddle._C_ops.stack(combine_7, 1)
        del combine_7

        # pd_op.reshape: (726x1x4xf32) <- (726x4xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(stack_3, full_int_array_2)
        del stack_3

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(data_8, full_int_array_3)
        del data_8

        # pd_op.add: (726x3x4xf32) <- (726x1x4xf32, 1x3x4xf32)
        add_18 = paddle._C_ops.add(reshape_22, reshape_23)
        del reshape_22, reshape_23

        # pd_op.reshape: (2178x4xf32) <- (726x3x4xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(add_18, full_int_array_4)
        del add_18

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1088"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("64"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (17xf32) <- (1xf64, 1xf64, 1xf64)
        arange_8 = paddle.arange(full_0, full_7, full_8, dtype="float32")
        del full_7

        # pd_op.arange: (11xf32) <- (1xf64, 1xf64, 1xf64)
        arange_9 = paddle.arange(full_0, full_3, full_8, dtype="float32")
        del full_0, full_3, full_8

        # builtin.combine: ([11xf32, 17xf32]) <- (11xf32, 17xf32)
        combine_8 = [arange_9, arange_8]
        del arange_8, arange_9

        # pd_op.meshgrid: ([11x17xf32, 11x17xf32]) <- ([11xf32, 17xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_8)
        del combine_8

        # builtin.split: (11x17xf32, 11x17xf32) <- ([11x17xf32, 11x17xf32])
        (
            split_8,
            split_9,
        ) = meshgrid_4
        del meshgrid_4

        # pd_op.reshape: (187xf32) <- (11x17xf32, 1xi64)
        reshape_24 = paddle._C_ops.reshape(split_9, full_int_array_1)
        del split_9

        # pd_op.reshape: (187xf32) <- (11x17xf32, 1xi64)
        reshape_25 = paddle._C_ops.reshape(split_8, full_int_array_1)
        del full_int_array_1, split_8

        # builtin.combine: ([187xf32, 187xf32, 187xf32, 187xf32]) <- (187xf32, 187xf32, 187xf32, 187xf32)
        combine_9 = [reshape_24, reshape_25, reshape_24, reshape_25]
        del reshape_24, reshape_25

        # pd_op.stack: (187x4xf32) <- ([187xf32, 187xf32, 187xf32, 187xf32])
        stack_4 = paddle._C_ops.stack(combine_9, 1)
        del combine_9

        # pd_op.reshape: (187x1x4xf32) <- (187x4xf32, 3xi64)
        reshape_26 = paddle._C_ops.reshape(stack_4, full_int_array_2)
        del full_int_array_2, stack_4

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(data_9, full_int_array_3)
        del data_9, full_int_array_3

        # pd_op.add: (187x3x4xf32) <- (187x1x4xf32, 1x3x4xf32)
        add_19 = paddle._C_ops.add(reshape_26, reshape_27)
        del reshape_26, reshape_27

        # pd_op.reshape: (561x4xf32) <- (187x3x4xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(add_19, full_int_array_4)
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
            full_int_array_4,
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
