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
        # pd_op.conv2d: (2x256x248x184xf32) <- (2x256x248x184xf32, 256x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_4, full_int_array_0)

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_3 = reshape_0

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_2 = reshape_0

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_1 = reshape_0

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_0 = reshape_0

        # pd_op.add: (2x256x248x184xf32) <- (2x256x248x184xf32, 1x256x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.relu: (2x256x248x184xf32) <- (2x256x248x184xf32)
        relu_0 = paddle._C_ops.relu(add_10)

        # pd_op.conv2d: (2x256x124x92xf32) <- (2x256x124x92xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_1, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x256x124x92xf32) <- (2x256x124x92xf32, 1x256x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_1, reshape_0)

        # pd_op.relu: (2x256x124x92xf32) <- (2x256x124x92xf32)
        relu_1 = paddle._C_ops.relu(add_11)

        # pd_op.conv2d: (2x256x62x46xf32) <- (2x256x62x46xf32, 256x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_2, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x256x62x46xf32) <- (2x256x62x46xf32, 1x256x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_2, reshape_0)

        # pd_op.relu: (2x256x62x46xf32) <- (2x256x62x46xf32)
        relu_2 = paddle._C_ops.relu(add_12)

        # pd_op.conv2d: (2x256x31x23xf32) <- (2x256x31x23xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            data_3, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x256x31x23xf32) <- (2x256x31x23xf32, 1x256x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_3, reshape_0)

        # pd_op.relu: (2x256x31x23xf32) <- (2x256x31x23xf32)
        relu_3 = paddle._C_ops.relu(add_13)

        # pd_op.conv2d: (2x256x16x12xf32) <- (2x256x16x12xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            data_4, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x256x16x12xf32) <- (2x256x16x12xf32, 1x256x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_4, reshape_0)

        # pd_op.relu: (2x256x16x12xf32) <- (2x256x16x12xf32)
        relu_4 = paddle._C_ops.relu(add_14)

        # pd_op.conv2d: (2x3x248x184xf32) <- (2x256x248x184xf32, 3x256x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_0, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x3x1x1xf32) <- (3xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_2, full_int_array_0)

        # pd_op.assign: (1x3x1x1xf32) <- (1x3x1x1xf32)
        assign_10 = reshape_1

        # pd_op.assign: (1x3x1x1xf32) <- (1x3x1x1xf32)
        assign_8 = reshape_1

        # pd_op.assign: (1x3x1x1xf32) <- (1x3x1x1xf32)
        assign_6 = reshape_1

        # pd_op.assign: (1x3x1x1xf32) <- (1x3x1x1xf32)
        assign_4 = reshape_1

        # pd_op.add: (2x3x248x184xf32) <- (2x3x248x184xf32, 1x3x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_5, reshape_1)

        # pd_op.conv2d: (2x12x248x184xf32) <- (2x256x248x184xf32, 12x256x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x12x1x1xf32) <- (12xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_0, full_int_array_0)

        # pd_op.assign: (1x12x1x1xf32) <- (1x12x1x1xf32)
        assign_11 = reshape_2

        # pd_op.assign: (1x12x1x1xf32) <- (1x12x1x1xf32)
        assign_9 = reshape_2

        # pd_op.assign: (1x12x1x1xf32) <- (1x12x1x1xf32)
        assign_7 = reshape_2

        # pd_op.assign: (1x12x1x1xf32) <- (1x12x1x1xf32)
        assign_5 = reshape_2

        # pd_op.add: (2x12x248x184xf32) <- (2x12x248x184xf32, 1x12x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_6, reshape_2)

        # pd_op.conv2d: (2x3x124x92xf32) <- (2x256x124x92xf32, 3x256x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_1, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x3x124x92xf32) <- (2x3x124x92xf32, 1x3x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_7, reshape_1)

        # pd_op.conv2d: (2x12x124x92xf32) <- (2x256x124x92xf32, 12x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_1, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x12x124x92xf32) <- (2x12x124x92xf32, 1x12x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_8, reshape_2)

        # pd_op.conv2d: (2x3x62x46xf32) <- (2x256x62x46xf32, 3x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_2, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x3x62x46xf32) <- (2x3x62x46xf32, 1x3x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_9, reshape_1)

        # pd_op.conv2d: (2x12x62x46xf32) <- (2x256x62x46xf32, 12x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_2, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x12x62x46xf32) <- (2x12x62x46xf32, 1x12x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_10, reshape_2)

        # pd_op.conv2d: (2x3x31x23xf32) <- (2x256x31x23xf32, 3x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_3, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x3x31x23xf32) <- (2x3x31x23xf32, 1x3x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_11, reshape_1)

        # pd_op.conv2d: (2x12x31x23xf32) <- (2x256x31x23xf32, 12x256x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_3, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x12x31x23xf32) <- (2x12x31x23xf32, 1x12x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_12, reshape_2)

        # pd_op.conv2d: (2x3x16x12xf32) <- (2x256x16x12xf32, 3x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_4, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x3x16x12xf32) <- (2x3x16x12xf32, 1x3x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_13, reshape_1)

        # pd_op.conv2d: (2x12x16x12xf32) <- (2x256x16x12xf32, 12x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_4, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (2x12x16x12xf32) <- (2x12x16x12xf32, 1x12x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_14, reshape_2)

        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("736"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("4"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (184xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("992"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (248xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_3, full_2, dtype="float32")

        # builtin.combine: ([248xf32, 184xf32]) <- (248xf32, 184xf32)
        combine_0 = [arange_1, arange_0]

        # pd_op.meshgrid: ([248x184xf32, 248x184xf32]) <- ([248xf32, 184xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (248x184xf32, 248x184xf32) <- ([248x184xf32, 248x184xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.reshape: (45632xf32) <- (248x184xf32, 1xi64)
        reshape_8 = paddle._C_ops.reshape(split_1, full_int_array_1)

        # pd_op.reshape: (45632xf32) <- (248x184xf32, 1xi64)
        reshape_9 = paddle._C_ops.reshape(split_0, full_int_array_1)

        # builtin.combine: ([45632xf32, 45632xf32, 45632xf32, 45632xf32]) <- (45632xf32, 45632xf32, 45632xf32, 45632xf32)
        combine_1 = [reshape_8, reshape_9, reshape_8, reshape_9]

        # pd_op.stack: (45632x4xf32) <- ([45632xf32, 45632xf32, 45632xf32, 45632xf32])
        stack_0 = paddle._C_ops.stack(combine_1, 1)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [-1, 1, 4]

        # pd_op.reshape: (45632x1x4xf32) <- (45632x4xf32, 3xi64)
        reshape_10 = paddle._C_ops.reshape(stack_0, full_int_array_2)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [1, -1, 4]

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(data_5, full_int_array_3)

        # pd_op.add: (45632x3x4xf32) <- (45632x1x4xf32, 1x3x4xf32)
        add_15 = paddle._C_ops.add(reshape_10, reshape_11)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [-1, 4]

        # pd_op.reshape: (136896x4xf32) <- (45632x3x4xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(add_15, full_int_array_4)

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("8"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (92xf32) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_1, full_4, dtype="float32")

        # pd_op.arange: (124xf32) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_0, full_3, full_4, dtype="float32")

        # builtin.combine: ([124xf32, 92xf32]) <- (124xf32, 92xf32)
        combine_2 = [arange_3, arange_2]

        # pd_op.meshgrid: ([124x92xf32, 124x92xf32]) <- ([124xf32, 92xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_2)

        # builtin.split: (124x92xf32, 124x92xf32) <- ([124x92xf32, 124x92xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1

        # pd_op.reshape: (11408xf32) <- (124x92xf32, 1xi64)
        reshape_12 = paddle._C_ops.reshape(split_3, full_int_array_1)

        # pd_op.reshape: (11408xf32) <- (124x92xf32, 1xi64)
        reshape_13 = paddle._C_ops.reshape(split_2, full_int_array_1)

        # builtin.combine: ([11408xf32, 11408xf32, 11408xf32, 11408xf32]) <- (11408xf32, 11408xf32, 11408xf32, 11408xf32)
        combine_3 = [reshape_12, reshape_13, reshape_12, reshape_13]

        # pd_op.stack: (11408x4xf32) <- ([11408xf32, 11408xf32, 11408xf32, 11408xf32])
        stack_1 = paddle._C_ops.stack(combine_3, 1)

        # pd_op.reshape: (11408x1x4xf32) <- (11408x4xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(stack_1, full_int_array_2)

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(data_6, full_int_array_3)

        # pd_op.add: (11408x3x4xf32) <- (11408x1x4xf32, 1x3x4xf32)
        add_16 = paddle._C_ops.add(reshape_14, reshape_15)

        # pd_op.reshape: (34224x4xf32) <- (11408x3x4xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(add_16, full_int_array_4)

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("16"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (46xf32) <- (1xf64, 1xf64, 1xf64)
        arange_4 = paddle.arange(full_0, full_1, full_5, dtype="float32")

        # pd_op.arange: (62xf32) <- (1xf64, 1xf64, 1xf64)
        arange_5 = paddle.arange(full_0, full_3, full_5, dtype="float32")

        # builtin.combine: ([62xf32, 46xf32]) <- (62xf32, 46xf32)
        combine_4 = [arange_5, arange_4]

        # pd_op.meshgrid: ([62x46xf32, 62x46xf32]) <- ([62xf32, 46xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_4)

        # builtin.split: (62x46xf32, 62x46xf32) <- ([62x46xf32, 62x46xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2

        # pd_op.reshape: (2852xf32) <- (62x46xf32, 1xi64)
        reshape_16 = paddle._C_ops.reshape(split_5, full_int_array_1)

        # pd_op.reshape: (2852xf32) <- (62x46xf32, 1xi64)
        reshape_17 = paddle._C_ops.reshape(split_4, full_int_array_1)

        # builtin.combine: ([2852xf32, 2852xf32, 2852xf32, 2852xf32]) <- (2852xf32, 2852xf32, 2852xf32, 2852xf32)
        combine_5 = [reshape_16, reshape_17, reshape_16, reshape_17]

        # pd_op.stack: (2852x4xf32) <- ([2852xf32, 2852xf32, 2852xf32, 2852xf32])
        stack_2 = paddle._C_ops.stack(combine_5, 1)

        # pd_op.reshape: (2852x1x4xf32) <- (2852x4xf32, 3xi64)
        reshape_18 = paddle._C_ops.reshape(stack_2, full_int_array_2)

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(data_7, full_int_array_3)

        # pd_op.add: (2852x3x4xf32) <- (2852x1x4xf32, 1x3x4xf32)
        add_17 = paddle._C_ops.add(reshape_18, reshape_19)

        # pd_op.reshape: (8556x4xf32) <- (2852x3x4xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(add_17, full_int_array_4)

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("32"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (23xf32) <- (1xf64, 1xf64, 1xf64)
        arange_6 = paddle.arange(full_0, full_1, full_6, dtype="float32")

        # pd_op.arange: (31xf32) <- (1xf64, 1xf64, 1xf64)
        arange_7 = paddle.arange(full_0, full_3, full_6, dtype="float32")

        # builtin.combine: ([31xf32, 23xf32]) <- (31xf32, 23xf32)
        combine_6 = [arange_7, arange_6]

        # pd_op.meshgrid: ([31x23xf32, 31x23xf32]) <- ([31xf32, 23xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (31x23xf32, 31x23xf32) <- ([31x23xf32, 31x23xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3

        # pd_op.reshape: (713xf32) <- (31x23xf32, 1xi64)
        reshape_20 = paddle._C_ops.reshape(split_7, full_int_array_1)

        # pd_op.reshape: (713xf32) <- (31x23xf32, 1xi64)
        reshape_21 = paddle._C_ops.reshape(split_6, full_int_array_1)

        # builtin.combine: ([713xf32, 713xf32, 713xf32, 713xf32]) <- (713xf32, 713xf32, 713xf32, 713xf32)
        combine_7 = [reshape_20, reshape_21, reshape_20, reshape_21]

        # pd_op.stack: (713x4xf32) <- ([713xf32, 713xf32, 713xf32, 713xf32])
        stack_3 = paddle._C_ops.stack(combine_7, 1)

        # pd_op.reshape: (713x1x4xf32) <- (713x4xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(stack_3, full_int_array_2)

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(data_8, full_int_array_3)

        # pd_op.add: (713x3x4xf32) <- (713x1x4xf32, 1x3x4xf32)
        add_18 = paddle._C_ops.add(reshape_22, reshape_23)

        # pd_op.reshape: (2139x4xf32) <- (713x3x4xf32, 2xi64)
        reshape_6 = paddle._C_ops.reshape(add_18, full_int_array_4)

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("768"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("64"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (12xf32) <- (1xf64, 1xf64, 1xf64)
        arange_8 = paddle.arange(full_0, full_7, full_8, dtype="float32")

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("1024"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (16xf32) <- (1xf64, 1xf64, 1xf64)
        arange_9 = paddle.arange(full_0, full_9, full_8, dtype="float32")

        # builtin.combine: ([16xf32, 12xf32]) <- (16xf32, 12xf32)
        combine_8 = [arange_9, arange_8]

        # pd_op.meshgrid: ([16x12xf32, 16x12xf32]) <- ([16xf32, 12xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_8)

        # builtin.split: (16x12xf32, 16x12xf32) <- ([16x12xf32, 16x12xf32])
        (
            split_8,
            split_9,
        ) = meshgrid_4

        # pd_op.reshape: (192xf32) <- (16x12xf32, 1xi64)
        reshape_24 = paddle._C_ops.reshape(split_9, full_int_array_1)

        # pd_op.reshape: (192xf32) <- (16x12xf32, 1xi64)
        reshape_25 = paddle._C_ops.reshape(split_8, full_int_array_1)

        # builtin.combine: ([192xf32, 192xf32, 192xf32, 192xf32]) <- (192xf32, 192xf32, 192xf32, 192xf32)
        combine_9 = [reshape_24, reshape_25, reshape_24, reshape_25]

        # pd_op.stack: (192x4xf32) <- ([192xf32, 192xf32, 192xf32, 192xf32])
        stack_4 = paddle._C_ops.stack(combine_9, 1)

        # pd_op.reshape: (192x1x4xf32) <- (192x4xf32, 3xi64)
        reshape_26 = paddle._C_ops.reshape(stack_4, full_int_array_2)

        # pd_op.reshape: (1x3x4xf32) <- (3x4xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(data_9, full_int_array_3)

        # pd_op.add: (192x3x4xf32) <- (192x1x4xf32, 1x3x4xf32)
        add_19 = paddle._C_ops.add(reshape_26, reshape_27)

        # pd_op.reshape: (576x4xf32) <- (192x3x4xf32, 2xi64)
        reshape_7 = paddle._C_ops.reshape(add_19, full_int_array_4)
        return (
            conv2d_0,
            reshape_0,
            relu_0,
            conv2d_1,
            assign_0,
            relu_1,
            conv2d_2,
            assign_1,
            relu_2,
            conv2d_3,
            assign_2,
            relu_3,
            conv2d_4,
            assign_3,
            relu_4,
            conv2d_5,
            reshape_1,
            conv2d_6,
            reshape_2,
            conv2d_7,
            assign_4,
            conv2d_8,
            assign_5,
            conv2d_9,
            assign_6,
            conv2d_10,
            assign_7,
            conv2d_11,
            assign_8,
            conv2d_12,
            assign_9,
            conv2d_13,
            assign_10,
            conv2d_14,
            assign_11,
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
            reshape_3,
            reshape_4,
            reshape_5,
            reshape_6,
            reshape_7,
        )
