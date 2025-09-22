import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("640"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("4"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (160xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("856"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (214xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_3, full_2, dtype="float32")
        del full_2

        # builtin.combine: ([214xf32, 160xf32]) <- (214xf32, 160xf32)
        combine_0 = [arange_1, arange_0]
        del arange_0, arange_1

        # pd_op.meshgrid: ([214x160xf32, 214x160xf32]) <- ([214xf32, 160xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (214x160xf32, 214x160xf32) <- ([214x160xf32, 214x160xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.reshape: (34240xf32) <- (214x160xf32, 1xi64)
        reshape_6 = paddle._C_ops.reshape(split_1, full_int_array_0)
        del split_1

        # pd_op.reshape: (34240xf32) <- (214x160xf32, 1xi64)
        reshape_7 = paddle._C_ops.reshape(split_0, full_int_array_0)
        del split_0

        # builtin.combine: ([34240xf32, 34240xf32, 34240xf32, 34240xf32]) <- (34240xf32, 34240xf32, 34240xf32, 34240xf32)
        combine_1 = [reshape_6, reshape_7, reshape_6, reshape_7]
        del reshape_6, reshape_7

        # pd_op.stack: (34240x4xf32) <- ([34240xf32, 34240xf32, 34240xf32, 34240xf32])
        stack_0 = paddle._C_ops.stack(combine_1, 1)
        del combine_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [-1, 1, 4]

        # pd_op.reshape: (34240x1x4xf32) <- (34240x4xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(stack_0, full_int_array_1)
        del stack_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, -1, 4]

        # pd_op.reshape: (1x1x4xf32) <- (1x4xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(data_0, full_int_array_2)
        del data_0

        # pd_op.add: (34240x1x4xf32) <- (34240x1x4xf32, 1x1x4xf32)
        add_0 = paddle._C_ops.add(reshape_8, reshape_9)
        del reshape_8, reshape_9

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [-1, 4]

        # pd_op.reshape: (34240x4xf32) <- (34240x1x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(add_0, full_int_array_3)
        del add_0

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("8"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (80xf32) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_1, full_4, dtype="float32")

        # pd_op.arange: (107xf32) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_0, full_3, full_4, dtype="float32")
        del full_3, full_4

        # builtin.combine: ([107xf32, 80xf32]) <- (107xf32, 80xf32)
        combine_2 = [arange_3, arange_2]
        del arange_2, arange_3

        # pd_op.meshgrid: ([107x80xf32, 107x80xf32]) <- ([107xf32, 80xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_2)
        del combine_2

        # builtin.split: (107x80xf32, 107x80xf32) <- ([107x80xf32, 107x80xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # pd_op.reshape: (8560xf32) <- (107x80xf32, 1xi64)
        reshape_10 = paddle._C_ops.reshape(split_3, full_int_array_0)
        del split_3

        # pd_op.reshape: (8560xf32) <- (107x80xf32, 1xi64)
        reshape_11 = paddle._C_ops.reshape(split_2, full_int_array_0)
        del split_2

        # builtin.combine: ([8560xf32, 8560xf32, 8560xf32, 8560xf32]) <- (8560xf32, 8560xf32, 8560xf32, 8560xf32)
        combine_3 = [reshape_10, reshape_11, reshape_10, reshape_11]
        del reshape_10, reshape_11

        # pd_op.stack: (8560x4xf32) <- ([8560xf32, 8560xf32, 8560xf32, 8560xf32])
        stack_1 = paddle._C_ops.stack(combine_3, 1)
        del combine_3

        # pd_op.reshape: (8560x1x4xf32) <- (8560x4xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(stack_1, full_int_array_1)
        del stack_1

        # pd_op.reshape: (1x1x4xf32) <- (1x4xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(data_1, full_int_array_2)
        del data_1

        # pd_op.add: (8560x1x4xf32) <- (8560x1x4xf32, 1x1x4xf32)
        add_1 = paddle._C_ops.add(reshape_12, reshape_13)
        del reshape_12, reshape_13

        # pd_op.reshape: (8560x4xf32) <- (8560x1x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(add_1, full_int_array_3)
        del add_1

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("16"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (40xf32) <- (1xf64, 1xf64, 1xf64)
        arange_4 = paddle.arange(full_0, full_1, full_5, dtype="float32")

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("864"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (54xf32) <- (1xf64, 1xf64, 1xf64)
        arange_5 = paddle.arange(full_0, full_6, full_5, dtype="float32")
        del full_5

        # builtin.combine: ([54xf32, 40xf32]) <- (54xf32, 40xf32)
        combine_4 = [arange_5, arange_4]
        del arange_4, arange_5

        # pd_op.meshgrid: ([54x40xf32, 54x40xf32]) <- ([54xf32, 40xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_4)
        del combine_4

        # builtin.split: (54x40xf32, 54x40xf32) <- ([54x40xf32, 54x40xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # pd_op.reshape: (2160xf32) <- (54x40xf32, 1xi64)
        reshape_14 = paddle._C_ops.reshape(split_5, full_int_array_0)
        del split_5

        # pd_op.reshape: (2160xf32) <- (54x40xf32, 1xi64)
        reshape_15 = paddle._C_ops.reshape(split_4, full_int_array_0)
        del split_4

        # builtin.combine: ([2160xf32, 2160xf32, 2160xf32, 2160xf32]) <- (2160xf32, 2160xf32, 2160xf32, 2160xf32)
        combine_5 = [reshape_14, reshape_15, reshape_14, reshape_15]
        del reshape_14, reshape_15

        # pd_op.stack: (2160x4xf32) <- ([2160xf32, 2160xf32, 2160xf32, 2160xf32])
        stack_2 = paddle._C_ops.stack(combine_5, 1)
        del combine_5

        # pd_op.reshape: (2160x1x4xf32) <- (2160x4xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(stack_2, full_int_array_1)
        del stack_2

        # pd_op.reshape: (1x1x4xf32) <- (1x4xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(data_2, full_int_array_2)
        del data_2

        # pd_op.add: (2160x1x4xf32) <- (2160x1x4xf32, 1x1x4xf32)
        add_2 = paddle._C_ops.add(reshape_16, reshape_17)
        del reshape_16, reshape_17

        # pd_op.reshape: (2160x4xf32) <- (2160x1x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(add_2, full_int_array_3)
        del add_2

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("32"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (20xf32) <- (1xf64, 1xf64, 1xf64)
        arange_6 = paddle.arange(full_0, full_1, full_7, dtype="float32")

        # pd_op.arange: (27xf32) <- (1xf64, 1xf64, 1xf64)
        arange_7 = paddle.arange(full_0, full_6, full_7, dtype="float32")
        del full_6, full_7

        # builtin.combine: ([27xf32, 20xf32]) <- (27xf32, 20xf32)
        combine_6 = [arange_7, arange_6]
        del arange_6, arange_7

        # pd_op.meshgrid: ([27x20xf32, 27x20xf32]) <- ([27xf32, 20xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_6)
        del combine_6

        # builtin.split: (27x20xf32, 27x20xf32) <- ([27x20xf32, 27x20xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3
        del meshgrid_3

        # pd_op.reshape: (540xf32) <- (27x20xf32, 1xi64)
        reshape_18 = paddle._C_ops.reshape(split_7, full_int_array_0)
        del split_7

        # pd_op.reshape: (540xf32) <- (27x20xf32, 1xi64)
        reshape_19 = paddle._C_ops.reshape(split_6, full_int_array_0)
        del split_6

        # builtin.combine: ([540xf32, 540xf32, 540xf32, 540xf32]) <- (540xf32, 540xf32, 540xf32, 540xf32)
        combine_7 = [reshape_18, reshape_19, reshape_18, reshape_19]
        del reshape_18, reshape_19

        # pd_op.stack: (540x4xf32) <- ([540xf32, 540xf32, 540xf32, 540xf32])
        stack_3 = paddle._C_ops.stack(combine_7, 1)
        del combine_7

        # pd_op.reshape: (540x1x4xf32) <- (540x4xf32, 3xi64)
        reshape_20 = paddle._C_ops.reshape(stack_3, full_int_array_1)
        del stack_3

        # pd_op.reshape: (1x1x4xf32) <- (1x4xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(data_3, full_int_array_2)
        del data_3

        # pd_op.add: (540x1x4xf32) <- (540x1x4xf32, 1x1x4xf32)
        add_3 = paddle._C_ops.add(reshape_20, reshape_21)
        del reshape_20, reshape_21

        # pd_op.reshape: (540x4xf32) <- (540x1x4xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(add_3, full_int_array_3)
        del add_3

        # pd_op.full: (1xf64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("64"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (10xf32) <- (1xf64, 1xf64, 1xf64)
        arange_8 = paddle.arange(full_0, full_1, full_8, dtype="float32")

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("896"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (14xf32) <- (1xf64, 1xf64, 1xf64)
        arange_9 = paddle.arange(full_0, full_9, full_8, dtype="float32")
        del full_8

        # builtin.combine: ([14xf32, 10xf32]) <- (14xf32, 10xf32)
        combine_8 = [arange_9, arange_8]
        del arange_8, arange_9

        # pd_op.meshgrid: ([14x10xf32, 14x10xf32]) <- ([14xf32, 10xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_8)
        del combine_8

        # builtin.split: (14x10xf32, 14x10xf32) <- ([14x10xf32, 14x10xf32])
        (
            split_8,
            split_9,
        ) = meshgrid_4
        del meshgrid_4

        # pd_op.reshape: (140xf32) <- (14x10xf32, 1xi64)
        reshape_22 = paddle._C_ops.reshape(split_9, full_int_array_0)
        del split_9

        # pd_op.reshape: (140xf32) <- (14x10xf32, 1xi64)
        reshape_23 = paddle._C_ops.reshape(split_8, full_int_array_0)
        del split_8

        # builtin.combine: ([140xf32, 140xf32, 140xf32, 140xf32]) <- (140xf32, 140xf32, 140xf32, 140xf32)
        combine_9 = [reshape_22, reshape_23, reshape_22, reshape_23]
        del reshape_22, reshape_23

        # pd_op.stack: (140x4xf32) <- ([140xf32, 140xf32, 140xf32, 140xf32])
        stack_4 = paddle._C_ops.stack(combine_9, 1)
        del combine_9

        # pd_op.reshape: (140x1x4xf32) <- (140x4xf32, 3xi64)
        reshape_24 = paddle._C_ops.reshape(stack_4, full_int_array_1)
        del stack_4

        # pd_op.reshape: (1x1x4xf32) <- (1x4xf32, 3xi64)
        reshape_25 = paddle._C_ops.reshape(data_4, full_int_array_2)
        del data_4

        # pd_op.add: (140x1x4xf32) <- (140x1x4xf32, 1x1x4xf32)
        add_4 = paddle._C_ops.add(reshape_24, reshape_25)
        del reshape_24, reshape_25

        # pd_op.reshape: (140x4xf32) <- (140x1x4xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(add_4, full_int_array_3)
        del add_4

        # pd_op.full: (1xf64) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("128"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (5xf32) <- (1xf64, 1xf64, 1xf64)
        arange_10 = paddle.arange(full_0, full_1, full_10, dtype="float32")
        del full_1

        # pd_op.arange: (7xf32) <- (1xf64, 1xf64, 1xf64)
        arange_11 = paddle.arange(full_0, full_9, full_10, dtype="float32")
        del full_0, full_10, full_9

        # builtin.combine: ([7xf32, 5xf32]) <- (7xf32, 5xf32)
        combine_10 = [arange_11, arange_10]
        del arange_10, arange_11

        # pd_op.meshgrid: ([7x5xf32, 7x5xf32]) <- ([7xf32, 5xf32])
        meshgrid_5 = paddle._C_ops.meshgrid(combine_10)
        del combine_10

        # builtin.split: (7x5xf32, 7x5xf32) <- ([7x5xf32, 7x5xf32])
        (
            split_10,
            split_11,
        ) = meshgrid_5
        del meshgrid_5

        # pd_op.reshape: (35xf32) <- (7x5xf32, 1xi64)
        reshape_26 = paddle._C_ops.reshape(split_11, full_int_array_0)
        del split_11

        # pd_op.reshape: (35xf32) <- (7x5xf32, 1xi64)
        reshape_27 = paddle._C_ops.reshape(split_10, full_int_array_0)
        del full_int_array_0, split_10

        # builtin.combine: ([35xf32, 35xf32, 35xf32, 35xf32]) <- (35xf32, 35xf32, 35xf32, 35xf32)
        combine_11 = [reshape_26, reshape_27, reshape_26, reshape_27]
        del reshape_26, reshape_27

        # pd_op.stack: (35x4xf32) <- ([35xf32, 35xf32, 35xf32, 35xf32])
        stack_5 = paddle._C_ops.stack(combine_11, 1)
        del combine_11

        # pd_op.reshape: (35x1x4xf32) <- (35x4xf32, 3xi64)
        reshape_28 = paddle._C_ops.reshape(stack_5, full_int_array_1)
        del full_int_array_1, stack_5

        # pd_op.reshape: (1x1x4xf32) <- (1x4xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(data_5, full_int_array_2)
        del data_5, full_int_array_2

        # pd_op.add: (35x1x4xf32) <- (35x1x4xf32, 1x1x4xf32)
        add_5 = paddle._C_ops.add(reshape_28, reshape_29)
        del reshape_28, reshape_29

        # pd_op.reshape: (35x4xf32) <- (35x1x4xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(add_5, full_int_array_3)
        del add_5, full_int_array_3

        return reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5
