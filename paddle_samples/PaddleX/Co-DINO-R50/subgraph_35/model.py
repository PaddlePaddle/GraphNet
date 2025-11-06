import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        # pd_op.full: (160xi32) <- ()
        full_0 = paddle._C_ops.full(
            [160], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (214xi32) <- ()
        full_1 = paddle._C_ops.full(
            [214], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.set_value_: (160xi32) <- (160xi32, 0xi64, 0xi64, 0xi64)
        set_value__0 = paddle._C_ops.set_value_(
            full_0,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_0

        # pd_op.set_value_: (214xi32) <- (214xi32, 0xi64, 0xi64, 0xi64)
        set_value__1 = paddle._C_ops.set_value_(
            full_1,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_1

        # builtin.combine: ([214xi32, 160xi32]) <- (214xi32, 160xi32)
        combine_0 = [set_value__1, set_value__0]

        # pd_op.meshgrid: ([214x160xi32, 214x160xi32]) <- ([214xi32, 160xi32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (214x160xi32, 214x160xi32) <- ([214x160xi32, 214x160xi32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.reshape: (34240xi32) <- (214x160xi32, 1xi64)
        reshape_6 = paddle._C_ops.reshape(split_0, full_int_array_1)
        del split_0

        # pd_op.reshape: (34240xi32) <- (214x160xi32, 1xi64)
        reshape_7 = paddle._C_ops.reshape(split_1, full_int_array_1)
        del split_1

        # pd_op.bitwise_and: (34240xi32) <- (34240xi32, 34240xi32)
        bitwise_and_0 = paddle._C_ops.bitwise_and(reshape_7, reshape_6)
        del reshape_6, reshape_7

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 1]

        # pd_op.reshape: (34240x1xi32) <- (34240xi32, 2xi64)
        reshape_8 = paddle._C_ops.reshape(bitwise_and_0, full_int_array_2)
        del bitwise_and_0

        # pd_op.expand: (34240x1xi32) <- (34240x1xi32, 2xi64)
        expand_0 = paddle._C_ops.expand(reshape_8, full_int_array_2)
        del reshape_8

        # pd_op.reshape: (34240xi32) <- (34240x1xi32, 1xi64)
        reshape_0 = paddle._C_ops.reshape(expand_0, full_int_array_1)
        del expand_0

        # pd_op.full: (80xi32) <- ()
        full_2 = paddle._C_ops.full(
            [80], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (107xi32) <- ()
        full_3 = paddle._C_ops.full(
            [107], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.set_value_: (80xi32) <- (80xi32, 0xi64, 0xi64, 0xi64)
        set_value__2 = paddle._C_ops.set_value_(
            full_2,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_2

        # pd_op.set_value_: (107xi32) <- (107xi32, 0xi64, 0xi64, 0xi64)
        set_value__3 = paddle._C_ops.set_value_(
            full_3,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_3

        # builtin.combine: ([107xi32, 80xi32]) <- (107xi32, 80xi32)
        combine_1 = [set_value__3, set_value__2]

        # pd_op.meshgrid: ([107x80xi32, 107x80xi32]) <- ([107xi32, 80xi32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_1)
        del combine_1

        # builtin.split: (107x80xi32, 107x80xi32) <- ([107x80xi32, 107x80xi32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # pd_op.reshape: (8560xi32) <- (107x80xi32, 1xi64)
        reshape_9 = paddle._C_ops.reshape(split_2, full_int_array_1)
        del split_2

        # pd_op.reshape: (8560xi32) <- (107x80xi32, 1xi64)
        reshape_10 = paddle._C_ops.reshape(split_3, full_int_array_1)
        del split_3

        # pd_op.bitwise_and: (8560xi32) <- (8560xi32, 8560xi32)
        bitwise_and_1 = paddle._C_ops.bitwise_and(reshape_10, reshape_9)
        del reshape_10, reshape_9

        # pd_op.reshape: (8560x1xi32) <- (8560xi32, 2xi64)
        reshape_11 = paddle._C_ops.reshape(bitwise_and_1, full_int_array_2)
        del bitwise_and_1

        # pd_op.expand: (8560x1xi32) <- (8560x1xi32, 2xi64)
        expand_1 = paddle._C_ops.expand(reshape_11, full_int_array_2)
        del reshape_11

        # pd_op.reshape: (8560xi32) <- (8560x1xi32, 1xi64)
        reshape_1 = paddle._C_ops.reshape(expand_1, full_int_array_1)
        del expand_1

        # pd_op.full: (40xi32) <- ()
        full_4 = paddle._C_ops.full(
            [40], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (54xi32) <- ()
        full_5 = paddle._C_ops.full(
            [54], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.set_value_: (40xi32) <- (40xi32, 0xi64, 0xi64, 0xi64)
        set_value__4 = paddle._C_ops.set_value_(
            full_4,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_4

        # pd_op.set_value_: (54xi32) <- (54xi32, 0xi64, 0xi64, 0xi64)
        set_value__5 = paddle._C_ops.set_value_(
            full_5,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_5

        # builtin.combine: ([54xi32, 40xi32]) <- (54xi32, 40xi32)
        combine_2 = [set_value__5, set_value__4]

        # pd_op.meshgrid: ([54x40xi32, 54x40xi32]) <- ([54xi32, 40xi32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_2)
        del combine_2

        # builtin.split: (54x40xi32, 54x40xi32) <- ([54x40xi32, 54x40xi32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # pd_op.reshape: (2160xi32) <- (54x40xi32, 1xi64)
        reshape_12 = paddle._C_ops.reshape(split_4, full_int_array_1)
        del split_4

        # pd_op.reshape: (2160xi32) <- (54x40xi32, 1xi64)
        reshape_13 = paddle._C_ops.reshape(split_5, full_int_array_1)
        del split_5

        # pd_op.bitwise_and: (2160xi32) <- (2160xi32, 2160xi32)
        bitwise_and_2 = paddle._C_ops.bitwise_and(reshape_13, reshape_12)
        del reshape_12, reshape_13

        # pd_op.reshape: (2160x1xi32) <- (2160xi32, 2xi64)
        reshape_14 = paddle._C_ops.reshape(bitwise_and_2, full_int_array_2)
        del bitwise_and_2

        # pd_op.expand: (2160x1xi32) <- (2160x1xi32, 2xi64)
        expand_2 = paddle._C_ops.expand(reshape_14, full_int_array_2)
        del reshape_14

        # pd_op.reshape: (2160xi32) <- (2160x1xi32, 1xi64)
        reshape_2 = paddle._C_ops.reshape(expand_2, full_int_array_1)
        del expand_2

        # pd_op.full: (20xi32) <- ()
        full_6 = paddle._C_ops.full(
            [20], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (27xi32) <- ()
        full_7 = paddle._C_ops.full(
            [27], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.set_value_: (20xi32) <- (20xi32, 0xi64, 0xi64, 0xi64)
        set_value__6 = paddle._C_ops.set_value_(
            full_6,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_6

        # pd_op.set_value_: (27xi32) <- (27xi32, 0xi64, 0xi64, 0xi64)
        set_value__7 = paddle._C_ops.set_value_(
            full_7,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_7

        # builtin.combine: ([27xi32, 20xi32]) <- (27xi32, 20xi32)
        combine_3 = [set_value__7, set_value__6]

        # pd_op.meshgrid: ([27x20xi32, 27x20xi32]) <- ([27xi32, 20xi32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_3)
        del combine_3

        # builtin.split: (27x20xi32, 27x20xi32) <- ([27x20xi32, 27x20xi32])
        (
            split_6,
            split_7,
        ) = meshgrid_3
        del meshgrid_3

        # pd_op.reshape: (540xi32) <- (27x20xi32, 1xi64)
        reshape_15 = paddle._C_ops.reshape(split_6, full_int_array_1)
        del split_6

        # pd_op.reshape: (540xi32) <- (27x20xi32, 1xi64)
        reshape_16 = paddle._C_ops.reshape(split_7, full_int_array_1)
        del split_7

        # pd_op.bitwise_and: (540xi32) <- (540xi32, 540xi32)
        bitwise_and_3 = paddle._C_ops.bitwise_and(reshape_16, reshape_15)
        del reshape_15, reshape_16

        # pd_op.reshape: (540x1xi32) <- (540xi32, 2xi64)
        reshape_17 = paddle._C_ops.reshape(bitwise_and_3, full_int_array_2)
        del bitwise_and_3

        # pd_op.expand: (540x1xi32) <- (540x1xi32, 2xi64)
        expand_3 = paddle._C_ops.expand(reshape_17, full_int_array_2)
        del reshape_17

        # pd_op.reshape: (540xi32) <- (540x1xi32, 1xi64)
        reshape_3 = paddle._C_ops.reshape(expand_3, full_int_array_1)
        del expand_3

        # pd_op.full: (10xi32) <- ()
        full_8 = paddle._C_ops.full(
            [10], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (14xi32) <- ()
        full_9 = paddle._C_ops.full(
            [14], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.set_value_: (10xi32) <- (10xi32, 0xi64, 0xi64, 0xi64)
        set_value__8 = paddle._C_ops.set_value_(
            full_8,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_8

        # pd_op.set_value_: (14xi32) <- (14xi32, 0xi64, 0xi64, 0xi64)
        set_value__9 = paddle._C_ops.set_value_(
            full_9,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_9

        # builtin.combine: ([14xi32, 10xi32]) <- (14xi32, 10xi32)
        combine_4 = [set_value__9, set_value__8]

        # pd_op.meshgrid: ([14x10xi32, 14x10xi32]) <- ([14xi32, 10xi32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_4)
        del combine_4

        # builtin.split: (14x10xi32, 14x10xi32) <- ([14x10xi32, 14x10xi32])
        (
            split_8,
            split_9,
        ) = meshgrid_4
        del meshgrid_4

        # pd_op.reshape: (140xi32) <- (14x10xi32, 1xi64)
        reshape_18 = paddle._C_ops.reshape(split_8, full_int_array_1)
        del split_8

        # pd_op.reshape: (140xi32) <- (14x10xi32, 1xi64)
        reshape_19 = paddle._C_ops.reshape(split_9, full_int_array_1)
        del split_9

        # pd_op.bitwise_and: (140xi32) <- (140xi32, 140xi32)
        bitwise_and_4 = paddle._C_ops.bitwise_and(reshape_19, reshape_18)
        del reshape_18, reshape_19

        # pd_op.reshape: (140x1xi32) <- (140xi32, 2xi64)
        reshape_20 = paddle._C_ops.reshape(bitwise_and_4, full_int_array_2)
        del bitwise_and_4

        # pd_op.expand: (140x1xi32) <- (140x1xi32, 2xi64)
        expand_4 = paddle._C_ops.expand(reshape_20, full_int_array_2)
        del reshape_20

        # pd_op.reshape: (140xi32) <- (140x1xi32, 1xi64)
        reshape_4 = paddle._C_ops.reshape(expand_4, full_int_array_1)
        del expand_4

        # pd_op.full: (5xi32) <- ()
        full_10 = paddle._C_ops.full(
            [5], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (7xi32) <- ()
        full_11 = paddle._C_ops.full(
            [7], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.set_value_: (5xi32) <- (5xi32, 0xi64, 0xi64, 0xi64)
        set_value__10 = paddle._C_ops.set_value_(
            full_10,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_10

        # pd_op.set_value_: (7xi32) <- (7xi32, 0xi64, 0xi64, 0xi64)
        set_value__11 = paddle._C_ops.set_value_(
            full_11,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_11, full_int_array_0

        # builtin.combine: ([7xi32, 5xi32]) <- (7xi32, 5xi32)
        combine_5 = [set_value__11, set_value__10]

        # pd_op.meshgrid: ([7x5xi32, 7x5xi32]) <- ([7xi32, 5xi32])
        meshgrid_5 = paddle._C_ops.meshgrid(combine_5)
        del combine_5

        # builtin.split: (7x5xi32, 7x5xi32) <- ([7x5xi32, 7x5xi32])
        (
            split_10,
            split_11,
        ) = meshgrid_5
        del meshgrid_5

        # pd_op.reshape: (35xi32) <- (7x5xi32, 1xi64)
        reshape_21 = paddle._C_ops.reshape(split_10, full_int_array_1)
        del split_10

        # pd_op.reshape: (35xi32) <- (7x5xi32, 1xi64)
        reshape_22 = paddle._C_ops.reshape(split_11, full_int_array_1)
        del split_11

        # pd_op.bitwise_and: (35xi32) <- (35xi32, 35xi32)
        bitwise_and_5 = paddle._C_ops.bitwise_and(reshape_22, reshape_21)
        del reshape_21, reshape_22

        # pd_op.reshape: (35x1xi32) <- (35xi32, 2xi64)
        reshape_23 = paddle._C_ops.reshape(bitwise_and_5, full_int_array_2)
        del bitwise_and_5

        # pd_op.expand: (35x1xi32) <- (35x1xi32, 2xi64)
        expand_5 = paddle._C_ops.expand(reshape_23, full_int_array_2)
        del full_int_array_2, reshape_23

        # pd_op.reshape: (35xi32) <- (35x1xi32, 1xi64)
        reshape_5 = paddle._C_ops.reshape(expand_5, full_int_array_1)
        del (
            expand_5,
            full_int_array_1,
            set_value__0,
            set_value__1,
            set_value__10,
            set_value__11,
            set_value__2,
            set_value__3,
            set_value__4,
            set_value__5,
            set_value__6,
            set_value__7,
            set_value__8,
            set_value__9,
        )

        return reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5
