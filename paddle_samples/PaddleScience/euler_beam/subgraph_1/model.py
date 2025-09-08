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
        # pd_op.matmul: (100x20xf32) <- (100x1xf32, 1x20xf32)
        matmul_0 = paddle._C_ops.matmul(data_1, parameter_8, False, False)

        # pd_op.add: (100x20xf32) <- (100x20xf32, 20xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_7)
        del matmul_0

        # pd_op.tanh: (100x20xf32) <- (100x20xf32)
        tanh_0 = paddle._C_ops.tanh(add_1)
        del add_1

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_1 = paddle._C_ops.matmul(tanh_0, parameter_6, False, False)

        # pd_op.add: (100x20xf32) <- (100x20xf32, 20xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_5)
        del matmul_1

        # pd_op.tanh: (100x20xf32) <- (100x20xf32)
        tanh_1 = paddle._C_ops.tanh(add_2)
        del add_2

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_2 = paddle._C_ops.matmul(tanh_1, parameter_4, False, False)

        # pd_op.add: (100x20xf32) <- (100x20xf32, 20xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_3)
        del matmul_2

        # pd_op.tanh: (100x20xf32) <- (100x20xf32)
        tanh_2 = paddle._C_ops.tanh(add_3)
        del add_3

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 20x1xf32)
        matmul_3 = paddle._C_ops.matmul(tanh_2, parameter_2, False, False)

        # pd_op.add: (100x1xf32) <- (100x1xf32, 1xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_1)
        del matmul_3

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (100x1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [100, 1],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (100x1xf32) <- (100x1xf32)
        assign_0 = full_1

        # pd_op.assign: (100x1xf32) <- (100x1xf32)
        assign_1 = full_1

        # pd_op.assign: (100x1xf32) <- (100x1xf32)
        assign_2 = full_1

        # pd_op.assign: (100x1xf32) <- (100x1xf32)
        assign_3 = full_1

        # pd_op.matmul: (100x20xf32) <- (100x1xf32, 20x1xf32)
        matmul_4 = paddle._C_ops.matmul(assign_3, parameter_2, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_0 = paddle._C_ops.multiply(tanh_2, tanh_2)

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.subtract: (100x20xf32) <- (xf32, 100x20xf32)
        subtract_0 = paddle._C_ops.subtract(full_2, multiply_0)
        del multiply_0

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_4 = subtract_0

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_5 = subtract_0

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_6 = subtract_0

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_1 = paddle._C_ops.multiply(matmul_4, subtract_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_7 = multiply_1
        del multiply_1

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_5 = paddle._C_ops.matmul(assign_7, parameter_4, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_2 = paddle._C_ops.multiply(tanh_1, tanh_1)

        # pd_op.subtract: (100x20xf32) <- (xf32, 100x20xf32)
        subtract_1 = paddle._C_ops.subtract(full_2, multiply_2)
        del multiply_2

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_8 = subtract_1

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_9 = subtract_1

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_10 = subtract_1

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_3 = paddle._C_ops.multiply(matmul_5, subtract_1)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_11 = multiply_3
        del multiply_3

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_6 = paddle._C_ops.matmul(assign_11, parameter_6, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_4 = paddle._C_ops.multiply(tanh_0, tanh_0)

        # pd_op.subtract: (100x20xf32) <- (xf32, 100x20xf32)
        subtract_2 = paddle._C_ops.subtract(full_2, multiply_4)
        del multiply_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_12 = subtract_2

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_13 = subtract_2

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_14 = subtract_2

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_5 = paddle._C_ops.multiply(matmul_6, subtract_2)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_15 = multiply_5
        del multiply_5

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 1x20xf32)
        matmul_7 = paddle._C_ops.matmul(assign_15, parameter_8, False, True)

        # pd_op.matmul: (1x20xf32) <- (100x1xf32, 100x20xf32)
        matmul_8 = paddle._C_ops.matmul(data_1, assign_15, True, False)
        del data_1

        # pd_op.matmul: (100x20xf32) <- (100x1xf32, 1x20xf32)
        matmul_9 = paddle._C_ops.matmul(full_1, parameter_8, False, False)

        # pd_op.matmul: (1x20xf32) <- (100x1xf32, 100x20xf32)
        matmul_10 = paddle._C_ops.matmul(full_1, assign_15, True, False)
        del assign_15

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_16 = matmul_9
        del matmul_9

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_17 = assign_16

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_18 = assign_16

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_6 = paddle._C_ops.multiply(assign_16, subtract_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_7 = paddle._C_ops.multiply(assign_16, matmul_6)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_11 = paddle._C_ops.matmul(multiply_6, parameter_6, False, False)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_7, full_3, float("0"), True)
        del multiply_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_19 = matmul_11
        del matmul_11

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_8 = paddle._C_ops.multiply(scale_1, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_9 = paddle._C_ops.multiply(assign_19, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_10 = paddle._C_ops.multiply(assign_19, matmul_5)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_12 = paddle._C_ops.matmul(multiply_9, parameter_4, False, False)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(multiply_10, full_3, float("0"), True)
        del multiply_10

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_20 = matmul_12
        del matmul_12

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_11 = paddle._C_ops.multiply(scale_2, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_12 = paddle._C_ops.multiply(assign_20, matmul_4)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(multiply_12, full_3, float("0"), True)
        del multiply_12

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_13 = paddle._C_ops.multiply(scale_3, tanh_2)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_0 = [multiply_13, multiply_13]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_5 = paddle._C_ops.add(multiply_13, multiply_13)
        del multiply_13

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_14 = paddle._C_ops.multiply(add_5, subtract_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_21 = multiply_14
        del multiply_14

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_13 = paddle._C_ops.matmul(assign_21, parameter_4, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32)
        combine_1 = [multiply_11, multiply_11, matmul_13]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_6 = paddle._C_ops.add(multiply_11, multiply_11)
        del multiply_11

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_7 = paddle._C_ops.add(add_6, matmul_13)
        del add_6, matmul_13

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_15 = paddle._C_ops.multiply(add_7, subtract_1)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_22 = multiply_15
        del multiply_15

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_14 = paddle._C_ops.matmul(assign_22, parameter_6, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32)
        combine_2 = [multiply_8, multiply_8, matmul_14]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_8 = paddle._C_ops.add(multiply_8, multiply_8)
        del multiply_8

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_9 = paddle._C_ops.add(add_8, matmul_14)
        del add_8, matmul_14

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_16 = paddle._C_ops.multiply(add_9, subtract_2)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_23 = multiply_16
        del multiply_16

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 1x20xf32)
        matmul_15 = paddle._C_ops.matmul(assign_23, parameter_8, False, True)

        # pd_op.matmul: (1x20xf32) <- (100x1xf32, 100x20xf32)
        matmul_16 = paddle._C_ops.matmul(full_1, assign_23, True, False)
        del assign_23

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_17 = paddle._C_ops.multiply(assign_16, add_9)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(multiply_6, full_0, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_24 = scale_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_25 = scale_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_26 = scale_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_27 = scale_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_28 = scale_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_29 = scale_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_30 = scale_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_31 = scale_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_32 = scale_4

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(multiply_17, full_3, float("0"), True)
        del multiply_17

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_18 = paddle._C_ops.multiply(scale_5, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_19 = paddle._C_ops.multiply(scale_4, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_20 = paddle._C_ops.multiply(scale_4, scale_1)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_17 = paddle._C_ops.matmul(scale_4, parameter_6, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_3 = [multiply_19, multiply_19]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_10 = paddle._C_ops.add(multiply_19, multiply_19)
        del multiply_19

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(add_10, full_3, float("0"), True)
        del add_10

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_33 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_34 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_35 = matmul_17
        del matmul_17

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_36 = assign_35

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_21 = paddle._C_ops.multiply(scale_6, assign_16)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_37 = multiply_21

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_22 = paddle._C_ops.multiply(assign_35, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_23 = paddle._C_ops.multiply(assign_35, add_7)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(multiply_22, full_0, float("0"), True)
        del multiply_22

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_38 = scale_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_39 = scale_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_40 = scale_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_41 = scale_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_42 = scale_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_43 = scale_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_44 = scale_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_45 = scale_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_46 = scale_7

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(multiply_23, full_3, float("0"), True)
        del multiply_23

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_24 = paddle._C_ops.multiply(scale_8, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_25 = paddle._C_ops.multiply(scale_7, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_26 = paddle._C_ops.multiply(scale_7, scale_2)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_18 = paddle._C_ops.matmul(scale_7, parameter_4, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_4 = [multiply_25, multiply_25]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_11 = paddle._C_ops.add(multiply_25, multiply_25)
        del multiply_25

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(add_11, full_3, float("0"), True)
        del add_11

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_47 = scale_9

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_48 = matmul_18
        del matmul_18

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_49 = assign_48

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_27 = paddle._C_ops.multiply(scale_9, matmul_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_28 = paddle._C_ops.multiply(scale_9, assign_19)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_29 = paddle._C_ops.multiply(assign_48, subtract_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_30 = paddle._C_ops.multiply(assign_48, add_5)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(multiply_29, full_0, float("0"), True)
        del multiply_29

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_50 = scale_10

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_51 = scale_10

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_52 = scale_10

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_53 = scale_10

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_54 = scale_10

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_55 = scale_10

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_56 = scale_10

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(multiply_30, full_3, float("0"), True)
        del multiply_30

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_31 = paddle._C_ops.multiply(scale_11, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_32 = paddle._C_ops.multiply(scale_10, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_33 = paddle._C_ops.multiply(scale_10, scale_3)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_5 = [multiply_32, multiply_32]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_12 = paddle._C_ops.add(multiply_32, multiply_32)
        del multiply_32

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(add_12, full_3, float("0"), True)
        del add_12

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_57 = scale_12

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_58 = scale_12

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_34 = paddle._C_ops.multiply(scale_12, matmul_4)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_59 = multiply_34
        del multiply_34

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_60 = assign_59

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_19 = paddle._C_ops.matmul(assign_59, parameter_4, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_35 = paddle._C_ops.multiply(matmul_19, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_36 = paddle._C_ops.multiply(matmul_19, assign_19)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_6 = [multiply_27, multiply_35]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_13 = paddle._C_ops.add(multiply_27, multiply_35)
        del multiply_27, multiply_35

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_61 = add_13
        del add_13

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_20 = paddle._C_ops.matmul(assign_61, parameter_6, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_37 = paddle._C_ops.multiply(matmul_20, assign_16)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_21 = paddle._C_ops.matmul(multiply_21, parameter_6, False, False)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(multiply_37, full_3, float("0"), True)
        del multiply_37

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_62 = matmul_21
        del matmul_21

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_38 = paddle._C_ops.multiply(scale_13, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_39 = paddle._C_ops.multiply(assign_62, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_40 = paddle._C_ops.multiply(assign_62, matmul_5)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_7 = [multiply_28, multiply_39]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_14 = paddle._C_ops.add(multiply_28, multiply_39)
        del multiply_28, multiply_39

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_22 = paddle._C_ops.matmul(add_14, parameter_4, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_8 = [multiply_36, multiply_40]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_15 = paddle._C_ops.add(multiply_36, multiply_40)
        del multiply_36, multiply_40

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(add_15, full_3, float("0"), True)
        del add_15

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_63 = matmul_22
        del matmul_22

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_41 = paddle._C_ops.multiply(scale_14, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_42 = paddle._C_ops.multiply(assign_63, matmul_4)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(multiply_42, full_3, float("0"), True)
        del multiply_42

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_43 = paddle._C_ops.multiply(scale_15, tanh_2)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_9 = [
            multiply_31,
            multiply_31,
            multiply_33,
            multiply_33,
            multiply_43,
            multiply_43,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_16 = paddle._C_ops.add(multiply_31, multiply_31)
        del multiply_31

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_17 = paddle._C_ops.add(add_16, multiply_33)
        del add_16

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_18 = paddle._C_ops.add(add_17, multiply_33)
        del add_17

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_19 = paddle._C_ops.add(add_18, multiply_43)
        del add_18

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_20 = paddle._C_ops.add(add_19, multiply_43)
        del add_19, multiply_43

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_44 = paddle._C_ops.multiply(add_20, subtract_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_64 = multiply_44
        del multiply_44

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_23 = paddle._C_ops.matmul(assign_64, parameter_4, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_10 = [
            multiply_24,
            multiply_24,
            multiply_26,
            multiply_26,
            multiply_41,
            multiply_41,
            matmul_23,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_21 = paddle._C_ops.add(multiply_24, multiply_24)
        del multiply_24

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_22 = paddle._C_ops.add(add_21, multiply_26)
        del add_21

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_23 = paddle._C_ops.add(add_22, multiply_26)
        del add_22

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_24 = paddle._C_ops.add(add_23, multiply_41)
        del add_23

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_25 = paddle._C_ops.add(add_24, multiply_41)
        del add_24, multiply_41

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_26 = paddle._C_ops.add(add_25, matmul_23)
        del add_25, matmul_23

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_45 = paddle._C_ops.multiply(add_26, subtract_1)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_65 = multiply_45
        del multiply_45

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_24 = paddle._C_ops.matmul(assign_65, parameter_6, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_11 = [
            multiply_18,
            multiply_18,
            multiply_20,
            multiply_20,
            multiply_38,
            multiply_38,
            matmul_24,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_27 = paddle._C_ops.add(multiply_18, multiply_18)
        del multiply_18

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_28 = paddle._C_ops.add(add_27, multiply_20)
        del add_27

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_29 = paddle._C_ops.add(add_28, multiply_20)
        del add_28

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_30 = paddle._C_ops.add(add_29, multiply_38)
        del add_29

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_31 = paddle._C_ops.add(add_30, multiply_38)
        del add_30, multiply_38

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_32 = paddle._C_ops.add(add_31, matmul_24)
        del add_31, matmul_24

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_46 = paddle._C_ops.multiply(add_32, subtract_2)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_66 = multiply_46
        del multiply_46

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 1x20xf32)
        matmul_25 = paddle._C_ops.matmul(assign_66, parameter_8, False, True)

        # pd_op.matmul: (1x20xf32) <- (100x1xf32, 100x20xf32)
        matmul_26 = paddle._C_ops.matmul(full_1, assign_66, True, False)
        del assign_66, full_1

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_47 = paddle._C_ops.multiply(assign_16, add_32)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(multiply_47, full_3, float("0"), True)
        del multiply_47

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_48 = paddle._C_ops.multiply(scale_16, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_49 = paddle._C_ops.multiply(scale_4, scale_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_50 = paddle._C_ops.multiply(scale_4, scale_4)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_51 = paddle._C_ops.multiply(scale_4, scale_13)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_52 = paddle._C_ops.multiply(assign_35, add_26)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(multiply_52, full_3, float("0"), True)
        del multiply_52

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_53 = paddle._C_ops.multiply(scale_17, tanh_1)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(assign_62, full_0, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_67 = scale_18

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_54 = paddle._C_ops.multiply(scale_7, scale_8)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_55 = paddle._C_ops.multiply(scale_7, scale_7)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_56 = paddle._C_ops.multiply(scale_7, scale_14)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_57 = paddle._C_ops.multiply(scale_18, matmul_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_58 = paddle._C_ops.multiply(scale_18, scale_9)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_59 = paddle._C_ops.multiply(scale_18, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_60 = paddle._C_ops.multiply(scale_18, matmul_19)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_61 = paddle._C_ops.multiply(scale_9, add_7)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_62 = paddle._C_ops.multiply(scale_9, assign_35)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_9, full_0, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_68 = scale_19

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_63 = paddle._C_ops.multiply(assign_48, add_20)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(multiply_63, full_3, float("0"), True)
        del multiply_63

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_64 = paddle._C_ops.multiply(scale_19, assign_19)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_65 = paddle._C_ops.multiply(scale_19, matmul_19)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_66 = paddle._C_ops.multiply(scale_19, matmul_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_67 = paddle._C_ops.multiply(scale_19, assign_62)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_68 = paddle._C_ops.multiply(scale_20, tanh_2)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_12 = [multiply_59, multiply_64]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_33 = paddle._C_ops.add(multiply_59, multiply_64)
        del multiply_59, multiply_64

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_27 = paddle._C_ops.matmul(add_33, parameter_4, False, False)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_69 = paddle._C_ops.multiply(scale_10, scale_11)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_70 = paddle._C_ops.multiply(scale_10, scale_10)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_71 = paddle._C_ops.multiply(scale_10, scale_15)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_69 = matmul_27
        del matmul_27

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_72 = paddle._C_ops.multiply(assign_69, matmul_4)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_73 = paddle._C_ops.multiply(scale_12, add_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_74 = paddle._C_ops.multiply(scale_12, assign_48)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(multiply_72, full_3, float("0"), True)
        del multiply_72

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(scale_21, full_0, float("0"), True)
        del scale_21

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_70 = scale_22

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(matmul_19, full_0, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_71 = scale_23

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_75 = paddle._C_ops.multiply(scale_22, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_76 = paddle._C_ops.multiply(scale_22, scale_10)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_13 = [multiply_33, multiply_75]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_34 = paddle._C_ops.add(multiply_33, multiply_75)
        del multiply_33, multiply_75

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(add_34, full_0, float("0"), True)
        del add_34

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_77 = paddle._C_ops.multiply(scale_23, assign_19)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_78 = paddle._C_ops.multiply(scale_23, scale_9)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_79 = paddle._C_ops.multiply(scale_23, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_80 = paddle._C_ops.multiply(scale_23, assign_62)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_14 = [scale_24, scale_24]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_35 = paddle._C_ops.add(scale_24, scale_24)
        del scale_24

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_81 = paddle._C_ops.multiply(add_35, subtract_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_82 = paddle._C_ops.multiply(add_35, assign_48)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_15 = [multiply_57, multiply_77]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_36 = paddle._C_ops.add(multiply_57, multiply_77)
        del multiply_57, multiply_77

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(add_36, full_3, float("0"), True)
        del add_36

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_16 = [multiply_66, multiply_79]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_37 = paddle._C_ops.add(multiply_66, multiply_79)
        del multiply_66, multiply_79

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_72 = add_37
        del add_37

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_17 = [multiply_73, multiply_81]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_38 = paddle._C_ops.add(multiply_73, multiply_81)
        del multiply_73, multiply_81

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_73 = add_38
        del add_38

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(scale_25, full_0, float("0"), True)
        del scale_25

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_74 = scale_26

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_28 = paddle._C_ops.matmul(assign_72, parameter_6, False, True)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_29 = paddle._C_ops.matmul(assign_73, parameter_4, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_83 = paddle._C_ops.multiply(matmul_28, assign_16)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(matmul_29, full_0, float("0"), True)
        del matmul_29

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_84 = paddle._C_ops.multiply(scale_26, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_85 = paddle._C_ops.multiply(scale_26, scale_7)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(multiply_83, full_3, float("0"), True)
        del multiply_83

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_18 = [multiply_26, multiply_84]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_39 = paddle._C_ops.add(multiply_26, multiply_84)
        del multiply_26, multiply_84

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(add_39, full_0, float("0"), True)
        del add_39

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(scale_28, full_0, float("0"), True)
        del scale_28

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_75 = scale_30

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32)
        combine_19 = [scale_27, scale_29, scale_29]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_40 = paddle._C_ops.add(scale_27, scale_29)
        del scale_27

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_41 = paddle._C_ops.add(add_40, scale_29)
        del add_40, scale_29

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_86 = paddle._C_ops.multiply(add_41, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_87 = paddle._C_ops.multiply(add_41, assign_35)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_20 = [multiply_61, multiply_86]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_42 = paddle._C_ops.add(multiply_61, multiply_86)
        del multiply_61, multiply_86

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_76 = add_42
        del add_42

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_88 = paddle._C_ops.multiply(scale_30, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_89 = paddle._C_ops.multiply(scale_30, scale_4)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_30 = paddle._C_ops.matmul(assign_76, parameter_6, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_21 = [multiply_20, multiply_88]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_43 = paddle._C_ops.add(multiply_20, multiply_88)
        del multiply_20, multiply_88

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(add_43, full_0, float("0"), True)
        del add_43

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(matmul_30, full_0, float("0"), True)
        del matmul_30

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32)
        combine_22 = [scale_31, scale_31, scale_32]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_44 = paddle._C_ops.add(scale_31, scale_31)
        del scale_31

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_45 = paddle._C_ops.add(add_44, scale_32)
        del add_44, scale_32

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_90 = paddle._C_ops.multiply(add_45, assign_16)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(multiply_21, full_0, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_77 = scale_33

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_78 = scale_33

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(multiply_90, full_3, float("0"), True)
        del multiply_90

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_91 = paddle._C_ops.multiply(scale_34, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_92 = paddle._C_ops.multiply(scale_33, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_93 = paddle._C_ops.multiply(scale_33, scale_1)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_31 = paddle._C_ops.matmul(scale_33, parameter_6, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_23 = [multiply_50, multiply_50, multiply_92, multiply_92]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_46 = paddle._C_ops.add(multiply_50, multiply_50)
        del multiply_50

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_47 = paddle._C_ops.add(add_46, multiply_92)
        del add_46

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_48 = paddle._C_ops.add(add_47, multiply_92)
        del add_47, multiply_92

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(add_48, full_3, float("0"), True)
        del add_48

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_79 = matmul_31
        del matmul_31

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_94 = paddle._C_ops.multiply(scale_35, assign_16)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_95 = paddle._C_ops.multiply(assign_79, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_96 = paddle._C_ops.multiply(assign_79, add_7)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_24 = [multiply_62, multiply_95]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_49 = paddle._C_ops.add(multiply_62, multiply_95)
        del multiply_62, multiply_95

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(add_49, full_0, float("0"), True)
        del add_49

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_80 = scale_36

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_81 = scale_36

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_25 = [multiply_87, multiply_96]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_50 = paddle._C_ops.add(multiply_87, multiply_96)
        del multiply_87, multiply_96

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(add_50, full_3, float("0"), True)
        del add_50

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_97 = paddle._C_ops.multiply(scale_37, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_98 = paddle._C_ops.multiply(scale_36, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_99 = paddle._C_ops.multiply(scale_36, scale_2)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_32 = paddle._C_ops.matmul(scale_36, parameter_4, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_26 = [multiply_55, multiply_55, multiply_98, multiply_98]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_51 = paddle._C_ops.add(multiply_55, multiply_55)
        del multiply_55

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_52 = paddle._C_ops.add(add_51, multiply_98)
        del add_51

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_53 = paddle._C_ops.add(add_52, multiply_98)
        del add_52, multiply_98

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(add_53, full_3, float("0"), True)
        del add_53

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_82 = matmul_32
        del matmul_32

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_100 = paddle._C_ops.multiply(scale_38, matmul_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_101 = paddle._C_ops.multiply(scale_38, assign_19)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_102 = paddle._C_ops.multiply(assign_82, subtract_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_103 = paddle._C_ops.multiply(assign_82, add_5)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_27 = [multiply_74, multiply_102]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_54 = paddle._C_ops.add(multiply_74, multiply_102)
        del multiply_102, multiply_74

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(add_54, full_0, float("0"), True)
        del add_54

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_83 = scale_39

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_28 = [multiply_82, multiply_103]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_55 = paddle._C_ops.add(multiply_82, multiply_103)
        del multiply_103, multiply_82

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_40 = paddle._C_ops.scale(add_55, full_3, float("0"), True)
        del add_55

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_104 = paddle._C_ops.multiply(scale_40, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_105 = paddle._C_ops.multiply(scale_39, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_106 = paddle._C_ops.multiply(scale_39, scale_3)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_29 = [multiply_70, multiply_70, multiply_105, multiply_105]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_56 = paddle._C_ops.add(multiply_70, multiply_70)
        del multiply_70

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_57 = paddle._C_ops.add(add_56, multiply_105)
        del add_56

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_58 = paddle._C_ops.add(add_57, multiply_105)
        del add_57, multiply_105

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_41 = paddle._C_ops.scale(add_58, full_3, float("0"), True)
        del add_58

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_107 = paddle._C_ops.multiply(scale_41, matmul_4)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_84 = multiply_107
        del multiply_107

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_33 = paddle._C_ops.matmul(assign_84, parameter_4, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_108 = paddle._C_ops.multiply(matmul_33, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_109 = paddle._C_ops.multiply(matmul_33, assign_19)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_30 = [multiply_65, multiply_78, multiply_100, multiply_108]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_59 = paddle._C_ops.add(multiply_65, multiply_78)
        del multiply_65, multiply_78

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_60 = paddle._C_ops.add(add_59, multiply_100)
        del add_59, multiply_100

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_61 = paddle._C_ops.add(add_60, multiply_108)
        del add_60, multiply_108

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_85 = add_61
        del add_61

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_34 = paddle._C_ops.matmul(assign_85, parameter_6, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_110 = paddle._C_ops.multiply(matmul_34, assign_16)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_35 = paddle._C_ops.matmul(multiply_94, parameter_6, False, False)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_42 = paddle._C_ops.scale(multiply_110, full_3, float("0"), True)
        del multiply_110

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_86 = matmul_35
        del matmul_35

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_111 = paddle._C_ops.multiply(scale_42, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_112 = paddle._C_ops.multiply(assign_86, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_113 = paddle._C_ops.multiply(assign_86, matmul_5)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_31 = [multiply_58, multiply_67, multiply_101, multiply_112]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_62 = paddle._C_ops.add(multiply_58, multiply_67)
        del multiply_58, multiply_67

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_63 = paddle._C_ops.add(add_62, multiply_101)
        del add_62, multiply_101

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_64 = paddle._C_ops.add(add_63, multiply_112)
        del add_63, multiply_112

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_36 = paddle._C_ops.matmul(add_64, parameter_4, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_32 = [multiply_60, multiply_80, multiply_109, multiply_113]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_65 = paddle._C_ops.add(multiply_60, multiply_80)
        del multiply_60, multiply_80

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_66 = paddle._C_ops.add(add_65, multiply_109)
        del add_65, multiply_109

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_67 = paddle._C_ops.add(add_66, multiply_113)
        del add_66, multiply_113

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_43 = paddle._C_ops.scale(add_67, full_3, float("0"), True)
        del add_67

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_87 = matmul_36
        del matmul_36

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_114 = paddle._C_ops.multiply(scale_43, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_115 = paddle._C_ops.multiply(assign_87, matmul_4)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_44 = paddle._C_ops.scale(multiply_115, full_3, float("0"), True)
        del multiply_115

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_116 = paddle._C_ops.multiply(scale_44, tanh_2)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_33 = [
            multiply_68,
            multiply_68,
            multiply_69,
            multiply_69,
            multiply_71,
            multiply_71,
            multiply_76,
            multiply_76,
            multiply_104,
            multiply_104,
            multiply_106,
            multiply_106,
            multiply_116,
            multiply_116,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_68 = paddle._C_ops.add(multiply_68, multiply_68)
        del multiply_68

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_69 = paddle._C_ops.add(add_68, multiply_69)
        del add_68

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_70 = paddle._C_ops.add(add_69, multiply_69)
        del add_69, multiply_69

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_71 = paddle._C_ops.add(add_70, multiply_71)
        del add_70

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_72 = paddle._C_ops.add(add_71, multiply_71)
        del add_71, multiply_71

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_73 = paddle._C_ops.add(add_72, multiply_76)
        del add_72

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_74 = paddle._C_ops.add(add_73, multiply_76)
        del add_73, multiply_76

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_75 = paddle._C_ops.add(add_74, multiply_104)
        del add_74

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_76 = paddle._C_ops.add(add_75, multiply_104)
        del add_75, multiply_104

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_77 = paddle._C_ops.add(add_76, multiply_106)
        del add_76

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_78 = paddle._C_ops.add(add_77, multiply_106)
        del add_77, multiply_106

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_79 = paddle._C_ops.add(add_78, multiply_116)
        del add_78

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_80 = paddle._C_ops.add(add_79, multiply_116)
        del add_79, multiply_116

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_117 = paddle._C_ops.multiply(add_80, subtract_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_88 = multiply_117
        del multiply_117

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_37 = paddle._C_ops.matmul(assign_88, parameter_4, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_34 = [
            multiply_53,
            multiply_53,
            multiply_54,
            multiply_54,
            multiply_56,
            multiply_56,
            multiply_85,
            multiply_85,
            multiply_97,
            multiply_97,
            multiply_99,
            multiply_99,
            multiply_114,
            multiply_114,
            matmul_37,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_81 = paddle._C_ops.add(multiply_53, multiply_53)
        del multiply_53

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_82 = paddle._C_ops.add(add_81, multiply_54)
        del add_81

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_83 = paddle._C_ops.add(add_82, multiply_54)
        del add_82, multiply_54

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_84 = paddle._C_ops.add(add_83, multiply_56)
        del add_83

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_85 = paddle._C_ops.add(add_84, multiply_56)
        del add_84, multiply_56

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_86 = paddle._C_ops.add(add_85, multiply_85)
        del add_85

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_87 = paddle._C_ops.add(add_86, multiply_85)
        del add_86, multiply_85

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_88 = paddle._C_ops.add(add_87, multiply_97)
        del add_87

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_89 = paddle._C_ops.add(add_88, multiply_97)
        del add_88, multiply_97

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_90 = paddle._C_ops.add(add_89, multiply_99)
        del add_89

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_91 = paddle._C_ops.add(add_90, multiply_99)
        del add_90, multiply_99

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_92 = paddle._C_ops.add(add_91, multiply_114)
        del add_91

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_93 = paddle._C_ops.add(add_92, multiply_114)
        del add_92, multiply_114

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_94 = paddle._C_ops.add(add_93, matmul_37)
        del add_93, matmul_37

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_118 = paddle._C_ops.multiply(add_94, subtract_1)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_89 = multiply_118
        del multiply_118

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_38 = paddle._C_ops.matmul(assign_89, parameter_6, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_35 = [
            multiply_48,
            multiply_48,
            multiply_49,
            multiply_49,
            multiply_51,
            multiply_51,
            multiply_89,
            multiply_89,
            multiply_91,
            multiply_91,
            multiply_93,
            multiply_93,
            multiply_111,
            multiply_111,
            matmul_38,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_95 = paddle._C_ops.add(multiply_48, multiply_48)
        del multiply_48

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_96 = paddle._C_ops.add(add_95, multiply_49)
        del add_95

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_97 = paddle._C_ops.add(add_96, multiply_49)
        del add_96, multiply_49

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_98 = paddle._C_ops.add(add_97, multiply_51)
        del add_97

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_99 = paddle._C_ops.add(add_98, multiply_51)
        del add_98, multiply_51

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_100 = paddle._C_ops.add(add_99, multiply_89)
        del add_99

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_101 = paddle._C_ops.add(add_100, multiply_89)
        del add_100, multiply_89

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_102 = paddle._C_ops.add(add_101, multiply_91)
        del add_101

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_103 = paddle._C_ops.add(add_102, multiply_91)
        del add_102, multiply_91

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_104 = paddle._C_ops.add(add_103, multiply_93)
        del add_103

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_105 = paddle._C_ops.add(add_104, multiply_93)
        del add_104, multiply_93

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_106 = paddle._C_ops.add(add_105, multiply_111)
        del add_105

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_107 = paddle._C_ops.add(add_106, multiply_111)
        del add_106, multiply_111

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_108 = paddle._C_ops.add(add_107, matmul_38)
        del add_107, matmul_38

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_119 = paddle._C_ops.multiply(add_108, subtract_2)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_90 = multiply_119
        del multiply_119

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 1x20xf32)
        matmul_39 = paddle._C_ops.matmul(assign_90, parameter_8, False, True)

        # pd_op.add: (100x1xf32) <- (xf32, 100x1xf32)
        add_109 = paddle._C_ops.add(parameter_0, matmul_39)
        del matmul_39, parameter_0

        # pd_op.subtract: (100x1xf32) <- (100x1xf32, 100x1xf32)
        subtract_3 = paddle._C_ops.subtract(add_109, data_4)
        del add_109, data_4

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [], float("2"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_91 = full_4

        # pd_op.assign: (xf32) <- (xf32)
        assign_92 = full_4

        # pd_op.assign: (xf32) <- (xf32)
        assign_93 = full_4

        # pd_op.assign: (xf32) <- (xf32)
        assign_94 = full_4

        # pd_op.elementwise_pow: (100x1xf32) <- (100x1xf32, xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(subtract_3, full_4)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [0, 1]

        # pd_op.sum: (xf32) <- (100x1xf32, 2xi64)
        sum_0 = paddle._C_ops.sum(
            elementwise_pow_0, full_int_array_1, paddle.float32, False
        )
        del elementwise_pow_0, full_int_array_1

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [], float("100"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_0, full_5)
        del full_5, sum_0

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_0, full_0, float("0"), True)

        # pd_op.matmul: (4x20xf32) <- (4x1xf32, 1x20xf32)
        matmul_40 = paddle._C_ops.matmul(data_3, parameter_8, False, False)

        # pd_op.add: (4x20xf32) <- (4x20xf32, 20xf32)
        add_110 = paddle._C_ops.add(matmul_40, parameter_7)
        del matmul_40, parameter_7

        # pd_op.tanh: (4x20xf32) <- (4x20xf32)
        tanh_3 = paddle._C_ops.tanh(add_110)
        del add_110

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_41 = paddle._C_ops.matmul(tanh_3, parameter_6, False, False)

        # pd_op.add: (4x20xf32) <- (4x20xf32, 20xf32)
        add_111 = paddle._C_ops.add(matmul_41, parameter_5)
        del matmul_41, parameter_5

        # pd_op.tanh: (4x20xf32) <- (4x20xf32)
        tanh_4 = paddle._C_ops.tanh(add_111)
        del add_111

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_42 = paddle._C_ops.matmul(tanh_4, parameter_4, False, False)

        # pd_op.add: (4x20xf32) <- (4x20xf32, 20xf32)
        add_112 = paddle._C_ops.add(matmul_42, parameter_3)
        del matmul_42, parameter_3

        # pd_op.tanh: (4x20xf32) <- (4x20xf32)
        tanh_5 = paddle._C_ops.tanh(add_112)
        del add_112

        # pd_op.matmul: (4x1xf32) <- (4x20xf32, 20x1xf32)
        matmul_43 = paddle._C_ops.matmul(tanh_5, parameter_2, False, False)

        # pd_op.add: (4x1xf32) <- (4x1xf32, 1xf32)
        add_113 = paddle._C_ops.add(matmul_43, parameter_1)
        del matmul_43, parameter_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (1x1xf32) <- (4x1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            add_113, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del add_113, full_int_array_2

        # pd_op.full: (4x1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [4, 1],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (4x1xf32) <- (4x1xf32)
        assign_95 = full_6

        # pd_op.assign: (4x1xf32) <- (4x1xf32)
        assign_96 = full_6

        # pd_op.assign: (4x1xf32) <- (4x1xf32)
        assign_97 = full_6

        # pd_op.matmul: (4x20xf32) <- (4x1xf32, 20x1xf32)
        matmul_44 = paddle._C_ops.matmul(assign_97, parameter_2, False, True)
        del parameter_2

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_120 = paddle._C_ops.multiply(tanh_5, tanh_5)

        # pd_op.subtract: (4x20xf32) <- (xf32, 4x20xf32)
        subtract_4 = paddle._C_ops.subtract(full_2, multiply_120)
        del multiply_120

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_98 = subtract_4

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_99 = subtract_4

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_121 = paddle._C_ops.multiply(matmul_44, subtract_4)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_100 = multiply_121
        del multiply_121

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_45 = paddle._C_ops.matmul(assign_100, parameter_4, False, True)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_122 = paddle._C_ops.multiply(tanh_4, tanh_4)

        # pd_op.subtract: (4x20xf32) <- (xf32, 4x20xf32)
        subtract_5 = paddle._C_ops.subtract(full_2, multiply_122)
        del multiply_122

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_101 = subtract_5

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_102 = subtract_5

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_123 = paddle._C_ops.multiply(matmul_45, subtract_5)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_103 = multiply_123
        del multiply_123

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_46 = paddle._C_ops.matmul(assign_103, parameter_6, False, True)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_124 = paddle._C_ops.multiply(tanh_3, tanh_3)

        # pd_op.subtract: (4x20xf32) <- (xf32, 4x20xf32)
        subtract_6 = paddle._C_ops.subtract(full_2, multiply_124)
        del full_2, multiply_124

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_104 = subtract_6

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_105 = subtract_6

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_125 = paddle._C_ops.multiply(matmul_46, subtract_6)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_106 = multiply_125
        del multiply_125

        # pd_op.matmul: (4x1xf32) <- (4x20xf32, 1x20xf32)
        matmul_47 = paddle._C_ops.matmul(assign_106, parameter_8, False, True)

        # pd_op.matmul: (1x20xf32) <- (4x1xf32, 4x20xf32)
        matmul_48 = paddle._C_ops.matmul(data_3, assign_106, True, False)
        del data_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.slice: (1x1xf32) <- (4x1xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            matmul_47, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del full_int_array_3, matmul_47

        # pd_op.matmul: (4x20xf32) <- (4x1xf32, 1x20xf32)
        matmul_49 = paddle._C_ops.matmul(full_6, parameter_8, False, False)

        # pd_op.matmul: (1x20xf32) <- (4x1xf32, 4x20xf32)
        matmul_50 = paddle._C_ops.matmul(full_6, assign_106, True, False)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_107 = matmul_49
        del matmul_49

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_108 = assign_107

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_126 = paddle._C_ops.multiply(assign_107, subtract_6)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_127 = paddle._C_ops.multiply(assign_107, matmul_46)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_51 = paddle._C_ops.matmul(multiply_126, parameter_6, False, False)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_45 = paddle._C_ops.scale(multiply_127, full_3, float("0"), True)
        del multiply_127

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_109 = matmul_51
        del matmul_51

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_128 = paddle._C_ops.multiply(scale_45, tanh_3)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_129 = paddle._C_ops.multiply(assign_109, subtract_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_130 = paddle._C_ops.multiply(assign_109, matmul_45)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_52 = paddle._C_ops.matmul(multiply_129, parameter_4, False, False)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_46 = paddle._C_ops.scale(multiply_130, full_3, float("0"), True)
        del multiply_130

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_110 = matmul_52
        del matmul_52

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_131 = paddle._C_ops.multiply(scale_46, tanh_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_132 = paddle._C_ops.multiply(assign_110, matmul_44)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_47 = paddle._C_ops.scale(multiply_132, full_3, float("0"), True)
        del multiply_132

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_133 = paddle._C_ops.multiply(scale_47, tanh_5)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_36 = [multiply_133, multiply_133]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_114 = paddle._C_ops.add(multiply_133, multiply_133)
        del multiply_133

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_134 = paddle._C_ops.multiply(add_114, subtract_4)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_111 = multiply_134
        del multiply_134

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_53 = paddle._C_ops.matmul(assign_111, parameter_4, False, True)

        # builtin.combine: ([4x20xf32, 4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32, 4x20xf32)
        combine_37 = [multiply_131, multiply_131, matmul_53]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_115 = paddle._C_ops.add(multiply_131, multiply_131)
        del multiply_131

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_116 = paddle._C_ops.add(add_115, matmul_53)
        del add_115, matmul_53

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_135 = paddle._C_ops.multiply(add_116, subtract_5)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_112 = multiply_135
        del multiply_135

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_54 = paddle._C_ops.matmul(assign_112, parameter_6, False, True)

        # builtin.combine: ([4x20xf32, 4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32, 4x20xf32)
        combine_38 = [multiply_128, multiply_128, matmul_54]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_117 = paddle._C_ops.add(multiply_128, multiply_128)
        del multiply_128

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_118 = paddle._C_ops.add(add_117, matmul_54)
        del add_117, matmul_54

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_136 = paddle._C_ops.multiply(add_118, subtract_6)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_113 = multiply_136
        del multiply_136

        # pd_op.matmul: (4x1xf32) <- (4x20xf32, 1x20xf32)
        matmul_55 = paddle._C_ops.matmul(assign_113, parameter_8, False, True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (1x1xf32) <- (4x1xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            matmul_55, [0], full_int_array_4, full_int_array_5, [1], []
        )
        del full_int_array_4, matmul_55

        # pd_op.matmul: (1x20xf32) <- (4x1xf32, 4x20xf32)
        matmul_56 = paddle._C_ops.matmul(full_6, assign_113, True, False)
        del full_6

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_137 = paddle._C_ops.multiply(assign_107, add_118)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_48 = paddle._C_ops.scale(multiply_126, full_0, float("0"), True)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_114 = scale_48

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_115 = scale_48

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_49 = paddle._C_ops.scale(multiply_137, full_3, float("0"), True)
        del multiply_137

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_138 = paddle._C_ops.multiply(scale_49, tanh_3)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_139 = paddle._C_ops.multiply(scale_48, tanh_3)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_140 = paddle._C_ops.multiply(scale_48, scale_45)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_57 = paddle._C_ops.matmul(scale_48, parameter_6, False, False)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_39 = [multiply_139, multiply_139]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_119 = paddle._C_ops.add(multiply_139, multiply_139)
        del multiply_139

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_50 = paddle._C_ops.scale(add_119, full_3, float("0"), True)
        del add_119

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_116 = matmul_57
        del matmul_57

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_141 = paddle._C_ops.multiply(scale_50, assign_107)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_142 = paddle._C_ops.multiply(assign_116, subtract_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_143 = paddle._C_ops.multiply(assign_116, add_116)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_51 = paddle._C_ops.scale(multiply_142, full_0, float("0"), True)
        del multiply_142

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_117 = scale_51

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_118 = scale_51

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_52 = paddle._C_ops.scale(multiply_143, full_3, float("0"), True)
        del multiply_143

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_144 = paddle._C_ops.multiply(scale_52, tanh_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_145 = paddle._C_ops.multiply(scale_51, tanh_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_146 = paddle._C_ops.multiply(scale_51, scale_46)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_58 = paddle._C_ops.matmul(scale_51, parameter_4, False, False)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_40 = [multiply_145, multiply_145]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_120 = paddle._C_ops.add(multiply_145, multiply_145)
        del multiply_145

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_53 = paddle._C_ops.scale(add_120, full_3, float("0"), True)
        del add_120

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_119 = matmul_58
        del matmul_58

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_147 = paddle._C_ops.multiply(scale_53, matmul_45)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_148 = paddle._C_ops.multiply(scale_53, assign_109)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_149 = paddle._C_ops.multiply(assign_119, subtract_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_150 = paddle._C_ops.multiply(assign_119, add_114)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_54 = paddle._C_ops.scale(multiply_149, full_0, float("0"), True)
        del multiply_149

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_120 = scale_54

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_55 = paddle._C_ops.scale(multiply_150, full_3, float("0"), True)
        del multiply_150

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_151 = paddle._C_ops.multiply(scale_55, tanh_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_152 = paddle._C_ops.multiply(scale_54, tanh_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_153 = paddle._C_ops.multiply(scale_54, scale_47)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_41 = [multiply_152, multiply_152]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_121 = paddle._C_ops.add(multiply_152, multiply_152)
        del multiply_152

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_56 = paddle._C_ops.scale(add_121, full_3, float("0"), True)
        del add_121

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_154 = paddle._C_ops.multiply(scale_56, matmul_44)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_121 = multiply_154
        del multiply_154

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_59 = paddle._C_ops.matmul(assign_121, parameter_4, False, True)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_155 = paddle._C_ops.multiply(matmul_59, subtract_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_156 = paddle._C_ops.multiply(matmul_59, assign_109)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_42 = [multiply_147, multiply_155]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_122 = paddle._C_ops.add(multiply_147, multiply_155)
        del multiply_147, multiply_155

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_122 = add_122
        del add_122

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_60 = paddle._C_ops.matmul(assign_122, parameter_6, False, True)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_157 = paddle._C_ops.multiply(matmul_60, assign_107)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_61 = paddle._C_ops.matmul(multiply_141, parameter_6, False, False)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_57 = paddle._C_ops.scale(multiply_157, full_3, float("0"), True)
        del multiply_157

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_123 = matmul_61
        del matmul_61

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_158 = paddle._C_ops.multiply(scale_57, tanh_3)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_159 = paddle._C_ops.multiply(assign_123, subtract_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_160 = paddle._C_ops.multiply(assign_123, matmul_45)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_43 = [multiply_148, multiply_159]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_123 = paddle._C_ops.add(multiply_148, multiply_159)
        del multiply_148, multiply_159

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_62 = paddle._C_ops.matmul(add_123, parameter_4, False, False)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_44 = [multiply_156, multiply_160]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_124 = paddle._C_ops.add(multiply_156, multiply_160)
        del multiply_156, multiply_160

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_58 = paddle._C_ops.scale(add_124, full_3, float("0"), True)
        del add_124

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_124 = matmul_62
        del matmul_62

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_161 = paddle._C_ops.multiply(scale_58, tanh_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_162 = paddle._C_ops.multiply(assign_124, matmul_44)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_59 = paddle._C_ops.scale(multiply_162, full_3, float("0"), True)
        del full_3, multiply_162

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_163 = paddle._C_ops.multiply(scale_59, tanh_5)

        # builtin.combine: ([4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32)
        combine_45 = [
            multiply_151,
            multiply_151,
            multiply_153,
            multiply_153,
            multiply_163,
            multiply_163,
        ]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_125 = paddle._C_ops.add(multiply_151, multiply_151)
        del multiply_151

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_126 = paddle._C_ops.add(add_125, multiply_153)
        del add_125

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_127 = paddle._C_ops.add(add_126, multiply_153)
        del add_126, multiply_153

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_128 = paddle._C_ops.add(add_127, multiply_163)
        del add_127

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_129 = paddle._C_ops.add(add_128, multiply_163)
        del add_128, multiply_163

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_164 = paddle._C_ops.multiply(add_129, subtract_4)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_125 = multiply_164
        del multiply_164

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_63 = paddle._C_ops.matmul(assign_125, parameter_4, False, True)
        del parameter_4

        # builtin.combine: ([4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32)
        combine_46 = [
            multiply_144,
            multiply_144,
            multiply_146,
            multiply_146,
            multiply_161,
            multiply_161,
            matmul_63,
        ]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_130 = paddle._C_ops.add(multiply_144, multiply_144)
        del multiply_144

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_131 = paddle._C_ops.add(add_130, multiply_146)
        del add_130

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_132 = paddle._C_ops.add(add_131, multiply_146)
        del add_131, multiply_146

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_133 = paddle._C_ops.add(add_132, multiply_161)
        del add_132

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_134 = paddle._C_ops.add(add_133, multiply_161)
        del add_133, multiply_161

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_135 = paddle._C_ops.add(add_134, matmul_63)
        del add_134, matmul_63

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_165 = paddle._C_ops.multiply(add_135, subtract_5)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_126 = multiply_165
        del multiply_165

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_64 = paddle._C_ops.matmul(assign_126, parameter_6, False, True)
        del parameter_6

        # builtin.combine: ([4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32)
        combine_47 = [
            multiply_138,
            multiply_138,
            multiply_140,
            multiply_140,
            multiply_158,
            multiply_158,
            matmul_64,
        ]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_136 = paddle._C_ops.add(multiply_138, multiply_138)
        del multiply_138

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_137 = paddle._C_ops.add(add_136, multiply_140)
        del add_136

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_138 = paddle._C_ops.add(add_137, multiply_140)
        del add_137, multiply_140

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_139 = paddle._C_ops.add(add_138, multiply_158)
        del add_138

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_140 = paddle._C_ops.add(add_139, multiply_158)
        del add_139, multiply_158

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_141 = paddle._C_ops.add(add_140, matmul_64)
        del add_140, matmul_64

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_166 = paddle._C_ops.multiply(add_141, subtract_6)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_127 = multiply_166
        del multiply_166

        # pd_op.matmul: (4x1xf32) <- (4x20xf32, 1x20xf32)
        matmul_65 = paddle._C_ops.matmul(assign_127, parameter_8, False, True)
        del parameter_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2147483647]

        # pd_op.slice: (1x1xf32) <- (4x1xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            matmul_65, [0], full_int_array_5, full_int_array_6, [1], []
        )
        del full_int_array_5, full_int_array_6, matmul_65

        # pd_op.subtract: (4x1xf32) <- (1x1xf32, 4x1xf32)
        subtract_7 = paddle._C_ops.subtract(slice_0, data_5)
        del data_5

        # pd_op.elementwise_pow: (4x1xf32) <- (4x1xf32, xf32)
        elementwise_pow_1 = paddle._C_ops.elementwise_pow(subtract_7, full_4)

        # pd_op.sum: (xf32) <- (4x1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(elementwise_pow_1, full_int_array_0, None, False)
        del elementwise_pow_1

        # pd_op.subtract: (4x1xf32) <- (1x1xf32, 4x1xf32)
        subtract_8 = paddle._C_ops.subtract(slice_1, data_6)
        del data_6

        # pd_op.elementwise_pow: (4x1xf32) <- (4x1xf32, xf32)
        elementwise_pow_2 = paddle._C_ops.elementwise_pow(subtract_8, full_4)

        # pd_op.sum: (xf32) <- (4x1xf32, 0xi64)
        sum_2 = paddle._C_ops.sum(elementwise_pow_2, full_int_array_0, None, False)
        del elementwise_pow_2

        # pd_op.subtract: (4x1xf32) <- (1x1xf32, 4x1xf32)
        subtract_9 = paddle._C_ops.subtract(slice_2, data_7)
        del data_7

        # pd_op.elementwise_pow: (4x1xf32) <- (4x1xf32, xf32)
        elementwise_pow_3 = paddle._C_ops.elementwise_pow(subtract_9, full_4)

        # pd_op.sum: (xf32) <- (4x1xf32, 0xi64)
        sum_3 = paddle._C_ops.sum(elementwise_pow_3, full_int_array_0, None, False)
        del elementwise_pow_3

        # pd_op.subtract: (4x1xf32) <- (1x1xf32, 4x1xf32)
        subtract_10 = paddle._C_ops.subtract(slice_3, data_8)
        del data_8

        # pd_op.elementwise_pow: (4x1xf32) <- (4x1xf32, xf32)
        elementwise_pow_4 = paddle._C_ops.elementwise_pow(subtract_10, full_4)

        # pd_op.sum: (xf32) <- (4x1xf32, 0xi64)
        sum_4 = paddle._C_ops.sum(elementwise_pow_4, full_int_array_0, None, False)
        del elementwise_pow_4, full_int_array_0

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_60 = paddle._C_ops.scale(sum_1, full_0, float("0"), True)
        del full_0

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_142 = paddle._C_ops.add(scale_60, sum_2)
        del scale_60

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_143 = paddle._C_ops.add(add_142, sum_3)
        del add_142

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_0 = paddle._C_ops.add(add_143, sum_4)
        del (
            add_108,
            add_114,
            add_116,
            add_118,
            add_123,
            add_129,
            add_135,
            add_14,
            add_141,
            add_143,
            add_20,
            add_26,
            add_32,
            add_33,
            add_35,
            add_41,
            add_45,
            add_5,
            add_64,
            add_7,
            add_80,
            add_9,
            add_94,
            assign_0,
            assign_1,
            assign_10,
            assign_100,
            assign_101,
            assign_102,
            assign_103,
            assign_104,
            assign_105,
            assign_106,
            assign_107,
            assign_108,
            assign_109,
            assign_11,
            assign_110,
            assign_111,
            assign_112,
            assign_113,
            assign_114,
            assign_115,
            assign_116,
            assign_117,
            assign_118,
            assign_119,
            assign_12,
            assign_120,
            assign_121,
            assign_122,
            assign_123,
            assign_124,
            assign_125,
            assign_126,
            assign_127,
            assign_13,
            assign_14,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
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
            divide_0,
            full_4,
            matmul_19,
            matmul_20,
            matmul_28,
            matmul_33,
            matmul_34,
            matmul_4,
            matmul_44,
            matmul_45,
            matmul_46,
            matmul_5,
            matmul_59,
            matmul_6,
            matmul_60,
            multiply_126,
            multiply_129,
            multiply_141,
            multiply_21,
            multiply_6,
            multiply_9,
            multiply_94,
            scale_1,
            scale_10,
            scale_11,
            scale_12,
            scale_13,
            scale_14,
            scale_15,
            scale_16,
            scale_17,
            scale_18,
            scale_19,
            scale_2,
            scale_20,
            scale_22,
            scale_23,
            scale_26,
            scale_3,
            scale_30,
            scale_33,
            scale_34,
            scale_35,
            scale_36,
            scale_37,
            scale_38,
            scale_39,
            scale_4,
            scale_40,
            scale_41,
            scale_42,
            scale_43,
            scale_44,
            scale_45,
            scale_46,
            scale_47,
            scale_48,
            scale_49,
            scale_5,
            scale_50,
            scale_51,
            scale_52,
            scale_53,
            scale_54,
            scale_55,
            scale_56,
            scale_57,
            scale_58,
            scale_59,
            scale_6,
            scale_7,
            scale_8,
            scale_9,
            slice_0,
            slice_1,
            slice_2,
            slice_3,
            subtract_0,
            subtract_1,
            subtract_10,
            subtract_2,
            subtract_3,
            subtract_4,
            subtract_5,
            subtract_6,
            subtract_7,
            subtract_8,
            subtract_9,
            sum_1,
            sum_2,
            sum_3,
            sum_4,
            tanh_0,
            tanh_1,
            tanh_2,
            tanh_3,
            tanh_4,
            tanh_5,
        )

        return add_0, scale_0
