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
        matmul_13 = paddle._C_ops.matmul(data_1, parameter_8, False, False)

        # pd_op.add: (100x20xf32) <- (100x20xf32, 20xf32)
        add_23 = paddle._C_ops.add(matmul_13, parameter_7)
        del matmul_13

        # pd_op.tanh: (100x20xf32) <- (100x20xf32)
        tanh_0 = paddle._C_ops.tanh(add_23)
        del add_23

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_14 = paddle._C_ops.matmul(tanh_0, parameter_6, False, False)

        # pd_op.add: (100x20xf32) <- (100x20xf32, 20xf32)
        add_24 = paddle._C_ops.add(matmul_14, parameter_5)
        del matmul_14

        # pd_op.tanh: (100x20xf32) <- (100x20xf32)
        tanh_1 = paddle._C_ops.tanh(add_24)
        del add_24

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_15 = paddle._C_ops.matmul(tanh_1, parameter_4, False, False)

        # pd_op.add: (100x20xf32) <- (100x20xf32, 20xf32)
        add_25 = paddle._C_ops.add(matmul_15, parameter_3)
        del matmul_15

        # pd_op.tanh: (100x20xf32) <- (100x20xf32)
        tanh_2 = paddle._C_ops.tanh(add_25)
        del add_25

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 20x1xf32)
        matmul_16 = paddle._C_ops.matmul(tanh_2, parameter_2, False, False)

        # pd_op.add: (100x1xf32) <- (100x1xf32, 1xf32)
        add_26 = paddle._C_ops.add(matmul_16, parameter_1)
        del matmul_16

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (100x1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [100, 1],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (100x1xf32) <- (100x1xf32)
        assign_30 = full_2

        # pd_op.assign: (100x1xf32) <- (100x1xf32)
        assign_12 = full_2

        # pd_op.assign: (100x1xf32) <- (100x1xf32)
        assign_3 = full_2

        # pd_op.assign: (100x1xf32) <- (100x1xf32)
        assign_0 = full_2

        # pd_op.matmul: (100x20xf32) <- (100x1xf32, 20x1xf32)
        matmul_0 = paddle._C_ops.matmul(assign_0, parameter_2, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_7 = paddle._C_ops.multiply(tanh_2, tanh_2)

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.subtract: (100x20xf32) <- (xf32, 100x20xf32)
        subtract_0 = paddle._C_ops.subtract(full_3, multiply_7)
        del multiply_7

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_82 = subtract_0

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_25 = subtract_0

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_7 = subtract_0

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_8 = paddle._C_ops.multiply(matmul_0, subtract_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_1 = multiply_8
        del multiply_8

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_1 = paddle._C_ops.matmul(assign_1, parameter_4, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_9 = paddle._C_ops.multiply(tanh_1, tanh_1)

        # pd_op.subtract: (100x20xf32) <- (xf32, 100x20xf32)
        subtract_1 = paddle._C_ops.subtract(full_3, multiply_9)
        del multiply_9

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_84 = subtract_1

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_27 = subtract_1

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_9 = subtract_1

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_10 = paddle._C_ops.multiply(matmul_1, subtract_1)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_2 = multiply_10
        del multiply_10

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_2 = paddle._C_ops.matmul(assign_2, parameter_6, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_11 = paddle._C_ops.multiply(tanh_0, tanh_0)

        # pd_op.subtract: (100x20xf32) <- (xf32, 100x20xf32)
        subtract_2 = paddle._C_ops.subtract(full_3, multiply_11)
        del multiply_11

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_86 = subtract_2

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_29 = subtract_2

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_11 = subtract_2

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_12 = paddle._C_ops.multiply(matmul_2, subtract_2)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_125 = multiply_12
        del multiply_12

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 1x20xf32)
        matmul_17 = paddle._C_ops.matmul(assign_125, parameter_8, False, True)

        # pd_op.matmul: (1x20xf32) <- (100x1xf32, 100x20xf32)
        matmul_18 = paddle._C_ops.matmul(data_1, assign_125, True, False)
        del data_1

        # pd_op.matmul: (100x20xf32) <- (100x1xf32, 1x20xf32)
        matmul_19 = paddle._C_ops.matmul(full_2, parameter_8, False, False)

        # pd_op.matmul: (1x20xf32) <- (100x1xf32, 100x20xf32)
        matmul_20 = paddle._C_ops.matmul(full_2, assign_125, True, False)
        del assign_125

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_4 = matmul_19
        del matmul_19

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_31 = assign_4

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_13 = assign_4

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_0 = paddle._C_ops.multiply(assign_4, subtract_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_13 = paddle._C_ops.multiply(assign_4, matmul_2)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_21 = paddle._C_ops.matmul(multiply_0, parameter_6, False, False)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(multiply_13, full_4, float("0"), True)
        del multiply_13

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_5 = matmul_21
        del matmul_21

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_14 = paddle._C_ops.multiply(scale_0, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_1 = paddle._C_ops.multiply(assign_5, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_15 = paddle._C_ops.multiply(assign_5, matmul_1)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_22 = paddle._C_ops.matmul(multiply_1, parameter_4, False, False)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_15, full_4, float("0"), True)
        del multiply_15

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_6 = matmul_22
        del matmul_22

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_16 = paddle._C_ops.multiply(scale_1, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_17 = paddle._C_ops.multiply(assign_6, matmul_0)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(multiply_17, full_4, float("0"), True)
        del multiply_17

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_18 = paddle._C_ops.multiply(scale_2, tanh_2)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_0 = [multiply_18, multiply_18]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_0 = paddle._C_ops.add(multiply_18, multiply_18)
        del multiply_18

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_19 = paddle._C_ops.multiply(add_0, subtract_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_8 = multiply_19
        del multiply_19

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_23 = paddle._C_ops.matmul(assign_8, parameter_4, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32)
        combine_1 = [multiply_16, multiply_16, matmul_23]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_27 = paddle._C_ops.add(multiply_16, multiply_16)
        del multiply_16

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_1 = paddle._C_ops.add(add_27, matmul_23)
        del add_27, matmul_23

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_20 = paddle._C_ops.multiply(add_1, subtract_1)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_10 = multiply_20
        del multiply_20

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_24 = paddle._C_ops.matmul(assign_10, parameter_6, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32)
        combine_2 = [multiply_14, multiply_14, matmul_24]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_28 = paddle._C_ops.add(multiply_14, multiply_14)
        del multiply_14

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_2 = paddle._C_ops.add(add_28, matmul_24)
        del add_28, matmul_24

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_21 = paddle._C_ops.multiply(add_2, subtract_2)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_126 = multiply_21
        del multiply_21

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 1x20xf32)
        matmul_25 = paddle._C_ops.matmul(assign_126, parameter_8, False, True)

        # pd_op.matmul: (1x20xf32) <- (100x1xf32, 100x20xf32)
        matmul_26 = paddle._C_ops.matmul(full_2, assign_126, True, False)
        del assign_126

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_22 = paddle._C_ops.multiply(assign_4, add_2)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(multiply_0, full_1, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_38 = scale_3

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_37 = scale_3

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_36 = scale_3

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_35 = scale_3

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_34 = scale_3

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_33 = scale_3

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_32 = scale_3

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_15 = scale_3

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_14 = scale_3

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(multiply_22, full_4, float("0"), True)
        del multiply_22

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_23 = paddle._C_ops.multiply(scale_4, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_24 = paddle._C_ops.multiply(scale_3, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_25 = paddle._C_ops.multiply(scale_3, scale_0)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_27 = paddle._C_ops.matmul(scale_3, parameter_6, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_3 = [multiply_24, multiply_24]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_29 = paddle._C_ops.add(multiply_24, multiply_24)
        del multiply_24

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(add_29, full_4, float("0"), True)
        del add_29

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_40 = scale_5

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_39 = scale_5

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_16 = matmul_27
        del matmul_27

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_41 = assign_16

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_2 = paddle._C_ops.multiply(scale_5, assign_4)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_42 = multiply_2

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_26 = paddle._C_ops.multiply(assign_16, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_27 = paddle._C_ops.multiply(assign_16, add_1)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(multiply_26, full_1, float("0"), True)
        del multiply_26

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_49 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_48 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_47 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_46 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_45 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_44 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_43 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_18 = scale_6

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_17 = scale_6

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(multiply_27, full_4, float("0"), True)
        del multiply_27

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_28 = paddle._C_ops.multiply(scale_7, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_29 = paddle._C_ops.multiply(scale_6, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_30 = paddle._C_ops.multiply(scale_6, scale_1)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_28 = paddle._C_ops.matmul(scale_6, parameter_4, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_4 = [multiply_29, multiply_29]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_30 = paddle._C_ops.add(multiply_29, multiply_29)
        del multiply_29

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(add_30, full_4, float("0"), True)
        del add_30

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_51 = scale_8

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_19 = matmul_28
        del matmul_28

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_52 = assign_19

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_31 = paddle._C_ops.multiply(scale_8, matmul_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_32 = paddle._C_ops.multiply(scale_8, assign_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_33 = paddle._C_ops.multiply(assign_19, subtract_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_34 = paddle._C_ops.multiply(assign_19, add_0)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(multiply_33, full_1, float("0"), True)
        del multiply_33

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_59 = scale_9

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_58 = scale_9

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_57 = scale_9

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_56 = scale_9

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_55 = scale_9

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_54 = scale_9

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_20 = scale_9

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(multiply_34, full_4, float("0"), True)
        del multiply_34

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_35 = paddle._C_ops.multiply(scale_10, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_36 = paddle._C_ops.multiply(scale_9, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_37 = paddle._C_ops.multiply(scale_9, scale_2)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_5 = [multiply_36, multiply_36]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_31 = paddle._C_ops.add(multiply_36, multiply_36)
        del multiply_36

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(add_31, full_4, float("0"), True)
        del add_31

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_62 = scale_11

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_61 = scale_11

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_38 = paddle._C_ops.multiply(scale_11, matmul_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_21 = multiply_38
        del multiply_38

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_63 = assign_21

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_3 = paddle._C_ops.matmul(assign_21, parameter_4, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_39 = paddle._C_ops.multiply(matmul_3, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_40 = paddle._C_ops.multiply(matmul_3, assign_5)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_6 = [multiply_31, multiply_39]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_32 = paddle._C_ops.add(multiply_31, multiply_39)
        del multiply_31, multiply_39

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_22 = add_32
        del add_32

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_4 = paddle._C_ops.matmul(assign_22, parameter_6, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_41 = paddle._C_ops.multiply(matmul_4, assign_4)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_29 = paddle._C_ops.matmul(multiply_2, parameter_6, False, False)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(multiply_41, full_4, float("0"), True)
        del multiply_41

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_23 = matmul_29
        del matmul_29

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_42 = paddle._C_ops.multiply(scale_12, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_43 = paddle._C_ops.multiply(assign_23, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_44 = paddle._C_ops.multiply(assign_23, matmul_1)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_7 = [multiply_32, multiply_43]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_3 = paddle._C_ops.add(multiply_32, multiply_43)
        del multiply_32, multiply_43

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_30 = paddle._C_ops.matmul(add_3, parameter_4, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_8 = [multiply_40, multiply_44]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_33 = paddle._C_ops.add(multiply_40, multiply_44)
        del multiply_40, multiply_44

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(add_33, full_4, float("0"), True)
        del add_33

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_24 = matmul_30
        del matmul_30

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_45 = paddle._C_ops.multiply(scale_13, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_46 = paddle._C_ops.multiply(assign_24, matmul_0)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(multiply_46, full_4, float("0"), True)
        del multiply_46

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_47 = paddle._C_ops.multiply(scale_14, tanh_2)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_9 = [
            multiply_35,
            multiply_35,
            multiply_37,
            multiply_37,
            multiply_47,
            multiply_47,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_34 = paddle._C_ops.add(multiply_35, multiply_35)
        del multiply_35

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_35 = paddle._C_ops.add(add_34, multiply_37)
        del add_34

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_36 = paddle._C_ops.add(add_35, multiply_37)
        del add_35

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_37 = paddle._C_ops.add(add_36, multiply_47)
        del add_36

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_4 = paddle._C_ops.add(add_37, multiply_47)
        del add_37, multiply_47

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_48 = paddle._C_ops.multiply(add_4, subtract_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_26 = multiply_48
        del multiply_48

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_31 = paddle._C_ops.matmul(assign_26, parameter_4, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_10 = [
            multiply_28,
            multiply_28,
            multiply_30,
            multiply_30,
            multiply_45,
            multiply_45,
            matmul_31,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_38 = paddle._C_ops.add(multiply_28, multiply_28)
        del multiply_28

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_39 = paddle._C_ops.add(add_38, multiply_30)
        del add_38

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_40 = paddle._C_ops.add(add_39, multiply_30)
        del add_39

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_41 = paddle._C_ops.add(add_40, multiply_45)
        del add_40

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_42 = paddle._C_ops.add(add_41, multiply_45)
        del add_41, multiply_45

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_5 = paddle._C_ops.add(add_42, matmul_31)
        del add_42, matmul_31

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_49 = paddle._C_ops.multiply(add_5, subtract_1)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_28 = multiply_49
        del multiply_49

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_32 = paddle._C_ops.matmul(assign_28, parameter_6, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_11 = [
            multiply_23,
            multiply_23,
            multiply_25,
            multiply_25,
            multiply_42,
            multiply_42,
            matmul_32,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_43 = paddle._C_ops.add(multiply_23, multiply_23)
        del multiply_23

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_44 = paddle._C_ops.add(add_43, multiply_25)
        del add_43

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_45 = paddle._C_ops.add(add_44, multiply_25)
        del add_44

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_46 = paddle._C_ops.add(add_45, multiply_42)
        del add_45

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_47 = paddle._C_ops.add(add_46, multiply_42)
        del add_46, multiply_42

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_6 = paddle._C_ops.add(add_47, matmul_32)
        del add_47, matmul_32

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_50 = paddle._C_ops.multiply(add_6, subtract_2)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_127 = multiply_50
        del multiply_50

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 1x20xf32)
        matmul_33 = paddle._C_ops.matmul(assign_127, parameter_8, False, True)

        # pd_op.matmul: (1x20xf32) <- (100x1xf32, 100x20xf32)
        matmul_34 = paddle._C_ops.matmul(full_2, assign_127, True, False)
        del assign_127, full_2

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_51 = paddle._C_ops.multiply(assign_4, add_6)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(multiply_51, full_4, float("0"), True)
        del multiply_51

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_52 = paddle._C_ops.multiply(scale_15, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_53 = paddle._C_ops.multiply(scale_3, scale_4)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_54 = paddle._C_ops.multiply(scale_3, scale_3)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_55 = paddle._C_ops.multiply(scale_3, scale_12)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_56 = paddle._C_ops.multiply(assign_16, add_5)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(multiply_56, full_4, float("0"), True)
        del multiply_56

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_57 = paddle._C_ops.multiply(scale_16, tanh_1)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(assign_23, full_1, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_50 = scale_17

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_58 = paddle._C_ops.multiply(scale_6, scale_7)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_59 = paddle._C_ops.multiply(scale_6, scale_6)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_60 = paddle._C_ops.multiply(scale_6, scale_13)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_61 = paddle._C_ops.multiply(scale_17, matmul_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_62 = paddle._C_ops.multiply(scale_17, scale_8)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_63 = paddle._C_ops.multiply(scale_17, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_64 = paddle._C_ops.multiply(scale_17, matmul_3)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_65 = paddle._C_ops.multiply(scale_8, add_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_66 = paddle._C_ops.multiply(scale_8, assign_16)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(scale_8, full_1, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_53 = scale_18

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_67 = paddle._C_ops.multiply(assign_19, add_4)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(multiply_67, full_4, float("0"), True)
        del multiply_67

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_68 = paddle._C_ops.multiply(scale_18, assign_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_69 = paddle._C_ops.multiply(scale_18, matmul_3)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_70 = paddle._C_ops.multiply(scale_18, matmul_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_71 = paddle._C_ops.multiply(scale_18, assign_23)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_72 = paddle._C_ops.multiply(scale_19, tanh_2)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_12 = [multiply_63, multiply_68]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_7 = paddle._C_ops.add(multiply_63, multiply_68)
        del multiply_63, multiply_68

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_35 = paddle._C_ops.matmul(add_7, parameter_4, False, False)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_73 = paddle._C_ops.multiply(scale_9, scale_10)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_74 = paddle._C_ops.multiply(scale_9, scale_9)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_75 = paddle._C_ops.multiply(scale_9, scale_14)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_60 = matmul_35
        del matmul_35

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_76 = paddle._C_ops.multiply(assign_60, matmul_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_77 = paddle._C_ops.multiply(scale_11, add_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_78 = paddle._C_ops.multiply(scale_11, assign_19)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_52 = paddle._C_ops.scale(multiply_76, full_4, float("0"), True)
        del multiply_76

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(scale_52, full_1, float("0"), True)
        del scale_52

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_64 = scale_20

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(matmul_3, full_1, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_65 = scale_21

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_79 = paddle._C_ops.multiply(scale_20, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_80 = paddle._C_ops.multiply(scale_20, scale_9)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_13 = [multiply_37, multiply_79]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_48 = paddle._C_ops.add(multiply_37, multiply_79)
        del multiply_37, multiply_79

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_53 = paddle._C_ops.scale(add_48, full_1, float("0"), True)
        del add_48

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_81 = paddle._C_ops.multiply(scale_21, assign_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_82 = paddle._C_ops.multiply(scale_21, scale_8)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_83 = paddle._C_ops.multiply(scale_21, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_84 = paddle._C_ops.multiply(scale_21, assign_23)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_14 = [scale_53, scale_53]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_8 = paddle._C_ops.add(scale_53, scale_53)
        del scale_53

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_85 = paddle._C_ops.multiply(add_8, subtract_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_86 = paddle._C_ops.multiply(add_8, assign_19)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_15 = [multiply_61, multiply_81]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_49 = paddle._C_ops.add(multiply_61, multiply_81)
        del multiply_61, multiply_81

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_54 = paddle._C_ops.scale(add_49, full_4, float("0"), True)
        del add_49

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_16 = [multiply_70, multiply_83]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_50 = paddle._C_ops.add(multiply_70, multiply_83)
        del multiply_70, multiply_83

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_66 = add_50
        del add_50

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_17 = [multiply_77, multiply_85]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_51 = paddle._C_ops.add(multiply_77, multiply_85)
        del multiply_77, multiply_85

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_67 = add_51
        del add_51

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(scale_54, full_1, float("0"), True)
        del scale_54

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_68 = scale_22

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_5 = paddle._C_ops.matmul(assign_66, parameter_6, False, True)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_36 = paddle._C_ops.matmul(assign_67, parameter_4, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_87 = paddle._C_ops.multiply(matmul_5, assign_4)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_55 = paddle._C_ops.scale(matmul_36, full_1, float("0"), True)
        del matmul_36

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_88 = paddle._C_ops.multiply(scale_22, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_89 = paddle._C_ops.multiply(scale_22, scale_6)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_56 = paddle._C_ops.scale(multiply_87, full_4, float("0"), True)
        del multiply_87

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_18 = [multiply_30, multiply_88]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_52 = paddle._C_ops.add(multiply_30, multiply_88)
        del multiply_30, multiply_88

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_57 = paddle._C_ops.scale(add_52, full_1, float("0"), True)
        del add_52

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(scale_56, full_1, float("0"), True)
        del scale_56

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_69 = scale_23

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32)
        combine_19 = [scale_55, scale_57, scale_57]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_53 = paddle._C_ops.add(scale_55, scale_57)
        del scale_55

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_9 = paddle._C_ops.add(add_53, scale_57)
        del add_53, scale_57

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_90 = paddle._C_ops.multiply(add_9, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_91 = paddle._C_ops.multiply(add_9, assign_16)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_20 = [multiply_65, multiply_90]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_54 = paddle._C_ops.add(multiply_65, multiply_90)
        del multiply_65, multiply_90

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_70 = add_54
        del add_54

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_92 = paddle._C_ops.multiply(scale_23, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_93 = paddle._C_ops.multiply(scale_23, scale_3)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_37 = paddle._C_ops.matmul(assign_70, parameter_6, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_21 = [multiply_25, multiply_92]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_55 = paddle._C_ops.add(multiply_25, multiply_92)
        del multiply_25, multiply_92

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_58 = paddle._C_ops.scale(add_55, full_1, float("0"), True)
        del add_55

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_59 = paddle._C_ops.scale(matmul_37, full_1, float("0"), True)
        del matmul_37

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32)
        combine_22 = [scale_58, scale_58, scale_59]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_56 = paddle._C_ops.add(scale_58, scale_58)
        del scale_58

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_10 = paddle._C_ops.add(add_56, scale_59)
        del add_56, scale_59

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_94 = paddle._C_ops.multiply(add_10, assign_4)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(multiply_2, full_1, float("0"), True)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_72 = scale_24

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_71 = scale_24

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(multiply_94, full_4, float("0"), True)
        del multiply_94

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_95 = paddle._C_ops.multiply(scale_25, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_96 = paddle._C_ops.multiply(scale_24, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_97 = paddle._C_ops.multiply(scale_24, scale_0)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_38 = paddle._C_ops.matmul(scale_24, parameter_6, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_23 = [multiply_54, multiply_54, multiply_96, multiply_96]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_57 = paddle._C_ops.add(multiply_54, multiply_54)
        del multiply_54

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_58 = paddle._C_ops.add(add_57, multiply_96)
        del add_57

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_59 = paddle._C_ops.add(add_58, multiply_96)
        del add_58, multiply_96

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(add_59, full_4, float("0"), True)
        del add_59

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_73 = matmul_38
        del matmul_38

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_3 = paddle._C_ops.multiply(scale_26, assign_4)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_98 = paddle._C_ops.multiply(assign_73, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_99 = paddle._C_ops.multiply(assign_73, add_1)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_24 = [multiply_66, multiply_98]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_60 = paddle._C_ops.add(multiply_66, multiply_98)
        del multiply_66, multiply_98

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(add_60, full_1, float("0"), True)
        del add_60

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_75 = scale_27

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_74 = scale_27

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_25 = [multiply_91, multiply_99]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_61 = paddle._C_ops.add(multiply_91, multiply_99)
        del multiply_91, multiply_99

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(add_61, full_4, float("0"), True)
        del add_61

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_100 = paddle._C_ops.multiply(scale_28, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_101 = paddle._C_ops.multiply(scale_27, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_102 = paddle._C_ops.multiply(scale_27, scale_1)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_39 = paddle._C_ops.matmul(scale_27, parameter_4, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_26 = [multiply_59, multiply_59, multiply_101, multiply_101]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_62 = paddle._C_ops.add(multiply_59, multiply_59)
        del multiply_59

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_63 = paddle._C_ops.add(add_62, multiply_101)
        del add_62

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_64 = paddle._C_ops.add(add_63, multiply_101)
        del add_63, multiply_101

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(add_64, full_4, float("0"), True)
        del add_64

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_76 = matmul_39
        del matmul_39

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_103 = paddle._C_ops.multiply(scale_29, matmul_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_104 = paddle._C_ops.multiply(scale_29, assign_5)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_105 = paddle._C_ops.multiply(assign_76, subtract_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_106 = paddle._C_ops.multiply(assign_76, add_0)

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_27 = [multiply_78, multiply_105]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_65 = paddle._C_ops.add(multiply_78, multiply_105)
        del multiply_105, multiply_78

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(add_65, full_1, float("0"), True)
        del add_65

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_77 = scale_30

        # builtin.combine: ([100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32)
        combine_28 = [multiply_86, multiply_106]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_66 = paddle._C_ops.add(multiply_86, multiply_106)
        del multiply_106, multiply_86

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(add_66, full_4, float("0"), True)
        del add_66

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_107 = paddle._C_ops.multiply(scale_31, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_108 = paddle._C_ops.multiply(scale_30, tanh_2)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_109 = paddle._C_ops.multiply(scale_30, scale_2)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_29 = [multiply_74, multiply_74, multiply_108, multiply_108]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_67 = paddle._C_ops.add(multiply_74, multiply_74)
        del multiply_74

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_68 = paddle._C_ops.add(add_67, multiply_108)
        del add_67

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_69 = paddle._C_ops.add(add_68, multiply_108)
        del add_68, multiply_108

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(add_69, full_4, float("0"), True)
        del add_69

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_110 = paddle._C_ops.multiply(scale_32, matmul_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_78 = multiply_110
        del multiply_110

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_6 = paddle._C_ops.matmul(assign_78, parameter_4, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_111 = paddle._C_ops.multiply(matmul_6, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_112 = paddle._C_ops.multiply(matmul_6, assign_5)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_30 = [multiply_69, multiply_82, multiply_103, multiply_111]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_70 = paddle._C_ops.add(multiply_69, multiply_82)
        del multiply_69, multiply_82

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_71 = paddle._C_ops.add(add_70, multiply_103)
        del add_70, multiply_103

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_72 = paddle._C_ops.add(add_71, multiply_111)
        del add_71, multiply_111

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_79 = add_72
        del add_72

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_7 = paddle._C_ops.matmul(assign_79, parameter_6, False, True)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_113 = paddle._C_ops.multiply(matmul_7, assign_4)

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_40 = paddle._C_ops.matmul(multiply_3, parameter_6, False, False)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(multiply_113, full_4, float("0"), True)
        del multiply_113

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_80 = matmul_40
        del matmul_40

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_114 = paddle._C_ops.multiply(scale_33, tanh_0)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_115 = paddle._C_ops.multiply(assign_80, subtract_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_116 = paddle._C_ops.multiply(assign_80, matmul_1)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_31 = [multiply_62, multiply_71, multiply_104, multiply_115]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_73 = paddle._C_ops.add(multiply_62, multiply_71)
        del multiply_62, multiply_71

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_74 = paddle._C_ops.add(add_73, multiply_104)
        del add_73, multiply_104

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_11 = paddle._C_ops.add(add_74, multiply_115)
        del add_74, multiply_115

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_41 = paddle._C_ops.matmul(add_11, parameter_4, False, False)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_32 = [multiply_64, multiply_84, multiply_112, multiply_116]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_75 = paddle._C_ops.add(multiply_64, multiply_84)
        del multiply_64, multiply_84

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_76 = paddle._C_ops.add(add_75, multiply_112)
        del add_75, multiply_112

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_77 = paddle._C_ops.add(add_76, multiply_116)
        del add_76, multiply_116

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(add_77, full_4, float("0"), True)
        del add_77

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_81 = matmul_41
        del matmul_41

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_117 = paddle._C_ops.multiply(scale_34, tanh_1)

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_118 = paddle._C_ops.multiply(assign_81, matmul_0)

        # pd_op.scale: (100x20xf32) <- (100x20xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(multiply_118, full_4, float("0"), True)
        del multiply_118

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_119 = paddle._C_ops.multiply(scale_35, tanh_2)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_33 = [
            multiply_72,
            multiply_72,
            multiply_73,
            multiply_73,
            multiply_75,
            multiply_75,
            multiply_80,
            multiply_80,
            multiply_107,
            multiply_107,
            multiply_109,
            multiply_109,
            multiply_119,
            multiply_119,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_78 = paddle._C_ops.add(multiply_72, multiply_72)
        del multiply_72

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_79 = paddle._C_ops.add(add_78, multiply_73)
        del add_78

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_80 = paddle._C_ops.add(add_79, multiply_73)
        del add_79, multiply_73

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_81 = paddle._C_ops.add(add_80, multiply_75)
        del add_80

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_82 = paddle._C_ops.add(add_81, multiply_75)
        del add_81, multiply_75

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_83 = paddle._C_ops.add(add_82, multiply_80)
        del add_82

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_84 = paddle._C_ops.add(add_83, multiply_80)
        del add_83, multiply_80

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_85 = paddle._C_ops.add(add_84, multiply_107)
        del add_84

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_86 = paddle._C_ops.add(add_85, multiply_107)
        del add_85, multiply_107

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_87 = paddle._C_ops.add(add_86, multiply_109)
        del add_86

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_88 = paddle._C_ops.add(add_87, multiply_109)
        del add_87, multiply_109

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_89 = paddle._C_ops.add(add_88, multiply_119)
        del add_88

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_12 = paddle._C_ops.add(add_89, multiply_119)
        del add_89, multiply_119

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_120 = paddle._C_ops.multiply(add_12, subtract_0)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_83 = multiply_120
        del multiply_120

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_42 = paddle._C_ops.matmul(assign_83, parameter_4, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_34 = [
            multiply_57,
            multiply_57,
            multiply_58,
            multiply_58,
            multiply_60,
            multiply_60,
            multiply_89,
            multiply_89,
            multiply_100,
            multiply_100,
            multiply_102,
            multiply_102,
            multiply_117,
            multiply_117,
            matmul_42,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_90 = paddle._C_ops.add(multiply_57, multiply_57)
        del multiply_57

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_91 = paddle._C_ops.add(add_90, multiply_58)
        del add_90

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_92 = paddle._C_ops.add(add_91, multiply_58)
        del add_91, multiply_58

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_93 = paddle._C_ops.add(add_92, multiply_60)
        del add_92

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_94 = paddle._C_ops.add(add_93, multiply_60)
        del add_93, multiply_60

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_95 = paddle._C_ops.add(add_94, multiply_89)
        del add_94

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_96 = paddle._C_ops.add(add_95, multiply_89)
        del add_95, multiply_89

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_97 = paddle._C_ops.add(add_96, multiply_100)
        del add_96

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_98 = paddle._C_ops.add(add_97, multiply_100)
        del add_97, multiply_100

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_99 = paddle._C_ops.add(add_98, multiply_102)
        del add_98

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_100 = paddle._C_ops.add(add_99, multiply_102)
        del add_99, multiply_102

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_101 = paddle._C_ops.add(add_100, multiply_117)
        del add_100

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_102 = paddle._C_ops.add(add_101, multiply_117)
        del add_101, multiply_117

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_13 = paddle._C_ops.add(add_102, matmul_42)
        del add_102, matmul_42

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_121 = paddle._C_ops.multiply(add_13, subtract_1)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_85 = multiply_121
        del multiply_121

        # pd_op.matmul: (100x20xf32) <- (100x20xf32, 20x20xf32)
        matmul_43 = paddle._C_ops.matmul(assign_85, parameter_6, False, True)

        # builtin.combine: ([100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32]) <- (100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32, 100x20xf32)
        combine_35 = [
            multiply_52,
            multiply_52,
            multiply_53,
            multiply_53,
            multiply_55,
            multiply_55,
            multiply_93,
            multiply_93,
            multiply_95,
            multiply_95,
            multiply_97,
            multiply_97,
            multiply_114,
            multiply_114,
            matmul_43,
        ]

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_103 = paddle._C_ops.add(multiply_52, multiply_52)
        del multiply_52

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_104 = paddle._C_ops.add(add_103, multiply_53)
        del add_103

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_105 = paddle._C_ops.add(add_104, multiply_53)
        del add_104, multiply_53

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_106 = paddle._C_ops.add(add_105, multiply_55)
        del add_105

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_107 = paddle._C_ops.add(add_106, multiply_55)
        del add_106, multiply_55

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_108 = paddle._C_ops.add(add_107, multiply_93)
        del add_107

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_109 = paddle._C_ops.add(add_108, multiply_93)
        del add_108, multiply_93

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_110 = paddle._C_ops.add(add_109, multiply_95)
        del add_109

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_111 = paddle._C_ops.add(add_110, multiply_95)
        del add_110, multiply_95

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_112 = paddle._C_ops.add(add_111, multiply_97)
        del add_111

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_113 = paddle._C_ops.add(add_112, multiply_97)
        del add_112, multiply_97

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_114 = paddle._C_ops.add(add_113, multiply_114)
        del add_113

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_115 = paddle._C_ops.add(add_114, multiply_114)
        del add_114, multiply_114

        # pd_op.add: (100x20xf32) <- (100x20xf32, 100x20xf32)
        add_14 = paddle._C_ops.add(add_115, matmul_43)
        del add_115, matmul_43

        # pd_op.multiply: (100x20xf32) <- (100x20xf32, 100x20xf32)
        multiply_122 = paddle._C_ops.multiply(add_14, subtract_2)

        # pd_op.assign: (100x20xf32) <- (100x20xf32)
        assign_87 = multiply_122
        del multiply_122

        # pd_op.matmul: (100x1xf32) <- (100x20xf32, 1x20xf32)
        matmul_44 = paddle._C_ops.matmul(assign_87, parameter_8, False, True)

        # pd_op.add: (100x1xf32) <- (xf32, 100x1xf32)
        add_116 = paddle._C_ops.add(parameter_0, matmul_44)
        del matmul_44, parameter_0

        # pd_op.subtract: (100x1xf32) <- (100x1xf32, 100x1xf32)
        subtract_3 = paddle._C_ops.subtract(add_116, data_4)
        del add_116, data_4

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("2"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_124 = full_0

        # pd_op.assign: (xf32) <- (xf32)
        assign_123 = full_0

        # pd_op.assign: (xf32) <- (xf32)
        assign_122 = full_0

        # pd_op.assign: (xf32) <- (xf32)
        assign_121 = full_0

        # pd_op.elementwise_pow: (100x1xf32) <- (100x1xf32, xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(subtract_3, full_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [0, 1]

        # pd_op.sum: (xf32) <- (100x1xf32, 2xi64)
        sum_4 = paddle._C_ops.sum(
            elementwise_pow_0, full_int_array_1, paddle.float32, False
        )
        del elementwise_pow_0, full_int_array_1

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [], float("100"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_4, full_5)
        del full_5, sum_4

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_51 = paddle._C_ops.scale(divide_0, full_1, float("0"), True)

        # pd_op.matmul: (4x20xf32) <- (4x1xf32, 1x20xf32)
        matmul_45 = paddle._C_ops.matmul(data_3, parameter_8, False, False)

        # pd_op.add: (4x20xf32) <- (4x20xf32, 20xf32)
        add_117 = paddle._C_ops.add(matmul_45, parameter_7)
        del matmul_45, parameter_7

        # pd_op.tanh: (4x20xf32) <- (4x20xf32)
        tanh_3 = paddle._C_ops.tanh(add_117)
        del add_117

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_46 = paddle._C_ops.matmul(tanh_3, parameter_6, False, False)

        # pd_op.add: (4x20xf32) <- (4x20xf32, 20xf32)
        add_118 = paddle._C_ops.add(matmul_46, parameter_5)
        del matmul_46, parameter_5

        # pd_op.tanh: (4x20xf32) <- (4x20xf32)
        tanh_4 = paddle._C_ops.tanh(add_118)
        del add_118

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_47 = paddle._C_ops.matmul(tanh_4, parameter_4, False, False)

        # pd_op.add: (4x20xf32) <- (4x20xf32, 20xf32)
        add_119 = paddle._C_ops.add(matmul_47, parameter_3)
        del matmul_47, parameter_3

        # pd_op.tanh: (4x20xf32) <- (4x20xf32)
        tanh_5 = paddle._C_ops.tanh(add_119)
        del add_119

        # pd_op.matmul: (4x1xf32) <- (4x20xf32, 20x1xf32)
        matmul_48 = paddle._C_ops.matmul(tanh_5, parameter_2, False, False)

        # pd_op.add: (4x1xf32) <- (4x1xf32, 1xf32)
        add_120 = paddle._C_ops.add(matmul_48, parameter_1)
        del matmul_48, parameter_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (1x1xf32) <- (4x1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            add_120, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del add_120, full_int_array_2

        # pd_op.full: (4x1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [4, 1],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (4x1xf32) <- (4x1xf32)
        assign_102 = full_6

        # pd_op.assign: (4x1xf32) <- (4x1xf32)
        assign_92 = full_6

        # pd_op.assign: (4x1xf32) <- (4x1xf32)
        assign_88 = full_6

        # pd_op.matmul: (4x20xf32) <- (4x1xf32, 20x1xf32)
        matmul_8 = paddle._C_ops.matmul(assign_88, parameter_2, False, True)
        del parameter_2

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_123 = paddle._C_ops.multiply(tanh_5, tanh_5)

        # pd_op.subtract: (4x20xf32) <- (xf32, 4x20xf32)
        subtract_4 = paddle._C_ops.subtract(full_3, multiply_123)
        del multiply_123

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_115 = subtract_4

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_96 = subtract_4

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_124 = paddle._C_ops.multiply(matmul_8, subtract_4)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_89 = multiply_124
        del multiply_124

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_9 = paddle._C_ops.matmul(assign_89, parameter_4, False, True)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_125 = paddle._C_ops.multiply(tanh_4, tanh_4)

        # pd_op.subtract: (4x20xf32) <- (xf32, 4x20xf32)
        subtract_5 = paddle._C_ops.subtract(full_3, multiply_125)
        del multiply_125

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_117 = subtract_5

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_98 = subtract_5

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_126 = paddle._C_ops.multiply(matmul_9, subtract_5)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_90 = multiply_126
        del multiply_126

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_10 = paddle._C_ops.matmul(assign_90, parameter_6, False, True)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_127 = paddle._C_ops.multiply(tanh_3, tanh_3)

        # pd_op.subtract: (4x20xf32) <- (xf32, 4x20xf32)
        subtract_6 = paddle._C_ops.subtract(full_3, multiply_127)
        del full_3, multiply_127

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_119 = subtract_6

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_100 = subtract_6

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_128 = paddle._C_ops.multiply(matmul_10, subtract_6)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_91 = multiply_128
        del multiply_128

        # pd_op.matmul: (4x1xf32) <- (4x20xf32, 1x20xf32)
        matmul_49 = paddle._C_ops.matmul(assign_91, parameter_8, False, True)

        # pd_op.matmul: (1x20xf32) <- (4x1xf32, 4x20xf32)
        matmul_50 = paddle._C_ops.matmul(data_3, assign_91, True, False)
        del data_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.slice: (1x1xf32) <- (4x1xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            matmul_49, [0], full_int_array_3, full_int_array_4, [1], []
        )
        del full_int_array_3, matmul_49

        # pd_op.matmul: (4x20xf32) <- (4x1xf32, 1x20xf32)
        matmul_51 = paddle._C_ops.matmul(full_6, parameter_8, False, False)

        # pd_op.matmul: (1x20xf32) <- (4x1xf32, 4x20xf32)
        matmul_52 = paddle._C_ops.matmul(full_6, assign_91, True, False)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_93 = matmul_51
        del matmul_51

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_103 = assign_93

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_4 = paddle._C_ops.multiply(assign_93, subtract_6)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_129 = paddle._C_ops.multiply(assign_93, matmul_10)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_53 = paddle._C_ops.matmul(multiply_4, parameter_6, False, False)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(multiply_129, full_4, float("0"), True)
        del multiply_129

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_94 = matmul_53
        del matmul_53

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_130 = paddle._C_ops.multiply(scale_36, tanh_3)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_5 = paddle._C_ops.multiply(assign_94, subtract_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_131 = paddle._C_ops.multiply(assign_94, matmul_9)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_54 = paddle._C_ops.matmul(multiply_5, parameter_4, False, False)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(multiply_131, full_4, float("0"), True)
        del multiply_131

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_95 = matmul_54
        del matmul_54

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_132 = paddle._C_ops.multiply(scale_37, tanh_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_133 = paddle._C_ops.multiply(assign_95, matmul_8)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(multiply_133, full_4, float("0"), True)
        del multiply_133

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_134 = paddle._C_ops.multiply(scale_38, tanh_5)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_36 = [multiply_134, multiply_134]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_15 = paddle._C_ops.add(multiply_134, multiply_134)
        del multiply_134

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_135 = paddle._C_ops.multiply(add_15, subtract_4)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_97 = multiply_135
        del multiply_135

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_55 = paddle._C_ops.matmul(assign_97, parameter_4, False, True)

        # builtin.combine: ([4x20xf32, 4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32, 4x20xf32)
        combine_37 = [multiply_132, multiply_132, matmul_55]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_121 = paddle._C_ops.add(multiply_132, multiply_132)
        del multiply_132

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_16 = paddle._C_ops.add(add_121, matmul_55)
        del add_121, matmul_55

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_136 = paddle._C_ops.multiply(add_16, subtract_5)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_99 = multiply_136
        del multiply_136

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_56 = paddle._C_ops.matmul(assign_99, parameter_6, False, True)

        # builtin.combine: ([4x20xf32, 4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32, 4x20xf32)
        combine_38 = [multiply_130, multiply_130, matmul_56]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_122 = paddle._C_ops.add(multiply_130, multiply_130)
        del multiply_130

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_17 = paddle._C_ops.add(add_122, matmul_56)
        del add_122, matmul_56

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_137 = paddle._C_ops.multiply(add_17, subtract_6)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_101 = multiply_137
        del multiply_137

        # pd_op.matmul: (4x1xf32) <- (4x20xf32, 1x20xf32)
        matmul_57 = paddle._C_ops.matmul(assign_101, parameter_8, False, True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (1x1xf32) <- (4x1xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            matmul_57, [0], full_int_array_4, full_int_array_5, [1], []
        )
        del full_int_array_4, matmul_57

        # pd_op.matmul: (1x20xf32) <- (4x1xf32, 4x20xf32)
        matmul_58 = paddle._C_ops.matmul(full_6, assign_101, True, False)
        del full_6

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_138 = paddle._C_ops.multiply(assign_93, add_17)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(multiply_4, full_1, float("0"), True)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_105 = scale_39

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_104 = scale_39

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_40 = paddle._C_ops.scale(multiply_138, full_4, float("0"), True)
        del multiply_138

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_139 = paddle._C_ops.multiply(scale_40, tanh_3)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_140 = paddle._C_ops.multiply(scale_39, tanh_3)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_141 = paddle._C_ops.multiply(scale_39, scale_36)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_59 = paddle._C_ops.matmul(scale_39, parameter_6, False, False)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_39 = [multiply_140, multiply_140]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_123 = paddle._C_ops.add(multiply_140, multiply_140)
        del multiply_140

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_41 = paddle._C_ops.scale(add_123, full_4, float("0"), True)
        del add_123

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_106 = matmul_59
        del matmul_59

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_6 = paddle._C_ops.multiply(scale_41, assign_93)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_142 = paddle._C_ops.multiply(assign_106, subtract_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_143 = paddle._C_ops.multiply(assign_106, add_16)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_42 = paddle._C_ops.scale(multiply_142, full_1, float("0"), True)
        del multiply_142

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_108 = scale_42

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_107 = scale_42

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_43 = paddle._C_ops.scale(multiply_143, full_4, float("0"), True)
        del multiply_143

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_144 = paddle._C_ops.multiply(scale_43, tanh_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_145 = paddle._C_ops.multiply(scale_42, tanh_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_146 = paddle._C_ops.multiply(scale_42, scale_37)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_60 = paddle._C_ops.matmul(scale_42, parameter_4, False, False)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_40 = [multiply_145, multiply_145]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_124 = paddle._C_ops.add(multiply_145, multiply_145)
        del multiply_145

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_44 = paddle._C_ops.scale(add_124, full_4, float("0"), True)
        del add_124

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_109 = matmul_60
        del matmul_60

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_147 = paddle._C_ops.multiply(scale_44, matmul_9)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_148 = paddle._C_ops.multiply(scale_44, assign_94)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_149 = paddle._C_ops.multiply(assign_109, subtract_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_150 = paddle._C_ops.multiply(assign_109, add_15)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_45 = paddle._C_ops.scale(multiply_149, full_1, float("0"), True)
        del multiply_149

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_110 = scale_45

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_46 = paddle._C_ops.scale(multiply_150, full_4, float("0"), True)
        del multiply_150

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_151 = paddle._C_ops.multiply(scale_46, tanh_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_152 = paddle._C_ops.multiply(scale_45, tanh_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_153 = paddle._C_ops.multiply(scale_45, scale_38)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_41 = [multiply_152, multiply_152]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_125 = paddle._C_ops.add(multiply_152, multiply_152)
        del multiply_152

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_47 = paddle._C_ops.scale(add_125, full_4, float("0"), True)
        del add_125

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_154 = paddle._C_ops.multiply(scale_47, matmul_8)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_111 = multiply_154
        del multiply_154

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_11 = paddle._C_ops.matmul(assign_111, parameter_4, False, True)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_155 = paddle._C_ops.multiply(matmul_11, subtract_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_156 = paddle._C_ops.multiply(matmul_11, assign_94)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_42 = [multiply_147, multiply_155]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_126 = paddle._C_ops.add(multiply_147, multiply_155)
        del multiply_147, multiply_155

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_112 = add_126
        del add_126

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_12 = paddle._C_ops.matmul(assign_112, parameter_6, False, True)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_157 = paddle._C_ops.multiply(matmul_12, assign_93)

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_61 = paddle._C_ops.matmul(multiply_6, parameter_6, False, False)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_48 = paddle._C_ops.scale(multiply_157, full_4, float("0"), True)
        del multiply_157

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_113 = matmul_61
        del matmul_61

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_158 = paddle._C_ops.multiply(scale_48, tanh_3)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_159 = paddle._C_ops.multiply(assign_113, subtract_5)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_160 = paddle._C_ops.multiply(assign_113, matmul_9)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_43 = [multiply_148, multiply_159]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_18 = paddle._C_ops.add(multiply_148, multiply_159)
        del multiply_148, multiply_159

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_62 = paddle._C_ops.matmul(add_18, parameter_4, False, False)

        # builtin.combine: ([4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32)
        combine_44 = [multiply_156, multiply_160]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_127 = paddle._C_ops.add(multiply_156, multiply_160)
        del multiply_156, multiply_160

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_49 = paddle._C_ops.scale(add_127, full_4, float("0"), True)
        del add_127

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_114 = matmul_62
        del matmul_62

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_161 = paddle._C_ops.multiply(scale_49, tanh_4)

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_162 = paddle._C_ops.multiply(assign_114, matmul_8)

        # pd_op.scale: (4x20xf32) <- (4x20xf32, 1xf32)
        scale_50 = paddle._C_ops.scale(multiply_162, full_4, float("0"), True)
        del full_4, multiply_162

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_163 = paddle._C_ops.multiply(scale_50, tanh_5)

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
        add_128 = paddle._C_ops.add(multiply_151, multiply_151)
        del multiply_151

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_129 = paddle._C_ops.add(add_128, multiply_153)
        del add_128

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_130 = paddle._C_ops.add(add_129, multiply_153)
        del add_129, multiply_153

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_131 = paddle._C_ops.add(add_130, multiply_163)
        del add_130

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_19 = paddle._C_ops.add(add_131, multiply_163)
        del add_131, multiply_163

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_164 = paddle._C_ops.multiply(add_19, subtract_4)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_116 = multiply_164
        del multiply_164

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_63 = paddle._C_ops.matmul(assign_116, parameter_4, False, True)
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
        add_132 = paddle._C_ops.add(multiply_144, multiply_144)
        del multiply_144

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_133 = paddle._C_ops.add(add_132, multiply_146)
        del add_132

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_134 = paddle._C_ops.add(add_133, multiply_146)
        del add_133, multiply_146

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_135 = paddle._C_ops.add(add_134, multiply_161)
        del add_134

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_136 = paddle._C_ops.add(add_135, multiply_161)
        del add_135, multiply_161

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_20 = paddle._C_ops.add(add_136, matmul_63)
        del add_136, matmul_63

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_165 = paddle._C_ops.multiply(add_20, subtract_5)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_118 = multiply_165
        del multiply_165

        # pd_op.matmul: (4x20xf32) <- (4x20xf32, 20x20xf32)
        matmul_64 = paddle._C_ops.matmul(assign_118, parameter_6, False, True)
        del parameter_6

        # builtin.combine: ([4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32]) <- (4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32, 4x20xf32)
        combine_47 = [
            multiply_139,
            multiply_139,
            multiply_141,
            multiply_141,
            multiply_158,
            multiply_158,
            matmul_64,
        ]

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_137 = paddle._C_ops.add(multiply_139, multiply_139)
        del multiply_139

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_138 = paddle._C_ops.add(add_137, multiply_141)
        del add_137

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_139 = paddle._C_ops.add(add_138, multiply_141)
        del add_138, multiply_141

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_140 = paddle._C_ops.add(add_139, multiply_158)
        del add_139

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_141 = paddle._C_ops.add(add_140, multiply_158)
        del add_140, multiply_158

        # pd_op.add: (4x20xf32) <- (4x20xf32, 4x20xf32)
        add_21 = paddle._C_ops.add(add_141, matmul_64)
        del add_141, matmul_64

        # pd_op.multiply: (4x20xf32) <- (4x20xf32, 4x20xf32)
        multiply_166 = paddle._C_ops.multiply(add_21, subtract_6)

        # pd_op.assign: (4x20xf32) <- (4x20xf32)
        assign_120 = multiply_166
        del multiply_166

        # pd_op.matmul: (4x1xf32) <- (4x20xf32, 1x20xf32)
        matmul_65 = paddle._C_ops.matmul(assign_120, parameter_8, False, True)
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
        elementwise_pow_1 = paddle._C_ops.elementwise_pow(subtract_7, full_0)

        # pd_op.sum: (xf32) <- (4x1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(elementwise_pow_1, full_int_array_0, None, False)
        del elementwise_pow_1

        # pd_op.subtract: (4x1xf32) <- (1x1xf32, 4x1xf32)
        subtract_8 = paddle._C_ops.subtract(slice_1, data_6)
        del data_6

        # pd_op.elementwise_pow: (4x1xf32) <- (4x1xf32, xf32)
        elementwise_pow_2 = paddle._C_ops.elementwise_pow(subtract_8, full_0)

        # pd_op.sum: (xf32) <- (4x1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(elementwise_pow_2, full_int_array_0, None, False)
        del elementwise_pow_2

        # pd_op.subtract: (4x1xf32) <- (1x1xf32, 4x1xf32)
        subtract_9 = paddle._C_ops.subtract(slice_2, data_7)
        del data_7

        # pd_op.elementwise_pow: (4x1xf32) <- (4x1xf32, xf32)
        elementwise_pow_3 = paddle._C_ops.elementwise_pow(subtract_9, full_0)

        # pd_op.sum: (xf32) <- (4x1xf32, 0xi64)
        sum_2 = paddle._C_ops.sum(elementwise_pow_3, full_int_array_0, None, False)
        del elementwise_pow_3

        # pd_op.subtract: (4x1xf32) <- (1x1xf32, 4x1xf32)
        subtract_10 = paddle._C_ops.subtract(slice_3, data_8)
        del data_8

        # pd_op.elementwise_pow: (4x1xf32) <- (4x1xf32, xf32)
        elementwise_pow_4 = paddle._C_ops.elementwise_pow(subtract_10, full_0)

        # pd_op.sum: (xf32) <- (4x1xf32, 0xi64)
        sum_3 = paddle._C_ops.sum(elementwise_pow_4, full_int_array_0, None, False)
        del elementwise_pow_4, full_int_array_0

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_60 = paddle._C_ops.scale(sum_0, full_1, float("0"), True)
        del full_1

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_142 = paddle._C_ops.add(scale_60, sum_1)
        del scale_60

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_143 = paddle._C_ops.add(add_142, sum_2)
        del add_142

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_22 = paddle._C_ops.add(add_143, sum_3)
        del add_143

        return (
            tanh_0,
            tanh_1,
            tanh_2,
            assign_0,
            matmul_0,
            subtract_0,
            assign_1,
            matmul_1,
            subtract_1,
            assign_2,
            matmul_2,
            subtract_2,
            assign_3,
            assign_4,
            multiply_0,
            scale_0,
            assign_5,
            multiply_1,
            scale_1,
            assign_6,
            scale_2,
            add_0,
            assign_7,
            assign_8,
            add_1,
            assign_9,
            assign_10,
            add_2,
            assign_11,
            assign_12,
            assign_13,
            scale_3,
            assign_14,
            assign_15,
            scale_4,
            scale_5,
            assign_16,
            multiply_2,
            scale_6,
            assign_17,
            assign_18,
            scale_7,
            scale_8,
            assign_19,
            scale_9,
            assign_20,
            scale_10,
            scale_11,
            assign_21,
            matmul_3,
            assign_22,
            matmul_4,
            scale_12,
            assign_23,
            add_3,
            scale_13,
            assign_24,
            scale_14,
            add_4,
            assign_25,
            assign_26,
            add_5,
            assign_27,
            assign_28,
            add_6,
            assign_29,
            assign_30,
            assign_31,
            assign_32,
            assign_33,
            assign_34,
            assign_35,
            assign_36,
            assign_37,
            assign_38,
            scale_15,
            assign_39,
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
            scale_16,
            scale_17,
            assign_50,
            assign_51,
            assign_52,
            scale_18,
            assign_53,
            assign_54,
            assign_55,
            assign_56,
            assign_57,
            assign_58,
            assign_59,
            scale_19,
            add_7,
            assign_60,
            assign_61,
            assign_62,
            assign_63,
            scale_20,
            assign_64,
            scale_21,
            assign_65,
            add_8,
            assign_66,
            assign_67,
            scale_22,
            assign_68,
            matmul_5,
            scale_23,
            assign_69,
            add_9,
            assign_70,
            add_10,
            scale_24,
            assign_71,
            assign_72,
            scale_25,
            scale_26,
            assign_73,
            multiply_3,
            scale_27,
            assign_74,
            assign_75,
            scale_28,
            scale_29,
            assign_76,
            scale_30,
            assign_77,
            scale_31,
            scale_32,
            assign_78,
            matmul_6,
            assign_79,
            matmul_7,
            scale_33,
            assign_80,
            add_11,
            scale_34,
            assign_81,
            scale_35,
            add_12,
            assign_82,
            assign_83,
            add_13,
            assign_84,
            assign_85,
            add_14,
            assign_86,
            assign_87,
            subtract_3,
            full_0,
            tanh_3,
            tanh_4,
            tanh_5,
            slice_0,
            assign_88,
            matmul_8,
            subtract_4,
            assign_89,
            matmul_9,
            subtract_5,
            assign_90,
            matmul_10,
            subtract_6,
            assign_91,
            slice_1,
            assign_92,
            assign_93,
            multiply_4,
            scale_36,
            assign_94,
            multiply_5,
            scale_37,
            assign_95,
            scale_38,
            add_15,
            assign_96,
            assign_97,
            add_16,
            assign_98,
            assign_99,
            add_17,
            assign_100,
            assign_101,
            slice_2,
            assign_102,
            assign_103,
            scale_39,
            assign_104,
            assign_105,
            scale_40,
            scale_41,
            assign_106,
            multiply_6,
            scale_42,
            assign_107,
            assign_108,
            scale_43,
            scale_44,
            assign_109,
            scale_45,
            assign_110,
            scale_46,
            scale_47,
            assign_111,
            matmul_11,
            assign_112,
            matmul_12,
            scale_48,
            assign_113,
            add_18,
            scale_49,
            assign_114,
            scale_50,
            add_19,
            assign_115,
            assign_116,
            add_20,
            assign_117,
            assign_118,
            add_21,
            assign_119,
            assign_120,
            slice_3,
            subtract_7,
            assign_121,
            subtract_8,
            assign_122,
            subtract_9,
            assign_123,
            subtract_10,
            assign_124,
            divide_0,
            sum_0,
            sum_1,
            sum_2,
            sum_3,
            add_22,
            scale_51,
        )
