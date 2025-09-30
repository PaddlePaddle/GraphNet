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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [34240]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_1

        # pd_op.slice: (1x34240x256xf32) <- (1x45640x256xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.transpose: (1x256x34240xf32) <- (1x34240x256xf32)
        transpose_0 = paddle._C_ops.transpose(slice_0, [0, 2, 1])
        del slice_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 256, 214, 160]

        # pd_op.reshape: (1x256x214x160xf32) <- (1x256x34240xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, full_int_array_2)
        del full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [42800]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_3

        # pd_op.slice: (1x8560x256xf32) <- (1x45640x256xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_2, [1], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.transpose: (1x256x8560xf32) <- (1x8560x256xf32)
        transpose_1 = paddle._C_ops.transpose(slice_1, [0, 2, 1])
        del slice_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 256, 107, 80]

        # pd_op.reshape: (1x256x107x80xf32) <- (1x256x8560xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_1, full_int_array_4)
        del full_int_array_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [44960]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_5

        # pd_op.slice: (1x2160x256xf32) <- (1x45640x256xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_2, [1], full_int_array_3, full_int_array_5, [1], []
        )

        # pd_op.transpose: (1x256x2160xf32) <- (1x2160x256xf32)
        transpose_2 = paddle._C_ops.transpose(slice_2, [0, 2, 1])
        del slice_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 256, 54, 40]

        # pd_op.reshape: (1x256x54x40xf32) <- (1x256x2160xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_2, full_int_array_6)
        del full_int_array_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [45500]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_7

        # pd_op.slice: (1x540x256xf32) <- (1x45640x256xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_2, [1], full_int_array_5, full_int_array_7, [1], []
        )

        # pd_op.transpose: (1x256x540xf32) <- (1x540x256xf32)
        transpose_3 = paddle._C_ops.transpose(slice_3, [0, 2, 1])
        del slice_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 256, 27, 20]

        # pd_op.reshape: (1x256x27x20xf32) <- (1x256x540xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_8)
        del full_int_array_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2147483647]

        # pd_op.slice: (1x140x256xf32) <- (1x45640x256xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_2, [1], full_int_array_7, full_int_array_9, [1], []
        )
        del data_2

        # pd_op.transpose: (1x256x140xf32) <- (1x140x256xf32)
        transpose_4 = paddle._C_ops.transpose(slice_4, [0, 2, 1])
        del slice_4

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 256, 14, 10]

        # pd_op.reshape: (1x256x14x10xf32) <- (1x256x140xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_4, full_int_array_10)
        del full_int_array_10

        # pd_op.conv2d: (1x256x7x5xf32) <- (1x256x14x10xf32, 256x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            reshape_4, parameter_51, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_51

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_50, full_int_array_11)
        del full_int_array_11, parameter_50

        # pd_op.add: (1x256x7x5xf32) <- (1x256x7x5xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_5)

        # pd_op.group_norm: (1x256x7x5xf32, 1x32xf32, 1x32xf32) <- (1x256x7x5xf32, 256xf32, 256xf32)
        group_norm_2, group_norm_0, group_norm_1 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_0, parameter_49, parameter_48, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_48, parameter_49

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_12

        # pd_op.slice: (1x1100x4xf32) <- (7x1x1100x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_12, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_19 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_20 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_21 = full_1

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(slice_5, full_0, full_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.001"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_22 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_23 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_24 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_25 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_26 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_27 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_28 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_30 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_31 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_32 = full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_33 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_34 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_35 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_36 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_37 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_38 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_39 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_40 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_41 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_42 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_43 = full_3

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(clip_0, full_2, full_3)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_44 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_45 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_46 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_47 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_48 = full_4

        # pd_op.scale: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(clip_0, full_4, float("1"), True)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(scale_0, full_2, full_3)

        # pd_op.divide: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        divide_0 = paddle._C_ops.divide(clip_1, clip_2)

        # pd_op.log: (1x1100x4xf32) <- (1x1100x4xf32)
        log_0 = paddle._C_ops.log(divide_0)

        # pd_op.slice: (1x1100x256xf32) <- (6x1x1100x256xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_12, [1], [0]
        )

        # pd_op.assign: (1x1100x256xf32) <- (1x1100x256xf32)
        assign_49 = slice_6

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_0 = paddle._C_ops.matmul(slice_6, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_46)
        del parameter_46

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(slice_6, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_44)
        del parameter_44

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_0 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(relu_0, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_42)
        del parameter_42

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_1 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_3 = paddle._C_ops.matmul(relu_1, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_40)
        del parameter_40

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        add_5 = paddle._C_ops.add(add_4, log_0)

        # pd_op.sigmoid: (1x1100x4xf32) <- (1x1100x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_5)
        del add_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_13

        # pd_op.slice: (1x1100x4xf32) <- (7x1x1100x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [0], full_int_array_12, full_int_array_13, [1], [0]
        )

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_3 = paddle._C_ops.clip(slice_7, full_0, full_1)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_4 = paddle._C_ops.clip(clip_3, full_2, full_3)

        # pd_op.scale: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(clip_3, full_4, float("1"), True)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_5 = paddle._C_ops.clip(scale_1, full_2, full_3)

        # pd_op.divide: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        divide_1 = paddle._C_ops.divide(clip_4, clip_5)

        # pd_op.log: (1x1100x4xf32) <- (1x1100x4xf32)
        log_1 = paddle._C_ops.log(divide_1)

        # pd_op.slice: (1x1100x256xf32) <- (6x1x1100x256xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_0, [0], full_int_array_12, full_int_array_13, [1], [0]
        )

        # pd_op.assign: (1x1100x256xf32) <- (1x1100x256xf32)
        assign_55 = slice_8

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_4 = paddle._C_ops.matmul(slice_8, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_6 = paddle._C_ops.add(matmul_4, parameter_38)
        del parameter_38

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_5 = paddle._C_ops.matmul(slice_8, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_7 = paddle._C_ops.add(matmul_5, parameter_36)
        del parameter_36

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_2 = paddle._C_ops.relu(add_7)
        del add_7

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_6 = paddle._C_ops.matmul(relu_2, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_34)
        del parameter_34

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_3 = paddle._C_ops.relu(add_8)
        del add_8

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_7 = paddle._C_ops.matmul(relu_3, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_32)
        del parameter_32

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        add_10 = paddle._C_ops.add(add_9, log_1)

        # pd_op.sigmoid: (1x1100x4xf32) <- (1x1100x4xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_10)
        del add_10

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_60 = full_int_array_14

        # pd_op.slice: (1x1100x4xf32) <- (7x1x1100x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_1, [0], full_int_array_13, full_int_array_14, [1], [0]
        )

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_6 = paddle._C_ops.clip(slice_9, full_0, full_1)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_7 = paddle._C_ops.clip(clip_6, full_2, full_3)

        # pd_op.scale: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(clip_6, full_4, float("1"), True)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_8 = paddle._C_ops.clip(scale_2, full_2, full_3)

        # pd_op.divide: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        divide_2 = paddle._C_ops.divide(clip_7, clip_8)

        # pd_op.log: (1x1100x4xf32) <- (1x1100x4xf32)
        log_2 = paddle._C_ops.log(divide_2)

        # pd_op.slice: (1x1100x256xf32) <- (6x1x1100x256xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_0, [0], full_int_array_13, full_int_array_14, [1], [0]
        )

        # pd_op.assign: (1x1100x256xf32) <- (1x1100x256xf32)
        assign_61 = slice_10

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_8 = paddle._C_ops.matmul(slice_10, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_11 = paddle._C_ops.add(matmul_8, parameter_30)
        del parameter_30

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_9 = paddle._C_ops.matmul(slice_10, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_28)
        del parameter_28

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_4 = paddle._C_ops.relu(add_12)
        del add_12

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_10 = paddle._C_ops.matmul(relu_4, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_26)
        del parameter_26

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_5 = paddle._C_ops.relu(add_13)
        del add_13

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_11 = paddle._C_ops.matmul(relu_5, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_14 = paddle._C_ops.add(matmul_11, parameter_24)
        del parameter_24

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        add_15 = paddle._C_ops.add(add_14, log_2)

        # pd_op.sigmoid: (1x1100x4xf32) <- (1x1100x4xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_15)
        del add_15

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_62 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_63 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_64 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_65 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_66 = full_int_array_15

        # pd_op.slice: (1x1100x4xf32) <- (7x1x1100x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_1, [0], full_int_array_14, full_int_array_15, [1], [0]
        )

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_9 = paddle._C_ops.clip(slice_11, full_0, full_1)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_10 = paddle._C_ops.clip(clip_9, full_2, full_3)

        # pd_op.scale: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(clip_9, full_4, float("1"), True)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_11 = paddle._C_ops.clip(scale_3, full_2, full_3)

        # pd_op.divide: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        divide_3 = paddle._C_ops.divide(clip_10, clip_11)

        # pd_op.log: (1x1100x4xf32) <- (1x1100x4xf32)
        log_3 = paddle._C_ops.log(divide_3)

        # pd_op.slice: (1x1100x256xf32) <- (6x1x1100x256xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            data_0, [0], full_int_array_14, full_int_array_15, [1], [0]
        )

        # pd_op.assign: (1x1100x256xf32) <- (1x1100x256xf32)
        assign_67 = slice_12

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_12 = paddle._C_ops.matmul(slice_12, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_16 = paddle._C_ops.add(matmul_12, parameter_22)
        del parameter_22

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_13 = paddle._C_ops.matmul(slice_12, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_17 = paddle._C_ops.add(matmul_13, parameter_20)
        del parameter_20

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_6 = paddle._C_ops.relu(add_17)
        del add_17

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_14 = paddle._C_ops.matmul(relu_6, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_18 = paddle._C_ops.add(matmul_14, parameter_18)
        del parameter_18

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_7 = paddle._C_ops.relu(add_18)
        del add_18

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_15 = paddle._C_ops.matmul(relu_7, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_19 = paddle._C_ops.add(matmul_15, parameter_16)
        del parameter_16

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        add_20 = paddle._C_ops.add(add_19, log_3)

        # pd_op.sigmoid: (1x1100x4xf32) <- (1x1100x4xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(add_20)
        del add_20

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [5]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_68 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_69 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_70 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_71 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_72 = full_int_array_16

        # pd_op.slice: (1x1100x4xf32) <- (7x1x1100x4xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            data_1, [0], full_int_array_15, full_int_array_16, [1], [0]
        )

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_12 = paddle._C_ops.clip(slice_13, full_0, full_1)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_13 = paddle._C_ops.clip(clip_12, full_2, full_3)

        # pd_op.scale: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(clip_12, full_4, float("1"), True)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_14 = paddle._C_ops.clip(scale_4, full_2, full_3)

        # pd_op.divide: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        divide_4 = paddle._C_ops.divide(clip_13, clip_14)

        # pd_op.log: (1x1100x4xf32) <- (1x1100x4xf32)
        log_4 = paddle._C_ops.log(divide_4)

        # pd_op.slice: (1x1100x256xf32) <- (6x1x1100x256xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            data_0, [0], full_int_array_15, full_int_array_16, [1], [0]
        )

        # pd_op.assign: (1x1100x256xf32) <- (1x1100x256xf32)
        assign_73 = slice_14

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_16 = paddle._C_ops.matmul(slice_14, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_21 = paddle._C_ops.add(matmul_16, parameter_14)
        del parameter_14

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_17 = paddle._C_ops.matmul(slice_14, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_22 = paddle._C_ops.add(matmul_17, parameter_12)
        del parameter_12

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_8 = paddle._C_ops.relu(add_22)
        del add_22

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_18 = paddle._C_ops.matmul(relu_8, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_23 = paddle._C_ops.add(matmul_18, parameter_10)
        del parameter_10

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_9 = paddle._C_ops.relu(add_23)
        del add_23

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_19 = paddle._C_ops.matmul(relu_9, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_24 = paddle._C_ops.add(matmul_19, parameter_8)
        del parameter_8

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        add_25 = paddle._C_ops.add(add_24, log_4)

        # pd_op.sigmoid: (1x1100x4xf32) <- (1x1100x4xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(add_25)
        del add_25

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [6]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_74 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_75 = full_int_array_17

        # pd_op.slice: (1x1100x4xf32) <- (7x1x1100x4xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            data_1, [0], full_int_array_16, full_int_array_17, [1], [0]
        )
        del data_1

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_15 = paddle._C_ops.clip(slice_15, full_0, full_1)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_16 = paddle._C_ops.clip(clip_15, full_2, full_3)

        # pd_op.scale: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(clip_15, full_4, float("1"), True)

        # pd_op.clip: (1x1100x4xf32) <- (1x1100x4xf32, 1xf32, 1xf32)
        clip_17 = paddle._C_ops.clip(scale_5, full_2, full_3)

        # pd_op.divide: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        divide_5 = paddle._C_ops.divide(clip_16, clip_17)

        # pd_op.log: (1x1100x4xf32) <- (1x1100x4xf32)
        log_5 = paddle._C_ops.log(divide_5)

        # pd_op.slice: (1x1100x256xf32) <- (6x1x1100x256xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            data_0, [0], full_int_array_16, full_int_array_17, [1], [0]
        )
        del data_0

        # pd_op.assign: (1x1100x256xf32) <- (1x1100x256xf32)
        assign_76 = slice_16

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_20 = paddle._C_ops.matmul(slice_16, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_26 = paddle._C_ops.add(matmul_20, parameter_6)
        del parameter_6

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_21 = paddle._C_ops.matmul(slice_16, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_27 = paddle._C_ops.add(matmul_21, parameter_4)
        del parameter_4

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_10 = paddle._C_ops.relu(add_27)
        del add_27

        # pd_op.matmul: (1x1100x256xf32) <- (1x1100x256xf32, 256x256xf32)
        matmul_22 = paddle._C_ops.matmul(relu_10, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (1x1100x256xf32) <- (1x1100x256xf32, 256xf32)
        add_28 = paddle._C_ops.add(matmul_22, parameter_2)
        del parameter_2

        # pd_op.relu: (1x1100x256xf32) <- (1x1100x256xf32)
        relu_11 = paddle._C_ops.relu(add_28)
        del add_28

        # pd_op.matmul: (1x1100x4xf32) <- (1x1100x256xf32, 256x4xf32)
        matmul_23 = paddle._C_ops.matmul(relu_11, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 4xf32)
        add_29 = paddle._C_ops.add(matmul_23, parameter_0)
        del parameter_0

        # pd_op.add: (1x1100x4xf32) <- (1x1100x4xf32, 1x1100x4xf32)
        add_30 = paddle._C_ops.add(add_29, log_5)

        # pd_op.sigmoid: (1x1100x4xf32) <- (1x1100x4xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(add_30)
        del add_30

        # builtin.combine: ([1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32]) <- (1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32)
        combine_0 = [add_1, add_6, add_11, add_16, add_21, add_26]

        # pd_op.stack: (6x1x1100x4xf32) <- ([1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # builtin.combine: ([1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32]) <- (1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32)
        combine_1 = [sigmoid_0, sigmoid_1, sigmoid_2, sigmoid_3, sigmoid_4, sigmoid_5]

        # pd_op.stack: (6x1x1100x4xf32) <- ([1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32, 1x1100x4xf32])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del (
            add_0,
            add_1,
            add_11,
            add_14,
            add_16,
            add_19,
            add_21,
            add_24,
            add_26,
            add_29,
            add_4,
            add_6,
            add_9,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
            assign_23,
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
            assign_66,
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
            assign_8,
            assign_9,
            clip_0,
            clip_1,
            clip_10,
            clip_11,
            clip_12,
            clip_13,
            clip_14,
            clip_15,
            clip_16,
            clip_17,
            clip_2,
            clip_3,
            clip_4,
            clip_5,
            clip_6,
            clip_7,
            clip_8,
            clip_9,
            combine_1,
            conv2d_0,
            divide_0,
            divide_1,
            divide_2,
            divide_3,
            divide_4,
            divide_5,
            full_0,
            full_1,
            full_2,
            full_3,
            full_4,
            full_int_array_0,
            full_int_array_1,
            full_int_array_12,
            full_int_array_13,
            full_int_array_14,
            full_int_array_15,
            full_int_array_16,
            full_int_array_17,
            full_int_array_3,
            full_int_array_5,
            full_int_array_7,
            full_int_array_9,
            log_0,
            log_1,
            log_2,
            log_3,
            log_4,
            log_5,
            matmul_0,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_19,
            matmul_2,
            matmul_20,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_3,
            matmul_4,
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_8,
            matmul_9,
            relu_0,
            relu_1,
            relu_10,
            relu_11,
            relu_2,
            relu_3,
            relu_4,
            relu_5,
            relu_6,
            relu_7,
            relu_8,
            relu_9,
            reshape_4,
            reshape_5,
            scale_0,
            scale_1,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            sigmoid_0,
            sigmoid_1,
            sigmoid_2,
            sigmoid_3,
            sigmoid_4,
            sigmoid_5,
            slice_10,
            slice_11,
            slice_12,
            slice_13,
            slice_14,
            slice_15,
            slice_16,
            slice_5,
            slice_6,
            slice_7,
            slice_8,
            slice_9,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            transpose_4,
        )

        return (
            stack_0,
            stack_1,
            reshape_0,
            reshape_1,
            reshape_2,
            reshape_3,
            group_norm_2,
        )
