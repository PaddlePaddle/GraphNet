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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("50"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (-1x50xf32) <- (-1xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            data_0 % paddle.cast(full_0, data_0.dtype), full_0
        )
        del data_0, full_0

        # pd_op.matmul: (-1x256x256xf32) <- (-1x256x96xf32, 96x256xf32)
        matmul_0 = paddle._C_ops.matmul(data_1, parameter_15, False, False)
        del parameter_15

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(data_2, parameter_14, False, False)
        del parameter_14

        # pd_op.add: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_13)
        del matmul_1, parameter_13

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (-1x1x256xf32) <- (-1x256xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(add_1, full_int_array_0)
        del add_1, full_int_array_0

        # pd_op.add: (-1x256x256xf32) <- (-1x256x256xf32, -1x1x256xf32)
        add_2 = paddle._C_ops.add(matmul_0, unsqueeze_0)
        del matmul_0, unsqueeze_0

        # pd_op.tanh: (-1x256x256xf32) <- (-1x256x256xf32)
        tanh_0 = paddle._C_ops.tanh(add_2)
        del add_2

        # pd_op.matmul: (-1x256x1xf32) <- (-1x256x256xf32, 256x1xf32)
        matmul_2 = paddle._C_ops.matmul(tanh_0, parameter_12, False, False)
        del parameter_12, tanh_0

        # pd_op.softmax: (-1x256x1xf32) <- (-1x256x1xf32)
        softmax_0 = paddle._C_ops.softmax(matmul_2, 1)
        del matmul_2

        # pd_op.transpose: (-1x1x256xf32) <- (-1x256x1xf32)
        transpose_0 = paddle._C_ops.transpose(softmax_0, [0, 2, 1])
        del softmax_0

        # pd_op.matmul: (-1x1x96xf32) <- (-1x1x256xf32, -1x256x96xf32)
        matmul_3 = paddle._C_ops.matmul(transpose_0, data_1, False, False)
        del data_1, transpose_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.squeeze: (-1x96xf32) <- (-1x1x96xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(matmul_3, full_int_array_1)
        del full_int_array_1, matmul_3

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x96xf32, -1x50xf32]) <- (-1x96xf32, -1x50xf32)
        combine_0 = [squeeze_0, one_hot_0]
        del one_hot_0, squeeze_0

        # pd_op.concat: (-1x146xf32) <- ([-1x96xf32, -1x50xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)
        del combine_0, full_1

        # pd_op.matmul: (-1x768xf32) <- (-1x146xf32, 768x146xf32)
        matmul_4 = paddle._C_ops.matmul(concat_0, parameter_11, False, True)
        del concat_0, parameter_11

        # pd_op.add: (-1x768xf32) <- (-1x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_4, parameter_10)
        del matmul_4, parameter_10

        # pd_op.matmul: (-1x768xf32) <- (-1x256xf32, 768x256xf32)
        matmul_5 = paddle._C_ops.matmul(data_2, parameter_9, False, True)
        del parameter_9

        # pd_op.add: (-1x768xf32) <- (-1x768xf32, 768xf32)
        add_4 = paddle._C_ops.add(matmul_5, parameter_8)
        del matmul_5, parameter_8

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x768xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add_3, 3, full_2)
        del add_3, full_2

        # builtin.split: (-1x256xf32, -1x256xf32, -1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32])
        (
            split_0,
            split_1,
            split_2,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x768xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(add_4, 3, full_3)
        del add_4, full_3

        # builtin.split: (-1x256xf32, -1x256xf32, -1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32])
        (
            split_3,
            split_4,
            split_5,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.add: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add_5 = paddle._C_ops.add(split_0, split_3)
        del split_0, split_3

        # pd_op.sigmoid: (-1x256xf32) <- (-1x256xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_5)
        del add_5

        # pd_op.add: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add_6 = paddle._C_ops.add(split_1, split_4)
        del split_1, split_4

        # pd_op.sigmoid: (-1x256xf32) <- (-1x256xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_6)
        del add_6

        # pd_op.multiply: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply_0 = paddle._C_ops.multiply(sigmoid_1, split_5)
        del sigmoid_1, split_5

        # pd_op.add: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add_7 = paddle._C_ops.add(split_2, multiply_0)
        del multiply_0, split_2

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_1 = paddle._C_ops.tanh(add_7)
        del add_7

        # pd_op.subtract: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        subtract_0 = paddle._C_ops.subtract(data_2, tanh_1)
        del data_2

        # pd_op.multiply: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_0, sigmoid_2)
        del sigmoid_2, subtract_0

        # pd_op.add: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add_8 = paddle._C_ops.add(multiply_1, tanh_1)
        del multiply_1, tanh_1

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_6 = paddle._C_ops.matmul(add_8, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add_9 = paddle._C_ops.add(matmul_6, parameter_6)
        del matmul_6, parameter_6

        # pd_op.matmul: (-1x50xf32) <- (-1x256xf32, 256x50xf32)
        matmul_7 = paddle._C_ops.matmul(add_9, parameter_5, False, False)
        del add_9, parameter_5

        # pd_op.add: (-1x50xf32) <- (-1x50xf32, 50xf32)
        add_0 = paddle._C_ops.add(matmul_7, parameter_4)
        del matmul_7, parameter_4

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_8 = paddle._C_ops.matmul(add_8, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add_10 = paddle._C_ops.add(matmul_8, parameter_2)
        del matmul_8, parameter_2

        # pd_op.matmul: (-1x8xf32) <- (-1x256xf32, 256x8xf32)
        matmul_9 = paddle._C_ops.matmul(add_10, parameter_1, False, False)
        del add_10, parameter_1

        # pd_op.add: (-1x8xf32) <- (-1x8xf32, 8xf32)
        add_11 = paddle._C_ops.add(matmul_9, parameter_0)
        del matmul_9, parameter_0

        # pd_op.sigmoid: (-1x8xf32) <- (-1x8xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_11)
        del add_11, add_8

        return add_0, sigmoid_0
