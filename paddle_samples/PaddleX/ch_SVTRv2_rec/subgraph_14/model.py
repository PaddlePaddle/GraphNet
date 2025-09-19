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
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.flatten: (4x384x40xf32) <- (4x384x1x40xf32)
        flatten_0 = paddle._C_ops.flatten(data_0, 2, 3)
        del data_0

        # pd_op.transpose: (4x40x384xf32) <- (4x384x40xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.matmul: (4x40x384xf32) <- (4x40x384xf32, 384x384xf32)
        matmul_1 = paddle._C_ops.matmul(transpose_0, parameter_42, False, False)
        del parameter_42

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.max: (xi64) <- (4xi64, 0xi64)
        max_0 = paddle._C_ops.max(data_2, full_int_array_0, False)
        del data_2, full_int_array_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(max_0, full_0, float("2"), True)
        del full_0, max_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_1

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [scale_0]
        del scale_0

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.slice: (4x-1xi64) <- (4x25xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_1, [1], full_int_array_1, stack_0, [-1], [])
        del data_1, stack_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.slice: (4x-1xi64) <- (4x-1xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            slice_0, [1], full_int_array_1, full_int_array_2, [1], []
        )
        del full_int_array_2, slice_0

        # pd_op.embedding: (4x-1x384xf32) <- (4x-1xi64, 6629x384xf32)
        embedding_0 = paddle._C_ops.embedding(slice_1, parameter_41, 0, False)
        del parameter_41

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("19.5959"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x-1x384xf32) <- (4x-1x384xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(embedding_0, full_1, float("0"), True)
        del embedding_0

        # pd_op.transpose: (-1x4x384xf32) <- (4x-1x384xf32)
        transpose_1 = paddle._C_ops.transpose(scale_1, [1, 0, 2])
        del scale_1

        # pd_op.shape64: (3xi64) <- (-1x4x384xf32)
        shape64_0 = paddle._C_ops.shape64(transpose_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_3

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_3, [1], [0]
        )
        del shape64_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [slice_2]
        del slice_2

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.slice: (-1x1x384xf32) <- (5000x1x384xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(data_3, [0], full_int_array_1, stack_1, [-1], [])
        del data_3, stack_1

        # pd_op.add: (-1x4x384xf32) <- (-1x4x384xf32, -1x1x384xf32)
        add_0 = paddle._C_ops.add(transpose_1, slice_3)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_19 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_20 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_21 = full_2

        # pd_op.dropout: (-1x4x384xf32, -1x4x384xui8) <- (-1x4x384xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_0, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_0

        # pd_op.transpose: (4x-1x384xf32) <- (-1x4x384xf32)
        transpose_2 = paddle._C_ops.transpose(dropout_0, [1, 0, 2])
        del dropout_0

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_1 = paddle._C_ops.shape64(transpose_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_4

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_2 = [slice_4, slice_4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_3, stack_2, paddle.float32
        )
        del full_3, stack_2

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_3 = [slice_4, slice_4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-inf"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi64)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(
            full_4, stack_3, paddle.float32
        )
        del full_4, stack_3

        # pd_op.triu: (-1x-1xf32) <- (-1x-1xf32)
        triu_0 = paddle._C_ops.triu(full_with_tensor_1, 1)
        del full_with_tensor_1

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        add_1 = paddle._C_ops.add(full_with_tensor_0, triu_0)
        del full_with_tensor_0, triu_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [0, 1]

        # pd_op.unsqueeze: (1x1x-1x-1xf32) <- (-1x-1xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(add_1, full_int_array_5)
        del add_1, full_int_array_5

        # pd_op.matmul: (4x-1x1152xf32) <- (4x-1x384xf32, 384x1152xf32)
        matmul_2 = paddle._C_ops.matmul(transpose_2, parameter_40, False, False)
        del parameter_40

        # pd_op.add: (4x-1x1152xf32) <- (4x-1x1152xf32, 1152xf32)
        add_2 = paddle._C_ops.add(matmul_2, parameter_39)
        del parameter_39

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_4 = [full_5, slice_4, full_6, full_7, full_8]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.reshape: (4x-1x3x12x32xf32) <- (4x-1x1152xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, stack_4)
        del stack_4

        # pd_op.transpose: (3x4x12x-1x32xf32) <- (4x-1x3x12x32xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_6

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_4, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (4x12x32x-1xf32) <- (4x12x-1x32xf32)
        transpose_4 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])
        del slice_6

        # pd_op.matmul: (4x12x-1x-1xf32) <- (4x12x-1x32xf32, 4x12x32x-1xf32)
        matmul_3 = paddle._C_ops.matmul(slice_5, transpose_4, False, False)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_30 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_31 = full_9

        # pd_op.scale: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_3, full_9, float("0"), True)
        del matmul_3

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_3 = paddle._C_ops.add(scale_2, unsqueeze_0)

        # pd_op.softmax: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(add_3, -1)
        del add_3

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x-1xf32, 4x12x-1x32xf32)
        matmul_4 = paddle._C_ops.matmul(softmax_0, slice_7, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [full_5, slice_4, full_10]
        del slice_4

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_5, stack_5)
        del stack_5

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_1, parameter_38, False, False)
        del parameter_38

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_4 = paddle._C_ops.add(matmul_5, parameter_37)
        del parameter_37

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_4, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_4

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_5 = paddle._C_ops.add(transpose_2, dropout_2)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_36, parameter_35, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_35, parameter_36

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_2 = paddle._C_ops.shape64(layer_norm_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del shape64_2

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_0, parameter_34, False, False)
        del parameter_34

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_6 = paddle._C_ops.add(matmul_6, parameter_33)
        del parameter_33

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [full_5, slice_8, full_7, full_8]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.reshape: (4x-1x12x32xf32) <- (4x-1x384xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_6, stack_6)
        del stack_6

        # pd_op.transpose: (4x12x-1x32xf32) <- (4x-1x12x32xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (4x40x768xf32) <- (4x40x384xf32, 384x768xf32)
        matmul_7 = paddle._C_ops.matmul(matmul_1, parameter_32, False, False)
        del parameter_32

        # pd_op.add: (4x40x768xf32) <- (4x40x768xf32, 768xf32)
        add_7 = paddle._C_ops.add(matmul_7, parameter_31)
        del parameter_31

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_7 = [0, 40, 2, 12, 32]

        # pd_op.reshape: (4x40x2x12x32xf32) <- (4x40x768xf32, 5xi64)
        reshape_3 = paddle._C_ops.reshape(add_7, full_int_array_7)

        # pd_op.transpose: (2x4x12x40x32xf32) <- (4x40x2x12x32xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_3, [2, 0, 3, 1, 4])
        del reshape_3

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (4x12x32x40xf32) <- (4x12x40x32xf32)
        transpose_8 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])
        del slice_9

        # pd_op.matmul: (4x12x-1x40xf32) <- (4x12x-1x32xf32, 4x12x32x40xf32)
        matmul_8 = paddle._C_ops.matmul(transpose_6, transpose_8, False, False)

        # pd_op.scale: (4x12x-1x40xf32) <- (4x12x-1x40xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_8, full_9, float("0"), True)
        del matmul_8

        # pd_op.softmax: (4x12x-1x40xf32) <- (4x12x-1x40xf32)
        softmax_1 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x40xf32, 4x12x40x32xf32)
        matmul_9 = paddle._C_ops.matmul(softmax_1, slice_10, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_9, [0, 2, 1, 3])
        del matmul_9

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_7 = [full_5, slice_8, full_10]
        del slice_8

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_9, stack_7)
        del stack_7

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_10 = paddle._C_ops.matmul(reshape_4, parameter_30, False, False)
        del parameter_30

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_8 = paddle._C_ops.add(matmul_10, parameter_29)
        del parameter_29

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_8, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_8

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_9 = paddle._C_ops.add(layer_norm_0, dropout_4)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_28, parameter_27, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_27, parameter_28

        # pd_op.matmul: (4x-1x1536xf32) <- (4x-1x384xf32, 384x1536xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_3, parameter_26, False, False)
        del parameter_26

        # pd_op.add: (4x-1x1536xf32) <- (4x-1x1536xf32, 1536xf32)
        add_10 = paddle._C_ops.add(matmul_11, parameter_25)
        del parameter_25

        # pd_op.relu: (4x-1x1536xf32) <- (4x-1x1536xf32)
        relu_0 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.dropout: (4x-1x1536xf32, 4x-1x1536xui8) <- (4x-1x1536xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_0, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x1536xf32, 1536x384xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_6, parameter_24, False, False)
        del parameter_24

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_11 = paddle._C_ops.add(matmul_12, parameter_23)
        del parameter_23

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_11, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_11

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_8, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_8

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_12 = paddle._C_ops.add(layer_norm_3, dropout_10)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_22, parameter_21, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_21, parameter_22

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_6)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del shape64_3

        # pd_op.matmul: (4x-1x1152xf32) <- (4x-1x384xf32, 384x1152xf32)
        matmul_13 = paddle._C_ops.matmul(layer_norm_6, parameter_20, False, False)
        del parameter_20

        # pd_op.add: (4x-1x1152xf32) <- (4x-1x1152xf32, 1152xf32)
        add_13 = paddle._C_ops.add(matmul_13, parameter_19)
        del parameter_19

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_8 = [full_5, slice_11, full_6, full_7, full_8]
        del full_6

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.reshape: (4x-1x3x12x32xf32) <- (4x-1x1152xf32, 5xi64)
        reshape_5 = paddle._C_ops.reshape(add_13, stack_8)
        del stack_8

        # pd_op.transpose: (3x4x12x-1x32xf32) <- (4x-1x3x12x32xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_5, [2, 0, 3, 1, 4])
        del reshape_5

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_4, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (4x12x32x-1xf32) <- (4x12x-1x32xf32)
        transpose_11 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (4x12x-1x-1xf32) <- (4x12x-1x32xf32, 4x12x32x-1xf32)
        matmul_14 = paddle._C_ops.matmul(slice_12, transpose_11, False, False)

        # pd_op.scale: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_14, full_9, float("0"), True)
        del matmul_14

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_14 = paddle._C_ops.add(scale_4, unsqueeze_0)

        # pd_op.softmax: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(add_14, -1)
        del add_14

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x-1xf32, 4x12x-1x32xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_2, slice_14, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])
        del matmul_15

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [full_5, slice_11, full_10]
        del slice_11

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_12, stack_9)
        del stack_9

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_6, parameter_18, False, False)
        del parameter_18

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_15 = paddle._C_ops.add(matmul_16, parameter_17)
        del parameter_17

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_15, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_15

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_16 = paddle._C_ops.add(layer_norm_6, dropout_12)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_16, parameter_15, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_15, parameter_16

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_4 = paddle._C_ops.shape64(layer_norm_9)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del shape64_4

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_9, parameter_14, False, False)
        del parameter_14

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_17 = paddle._C_ops.add(matmul_17, parameter_13)
        del parameter_13

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_10 = [full_5, slice_15, full_7, full_8]
        del full_7, full_8

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.reshape: (4x-1x12x32xf32) <- (4x-1x384xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(add_17, stack_10)
        del stack_10

        # pd_op.transpose: (4x12x-1x32xf32) <- (4x-1x12x32xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_7, [0, 2, 1, 3])
        del reshape_7

        # pd_op.matmul: (4x40x768xf32) <- (4x40x384xf32, 384x768xf32)
        matmul_18 = paddle._C_ops.matmul(matmul_1, parameter_12, False, False)
        del parameter_12

        # pd_op.add: (4x40x768xf32) <- (4x40x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(matmul_18, parameter_11)
        del parameter_11

        # pd_op.reshape: (4x40x2x12x32xf32) <- (4x40x768xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_18, full_int_array_7)
        del full_int_array_7

        # pd_op.transpose: (2x4x12x40x32xf32) <- (4x40x2x12x32xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_1, full_int_array_3, [1], [0]
        )
        del full_int_array_1

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del full_int_array_3, full_int_array_4

        # pd_op.transpose: (4x12x32x40xf32) <- (4x12x40x32xf32)
        transpose_15 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])
        del slice_16

        # pd_op.matmul: (4x12x-1x40xf32) <- (4x12x-1x32xf32, 4x12x32x40xf32)
        matmul_19 = paddle._C_ops.matmul(transpose_13, transpose_15, False, False)

        # pd_op.scale: (4x12x-1x40xf32) <- (4x12x-1x40xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_19, full_9, float("0"), True)
        del matmul_19

        # pd_op.softmax: (4x12x-1x40xf32) <- (4x12x-1x40xf32)
        softmax_3 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x40xf32, 4x12x40x32xf32)
        matmul_20 = paddle._C_ops.matmul(softmax_3, slice_17, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_16 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [full_5, slice_15, full_10]
        del full_10, full_5, slice_15

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_16, stack_11)
        del stack_11

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_9, parameter_10, False, False)
        del parameter_10

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_19 = paddle._C_ops.add(matmul_21, parameter_9)
        del parameter_9

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_19, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_19

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_20 = paddle._C_ops.add(layer_norm_9, dropout_14)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_8, parameter_7, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_7, parameter_8

        # pd_op.matmul: (4x-1x1536xf32) <- (4x-1x384xf32, 384x1536xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_12, parameter_6, False, False)
        del parameter_6

        # pd_op.add: (4x-1x1536xf32) <- (4x-1x1536xf32, 1536xf32)
        add_21 = paddle._C_ops.add(matmul_22, parameter_5)
        del parameter_5

        # pd_op.relu: (4x-1x1536xf32) <- (4x-1x1536xf32)
        relu_1 = paddle._C_ops.relu(add_21)
        del add_21

        # pd_op.dropout: (4x-1x1536xf32, 4x-1x1536xui8) <- (4x-1x1536xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_1, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x1536xf32, 1536x384xf32)
        matmul_23 = paddle._C_ops.matmul(dropout_16, parameter_4, False, False)
        del parameter_4

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_22 = paddle._C_ops.add(matmul_23, parameter_3)
        del parameter_3

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_22, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_22

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_18, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_18

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_23 = paddle._C_ops.add(layer_norm_12, dropout_20)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_2, parameter_1, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_1, parameter_2

        # pd_op.matmul: (4x-1x6629xf32) <- (4x-1x384xf32, 384x6629xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_15, parameter_0, False, False)
        del (
            add_12,
            add_13,
            add_16,
            add_17,
            add_18,
            add_2,
            add_20,
            add_23,
            add_5,
            add_6,
            add_7,
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
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            dropout_1,
            dropout_10,
            dropout_11,
            dropout_12,
            dropout_13,
            dropout_14,
            dropout_15,
            dropout_16,
            dropout_17,
            dropout_19,
            dropout_2,
            dropout_20,
            dropout_21,
            dropout_3,
            dropout_4,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_9,
            full_1,
            full_2,
            full_9,
            full_int_array_6,
            layer_norm_0,
            layer_norm_1,
            layer_norm_10,
            layer_norm_11,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            layer_norm_15,
            layer_norm_16,
            layer_norm_17,
            layer_norm_2,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_13,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_2,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_5,
            matmul_6,
            matmul_7,
            parameter_0,
            relu_0,
            relu_1,
            reshape_1,
            reshape_4,
            reshape_6,
            reshape_9,
            scale_2,
            scale_4,
            slice_1,
            slice_10,
            slice_12,
            slice_14,
            slice_17,
            slice_3,
            slice_5,
            slice_7,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            transpose_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_12,
            transpose_13,
            transpose_14,
            transpose_15,
            transpose_16,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_0,
        )

        return matmul_0
