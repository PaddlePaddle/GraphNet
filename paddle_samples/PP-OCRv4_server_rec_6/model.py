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
        parameter_52,
        parameter_53,
        parameter_54,
        parameter_55,
        parameter_56,
        parameter_57,
        parameter_58,
        parameter_59,
        parameter_60,
        parameter_61,
        parameter_62,
        parameter_63,
        parameter_64,
        parameter_65,
        parameter_66,
        parameter_67,
        parameter_68,
        parameter_69,
        parameter_70,
        parameter_71,
        parameter_72,
        parameter_73,
        parameter_74,
        parameter_75,
        parameter_76,
        parameter_77,
        parameter_78,
        parameter_79,
        parameter_80,
        parameter_81,
        parameter_82,
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.flatten: (-1x1024x40xf32) <- (-1x1024x1x40xf32)
        flatten_0 = paddle._C_ops.flatten(data_0, 2, 3)

        # pd_op.transpose: (-1x40x1024xf32) <- (-1x1024x40xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.matmul: (-1x40x384xf32) <- (-1x40x1024xf32, 1024x384xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_0, parameter_82, False, False)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_1 = []

        # pd_op.max: (xi64) <- (-1xi64, 0xi64)
        max_0 = paddle._C_ops.max(data_2, full_int_array_1, False)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_4 = paddle._C_ops.scale(max_0, full_3, float("2"), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_32 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_2

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [scale_4]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.slice: (-1x-1xi64) <- (-1x25xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(data_1, [1], full_int_array_2, stack_0, [-1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.slice: (-1x-1xi64) <- (-1x-1xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            slice_14, [1], full_int_array_2, full_int_array_3, [1], []
        )

        # pd_op.embedding: (-1x-1x384xf32) <- (-1x-1xi64, 6629x384xf32)
        embedding_0 = paddle._C_ops.embedding(slice_0, parameter_81, 0, False)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("19.5959"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1x384xf32) <- (-1x-1x384xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(embedding_0, full_0, float("0"), True)

        # pd_op.transpose: (-1x-1x384xf32) <- (-1x-1x384xf32)
        transpose_1 = paddle._C_ops.transpose(scale_5, [1, 0, 2])

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_0 = paddle._C_ops.shape64(transpose_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_4

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_60 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_5

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [slice_15]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.slice: (-1x1x384xf32) <- (5000x1x384xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(data_3, [0], full_int_array_2, stack_1, [-1], [])

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x1x384xf32)
        add_24 = paddle._C_ops.add(transpose_1, slice_1)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_65 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_64 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_63 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_62 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_56 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_48 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_47 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_46 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_45 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_39 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_31 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_30 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_28 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_22 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_1

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_37, dropout_0 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_24, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.transpose: (-1x-1x384xf32) <- (-1x-1x384xf32)
        transpose_2 = paddle._C_ops.transpose(dropout_37, [1, 0, 2])

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_1 = paddle._C_ops.shape64(transpose_2)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_2 = paddle._C_ops.shape64(transpose_2)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_2 = [slice_18, slice_18]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_4, stack_2, paddle.float32
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_3 = [slice_18, slice_18]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("-inf"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi64)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(
            full_5, stack_3, paddle.float32
        )

        # pd_op.triu: (-1x-1xf32) <- (-1x-1xf32)
        triu_0 = paddle._C_ops.triu(full_with_tensor_1, 1)

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        add_25 = paddle._C_ops.add(full_with_tensor_0, triu_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [0, 1]

        # pd_op.unsqueeze: (1x1x-1x-1xf32) <- (-1x-1xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(add_25, full_int_array_6)

        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x384xf32, 384x1152xf32)
        matmul_1 = paddle._C_ops.matmul(transpose_2, parameter_80, False, False)

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_0 = paddle._C_ops.add(matmul_1, parameter_79)

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_4 = [full_6, slice_18, full_7, full_8, full_9]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)

        # pd_op.reshape: (-1x-1x3x12x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_0, stack_4)

        # pd_op.transpose: (3x-1x12x-1x32xf32) <- (-1x-1x3x12x32xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_0

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_5, full_int_array_0, [1], [0]
        )

        # pd_op.transpose: (-1x12x32x-1xf32) <- (-1x12x-1x32xf32)
        transpose_4 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x-1x-1xf32) <- (-1x12x-1x32xf32, -1x12x32x-1xf32)
        matmul_30 = paddle._C_ops.matmul(slice_2, transpose_4, False, False)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_61 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_55 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_44 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_38 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_27 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_21 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_2

        # pd_op.scale: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_30, full_2, float("0"), True)

        # pd_op.add: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_26 = paddle._C_ops.add(scale_0, unsqueeze_0)

        # pd_op.softmax: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(add_26, -1)

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x-1xf32, -1x12x-1x32xf32)
        matmul_31 = paddle._C_ops.matmul(softmax_0, slice_3, False, False)

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_31, [0, 2, 1, 3])

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [full_6, slice_18, full_10]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_5, stack_5)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_2 = paddle._C_ops.matmul(reshape_0, parameter_78, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_27 = paddle._C_ops.add(matmul_2, parameter_77)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_1, dropout_2 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_27, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_1 = paddle._C_ops.add(transpose_2, dropout_1)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_76, parameter_75, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_4 = paddle._C_ops.shape64(layer_norm_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.shape64: (3xi64) <- (-1x40x384xf32)
        shape64_5 = paddle._C_ops.shape64(matmul_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_3 = paddle._C_ops.matmul(layer_norm_0, parameter_74, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_2 = paddle._C_ops.add(matmul_3, parameter_73)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [full_6, slice_21, full_8, full_9]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)

        # pd_op.reshape: (-1x-1x12x32xf32) <- (-1x-1x384xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_2, stack_6)

        # pd_op.transpose: (-1x12x-1x32xf32) <- (-1x-1x12x32xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])

        # pd_op.matmul: (-1x40x768xf32) <- (-1x40x384xf32, 384x768xf32)
        matmul_4 = paddle._C_ops.matmul(matmul_0, parameter_72, False, False)

        # pd_op.add: (-1x40x768xf32) <- (-1x40x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_4, parameter_71)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_7 = [0, 40, 2, 12, 32]

        # pd_op.reshape: (-1x40x2x12x32xf32) <- (-1x40x768xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_3, full_int_array_7)

        # pd_op.transpose: (2x-1x12x40x32xf32) <- (-1x40x2x12x32xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x40x32xf32) <- (2x-1x12x40x32xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (-1x12x40x32xf32) <- (2x-1x12x40x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (-1x12x32x40xf32) <- (-1x12x40x32xf32)
        transpose_8 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x-1x40xf32) <- (-1x12x-1x32xf32, -1x12x32x40xf32)
        matmul_32 = paddle._C_ops.matmul(transpose_6, transpose_8, False, False)

        # pd_op.scale: (-1x12x-1x40xf32) <- (-1x12x-1x40xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_32, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x-1x40xf32) <- (-1x12x-1x40xf32)
        softmax_1 = paddle._C_ops.softmax(scale_6, -1)

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x40xf32, -1x12x40x32xf32)
        matmul_33 = paddle._C_ops.matmul(softmax_1, slice_4, False, False)

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_33, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_7 = [full_6, slice_21, full_10]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_9, stack_7)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_1, parameter_70, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_28 = paddle._C_ops.add(matmul_5, parameter_69)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_3, dropout_4 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_28, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_4 = paddle._C_ops.add(layer_norm_0, dropout_3)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_68, parameter_67, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_66, False, False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_29 = paddle._C_ops.add(matmul_6, parameter_65)

        # pd_op.relu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        relu_0 = paddle._C_ops.relu(add_29)

        # pd_op.dropout: (-1x-1x1536xf32, -1x-1x1536xui8) <- (-1x-1x1536xf32, None, 1xf32)
        dropout_5, dropout_6 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_0, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_7 = paddle._C_ops.matmul(dropout_5, parameter_64, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_30 = paddle._C_ops.add(matmul_7, parameter_63)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_38, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_30, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_38, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_5 = paddle._C_ops.add(layer_norm_3, dropout_8)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_62, parameter_61, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_6 = paddle._C_ops.shape64(layer_norm_6)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_7 = paddle._C_ops.shape64(layer_norm_6)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x384xf32, 384x1152xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_60, False, False)

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_6 = paddle._C_ops.add(matmul_8, parameter_59)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_8 = [full_6, slice_25, full_7, full_8, full_9]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)

        # pd_op.reshape: (-1x-1x3x12x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_11 = paddle._C_ops.reshape(add_6, stack_8)

        # pd_op.transpose: (3x-1x12x-1x32xf32) <- (-1x-1x3x12x32xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_11, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_5, full_int_array_0, [1], [0]
        )

        # pd_op.transpose: (-1x12x32x-1xf32) <- (-1x12x-1x32xf32)
        transpose_11 = paddle._C_ops.transpose(slice_26, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x-1x-1xf32) <- (-1x12x-1x32xf32, -1x12x32x-1xf32)
        matmul_34 = paddle._C_ops.matmul(slice_5, transpose_11, False, False)

        # pd_op.scale: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_34, full_2, float("0"), True)

        # pd_op.add: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_31 = paddle._C_ops.add(scale_1, unsqueeze_0)

        # pd_op.softmax: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(add_31, -1)

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x-1xf32, -1x12x-1x32xf32)
        matmul_35 = paddle._C_ops.matmul(softmax_2, slice_6, False, False)

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_35, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [full_6, slice_25, full_10]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_12, stack_9)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_2, parameter_58, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_32 = paddle._C_ops.add(matmul_9, parameter_57)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_32, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_7 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_56, parameter_55, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_8 = paddle._C_ops.shape64(layer_norm_9)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_9 = paddle._C_ops.shape64(layer_norm_9)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_9, parameter_54, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_8 = paddle._C_ops.add(matmul_10, parameter_53)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_10 = [full_6, slice_28, full_8, full_9]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)

        # pd_op.reshape: (-1x-1x12x32xf32) <- (-1x-1x384xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_8, stack_10)

        # pd_op.transpose: (-1x12x-1x32xf32) <- (-1x-1x12x32xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])

        # pd_op.matmul: (-1x40x768xf32) <- (-1x40x384xf32, 384x768xf32)
        matmul_11 = paddle._C_ops.matmul(matmul_0, parameter_52, False, False)

        # pd_op.add: (-1x40x768xf32) <- (-1x40x768xf32, 768xf32)
        add_9 = paddle._C_ops.add(matmul_11, parameter_51)

        # pd_op.reshape: (-1x40x2x12x32xf32) <- (-1x40x768xf32, 5xi64)
        reshape_13 = paddle._C_ops.reshape(add_9, full_int_array_7)

        # pd_op.transpose: (2x-1x12x40x32xf32) <- (-1x40x2x12x32xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_13, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x40x32xf32) <- (2x-1x12x40x32xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (-1x12x40x32xf32) <- (2x-1x12x40x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (-1x12x32x40xf32) <- (-1x12x40x32xf32)
        transpose_15 = paddle._C_ops.transpose(slice_29, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x-1x40xf32) <- (-1x12x-1x32xf32, -1x12x32x40xf32)
        matmul_36 = paddle._C_ops.matmul(transpose_13, transpose_15, False, False)

        # pd_op.scale: (-1x12x-1x40xf32) <- (-1x12x-1x40xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_36, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x-1x40xf32) <- (-1x12x-1x40xf32)
        softmax_3 = paddle._C_ops.softmax(scale_7, -1)

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x40xf32, -1x12x40x32xf32)
        matmul_37 = paddle._C_ops.matmul(softmax_3, slice_7, False, False)

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_16 = paddle._C_ops.transpose(matmul_37, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [full_6, slice_28, full_10]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_16, stack_11)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_12 = paddle._C_ops.matmul(reshape_3, parameter_50, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_33 = paddle._C_ops.add(matmul_12, parameter_49)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_33, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_10 = paddle._C_ops.add(layer_norm_9, dropout_12)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_48, parameter_47, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_13 = paddle._C_ops.matmul(layer_norm_12, parameter_46, False, False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_34 = paddle._C_ops.add(matmul_13, parameter_45)

        # pd_op.relu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        relu_1 = paddle._C_ops.relu(add_34)

        # pd_op.dropout: (-1x-1x1536xf32, -1x-1x1536xui8) <- (-1x-1x1536xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_1, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_14 = paddle._C_ops.matmul(dropout_14, parameter_44, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_35 = paddle._C_ops.add(matmul_14, parameter_43)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_39, dropout_16 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_35, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_17, dropout_18 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_39, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_11 = paddle._C_ops.add(layer_norm_12, dropout_17)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_42, parameter_41, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_10 = paddle._C_ops.shape64(layer_norm_15)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_11 = paddle._C_ops.shape64(layer_norm_15)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x384xf32, 384x1152xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_15, parameter_40, False, False)

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_12 = paddle._C_ops.add(matmul_15, parameter_39)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_12 = [full_6, slice_31, full_7, full_8, full_9]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_12, 0)

        # pd_op.reshape: (-1x-1x3x12x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_14 = paddle._C_ops.reshape(add_12, stack_12)

        # pd_op.transpose: (3x-1x12x-1x32xf32) <- (-1x-1x3x12x32xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_14, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_17, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_17, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_17, [0], full_int_array_5, full_int_array_0, [1], [0]
        )

        # pd_op.transpose: (-1x12x32x-1xf32) <- (-1x12x-1x32xf32)
        transpose_18 = paddle._C_ops.transpose(slice_32, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x-1x-1xf32) <- (-1x12x-1x32xf32, -1x12x32x-1xf32)
        matmul_38 = paddle._C_ops.matmul(slice_8, transpose_18, False, False)

        # pd_op.scale: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_38, full_2, float("0"), True)

        # pd_op.add: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_36 = paddle._C_ops.add(scale_2, unsqueeze_0)

        # pd_op.softmax: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32)
        softmax_4 = paddle._C_ops.softmax(add_36, -1)

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x-1xf32, -1x12x-1x32xf32)
        matmul_39 = paddle._C_ops.matmul(softmax_4, slice_9, False, False)

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_39, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_13 = [full_6, slice_31, full_10]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_13, 0)

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_19, stack_13)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_4, parameter_38, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_37 = paddle._C_ops.add(matmul_16, parameter_37)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_19, dropout_20 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_37, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_13 = paddle._C_ops.add(layer_norm_15, dropout_19)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_36, parameter_35, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_12 = paddle._C_ops.shape64(layer_norm_18)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_13 = paddle._C_ops.shape64(layer_norm_18)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_18, parameter_34, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_14 = paddle._C_ops.add(matmul_17, parameter_33)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_14 = [full_6, slice_34, full_8, full_9]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_14, 0)

        # pd_op.reshape: (-1x-1x12x32xf32) <- (-1x-1x384xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(add_14, stack_14)

        # pd_op.transpose: (-1x12x-1x32xf32) <- (-1x-1x12x32xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_15, [0, 2, 1, 3])

        # pd_op.matmul: (-1x40x768xf32) <- (-1x40x384xf32, 384x768xf32)
        matmul_18 = paddle._C_ops.matmul(matmul_0, parameter_32, False, False)

        # pd_op.add: (-1x40x768xf32) <- (-1x40x768xf32, 768xf32)
        add_15 = paddle._C_ops.add(matmul_18, parameter_31)

        # pd_op.reshape: (-1x40x2x12x32xf32) <- (-1x40x768xf32, 5xi64)
        reshape_16 = paddle._C_ops.reshape(add_15, full_int_array_7)

        # pd_op.transpose: (2x-1x12x40x32xf32) <- (-1x40x2x12x32xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_16, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x40x32xf32) <- (2x-1x12x40x32xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            transpose_21, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (-1x12x40x32xf32) <- (2x-1x12x40x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_21, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (-1x12x32x40xf32) <- (-1x12x40x32xf32)
        transpose_22 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x-1x40xf32) <- (-1x12x-1x32xf32, -1x12x32x40xf32)
        matmul_40 = paddle._C_ops.matmul(transpose_20, transpose_22, False, False)

        # pd_op.scale: (-1x12x-1x40xf32) <- (-1x12x-1x40xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_40, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x-1x40xf32) <- (-1x12x-1x40xf32)
        softmax_5 = paddle._C_ops.softmax(scale_8, -1)

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x40xf32, -1x12x40x32xf32)
        matmul_41 = paddle._C_ops.matmul(softmax_5, slice_10, False, False)

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_41, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_15 = [full_6, slice_34, full_10]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_15, 0)

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_23, stack_15)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_5, parameter_30, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_38 = paddle._C_ops.add(matmul_19, parameter_29)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_21, dropout_22 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_38, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_16 = paddle._C_ops.add(layer_norm_18, dropout_21)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_28, parameter_27, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_21, parameter_26, False, False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_39 = paddle._C_ops.add(matmul_20, parameter_25)

        # pd_op.relu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        relu_2 = paddle._C_ops.relu(add_39)

        # pd_op.dropout: (-1x-1x1536xf32, -1x-1x1536xui8) <- (-1x-1x1536xf32, None, 1xf32)
        dropout_23, dropout_24 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_2, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_21 = paddle._C_ops.matmul(dropout_23, parameter_24, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_40 = paddle._C_ops.add(matmul_21, parameter_23)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_40, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_40, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_40, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_17 = paddle._C_ops.add(layer_norm_21, dropout_26)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_22, parameter_21, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_14 = paddle._C_ops.shape64(layer_norm_24)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_15 = paddle._C_ops.shape64(layer_norm_24)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x384xf32, 384x1152xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_24, parameter_20, False, False)

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_18 = paddle._C_ops.add(matmul_22, parameter_19)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_16 = [full_6, slice_37, full_7, full_8, full_9]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_16, 0)

        # pd_op.reshape: (-1x-1x3x12x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_17 = paddle._C_ops.reshape(add_18, stack_16)

        # pd_op.transpose: (3x-1x12x-1x32xf32) <- (-1x-1x3x12x32xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_17, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_24, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            transpose_24, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_24, [0], full_int_array_5, full_int_array_0, [1], [0]
        )

        # pd_op.transpose: (-1x12x32x-1xf32) <- (-1x12x-1x32xf32)
        transpose_25 = paddle._C_ops.transpose(slice_38, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x-1x-1xf32) <- (-1x12x-1x32xf32, -1x12x32x-1xf32)
        matmul_42 = paddle._C_ops.matmul(slice_11, transpose_25, False, False)

        # pd_op.scale: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_42, full_2, float("0"), True)

        # pd_op.add: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_41 = paddle._C_ops.add(scale_3, unsqueeze_0)

        # pd_op.softmax: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32)
        softmax_6 = paddle._C_ops.softmax(add_41, -1)

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x-1xf32, -1x12x-1x32xf32)
        matmul_43 = paddle._C_ops.matmul(softmax_6, slice_12, False, False)

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_26 = paddle._C_ops.transpose(matmul_43, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_17 = [full_6, slice_37, full_10]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_17, 0)

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_26, stack_17)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_23 = paddle._C_ops.matmul(reshape_6, parameter_18, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_42 = paddle._C_ops.add(matmul_23, parameter_17)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_42, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_19 = paddle._C_ops.add(layer_norm_24, dropout_28)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_16, parameter_15, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_16 = paddle._C_ops.shape64(layer_norm_27)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            shape64_16, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_17 = paddle._C_ops.shape64(layer_norm_27)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            shape64_17, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_27, parameter_14, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_20 = paddle._C_ops.add(matmul_24, parameter_13)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_18 = [full_6, slice_40, full_8, full_9]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_18 = paddle._C_ops.stack(combine_18, 0)

        # pd_op.reshape: (-1x-1x12x32xf32) <- (-1x-1x384xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_20, stack_18)

        # pd_op.transpose: (-1x12x-1x32xf32) <- (-1x-1x12x32xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])

        # pd_op.matmul: (-1x40x768xf32) <- (-1x40x384xf32, 384x768xf32)
        matmul_25 = paddle._C_ops.matmul(matmul_0, parameter_12, False, False)

        # pd_op.add: (-1x40x768xf32) <- (-1x40x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_25, parameter_11)

        # pd_op.reshape: (-1x40x2x12x32xf32) <- (-1x40x768xf32, 5xi64)
        reshape_19 = paddle._C_ops.reshape(add_21, full_int_array_7)

        # pd_op.transpose: (2x-1x12x40x32xf32) <- (-1x40x2x12x32xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_19, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x40x32xf32) <- (2x-1x12x40x32xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (-1x12x40x32xf32) <- (2x-1x12x40x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (-1x12x32x40xf32) <- (-1x12x40x32xf32)
        transpose_29 = paddle._C_ops.transpose(slice_41, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x-1x40xf32) <- (-1x12x-1x32xf32, -1x12x32x40xf32)
        matmul_44 = paddle._C_ops.matmul(transpose_27, transpose_29, False, False)

        # pd_op.scale: (-1x12x-1x40xf32) <- (-1x12x-1x40xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_44, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x-1x40xf32) <- (-1x12x-1x40xf32)
        softmax_7 = paddle._C_ops.softmax(scale_9, -1)

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x40xf32, -1x12x40x32xf32)
        matmul_45 = paddle._C_ops.matmul(softmax_7, slice_13, False, False)

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_19 = [full_6, slice_40, full_10]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_19 = paddle._C_ops.stack(combine_19, 0)

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_30, stack_19)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_26 = paddle._C_ops.matmul(reshape_7, parameter_10, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_43 = paddle._C_ops.add(matmul_26, parameter_9)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_43, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_22 = paddle._C_ops.add(layer_norm_27, dropout_30)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_8, parameter_7, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_27 = paddle._C_ops.matmul(layer_norm_30, parameter_6, False, False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_44 = paddle._C_ops.add(matmul_27, parameter_5)

        # pd_op.relu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        relu_3 = paddle._C_ops.relu(add_44)

        # pd_op.dropout: (-1x-1x1536xf32, -1x-1x1536xui8) <- (-1x-1x1536xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_3, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_32, parameter_4, False, False)

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_45 = paddle._C_ops.add(matmul_28, parameter_3)

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_41, dropout_34 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_45, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.dropout: (-1x-1x384xf32, -1x-1x384xui8) <- (-1x-1x384xf32, None, 1xf32)
        dropout_35, dropout_36 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_41, None, full_1, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_23 = paddle._C_ops.add(layer_norm_30, dropout_35)

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_2, parameter_1, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x6629xf32) <- (-1x-1x384xf32, 384x6629xf32)
        matmul_29 = paddle._C_ops.matmul(layer_norm_33, parameter_0, False, False)
        return (
            transpose_0,
            matmul_0,
            slice_0,
            full_0,
            transpose_1,
            slice_1,
            full_1,
            dropout_0,
            transpose_2,
            unsqueeze_0,
            matmul_1,
            add_0,
            transpose_3,
            assign_0,
            assign_1,
            slice_2,
            assign_2,
            assign_3,
            assign_4,
            full_int_array_0,
            slice_3,
            transpose_4,
            full_2,
            scale_0,
            softmax_0,
            transpose_5,
            reshape_0,
            matmul_2,
            assign_5,
            dropout_1,
            dropout_2,
            add_1,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_3,
            add_2,
            transpose_6,
            matmul_4,
            add_3,
            transpose_7,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            slice_4,
            transpose_8,
            assign_10,
            softmax_1,
            transpose_9,
            reshape_1,
            matmul_5,
            assign_11,
            dropout_3,
            dropout_4,
            add_4,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            matmul_6,
            relu_0,
            assign_12,
            dropout_5,
            dropout_6,
            matmul_7,
            assign_13,
            dropout_7,
            assign_14,
            dropout_8,
            dropout_9,
            add_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            matmul_8,
            add_6,
            transpose_10,
            assign_15,
            assign_16,
            slice_5,
            assign_17,
            assign_18,
            assign_19,
            assign_20,
            slice_6,
            transpose_11,
            assign_21,
            scale_1,
            softmax_2,
            transpose_12,
            reshape_2,
            matmul_9,
            assign_22,
            dropout_10,
            dropout_11,
            add_7,
            layer_norm_9,
            layer_norm_10,
            layer_norm_11,
            matmul_10,
            add_8,
            transpose_13,
            matmul_11,
            add_9,
            transpose_14,
            assign_23,
            assign_24,
            assign_25,
            assign_26,
            slice_7,
            transpose_15,
            assign_27,
            softmax_3,
            transpose_16,
            reshape_3,
            matmul_12,
            assign_28,
            dropout_12,
            dropout_13,
            add_10,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            matmul_13,
            relu_1,
            assign_29,
            dropout_14,
            dropout_15,
            matmul_14,
            assign_30,
            dropout_16,
            assign_31,
            dropout_17,
            dropout_18,
            add_11,
            layer_norm_15,
            layer_norm_16,
            layer_norm_17,
            matmul_15,
            add_12,
            transpose_17,
            assign_32,
            assign_33,
            slice_8,
            assign_34,
            assign_35,
            assign_36,
            assign_37,
            slice_9,
            transpose_18,
            assign_38,
            scale_2,
            softmax_4,
            transpose_19,
            reshape_4,
            matmul_16,
            assign_39,
            dropout_19,
            dropout_20,
            add_13,
            layer_norm_18,
            layer_norm_19,
            layer_norm_20,
            matmul_17,
            add_14,
            transpose_20,
            matmul_18,
            add_15,
            transpose_21,
            assign_40,
            assign_41,
            assign_42,
            assign_43,
            slice_10,
            transpose_22,
            assign_44,
            softmax_5,
            transpose_23,
            reshape_5,
            matmul_19,
            assign_45,
            dropout_21,
            dropout_22,
            add_16,
            layer_norm_21,
            layer_norm_22,
            layer_norm_23,
            matmul_20,
            relu_2,
            assign_46,
            dropout_23,
            dropout_24,
            matmul_21,
            assign_47,
            dropout_25,
            assign_48,
            dropout_26,
            dropout_27,
            add_17,
            layer_norm_24,
            layer_norm_25,
            layer_norm_26,
            matmul_22,
            add_18,
            transpose_24,
            assign_49,
            assign_50,
            slice_11,
            assign_51,
            assign_52,
            assign_53,
            assign_54,
            slice_12,
            transpose_25,
            assign_55,
            scale_3,
            softmax_6,
            transpose_26,
            reshape_6,
            matmul_23,
            assign_56,
            dropout_28,
            dropout_29,
            add_19,
            layer_norm_27,
            layer_norm_28,
            layer_norm_29,
            matmul_24,
            add_20,
            transpose_27,
            matmul_25,
            add_21,
            transpose_28,
            assign_57,
            assign_58,
            assign_59,
            assign_60,
            slice_13,
            transpose_29,
            assign_61,
            softmax_7,
            transpose_30,
            reshape_7,
            matmul_26,
            assign_62,
            dropout_30,
            dropout_31,
            add_22,
            layer_norm_30,
            layer_norm_31,
            layer_norm_32,
            matmul_27,
            relu_3,
            assign_63,
            dropout_32,
            dropout_33,
            matmul_28,
            assign_64,
            dropout_34,
            assign_65,
            dropout_35,
            dropout_36,
            add_23,
            layer_norm_33,
            layer_norm_34,
            layer_norm_35,
            matmul_29,
        )
