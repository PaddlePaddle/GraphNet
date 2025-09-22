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
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # builtin.combine: ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32]) <- (512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32)
        combine_0 = [data_0, data_1, data_2, data_3]
        del data_0, data_1, data_2, data_3

        # pd_op.concat: (2048x4xf32) <- ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.assign: (2048x4xf32) <- (2048x4xf32)
        assign_4 = concat_0

        # pd_op.distribute_fpn_proposals: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], [-1xi32, -1xi32, -1xi32, -1xi32], -1x1xi32) <- (2048x4xf32, 4xi64)
        (
            distribute_fpn_proposals_0,
            distribute_fpn_proposals_1,
            distribute_fpn_proposals_2,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.distribute_fpn_proposals(
                concat_0, data_4, 2, 5, 4, 224, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_4

        # builtin.split: (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = distribute_fpn_proposals_0
        del distribute_fpn_proposals_0

        # builtin.split: (-1xi32, -1xi32, -1xi32, -1xi32) <- ([-1xi32, -1xi32, -1xi32, -1xi32])
        (
            split_8,
            split_9,
            split_10,
            split_11,
        ) = distribute_fpn_proposals_1
        del distribute_fpn_proposals_1

        # pd_op.roi_align: (-1x256x7x7xf32) <- (4x256x-1x-1xf32, -1x4xf32, -1xi32)
        roi_align_0 = paddle._C_ops.roi_align(
            data_5, split_4, split_8, 7, 7, float("0.25"), 0, True
        )
        del data_5

        # pd_op.roi_align: (-1x256x7x7xf32) <- (4x256x-1x-1xf32, -1x4xf32, -1xi32)
        roi_align_1 = paddle._C_ops.roi_align(
            data_6, split_5, split_9, 7, 7, float("0.125"), 0, True
        )
        del data_6

        # pd_op.roi_align: (-1x256x7x7xf32) <- (4x256x-1x-1xf32, -1x4xf32, -1xi32)
        roi_align_2 = paddle._C_ops.roi_align(
            data_7, split_6, split_10, 7, 7, float("0.0625"), 0, True
        )
        del data_7

        # pd_op.roi_align: (-1x256x7x7xf32) <- (4x256x-1x-1xf32, -1x4xf32, -1xi32)
        roi_align_3 = paddle._C_ops.roi_align(
            data_8, split_7, split_11, 7, 7, float("0.03125"), 0, True
        )
        del data_8

        # builtin.combine: ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32]) <- (-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32)
        combine_1 = [roi_align_0, roi_align_1, roi_align_2, roi_align_3]

        # pd_op.concat: (-1x256x7x7xf32) <- ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.gather: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(concat_1, distribute_fpn_proposals_2, full_0)

        # pd_op.flatten: (-1x12544xf32) <- (-1x256x7x7xf32)
        flatten_0 = paddle._C_ops.flatten(gather_0, 1, 3)

        # pd_op.matmul: (-1x1024xf32) <- (-1x12544xf32, 12544x1024xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_6)
        del parameter_6

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_0 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_4)
        del parameter_4

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_1 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.matmul: (-1x5xf32) <- (-1x1024xf32, 1024x5xf32)
        matmul_2 = paddle._C_ops.matmul(relu_1, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (-1x5xf32) <- (-1x5xf32, 5xf32)
        add_0 = paddle._C_ops.add(matmul_2, parameter_2)
        del parameter_2

        # pd_op.matmul: (-1x4xf32) <- (-1x1024xf32, 1024x4xf32)
        matmul_3 = paddle._C_ops.matmul(relu_1, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (-1x4xf32) <- (-1x4xf32, 4xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_0)
        del parameter_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_1

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_3

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.assign: (2048xf32) <- (2048xf32)
        assign_22 = slice_1

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_4

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (2048xf32) <- (2048x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del concat_0

        # pd_op.assign: (2048xf32) <- (2048xf32)
        assign_27 = slice_3

        # pd_op.subtract: (2048xf32) <- (2048xf32, 2048xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_28 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_30 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_31 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_32 = full_1

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_1, float("0"), True)

        # pd_op.add: (2048xf32) <- (2048xf32, 2048xf32)
        add_4 = paddle._C_ops.add(slice_1, scale_0)

        # pd_op.scale: (2048xf32) <- (2048xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_1, full_1, float("0"), True)

        # pd_op.add: (2048xf32) <- (2048xf32, 2048xf32)
        add_5 = paddle._C_ops.add(slice_3, scale_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_5

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            add_3, [1], full_int_array_2, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.0333333"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_36 = full_2

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(strided_slice_0, full_2, float("0"), True)
        del strided_slice_0

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            add_3, [1], full_int_array_3, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(strided_slice_1, full_2, float("0"), True)
        del strided_slice_1

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            add_3, [1], full_int_array_0, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.0666667"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_37 = full_3

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(strided_slice_2, full_3, float("0"), True)
        del strided_slice_2

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            add_3, [1], full_int_array_1, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(strided_slice_3, full_3, float("0"), True)
        del strided_slice_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_38 = full_4

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("4.13517"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_39 = full_5

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(scale_4, full_4, full_5)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(scale_5, full_4, full_5)

        # pd_op.unsqueeze: (2048x1xf32) <- (2048xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(subtract_0, full_int_array_3)

        # pd_op.assign: (2048x1xf32) <- (2048x1xf32)
        assign_40 = unsqueeze_0

        # pd_op.multiply: (2048x1xf32) <- (-1x1xf32, 2048x1xf32)
        multiply_0 = paddle._C_ops.multiply(scale_2, unsqueeze_0)

        # pd_op.unsqueeze: (2048x1xf32) <- (2048xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(add_4, full_int_array_3)

        # pd_op.add: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        add_6 = paddle._C_ops.add(multiply_0, unsqueeze_1)

        # pd_op.unsqueeze: (2048x1xf32) <- (2048xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(subtract_1, full_int_array_3)

        # pd_op.assign: (2048x1xf32) <- (2048x1xf32)
        assign_41 = unsqueeze_2

        # pd_op.multiply: (2048x1xf32) <- (-1x1xf32, 2048x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_3, unsqueeze_2)

        # pd_op.unsqueeze: (2048x1xf32) <- (2048xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(add_5, full_int_array_3)

        # pd_op.add: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        add_7 = paddle._C_ops.add(multiply_1, unsqueeze_3)

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_0 = paddle._C_ops.exp(clip_0)
        del clip_0

        # pd_op.multiply: (2048x1xf32) <- (-1x1xf32, 2048x1xf32)
        multiply_2 = paddle._C_ops.multiply(exp_0, unsqueeze_0)

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_1 = paddle._C_ops.exp(clip_1)
        del clip_1

        # pd_op.multiply: (2048x1xf32) <- (-1x1xf32, 2048x1xf32)
        multiply_3 = paddle._C_ops.multiply(exp_1, unsqueeze_2)

        # pd_op.scale: (2048x1xf32) <- (2048x1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(multiply_2, full_1, float("0"), True)
        del multiply_2

        # pd_op.assign: (2048x1xf32) <- (2048x1xf32)
        assign_42 = scale_6

        # pd_op.subtract: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        subtract_2 = paddle._C_ops.subtract(add_6, scale_6)

        # pd_op.scale: (2048x1xf32) <- (2048x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(multiply_3, full_1, float("0"), True)
        del multiply_3

        # pd_op.assign: (2048x1xf32) <- (2048x1xf32)
        assign_43 = scale_7

        # pd_op.subtract: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        subtract_3 = paddle._C_ops.subtract(add_7, scale_7)

        # pd_op.add: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        add_8 = paddle._C_ops.add(add_6, scale_6)

        # pd_op.add: (2048x1xf32) <- (2048x1xf32, 2048x1xf32)
        add_9 = paddle._C_ops.add(add_7, scale_7)

        # builtin.combine: ([2048x1xf32, 2048x1xf32, 2048x1xf32, 2048x1xf32]) <- (2048x1xf32, 2048x1xf32, 2048x1xf32, 2048x1xf32)
        combine_2 = [subtract_2, subtract_3, add_8, add_9]

        # pd_op.stack: (2048x1x4xf32) <- ([2048x1xf32, 2048x1xf32, 2048x1xf32, 2048x1xf32])
        stack_0 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_0 = paddle._C_ops.shape64(add_3)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [-1, 4]

        # pd_op.reshape: (2048x4xf32) <- (2048x1x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_6)
        del full_int_array_6

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [512, 512, 512, 512]

        # pd_op.split: ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32]) <- (2048x4xf32, 4xi64, 1xi32)
        split_12 = paddle._C_ops.split(reshape_0, full_int_array_7, full_0)
        del full_0, full_int_array_7, reshape_0

        # builtin.split: (512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32) <- ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_12
        del (
            add_3,
            add_4,
            add_5,
            add_6,
            add_7,
            add_8,
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
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            concat_1,
            distribute_fpn_proposals_2,
            exp_0,
            exp_1,
            flatten_0,
            full_1,
            full_2,
            full_3,
            full_4,
            full_5,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            full_int_array_5,
            gather_0,
            matmul_0,
            matmul_1,
            matmul_2,
            matmul_3,
            multiply_0,
            multiply_1,
            relu_0,
            relu_1,
            roi_align_0,
            roi_align_1,
            roi_align_2,
            roi_align_3,
            scale_0,
            scale_1,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            scale_6,
            scale_7,
            slice_0,
            slice_1,
            slice_2,
            slice_3,
            split_10,
            split_11,
            split_12,
            split_4,
            split_5,
            split_6,
            split_7,
            split_8,
            split_9,
            stack_0,
            subtract_0,
            subtract_1,
            subtract_2,
            subtract_3,
            unsqueeze_0,
            unsqueeze_1,
            unsqueeze_2,
            unsqueeze_3,
        )

        return add_0, split_0, split_1, split_2, split_3
