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
    ):
        # pd_op.distribute_fpn_proposals: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], [-1xi32, -1xi32, -1xi32, -1xi32], -1x1xi32) <- (512x4xf32, 1xi64)
        (
            distribute_fpn_proposals_1,
            distribute_fpn_proposals_2,
            distribute_fpn_proposals_0,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.distribute_fpn_proposals(data_0, data_1, 2, 5, 4, 224, False),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # builtin.split: (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = distribute_fpn_proposals_1

        # builtin.split: (-1xi32, -1xi32, -1xi32, -1xi32) <- ([-1xi32, -1xi32, -1xi32, -1xi32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = distribute_fpn_proposals_2

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x-1x-1xf32, -1x4xf32, -1xi32)
        roi_align_0 = paddle._C_ops.roi_align(
            data_2, split_0, split_4, 7, 7, float("0.25"), 0, True
        )

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x-1x-1xf32, -1x4xf32, -1xi32)
        roi_align_1 = paddle._C_ops.roi_align(
            data_3, split_1, split_5, 7, 7, float("0.125"), 0, True
        )

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x-1x-1xf32, -1x4xf32, -1xi32)
        roi_align_2 = paddle._C_ops.roi_align(
            data_4, split_2, split_6, 7, 7, float("0.0625"), 0, True
        )

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x-1x-1xf32, -1x4xf32, -1xi32)
        roi_align_3 = paddle._C_ops.roi_align(
            data_5, split_3, split_7, 7, 7, float("0.03125"), 0, True
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_40 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # builtin.combine: ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32]) <- (-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32)
        combine_0 = [roi_align_0, roi_align_1, roi_align_2, roi_align_3]

        # pd_op.concat: (-1x256x7x7xf32) <- ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.gather: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(concat_0, distribute_fpn_proposals_0, full_0)

        # pd_op.flatten: (-1x12544xf32) <- (-1x256x7x7xf32)
        flatten_0 = paddle._C_ops.flatten(gather_0, 1, 3)

        # pd_op.matmul: (-1x1024xf32) <- (-1x12544xf32, 12544x1024xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_7, False, False)

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_8 = paddle._C_ops.add(matmul_0, parameter_6)

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_0 = paddle._C_ops.relu(add_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_5, False, False)

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_9 = paddle._C_ops.add(matmul_1, parameter_4)

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_1 = paddle._C_ops.relu(add_9)

        # pd_op.matmul: (-1x3xf32) <- (-1x1024xf32, 1024x3xf32)
        matmul_2 = paddle._C_ops.matmul(relu_1, parameter_3, False, False)

        # pd_op.add: (-1x3xf32) <- (-1x3xf32, 3xf32)
        add_6 = paddle._C_ops.add(matmul_2, parameter_2)

        # pd_op.matmul: (-1x4xf32) <- (-1x1024xf32, 1024x4xf32)
        matmul_3 = paddle._C_ops.matmul(relu_1, parameter_1, False, False)

        # pd_op.add: (-1x4xf32) <- (-1x4xf32, 4xf32)
        add_7 = paddle._C_ops.add(matmul_3, parameter_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_32 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_29 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_3

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.assign: (512xf32) <- (512xf32)
        assign_6 = slice_1

        # pd_op.subtract: (512xf32) <- (512xf32, 512xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_4

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.assign: (512xf32) <- (512xf32)
        assign_9 = slice_3

        # pd_op.subtract: (512xf32) <- (512xf32, 512xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_38 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_36 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_35 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_34 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_1

        # pd_op.scale: (512xf32) <- (512xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_1, float("0"), True)

        # pd_op.add: (512xf32) <- (512xf32, 512xf32)
        add_0 = paddle._C_ops.add(slice_1, scale_0)

        # pd_op.scale: (512xf32) <- (512xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_1, full_1, float("0"), True)

        # pd_op.add: (512xf32) <- (512xf32, 512xf32)
        add_1 = paddle._C_ops.add(slice_3, scale_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_5

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            add_7, [1], full_int_array_2, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_2

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(strided_slice_0, full_2, float("0"), True)

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            add_7, [1], full_int_array_3, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(strided_slice_1, full_2, float("0"), True)

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            add_7, [1], full_int_array_0, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_23 = full_3

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(strided_slice_2, full_3, float("0"), True)

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            add_7, [1], full_int_array_1, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(strided_slice_3, full_3, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_24 = full_4

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("4.13517"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_25 = full_5

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(scale_4, full_4, full_5)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(scale_5, full_4, full_5)

        # pd_op.unsqueeze: (512x1xf32) <- (512xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(subtract_0, full_int_array_3)

        # pd_op.assign: (512x1xf32) <- (512x1xf32)
        assign_31 = unsqueeze_0

        # pd_op.multiply: (512x1xf32) <- (-1x1xf32, 512x1xf32)
        multiply_0 = paddle._C_ops.multiply(scale_2, unsqueeze_0)

        # pd_op.unsqueeze: (512x1xf32) <- (512xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(add_0, full_int_array_3)

        # pd_op.add: (512x1xf32) <- (512x1xf32, 512x1xf32)
        add_2 = paddle._C_ops.add(multiply_0, unsqueeze_1)

        # pd_op.unsqueeze: (512x1xf32) <- (512xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(subtract_1, full_int_array_3)

        # pd_op.assign: (512x1xf32) <- (512x1xf32)
        assign_33 = unsqueeze_2

        # pd_op.multiply: (512x1xf32) <- (-1x1xf32, 512x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_3, unsqueeze_2)

        # pd_op.unsqueeze: (512x1xf32) <- (512xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(add_1, full_int_array_3)

        # pd_op.add: (512x1xf32) <- (512x1xf32, 512x1xf32)
        add_3 = paddle._C_ops.add(multiply_1, unsqueeze_3)

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_0 = paddle._C_ops.exp(clip_0)

        # pd_op.multiply: (512x1xf32) <- (-1x1xf32, 512x1xf32)
        multiply_2 = paddle._C_ops.multiply(exp_0, unsqueeze_0)

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_1 = paddle._C_ops.exp(clip_1)

        # pd_op.multiply: (512x1xf32) <- (-1x1xf32, 512x1xf32)
        multiply_3 = paddle._C_ops.multiply(exp_1, unsqueeze_2)

        # pd_op.scale: (512x1xf32) <- (512x1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(multiply_2, full_1, float("0"), True)

        # pd_op.assign: (512x1xf32) <- (512x1xf32)
        assign_37 = scale_6

        # pd_op.subtract: (512x1xf32) <- (512x1xf32, 512x1xf32)
        subtract_2 = paddle._C_ops.subtract(add_2, scale_6)

        # pd_op.scale: (512x1xf32) <- (512x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(multiply_3, full_1, float("0"), True)

        # pd_op.assign: (512x1xf32) <- (512x1xf32)
        assign_39 = scale_7

        # pd_op.subtract: (512x1xf32) <- (512x1xf32, 512x1xf32)
        subtract_3 = paddle._C_ops.subtract(add_3, scale_7)

        # pd_op.add: (512x1xf32) <- (512x1xf32, 512x1xf32)
        add_4 = paddle._C_ops.add(add_2, scale_6)

        # pd_op.add: (512x1xf32) <- (512x1xf32, 512x1xf32)
        add_5 = paddle._C_ops.add(add_3, scale_7)

        # builtin.combine: ([512x1xf32, 512x1xf32, 512x1xf32, 512x1xf32]) <- (512x1xf32, 512x1xf32, 512x1xf32, 512x1xf32)
        combine_1 = [subtract_2, subtract_3, add_4, add_5]

        # pd_op.stack: (512x1x4xf32) <- ([512x1xf32, 512x1xf32, 512x1xf32, 512x1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_0 = paddle._C_ops.shape64(add_7)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [-1, 4]

        # pd_op.reshape: (512x4xf32) <- (512x1x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [512]

        # pd_op.split: ([512x4xf32]) <- (512x4xf32, 1xi64, 1xi32)
        split_9 = paddle._C_ops.split(reshape_0, full_int_array_7, full_0)

        # builtin.split: (512x4xf32) <- ([512x4xf32])
        (split_8,) = split_9
        return (
            distribute_fpn_proposals_0,
            split_0,
            split_1,
            split_2,
            split_3,
            split_4,
            split_5,
            split_6,
            split_7,
            roi_align_0,
            roi_align_1,
            roi_align_2,
            roi_align_3,
            full_0,
            concat_0,
            assign_0,
            flatten_0,
            matmul_0,
            relu_0,
            matmul_1,
            matmul_2,
            matmul_3,
            full_int_array_0,
            full_int_array_1,
            slice_0,
            full_int_array_2,
            full_int_array_3,
            slice_1,
            subtract_0,
            assign_1,
            full_int_array_4,
            slice_2,
            assign_2,
            assign_3,
            slice_3,
            subtract_1,
            assign_4,
            assign_5,
            assign_6,
            full_1,
            scale_0,
            add_0,
            assign_7,
            assign_8,
            assign_9,
            assign_10,
            scale_1,
            add_1,
            assign_11,
            full_int_array_5,
            assign_12,
            full_2,
            scale_2,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            scale_3,
            assign_17,
            assign_18,
            assign_19,
            full_3,
            scale_4,
            assign_20,
            assign_21,
            assign_22,
            assign_23,
            scale_5,
            full_4,
            full_5,
            assign_24,
            assign_25,
            assign_26,
            unsqueeze_0,
            multiply_0,
            assign_27,
            unsqueeze_1,
            add_2,
            assign_28,
            unsqueeze_2,
            multiply_1,
            assign_29,
            unsqueeze_3,
            add_3,
            exp_0,
            assign_30,
            assign_31,
            exp_1,
            assign_32,
            assign_33,
            assign_34,
            scale_6,
            subtract_2,
            assign_35,
            scale_7,
            subtract_3,
            assign_36,
            assign_37,
            add_4,
            assign_38,
            assign_39,
            add_5,
            stack_0,
            assign_40,
            gather_0,
            relu_1,
            add_6,
            add_7,
            split_8,
        )
