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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
    ):
        # pd_op.distribute_fpn_proposals: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], [-1xi32, -1xi32, -1xi32, -1xi32], -1x1xi32) <- (1000x4xf32, 1xi64)
        (
            distribute_fpn_proposals_0,
            distribute_fpn_proposals_1,
            distribute_fpn_proposals_2,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.distribute_fpn_proposals(data_0, data_1, 2, 5, 4, 224, False),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_1

        # builtin.split: (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = distribute_fpn_proposals_0
        del distribute_fpn_proposals_0

        # builtin.split: (-1xi32, -1xi32, -1xi32, -1xi32) <- ([-1xi32, -1xi32, -1xi32, -1xi32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = distribute_fpn_proposals_1
        del distribute_fpn_proposals_1

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x304x200xf32, -1x4xf32, -1xi32)
        roi_align_0 = paddle._C_ops.roi_align(
            data_2, split_0, split_4, 7, 7, float("0.25"), 0, True
        )
        del split_0, split_4

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x152x100xf32, -1x4xf32, -1xi32)
        roi_align_1 = paddle._C_ops.roi_align(
            data_3, split_1, split_5, 7, 7, float("0.125"), 0, True
        )
        del split_1, split_5

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x76x50xf32, -1x4xf32, -1xi32)
        roi_align_2 = paddle._C_ops.roi_align(
            data_4, split_2, split_6, 7, 7, float("0.0625"), 0, True
        )
        del split_2, split_6

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x38x25xf32, -1x4xf32, -1xi32)
        roi_align_3 = paddle._C_ops.roi_align(
            data_5, split_3, split_7, 7, 7, float("0.03125"), 0, True
        )
        del split_3, split_7

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32]) <- (-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32)
        combine_0 = [roi_align_0, roi_align_1, roi_align_2, roi_align_3]
        del roi_align_0, roi_align_1, roi_align_2, roi_align_3

        # pd_op.concat: (-1x256x7x7xf32) <- ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.gather: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(concat_0, distribute_fpn_proposals_2, full_0)
        del concat_0, distribute_fpn_proposals_2

        # pd_op.flatten: (-1x12544xf32) <- (-1x256x7x7xf32)
        flatten_0 = paddle._C_ops.flatten(gather_0, 1, 3)
        del gather_0

        # pd_op.matmul: (-1x1024xf32) <- (-1x12544xf32, 12544x1024xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_23, False, False)
        del flatten_0, parameter_23

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_22)
        del matmul_0, parameter_22

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_0 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_21, False, False)
        del parameter_21, relu_0

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_20)
        del matmul_1, parameter_20

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_1 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.matmul: (-1x5xf32) <- (-1x1024xf32, 1024x5xf32)
        matmul_2 = paddle._C_ops.matmul(relu_1, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (-1x5xf32) <- (-1x5xf32, 5xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_18)
        del matmul_2, parameter_18

        # pd_op.matmul: (-1x4xf32) <- (-1x1024xf32, 1024x4xf32)
        matmul_3 = paddle._C_ops.matmul(relu_1, parameter_17, False, False)
        del parameter_17, relu_1

        # pd_op.add: (-1x4xf32) <- (-1x4xf32, 4xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_16)
        del matmul_3, parameter_16

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (1000xf32) <- (1000xf32, 1000xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_0

        # pd_op.subtract: (1000xf32) <- (1000xf32, 1000xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_2

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1000xf32) <- (1000xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_0, full_1, float("0"), True)

        # pd_op.add: (1000xf32) <- (1000xf32, 1000xf32)
        add_5 = paddle._C_ops.add(slice_1, scale_1)
        del scale_1, slice_1

        # pd_op.scale: (1000xf32) <- (1000xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(subtract_1, full_1, float("0"), True)

        # pd_op.add: (1000xf32) <- (1000xf32, 1000xf32)
        add_6 = paddle._C_ops.add(slice_3, scale_2)
        del scale_2, slice_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            add_4, [1], full_int_array_2, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(strided_slice_0, full_2, float("0"), True)
        del strided_slice_0

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            add_4, [1], full_int_array_3, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(strided_slice_1, full_2, float("0"), True)
        del strided_slice_1

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            add_4, [1], full_int_array_0, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(strided_slice_2, full_3, float("0"), True)
        del strided_slice_2

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            add_4, [1], full_int_array_1, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(strided_slice_3, full_3, float("0"), True)
        del full_3, strided_slice_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("4.13517"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(scale_5, full_4, full_5)
        del scale_5

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(scale_6, full_4, full_5)
        del scale_6

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(subtract_0, full_int_array_3)
        del subtract_0

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_0 = paddle._C_ops.multiply(scale_3, unsqueeze_0)
        del scale_3

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(add_5, full_int_array_3)
        del add_5

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_7 = paddle._C_ops.add(multiply_0, unsqueeze_1)
        del multiply_0, unsqueeze_1

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(subtract_1, full_int_array_3)
        del subtract_1

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_4, unsqueeze_2)
        del scale_4

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(add_6, full_int_array_3)
        del add_6

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_8 = paddle._C_ops.add(multiply_1, unsqueeze_3)
        del multiply_1, unsqueeze_3

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_0 = paddle._C_ops.exp(clip_0)
        del clip_0

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_2 = paddle._C_ops.multiply(exp_0, unsqueeze_0)
        del exp_0, unsqueeze_0

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_1 = paddle._C_ops.exp(clip_1)
        del clip_1

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_3 = paddle._C_ops.multiply(exp_1, unsqueeze_2)
        del exp_1, unsqueeze_2

        # pd_op.scale: (1000x1xf32) <- (1000x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(multiply_2, full_1, float("0"), True)
        del multiply_2

        # pd_op.subtract: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        subtract_2 = paddle._C_ops.subtract(add_7, scale_7)

        # pd_op.scale: (1000x1xf32) <- (1000x1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(multiply_3, full_1, float("0"), True)
        del multiply_3

        # pd_op.subtract: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        subtract_3 = paddle._C_ops.subtract(add_8, scale_8)

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_9 = paddle._C_ops.add(add_7, scale_7)
        del add_7, scale_7

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_10 = paddle._C_ops.add(add_8, scale_8)
        del add_8, scale_8

        # builtin.combine: ([1000x1xf32, 1000x1xf32, 1000x1xf32, 1000x1xf32]) <- (1000x1xf32, 1000x1xf32, 1000x1xf32, 1000x1xf32)
        combine_1 = [subtract_2, subtract_3, add_9, add_10]
        del add_10, add_9, subtract_2, subtract_3

        # pd_op.stack: (1000x1x4xf32) <- ([1000x1xf32, 1000x1xf32, 1000x1xf32, 1000x1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_0 = paddle._C_ops.shape64(add_4)
        del add_4

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [-1, 4]

        # pd_op.reshape: (1000x4xf32) <- (1000x1x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_6)
        del stack_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1000]

        # pd_op.split: ([1000x4xf32]) <- (1000x4xf32, 1xi64, 1xi32)
        split_8 = paddle._C_ops.split(reshape_0, full_int_array_7, full_0)
        del reshape_0

        # builtin.split: (1000x4xf32) <- ([1000x4xf32])
        (split_9,) = split_8
        del split_8

        # pd_op.slice: (2xf32) <- (1x2xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_6, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del data_6

        # pd_op.slice: (xf32) <- (2xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xf32) <- (2xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_3, full_int_array_0, [1], [0]
        )
        del slice_5

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            split_9, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1000xf32) <- (1000xf32, 1xf32, xf32)
        clip_2 = paddle._C_ops.clip(slice_8, full_6, slice_7)
        del slice_8

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            split_9, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.clip: (1000xf32) <- (1000xf32, 1xf32, xf32)
        clip_3 = paddle._C_ops.clip(slice_9, full_6, slice_6)
        del slice_9

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            split_9, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.clip: (1000xf32) <- (1000xf32, 1xf32, xf32)
        clip_4 = paddle._C_ops.clip(slice_10, full_6, slice_7)
        del slice_10

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            split_9, [1], full_int_array_1, full_int_array_4, [1], [1]
        )
        del split_9

        # pd_op.clip: (1000xf32) <- (1000xf32, 1xf32, xf32)
        clip_5 = paddle._C_ops.clip(slice_11, full_6, slice_6)
        del slice_11

        # builtin.combine: ([1000xf32, 1000xf32, 1000xf32, 1000xf32]) <- (1000xf32, 1000xf32, 1000xf32, 1000xf32)
        combine_2 = [clip_2, clip_3, clip_4, clip_5]
        del clip_2, clip_3, clip_4, clip_5

        # pd_op.stack: (1000x4xf32) <- ([1000xf32, 1000xf32, 1000xf32, 1000xf32])
        stack_1 = paddle._C_ops.stack(combine_2, 1)
        del combine_2

        # pd_op.shape64: (2xi64) <- (1000x4xf32)
        shape64_1 = paddle._C_ops.shape64(stack_1)

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del shape64_1

        # builtin.combine: ([1xi64]) <- (1xi64)
        combine_3 = [slice_12]
        del slice_12

        # pd_op.concat: (1xi64) <- ([1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.distribute_fpn_proposals: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], [-1xi32, -1xi32, -1xi32, -1xi32], -1x1xi32) <- (1000x4xf32, 1xi64)
        (
            distribute_fpn_proposals_3,
            distribute_fpn_proposals_4,
            distribute_fpn_proposals_5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.distribute_fpn_proposals(
                stack_1, concat_1, 2, 5, 4, 224, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del concat_1

        # builtin.split: (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32])
        (
            split_10,
            split_11,
            split_12,
            split_13,
        ) = distribute_fpn_proposals_3
        del distribute_fpn_proposals_3

        # builtin.split: (-1xi32, -1xi32, -1xi32, -1xi32) <- ([-1xi32, -1xi32, -1xi32, -1xi32])
        (
            split_14,
            split_15,
            split_16,
            split_17,
        ) = distribute_fpn_proposals_4
        del distribute_fpn_proposals_4

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x304x200xf32, -1x4xf32, -1xi32)
        roi_align_4 = paddle._C_ops.roi_align(
            data_2, split_10, split_14, 7, 7, float("0.25"), 0, True
        )
        del split_10, split_14

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x152x100xf32, -1x4xf32, -1xi32)
        roi_align_5 = paddle._C_ops.roi_align(
            data_3, split_11, split_15, 7, 7, float("0.125"), 0, True
        )
        del split_11, split_15

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x76x50xf32, -1x4xf32, -1xi32)
        roi_align_6 = paddle._C_ops.roi_align(
            data_4, split_12, split_16, 7, 7, float("0.0625"), 0, True
        )
        del split_12, split_16

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x38x25xf32, -1x4xf32, -1xi32)
        roi_align_7 = paddle._C_ops.roi_align(
            data_5, split_13, split_17, 7, 7, float("0.03125"), 0, True
        )
        del split_13, split_17

        # builtin.combine: ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32]) <- (-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32)
        combine_4 = [roi_align_4, roi_align_5, roi_align_6, roi_align_7]
        del roi_align_4, roi_align_5, roi_align_6, roi_align_7

        # pd_op.concat: (-1x256x7x7xf32) <- ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4

        # pd_op.gather: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1xi32, 1xi32)
        gather_1 = paddle._C_ops.gather(concat_2, distribute_fpn_proposals_5, full_0)
        del concat_2, distribute_fpn_proposals_5

        # pd_op.flatten: (-1x12544xf32) <- (-1x256x7x7xf32)
        flatten_1 = paddle._C_ops.flatten(gather_1, 1, 3)
        del gather_1

        # pd_op.matmul: (-1x1024xf32) <- (-1x12544xf32, 12544x1024xf32)
        matmul_4 = paddle._C_ops.matmul(flatten_1, parameter_15, False, False)
        del flatten_1, parameter_15

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_11 = paddle._C_ops.add(matmul_4, parameter_14)
        del matmul_4, parameter_14

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_2 = paddle._C_ops.relu(add_11)
        del add_11

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_5 = paddle._C_ops.matmul(relu_2, parameter_13, False, False)
        del parameter_13, relu_2

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_12 = paddle._C_ops.add(matmul_5, parameter_12)
        del matmul_5, parameter_12

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_3 = paddle._C_ops.relu(add_12)
        del add_12

        # pd_op.matmul: (-1x5xf32) <- (-1x1024xf32, 1024x5xf32)
        matmul_6 = paddle._C_ops.matmul(relu_3, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (-1x5xf32) <- (-1x5xf32, 5xf32)
        add_13 = paddle._C_ops.add(matmul_6, parameter_10)
        del matmul_6, parameter_10

        # pd_op.matmul: (-1x4xf32) <- (-1x1024xf32, 1024x4xf32)
        matmul_7 = paddle._C_ops.matmul(relu_3, parameter_9, False, False)
        del parameter_9, relu_3

        # pd_op.add: (-1x4xf32) <- (-1x4xf32, 4xf32)
        add_14 = paddle._C_ops.add(matmul_7, parameter_8)
        del matmul_7, parameter_8

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            stack_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            stack_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (1000xf32) <- (1000xf32, 1000xf32)
        subtract_4 = paddle._C_ops.subtract(slice_13, slice_14)
        del slice_13

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            stack_1, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            stack_1, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del stack_1

        # pd_op.subtract: (1000xf32) <- (1000xf32, 1000xf32)
        subtract_5 = paddle._C_ops.subtract(slice_15, slice_16)
        del slice_15

        # pd_op.scale: (1000xf32) <- (1000xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(subtract_4, full_1, float("0"), True)

        # pd_op.add: (1000xf32) <- (1000xf32, 1000xf32)
        add_15 = paddle._C_ops.add(slice_14, scale_9)
        del scale_9, slice_14

        # pd_op.scale: (1000xf32) <- (1000xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(subtract_5, full_1, float("0"), True)

        # pd_op.add: (1000xf32) <- (1000xf32, 1000xf32)
        add_16 = paddle._C_ops.add(slice_16, scale_10)
        del scale_10, slice_16

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            add_14, [1], full_int_array_2, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(strided_slice_4, full_7, float("0"), True)
        del strided_slice_4

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            add_14, [1], full_int_array_3, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(strided_slice_5, full_7, float("0"), True)
        del full_7, strided_slice_5

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            add_14, [1], full_int_array_0, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(strided_slice_6, full_2, float("0"), True)
        del strided_slice_6

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            add_14, [1], full_int_array_1, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(strided_slice_7, full_2, float("0"), True)
        del full_2, strided_slice_7

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_6 = paddle._C_ops.clip(scale_13, full_4, full_5)
        del scale_13

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_7 = paddle._C_ops.clip(scale_14, full_4, full_5)
        del scale_14

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(subtract_4, full_int_array_3)
        del subtract_4

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_4 = paddle._C_ops.multiply(scale_11, unsqueeze_4)
        del scale_11

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(add_15, full_int_array_3)
        del add_15

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_17 = paddle._C_ops.add(multiply_4, unsqueeze_5)
        del multiply_4, unsqueeze_5

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(subtract_5, full_int_array_3)
        del subtract_5

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_5 = paddle._C_ops.multiply(scale_12, unsqueeze_6)
        del scale_12

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(add_16, full_int_array_3)
        del add_16

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_18 = paddle._C_ops.add(multiply_5, unsqueeze_7)
        del multiply_5, unsqueeze_7

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_2 = paddle._C_ops.exp(clip_6)
        del clip_6

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_6 = paddle._C_ops.multiply(exp_2, unsqueeze_4)
        del exp_2, unsqueeze_4

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_3 = paddle._C_ops.exp(clip_7)
        del clip_7

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_7 = paddle._C_ops.multiply(exp_3, unsqueeze_6)
        del exp_3, unsqueeze_6

        # pd_op.scale: (1000x1xf32) <- (1000x1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(multiply_6, full_1, float("0"), True)
        del multiply_6

        # pd_op.subtract: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        subtract_6 = paddle._C_ops.subtract(add_17, scale_15)

        # pd_op.scale: (1000x1xf32) <- (1000x1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(multiply_7, full_1, float("0"), True)
        del multiply_7

        # pd_op.subtract: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        subtract_7 = paddle._C_ops.subtract(add_18, scale_16)

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_19 = paddle._C_ops.add(add_17, scale_15)
        del add_17, scale_15

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_20 = paddle._C_ops.add(add_18, scale_16)
        del add_18, scale_16

        # builtin.combine: ([1000x1xf32, 1000x1xf32, 1000x1xf32, 1000x1xf32]) <- (1000x1xf32, 1000x1xf32, 1000x1xf32, 1000x1xf32)
        combine_5 = [subtract_6, subtract_7, add_19, add_20]
        del add_19, add_20, subtract_6, subtract_7

        # pd_op.stack: (1000x1x4xf32) <- ([1000x1xf32, 1000x1xf32, 1000x1xf32, 1000x1xf32])
        stack_2 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_2 = paddle._C_ops.shape64(add_14)
        del add_14

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_2

        # pd_op.reshape: (1000x4xf32) <- (1000x1x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_2, full_int_array_6)
        del stack_2

        # pd_op.split: ([1000x4xf32]) <- (1000x4xf32, 1xi64, 1xi32)
        split_18 = paddle._C_ops.split(reshape_1, full_int_array_7, full_0)
        del reshape_1

        # builtin.split: (1000x4xf32) <- ([1000x4xf32])
        (split_19,) = split_18
        del split_18

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            split_19, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.clip: (1000xf32) <- (1000xf32, 1xf32, xf32)
        clip_8 = paddle._C_ops.clip(slice_18, full_6, slice_7)
        del slice_18

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            split_19, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.clip: (1000xf32) <- (1000xf32, 1xf32, xf32)
        clip_9 = paddle._C_ops.clip(slice_19, full_6, slice_6)
        del slice_19

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            split_19, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.clip: (1000xf32) <- (1000xf32, 1xf32, xf32)
        clip_10 = paddle._C_ops.clip(slice_20, full_6, slice_7)
        del slice_20, slice_7

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            split_19, [1], full_int_array_1, full_int_array_4, [1], [1]
        )
        del split_19

        # pd_op.clip: (1000xf32) <- (1000xf32, 1xf32, xf32)
        clip_11 = paddle._C_ops.clip(slice_21, full_6, slice_6)
        del full_6, slice_21, slice_6

        # builtin.combine: ([1000xf32, 1000xf32, 1000xf32, 1000xf32]) <- (1000xf32, 1000xf32, 1000xf32, 1000xf32)
        combine_6 = [clip_8, clip_9, clip_10, clip_11]
        del clip_10, clip_11, clip_8, clip_9

        # pd_op.stack: (1000x4xf32) <- ([1000xf32, 1000xf32, 1000xf32, 1000xf32])
        stack_3 = paddle._C_ops.stack(combine_6, 1)
        del combine_6

        # pd_op.shape64: (2xi64) <- (1000x4xf32)
        shape64_3 = paddle._C_ops.shape64(stack_3)

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del shape64_3

        # builtin.combine: ([1xi64]) <- (1xi64)
        combine_7 = [slice_22]
        del slice_22

        # pd_op.concat: (1xi64) <- ([1xi64], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_7, full_0)
        del combine_7

        # pd_op.distribute_fpn_proposals: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], [-1xi32, -1xi32, -1xi32, -1xi32], -1x1xi32) <- (1000x4xf32, 1xi64)
        (
            distribute_fpn_proposals_6,
            distribute_fpn_proposals_7,
            distribute_fpn_proposals_8,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.distribute_fpn_proposals(
                stack_3, concat_3, 2, 5, 4, 224, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del concat_3

        # builtin.split: (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32])
        (
            split_20,
            split_21,
            split_22,
            split_23,
        ) = distribute_fpn_proposals_6
        del distribute_fpn_proposals_6

        # builtin.split: (-1xi32, -1xi32, -1xi32, -1xi32) <- ([-1xi32, -1xi32, -1xi32, -1xi32])
        (
            split_24,
            split_25,
            split_26,
            split_27,
        ) = distribute_fpn_proposals_7
        del distribute_fpn_proposals_7

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x304x200xf32, -1x4xf32, -1xi32)
        roi_align_8 = paddle._C_ops.roi_align(
            data_2, split_20, split_24, 7, 7, float("0.25"), 0, True
        )
        del data_2, split_20, split_24

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x152x100xf32, -1x4xf32, -1xi32)
        roi_align_9 = paddle._C_ops.roi_align(
            data_3, split_21, split_25, 7, 7, float("0.125"), 0, True
        )
        del data_3, split_21, split_25

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x76x50xf32, -1x4xf32, -1xi32)
        roi_align_10 = paddle._C_ops.roi_align(
            data_4, split_22, split_26, 7, 7, float("0.0625"), 0, True
        )
        del data_4, split_22, split_26

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x38x25xf32, -1x4xf32, -1xi32)
        roi_align_11 = paddle._C_ops.roi_align(
            data_5, split_23, split_27, 7, 7, float("0.03125"), 0, True
        )
        del data_5, split_23, split_27

        # builtin.combine: ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32]) <- (-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32)
        combine_8 = [roi_align_8, roi_align_9, roi_align_10, roi_align_11]
        del roi_align_10, roi_align_11, roi_align_8, roi_align_9

        # pd_op.concat: (-1x256x7x7xf32) <- ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_8, full_0)
        del combine_8

        # pd_op.gather: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1xi32, 1xi32)
        gather_2 = paddle._C_ops.gather(concat_4, distribute_fpn_proposals_8, full_0)
        del concat_4, distribute_fpn_proposals_8

        # pd_op.flatten: (-1x12544xf32) <- (-1x256x7x7xf32)
        flatten_2 = paddle._C_ops.flatten(gather_2, 1, 3)
        del gather_2

        # pd_op.matmul: (-1x1024xf32) <- (-1x12544xf32, 12544x1024xf32)
        matmul_8 = paddle._C_ops.matmul(flatten_2, parameter_7, False, False)
        del flatten_2, parameter_7

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_21 = paddle._C_ops.add(matmul_8, parameter_6)
        del matmul_8, parameter_6

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_4 = paddle._C_ops.relu(add_21)
        del add_21

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_9 = paddle._C_ops.matmul(relu_4, parameter_5, False, False)
        del parameter_5, relu_4

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_22 = paddle._C_ops.add(matmul_9, parameter_4)
        del matmul_9, parameter_4

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_5 = paddle._C_ops.relu(add_22)
        del add_22

        # pd_op.matmul: (-1x5xf32) <- (-1x1024xf32, 1024x5xf32)
        matmul_10 = paddle._C_ops.matmul(relu_5, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (-1x5xf32) <- (-1x5xf32, 5xf32)
        add_23 = paddle._C_ops.add(matmul_10, parameter_2)
        del matmul_10, parameter_2

        # pd_op.matmul: (-1x4xf32) <- (-1x1024xf32, 1024x4xf32)
        matmul_11 = paddle._C_ops.matmul(relu_5, parameter_1, False, False)
        del parameter_1, relu_5

        # pd_op.add: (-1x4xf32) <- (-1x4xf32, 4xf32)
        add_0 = paddle._C_ops.add(matmul_11, parameter_0)
        del matmul_11, parameter_0

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            stack_3, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            stack_3, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (1000xf32) <- (1000xf32, 1000xf32)
        subtract_8 = paddle._C_ops.subtract(slice_23, slice_24)
        del slice_23

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            stack_3, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (1000xf32) <- (1000x4xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            stack_3, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (1000xf32) <- (1000xf32, 1000xf32)
        subtract_9 = paddle._C_ops.subtract(slice_25, slice_26)
        del slice_25

        # pd_op.scale: (1000xf32) <- (1000xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(subtract_8, full_1, float("0"), True)

        # pd_op.add: (1000xf32) <- (1000xf32, 1000xf32)
        add_24 = paddle._C_ops.add(slice_24, scale_17)
        del scale_17, slice_24

        # pd_op.scale: (1000xf32) <- (1000xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(subtract_9, full_1, float("0"), True)

        # pd_op.add: (1000xf32) <- (1000xf32, 1000xf32)
        add_25 = paddle._C_ops.add(slice_26, scale_18)
        del scale_18, slice_26

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            add_0, [1], full_int_array_2, full_int_array_5, full_int_array_4
        )

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.0333333"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(strided_slice_8, full_8, float("0"), True)
        del strided_slice_8

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            add_0, [1], full_int_array_3, full_int_array_5, full_int_array_4
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(strided_slice_9, full_8, float("0"), True)
        del full_8, strided_slice_9

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            add_0, [1], full_int_array_0, full_int_array_5, full_int_array_4
        )
        del full_int_array_0

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.0666667"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(strided_slice_10, full_9, float("0"), True)
        del strided_slice_10

        # pd_op.strided_slice: (-1x1xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            add_0, [1], full_int_array_1, full_int_array_5, full_int_array_4
        )
        del full_int_array_1, full_int_array_4, full_int_array_5

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(strided_slice_11, full_9, float("0"), True)
        del full_9, strided_slice_11

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_12 = paddle._C_ops.clip(scale_21, full_4, full_5)
        del scale_21

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_13 = paddle._C_ops.clip(scale_22, full_4, full_5)
        del full_4, full_5, scale_22

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(subtract_8, full_int_array_3)
        del subtract_8

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_8 = paddle._C_ops.multiply(scale_19, unsqueeze_8)
        del scale_19

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(add_24, full_int_array_3)
        del add_24

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_26 = paddle._C_ops.add(multiply_8, unsqueeze_9)
        del multiply_8, unsqueeze_9

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(subtract_9, full_int_array_3)
        del subtract_9

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_9 = paddle._C_ops.multiply(scale_20, unsqueeze_10)
        del scale_20

        # pd_op.unsqueeze: (1000x1xf32) <- (1000xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(add_25, full_int_array_3)
        del add_25

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_27 = paddle._C_ops.add(multiply_9, unsqueeze_11)
        del multiply_9, unsqueeze_11

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_4 = paddle._C_ops.exp(clip_12)
        del clip_12

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_10 = paddle._C_ops.multiply(exp_4, unsqueeze_8)
        del exp_4, unsqueeze_8

        # pd_op.exp: (-1x1xf32) <- (-1x1xf32)
        exp_5 = paddle._C_ops.exp(clip_13)
        del clip_13

        # pd_op.multiply: (1000x1xf32) <- (-1x1xf32, 1000x1xf32)
        multiply_11 = paddle._C_ops.multiply(exp_5, unsqueeze_10)
        del exp_5, unsqueeze_10

        # pd_op.scale: (1000x1xf32) <- (1000x1xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(multiply_10, full_1, float("0"), True)
        del multiply_10

        # pd_op.subtract: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        subtract_10 = paddle._C_ops.subtract(add_26, scale_23)

        # pd_op.scale: (1000x1xf32) <- (1000x1xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(multiply_11, full_1, float("0"), True)
        del full_1, multiply_11

        # pd_op.subtract: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        subtract_11 = paddle._C_ops.subtract(add_27, scale_24)

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_28 = paddle._C_ops.add(add_26, scale_23)
        del add_26, scale_23

        # pd_op.add: (1000x1xf32) <- (1000x1xf32, 1000x1xf32)
        add_29 = paddle._C_ops.add(add_27, scale_24)
        del add_27, scale_24

        # builtin.combine: ([1000x1xf32, 1000x1xf32, 1000x1xf32, 1000x1xf32]) <- (1000x1xf32, 1000x1xf32, 1000x1xf32, 1000x1xf32)
        combine_9 = [subtract_10, subtract_11, add_28, add_29]
        del add_28, add_29, subtract_10, subtract_11

        # pd_op.stack: (1000x1x4xf32) <- ([1000x1xf32, 1000x1xf32, 1000x1xf32, 1000x1xf32])
        stack_4 = paddle._C_ops.stack(combine_9, -1)
        del combine_9

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        shape64_4 = paddle._C_ops.shape64(add_0)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_4

        # pd_op.reshape: (1000x4xf32) <- (1000x1x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(stack_4, full_int_array_6)
        del full_int_array_6, stack_4

        # pd_op.split: ([1000x4xf32]) <- (1000x4xf32, 1xi64, 1xi32)
        split_28 = paddle._C_ops.split(reshape_2, full_int_array_7, full_0)
        del full_0, full_int_array_7, reshape_2

        # builtin.split: (1000x4xf32) <- ([1000x4xf32])
        (split_29,) = split_28
        del split_28

        # pd_op.softmax: (-1x5xf32) <- (-1x5xf32)
        softmax_0 = paddle._C_ops.softmax(add_3, -1)
        del add_3

        # pd_op.softmax: (-1x5xf32) <- (-1x5xf32)
        softmax_1 = paddle._C_ops.softmax(add_13, -1)
        del add_13

        # pd_op.softmax: (-1x5xf32) <- (-1x5xf32)
        softmax_2 = paddle._C_ops.softmax(add_23, -1)
        del add_23

        # builtin.combine: ([-1x5xf32, -1x5xf32, -1x5xf32]) <- (-1x5xf32, -1x5xf32, -1x5xf32)
        combine_10 = [softmax_0, softmax_1, softmax_2]
        del softmax_0, softmax_1, softmax_2

        # pd_op.add_n: (-1x5xf32) <- ([-1x5xf32, -1x5xf32, -1x5xf32])
        add_n_0 = paddle._C_ops.add_n(combine_10)
        del combine_10

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("0.333333"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x5xf32) <- (-1x5xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_n_0, full_10, float("0"), True)
        del add_n_0, full_10, stack_3

        return add_0, scale_0
