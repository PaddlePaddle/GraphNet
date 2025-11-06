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
    ):
        # pd_op.distribute_fpn_proposals: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], [-1xi32, -1xi32, -1xi32, -1xi32, -1xi32], -1x1xi32) <- (512x4xf32, 1xi64)
        (
            distribute_fpn_proposals_0,
            distribute_fpn_proposals_1,
            distribute_fpn_proposals_2,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.distribute_fpn_proposals(data_0, data_1, 2, 6, 4, 224, False),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del data_0, data_1

        # builtin.split: (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
            split_4,
        ) = distribute_fpn_proposals_0
        del distribute_fpn_proposals_0

        # builtin.split: (-1xi32, -1xi32, -1xi32, -1xi32, -1xi32) <- ([-1xi32, -1xi32, -1xi32, -1xi32, -1xi32])
        (
            split_5,
            split_6,
            split_7,
            split_8,
            split_9,
        ) = distribute_fpn_proposals_1
        del distribute_fpn_proposals_1

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x214x160xf32, -1x4xf32, -1xi32)
        roi_align_0 = paddle._C_ops.roi_align(
            data_2, split_0, split_5, 7, 7, float("0.25"), 0, True
        )
        del data_2

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x107x80xf32, -1x4xf32, -1xi32)
        roi_align_1 = paddle._C_ops.roi_align(
            data_3, split_1, split_6, 7, 7, float("0.125"), 0, True
        )
        del data_3

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x54x40xf32, -1x4xf32, -1xi32)
        roi_align_2 = paddle._C_ops.roi_align(
            data_4, split_2, split_7, 7, 7, float("0.0625"), 0, True
        )
        del data_4

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x27x20xf32, -1x4xf32, -1xi32)
        roi_align_3 = paddle._C_ops.roi_align(
            data_5, split_3, split_8, 7, 7, float("0.03125"), 0, True
        )
        del data_5

        # pd_op.roi_align: (-1x256x7x7xf32) <- (1x256x14x10xf32, -1x4xf32, -1xi32)
        roi_align_4 = paddle._C_ops.roi_align(
            data_6, split_4, split_9, 7, 7, float("0.015625"), 0, True
        )
        del data_6

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # builtin.combine: ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32]) <- (-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32)
        combine_0 = [roi_align_0, roi_align_1, roi_align_2, roi_align_3, roi_align_4]

        # pd_op.concat: (-1x256x7x7xf32) <- ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.gather: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(concat_0, distribute_fpn_proposals_2, full_0)

        # pd_op.flatten: (-1x12544xf32) <- (-1x256x7x7xf32)
        flatten_0 = paddle._C_ops.flatten(gather_0, 1, 3)

        # pd_op.matmul: (-1x1024xf32) <- (-1x12544xf32, 12544x1024xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_6)
        del parameter_6

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_0 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_4)
        del parameter_4

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_1 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.matmul: (-1x5xf32) <- (-1x1024xf32, 1024x5xf32)
        matmul_2 = paddle._C_ops.matmul(relu_1, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (-1x5xf32) <- (-1x5xf32, 5xf32)
        add_0 = paddle._C_ops.add(matmul_2, parameter_2)
        del parameter_2

        # pd_op.matmul: (-1x16xf32) <- (-1x1024xf32, 1024x16xf32)
        matmul_3 = paddle._C_ops.matmul(relu_1, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add_1 = paddle._C_ops.add(matmul_3, parameter_0)
        del (
            assign_0,
            concat_0,
            distribute_fpn_proposals_2,
            flatten_0,
            full_0,
            gather_0,
            matmul_0,
            matmul_1,
            matmul_2,
            matmul_3,
            parameter_0,
            relu_0,
            relu_1,
            roi_align_0,
            roi_align_1,
            roi_align_2,
            roi_align_3,
            roi_align_4,
            split_0,
            split_1,
            split_2,
            split_3,
            split_4,
            split_5,
            split_6,
            split_7,
            split_8,
            split_9,
        )

        return add_0, add_1
