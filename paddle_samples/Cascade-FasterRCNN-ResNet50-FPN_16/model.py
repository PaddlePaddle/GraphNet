import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8
    ):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # builtin.combine: ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32]) <- (512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32)
        combine_0 = [data_4, data_5, data_6, data_7]

        # pd_op.concat: (2048x4xf32) <- ([512x4xf32, 512x4xf32, 512x4xf32, 512x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.distribute_fpn_proposals: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], [-1xi32, -1xi32, -1xi32, -1xi32], -1x1xi32) <- (2048x4xf32, 4xi64)
        (
            distribute_fpn_proposals_1,
            distribute_fpn_proposals_2,
            distribute_fpn_proposals_0,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.distribute_fpn_proposals(
                concat_1, data_8, 2, 5, 4, 224, False
            ),
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

        # pd_op.roi_align: (-1x256x7x7xf32) <- (4x256x256x192xf32, -1x4xf32, -1xi32)
        roi_align_0 = paddle._C_ops.roi_align(
            data_0, split_0, split_4, 7, 7, float("0.25"), 0, True
        )

        # pd_op.roi_align: (-1x256x7x7xf32) <- (4x256x128x96xf32, -1x4xf32, -1xi32)
        roi_align_1 = paddle._C_ops.roi_align(
            data_1, split_1, split_5, 7, 7, float("0.125"), 0, True
        )

        # pd_op.roi_align: (-1x256x7x7xf32) <- (4x256x64x48xf32, -1x4xf32, -1xi32)
        roi_align_2 = paddle._C_ops.roi_align(
            data_2, split_2, split_6, 7, 7, float("0.0625"), 0, True
        )

        # pd_op.roi_align: (-1x256x7x7xf32) <- (4x256x32x24xf32, -1x4xf32, -1xi32)
        roi_align_3 = paddle._C_ops.roi_align(
            data_3, split_3, split_7, 7, 7, float("0.03125"), 0, True
        )

        # builtin.combine: ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32]) <- (-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32)
        combine_1 = [roi_align_0, roi_align_1, roi_align_2, roi_align_3]

        # pd_op.concat: (-1x256x7x7xf32) <- ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.gather: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(concat_0, distribute_fpn_proposals_0, full_0)
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
            assign_0,
            concat_0,
            assign_1,
            gather_0,
        )
