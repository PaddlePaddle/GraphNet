import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.divide: (1x2xf32) <- (1x2xf32, 1x2xf32)
        divide_0 = paddle._C_ops.divide(data_3, data_4)
        del data_3, data_4

        # pd_op.cast: (1x2xi32) <- (1x2xf32)
        cast_0 = paddle._C_ops.cast(divide_0, paddle.int32)
        del divide_0

        # pd_op.yolo_box: (1x1083x4xf32, 1x1083x4xf32) <- (1x27x19x19xf32, 1x2xi32)
        yolo_box_0, yolo_box_1 = (lambda x, f: f(x))(
            paddle._C_ops.yolo_box(
                data_0,
                cast_0,
                [116, 90, 156, 198, 373, 326],
                4,
                float("0.005"),
                32,
                True,
                float("1"),
                False,
                float("0.5"),
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_0

        # pd_op.transpose: (1x4x1083xf32) <- (1x1083x4xf32)
        transpose_0 = paddle._C_ops.transpose(yolo_box_1, [0, 2, 1])
        del yolo_box_1

        # pd_op.yolo_box: (1x4332x4xf32, 1x4332x4xf32) <- (1x27x38x38xf32, 1x2xi32)
        yolo_box_2, yolo_box_3 = (lambda x, f: f(x))(
            paddle._C_ops.yolo_box(
                data_1,
                cast_0,
                [30, 61, 62, 45, 59, 119],
                4,
                float("0.005"),
                16,
                True,
                float("1"),
                False,
                float("0.5"),
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_1

        # pd_op.transpose: (1x4x4332xf32) <- (1x4332x4xf32)
        transpose_1 = paddle._C_ops.transpose(yolo_box_3, [0, 2, 1])
        del yolo_box_3

        # pd_op.yolo_box: (1x17328x4xf32, 1x17328x4xf32) <- (1x27x76x76xf32, 1x2xi32)
        yolo_box_4, yolo_box_5 = (lambda x, f: f(x))(
            paddle._C_ops.yolo_box(
                data_2,
                cast_0,
                [10, 13, 16, 30, 33, 23],
                4,
                float("0.005"),
                8,
                True,
                float("1"),
                False,
                float("0.5"),
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del cast_0, data_2

        # pd_op.transpose: (1x4x17328xf32) <- (1x17328x4xf32)
        transpose_2 = paddle._C_ops.transpose(yolo_box_5, [0, 2, 1])
        del yolo_box_5

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x1083x4xf32, 1x4332x4xf32, 1x17328x4xf32]) <- (1x1083x4xf32, 1x4332x4xf32, 1x17328x4xf32)
        combine_0 = [yolo_box_0, yolo_box_2, yolo_box_4]
        del yolo_box_0, yolo_box_2, yolo_box_4

        # pd_op.concat: (1x22743x4xf32) <- ([1x1083x4xf32, 1x4332x4xf32, 1x17328x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, full_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x4x1083xf32, 1x4x4332xf32, 1x4x17328xf32]) <- (1x4x1083xf32, 1x4x4332xf32, 1x4x17328xf32)
        combine_1 = [transpose_0, transpose_1, transpose_2]
        del transpose_0, transpose_1, transpose_2

        # pd_op.concat: (1x4x22743xf32) <- ([1x4x1083xf32, 1x4x4332xf32, 1x4x17328xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)
        del combine_1, full_1

        return concat_0, concat_1
