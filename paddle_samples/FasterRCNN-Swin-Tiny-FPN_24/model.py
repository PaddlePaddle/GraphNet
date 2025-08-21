import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, parameter_2, data_0):
        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [-1, 240, 176, 96]

        # pd_op.reshape: (2x240x176x96xf32) <- (2x42240x96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(data_0, full_int_array_6)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2147483647, 2147483647]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_1

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_1

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_5 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_2

        # pd_op.strided_slice: (2x120x88x96xf32) <- (2x240x176x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            reshape_0, [1, 2], full_int_array_0, full_int_array_1, full_int_array_2
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [1, 0]

        # pd_op.strided_slice: (2x120x88x96xf32) <- (2x240x176x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            reshape_0, [1, 2], full_int_array_3, full_int_array_1, full_int_array_2
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [0, 1]

        # pd_op.strided_slice: (2x120x88x96xf32) <- (2x240x176x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            reshape_0, [1, 2], full_int_array_4, full_int_array_1, full_int_array_2
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 1]

        # pd_op.strided_slice: (2x120x88x96xf32) <- (2x240x176x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            reshape_0, [1, 2], full_int_array_5, full_int_array_1, full_int_array_2
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32]) <- (2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32)
        combine_0 = [strided_slice_0, strided_slice_1, strided_slice_2, strided_slice_3]

        # pd_op.concat: (2x120x88x384xf32) <- ([2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [-1, 10560, 384]

        # pd_op.reshape: (2x10560x384xf32) <- (2x120x88x384xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(concat_0, full_int_array_7)

        # pd_op.layer_norm: (2x10560x384xf32, 2x10560xf32, 2x10560xf32) <- (2x10560x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_1, parameter_2, parameter_1, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x10560x192xf32) <- (2x10560x384xf32, 384x192xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_0, False, False)
        return (
            reshape_0,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            strided_slice_0,
            full_int_array_3,
            assign_0,
            assign_1,
            strided_slice_1,
            full_int_array_4,
            assign_2,
            assign_3,
            strided_slice_2,
            full_int_array_5,
            assign_4,
            assign_5,
            strided_slice_3,
            full_0,
            concat_0,
            reshape_1,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
        )
