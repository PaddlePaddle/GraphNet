import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, parameter_0, parameter_1, parameter_2, parameter_3, data_0, data_1
    ):
        # pd_op.share_data_: (1x-1x4xf32) <- (1x-1x4xf32)
        share_data__0 = data_0.detach()
        del data_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.unsqueeze: (1x-1x1x4xf32) <- (1x-1x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(share_data__0, full_int_array_0)
        del share_data__0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [1, 1, 2]

        # pd_op.tile: (1x5x4xf32) <- (1x5x2xf32, 3xi64)
        tile_0 = paddle._C_ops.tile(data_1, full_int_array_1)
        del data_1, full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.unsqueeze: (1x1x5x4xf32) <- (1x5x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(tile_0, full_int_array_2)
        del tile_0

        # pd_op.multiply: (1x-1x5x4xf32) <- (1x-1x1x4xf32, 1x1x5x4xf32)
        multiply_0 = paddle._C_ops.multiply(unsqueeze_0, unsqueeze_1)
        del unsqueeze_0, unsqueeze_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.slice: (1x-1x4xf32) <- (1x-1x5x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            multiply_0, [2], full_int_array_3, full_int_array_2, [1], [2]
        )

        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("128"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (128xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="int64")
        del full_0, full_1, full_2

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_3,
            [],
            paddle.int64,
            [float("2")],
            paddle.framework._current_expected_place(),
        )
        del full_3

        # pd_op.floor_divide: (128xi64) <- (128xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(arange_0, assign_value__0)
        del arange_0, assign_value__0

        # pd_op.cast: (128xf32) <- (128xi64)
        cast_0 = paddle._C_ops.cast(floor_divide_0, paddle.float32)
        del floor_divide_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_4, float("0"), True)
        del cast_0, full_4

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.0078125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_5, float("0"), True)
        del full_5, scale_0

        # pd_op.full: (128xf32) <- ()
        full_6 = paddle._C_ops.full(
            [128],
            float("10000"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.elementwise_pow: (128xf32) <- (128xf32, 128xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(full_6, scale_1)
        del full_6, scale_1

        # pd_op.full: (128xf32) <- ()
        full_7 = paddle._C_ops.full(
            [128],
            float("6.28319"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.divide: (128xf32) <- (128xf32, 128xf32)
        divide_0 = paddle._C_ops.divide(full_7, elementwise_pow_0)
        del elementwise_pow_0, full_7

        # pd_op.shape64: (3xi64) <- (1x-1x4xf32)
        shape64_0 = paddle._C_ops.shape64(slice_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_0, [1], [0]
        )
        del shape64_0

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32]) <- (1x-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(slice_0, 4, full_8)
        del slice_0

        # builtin.split: (1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32) <- ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.multiply: (1x-1x128xf32) <- (1x-1x1xf32, 128xf32)
        multiply_1 = paddle._C_ops.multiply(split_0, divide_0)
        del split_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.strided_slice: (1x-1x64xf32) <- (1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            multiply_1, [2], full_int_array_3, full_int_array_4, full_int_array_0
        )

        # pd_op.sin: (1x-1x64xf32) <- (1x-1x64xf32)
        sin_0 = paddle._C_ops.sin(strided_slice_0)
        del strided_slice_0

        # pd_op.strided_slice: (1x-1x64xf32) <- (1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            multiply_1, [2], full_int_array_2, full_int_array_4, full_int_array_0
        )
        del multiply_1

        # pd_op.cos: (1x-1x64xf32) <- (1x-1x64xf32)
        cos_0 = paddle._C_ops.cos(strided_slice_1)
        del strided_slice_1

        # builtin.combine: ([1x-1x64xf32, 1x-1x64xf32]) <- (1x-1x64xf32, 1x-1x64xf32)
        combine_0 = [sin_0, cos_0]
        del cos_0, sin_0

        # pd_op.stack: (1x-1x64x2xf32) <- ([1x-1x64xf32, 1x-1x64xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 3)
        del combine_0

        # pd_op.flatten: (1x-1x128xf32) <- (1x-1x64x2xf32)
        flatten_0 = paddle._C_ops.flatten(stack_0, 2, 3)
        del stack_0

        # pd_op.multiply: (1x-1x128xf32) <- (1x-1x1xf32, 128xf32)
        multiply_2 = paddle._C_ops.multiply(split_1, divide_0)
        del split_1

        # pd_op.strided_slice: (1x-1x64xf32) <- (1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            multiply_2, [2], full_int_array_3, full_int_array_4, full_int_array_0
        )

        # pd_op.sin: (1x-1x64xf32) <- (1x-1x64xf32)
        sin_1 = paddle._C_ops.sin(strided_slice_2)
        del strided_slice_2

        # pd_op.strided_slice: (1x-1x64xf32) <- (1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            multiply_2, [2], full_int_array_2, full_int_array_4, full_int_array_0
        )
        del multiply_2

        # pd_op.cos: (1x-1x64xf32) <- (1x-1x64xf32)
        cos_1 = paddle._C_ops.cos(strided_slice_3)
        del strided_slice_3

        # builtin.combine: ([1x-1x64xf32, 1x-1x64xf32]) <- (1x-1x64xf32, 1x-1x64xf32)
        combine_1 = [sin_1, cos_1]
        del cos_1, sin_1

        # pd_op.stack: (1x-1x64x2xf32) <- ([1x-1x64xf32, 1x-1x64xf32])
        stack_1 = paddle._C_ops.stack(combine_1, 3)
        del combine_1

        # pd_op.flatten: (1x-1x128xf32) <- (1x-1x64x2xf32)
        flatten_1 = paddle._C_ops.flatten(stack_1, 2, 3)
        del stack_1

        # pd_op.multiply: (1x-1x128xf32) <- (1x-1x1xf32, 128xf32)
        multiply_3 = paddle._C_ops.multiply(split_2, divide_0)
        del split_2

        # pd_op.strided_slice: (1x-1x64xf32) <- (1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            multiply_3, [2], full_int_array_3, full_int_array_4, full_int_array_0
        )

        # pd_op.sin: (1x-1x64xf32) <- (1x-1x64xf32)
        sin_2 = paddle._C_ops.sin(strided_slice_4)
        del strided_slice_4

        # pd_op.strided_slice: (1x-1x64xf32) <- (1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            multiply_3, [2], full_int_array_2, full_int_array_4, full_int_array_0
        )
        del multiply_3

        # pd_op.cos: (1x-1x64xf32) <- (1x-1x64xf32)
        cos_2 = paddle._C_ops.cos(strided_slice_5)
        del strided_slice_5

        # builtin.combine: ([1x-1x64xf32, 1x-1x64xf32]) <- (1x-1x64xf32, 1x-1x64xf32)
        combine_2 = [sin_2, cos_2]
        del cos_2, sin_2

        # pd_op.stack: (1x-1x64x2xf32) <- ([1x-1x64xf32, 1x-1x64xf32])
        stack_2 = paddle._C_ops.stack(combine_2, 3)
        del combine_2

        # pd_op.flatten: (1x-1x128xf32) <- (1x-1x64x2xf32)
        flatten_2 = paddle._C_ops.flatten(stack_2, 2, 3)
        del stack_2

        # pd_op.multiply: (1x-1x128xf32) <- (1x-1x1xf32, 128xf32)
        multiply_4 = paddle._C_ops.multiply(split_3, divide_0)
        del divide_0, split_3

        # pd_op.strided_slice: (1x-1x64xf32) <- (1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            multiply_4, [2], full_int_array_3, full_int_array_4, full_int_array_0
        )
        del full_int_array_3

        # pd_op.sin: (1x-1x64xf32) <- (1x-1x64xf32)
        sin_3 = paddle._C_ops.sin(strided_slice_6)
        del strided_slice_6

        # pd_op.strided_slice: (1x-1x64xf32) <- (1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            multiply_4, [2], full_int_array_2, full_int_array_4, full_int_array_0
        )
        del full_int_array_0, full_int_array_2, full_int_array_4, multiply_4

        # pd_op.cos: (1x-1x64xf32) <- (1x-1x64xf32)
        cos_3 = paddle._C_ops.cos(strided_slice_7)
        del strided_slice_7

        # builtin.combine: ([1x-1x64xf32, 1x-1x64xf32]) <- (1x-1x64xf32, 1x-1x64xf32)
        combine_3 = [sin_3, cos_3]
        del cos_3, sin_3

        # pd_op.stack: (1x-1x64x2xf32) <- ([1x-1x64xf32, 1x-1x64xf32])
        stack_3 = paddle._C_ops.stack(combine_3, 3)
        del combine_3

        # pd_op.flatten: (1x-1x128xf32) <- (1x-1x64x2xf32)
        flatten_3 = paddle._C_ops.flatten(stack_3, 2, 3)
        del stack_3

        # builtin.combine: ([1x-1x128xf32, 1x-1x128xf32, 1x-1x128xf32, 1x-1x128xf32]) <- (1x-1x128xf32, 1x-1x128xf32, 1x-1x128xf32, 1x-1x128xf32)
        combine_4 = [flatten_1, flatten_0, flatten_2, flatten_3]
        del flatten_0, flatten_1, flatten_2, flatten_3

        # pd_op.concat: (1x-1x512xf32) <- ([1x-1x128xf32, 1x-1x128xf32, 1x-1x128xf32, 1x-1x128xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_4, full_8)
        del combine_4, full_8

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x512xf32, 512x256xf32)
        matmul_0 = paddle._C_ops.matmul(concat_0, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_2)
        del parameter_2

        # pd_op.relu: (1x-1x256xf32) <- (1x-1x256xf32)
        relu_0 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_0 = paddle._C_ops.add(matmul_1, parameter_0)
        del concat_0, matmul_0, matmul_1, multiply_0, parameter_0, relu_0

        return add_0
