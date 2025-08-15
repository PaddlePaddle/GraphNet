import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input0: paddle.Tensor,
        t0: paddle.Tensor,
        t1: paddle.Tensor,
        t2: paddle.Tensor,
        t3: paddle.Tensor,
        t4: paddle.Tensor,
        t5: paddle.Tensor,
        t6: paddle.Tensor,
        t7: paddle.Tensor,
        t8: paddle.Tensor,
        t9: paddle.Tensor,
        t10: paddle.Tensor,
        t11: paddle.Tensor,
        t12: paddle.Tensor,
        t13: paddle.Tensor,
        t14: paddle.Tensor,
        t15: paddle.Tensor,
        t16: paddle.Tensor,
        t17: paddle.Tensor,
        t18: paddle.Tensor,
        t19: paddle.Tensor,
        t20: paddle.Tensor,
        t21: paddle.Tensor,
        t22: paddle.Tensor,
        t23: paddle.Tensor,
        t24: paddle.Tensor,
        t25: paddle.Tensor,
        t26: paddle.Tensor,
        t27: paddle.Tensor,
        t28: paddle.Tensor,
        t29: paddle.Tensor,
        t30: paddle.Tensor,
        t31: paddle.Tensor,
        t32: paddle.Tensor,
        t33: paddle.Tensor,
        t34: paddle.Tensor,
        t35: paddle.Tensor,
        t36: paddle.Tensor,
        t37: paddle.Tensor,
        t38: paddle.Tensor,
        t39: paddle.Tensor,
        t40: paddle.Tensor,
        t41: paddle.Tensor,
        t42: paddle.Tensor,
        t43: paddle.Tensor,
        t44: paddle.Tensor,
        t45: paddle.Tensor,
        t46: paddle.Tensor,
        t47: paddle.Tensor,
        t48: paddle.Tensor,
        t49: paddle.Tensor,
        t50: paddle.Tensor,
        t51: paddle.Tensor,
        t52: paddle.Tensor,
        t53: paddle.Tensor,
        t54: paddle.Tensor,
        t55: paddle.Tensor,
        t56: paddle.Tensor,
        t57: paddle.Tensor,
        t58: paddle.Tensor,
        t59: paddle.Tensor,
        t60: paddle.Tensor,
        t61: paddle.Tensor,
        t62: paddle.Tensor,
        t63: paddle.Tensor,
        t64: paddle.Tensor,
        t65: paddle.Tensor,
        t66: paddle.Tensor,
        t67: paddle.Tensor,
        t68: paddle.Tensor,
        t69: paddle.Tensor,
        t70: paddle.Tensor,
        t71: paddle.Tensor,
        t72: paddle.Tensor,
        t73: paddle.Tensor,
        t74: paddle.Tensor,
        t75: paddle.Tensor,
        t76: paddle.Tensor,
        t77: paddle.Tensor,
        t78: paddle.Tensor,
        t79: paddle.Tensor,
        t80: paddle.Tensor,
        t81: paddle.Tensor,
        t82: paddle.Tensor,
        t83: paddle.Tensor,
        t84: paddle.Tensor,
        t85: paddle.Tensor,
        t86: paddle.Tensor,
        t87: paddle.Tensor,
        t88: paddle.Tensor,
        t89: paddle.Tensor,
        t90: paddle.Tensor,
        t91: paddle.Tensor,
        t92: paddle.Tensor,
        t93: paddle.Tensor,
        t94: paddle.Tensor,
        t95: paddle.Tensor,
        t96: paddle.Tensor,
        t97: paddle.Tensor,
        t98: paddle.Tensor,
        t99: paddle.Tensor,
        t100: paddle.Tensor,
        t101: paddle.Tensor,
        t102: paddle.Tensor,
        t103: paddle.Tensor,
        t104: paddle.Tensor,
        t105: paddle.Tensor,
        t106: paddle.Tensor,
        t107: paddle.Tensor,
        t108: paddle.Tensor,
        t109: paddle.Tensor,
        t110: paddle.Tensor,
        t111: paddle.Tensor,
        t112: paddle.Tensor,
        t113: paddle.Tensor,
        t114: paddle.Tensor,
        t115: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        t116 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t117, t118, t119, t120, t121, t122 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t116,
                t1,
                t2,
                t3,
                t4,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t116, t4, t3, t2, t1

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t123 = paddle._C_ops.relu(t117)
        del t117

        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 32x32x3x3xf32)
        t124 = paddle._C_ops.conv2d(
            t123, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5, t123

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t125, t126, t127, t128, t129, t130 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t124,
                t6,
                t7,
                t8,
                t9,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t124, t9, t8, t7, t6

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t131 = paddle._C_ops.relu(t125)
        del t125

        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x32x112x112xf32, 64x32x3x3xf32)
        t132 = paddle._C_ops.conv2d(
            t131, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10, t131

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t133, t134, t135, t136, t137, t138 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t132,
                t11,
                t12,
                t13,
                t14,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t132, t14, t13, t12, t11

        # pd_op.relu: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        t139 = paddle._C_ops.relu(t133)
        del t133

        # pd_op.full_int_array: (2xi64) <- ()
        t140 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        t141 = paddle._C_ops.pool2d(
            t139,
            t140,
            [2, 2],
            [1, 1],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )
        del t140, t139

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t142 = paddle._C_ops.conv2d(
            t141, t15, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t143, t144, t145, t146, t147, t148 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t142,
                t16,
                t17,
                t18,
                t19,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t142, t16, t19, t18, t17

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t149 = paddle._C_ops.relu(t143)
        del t143

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t150 = paddle._C_ops.conv2d(
            t149, t20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20, t149

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t151, t152, t153, t154, t155, t156 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t150,
                t21,
                t22,
                t23,
                t24,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t150, t24, t23, t22, t21

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        t157 = paddle._C_ops.conv2d(
            t141, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25, t141

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t158, t159, t160, t161, t162, t163 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t157,
                t26,
                t27,
                t28,
                t29,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t157, t29, t28, t27, t26

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t164 = paddle._C_ops.add(t151, t158)
        del t151, t158

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t165 = paddle._C_ops.relu(t164)
        del t164

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t166 = paddle._C_ops.conv2d(
            t165, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t167, t168, t169, t170, t171, t172 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t166,
                t31,
                t32,
                t33,
                t34,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t166, t34, t33, t32, t31

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t173 = paddle._C_ops.relu(t167)
        del t167

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t174 = paddle._C_ops.conv2d(
            t173, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35, t173

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t175, t176, t177, t178, t179, t180 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t174,
                t36,
                t37,
                t38,
                t39,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t174, t39, t38, t37, t36

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t181 = paddle._C_ops.add(t175, t165)
        del t175, t165

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t182 = paddle._C_ops.relu(t181)
        del t181

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x56x56xf32, 128x64x3x3xf32)
        t183 = paddle._C_ops.conv2d(
            t182, t40, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t184, t185, t186, t187, t188, t189 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t183,
                t41,
                t42,
                t43,
                t44,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t183, t44, t43, t42, t41

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t190 = paddle._C_ops.relu(t184)
        del t184

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t191 = paddle._C_ops.conv2d(
            t190, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45, t190

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t192, t193, t194, t195, t196, t197 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t191,
                t46,
                t47,
                t48,
                t49,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t191, t49, t48, t47, t46

        # pd_op.full_int_array: (2xi64) <- ()
        t198 = [2, 2]

        # pd_op.pool2d: (-1x64x28x28xf32) <- (-1x64x56x56xf32, 2xi64)
        t199 = paddle._C_ops.pool2d(
            t182, t198, [2, 2], [0, 0], True, True, "NCHW", "avg", False, False, "SAME"
        )
        del t182

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x28x28xf32, 128x64x1x1xf32)
        t200 = paddle._C_ops.conv2d(
            t199, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50, t199

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t201, t202, t203, t204, t205, t206 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t200,
                t51,
                t52,
                t53,
                t54,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t200, t54, t53, t52, t51

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t207 = paddle._C_ops.add(t192, t201)
        del t192, t201

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t208 = paddle._C_ops.relu(t207)
        del t207

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t209 = paddle._C_ops.conv2d(
            t208, t55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t210, t211, t212, t213, t214, t215 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t209,
                t56,
                t57,
                t58,
                t59,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t209, t59, t58, t57, t56

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t216 = paddle._C_ops.relu(t210)
        del t210

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t217 = paddle._C_ops.conv2d(
            t216, t60, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60, t216

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t218, t219, t220, t221, t222, t223 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t217,
                t61,
                t62,
                t63,
                t64,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t217, t64, t63, t62, t61

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t224 = paddle._C_ops.add(t218, t208)
        del t218, t208

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t225 = paddle._C_ops.relu(t224)
        del t224

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x28x28xf32, 256x128x3x3xf32)
        t226 = paddle._C_ops.conv2d(
            t225, t65, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t227, t228, t229, t230, t231, t232 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t226,
                t66,
                t67,
                t68,
                t69,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t226, t69, t68, t67, t66

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t233 = paddle._C_ops.relu(t227)
        del t227

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t234 = paddle._C_ops.conv2d(
            t233, t70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70, t233

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t235, t236, t237, t238, t239, t240 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t234,
                t71,
                t72,
                t73,
                t74,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t234, t74, t73, t72, t71

        # pd_op.pool2d: (-1x128x14x14xf32) <- (-1x128x28x28xf32, 2xi64)
        t241 = paddle._C_ops.pool2d(
            t225, t198, [2, 2], [0, 0], True, True, "NCHW", "avg", False, False, "SAME"
        )
        del t225

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x14x14xf32, 256x128x1x1xf32)
        t242 = paddle._C_ops.conv2d(
            t241, t75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75, t241

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t243, t244, t245, t246, t247, t248 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t242,
                t76,
                t77,
                t78,
                t79,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t242, t79, t78, t77, t76

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t249 = paddle._C_ops.add(t235, t243)
        del t235, t243

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t250 = paddle._C_ops.relu(t249)
        del t249

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t251 = paddle._C_ops.conv2d(
            t250, t80, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t252, t253, t254, t255, t256, t257 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t251,
                t81,
                t82,
                t83,
                t84,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t251, t84, t83, t82, t81

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t258 = paddle._C_ops.relu(t252)
        del t252

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t259 = paddle._C_ops.conv2d(
            t258, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85, t258

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t260, t261, t262, t263, t264, t265 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t259,
                t86,
                t87,
                t88,
                t89,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t259, t89, t88, t87, t86

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t266 = paddle._C_ops.add(t260, t250)
        del t260, t250

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t267 = paddle._C_ops.relu(t266)
        del t266

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x14x14xf32, 512x256x3x3xf32)
        t268 = paddle._C_ops.conv2d(
            t267, t90, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t269, t270, t271, t272, t273, t274 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t268,
                t91,
                t92,
                t93,
                t94,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t268, t94, t93, t92, t91

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t275 = paddle._C_ops.relu(t269)
        del t269

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t276 = paddle._C_ops.conv2d(
            t275, t95, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95, t275

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t277, t278, t279, t280, t281, t282 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t276,
                t96,
                t97,
                t98,
                t99,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t276, t99, t98, t97, t96

        # pd_op.pool2d: (-1x256x7x7xf32) <- (-1x256x14x14xf32, 2xi64)
        t283 = paddle._C_ops.pool2d(
            t267, t198, [2, 2], [0, 0], True, True, "NCHW", "avg", False, False, "SAME"
        )
        del t198, t267

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x7x7xf32, 512x256x1x1xf32)
        t284 = paddle._C_ops.conv2d(
            t283, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100, t283

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t285, t286, t287, t288, t289, t290 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t284,
                t101,
                t102,
                t103,
                t104,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t284, t104, t103, t102, t101

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t291 = paddle._C_ops.add(t277, t285)
        del t277, t285

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t292 = paddle._C_ops.relu(t291)
        del t291

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t293 = paddle._C_ops.conv2d(
            t292, t105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t294, t295, t296, t297, t298, t299 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t293,
                t106,
                t107,
                t108,
                t109,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t293, t106, t109, t108, t107

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t300 = paddle._C_ops.relu(t294)
        del t294

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t301 = paddle._C_ops.conv2d(
            t300, t110, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110, t300

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t302, t303, t304, t305, t306, t307 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t301,
                t111,
                t112,
                t113,
                t114,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t301, t114, t113, t112, t111

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t308 = paddle._C_ops.add(t302, t292)
        del t302, t292

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t309 = paddle._C_ops.relu(t308)
        del t308

        # pd_op.full_int_array: (2xi64) <- ()
        t310 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x7x7xf32, 2xi64)
        t311 = paddle._C_ops.pool2d(
            t309,
            t310,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )
        del t310, t309

        # pd_op.flatten: (-1x512xf32) <- (-1x512x1x1xf32)
        t312 = paddle._C_ops.flatten(t311, 1, 3)
        del t311

        # pd_op.matmul: (-1x102xf32) <- (-1x512xf32, 512x102xf32)
        t313 = paddle._C_ops.matmul(t312, t115, False, False)
        del t312, t115

        return t313
