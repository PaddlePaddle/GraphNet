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
        t116: paddle.Tensor,
        t117: paddle.Tensor,
        t118: paddle.Tensor,
        t119: paddle.Tensor,
        t120: paddle.Tensor,
        t121: paddle.Tensor,
        t122: paddle.Tensor,
        t123: paddle.Tensor,
        t124: paddle.Tensor,
        t125: paddle.Tensor,
        t126: paddle.Tensor,
        t127: paddle.Tensor,
        t128: paddle.Tensor,
        t129: paddle.Tensor,
        t130: paddle.Tensor,
        t131: paddle.Tensor,
        t132: paddle.Tensor,
        t133: paddle.Tensor,
        t134: paddle.Tensor,
        t135: paddle.Tensor,
        t136: paddle.Tensor,
        t137: paddle.Tensor,
        t138: paddle.Tensor,
        t139: paddle.Tensor,
        t140: paddle.Tensor,
        t141: paddle.Tensor,
        t142: paddle.Tensor,
        t143: paddle.Tensor,
        t144: paddle.Tensor,
    ):
        t145 = None
        # pd_op.conv2d: (256x8x112x112xf32) <- (256x3x224x224xf32, 8x3x3x3xf32)
        t146 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t147, t148, t149, t150, t151, t152 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t146,
                t1,
                t2,
                t3,
                t4,
                False,
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
        del t4, t3, t2, t1

        # pd_op.hardswish: (256x8x112x112xf32) <- (256x8x112x112xf32)
        t153 = paddle._C_ops.hardswish(t147)

        # pd_op.depthwise_conv2d: (256x8x112x112xf32) <- (256x8x112x112xf32, 8x1x3x3xf32)
        t154 = paddle._C_ops.depthwise_conv2d(
            t153, t5, [1, 1], [1, 1], "EXPLICIT", 8, [1, 1], "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t155, t156, t157, t158, t159, t160 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t154,
                t6,
                t7,
                t8,
                t9,
                False,
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
        del t9, t8, t7, t6

        # pd_op.hardswish: (256x8x112x112xf32) <- (256x8x112x112xf32)
        t161 = paddle._C_ops.hardswish(t155)

        # pd_op.conv2d: (256x8x112x112xf32) <- (256x8x112x112xf32, 8x8x1x1xf32)
        t162 = paddle._C_ops.conv2d(
            t161, t10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t163, t164, t165, t166, t167, t168 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t162,
                t11,
                t12,
                t13,
                t14,
                False,
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
        del t14, t13, t12, t11

        # pd_op.hardswish: (256x8x112x112xf32) <- (256x8x112x112xf32)
        t169 = paddle._C_ops.hardswish(t163)

        # pd_op.depthwise_conv2d: (256x8x56x56xf32) <- (256x8x112x112xf32, 8x1x3x3xf32)
        t170 = paddle._C_ops.depthwise_conv2d(
            t169, t15, [2, 2], [1, 1], "EXPLICIT", 8, [1, 1], "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (256x8x56x56xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (256x8x56x56xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t171, t172, t173, t174, t175, t176 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t170,
                t16,
                t17,
                t18,
                t19,
                False,
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
        del t19, t18, t17, t16

        # pd_op.hardswish: (256x8x56x56xf32) <- (256x8x56x56xf32)
        t177 = paddle._C_ops.hardswish(t171)

        # pd_op.conv2d: (256x16x56x56xf32) <- (256x8x56x56xf32, 16x8x1x1xf32)
        t178 = paddle._C_ops.conv2d(
            t177, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (256x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (256x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t179, t180, t181, t182, t183, t184 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t178,
                t21,
                t22,
                t23,
                t24,
                False,
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
        del t24, t23, t22, t21

        # pd_op.hardswish: (256x16x56x56xf32) <- (256x16x56x56xf32)
        t185 = paddle._C_ops.hardswish(t179)

        # pd_op.depthwise_conv2d: (256x16x56x56xf32) <- (256x16x56x56xf32, 16x1x3x3xf32)
        t186 = paddle._C_ops.depthwise_conv2d(
            t185, t25, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (256x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (256x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t187, t188, t189, t190, t191, t192 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t186,
                t26,
                t27,
                t28,
                t29,
                False,
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
        del t29, t28, t27, t26

        # pd_op.hardswish: (256x16x56x56xf32) <- (256x16x56x56xf32)
        t193 = paddle._C_ops.hardswish(t187)

        # pd_op.conv2d: (256x16x56x56xf32) <- (256x16x56x56xf32, 16x16x1x1xf32)
        t194 = paddle._C_ops.conv2d(
            t193, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (256x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (256x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t195, t196, t197, t198, t199, t200 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t194,
                t31,
                t32,
                t33,
                t34,
                False,
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
        del t34, t33, t32, t31

        # pd_op.hardswish: (256x16x56x56xf32) <- (256x16x56x56xf32)
        t201 = paddle._C_ops.hardswish(t195)

        # pd_op.depthwise_conv2d: (256x16x28x28xf32) <- (256x16x56x56xf32, 16x1x3x3xf32)
        t202 = paddle._C_ops.depthwise_conv2d(
            t201, t35, [2, 2], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (256x16x28x28xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (256x16x28x28xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t203, t204, t205, t206, t207, t208 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t202,
                t36,
                t37,
                t38,
                t39,
                False,
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
        del t39, t38, t37, t36

        # pd_op.hardswish: (256x16x28x28xf32) <- (256x16x28x28xf32)
        t209 = paddle._C_ops.hardswish(t203)

        # pd_op.conv2d: (256x32x28x28xf32) <- (256x16x28x28xf32, 32x16x1x1xf32)
        t210 = paddle._C_ops.conv2d(
            t209, t40, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t211, t212, t213, t214, t215, t216 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t210,
                t41,
                t42,
                t43,
                t44,
                False,
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
        del t44, t43, t42, t41

        # pd_op.hardswish: (256x32x28x28xf32) <- (256x32x28x28xf32)
        t217 = paddle._C_ops.hardswish(t211)

        # pd_op.depthwise_conv2d: (256x32x28x28xf32) <- (256x32x28x28xf32, 32x1x3x3xf32)
        t218 = paddle._C_ops.depthwise_conv2d(
            t217, t45, [1, 1], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t219, t220, t221, t222, t223, t224 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t218,
                t46,
                t47,
                t48,
                t49,
                False,
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
        del t49, t48, t47, t46

        # pd_op.hardswish: (256x32x28x28xf32) <- (256x32x28x28xf32)
        t225 = paddle._C_ops.hardswish(t219)

        # pd_op.conv2d: (256x32x28x28xf32) <- (256x32x28x28xf32, 32x32x1x1xf32)
        t226 = paddle._C_ops.conv2d(
            t225, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t227, t228, t229, t230, t231, t232 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t226,
                t51,
                t52,
                t53,
                t54,
                False,
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
        del t54, t53, t52, t51

        # pd_op.hardswish: (256x32x28x28xf32) <- (256x32x28x28xf32)
        t233 = paddle._C_ops.hardswish(t227)

        # pd_op.depthwise_conv2d: (256x32x14x14xf32) <- (256x32x28x28xf32, 32x1x3x3xf32)
        t234 = paddle._C_ops.depthwise_conv2d(
            t233, t55, [2, 2], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (256x32x14x14xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x14x14xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t235, t236, t237, t238, t239, t240 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t234,
                t56,
                t57,
                t58,
                t59,
                False,
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
        del t59, t58, t57, t56

        # pd_op.hardswish: (256x32x14x14xf32) <- (256x32x14x14xf32)
        t241 = paddle._C_ops.hardswish(t235)

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x32x14x14xf32, 64x32x1x1xf32)
        t242 = paddle._C_ops.conv2d(
            t241, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t243, t244, t245, t246, t247, t248 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t242,
                t61,
                t62,
                t63,
                t64,
                False,
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
        del t64, t63, t62, t61

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t249 = paddle._C_ops.hardswish(t243)

        # pd_op.depthwise_conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x1x5x5xf32)
        t250 = paddle._C_ops.depthwise_conv2d(
            t249, t65, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t251, t252, t253, t254, t255, t256 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t250,
                t66,
                t67,
                t68,
                t69,
                False,
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
        del t69, t68, t67, t66

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t257 = paddle._C_ops.hardswish(t251)

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x64x1x1xf32)
        t258 = paddle._C_ops.conv2d(
            t257, t70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t259, t260, t261, t262, t263, t264 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t258,
                t71,
                t72,
                t73,
                t74,
                False,
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
        del t74, t73, t72, t71

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t265 = paddle._C_ops.hardswish(t259)

        # pd_op.depthwise_conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x1x5x5xf32)
        t266 = paddle._C_ops.depthwise_conv2d(
            t265, t75, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t267, t268, t269, t270, t271, t272 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t266,
                t76,
                t77,
                t78,
                t79,
                False,
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
        del t79, t78, t77, t76

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t273 = paddle._C_ops.hardswish(t267)

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x64x1x1xf32)
        t274 = paddle._C_ops.conv2d(
            t273, t80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t275, t276, t277, t278, t279, t280 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t274,
                t81,
                t82,
                t83,
                t84,
                False,
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
        del t84, t83, t82, t81

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t281 = paddle._C_ops.hardswish(t275)

        # pd_op.depthwise_conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x1x5x5xf32)
        t282 = paddle._C_ops.depthwise_conv2d(
            t281, t85, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t283, t284, t285, t286, t287, t288 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t282,
                t86,
                t87,
                t88,
                t89,
                False,
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
        del t89, t88, t87, t86

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t289 = paddle._C_ops.hardswish(t283)

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x64x1x1xf32)
        t290 = paddle._C_ops.conv2d(
            t289, t90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t291, t292, t293, t294, t295, t296 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t290,
                t91,
                t92,
                t93,
                t94,
                False,
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
        del t94, t93, t92, t91

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t297 = paddle._C_ops.hardswish(t291)

        # pd_op.depthwise_conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x1x5x5xf32)
        t298 = paddle._C_ops.depthwise_conv2d(
            t297, t95, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t299, t300, t301, t302, t303, t304 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t298,
                t96,
                t97,
                t98,
                t99,
                False,
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
        del t99, t98, t97, t96

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t305 = paddle._C_ops.hardswish(t299)

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x64x1x1xf32)
        t306 = paddle._C_ops.conv2d(
            t305, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t307, t308, t309, t310, t311, t312 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t306,
                t101,
                t102,
                t103,
                t104,
                False,
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
        del t104, t103, t102, t101

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t313 = paddle._C_ops.hardswish(t307)

        # pd_op.depthwise_conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x1x5x5xf32)
        t314 = paddle._C_ops.depthwise_conv2d(
            t313, t105, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t315, t316, t317, t318, t319, t320 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t314,
                t106,
                t107,
                t108,
                t109,
                False,
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
        del t109, t108, t107, t106

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t321 = paddle._C_ops.hardswish(t315)

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x64x1x1xf32)
        t322 = paddle._C_ops.conv2d(
            t321, t110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t323, t324, t325, t326, t327, t328 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t322,
                t111,
                t112,
                t113,
                t114,
                False,
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
        del t114, t113, t112, t111

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t329 = paddle._C_ops.hardswish(t323)

        # pd_op.depthwise_conv2d: (256x64x7x7xf32) <- (256x64x14x14xf32, 64x1x5x5xf32)
        t330 = paddle._C_ops.depthwise_conv2d(
            t329, t115, [2, 2], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (256x64x7x7xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x7x7xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t331, t332, t333, t334, t335, t336 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t330,
                t116,
                t117,
                t118,
                t119,
                False,
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
        del t119, t118, t117, t116

        # pd_op.hardswish: (256x64x7x7xf32) <- (256x64x7x7xf32)
        t337 = paddle._C_ops.hardswish(t331)

        # pd_op.full_int_array: (2xi64) <- ()
        t338 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        t339 = t338

        # pd_op.assign: (2xi64) <- (2xi64)
        t340 = t338

        # pd_op.pool2d: (256x64x1x1xf32) <- (256x64x7x7xf32, 2xi64)
        t341 = paddle._C_ops.pool2d(
            t337,
            t338,
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

        # pd_op.conv2d: (256x16x1x1xf32) <- (256x64x1x1xf32, 16x64x1x1xf32)
        t342 = paddle._C_ops.conv2d(
            t341, t120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.full_int_array: (4xi64) <- ()
        t343 = [1, -1, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32) <- (16xf32, 4xi64)
        t344 = paddle._C_ops.reshape(t121, t343)
        del t121

        # pd_op.add: (256x16x1x1xf32) <- (256x16x1x1xf32, 1x16x1x1xf32)
        t345 = paddle._C_ops.add(t342, t344)

        # pd_op.relu: (256x16x1x1xf32) <- (256x16x1x1xf32)
        t346 = paddle._C_ops.relu(t345)
        del t345

        # pd_op.conv2d: (256x64x1x1xf32) <- (256x16x1x1xf32, 64x16x1x1xf32)
        t347 = paddle._C_ops.conv2d(
            t346, t122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t122

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t348 = paddle._C_ops.reshape(t123, t343)
        del t123

        # pd_op.add: (256x64x1x1xf32) <- (256x64x1x1xf32, 1x64x1x1xf32)
        t349 = paddle._C_ops.add(t347, t348)

        # pd_op.hardsigmoid: (256x64x1x1xf32) <- (256x64x1x1xf32)
        t350 = paddle._C_ops.hardsigmoid(t349, float("0.166667"), float("0.5"))
        del t349

        # pd_op.multiply: (256x64x7x7xf32) <- (256x64x7x7xf32, 256x64x1x1xf32)
        t351 = paddle._C_ops.multiply(t337, t350)

        # pd_op.conv2d: (256x128x7x7xf32) <- (256x64x7x7xf32, 128x64x1x1xf32)
        t352 = paddle._C_ops.conv2d(
            t351, t124, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t124

        # pd_op.batch_norm_: (256x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (256x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t353, t354, t355, t356, t357, t358 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t352,
                t125,
                t126,
                t127,
                t128,
                False,
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
        del t128, t127, t126, t125

        # pd_op.hardswish: (256x128x7x7xf32) <- (256x128x7x7xf32)
        t359 = paddle._C_ops.hardswish(t353)

        # pd_op.depthwise_conv2d: (256x128x7x7xf32) <- (256x128x7x7xf32, 128x1x5x5xf32)
        t360 = paddle._C_ops.depthwise_conv2d(
            t359, t129, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t129

        # pd_op.batch_norm_: (256x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (256x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t361, t362, t363, t364, t365, t366 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t360,
                t130,
                t131,
                t132,
                t133,
                False,
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
        del t133, t132, t131, t130

        # pd_op.hardswish: (256x128x7x7xf32) <- (256x128x7x7xf32)
        t367 = paddle._C_ops.hardswish(t361)

        # pd_op.pool2d: (256x128x1x1xf32) <- (256x128x7x7xf32, 2xi64)
        t368 = paddle._C_ops.pool2d(
            t367,
            t338,
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

        # pd_op.conv2d: (256x32x1x1xf32) <- (256x128x1x1xf32, 32x128x1x1xf32)
        t369 = paddle._C_ops.conv2d(
            t368, t134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t134

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t370 = paddle._C_ops.reshape(t135, t343)
        del t135

        # pd_op.add: (256x32x1x1xf32) <- (256x32x1x1xf32, 1x32x1x1xf32)
        t371 = paddle._C_ops.add(t369, t370)

        # pd_op.relu: (256x32x1x1xf32) <- (256x32x1x1xf32)
        t372 = paddle._C_ops.relu(t371)
        del t371

        # pd_op.conv2d: (256x128x1x1xf32) <- (256x32x1x1xf32, 128x32x1x1xf32)
        t373 = paddle._C_ops.conv2d(
            t372, t136, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t136

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t374 = paddle._C_ops.reshape(t137, t343)
        del t343, t137

        # pd_op.add: (256x128x1x1xf32) <- (256x128x1x1xf32, 1x128x1x1xf32)
        t375 = paddle._C_ops.add(t373, t374)

        # pd_op.hardsigmoid: (256x128x1x1xf32) <- (256x128x1x1xf32)
        t376 = paddle._C_ops.hardsigmoid(t375, float("0.166667"), float("0.5"))
        del t375

        # pd_op.multiply: (256x128x7x7xf32) <- (256x128x7x7xf32, 256x128x1x1xf32)
        t377 = paddle._C_ops.multiply(t367, t376)

        # pd_op.conv2d: (256x128x7x7xf32) <- (256x128x7x7xf32, 128x128x1x1xf32)
        t378 = paddle._C_ops.conv2d(
            t377, t138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t138

        # pd_op.batch_norm_: (256x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (256x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t379, t380, t381, t382, t383, t384 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t378,
                t139,
                t140,
                t141,
                t142,
                False,
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
        del t142, t141, t140, t139

        # pd_op.hardswish: (256x128x7x7xf32) <- (256x128x7x7xf32)
        t385 = paddle._C_ops.hardswish(t379)

        # pd_op.pool2d: (256x128x1x1xf32) <- (256x128x7x7xf32, 2xi64)
        t386 = paddle._C_ops.pool2d(
            t385,
            t338,
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

        # pd_op.conv2d: (256x1280x1x1xf32) <- (256x128x1x1xf32, 1280x128x1x1xf32)
        t387 = paddle._C_ops.conv2d(
            t386, t143, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t143

        # pd_op.hardswish: (256x1280x1x1xf32) <- (256x1280x1x1xf32)
        t388 = paddle._C_ops.hardswish(t387)

        # pd_op.full: (1xf32) <- ()
        t389 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (256x1280x1x1xf32, 256x1280x1x1xui8) <- (256x1280x1x1xf32, None, 1xf32)
        t390, t391 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                t388, None, t389, False, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t388

        # pd_op.flatten: (256x1280xf32) <- (256x1280x1x1xf32)
        t392 = paddle._C_ops.flatten(t390, 1, 3)

        # pd_op.matmul: (256x102xf32) <- (256x1280xf32, 1280x102xf32)
        t393 = paddle._C_ops.matmul(t392, t144, False, False)
        del t144

        return (
            t146,
            t147,
            t148,
            t149,
            t150,
            t151,
            t152,
            t153,
            t154,
            t155,
            t156,
            t157,
            t158,
            t159,
            t160,
            t161,
            t162,
            t163,
            t164,
            t165,
            t166,
            t167,
            t168,
            t169,
            t170,
            t171,
            t172,
            t173,
            t174,
            t175,
            t176,
            t177,
            t178,
            t179,
            t180,
            t181,
            t182,
            t183,
            t184,
            t185,
            t186,
            t187,
            t188,
            t189,
            t190,
            t191,
            t192,
            t193,
            t194,
            t195,
            t196,
            t197,
            t198,
            t199,
            t200,
            t201,
            t202,
            t203,
            t204,
            t205,
            t206,
            t207,
            t208,
            t209,
            t210,
            t211,
            t212,
            t213,
            t214,
            t215,
            t216,
            t217,
            t218,
            t219,
            t220,
            t221,
            t222,
            t223,
            t224,
            t225,
            t226,
            t227,
            t228,
            t229,
            t230,
            t231,
            t232,
            t233,
            t234,
            t235,
            t236,
            t237,
            t238,
            t239,
            t240,
            t241,
            t242,
            t243,
            t244,
            t245,
            t246,
            t247,
            t248,
            t249,
            t250,
            t251,
            t252,
            t253,
            t254,
            t255,
            t256,
            t257,
            t258,
            t259,
            t260,
            t261,
            t262,
            t263,
            t264,
            t265,
            t266,
            t267,
            t268,
            t269,
            t270,
            t271,
            t272,
            t273,
            t274,
            t275,
            t276,
            t277,
            t278,
            t279,
            t280,
            t281,
            t282,
            t283,
            t284,
            t285,
            t286,
            t287,
            t288,
            t289,
            t290,
            t291,
            t292,
            t293,
            t294,
            t295,
            t296,
            t297,
            t298,
            t299,
            t300,
            t301,
            t302,
            t303,
            t304,
            t305,
            t306,
            t307,
            t308,
            t309,
            t310,
            t311,
            t312,
            t313,
            t314,
            t315,
            t316,
            t317,
            t318,
            t319,
            t320,
            t321,
            t322,
            t323,
            t324,
            t325,
            t326,
            t327,
            t328,
            t329,
            t330,
            t331,
            t332,
            t333,
            t334,
            t335,
            t336,
            t337,
            t338,
            t339,
            t340,
            t341,
            t342,
            t344,
            t346,
            t347,
            t348,
            t350,
            t351,
            t352,
            t353,
            t354,
            t355,
            t356,
            t357,
            t358,
            t359,
            t360,
            t361,
            t362,
            t363,
            t364,
            t365,
            t366,
            t367,
            t368,
            t369,
            t370,
            t372,
            t373,
            t374,
            t376,
            t377,
            t378,
            t379,
            t380,
            t381,
            t382,
            t383,
            t384,
            t385,
            t386,
            t387,
            t389,
            t390,
            t391,
            t392,
            t393,
        )
