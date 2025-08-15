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
    ):
        # pd_op.conv2d: (64x8x112x112xf32) <- (64x3x224x224xf32, 8x3x3x3xf32)
        t136 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (64x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (64x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t137, t138, t139, t140, t141, t142 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t136,
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

        # pd_op.relu: (64x8x112x112xf32) <- (64x8x112x112xf32)
        t143 = paddle._C_ops.relu(t137)
        del t137

        # pd_op.depthwise_conv2d: (64x8x112x112xf32) <- (64x8x112x112xf32, 8x1x3x3xf32)
        t144 = paddle._C_ops.depthwise_conv2d(
            t143, t5, [1, 1], [1, 1], "EXPLICIT", 8, [1, 1], "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (64x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (64x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t145, t146, t147, t148, t149, t150 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t144,
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

        # pd_op.relu: (64x8x112x112xf32) <- (64x8x112x112xf32)
        t151 = paddle._C_ops.relu(t145)
        del t145

        # pd_op.conv2d: (64x16x112x112xf32) <- (64x8x112x112xf32, 16x8x1x1xf32)
        t152 = paddle._C_ops.conv2d(
            t151, t10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (64x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (64x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t153, t154, t155, t156, t157, t158 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t152,
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

        # pd_op.relu: (64x16x112x112xf32) <- (64x16x112x112xf32)
        t159 = paddle._C_ops.relu(t153)
        del t153

        # pd_op.depthwise_conv2d: (64x16x56x56xf32) <- (64x16x112x112xf32, 16x1x3x3xf32)
        t160 = paddle._C_ops.depthwise_conv2d(
            t159, t15, [2, 2], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (64x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (64x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t161, t162, t163, t164, t165, t166 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t160,
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

        # pd_op.relu: (64x16x56x56xf32) <- (64x16x56x56xf32)
        t167 = paddle._C_ops.relu(t161)
        del t161

        # pd_op.conv2d: (64x32x56x56xf32) <- (64x16x56x56xf32, 32x16x1x1xf32)
        t168 = paddle._C_ops.conv2d(
            t167, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t169, t170, t171, t172, t173, t174 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t168,
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

        # pd_op.relu: (64x32x56x56xf32) <- (64x32x56x56xf32)
        t175 = paddle._C_ops.relu(t169)
        del t169

        # pd_op.depthwise_conv2d: (64x32x56x56xf32) <- (64x32x56x56xf32, 32x1x3x3xf32)
        t176 = paddle._C_ops.depthwise_conv2d(
            t175, t25, [1, 1], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t177, t178, t179, t180, t181, t182 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t176,
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

        # pd_op.relu: (64x32x56x56xf32) <- (64x32x56x56xf32)
        t183 = paddle._C_ops.relu(t177)
        del t177

        # pd_op.conv2d: (64x32x56x56xf32) <- (64x32x56x56xf32, 32x32x1x1xf32)
        t184 = paddle._C_ops.conv2d(
            t183, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t185, t186, t187, t188, t189, t190 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t184,
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

        # pd_op.relu: (64x32x56x56xf32) <- (64x32x56x56xf32)
        t191 = paddle._C_ops.relu(t185)
        del t185

        # pd_op.depthwise_conv2d: (64x32x28x28xf32) <- (64x32x56x56xf32, 32x1x3x3xf32)
        t192 = paddle._C_ops.depthwise_conv2d(
            t191, t35, [2, 2], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (64x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t193, t194, t195, t196, t197, t198 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t192,
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
        del t36, t39, t38, t37

        # pd_op.relu: (64x32x28x28xf32) <- (64x32x28x28xf32)
        t199 = paddle._C_ops.relu(t193)
        del t193

        # pd_op.conv2d: (64x64x28x28xf32) <- (64x32x28x28xf32, 64x32x1x1xf32)
        t200 = paddle._C_ops.conv2d(
            t199, t40, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (64x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (64x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t201, t202, t203, t204, t205, t206 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t200,
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

        # pd_op.relu: (64x64x28x28xf32) <- (64x64x28x28xf32)
        t207 = paddle._C_ops.relu(t201)
        del t201

        # pd_op.depthwise_conv2d: (64x64x28x28xf32) <- (64x64x28x28xf32, 64x1x3x3xf32)
        t208 = paddle._C_ops.depthwise_conv2d(
            t207, t45, [1, 1], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (64x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (64x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t209, t210, t211, t212, t213, t214 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t208,
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

        # pd_op.relu: (64x64x28x28xf32) <- (64x64x28x28xf32)
        t215 = paddle._C_ops.relu(t209)
        del t209

        # pd_op.conv2d: (64x64x28x28xf32) <- (64x64x28x28xf32, 64x64x1x1xf32)
        t216 = paddle._C_ops.conv2d(
            t215, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (64x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (64x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t217, t218, t219, t220, t221, t222 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t216,
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

        # pd_op.relu: (64x64x28x28xf32) <- (64x64x28x28xf32)
        t223 = paddle._C_ops.relu(t217)
        del t217

        # pd_op.depthwise_conv2d: (64x64x14x14xf32) <- (64x64x28x28xf32, 64x1x3x3xf32)
        t224 = paddle._C_ops.depthwise_conv2d(
            t223, t55, [2, 2], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (64x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (64x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t225, t226, t227, t228, t229, t230 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t224,
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

        # pd_op.relu: (64x64x14x14xf32) <- (64x64x14x14xf32)
        t231 = paddle._C_ops.relu(t225)
        del t225

        # pd_op.conv2d: (64x128x14x14xf32) <- (64x64x14x14xf32, 128x64x1x1xf32)
        t232 = paddle._C_ops.conv2d(
            t231, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t233, t234, t235, t236, t237, t238 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t232,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t239 = paddle._C_ops.relu(t233)
        del t233

        # pd_op.depthwise_conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x1x3x3xf32)
        t240 = paddle._C_ops.depthwise_conv2d(
            t239, t65, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t241, t242, t243, t244, t245, t246 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t240,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t247 = paddle._C_ops.relu(t241)
        del t241

        # pd_op.conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x128x1x1xf32)
        t248 = paddle._C_ops.conv2d(
            t247, t70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t249, t250, t251, t252, t253, t254 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t248,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t255 = paddle._C_ops.relu(t249)
        del t249

        # pd_op.depthwise_conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x1x3x3xf32)
        t256 = paddle._C_ops.depthwise_conv2d(
            t255, t75, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t257, t258, t259, t260, t261, t262 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t256,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t263 = paddle._C_ops.relu(t257)
        del t257

        # pd_op.conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x128x1x1xf32)
        t264 = paddle._C_ops.conv2d(
            t263, t80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t265, t266, t267, t268, t269, t270 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t264,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t271 = paddle._C_ops.relu(t265)
        del t265

        # pd_op.depthwise_conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x1x3x3xf32)
        t272 = paddle._C_ops.depthwise_conv2d(
            t271, t85, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t273, t274, t275, t276, t277, t278 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t272,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t279 = paddle._C_ops.relu(t273)
        del t273

        # pd_op.conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x128x1x1xf32)
        t280 = paddle._C_ops.conv2d(
            t279, t90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t281, t282, t283, t284, t285, t286 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t280,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t287 = paddle._C_ops.relu(t281)
        del t281

        # pd_op.depthwise_conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x1x3x3xf32)
        t288 = paddle._C_ops.depthwise_conv2d(
            t287, t95, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t289, t290, t291, t292, t293, t294 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t288,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t295 = paddle._C_ops.relu(t289)
        del t289

        # pd_op.conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x128x1x1xf32)
        t296 = paddle._C_ops.conv2d(
            t295, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t297, t298, t299, t300, t301, t302 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t296,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t303 = paddle._C_ops.relu(t297)
        del t297

        # pd_op.depthwise_conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x1x3x3xf32)
        t304 = paddle._C_ops.depthwise_conv2d(
            t303, t105, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t305, t306, t307, t308, t309, t310 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t304,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t311 = paddle._C_ops.relu(t305)
        del t305

        # pd_op.conv2d: (64x128x14x14xf32) <- (64x128x14x14xf32, 128x128x1x1xf32)
        t312 = paddle._C_ops.conv2d(
            t311, t110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t313, t314, t315, t316, t317, t318 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t312,
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

        # pd_op.relu: (64x128x14x14xf32) <- (64x128x14x14xf32)
        t319 = paddle._C_ops.relu(t313)
        del t313

        # pd_op.depthwise_conv2d: (64x128x7x7xf32) <- (64x128x14x14xf32, 128x1x3x3xf32)
        t320 = paddle._C_ops.depthwise_conv2d(
            t319, t115, [2, 2], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (64x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t321, t322, t323, t324, t325, t326 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t320,
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

        # pd_op.relu: (64x128x7x7xf32) <- (64x128x7x7xf32)
        t327 = paddle._C_ops.relu(t321)
        del t321

        # pd_op.conv2d: (64x256x7x7xf32) <- (64x128x7x7xf32, 256x128x1x1xf32)
        t328 = paddle._C_ops.conv2d(
            t327, t120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (64x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (64x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t329, t330, t331, t332, t333, t334 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t328,
                t121,
                t122,
                t123,
                t124,
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
        del t124, t123, t122, t121

        # pd_op.relu: (64x256x7x7xf32) <- (64x256x7x7xf32)
        t335 = paddle._C_ops.relu(t329)
        del t329

        # pd_op.depthwise_conv2d: (64x256x7x7xf32) <- (64x256x7x7xf32, 256x1x3x3xf32)
        t336 = paddle._C_ops.depthwise_conv2d(
            t335, t125, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (64x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (64x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t337, t338, t339, t340, t341, t342 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t336,
                t126,
                t127,
                t128,
                t129,
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
        del t126, t129, t128, t127

        # pd_op.relu: (64x256x7x7xf32) <- (64x256x7x7xf32)
        t343 = paddle._C_ops.relu(t337)
        del t337

        # pd_op.conv2d: (64x256x7x7xf32) <- (64x256x7x7xf32, 256x256x1x1xf32)
        t344 = paddle._C_ops.conv2d(
            t343, t130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (64x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (64x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t345, t346, t347, t348, t349, t350 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t344,
                t131,
                t132,
                t133,
                t134,
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
        del t134, t133, t132, t131

        # pd_op.relu: (64x256x7x7xf32) <- (64x256x7x7xf32)
        t351 = paddle._C_ops.relu(t345)
        del t345

        # pd_op.full_int_array: (2xi64) <- ()
        t352 = [1, 1]

        # pd_op.pool2d: (64x256x1x1xf32) <- (64x256x7x7xf32, 2xi64)
        t353 = paddle._C_ops.pool2d(
            t351,
            t352,
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

        # pd_op.flatten: (64x256xf32) <- (64x256x1x1xf32)
        t354 = paddle._C_ops.flatten(t353, 1, 3)

        # pd_op.matmul: (64x102xf32) <- (64x256xf32, 256x102xf32)
        t355 = paddle._C_ops.matmul(t354, t135, False, False)
        del t135

        return (
            t136,
            t138,
            t139,
            t140,
            t141,
            t142,
            t143,
            t144,
            t146,
            t147,
            t148,
            t149,
            t150,
            t151,
            t152,
            t154,
            t155,
            t156,
            t157,
            t158,
            t159,
            t160,
            t162,
            t163,
            t164,
            t165,
            t166,
            t167,
            t168,
            t170,
            t171,
            t172,
            t173,
            t174,
            t175,
            t176,
            t178,
            t179,
            t180,
            t181,
            t182,
            t183,
            t184,
            t186,
            t187,
            t188,
            t189,
            t190,
            t191,
            t192,
            t194,
            t195,
            t196,
            t197,
            t198,
            t199,
            t200,
            t202,
            t203,
            t204,
            t205,
            t206,
            t207,
            t208,
            t210,
            t211,
            t212,
            t213,
            t214,
            t215,
            t216,
            t218,
            t219,
            t220,
            t221,
            t222,
            t223,
            t224,
            t226,
            t227,
            t228,
            t229,
            t230,
            t231,
            t232,
            t234,
            t235,
            t236,
            t237,
            t238,
            t239,
            t240,
            t242,
            t243,
            t244,
            t245,
            t246,
            t247,
            t248,
            t250,
            t251,
            t252,
            t253,
            t254,
            t255,
            t256,
            t258,
            t259,
            t260,
            t261,
            t262,
            t263,
            t264,
            t266,
            t267,
            t268,
            t269,
            t270,
            t271,
            t272,
            t274,
            t275,
            t276,
            t277,
            t278,
            t279,
            t280,
            t282,
            t283,
            t284,
            t285,
            t286,
            t287,
            t288,
            t290,
            t291,
            t292,
            t293,
            t294,
            t295,
            t296,
            t298,
            t299,
            t300,
            t301,
            t302,
            t303,
            t304,
            t306,
            t307,
            t308,
            t309,
            t310,
            t311,
            t312,
            t314,
            t315,
            t316,
            t317,
            t318,
            t319,
            t320,
            t322,
            t323,
            t324,
            t325,
            t326,
            t327,
            t328,
            t330,
            t331,
            t332,
            t333,
            t334,
            t335,
            t336,
            t338,
            t339,
            t340,
            t341,
            t342,
            t343,
            t344,
            t346,
            t347,
            t348,
            t349,
            t350,
            t351,
            t352,
            t353,
            t354,
            t355,
        )
