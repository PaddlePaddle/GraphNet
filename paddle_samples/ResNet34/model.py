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
        t145: paddle.Tensor,
        t146: paddle.Tensor,
        t147: paddle.Tensor,
        t148: paddle.Tensor,
        t149: paddle.Tensor,
        t150: paddle.Tensor,
        t151: paddle.Tensor,
        t152: paddle.Tensor,
        t153: paddle.Tensor,
        t154: paddle.Tensor,
        t155: paddle.Tensor,
        t156: paddle.Tensor,
        t157: paddle.Tensor,
        t158: paddle.Tensor,
        t159: paddle.Tensor,
        t160: paddle.Tensor,
        t161: paddle.Tensor,
        t162: paddle.Tensor,
        t163: paddle.Tensor,
        t164: paddle.Tensor,
        t165: paddle.Tensor,
        t166: paddle.Tensor,
        t167: paddle.Tensor,
        t168: paddle.Tensor,
        t169: paddle.Tensor,
        t170: paddle.Tensor,
        t171: paddle.Tensor,
        t172: paddle.Tensor,
        t173: paddle.Tensor,
        t174: paddle.Tensor,
        t175: paddle.Tensor,
        t176: paddle.Tensor,
        t177: paddle.Tensor,
        t178: paddle.Tensor,
        t179: paddle.Tensor,
        t180: paddle.Tensor,
        t181: paddle.Tensor,
        t182: paddle.Tensor,
        t183: paddle.Tensor,
        t184: paddle.Tensor,
        t185: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x3x224x224xf32, 64x3x7x7xf32)
        t186 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t187, t188, t189, t190, t191, t192 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t186,
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

        # pd_op.relu: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        t193 = paddle._C_ops.relu(t187)
        del t187

        # pd_op.full_int_array: (2xi64) <- ()
        t194 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        t195 = paddle._C_ops.pool2d(
            t193,
            t194,
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

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t196 = paddle._C_ops.conv2d(
            t195, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t197, t198, t199, t200, t201, t202 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t196,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t203 = paddle._C_ops.relu(t197)
        del t197

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t204 = paddle._C_ops.conv2d(
            t203, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t205, t206, t207, t208, t209, t210 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t204,
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

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        t211 = paddle._C_ops.conv2d(
            t195, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t212, t213, t214, t215, t216, t217 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t211,
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

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t218 = paddle._C_ops.add(t205, t212)

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t219 = paddle._C_ops.relu(t218)
        del t218

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t220 = paddle._C_ops.conv2d(
            t219, t20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t221, t222, t223, t224, t225, t226 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t220,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t227 = paddle._C_ops.relu(t221)
        del t221

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t228 = paddle._C_ops.conv2d(
            t227, t25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t229, t230, t231, t232, t233, t234 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t228,
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

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t235 = paddle._C_ops.add(t229, t219)

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t236 = paddle._C_ops.relu(t235)
        del t235

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t237 = paddle._C_ops.conv2d(
            t236, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t238, t239, t240, t241, t242, t243 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t237,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t244 = paddle._C_ops.relu(t238)
        del t238

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t245 = paddle._C_ops.conv2d(
            t244, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t246, t247, t248, t249, t250, t251 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t245,
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

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t252 = paddle._C_ops.add(t246, t236)

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t253 = paddle._C_ops.relu(t252)
        del t252

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x56x56xf32, 128x64x3x3xf32)
        t254 = paddle._C_ops.conv2d(
            t253, t40, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t255, t256, t257, t258, t259, t260 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t254,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t261 = paddle._C_ops.relu(t255)
        del t255

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t262 = paddle._C_ops.conv2d(
            t261, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t263, t264, t265, t266, t267, t268 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t262,
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

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x56x56xf32, 128x64x1x1xf32)
        t269 = paddle._C_ops.conv2d(
            t253, t50, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t270, t271, t272, t273, t274, t275 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t269,
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

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t276 = paddle._C_ops.add(t263, t270)

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t277 = paddle._C_ops.relu(t276)
        del t276

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t278 = paddle._C_ops.conv2d(
            t277, t55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t279, t280, t281, t282, t283, t284 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t278,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t285 = paddle._C_ops.relu(t279)
        del t279

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t286 = paddle._C_ops.conv2d(
            t285, t60, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t287, t288, t289, t290, t291, t292 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t286,
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

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t293 = paddle._C_ops.add(t287, t277)

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t294 = paddle._C_ops.relu(t293)
        del t293

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t295 = paddle._C_ops.conv2d(
            t294, t65, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t296, t297, t298, t299, t300, t301 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t295,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t302 = paddle._C_ops.relu(t296)
        del t296

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t303 = paddle._C_ops.conv2d(
            t302, t70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t304, t305, t306, t307, t308, t309 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t303,
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

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t310 = paddle._C_ops.add(t304, t294)

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t311 = paddle._C_ops.relu(t310)
        del t310

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t312 = paddle._C_ops.conv2d(
            t311, t75, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t313, t314, t315, t316, t317, t318 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t312,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t319 = paddle._C_ops.relu(t313)
        del t313

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t320 = paddle._C_ops.conv2d(
            t319, t80, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t321, t322, t323, t324, t325, t326 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t320,
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

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t327 = paddle._C_ops.add(t321, t311)

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t328 = paddle._C_ops.relu(t327)
        del t327

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x28x28xf32, 256x128x3x3xf32)
        t329 = paddle._C_ops.conv2d(
            t328, t85, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t330, t331, t332, t333, t334, t335 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t329,
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
        del t86, t89, t88, t87

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t336 = paddle._C_ops.relu(t330)
        del t330

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t337 = paddle._C_ops.conv2d(
            t336, t90, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t338, t339, t340, t341, t342, t343 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t337,
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

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x28x28xf32, 256x128x1x1xf32)
        t344 = paddle._C_ops.conv2d(
            t328, t95, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t345, t346, t347, t348, t349, t350 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t344,
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

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t351 = paddle._C_ops.add(t338, t345)

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t352 = paddle._C_ops.relu(t351)
        del t351

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t353 = paddle._C_ops.conv2d(
            t352, t100, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t354, t355, t356, t357, t358, t359 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t353,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t360 = paddle._C_ops.relu(t354)
        del t354

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t361 = paddle._C_ops.conv2d(
            t360, t105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t362, t363, t364, t365, t366, t367 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t361,
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

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t368 = paddle._C_ops.add(t362, t352)

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t369 = paddle._C_ops.relu(t368)
        del t368

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t370 = paddle._C_ops.conv2d(
            t369, t110, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t371, t372, t373, t374, t375, t376 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t370,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t377 = paddle._C_ops.relu(t371)
        del t371

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t378 = paddle._C_ops.conv2d(
            t377, t115, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t379, t380, t381, t382, t383, t384 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t378,
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

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t385 = paddle._C_ops.add(t379, t369)

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t386 = paddle._C_ops.relu(t385)
        del t385

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t387 = paddle._C_ops.conv2d(
            t386, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t388, t389, t390, t391, t392, t393 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t387,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t394 = paddle._C_ops.relu(t388)
        del t388

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t395 = paddle._C_ops.conv2d(
            t394, t125, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t396, t397, t398, t399, t400, t401 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t395,
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
        del t129, t128, t127, t126

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t402 = paddle._C_ops.add(t396, t386)

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t403 = paddle._C_ops.relu(t402)
        del t402

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t404 = paddle._C_ops.conv2d(
            t403, t130, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t405, t406, t407, t408, t409, t410 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t404,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t411 = paddle._C_ops.relu(t405)
        del t405

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t412 = paddle._C_ops.conv2d(
            t411, t135, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t413, t414, t415, t416, t417, t418 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t412,
                t136,
                t137,
                t138,
                t139,
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
        del t139, t138, t137, t136

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t419 = paddle._C_ops.add(t413, t403)

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t420 = paddle._C_ops.relu(t419)
        del t419

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t421 = paddle._C_ops.conv2d(
            t420, t140, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t422, t423, t424, t425, t426, t427 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t421,
                t141,
                t142,
                t143,
                t144,
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
        del t144, t143, t142, t141

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t428 = paddle._C_ops.relu(t422)
        del t422

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t429 = paddle._C_ops.conv2d(
            t428, t145, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t430, t431, t432, t433, t434, t435 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t429,
                t146,
                t147,
                t148,
                t149,
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
        del t149, t148, t147, t146

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t436 = paddle._C_ops.add(t430, t420)

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t437 = paddle._C_ops.relu(t436)
        del t436

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x14x14xf32, 512x256x3x3xf32)
        t438 = paddle._C_ops.conv2d(
            t437, t150, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t439, t440, t441, t442, t443, t444 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t438,
                t151,
                t152,
                t153,
                t154,
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
        del t154, t153, t152, t151

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t445 = paddle._C_ops.relu(t439)
        del t439

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t446 = paddle._C_ops.conv2d(
            t445, t155, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t447, t448, t449, t450, t451, t452 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t446,
                t156,
                t157,
                t158,
                t159,
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
        del t159, t158, t157, t156

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        t453 = paddle._C_ops.conv2d(
            t437, t160, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t454, t455, t456, t457, t458, t459 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t453,
                t161,
                t162,
                t163,
                t164,
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
        del t164, t163, t162, t161

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t460 = paddle._C_ops.add(t447, t454)

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t461 = paddle._C_ops.relu(t460)
        del t460

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t462 = paddle._C_ops.conv2d(
            t461, t165, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t463, t464, t465, t466, t467, t468 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t462,
                t166,
                t167,
                t168,
                t169,
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
        del t169, t168, t167, t166

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t469 = paddle._C_ops.relu(t463)
        del t463

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t470 = paddle._C_ops.conv2d(
            t469, t170, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t471, t472, t473, t474, t475, t476 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t470,
                t171,
                t172,
                t173,
                t174,
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
        del t174, t173, t172, t171

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t477 = paddle._C_ops.add(t471, t461)

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t478 = paddle._C_ops.relu(t477)
        del t477

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t479 = paddle._C_ops.conv2d(
            t478, t175, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t480, t481, t482, t483, t484, t485 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t479,
                t176,
                t177,
                t178,
                t179,
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
        del t176, t179, t178, t177

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t486 = paddle._C_ops.relu(t480)
        del t480

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t487 = paddle._C_ops.conv2d(
            t486, t180, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t488, t489, t490, t491, t492, t493 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t487,
                t181,
                t182,
                t183,
                t184,
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
        del t184, t183, t182, t181

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t494 = paddle._C_ops.add(t488, t478)

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t495 = paddle._C_ops.relu(t494)
        del t494

        # pd_op.full_int_array: (2xi64) <- ()
        t496 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x7x7xf32, 2xi64)
        t497 = paddle._C_ops.pool2d(
            t495,
            t496,
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

        # pd_op.flatten: (-1x512xf32) <- (-1x512x1x1xf32)
        t498 = paddle._C_ops.flatten(t497, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x512xf32, 512x102xf32)
        t499 = paddle._C_ops.matmul(t498, t185, False, False)
        del t185

        return (
            t186,
            t188,
            t189,
            t190,
            t191,
            t192,
            t193,
            t194,
            t195,
            t196,
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
            t219,
            t220,
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
            t236,
            t237,
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
            t253,
            t254,
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
            t277,
            t278,
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
            t294,
            t295,
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
            t311,
            t312,
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
            t328,
            t329,
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
            t343,
            t344,
            t345,
            t346,
            t347,
            t348,
            t349,
            t350,
            t352,
            t353,
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
            t369,
            t370,
            t372,
            t373,
            t374,
            t375,
            t376,
            t377,
            t378,
            t379,
            t380,
            t381,
            t382,
            t383,
            t384,
            t386,
            t387,
            t389,
            t390,
            t391,
            t392,
            t393,
            t394,
            t395,
            t396,
            t397,
            t398,
            t399,
            t400,
            t401,
            t403,
            t404,
            t406,
            t407,
            t408,
            t409,
            t410,
            t411,
            t412,
            t413,
            t414,
            t415,
            t416,
            t417,
            t418,
            t420,
            t421,
            t423,
            t424,
            t425,
            t426,
            t427,
            t428,
            t429,
            t430,
            t431,
            t432,
            t433,
            t434,
            t435,
            t437,
            t438,
            t440,
            t441,
            t442,
            t443,
            t444,
            t445,
            t446,
            t447,
            t448,
            t449,
            t450,
            t451,
            t452,
            t453,
            t454,
            t455,
            t456,
            t457,
            t458,
            t459,
            t461,
            t462,
            t464,
            t465,
            t466,
            t467,
            t468,
            t469,
            t470,
            t471,
            t472,
            t473,
            t474,
            t475,
            t476,
            t478,
            t479,
            t481,
            t482,
            t483,
            t484,
            t485,
            t486,
            t487,
            t488,
            t489,
            t490,
            t491,
            t492,
            t493,
            t495,
            t496,
            t497,
            t498,
            t499,
        )
