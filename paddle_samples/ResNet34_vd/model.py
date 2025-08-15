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
        t186: paddle.Tensor,
        t187: paddle.Tensor,
        t188: paddle.Tensor,
        t189: paddle.Tensor,
        t190: paddle.Tensor,
        t191: paddle.Tensor,
        t192: paddle.Tensor,
        t193: paddle.Tensor,
        t194: paddle.Tensor,
        t195: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        t196 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t197, t198, t199, t200, t201, t202 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t196,
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
        del t196, t4, t3, t2, t1

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t203 = paddle._C_ops.relu(t197)
        del t197

        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 32x32x3x3xf32)
        t204 = paddle._C_ops.conv2d(
            t203, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5, t203

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t205, t206, t207, t208, t209, t210 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t204,
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
        del t204, t9, t8, t7, t6

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t211 = paddle._C_ops.relu(t205)
        del t205

        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x32x112x112xf32, 64x32x3x3xf32)
        t212 = paddle._C_ops.conv2d(
            t211, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10, t211

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t213, t214, t215, t216, t217, t218 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t212,
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
        del t212, t14, t13, t12, t11

        # pd_op.relu: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        t219 = paddle._C_ops.relu(t213)
        del t213

        # pd_op.full_int_array: (2xi64) <- ()
        t220 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        t221 = paddle._C_ops.pool2d(
            t219,
            t220,
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
        del t220, t219

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t222 = paddle._C_ops.conv2d(
            t221, t15, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t223, t224, t225, t226, t227, t228 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t222,
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
        del t222, t19, t18, t17, t16

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t229 = paddle._C_ops.relu(t223)
        del t223

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t230 = paddle._C_ops.conv2d(
            t229, t20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20, t229

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t231, t232, t233, t234, t235, t236 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t230,
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
        del t230, t24, t23, t22, t21

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        t237 = paddle._C_ops.conv2d(
            t221, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25, t221

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t238, t239, t240, t241, t242, t243 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t237,
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
        del t237, t29, t28, t27, t26

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t244 = paddle._C_ops.add(t231, t238)
        del t231, t238

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t245 = paddle._C_ops.relu(t244)
        del t244

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t246 = paddle._C_ops.conv2d(
            t245, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t247, t248, t249, t250, t251, t252 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t246,
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
        del t246, t34, t33, t32, t31

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t253 = paddle._C_ops.relu(t247)
        del t247

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t254 = paddle._C_ops.conv2d(
            t253, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35, t253

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t255, t256, t257, t258, t259, t260 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t254,
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
        del t254, t39, t38, t37, t36

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t261 = paddle._C_ops.add(t255, t245)
        del t255, t245

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t262 = paddle._C_ops.relu(t261)
        del t261

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t263 = paddle._C_ops.conv2d(
            t262, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t264, t265, t266, t267, t268, t269 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t263,
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
        del t263, t44, t43, t42, t41

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t270 = paddle._C_ops.relu(t264)
        del t264

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t271 = paddle._C_ops.conv2d(
            t270, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45, t270

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t272, t273, t274, t275, t276, t277 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t271,
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
        del t271, t49, t48, t47, t46

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t278 = paddle._C_ops.add(t272, t262)
        del t272, t262

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t279 = paddle._C_ops.relu(t278)
        del t278

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x56x56xf32, 128x64x3x3xf32)
        t280 = paddle._C_ops.conv2d(
            t279, t50, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t281, t282, t283, t284, t285, t286 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t280,
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
        del t280, t54, t53, t52, t51

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t287 = paddle._C_ops.relu(t281)
        del t281

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t288 = paddle._C_ops.conv2d(
            t287, t55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55, t287

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t289, t290, t291, t292, t293, t294 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t288,
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
        del t288, t59, t58, t57, t56

        # pd_op.full_int_array: (2xi64) <- ()
        t295 = [2, 2]

        # pd_op.pool2d: (-1x64x28x28xf32) <- (-1x64x56x56xf32, 2xi64)
        t296 = paddle._C_ops.pool2d(
            t279, t295, [2, 2], [0, 0], True, True, "NCHW", "avg", False, False, "SAME"
        )
        del t279

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x28x28xf32, 128x64x1x1xf32)
        t297 = paddle._C_ops.conv2d(
            t296, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60, t296

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t298, t299, t300, t301, t302, t303 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t297,
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
        del t297, t64, t63, t62, t61

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t304 = paddle._C_ops.add(t289, t298)
        del t289, t298

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t305 = paddle._C_ops.relu(t304)
        del t304

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t306 = paddle._C_ops.conv2d(
            t305, t65, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t307, t308, t309, t310, t311, t312 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t306,
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
        del t306, t69, t68, t67, t66

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t313 = paddle._C_ops.relu(t307)
        del t307

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t314 = paddle._C_ops.conv2d(
            t313, t70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70, t313

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t315, t316, t317, t318, t319, t320 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t314,
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
        del t314, t74, t73, t72, t71

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t321 = paddle._C_ops.add(t315, t305)
        del t315, t305

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t322 = paddle._C_ops.relu(t321)
        del t321

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t323 = paddle._C_ops.conv2d(
            t322, t75, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t324, t325, t326, t327, t328, t329 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t323,
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
        del t323, t79, t78, t77, t76

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t330 = paddle._C_ops.relu(t324)
        del t324

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t331 = paddle._C_ops.conv2d(
            t330, t80, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80, t330

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t332, t333, t334, t335, t336, t337 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t331,
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
        del t331, t84, t83, t82, t81

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t338 = paddle._C_ops.add(t332, t322)
        del t332, t322

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t339 = paddle._C_ops.relu(t338)
        del t338

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t340 = paddle._C_ops.conv2d(
            t339, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t341, t342, t343, t344, t345, t346 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t340,
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
        del t340, t89, t88, t87, t86

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t347 = paddle._C_ops.relu(t341)
        del t341

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t348 = paddle._C_ops.conv2d(
            t347, t90, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90, t347

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t349, t350, t351, t352, t353, t354 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t348,
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
        del t348, t94, t93, t92, t91

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t355 = paddle._C_ops.add(t349, t339)
        del t349, t339

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t356 = paddle._C_ops.relu(t355)
        del t355

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x28x28xf32, 256x128x3x3xf32)
        t357 = paddle._C_ops.conv2d(
            t356, t95, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t358, t359, t360, t361, t362, t363 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t357,
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
        del t357, t96, t99, t98, t97

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t364 = paddle._C_ops.relu(t358)
        del t358

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t365 = paddle._C_ops.conv2d(
            t364, t100, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100, t364

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t366, t367, t368, t369, t370, t371 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t365,
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
        del t365, t104, t103, t102, t101

        # pd_op.pool2d: (-1x128x14x14xf32) <- (-1x128x28x28xf32, 2xi64)
        t372 = paddle._C_ops.pool2d(
            t356, t295, [2, 2], [0, 0], True, True, "NCHW", "avg", False, False, "SAME"
        )
        del t356

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x14x14xf32, 256x128x1x1xf32)
        t373 = paddle._C_ops.conv2d(
            t372, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105, t372

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t374, t375, t376, t377, t378, t379 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t373,
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
        del t373, t109, t108, t107, t106

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t380 = paddle._C_ops.add(t366, t374)
        del t366, t374

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t381 = paddle._C_ops.relu(t380)
        del t380

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t382 = paddle._C_ops.conv2d(
            t381, t110, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t383, t384, t385, t386, t387, t388 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t382,
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
        del t382, t114, t113, t112, t111

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t389 = paddle._C_ops.relu(t383)
        del t383

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t390 = paddle._C_ops.conv2d(
            t389, t115, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115, t389

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t391, t392, t393, t394, t395, t396 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t390,
                t116,
                t117,
                t118,
                t119,
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
        del t390, t119, t118, t117, t116

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t397 = paddle._C_ops.add(t391, t381)
        del t391, t381

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t398 = paddle._C_ops.relu(t397)
        del t397

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t399 = paddle._C_ops.conv2d(
            t398, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t400, t401, t402, t403, t404, t405 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t399,
                t121,
                t122,
                t123,
                t124,
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
        del t399, t124, t123, t122, t121

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t406 = paddle._C_ops.relu(t400)
        del t400

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t407 = paddle._C_ops.conv2d(
            t406, t125, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125, t406

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t408, t409, t410, t411, t412, t413 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t407,
                t126,
                t127,
                t128,
                t129,
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
        del t407, t129, t128, t127, t126

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t414 = paddle._C_ops.add(t408, t398)
        del t408, t398

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t415 = paddle._C_ops.relu(t414)
        del t414

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t416 = paddle._C_ops.conv2d(
            t415, t130, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t417, t418, t419, t420, t421, t422 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t416,
                t131,
                t132,
                t133,
                t134,
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
        del t416, t134, t133, t132, t131

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t423 = paddle._C_ops.relu(t417)
        del t417

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t424 = paddle._C_ops.conv2d(
            t423, t135, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135, t423

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t425, t426, t427, t428, t429, t430 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t424,
                t136,
                t137,
                t138,
                t139,
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
        del t424, t139, t138, t137, t136

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t431 = paddle._C_ops.add(t425, t415)
        del t425, t415

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t432 = paddle._C_ops.relu(t431)
        del t431

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t433 = paddle._C_ops.conv2d(
            t432, t140, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t434, t435, t436, t437, t438, t439 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t433,
                t141,
                t142,
                t143,
                t144,
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
        del t433, t144, t143, t142, t141

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t440 = paddle._C_ops.relu(t434)
        del t434

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t441 = paddle._C_ops.conv2d(
            t440, t145, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145, t440

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t442, t443, t444, t445, t446, t447 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t441,
                t146,
                t147,
                t148,
                t149,
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
        del t441, t149, t148, t147, t146

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t448 = paddle._C_ops.add(t442, t432)
        del t442, t432

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t449 = paddle._C_ops.relu(t448)
        del t448

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t450 = paddle._C_ops.conv2d(
            t449, t150, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t451, t452, t453, t454, t455, t456 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t450,
                t151,
                t152,
                t153,
                t154,
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
        del t450, t154, t153, t152, t151

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t457 = paddle._C_ops.relu(t451)
        del t451

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t458 = paddle._C_ops.conv2d(
            t457, t155, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155, t457

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t459, t460, t461, t462, t463, t464 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t458,
                t156,
                t157,
                t158,
                t159,
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
        del t458, t159, t158, t157, t156

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t465 = paddle._C_ops.add(t459, t449)
        del t459, t449

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t466 = paddle._C_ops.relu(t465)
        del t465

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x14x14xf32, 512x256x3x3xf32)
        t467 = paddle._C_ops.conv2d(
            t466, t160, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t468, t469, t470, t471, t472, t473 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t467,
                t161,
                t162,
                t163,
                t164,
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
        del t467, t164, t163, t162, t161

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t474 = paddle._C_ops.relu(t468)
        del t468

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t475 = paddle._C_ops.conv2d(
            t474, t165, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165, t474

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t476, t477, t478, t479, t480, t481 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t475,
                t166,
                t167,
                t168,
                t169,
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
        del t475, t169, t168, t167, t166

        # pd_op.pool2d: (-1x256x7x7xf32) <- (-1x256x14x14xf32, 2xi64)
        t482 = paddle._C_ops.pool2d(
            t466, t295, [2, 2], [0, 0], True, True, "NCHW", "avg", False, False, "SAME"
        )
        del t295, t466

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x7x7xf32, 512x256x1x1xf32)
        t483 = paddle._C_ops.conv2d(
            t482, t170, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170, t482

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t484, t485, t486, t487, t488, t489 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t483,
                t171,
                t172,
                t173,
                t174,
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
        del t483, t174, t173, t172, t171

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t490 = paddle._C_ops.add(t476, t484)
        del t476, t484

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t491 = paddle._C_ops.relu(t490)
        del t490

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t492 = paddle._C_ops.conv2d(
            t491, t175, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t493, t494, t495, t496, t497, t498 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t492,
                t176,
                t177,
                t178,
                t179,
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
        del t492, t179, t178, t177, t176

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t499 = paddle._C_ops.relu(t493)
        del t493

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t500 = paddle._C_ops.conv2d(
            t499, t180, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180, t499

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t501, t502, t503, t504, t505, t506 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t500,
                t181,
                t182,
                t183,
                t184,
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
        del t500, t184, t183, t182, t181

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t507 = paddle._C_ops.add(t501, t491)
        del t501, t491

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t508 = paddle._C_ops.relu(t507)
        del t507

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t509 = paddle._C_ops.conv2d(
            t508, t185, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t510, t511, t512, t513, t514, t515 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t509,
                t186,
                t187,
                t188,
                t189,
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
        del t509, t186, t189, t188, t187

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t516 = paddle._C_ops.relu(t510)
        del t510

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t517 = paddle._C_ops.conv2d(
            t516, t190, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190, t516

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t518, t519, t520, t521, t522, t523 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t517,
                t191,
                t192,
                t193,
                t194,
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
        del t517, t194, t193, t192, t191

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t524 = paddle._C_ops.add(t518, t508)
        del t518, t508

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t525 = paddle._C_ops.relu(t524)
        del t524

        # pd_op.full_int_array: (2xi64) <- ()
        t526 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x7x7xf32, 2xi64)
        t527 = paddle._C_ops.pool2d(
            t525,
            t526,
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
        del t526, t525

        # pd_op.flatten: (-1x512xf32) <- (-1x512x1x1xf32)
        t528 = paddle._C_ops.flatten(t527, 1, 3)
        del t527

        # pd_op.matmul: (-1x102xf32) <- (-1x512xf32, 512x102xf32)
        t529 = paddle._C_ops.matmul(t528, t195, False, False)
        del t528, t195

        return t529
