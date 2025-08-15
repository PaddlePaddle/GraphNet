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
        t196: paddle.Tensor,
        t197: paddle.Tensor,
        t198: paddle.Tensor,
        t199: paddle.Tensor,
        t200: paddle.Tensor,
        t201: paddle.Tensor,
        t202: paddle.Tensor,
        t203: paddle.Tensor,
        t204: paddle.Tensor,
        t205: paddle.Tensor,
        t206: paddle.Tensor,
        t207: paddle.Tensor,
        t208: paddle.Tensor,
        t209: paddle.Tensor,
        t210: paddle.Tensor,
        t211: paddle.Tensor,
        t212: paddle.Tensor,
    ):
        t213 = None
        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x3x224x224xf32, 24x3x3x3xf32)
        t214 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t215, t216, t217, t218, t219, t220 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t214,
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
        del t214, t4, t3, t2, t1

        # pd_op.hardswish: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        t221 = paddle._C_ops.hardswish(t215)
        del t215

        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x24x112x112xf32, 24x24x1x1xf32)
        t222 = paddle._C_ops.conv2d(
            t221, t5, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t221, t5

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t223, t224, t225, t226, t227, t228 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t222,
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
        del t222, t9, t8, t7, t6

        # pd_op.relu: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        t229 = paddle._C_ops.relu(t223)
        del t223

        # pd_op.depthwise_conv2d: (-1x24x56x56xf32) <- (-1x24x112x112xf32, 24x1x3x3xf32)
        t230 = paddle._C_ops.depthwise_conv2d(
            t229, t10, [2, 2], [1, 1], "EXPLICIT", 24, [1, 1], "NCHW"
        )
        del t10, t229

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t231, t232, t233, t234, t235, t236 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t230,
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
        del t230, t14, t13, t12, t11

        # pd_op.relu: (-1x24x56x56xf32) <- (-1x24x56x56xf32)
        t237 = paddle._C_ops.relu(t231)
        del t231

        # pd_op.full_int_array: (2xi64) <- ()
        t238 = [1, 1]

        # pd_op.pool2d: (-1x24x1x1xf32) <- (-1x24x56x56xf32, 2xi64)
        t239 = paddle._C_ops.pool2d(
            t237,
            t238,
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

        # pd_op.conv2d: (-1x6x1x1xf32) <- (-1x24x1x1xf32, 6x24x1x1xf32)
        t240 = paddle._C_ops.conv2d(
            t239, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15, t239

        # pd_op.full_int_array: (4xi64) <- ()
        t241 = [1, -1, 1, 1]

        # pd_op.reshape: (1x6x1x1xf32) <- (6xf32, 4xi64)
        t242 = paddle._C_ops.reshape(t16, t241)
        del t16

        # pd_op.add: (-1x6x1x1xf32) <- (-1x6x1x1xf32, 1x6x1x1xf32)
        t243 = paddle._C_ops.add(t240, t242)
        del t240, t242

        # pd_op.relu: (-1x6x1x1xf32) <- (-1x6x1x1xf32)
        t244 = paddle._C_ops.relu(t243)
        del t243

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x6x1x1xf32, 24x6x1x1xf32)
        t245 = paddle._C_ops.conv2d(
            t244, t17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t17, t244

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t246 = paddle._C_ops.reshape(t18, t241)
        del t18

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        t247 = paddle._C_ops.add(t245, t246)
        del t245, t246

        # pd_op.hardsigmoid: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        t248 = paddle._C_ops.hardsigmoid(t247, float("0.2"), float("0.5"))
        del t247

        # pd_op.multiply: (-1x24x56x56xf32) <- (-1x24x56x56xf32, -1x24x1x1xf32)
        t249 = paddle._C_ops.multiply(t237, t248)
        del t248, t237

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x24x1x1xf32)
        t250 = paddle._C_ops.conv2d(
            t249, t19, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t249, t19

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t251, t252, t253, t254, t255, t256 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t250,
                t20,
                t21,
                t22,
                t23,
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
        del t250, t23, t22, t21, t20

        # pd_op.conv2d: (-1x88x56x56xf32) <- (-1x24x56x56xf32, 88x24x1x1xf32)
        t257 = paddle._C_ops.conv2d(
            t251, t24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t251, t24

        # pd_op.batch_norm_: (-1x88x56x56xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x56x56xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t258, t259, t260, t261, t262, t263 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t257,
                t25,
                t26,
                t27,
                t28,
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
        del t257, t28, t27, t26, t25

        # pd_op.relu: (-1x88x56x56xf32) <- (-1x88x56x56xf32)
        t264 = paddle._C_ops.relu(t258)
        del t258

        # pd_op.depthwise_conv2d: (-1x88x28x28xf32) <- (-1x88x56x56xf32, 88x1x3x3xf32)
        t265 = paddle._C_ops.depthwise_conv2d(
            t264, t29, [2, 2], [1, 1], "EXPLICIT", 88, [1, 1], "NCHW"
        )
        del t29, t264

        # pd_op.batch_norm_: (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t266, t267, t268, t269, t270, t271 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t265,
                t30,
                t31,
                t32,
                t33,
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
        del t265, t33, t32, t31, t30

        # pd_op.relu: (-1x88x28x28xf32) <- (-1x88x28x28xf32)
        t272 = paddle._C_ops.relu(t266)
        del t266

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x88x28x28xf32, 32x88x1x1xf32)
        t273 = paddle._C_ops.conv2d(
            t272, t34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t34, t272

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t274, t275, t276, t277, t278, t279 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t273,
                t35,
                t36,
                t37,
                t38,
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
        del t273, t38, t37, t36, t35

        # pd_op.conv2d: (-1x112x28x28xf32) <- (-1x32x28x28xf32, 112x32x1x1xf32)
        t280 = paddle._C_ops.conv2d(
            t274, t39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t39

        # pd_op.batch_norm_: (-1x112x28x28xf32, 112xf32, 112xf32, 112xf32, 112xf32, -1xui8) <- (-1x112x28x28xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        t281, t282, t283, t284, t285, t286 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t280,
                t40,
                t41,
                t42,
                t43,
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
        del t280, t43, t42, t41, t40

        # pd_op.relu: (-1x112x28x28xf32) <- (-1x112x28x28xf32)
        t287 = paddle._C_ops.relu(t281)
        del t281

        # pd_op.depthwise_conv2d: (-1x112x28x28xf32) <- (-1x112x28x28xf32, 112x1x3x3xf32)
        t288 = paddle._C_ops.depthwise_conv2d(
            t287, t44, [1, 1], [1, 1], "EXPLICIT", 112, [1, 1], "NCHW"
        )
        del t44, t287

        # pd_op.batch_norm_: (-1x112x28x28xf32, 112xf32, 112xf32, 112xf32, 112xf32, -1xui8) <- (-1x112x28x28xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        t289, t290, t291, t292, t293, t294 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t288,
                t45,
                t46,
                t47,
                t48,
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
        del t288, t48, t47, t46, t45

        # pd_op.relu: (-1x112x28x28xf32) <- (-1x112x28x28xf32)
        t295 = paddle._C_ops.relu(t289)
        del t289

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x112x28x28xf32, 32x112x1x1xf32)
        t296 = paddle._C_ops.conv2d(
            t295, t49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t49, t295

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t297, t298, t299, t300, t301, t302 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t296,
                t50,
                t51,
                t52,
                t53,
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
        del t296, t53, t52, t51, t50

        # pd_op.add: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x28x28xf32)
        t303 = paddle._C_ops.add(t274, t297)
        del t274, t297

        # pd_op.conv2d: (-1x120x28x28xf32) <- (-1x32x28x28xf32, 120x32x1x1xf32)
        t304 = paddle._C_ops.conv2d(
            t303, t54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t303, t54

        # pd_op.batch_norm_: (-1x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t305, t306, t307, t308, t309, t310 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t304,
                t55,
                t56,
                t57,
                t58,
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
        del t304, t58, t57, t56, t55

        # pd_op.hardswish: (-1x120x28x28xf32) <- (-1x120x28x28xf32)
        t311 = paddle._C_ops.hardswish(t305)
        del t305

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x28x28xf32, 120x1x5x5xf32)
        t312 = paddle._C_ops.depthwise_conv2d(
            t311, t59, [2, 2], [2, 2], "EXPLICIT", 120, [1, 1], "NCHW"
        )
        del t311, t59

        # pd_op.batch_norm_: (-1x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t313, t314, t315, t316, t317, t318 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t312,
                t60,
                t61,
                t62,
                t63,
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
        del t312, t63, t62, t61, t60

        # pd_op.hardswish: (-1x120x14x14xf32) <- (-1x120x14x14xf32)
        t319 = paddle._C_ops.hardswish(t313)
        del t313

        # pd_op.pool2d: (-1x120x1x1xf32) <- (-1x120x14x14xf32, 2xi64)
        t320 = paddle._C_ops.pool2d(
            t319,
            t238,
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

        # pd_op.conv2d: (-1x30x1x1xf32) <- (-1x120x1x1xf32, 30x120x1x1xf32)
        t321 = paddle._C_ops.conv2d(
            t320, t64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t64, t320

        # pd_op.reshape: (1x30x1x1xf32) <- (30xf32, 4xi64)
        t322 = paddle._C_ops.reshape(t65, t241)
        del t65

        # pd_op.add: (-1x30x1x1xf32) <- (-1x30x1x1xf32, 1x30x1x1xf32)
        t323 = paddle._C_ops.add(t321, t322)
        del t321, t322

        # pd_op.relu: (-1x30x1x1xf32) <- (-1x30x1x1xf32)
        t324 = paddle._C_ops.relu(t323)
        del t323

        # pd_op.conv2d: (-1x120x1x1xf32) <- (-1x30x1x1xf32, 120x30x1x1xf32)
        t325 = paddle._C_ops.conv2d(
            t324, t66, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t66, t324

        # pd_op.reshape: (1x120x1x1xf32) <- (120xf32, 4xi64)
        t326 = paddle._C_ops.reshape(t67, t241)
        del t67

        # pd_op.add: (-1x120x1x1xf32) <- (-1x120x1x1xf32, 1x120x1x1xf32)
        t327 = paddle._C_ops.add(t325, t326)
        del t325, t326

        # pd_op.hardsigmoid: (-1x120x1x1xf32) <- (-1x120x1x1xf32)
        t328 = paddle._C_ops.hardsigmoid(t327, float("0.2"), float("0.5"))
        del t327

        # pd_op.multiply: (-1x120x14x14xf32) <- (-1x120x14x14xf32, -1x120x1x1xf32)
        t329 = paddle._C_ops.multiply(t319, t328)
        del t328, t319

        # pd_op.conv2d: (-1x48x14x14xf32) <- (-1x120x14x14xf32, 48x120x1x1xf32)
        t330 = paddle._C_ops.conv2d(
            t329, t68, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t329, t68

        # pd_op.batch_norm_: (-1x48x14x14xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x14x14xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t331, t332, t333, t334, t335, t336 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t330,
                t69,
                t70,
                t71,
                t72,
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
        del t330, t72, t71, t70, t69

        # pd_op.conv2d: (-1x304x14x14xf32) <- (-1x48x14x14xf32, 304x48x1x1xf32)
        t337 = paddle._C_ops.conv2d(
            t331, t73, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t73

        # pd_op.batch_norm_: (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32, -1xui8) <- (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32)
        t338, t339, t340, t341, t342, t343 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t337,
                t74,
                t75,
                t76,
                t77,
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
        del t337, t77, t76, t75, t74

        # pd_op.hardswish: (-1x304x14x14xf32) <- (-1x304x14x14xf32)
        t344 = paddle._C_ops.hardswish(t338)
        del t338

        # pd_op.depthwise_conv2d: (-1x304x14x14xf32) <- (-1x304x14x14xf32, 304x1x5x5xf32)
        t345 = paddle._C_ops.depthwise_conv2d(
            t344, t78, [1, 1], [2, 2], "EXPLICIT", 304, [1, 1], "NCHW"
        )
        del t344, t78

        # pd_op.batch_norm_: (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32, -1xui8) <- (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32)
        t346, t347, t348, t349, t350, t351 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t345,
                t79,
                t80,
                t81,
                t82,
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
        del t345, t82, t81, t80, t79

        # pd_op.hardswish: (-1x304x14x14xf32) <- (-1x304x14x14xf32)
        t352 = paddle._C_ops.hardswish(t346)
        del t346

        # pd_op.pool2d: (-1x304x1x1xf32) <- (-1x304x14x14xf32, 2xi64)
        t353 = paddle._C_ops.pool2d(
            t352,
            t238,
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

        # pd_op.conv2d: (-1x76x1x1xf32) <- (-1x304x1x1xf32, 76x304x1x1xf32)
        t354 = paddle._C_ops.conv2d(
            t353, t83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t83, t353

        # pd_op.reshape: (1x76x1x1xf32) <- (76xf32, 4xi64)
        t355 = paddle._C_ops.reshape(t84, t241)
        del t84

        # pd_op.add: (-1x76x1x1xf32) <- (-1x76x1x1xf32, 1x76x1x1xf32)
        t356 = paddle._C_ops.add(t354, t355)
        del t354, t355

        # pd_op.relu: (-1x76x1x1xf32) <- (-1x76x1x1xf32)
        t357 = paddle._C_ops.relu(t356)
        del t356

        # pd_op.conv2d: (-1x304x1x1xf32) <- (-1x76x1x1xf32, 304x76x1x1xf32)
        t358 = paddle._C_ops.conv2d(
            t357, t85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85, t357

        # pd_op.reshape: (1x304x1x1xf32) <- (304xf32, 4xi64)
        t359 = paddle._C_ops.reshape(t86, t241)
        del t86

        # pd_op.add: (-1x304x1x1xf32) <- (-1x304x1x1xf32, 1x304x1x1xf32)
        t360 = paddle._C_ops.add(t358, t359)
        del t358, t359

        # pd_op.hardsigmoid: (-1x304x1x1xf32) <- (-1x304x1x1xf32)
        t361 = paddle._C_ops.hardsigmoid(t360, float("0.2"), float("0.5"))
        del t360

        # pd_op.multiply: (-1x304x14x14xf32) <- (-1x304x14x14xf32, -1x304x1x1xf32)
        t362 = paddle._C_ops.multiply(t352, t361)
        del t361, t352

        # pd_op.conv2d: (-1x48x14x14xf32) <- (-1x304x14x14xf32, 48x304x1x1xf32)
        t363 = paddle._C_ops.conv2d(
            t362, t87, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t362, t87

        # pd_op.batch_norm_: (-1x48x14x14xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x14x14xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t364, t365, t366, t367, t368, t369 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t363,
                t88,
                t89,
                t90,
                t91,
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
        del t363, t91, t90, t89, t88

        # pd_op.add: (-1x48x14x14xf32) <- (-1x48x14x14xf32, -1x48x14x14xf32)
        t370 = paddle._C_ops.add(t331, t364)
        del t331, t364

        # pd_op.conv2d: (-1x304x14x14xf32) <- (-1x48x14x14xf32, 304x48x1x1xf32)
        t371 = paddle._C_ops.conv2d(
            t370, t92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t92

        # pd_op.batch_norm_: (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32, -1xui8) <- (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32)
        t372, t373, t374, t375, t376, t377 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t371,
                t93,
                t94,
                t95,
                t96,
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
        del t371, t96, t95, t94, t93

        # pd_op.hardswish: (-1x304x14x14xf32) <- (-1x304x14x14xf32)
        t378 = paddle._C_ops.hardswish(t372)
        del t372

        # pd_op.depthwise_conv2d: (-1x304x14x14xf32) <- (-1x304x14x14xf32, 304x1x5x5xf32)
        t379 = paddle._C_ops.depthwise_conv2d(
            t378, t97, [1, 1], [2, 2], "EXPLICIT", 304, [1, 1], "NCHW"
        )
        del t378, t97

        # pd_op.batch_norm_: (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32, -1xui8) <- (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32)
        t380, t381, t382, t383, t384, t385 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t379,
                t98,
                t99,
                t100,
                t101,
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
        del t379, t101, t100, t99, t98

        # pd_op.hardswish: (-1x304x14x14xf32) <- (-1x304x14x14xf32)
        t386 = paddle._C_ops.hardswish(t380)
        del t380

        # pd_op.pool2d: (-1x304x1x1xf32) <- (-1x304x14x14xf32, 2xi64)
        t387 = paddle._C_ops.pool2d(
            t386,
            t238,
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

        # pd_op.conv2d: (-1x76x1x1xf32) <- (-1x304x1x1xf32, 76x304x1x1xf32)
        t388 = paddle._C_ops.conv2d(
            t387, t102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t102, t387

        # pd_op.reshape: (1x76x1x1xf32) <- (76xf32, 4xi64)
        t389 = paddle._C_ops.reshape(t103, t241)
        del t103

        # pd_op.add: (-1x76x1x1xf32) <- (-1x76x1x1xf32, 1x76x1x1xf32)
        t390 = paddle._C_ops.add(t388, t389)
        del t388, t389

        # pd_op.relu: (-1x76x1x1xf32) <- (-1x76x1x1xf32)
        t391 = paddle._C_ops.relu(t390)
        del t390

        # pd_op.conv2d: (-1x304x1x1xf32) <- (-1x76x1x1xf32, 304x76x1x1xf32)
        t392 = paddle._C_ops.conv2d(
            t391, t104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t104, t391

        # pd_op.reshape: (1x304x1x1xf32) <- (304xf32, 4xi64)
        t393 = paddle._C_ops.reshape(t105, t241)
        del t105

        # pd_op.add: (-1x304x1x1xf32) <- (-1x304x1x1xf32, 1x304x1x1xf32)
        t394 = paddle._C_ops.add(t392, t393)
        del t392, t393

        # pd_op.hardsigmoid: (-1x304x1x1xf32) <- (-1x304x1x1xf32)
        t395 = paddle._C_ops.hardsigmoid(t394, float("0.2"), float("0.5"))
        del t394

        # pd_op.multiply: (-1x304x14x14xf32) <- (-1x304x14x14xf32, -1x304x1x1xf32)
        t396 = paddle._C_ops.multiply(t386, t395)
        del t395, t386

        # pd_op.conv2d: (-1x48x14x14xf32) <- (-1x304x14x14xf32, 48x304x1x1xf32)
        t397 = paddle._C_ops.conv2d(
            t396, t106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t396, t106

        # pd_op.batch_norm_: (-1x48x14x14xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x14x14xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t398, t399, t400, t401, t402, t403 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t397,
                t107,
                t108,
                t109,
                t110,
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
        del t397, t110, t109, t108, t107

        # pd_op.add: (-1x48x14x14xf32) <- (-1x48x14x14xf32, -1x48x14x14xf32)
        t404 = paddle._C_ops.add(t370, t398)
        del t370, t398

        # pd_op.conv2d: (-1x152x14x14xf32) <- (-1x48x14x14xf32, 152x48x1x1xf32)
        t405 = paddle._C_ops.conv2d(
            t404, t111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t404, t111

        # pd_op.batch_norm_: (-1x152x14x14xf32, 152xf32, 152xf32, 152xf32, 152xf32, -1xui8) <- (-1x152x14x14xf32, 152xf32, 152xf32, 152xf32, 152xf32)
        t406, t407, t408, t409, t410, t411 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t405,
                t112,
                t113,
                t114,
                t115,
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
        del t405, t113, t112, t115, t114

        # pd_op.hardswish: (-1x152x14x14xf32) <- (-1x152x14x14xf32)
        t412 = paddle._C_ops.hardswish(t406)
        del t406

        # pd_op.depthwise_conv2d: (-1x152x14x14xf32) <- (-1x152x14x14xf32, 152x1x5x5xf32)
        t413 = paddle._C_ops.depthwise_conv2d(
            t412, t116, [1, 1], [2, 2], "EXPLICIT", 152, [1, 1], "NCHW"
        )
        del t412, t116

        # pd_op.batch_norm_: (-1x152x14x14xf32, 152xf32, 152xf32, 152xf32, 152xf32, -1xui8) <- (-1x152x14x14xf32, 152xf32, 152xf32, 152xf32, 152xf32)
        t414, t415, t416, t417, t418, t419 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t413,
                t117,
                t118,
                t119,
                t120,
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
        del t413, t120, t119, t118, t117

        # pd_op.hardswish: (-1x152x14x14xf32) <- (-1x152x14x14xf32)
        t420 = paddle._C_ops.hardswish(t414)
        del t414

        # pd_op.pool2d: (-1x152x1x1xf32) <- (-1x152x14x14xf32, 2xi64)
        t421 = paddle._C_ops.pool2d(
            t420,
            t238,
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

        # pd_op.conv2d: (-1x38x1x1xf32) <- (-1x152x1x1xf32, 38x152x1x1xf32)
        t422 = paddle._C_ops.conv2d(
            t421, t121, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t121, t421

        # pd_op.reshape: (1x38x1x1xf32) <- (38xf32, 4xi64)
        t423 = paddle._C_ops.reshape(t122, t241)
        del t122

        # pd_op.add: (-1x38x1x1xf32) <- (-1x38x1x1xf32, 1x38x1x1xf32)
        t424 = paddle._C_ops.add(t422, t423)
        del t422, t423

        # pd_op.relu: (-1x38x1x1xf32) <- (-1x38x1x1xf32)
        t425 = paddle._C_ops.relu(t424)
        del t424

        # pd_op.conv2d: (-1x152x1x1xf32) <- (-1x38x1x1xf32, 152x38x1x1xf32)
        t426 = paddle._C_ops.conv2d(
            t425, t123, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t123, t425

        # pd_op.reshape: (1x152x1x1xf32) <- (152xf32, 4xi64)
        t427 = paddle._C_ops.reshape(t124, t241)
        del t124

        # pd_op.add: (-1x152x1x1xf32) <- (-1x152x1x1xf32, 1x152x1x1xf32)
        t428 = paddle._C_ops.add(t426, t427)
        del t426, t427

        # pd_op.hardsigmoid: (-1x152x1x1xf32) <- (-1x152x1x1xf32)
        t429 = paddle._C_ops.hardsigmoid(t428, float("0.2"), float("0.5"))
        del t428

        # pd_op.multiply: (-1x152x14x14xf32) <- (-1x152x14x14xf32, -1x152x1x1xf32)
        t430 = paddle._C_ops.multiply(t420, t429)
        del t429, t420

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x152x14x14xf32, 64x152x1x1xf32)
        t431 = paddle._C_ops.conv2d(
            t430, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t430, t125

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t432, t433, t434, t435, t436, t437 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t431,
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
        del t431, t129, t128, t127, t126

        # pd_op.conv2d: (-1x184x14x14xf32) <- (-1x64x14x14xf32, 184x64x1x1xf32)
        t438 = paddle._C_ops.conv2d(
            t432, t130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (-1x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (-1x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        t439, t440, t441, t442, t443, t444 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t438,
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
        del t438, t134, t133, t132, t131

        # pd_op.hardswish: (-1x184x14x14xf32) <- (-1x184x14x14xf32)
        t445 = paddle._C_ops.hardswish(t439)
        del t439

        # pd_op.depthwise_conv2d: (-1x184x14x14xf32) <- (-1x184x14x14xf32, 184x1x5x5xf32)
        t446 = paddle._C_ops.depthwise_conv2d(
            t445, t135, [1, 1], [2, 2], "EXPLICIT", 184, [1, 1], "NCHW"
        )
        del t445, t135

        # pd_op.batch_norm_: (-1x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (-1x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        t447, t448, t449, t450, t451, t452 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t446,
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
        del t446, t139, t138, t137, t136

        # pd_op.hardswish: (-1x184x14x14xf32) <- (-1x184x14x14xf32)
        t453 = paddle._C_ops.hardswish(t447)
        del t447

        # pd_op.pool2d: (-1x184x1x1xf32) <- (-1x184x14x14xf32, 2xi64)
        t454 = paddle._C_ops.pool2d(
            t453,
            t238,
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

        # pd_op.conv2d: (-1x46x1x1xf32) <- (-1x184x1x1xf32, 46x184x1x1xf32)
        t455 = paddle._C_ops.conv2d(
            t454, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140, t454

        # pd_op.reshape: (1x46x1x1xf32) <- (46xf32, 4xi64)
        t456 = paddle._C_ops.reshape(t141, t241)
        del t141

        # pd_op.add: (-1x46x1x1xf32) <- (-1x46x1x1xf32, 1x46x1x1xf32)
        t457 = paddle._C_ops.add(t455, t456)
        del t455, t456

        # pd_op.relu: (-1x46x1x1xf32) <- (-1x46x1x1xf32)
        t458 = paddle._C_ops.relu(t457)
        del t457

        # pd_op.conv2d: (-1x184x1x1xf32) <- (-1x46x1x1xf32, 184x46x1x1xf32)
        t459 = paddle._C_ops.conv2d(
            t458, t142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t142, t458

        # pd_op.reshape: (1x184x1x1xf32) <- (184xf32, 4xi64)
        t460 = paddle._C_ops.reshape(t143, t241)
        del t143

        # pd_op.add: (-1x184x1x1xf32) <- (-1x184x1x1xf32, 1x184x1x1xf32)
        t461 = paddle._C_ops.add(t459, t460)
        del t459, t460

        # pd_op.hardsigmoid: (-1x184x1x1xf32) <- (-1x184x1x1xf32)
        t462 = paddle._C_ops.hardsigmoid(t461, float("0.2"), float("0.5"))
        del t461

        # pd_op.multiply: (-1x184x14x14xf32) <- (-1x184x14x14xf32, -1x184x1x1xf32)
        t463 = paddle._C_ops.multiply(t453, t462)
        del t462, t453

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x184x14x14xf32, 64x184x1x1xf32)
        t464 = paddle._C_ops.conv2d(
            t463, t144, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t463, t144

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t465, t466, t467, t468, t469, t470 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t464,
                t145,
                t146,
                t147,
                t148,
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
        del t464, t148, t147, t146, t145

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        t471 = paddle._C_ops.add(t432, t465)
        del t432, t465

        # pd_op.conv2d: (-1x360x14x14xf32) <- (-1x64x14x14xf32, 360x64x1x1xf32)
        t472 = paddle._C_ops.conv2d(
            t471, t149, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t471, t149

        # pd_op.batch_norm_: (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32, -1xui8) <- (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        t473, t474, t475, t476, t477, t478 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t472,
                t150,
                t151,
                t152,
                t153,
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
        del t472, t153, t152, t151, t150

        # pd_op.hardswish: (-1x360x14x14xf32) <- (-1x360x14x14xf32)
        t479 = paddle._C_ops.hardswish(t473)
        del t473

        # pd_op.depthwise_conv2d: (-1x360x7x7xf32) <- (-1x360x14x14xf32, 360x1x5x5xf32)
        t480 = paddle._C_ops.depthwise_conv2d(
            t479, t154, [2, 2], [2, 2], "EXPLICIT", 360, [1, 1], "NCHW"
        )
        del t479, t154

        # pd_op.batch_norm_: (-1x360x7x7xf32, 360xf32, 360xf32, 360xf32, 360xf32, -1xui8) <- (-1x360x7x7xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        t481, t482, t483, t484, t485, t486 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t480,
                t155,
                t156,
                t157,
                t158,
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
        del t480, t158, t157, t156, t155

        # pd_op.hardswish: (-1x360x7x7xf32) <- (-1x360x7x7xf32)
        t487 = paddle._C_ops.hardswish(t481)
        del t481

        # pd_op.pool2d: (-1x360x1x1xf32) <- (-1x360x7x7xf32, 2xi64)
        t488 = paddle._C_ops.pool2d(
            t487,
            t238,
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

        # pd_op.conv2d: (-1x90x1x1xf32) <- (-1x360x1x1xf32, 90x360x1x1xf32)
        t489 = paddle._C_ops.conv2d(
            t488, t159, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t159, t488

        # pd_op.reshape: (1x90x1x1xf32) <- (90xf32, 4xi64)
        t490 = paddle._C_ops.reshape(t160, t241)
        del t160

        # pd_op.add: (-1x90x1x1xf32) <- (-1x90x1x1xf32, 1x90x1x1xf32)
        t491 = paddle._C_ops.add(t489, t490)
        del t489, t490

        # pd_op.relu: (-1x90x1x1xf32) <- (-1x90x1x1xf32)
        t492 = paddle._C_ops.relu(t491)
        del t491

        # pd_op.conv2d: (-1x360x1x1xf32) <- (-1x90x1x1xf32, 360x90x1x1xf32)
        t493 = paddle._C_ops.conv2d(
            t492, t161, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t161, t492

        # pd_op.reshape: (1x360x1x1xf32) <- (360xf32, 4xi64)
        t494 = paddle._C_ops.reshape(t162, t241)
        del t162

        # pd_op.add: (-1x360x1x1xf32) <- (-1x360x1x1xf32, 1x360x1x1xf32)
        t495 = paddle._C_ops.add(t493, t494)
        del t493, t494

        # pd_op.hardsigmoid: (-1x360x1x1xf32) <- (-1x360x1x1xf32)
        t496 = paddle._C_ops.hardsigmoid(t495, float("0.2"), float("0.5"))
        del t495

        # pd_op.multiply: (-1x360x7x7xf32) <- (-1x360x7x7xf32, -1x360x1x1xf32)
        t497 = paddle._C_ops.multiply(t487, t496)
        del t496, t487

        # pd_op.conv2d: (-1x120x7x7xf32) <- (-1x360x7x7xf32, 120x360x1x1xf32)
        t498 = paddle._C_ops.conv2d(
            t497, t163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t497, t163

        # pd_op.batch_norm_: (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t499, t500, t501, t502, t503, t504 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t498,
                t164,
                t165,
                t166,
                t167,
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
        del t498, t167, t166, t165, t164

        # pd_op.conv2d: (-1x720x7x7xf32) <- (-1x120x7x7xf32, 720x120x1x1xf32)
        t505 = paddle._C_ops.conv2d(
            t499, t168, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t168

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t506, t507, t508, t509, t510, t511 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t505,
                t169,
                t170,
                t171,
                t172,
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
        del t505, t172, t171, t170, t169

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t512 = paddle._C_ops.hardswish(t506)
        del t506

        # pd_op.depthwise_conv2d: (-1x720x7x7xf32) <- (-1x720x7x7xf32, 720x1x5x5xf32)
        t513 = paddle._C_ops.depthwise_conv2d(
            t512, t173, [1, 1], [2, 2], "EXPLICIT", 720, [1, 1], "NCHW"
        )
        del t512, t173

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t514, t515, t516, t517, t518, t519 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t513,
                t174,
                t175,
                t176,
                t177,
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
        del t513, t177, t176, t175, t174

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t520 = paddle._C_ops.hardswish(t514)
        del t514

        # pd_op.pool2d: (-1x720x1x1xf32) <- (-1x720x7x7xf32, 2xi64)
        t521 = paddle._C_ops.pool2d(
            t520,
            t238,
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

        # pd_op.conv2d: (-1x180x1x1xf32) <- (-1x720x1x1xf32, 180x720x1x1xf32)
        t522 = paddle._C_ops.conv2d(
            t521, t178, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t178, t521

        # pd_op.reshape: (1x180x1x1xf32) <- (180xf32, 4xi64)
        t523 = paddle._C_ops.reshape(t179, t241)
        del t179

        # pd_op.add: (-1x180x1x1xf32) <- (-1x180x1x1xf32, 1x180x1x1xf32)
        t524 = paddle._C_ops.add(t522, t523)
        del t522, t523

        # pd_op.relu: (-1x180x1x1xf32) <- (-1x180x1x1xf32)
        t525 = paddle._C_ops.relu(t524)
        del t524

        # pd_op.conv2d: (-1x720x1x1xf32) <- (-1x180x1x1xf32, 720x180x1x1xf32)
        t526 = paddle._C_ops.conv2d(
            t525, t180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180, t525

        # pd_op.reshape: (1x720x1x1xf32) <- (720xf32, 4xi64)
        t527 = paddle._C_ops.reshape(t181, t241)
        del t181

        # pd_op.add: (-1x720x1x1xf32) <- (-1x720x1x1xf32, 1x720x1x1xf32)
        t528 = paddle._C_ops.add(t526, t527)
        del t526, t527

        # pd_op.hardsigmoid: (-1x720x1x1xf32) <- (-1x720x1x1xf32)
        t529 = paddle._C_ops.hardsigmoid(t528, float("0.2"), float("0.5"))
        del t528

        # pd_op.multiply: (-1x720x7x7xf32) <- (-1x720x7x7xf32, -1x720x1x1xf32)
        t530 = paddle._C_ops.multiply(t520, t529)
        del t529, t520

        # pd_op.conv2d: (-1x120x7x7xf32) <- (-1x720x7x7xf32, 120x720x1x1xf32)
        t531 = paddle._C_ops.conv2d(
            t530, t182, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t530, t182

        # pd_op.batch_norm_: (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t532, t533, t534, t535, t536, t537 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t531,
                t183,
                t184,
                t185,
                t186,
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
        del t531, t186, t185, t184, t183

        # pd_op.add: (-1x120x7x7xf32) <- (-1x120x7x7xf32, -1x120x7x7xf32)
        t538 = paddle._C_ops.add(t499, t532)
        del t499, t532

        # pd_op.conv2d: (-1x720x7x7xf32) <- (-1x120x7x7xf32, 720x120x1x1xf32)
        t539 = paddle._C_ops.conv2d(
            t538, t187, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t187

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t540, t541, t542, t543, t544, t545 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t539,
                t188,
                t189,
                t190,
                t191,
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
        del t539, t191, t190, t189, t188

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t546 = paddle._C_ops.hardswish(t540)
        del t540

        # pd_op.depthwise_conv2d: (-1x720x7x7xf32) <- (-1x720x7x7xf32, 720x1x5x5xf32)
        t547 = paddle._C_ops.depthwise_conv2d(
            t546, t192, [1, 1], [2, 2], "EXPLICIT", 720, [1, 1], "NCHW"
        )
        del t546, t192

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t548, t549, t550, t551, t552, t553 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t547,
                t193,
                t194,
                t195,
                t196,
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
        del t547, t196, t195, t194, t193

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t554 = paddle._C_ops.hardswish(t548)
        del t548

        # pd_op.pool2d: (-1x720x1x1xf32) <- (-1x720x7x7xf32, 2xi64)
        t555 = paddle._C_ops.pool2d(
            t554,
            t238,
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

        # pd_op.conv2d: (-1x180x1x1xf32) <- (-1x720x1x1xf32, 180x720x1x1xf32)
        t556 = paddle._C_ops.conv2d(
            t555, t197, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t197, t555

        # pd_op.reshape: (1x180x1x1xf32) <- (180xf32, 4xi64)
        t557 = paddle._C_ops.reshape(t198, t241)
        del t198

        # pd_op.add: (-1x180x1x1xf32) <- (-1x180x1x1xf32, 1x180x1x1xf32)
        t558 = paddle._C_ops.add(t556, t557)
        del t556, t557

        # pd_op.relu: (-1x180x1x1xf32) <- (-1x180x1x1xf32)
        t559 = paddle._C_ops.relu(t558)
        del t558

        # pd_op.conv2d: (-1x720x1x1xf32) <- (-1x180x1x1xf32, 720x180x1x1xf32)
        t560 = paddle._C_ops.conv2d(
            t559, t199, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t199, t559

        # pd_op.reshape: (1x720x1x1xf32) <- (720xf32, 4xi64)
        t561 = paddle._C_ops.reshape(t200, t241)
        del t241, t200

        # pd_op.add: (-1x720x1x1xf32) <- (-1x720x1x1xf32, 1x720x1x1xf32)
        t562 = paddle._C_ops.add(t560, t561)
        del t560, t561

        # pd_op.hardsigmoid: (-1x720x1x1xf32) <- (-1x720x1x1xf32)
        t563 = paddle._C_ops.hardsigmoid(t562, float("0.2"), float("0.5"))
        del t562

        # pd_op.multiply: (-1x720x7x7xf32) <- (-1x720x7x7xf32, -1x720x1x1xf32)
        t564 = paddle._C_ops.multiply(t554, t563)
        del t563, t554

        # pd_op.conv2d: (-1x120x7x7xf32) <- (-1x720x7x7xf32, 120x720x1x1xf32)
        t565 = paddle._C_ops.conv2d(
            t564, t201, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t564, t201

        # pd_op.batch_norm_: (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t566, t567, t568, t569, t570, t571 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t565,
                t202,
                t203,
                t204,
                t205,
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
        del t565, t203, t202, t205, t204

        # pd_op.add: (-1x120x7x7xf32) <- (-1x120x7x7xf32, -1x120x7x7xf32)
        t572 = paddle._C_ops.add(t538, t566)
        del t538, t566

        # pd_op.conv2d: (-1x720x7x7xf32) <- (-1x120x7x7xf32, 720x120x1x1xf32)
        t573 = paddle._C_ops.conv2d(
            t572, t206, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t572, t206

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t574, t575, t576, t577, t578, t579 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t573,
                t207,
                t208,
                t209,
                t210,
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
        del t573, t210, t209, t208, t207

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t580 = paddle._C_ops.hardswish(t574)
        del t574

        # pd_op.pool2d: (-1x720x1x1xf32) <- (-1x720x7x7xf32, 2xi64)
        t581 = paddle._C_ops.pool2d(
            t580,
            t238,
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
        del t238, t580

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x720x1x1xf32, 1280x720x1x1xf32)
        t582 = paddle._C_ops.conv2d(
            t581, t211, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t211, t581

        # pd_op.hardswish: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        t583 = paddle._C_ops.hardswish(t582)
        del t582

        # pd_op.full: (1xf32) <- ()
        t584 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x1280x1x1xf32, -1x1280x1x1xui8) <- (-1x1280x1x1xf32, None, 1xf32)
        t585, t586 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                t583, None, t584, True, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t584, t583

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        t587 = paddle._C_ops.flatten(t585, 1, 3)
        del t585

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        t588 = paddle._C_ops.matmul(t587, t212, False, False)
        del t587, t212

        return t588
