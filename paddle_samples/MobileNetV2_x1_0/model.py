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
        t213: paddle.Tensor,
        t214: paddle.Tensor,
        t215: paddle.Tensor,
        t216: paddle.Tensor,
        t217: paddle.Tensor,
        t218: paddle.Tensor,
        t219: paddle.Tensor,
        t220: paddle.Tensor,
        t221: paddle.Tensor,
        t222: paddle.Tensor,
        t223: paddle.Tensor,
        t224: paddle.Tensor,
        t225: paddle.Tensor,
        t226: paddle.Tensor,
        t227: paddle.Tensor,
        t228: paddle.Tensor,
        t229: paddle.Tensor,
        t230: paddle.Tensor,
        t231: paddle.Tensor,
        t232: paddle.Tensor,
        t233: paddle.Tensor,
        t234: paddle.Tensor,
        t235: paddle.Tensor,
        t236: paddle.Tensor,
        t237: paddle.Tensor,
        t238: paddle.Tensor,
        t239: paddle.Tensor,
        t240: paddle.Tensor,
        t241: paddle.Tensor,
        t242: paddle.Tensor,
        t243: paddle.Tensor,
        t244: paddle.Tensor,
        t245: paddle.Tensor,
        t246: paddle.Tensor,
        t247: paddle.Tensor,
        t248: paddle.Tensor,
        t249: paddle.Tensor,
        t250: paddle.Tensor,
        t251: paddle.Tensor,
        t252: paddle.Tensor,
        t253: paddle.Tensor,
        t254: paddle.Tensor,
        t255: paddle.Tensor,
        t256: paddle.Tensor,
        t257: paddle.Tensor,
        t258: paddle.Tensor,
        t259: paddle.Tensor,
        t260: paddle.Tensor,
        t261: paddle.Tensor,
        t262: paddle.Tensor,
        t263: paddle.Tensor,
        t264: paddle.Tensor,
        t265: paddle.Tensor,
    ):
        # pd_op.conv2d: (256x32x112x112xf32) <- (256x3x224x224xf32, 32x3x3x3xf32)
        t266 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (256x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t267, t268, t269, t270, t271, t272 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t266,
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

        # pd_op.relu6: (256x32x112x112xf32) <- (256x32x112x112xf32)
        t273 = paddle._C_ops.relu6(t267)
        del t267

        # pd_op.conv2d: (256x32x112x112xf32) <- (256x32x112x112xf32, 32x32x1x1xf32)
        t274 = paddle._C_ops.conv2d(
            t273, t5, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (256x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t275, t276, t277, t278, t279, t280 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t274,
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

        # pd_op.relu6: (256x32x112x112xf32) <- (256x32x112x112xf32)
        t281 = paddle._C_ops.relu6(t275)
        del t275

        # pd_op.depthwise_conv2d: (256x32x112x112xf32) <- (256x32x112x112xf32, 32x1x3x3xf32)
        t282 = paddle._C_ops.depthwise_conv2d(
            t281, t10, [1, 1], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (256x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t283, t284, t285, t286, t287, t288 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t282,
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

        # pd_op.relu6: (256x32x112x112xf32) <- (256x32x112x112xf32)
        t289 = paddle._C_ops.relu6(t283)
        del t283

        # pd_op.conv2d: (256x16x112x112xf32) <- (256x32x112x112xf32, 16x32x1x1xf32)
        t290 = paddle._C_ops.conv2d(
            t289, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (256x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (256x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t291, t292, t293, t294, t295, t296 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t290,
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

        # pd_op.conv2d: (256x96x112x112xf32) <- (256x16x112x112xf32, 96x16x1x1xf32)
        t297 = paddle._C_ops.conv2d(
            t291, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (256x96x112x112xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (256x96x112x112xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t298, t299, t300, t301, t302, t303 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t297,
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

        # pd_op.relu6: (256x96x112x112xf32) <- (256x96x112x112xf32)
        t304 = paddle._C_ops.relu6(t298)
        del t298

        # pd_op.depthwise_conv2d: (256x96x56x56xf32) <- (256x96x112x112xf32, 96x1x3x3xf32)
        t305 = paddle._C_ops.depthwise_conv2d(
            t304, t25, [2, 2], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (256x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (256x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t306, t307, t308, t309, t310, t311 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t305,
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

        # pd_op.relu6: (256x96x56x56xf32) <- (256x96x56x56xf32)
        t312 = paddle._C_ops.relu6(t306)
        del t306

        # pd_op.conv2d: (256x24x56x56xf32) <- (256x96x56x56xf32, 24x96x1x1xf32)
        t313 = paddle._C_ops.conv2d(
            t312, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (256x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (256x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t314, t315, t316, t317, t318, t319 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t313,
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

        # pd_op.conv2d: (256x144x56x56xf32) <- (256x24x56x56xf32, 144x24x1x1xf32)
        t320 = paddle._C_ops.conv2d(
            t314, t35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (256x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (256x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        t321, t322, t323, t324, t325, t326 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t320,
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

        # pd_op.relu6: (256x144x56x56xf32) <- (256x144x56x56xf32)
        t327 = paddle._C_ops.relu6(t321)
        del t321

        # pd_op.depthwise_conv2d: (256x144x56x56xf32) <- (256x144x56x56xf32, 144x1x3x3xf32)
        t328 = paddle._C_ops.depthwise_conv2d(
            t327, t40, [1, 1], [1, 1], "EXPLICIT", 144, [1, 1], "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (256x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (256x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        t329, t330, t331, t332, t333, t334 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t328,
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

        # pd_op.relu6: (256x144x56x56xf32) <- (256x144x56x56xf32)
        t335 = paddle._C_ops.relu6(t329)
        del t329

        # pd_op.conv2d: (256x24x56x56xf32) <- (256x144x56x56xf32, 24x144x1x1xf32)
        t336 = paddle._C_ops.conv2d(
            t335, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (256x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (256x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t337, t338, t339, t340, t341, t342 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t336,
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

        # pd_op.add: (256x24x56x56xf32) <- (256x24x56x56xf32, 256x24x56x56xf32)
        t343 = paddle._C_ops.add(t314, t337)

        # pd_op.conv2d: (256x144x56x56xf32) <- (256x24x56x56xf32, 144x24x1x1xf32)
        t344 = paddle._C_ops.conv2d(
            t343, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (256x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (256x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        t345, t346, t347, t348, t349, t350 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t344,
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

        # pd_op.relu6: (256x144x56x56xf32) <- (256x144x56x56xf32)
        t351 = paddle._C_ops.relu6(t345)
        del t345

        # pd_op.depthwise_conv2d: (256x144x28x28xf32) <- (256x144x56x56xf32, 144x1x3x3xf32)
        t352 = paddle._C_ops.depthwise_conv2d(
            t351, t55, [2, 2], [1, 1], "EXPLICIT", 144, [1, 1], "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (256x144x28x28xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (256x144x28x28xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        t353, t354, t355, t356, t357, t358 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t352,
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

        # pd_op.relu6: (256x144x28x28xf32) <- (256x144x28x28xf32)
        t359 = paddle._C_ops.relu6(t353)
        del t353

        # pd_op.conv2d: (256x32x28x28xf32) <- (256x144x28x28xf32, 32x144x1x1xf32)
        t360 = paddle._C_ops.conv2d(
            t359, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t361, t362, t363, t364, t365, t366 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t360,
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

        # pd_op.conv2d: (256x192x28x28xf32) <- (256x32x28x28xf32, 192x32x1x1xf32)
        t367 = paddle._C_ops.conv2d(
            t361, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t368, t369, t370, t371, t372, t373 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t367,
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

        # pd_op.relu6: (256x192x28x28xf32) <- (256x192x28x28xf32)
        t374 = paddle._C_ops.relu6(t368)
        del t368

        # pd_op.depthwise_conv2d: (256x192x28x28xf32) <- (256x192x28x28xf32, 192x1x3x3xf32)
        t375 = paddle._C_ops.depthwise_conv2d(
            t374, t70, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t376, t377, t378, t379, t380, t381 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t375,
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

        # pd_op.relu6: (256x192x28x28xf32) <- (256x192x28x28xf32)
        t382 = paddle._C_ops.relu6(t376)
        del t376

        # pd_op.conv2d: (256x32x28x28xf32) <- (256x192x28x28xf32, 32x192x1x1xf32)
        t383 = paddle._C_ops.conv2d(
            t382, t75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t384, t385, t386, t387, t388, t389 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t383,
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

        # pd_op.add: (256x32x28x28xf32) <- (256x32x28x28xf32, 256x32x28x28xf32)
        t390 = paddle._C_ops.add(t361, t384)

        # pd_op.conv2d: (256x192x28x28xf32) <- (256x32x28x28xf32, 192x32x1x1xf32)
        t391 = paddle._C_ops.conv2d(
            t390, t80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t392, t393, t394, t395, t396, t397 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t391,
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

        # pd_op.relu6: (256x192x28x28xf32) <- (256x192x28x28xf32)
        t398 = paddle._C_ops.relu6(t392)
        del t392

        # pd_op.depthwise_conv2d: (256x192x28x28xf32) <- (256x192x28x28xf32, 192x1x3x3xf32)
        t399 = paddle._C_ops.depthwise_conv2d(
            t398, t85, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t400, t401, t402, t403, t404, t405 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t399,
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

        # pd_op.relu6: (256x192x28x28xf32) <- (256x192x28x28xf32)
        t406 = paddle._C_ops.relu6(t400)
        del t400

        # pd_op.conv2d: (256x32x28x28xf32) <- (256x192x28x28xf32, 32x192x1x1xf32)
        t407 = paddle._C_ops.conv2d(
            t406, t90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (256x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t408, t409, t410, t411, t412, t413 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t407,
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

        # pd_op.add: (256x32x28x28xf32) <- (256x32x28x28xf32, 256x32x28x28xf32)
        t414 = paddle._C_ops.add(t390, t408)

        # pd_op.conv2d: (256x192x28x28xf32) <- (256x32x28x28xf32, 192x32x1x1xf32)
        t415 = paddle._C_ops.conv2d(
            t414, t95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (256x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t416, t417, t418, t419, t420, t421 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t415,
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

        # pd_op.relu6: (256x192x28x28xf32) <- (256x192x28x28xf32)
        t422 = paddle._C_ops.relu6(t416)
        del t416

        # pd_op.depthwise_conv2d: (256x192x14x14xf32) <- (256x192x28x28xf32, 192x1x3x3xf32)
        t423 = paddle._C_ops.depthwise_conv2d(
            t422, t100, [2, 2], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (256x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (256x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t424, t425, t426, t427, t428, t429 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t423,
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

        # pd_op.relu6: (256x192x14x14xf32) <- (256x192x14x14xf32)
        t430 = paddle._C_ops.relu6(t424)
        del t424

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x192x14x14xf32, 64x192x1x1xf32)
        t431 = paddle._C_ops.conv2d(
            t430, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t432, t433, t434, t435, t436, t437 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t431,
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

        # pd_op.conv2d: (256x384x14x14xf32) <- (256x64x14x14xf32, 384x64x1x1xf32)
        t438 = paddle._C_ops.conv2d(
            t432, t110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t439, t440, t441, t442, t443, t444 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t438,
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

        # pd_op.relu6: (256x384x14x14xf32) <- (256x384x14x14xf32)
        t445 = paddle._C_ops.relu6(t439)
        del t439

        # pd_op.depthwise_conv2d: (256x384x14x14xf32) <- (256x384x14x14xf32, 384x1x3x3xf32)
        t446 = paddle._C_ops.depthwise_conv2d(
            t445, t115, [1, 1], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t447, t448, t449, t450, t451, t452 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t446,
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

        # pd_op.relu6: (256x384x14x14xf32) <- (256x384x14x14xf32)
        t453 = paddle._C_ops.relu6(t447)
        del t447

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x384x14x14xf32, 64x384x1x1xf32)
        t454 = paddle._C_ops.conv2d(
            t453, t120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t455, t456, t457, t458, t459, t460 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t454,
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

        # pd_op.add: (256x64x14x14xf32) <- (256x64x14x14xf32, 256x64x14x14xf32)
        t461 = paddle._C_ops.add(t432, t455)

        # pd_op.conv2d: (256x384x14x14xf32) <- (256x64x14x14xf32, 384x64x1x1xf32)
        t462 = paddle._C_ops.conv2d(
            t461, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t463, t464, t465, t466, t467, t468 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t462,
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

        # pd_op.relu6: (256x384x14x14xf32) <- (256x384x14x14xf32)
        t469 = paddle._C_ops.relu6(t463)
        del t463

        # pd_op.depthwise_conv2d: (256x384x14x14xf32) <- (256x384x14x14xf32, 384x1x3x3xf32)
        t470 = paddle._C_ops.depthwise_conv2d(
            t469, t130, [1, 1], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t471, t472, t473, t474, t475, t476 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t470,
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

        # pd_op.relu6: (256x384x14x14xf32) <- (256x384x14x14xf32)
        t477 = paddle._C_ops.relu6(t471)
        del t471

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x384x14x14xf32, 64x384x1x1xf32)
        t478 = paddle._C_ops.conv2d(
            t477, t135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t479, t480, t481, t482, t483, t484 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t478,
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

        # pd_op.add: (256x64x14x14xf32) <- (256x64x14x14xf32, 256x64x14x14xf32)
        t485 = paddle._C_ops.add(t461, t479)

        # pd_op.conv2d: (256x384x14x14xf32) <- (256x64x14x14xf32, 384x64x1x1xf32)
        t486 = paddle._C_ops.conv2d(
            t485, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t487, t488, t489, t490, t491, t492 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t486,
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

        # pd_op.relu6: (256x384x14x14xf32) <- (256x384x14x14xf32)
        t493 = paddle._C_ops.relu6(t487)
        del t487

        # pd_op.depthwise_conv2d: (256x384x14x14xf32) <- (256x384x14x14xf32, 384x1x3x3xf32)
        t494 = paddle._C_ops.depthwise_conv2d(
            t493, t145, [1, 1], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t145

        # pd_op.batch_norm_: (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t495, t496, t497, t498, t499, t500 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t494,
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

        # pd_op.relu6: (256x384x14x14xf32) <- (256x384x14x14xf32)
        t501 = paddle._C_ops.relu6(t495)
        del t495

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x384x14x14xf32, 64x384x1x1xf32)
        t502 = paddle._C_ops.conv2d(
            t501, t150, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t503, t504, t505, t506, t507, t508 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t502,
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

        # pd_op.add: (256x64x14x14xf32) <- (256x64x14x14xf32, 256x64x14x14xf32)
        t509 = paddle._C_ops.add(t485, t503)

        # pd_op.conv2d: (256x384x14x14xf32) <- (256x64x14x14xf32, 384x64x1x1xf32)
        t510 = paddle._C_ops.conv2d(
            t509, t155, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # pd_op.batch_norm_: (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t511, t512, t513, t514, t515, t516 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t510,
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

        # pd_op.relu6: (256x384x14x14xf32) <- (256x384x14x14xf32)
        t517 = paddle._C_ops.relu6(t511)
        del t511

        # pd_op.depthwise_conv2d: (256x384x14x14xf32) <- (256x384x14x14xf32, 384x1x3x3xf32)
        t518 = paddle._C_ops.depthwise_conv2d(
            t517, t160, [1, 1], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (256x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t519, t520, t521, t522, t523, t524 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t518,
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

        # pd_op.relu6: (256x384x14x14xf32) <- (256x384x14x14xf32)
        t525 = paddle._C_ops.relu6(t519)
        del t519

        # pd_op.conv2d: (256x96x14x14xf32) <- (256x384x14x14xf32, 96x384x1x1xf32)
        t526 = paddle._C_ops.conv2d(
            t525, t165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165

        # pd_op.batch_norm_: (256x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (256x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t527, t528, t529, t530, t531, t532 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t526,
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
        del t166, t169, t168, t167

        # pd_op.conv2d: (256x576x14x14xf32) <- (256x96x14x14xf32, 576x96x1x1xf32)
        t533 = paddle._C_ops.conv2d(
            t527, t170, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t534, t535, t536, t537, t538, t539 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t533,
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

        # pd_op.relu6: (256x576x14x14xf32) <- (256x576x14x14xf32)
        t540 = paddle._C_ops.relu6(t534)
        del t534

        # pd_op.depthwise_conv2d: (256x576x14x14xf32) <- (256x576x14x14xf32, 576x1x3x3xf32)
        t541 = paddle._C_ops.depthwise_conv2d(
            t540, t175, [1, 1], [1, 1], "EXPLICIT", 576, [1, 1], "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t542, t543, t544, t545, t546, t547 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t541,
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
        del t179, t178, t177, t176

        # pd_op.relu6: (256x576x14x14xf32) <- (256x576x14x14xf32)
        t548 = paddle._C_ops.relu6(t542)
        del t542

        # pd_op.conv2d: (256x96x14x14xf32) <- (256x576x14x14xf32, 96x576x1x1xf32)
        t549 = paddle._C_ops.conv2d(
            t548, t180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180

        # pd_op.batch_norm_: (256x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (256x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t550, t551, t552, t553, t554, t555 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t549,
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

        # pd_op.add: (256x96x14x14xf32) <- (256x96x14x14xf32, 256x96x14x14xf32)
        t556 = paddle._C_ops.add(t527, t550)

        # pd_op.conv2d: (256x576x14x14xf32) <- (256x96x14x14xf32, 576x96x1x1xf32)
        t557 = paddle._C_ops.conv2d(
            t556, t185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185

        # pd_op.batch_norm_: (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t558, t559, t560, t561, t562, t563 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t557,
                t186,
                t187,
                t188,
                t189,
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
        del t189, t188, t187, t186

        # pd_op.relu6: (256x576x14x14xf32) <- (256x576x14x14xf32)
        t564 = paddle._C_ops.relu6(t558)
        del t558

        # pd_op.depthwise_conv2d: (256x576x14x14xf32) <- (256x576x14x14xf32, 576x1x3x3xf32)
        t565 = paddle._C_ops.depthwise_conv2d(
            t564, t190, [1, 1], [1, 1], "EXPLICIT", 576, [1, 1], "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t566, t567, t568, t569, t570, t571 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t565,
                t191,
                t192,
                t193,
                t194,
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
        del t194, t193, t192, t191

        # pd_op.relu6: (256x576x14x14xf32) <- (256x576x14x14xf32)
        t572 = paddle._C_ops.relu6(t566)
        del t566

        # pd_op.conv2d: (256x96x14x14xf32) <- (256x576x14x14xf32, 96x576x1x1xf32)
        t573 = paddle._C_ops.conv2d(
            t572, t195, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195

        # pd_op.batch_norm_: (256x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (256x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t574, t575, t576, t577, t578, t579 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t573,
                t196,
                t197,
                t198,
                t199,
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
        del t199, t198, t197, t196

        # pd_op.add: (256x96x14x14xf32) <- (256x96x14x14xf32, 256x96x14x14xf32)
        t580 = paddle._C_ops.add(t556, t574)

        # pd_op.conv2d: (256x576x14x14xf32) <- (256x96x14x14xf32, 576x96x1x1xf32)
        t581 = paddle._C_ops.conv2d(
            t580, t200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (256x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t582, t583, t584, t585, t586, t587 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t581,
                t201,
                t202,
                t203,
                t204,
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
        del t204, t203, t202, t201

        # pd_op.relu6: (256x576x14x14xf32) <- (256x576x14x14xf32)
        t588 = paddle._C_ops.relu6(t582)
        del t582

        # pd_op.depthwise_conv2d: (256x576x7x7xf32) <- (256x576x14x14xf32, 576x1x3x3xf32)
        t589 = paddle._C_ops.depthwise_conv2d(
            t588, t205, [2, 2], [1, 1], "EXPLICIT", 576, [1, 1], "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (256x576x7x7xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (256x576x7x7xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t590, t591, t592, t593, t594, t595 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t589,
                t206,
                t207,
                t208,
                t209,
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
        del t209, t208, t207, t206

        # pd_op.relu6: (256x576x7x7xf32) <- (256x576x7x7xf32)
        t596 = paddle._C_ops.relu6(t590)
        del t590

        # pd_op.conv2d: (256x160x7x7xf32) <- (256x576x7x7xf32, 160x576x1x1xf32)
        t597 = paddle._C_ops.conv2d(
            t596, t210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (256x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (256x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t598, t599, t600, t601, t602, t603 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t597,
                t211,
                t212,
                t213,
                t214,
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
        del t214, t213, t212, t211

        # pd_op.conv2d: (256x960x7x7xf32) <- (256x160x7x7xf32, 960x160x1x1xf32)
        t604 = paddle._C_ops.conv2d(
            t598, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.batch_norm_: (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t605, t606, t607, t608, t609, t610 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t604,
                t216,
                t217,
                t218,
                t219,
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
        del t219, t218, t217, t216

        # pd_op.relu6: (256x960x7x7xf32) <- (256x960x7x7xf32)
        t611 = paddle._C_ops.relu6(t605)
        del t605

        # pd_op.depthwise_conv2d: (256x960x7x7xf32) <- (256x960x7x7xf32, 960x1x3x3xf32)
        t612 = paddle._C_ops.depthwise_conv2d(
            t611, t220, [1, 1], [1, 1], "EXPLICIT", 960, [1, 1], "NCHW"
        )
        del t220

        # pd_op.batch_norm_: (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t613, t614, t615, t616, t617, t618 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t612,
                t221,
                t222,
                t223,
                t224,
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
        del t224, t223, t222, t221

        # pd_op.relu6: (256x960x7x7xf32) <- (256x960x7x7xf32)
        t619 = paddle._C_ops.relu6(t613)
        del t613

        # pd_op.conv2d: (256x160x7x7xf32) <- (256x960x7x7xf32, 160x960x1x1xf32)
        t620 = paddle._C_ops.conv2d(
            t619, t225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225

        # pd_op.batch_norm_: (256x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (256x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t621, t622, t623, t624, t625, t626 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t620,
                t226,
                t227,
                t228,
                t229,
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
        del t229, t228, t227, t226

        # pd_op.add: (256x160x7x7xf32) <- (256x160x7x7xf32, 256x160x7x7xf32)
        t627 = paddle._C_ops.add(t598, t621)

        # pd_op.conv2d: (256x960x7x7xf32) <- (256x160x7x7xf32, 960x160x1x1xf32)
        t628 = paddle._C_ops.conv2d(
            t627, t230, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t230

        # pd_op.batch_norm_: (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t629, t630, t631, t632, t633, t634 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t628,
                t231,
                t232,
                t233,
                t234,
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
        del t234, t233, t232, t231

        # pd_op.relu6: (256x960x7x7xf32) <- (256x960x7x7xf32)
        t635 = paddle._C_ops.relu6(t629)
        del t629

        # pd_op.depthwise_conv2d: (256x960x7x7xf32) <- (256x960x7x7xf32, 960x1x3x3xf32)
        t636 = paddle._C_ops.depthwise_conv2d(
            t635, t235, [1, 1], [1, 1], "EXPLICIT", 960, [1, 1], "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t637, t638, t639, t640, t641, t642 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t636,
                t236,
                t237,
                t238,
                t239,
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
        del t239, t238, t237, t236

        # pd_op.relu6: (256x960x7x7xf32) <- (256x960x7x7xf32)
        t643 = paddle._C_ops.relu6(t637)
        del t637

        # pd_op.conv2d: (256x160x7x7xf32) <- (256x960x7x7xf32, 160x960x1x1xf32)
        t644 = paddle._C_ops.conv2d(
            t643, t240, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t240

        # pd_op.batch_norm_: (256x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (256x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t645, t646, t647, t648, t649, t650 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t644,
                t241,
                t242,
                t243,
                t244,
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
        del t244, t243, t242, t241

        # pd_op.add: (256x160x7x7xf32) <- (256x160x7x7xf32, 256x160x7x7xf32)
        t651 = paddle._C_ops.add(t627, t645)

        # pd_op.conv2d: (256x960x7x7xf32) <- (256x160x7x7xf32, 960x160x1x1xf32)
        t652 = paddle._C_ops.conv2d(
            t651, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t653, t654, t655, t656, t657, t658 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t652,
                t246,
                t247,
                t248,
                t249,
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
        del t249, t248, t247, t246

        # pd_op.relu6: (256x960x7x7xf32) <- (256x960x7x7xf32)
        t659 = paddle._C_ops.relu6(t653)
        del t653

        # pd_op.depthwise_conv2d: (256x960x7x7xf32) <- (256x960x7x7xf32, 960x1x3x3xf32)
        t660 = paddle._C_ops.depthwise_conv2d(
            t659, t250, [1, 1], [1, 1], "EXPLICIT", 960, [1, 1], "NCHW"
        )
        del t250

        # pd_op.batch_norm_: (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (256x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t661, t662, t663, t664, t665, t666 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t660,
                t251,
                t252,
                t253,
                t254,
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
        del t254, t253, t252, t251

        # pd_op.relu6: (256x960x7x7xf32) <- (256x960x7x7xf32)
        t667 = paddle._C_ops.relu6(t661)
        del t661

        # pd_op.conv2d: (256x320x7x7xf32) <- (256x960x7x7xf32, 320x960x1x1xf32)
        t668 = paddle._C_ops.conv2d(
            t667, t255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255

        # pd_op.batch_norm_: (256x320x7x7xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (256x320x7x7xf32, 320xf32, 320xf32, 320xf32, 320xf32)
        t669, t670, t671, t672, t673, t674 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t668,
                t256,
                t257,
                t258,
                t259,
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
        del t256, t259, t258, t257

        # pd_op.conv2d: (256x1280x7x7xf32) <- (256x320x7x7xf32, 1280x320x1x1xf32)
        t675 = paddle._C_ops.conv2d(
            t669, t260, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t260

        # pd_op.batch_norm_: (256x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32, -1xui8) <- (256x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        t676, t677, t678, t679, t680, t681 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t675,
                t261,
                t262,
                t263,
                t264,
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
        del t264, t263, t262, t261

        # pd_op.relu6: (256x1280x7x7xf32) <- (256x1280x7x7xf32)
        t682 = paddle._C_ops.relu6(t676)
        del t676

        # pd_op.full_int_array: (2xi64) <- ()
        t683 = [1, 1]

        # pd_op.pool2d: (256x1280x1x1xf32) <- (256x1280x7x7xf32, 2xi64)
        t684 = paddle._C_ops.pool2d(
            t682,
            t683,
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

        # pd_op.flatten: (256x1280xf32) <- (256x1280x1x1xf32)
        t685 = paddle._C_ops.flatten(t684, 1, 3)

        # pd_op.matmul: (256x102xf32) <- (256x1280xf32, 1280x102xf32)
        t686 = paddle._C_ops.matmul(t685, t265, False, False)
        del t265

        return (
            t266,
            t268,
            t269,
            t270,
            t271,
            t272,
            t273,
            t274,
            t276,
            t277,
            t278,
            t279,
            t280,
            t281,
            t282,
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
            t299,
            t300,
            t301,
            t302,
            t303,
            t304,
            t305,
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
            t337,
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
            t369,
            t370,
            t371,
            t372,
            t373,
            t374,
            t375,
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
            t388,
            t389,
            t390,
            t391,
            t393,
            t394,
            t395,
            t396,
            t397,
            t398,
            t399,
            t401,
            t402,
            t403,
            t404,
            t405,
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
            t417,
            t418,
            t419,
            t420,
            t421,
            t422,
            t423,
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
            t436,
            t437,
            t438,
            t440,
            t441,
            t442,
            t443,
            t444,
            t445,
            t446,
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
            t460,
            t461,
            t462,
            t464,
            t465,
            t466,
            t467,
            t468,
            t469,
            t470,
            t472,
            t473,
            t474,
            t475,
            t476,
            t477,
            t478,
            t479,
            t480,
            t481,
            t482,
            t483,
            t484,
            t485,
            t486,
            t488,
            t489,
            t490,
            t491,
            t492,
            t493,
            t494,
            t496,
            t497,
            t498,
            t499,
            t500,
            t501,
            t502,
            t503,
            t504,
            t505,
            t506,
            t507,
            t508,
            t509,
            t510,
            t512,
            t513,
            t514,
            t515,
            t516,
            t517,
            t518,
            t520,
            t521,
            t522,
            t523,
            t524,
            t525,
            t526,
            t527,
            t528,
            t529,
            t530,
            t531,
            t532,
            t533,
            t535,
            t536,
            t537,
            t538,
            t539,
            t540,
            t541,
            t543,
            t544,
            t545,
            t546,
            t547,
            t548,
            t549,
            t550,
            t551,
            t552,
            t553,
            t554,
            t555,
            t556,
            t557,
            t559,
            t560,
            t561,
            t562,
            t563,
            t564,
            t565,
            t567,
            t568,
            t569,
            t570,
            t571,
            t572,
            t573,
            t574,
            t575,
            t576,
            t577,
            t578,
            t579,
            t580,
            t581,
            t583,
            t584,
            t585,
            t586,
            t587,
            t588,
            t589,
            t591,
            t592,
            t593,
            t594,
            t595,
            t596,
            t597,
            t598,
            t599,
            t600,
            t601,
            t602,
            t603,
            t604,
            t606,
            t607,
            t608,
            t609,
            t610,
            t611,
            t612,
            t614,
            t615,
            t616,
            t617,
            t618,
            t619,
            t620,
            t621,
            t622,
            t623,
            t624,
            t625,
            t626,
            t627,
            t628,
            t630,
            t631,
            t632,
            t633,
            t634,
            t635,
            t636,
            t638,
            t639,
            t640,
            t641,
            t642,
            t643,
            t644,
            t645,
            t646,
            t647,
            t648,
            t649,
            t650,
            t651,
            t652,
            t654,
            t655,
            t656,
            t657,
            t658,
            t659,
            t660,
            t662,
            t663,
            t664,
            t665,
            t666,
            t667,
            t668,
            t669,
            t670,
            t671,
            t672,
            t673,
            t674,
            t675,
            t677,
            t678,
            t679,
            t680,
            t681,
            t682,
            t683,
            t684,
            t685,
            t686,
        )
