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
        t266: paddle.Tensor,
        t267: paddle.Tensor,
        t268: paddle.Tensor,
        t269: paddle.Tensor,
    ):
        t270 = None
        # pd_op.conv2d: (128x32x112x112xf32) <- (128x3x224x224xf32, 32x3x3x3xf32)
        t271 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (128x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (128x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t272, t273, t274, t275, t276, t277 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t271,
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

        # pd_op.relu: (128x32x112x112xf32) <- (128x32x112x112xf32)
        t278 = paddle._C_ops.relu(t272)
        del t272

        # pd_op.depthwise_conv2d: (128x32x112x112xf32) <- (128x32x112x112xf32, 32x1x3x3xf32)
        t279 = paddle._C_ops.depthwise_conv2d(
            t278, t5, [1, 1], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (128x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (128x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t280, t281, t282, t283, t284, t285 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t279,
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

        # pd_op.relu: (128x32x112x112xf32) <- (128x32x112x112xf32)
        t286 = paddle._C_ops.relu(t280)
        del t280

        # pd_op.conv2d: (128x64x112x112xf32) <- (128x32x112x112xf32, 64x32x1x1xf32)
        t287 = paddle._C_ops.conv2d(
            t286, t10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (128x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t288, t289, t290, t291, t292, t293 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t287,
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

        # pd_op.relu: (128x64x112x112xf32) <- (128x64x112x112xf32)
        t294 = paddle._C_ops.relu(t288)
        del t288

        # pd_op.depthwise_conv2d: (128x64x56x56xf32) <- (128x64x112x112xf32, 64x1x3x3xf32)
        t295 = paddle._C_ops.depthwise_conv2d(
            t294, t15, [2, 2], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (128x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t296, t297, t298, t299, t300, t301 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t295,
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

        # pd_op.relu: (128x64x56x56xf32) <- (128x64x56x56xf32)
        t302 = paddle._C_ops.relu(t296)
        del t296

        # pd_op.conv2d: (128x128x56x56xf32) <- (128x64x56x56xf32, 128x64x1x1xf32)
        t303 = paddle._C_ops.conv2d(
            t302, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t304, t305, t306, t307, t308, t309 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t303,
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

        # pd_op.relu: (128x128x56x56xf32) <- (128x128x56x56xf32)
        t310 = paddle._C_ops.relu(t304)
        del t304

        # pd_op.depthwise_conv2d: (128x128x56x56xf32) <- (128x128x56x56xf32, 128x1x3x3xf32)
        t311 = paddle._C_ops.depthwise_conv2d(
            t310, t25, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t312, t313, t314, t315, t316, t317 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t311,
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

        # pd_op.relu: (128x128x56x56xf32) <- (128x128x56x56xf32)
        t318 = paddle._C_ops.relu(t312)
        del t312

        # pd_op.conv2d: (128x128x56x56xf32) <- (128x128x56x56xf32, 128x128x1x1xf32)
        t319 = paddle._C_ops.conv2d(
            t318, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t320, t321, t322, t323, t324, t325 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t319,
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

        # pd_op.relu: (128x128x56x56xf32) <- (128x128x56x56xf32)
        t326 = paddle._C_ops.relu(t320)
        del t320

        # pd_op.depthwise_conv2d: (128x128x28x28xf32) <- (128x128x56x56xf32, 128x1x3x3xf32)
        t327 = paddle._C_ops.depthwise_conv2d(
            t326, t35, [2, 2], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t328, t329, t330, t331, t332, t333 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t327,
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

        # pd_op.relu: (128x128x28x28xf32) <- (128x128x28x28xf32)
        t334 = paddle._C_ops.relu(t328)
        del t328

        # pd_op.conv2d: (128x256x28x28xf32) <- (128x128x28x28xf32, 256x128x1x1xf32)
        t335 = paddle._C_ops.conv2d(
            t334, t40, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t336, t337, t338, t339, t340, t341 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t335,
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

        # pd_op.relu: (128x256x28x28xf32) <- (128x256x28x28xf32)
        t342 = paddle._C_ops.relu(t336)
        del t336

        # pd_op.depthwise_conv2d: (128x256x28x28xf32) <- (128x256x28x28xf32, 256x1x3x3xf32)
        t343 = paddle._C_ops.depthwise_conv2d(
            t342, t45, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t344, t345, t346, t347, t348, t349 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t343,
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

        # pd_op.relu: (128x256x28x28xf32) <- (128x256x28x28xf32)
        t350 = paddle._C_ops.relu(t344)
        del t344

        # pd_op.conv2d: (128x256x28x28xf32) <- (128x256x28x28xf32, 256x256x1x1xf32)
        t351 = paddle._C_ops.conv2d(
            t350, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t352, t353, t354, t355, t356, t357 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t351,
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

        # pd_op.relu: (128x256x28x28xf32) <- (128x256x28x28xf32)
        t358 = paddle._C_ops.relu(t352)
        del t352

        # pd_op.depthwise_conv2d: (128x256x14x14xf32) <- (128x256x28x28xf32, 256x1x5x5xf32)
        t359 = paddle._C_ops.depthwise_conv2d(
            t358, t55, [2, 2], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t360, t361, t362, t363, t364, t365 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t359,
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

        # pd_op.depthwise_conv2d: (128x256x14x14xf32) <- (128x256x28x28xf32, 256x1x3x3xf32)
        t366 = paddle._C_ops.depthwise_conv2d(
            t358, t60, [2, 2], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t367, t368, t369, t370, t371, t372 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t366,
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

        # pd_op.add: (128x256x14x14xf32) <- (128x256x14x14xf32, 128x256x14x14xf32)
        t373 = paddle._C_ops.add(t360, t367)

        # pd_op.relu: (128x256x14x14xf32) <- (128x256x14x14xf32)
        t374 = paddle._C_ops.relu(t373)
        del t373

        # pd_op.full_int_array: (2xi64) <- ()
        t375 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        t376 = t375

        # pd_op.assign: (2xi64) <- (2xi64)
        t377 = t375

        # pd_op.assign: (2xi64) <- (2xi64)
        t378 = t375

        # pd_op.assign: (2xi64) <- (2xi64)
        t379 = t375

        # pd_op.assign: (2xi64) <- (2xi64)
        t380 = t375

        # pd_op.assign: (2xi64) <- (2xi64)
        t381 = t375

        # pd_op.pool2d: (128x256x1x1xf32) <- (128x256x14x14xf32, 2xi64)
        t382 = paddle._C_ops.pool2d(
            t374,
            t375,
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

        # pd_op.conv2d: (128x64x1x1xf32) <- (128x256x1x1xf32, 64x256x1x1xf32)
        t383 = paddle._C_ops.conv2d(
            t382, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.full_int_array: (4xi64) <- ()
        t384 = [1, -1, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t385 = paddle._C_ops.reshape(t66, t384)
        del t66

        # pd_op.add: (128x64x1x1xf32) <- (128x64x1x1xf32, 1x64x1x1xf32)
        t386 = paddle._C_ops.add(t383, t385)

        # pd_op.relu: (128x64x1x1xf32) <- (128x64x1x1xf32)
        t387 = paddle._C_ops.relu(t386)
        del t386

        # pd_op.conv2d: (128x256x1x1xf32) <- (128x64x1x1xf32, 256x64x1x1xf32)
        t388 = paddle._C_ops.conv2d(
            t387, t67, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t67

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t389 = paddle._C_ops.reshape(t68, t384)
        del t68

        # pd_op.add: (128x256x1x1xf32) <- (128x256x1x1xf32, 1x256x1x1xf32)
        t390 = paddle._C_ops.add(t388, t389)

        # pd_op.sigmoid: (128x256x1x1xf32) <- (128x256x1x1xf32)
        t391 = paddle._C_ops.sigmoid(t390)
        del t390

        # pd_op.multiply: (128x256x14x14xf32) <- (128x256x14x14xf32, 128x256x1x1xf32)
        t392 = paddle._C_ops.multiply(t374, t391)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x256x14x14xf32, 256x256x1x1xf32)
        t393 = paddle._C_ops.conv2d(
            t392, t69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t69

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t394, t395, t396, t397, t398, t399 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t393,
                t70,
                t71,
                t72,
                t73,
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
        del t73, t72, t71, t70

        # pd_op.relu: (128x256x14x14xf32) <- (128x256x14x14xf32)
        t400 = paddle._C_ops.relu(t394)
        del t394

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x256x14x14xf32, 512x256x1x1xf32)
        t401 = paddle._C_ops.conv2d(
            t400, t74, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t74

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t402, t403, t404, t405, t406, t407 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t401,
                t75,
                t76,
                t77,
                t78,
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
        del t78, t77, t76, t75

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t408 = paddle._C_ops.relu(t402)
        del t402

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x5x5xf32)
        t409 = paddle._C_ops.depthwise_conv2d(
            t408, t79, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t79

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t410, t411, t412, t413, t414, t415 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t409,
                t80,
                t81,
                t82,
                t83,
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
        del t83, t82, t81, t80

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x3x3xf32)
        t416 = paddle._C_ops.depthwise_conv2d(
            t408, t84, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t84

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t417, t418, t419, t420, t421, t422 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t416,
                t85,
                t86,
                t87,
                t88,
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
        del t88, t87, t86, t85

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t423 = paddle._C_ops.add(t410, t417)

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x1x1xf32)
        t424 = paddle._C_ops.depthwise_conv2d(
            t408, t89, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t89

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t425, t426, t427, t428, t429, t430 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t424,
                t90,
                t91,
                t92,
                t93,
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
        del t93, t92, t91, t90

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t431 = paddle._C_ops.add(t423, t425)

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t432 = paddle._C_ops.relu(t431)
        del t431

        # pd_op.pool2d: (128x512x1x1xf32) <- (128x512x14x14xf32, 2xi64)
        t433 = paddle._C_ops.pool2d(
            t432,
            t375,
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

        # pd_op.conv2d: (128x128x1x1xf32) <- (128x512x1x1xf32, 128x512x1x1xf32)
        t434 = paddle._C_ops.conv2d(
            t433, t94, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t94

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t435 = paddle._C_ops.reshape(t95, t384)
        del t95

        # pd_op.add: (128x128x1x1xf32) <- (128x128x1x1xf32, 1x128x1x1xf32)
        t436 = paddle._C_ops.add(t434, t435)

        # pd_op.relu: (128x128x1x1xf32) <- (128x128x1x1xf32)
        t437 = paddle._C_ops.relu(t436)
        del t436

        # pd_op.conv2d: (128x512x1x1xf32) <- (128x128x1x1xf32, 512x128x1x1xf32)
        t438 = paddle._C_ops.conv2d(
            t437, t96, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t96

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t439 = paddle._C_ops.reshape(t97, t384)
        del t97

        # pd_op.add: (128x512x1x1xf32) <- (128x512x1x1xf32, 1x512x1x1xf32)
        t440 = paddle._C_ops.add(t438, t439)

        # pd_op.sigmoid: (128x512x1x1xf32) <- (128x512x1x1xf32)
        t441 = paddle._C_ops.sigmoid(t440)
        del t440

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x1x1xf32)
        t442 = paddle._C_ops.multiply(t432, t441)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x512x14x14xf32, 256x512x1x1xf32)
        t443 = paddle._C_ops.conv2d(
            t442, t98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t98

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t444, t445, t446, t447, t448, t449 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t443,
                t99,
                t100,
                t101,
                t102,
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
        del t102, t101, t100, t99

        # pd_op.relu: (128x256x14x14xf32) <- (128x256x14x14xf32)
        t450 = paddle._C_ops.relu(t444)
        del t444

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x256x14x14xf32, 512x256x1x1xf32)
        t451 = paddle._C_ops.conv2d(
            t450, t103, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t103

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t452, t453, t454, t455, t456, t457 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t451,
                t104,
                t105,
                t106,
                t107,
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
        del t107, t106, t105, t104

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t458 = paddle._C_ops.relu(t452)
        del t452

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x5x5xf32)
        t459 = paddle._C_ops.depthwise_conv2d(
            t458, t108, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t108

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t460, t461, t462, t463, t464, t465 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t459,
                t109,
                t110,
                t111,
                t112,
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
        del t112, t111, t110, t109

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x3x3xf32)
        t466 = paddle._C_ops.depthwise_conv2d(
            t458, t113, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t113

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t467, t468, t469, t470, t471, t472 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t466,
                t114,
                t115,
                t116,
                t117,
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
        del t117, t116, t115, t114

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t473 = paddle._C_ops.add(t460, t467)

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x1x1xf32)
        t474 = paddle._C_ops.depthwise_conv2d(
            t458, t118, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t118

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t475, t476, t477, t478, t479, t480 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t474,
                t119,
                t120,
                t121,
                t122,
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
        del t122, t121, t120, t119

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t481 = paddle._C_ops.add(t473, t475)

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t482 = paddle._C_ops.relu(t481)
        del t481

        # pd_op.pool2d: (128x512x1x1xf32) <- (128x512x14x14xf32, 2xi64)
        t483 = paddle._C_ops.pool2d(
            t482,
            t375,
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

        # pd_op.conv2d: (128x128x1x1xf32) <- (128x512x1x1xf32, 128x512x1x1xf32)
        t484 = paddle._C_ops.conv2d(
            t483, t123, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t123

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t485 = paddle._C_ops.reshape(t124, t384)
        del t124

        # pd_op.add: (128x128x1x1xf32) <- (128x128x1x1xf32, 1x128x1x1xf32)
        t486 = paddle._C_ops.add(t484, t485)

        # pd_op.relu: (128x128x1x1xf32) <- (128x128x1x1xf32)
        t487 = paddle._C_ops.relu(t486)
        del t486

        # pd_op.conv2d: (128x512x1x1xf32) <- (128x128x1x1xf32, 512x128x1x1xf32)
        t488 = paddle._C_ops.conv2d(
            t487, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t489 = paddle._C_ops.reshape(t126, t384)
        del t126

        # pd_op.add: (128x512x1x1xf32) <- (128x512x1x1xf32, 1x512x1x1xf32)
        t490 = paddle._C_ops.add(t488, t489)

        # pd_op.sigmoid: (128x512x1x1xf32) <- (128x512x1x1xf32)
        t491 = paddle._C_ops.sigmoid(t490)
        del t490

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x1x1xf32)
        t492 = paddle._C_ops.multiply(t482, t491)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x512x14x14xf32, 256x512x1x1xf32)
        t493 = paddle._C_ops.conv2d(
            t492, t127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t127

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t494, t495, t496, t497, t498, t499 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t493,
                t128,
                t129,
                t130,
                t131,
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
        del t131, t130, t129, t128

        # pd_op.relu: (128x256x14x14xf32) <- (128x256x14x14xf32)
        t500 = paddle._C_ops.relu(t494)
        del t494

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x256x14x14xf32, 512x256x1x1xf32)
        t501 = paddle._C_ops.conv2d(
            t500, t132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t132

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t502, t503, t504, t505, t506, t507 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t501,
                t133,
                t134,
                t135,
                t136,
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
        del t136, t135, t134, t133

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t508 = paddle._C_ops.relu(t502)
        del t502

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x5x5xf32)
        t509 = paddle._C_ops.depthwise_conv2d(
            t508, t137, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t137

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t510, t511, t512, t513, t514, t515 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t509,
                t138,
                t139,
                t140,
                t141,
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
        del t141, t140, t139, t138

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x3x3xf32)
        t516 = paddle._C_ops.depthwise_conv2d(
            t508, t142, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t142

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t517, t518, t519, t520, t521, t522 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t516,
                t143,
                t144,
                t145,
                t146,
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
        del t146, t145, t144, t143

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t523 = paddle._C_ops.add(t510, t517)

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x1x1xf32)
        t524 = paddle._C_ops.depthwise_conv2d(
            t508, t147, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t147

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t525, t526, t527, t528, t529, t530 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t524,
                t148,
                t149,
                t150,
                t151,
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
        del t151, t150, t149, t148

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t531 = paddle._C_ops.add(t523, t525)

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t532 = paddle._C_ops.relu(t531)
        del t531

        # pd_op.pool2d: (128x512x1x1xf32) <- (128x512x14x14xf32, 2xi64)
        t533 = paddle._C_ops.pool2d(
            t532,
            t375,
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

        # pd_op.conv2d: (128x128x1x1xf32) <- (128x512x1x1xf32, 128x512x1x1xf32)
        t534 = paddle._C_ops.conv2d(
            t533, t152, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t152

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t535 = paddle._C_ops.reshape(t153, t384)
        del t153

        # pd_op.add: (128x128x1x1xf32) <- (128x128x1x1xf32, 1x128x1x1xf32)
        t536 = paddle._C_ops.add(t534, t535)

        # pd_op.relu: (128x128x1x1xf32) <- (128x128x1x1xf32)
        t537 = paddle._C_ops.relu(t536)
        del t536

        # pd_op.conv2d: (128x512x1x1xf32) <- (128x128x1x1xf32, 512x128x1x1xf32)
        t538 = paddle._C_ops.conv2d(
            t537, t154, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t154

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t539 = paddle._C_ops.reshape(t155, t384)
        del t155

        # pd_op.add: (128x512x1x1xf32) <- (128x512x1x1xf32, 1x512x1x1xf32)
        t540 = paddle._C_ops.add(t538, t539)

        # pd_op.sigmoid: (128x512x1x1xf32) <- (128x512x1x1xf32)
        t541 = paddle._C_ops.sigmoid(t540)
        del t540

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x1x1xf32)
        t542 = paddle._C_ops.multiply(t532, t541)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x512x14x14xf32, 256x512x1x1xf32)
        t543 = paddle._C_ops.conv2d(
            t542, t156, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t544, t545, t546, t547, t548, t549 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t543,
                t157,
                t158,
                t159,
                t160,
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
        del t160, t159, t158, t157

        # pd_op.relu: (128x256x14x14xf32) <- (128x256x14x14xf32)
        t550 = paddle._C_ops.relu(t544)
        del t544

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x256x14x14xf32, 512x256x1x1xf32)
        t551 = paddle._C_ops.conv2d(
            t550, t161, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t161

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t552, t553, t554, t555, t556, t557 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t551,
                t162,
                t163,
                t164,
                t165,
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
        del t165, t164, t163, t162

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t558 = paddle._C_ops.relu(t552)
        del t552

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x5x5xf32)
        t559 = paddle._C_ops.depthwise_conv2d(
            t558, t166, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t166

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t560, t561, t562, t563, t564, t565 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t559,
                t167,
                t168,
                t169,
                t170,
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
        del t170, t169, t168, t167

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x3x3xf32)
        t566 = paddle._C_ops.depthwise_conv2d(
            t558, t171, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t171

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t567, t568, t569, t570, t571, t572 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t566,
                t172,
                t173,
                t174,
                t175,
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
        del t175, t174, t173, t172

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t573 = paddle._C_ops.add(t560, t567)

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x1x1xf32)
        t574 = paddle._C_ops.depthwise_conv2d(
            t558, t176, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t176

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t575, t576, t577, t578, t579, t580 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t574,
                t177,
                t178,
                t179,
                t180,
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
        del t180, t179, t178, t177

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t581 = paddle._C_ops.add(t573, t575)

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t582 = paddle._C_ops.relu(t581)
        del t581

        # pd_op.pool2d: (128x512x1x1xf32) <- (128x512x14x14xf32, 2xi64)
        t583 = paddle._C_ops.pool2d(
            t582,
            t375,
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

        # pd_op.conv2d: (128x128x1x1xf32) <- (128x512x1x1xf32, 128x512x1x1xf32)
        t584 = paddle._C_ops.conv2d(
            t583, t181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t181

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t585 = paddle._C_ops.reshape(t182, t384)
        del t182

        # pd_op.add: (128x128x1x1xf32) <- (128x128x1x1xf32, 1x128x1x1xf32)
        t586 = paddle._C_ops.add(t584, t585)

        # pd_op.relu: (128x128x1x1xf32) <- (128x128x1x1xf32)
        t587 = paddle._C_ops.relu(t586)
        del t586

        # pd_op.conv2d: (128x512x1x1xf32) <- (128x128x1x1xf32, 512x128x1x1xf32)
        t588 = paddle._C_ops.conv2d(
            t587, t183, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t183

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t589 = paddle._C_ops.reshape(t184, t384)
        del t184

        # pd_op.add: (128x512x1x1xf32) <- (128x512x1x1xf32, 1x512x1x1xf32)
        t590 = paddle._C_ops.add(t588, t589)

        # pd_op.sigmoid: (128x512x1x1xf32) <- (128x512x1x1xf32)
        t591 = paddle._C_ops.sigmoid(t590)
        del t590

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x1x1xf32)
        t592 = paddle._C_ops.multiply(t582, t591)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x512x14x14xf32, 256x512x1x1xf32)
        t593 = paddle._C_ops.conv2d(
            t592, t185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t594, t595, t596, t597, t598, t599 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t593,
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

        # pd_op.relu: (128x256x14x14xf32) <- (128x256x14x14xf32)
        t600 = paddle._C_ops.relu(t594)
        del t594

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x256x14x14xf32, 512x256x1x1xf32)
        t601 = paddle._C_ops.conv2d(
            t600, t190, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t602, t603, t604, t605, t606, t607 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t601,
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

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t608 = paddle._C_ops.relu(t602)
        del t602

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x5x5xf32)
        t609 = paddle._C_ops.depthwise_conv2d(
            t608, t195, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t195

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t610, t611, t612, t613, t614, t615 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t609,
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

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x3x3xf32)
        t616 = paddle._C_ops.depthwise_conv2d(
            t608, t200, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t617, t618, t619, t620, t621, t622 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t616,
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

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t623 = paddle._C_ops.add(t610, t617)

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x1x1xf32)
        t624 = paddle._C_ops.depthwise_conv2d(
            t608, t205, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t625, t626, t627, t628, t629, t630 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t624,
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

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t631 = paddle._C_ops.add(t623, t625)

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t632 = paddle._C_ops.relu(t631)
        del t631

        # pd_op.pool2d: (128x512x1x1xf32) <- (128x512x14x14xf32, 2xi64)
        t633 = paddle._C_ops.pool2d(
            t632,
            t375,
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

        # pd_op.conv2d: (128x128x1x1xf32) <- (128x512x1x1xf32, 128x512x1x1xf32)
        t634 = paddle._C_ops.conv2d(
            t633, t210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t635 = paddle._C_ops.reshape(t211, t384)
        del t211

        # pd_op.add: (128x128x1x1xf32) <- (128x128x1x1xf32, 1x128x1x1xf32)
        t636 = paddle._C_ops.add(t634, t635)

        # pd_op.relu: (128x128x1x1xf32) <- (128x128x1x1xf32)
        t637 = paddle._C_ops.relu(t636)
        del t636

        # pd_op.conv2d: (128x512x1x1xf32) <- (128x128x1x1xf32, 512x128x1x1xf32)
        t638 = paddle._C_ops.conv2d(
            t637, t212, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t212

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t639 = paddle._C_ops.reshape(t213, t384)
        del t213

        # pd_op.add: (128x512x1x1xf32) <- (128x512x1x1xf32, 1x512x1x1xf32)
        t640 = paddle._C_ops.add(t638, t639)

        # pd_op.sigmoid: (128x512x1x1xf32) <- (128x512x1x1xf32)
        t641 = paddle._C_ops.sigmoid(t640)
        del t640

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x1x1xf32)
        t642 = paddle._C_ops.multiply(t632, t641)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x512x14x14xf32, 256x512x1x1xf32)
        t643 = paddle._C_ops.conv2d(
            t642, t214, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t214

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t644, t645, t646, t647, t648, t649 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t643,
                t215,
                t216,
                t217,
                t218,
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
        del t218, t217, t216, t215

        # pd_op.relu: (128x256x14x14xf32) <- (128x256x14x14xf32)
        t650 = paddle._C_ops.relu(t644)
        del t644

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x256x14x14xf32, 512x256x1x1xf32)
        t651 = paddle._C_ops.conv2d(
            t650, t219, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t219

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t652, t653, t654, t655, t656, t657 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t651,
                t220,
                t221,
                t222,
                t223,
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
        del t223, t222, t221, t220

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t658 = paddle._C_ops.relu(t652)
        del t652

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x5x5xf32)
        t659 = paddle._C_ops.depthwise_conv2d(
            t658, t224, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t224

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t660, t661, t662, t663, t664, t665 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t659,
                t225,
                t226,
                t227,
                t228,
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
        del t228, t227, t226, t225

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x3x3xf32)
        t666 = paddle._C_ops.depthwise_conv2d(
            t658, t229, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t229

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t667, t668, t669, t670, t671, t672 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t666,
                t230,
                t231,
                t232,
                t233,
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
        del t233, t232, t231, t230

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t673 = paddle._C_ops.add(t660, t667)

        # pd_op.depthwise_conv2d: (128x512x14x14xf32) <- (128x512x14x14xf32, 512x1x1x1xf32)
        t674 = paddle._C_ops.depthwise_conv2d(
            t658, t234, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t234

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t675, t676, t677, t678, t679, t680 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t674,
                t235,
                t236,
                t237,
                t238,
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
        del t238, t237, t236, t235

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t681 = paddle._C_ops.add(t673, t675)

        # pd_op.relu: (128x512x14x14xf32) <- (128x512x14x14xf32)
        t682 = paddle._C_ops.relu(t681)
        del t681

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t683 = paddle._C_ops.conv2d(
            t682, t239, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t239

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t684, t685, t686, t687, t688, t689 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t683,
                t240,
                t241,
                t242,
                t243,
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
        del t243, t242, t241, t240

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t690 = paddle._C_ops.relu(t684)
        del t684

        # pd_op.depthwise_conv2d: (128x1024x14x14xf32) <- (128x1024x14x14xf32, 1024x1x5x5xf32)
        t691 = paddle._C_ops.depthwise_conv2d(
            t690, t244, [1, 1], [2, 2], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del t244

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t692, t693, t694, t695, t696, t697 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t691,
                t245,
                t246,
                t247,
                t248,
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
        del t248, t247, t246, t245

        # pd_op.depthwise_conv2d: (128x1024x14x14xf32) <- (128x1024x14x14xf32, 1024x1x3x3xf32)
        t698 = paddle._C_ops.depthwise_conv2d(
            t690, t249, [1, 1], [1, 1], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del t249

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t699, t700, t701, t702, t703, t704 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t698,
                t250,
                t251,
                t252,
                t253,
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
        del t253, t252, t251, t250

        # pd_op.add: (128x1024x14x14xf32) <- (128x1024x14x14xf32, 128x1024x14x14xf32)
        t705 = paddle._C_ops.add(t692, t699)

        # pd_op.depthwise_conv2d: (128x1024x14x14xf32) <- (128x1024x14x14xf32, 1024x1x1x1xf32)
        t706 = paddle._C_ops.depthwise_conv2d(
            t690, t254, [1, 1], [0, 0], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del t254

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t707, t708, t709, t710, t711, t712 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t706,
                t255,
                t256,
                t257,
                t258,
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
        del t258, t257, t256, t255

        # pd_op.add: (128x1024x14x14xf32) <- (128x1024x14x14xf32, 128x1024x14x14xf32)
        t713 = paddle._C_ops.add(t705, t707)

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t714 = paddle._C_ops.relu(t713)
        del t713

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x1024x14x14xf32, 1024x1024x1x1xf32)
        t715 = paddle._C_ops.conv2d(
            t714, t259, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t259

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t716, t717, t718, t719, t720, t721 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t715,
                t260,
                t261,
                t262,
                t263,
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
        del t260, t263, t262, t261

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t722 = paddle._C_ops.relu(t716)
        del t716

        # pd_op.add: (128x1024x14x14xf32) <- (128x1024x14x14xf32, 128x1024x14x14xf32)
        t723 = paddle._C_ops.add(t722, t690)

        # pd_op.pool2d: (128x1024x1x1xf32) <- (128x1024x14x14xf32, 2xi64)
        t724 = paddle._C_ops.pool2d(
            t723,
            t375,
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

        # pd_op.conv2d: (128x512x1x1xf32) <- (128x1024x1x1xf32, 512x1024x1x1xf32)
        t725 = paddle._C_ops.conv2d(
            t724, t264, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t264

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t726 = paddle._C_ops.reshape(t265, t384)
        del t384, t265

        # pd_op.add: (128x512x1x1xf32) <- (128x512x1x1xf32, 1x512x1x1xf32)
        t727 = paddle._C_ops.add(t725, t726)

        # pd_op.full: (1xf32) <- ()
        t728 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (128x512x1x1xf32, 128x512x1x1xui8) <- (128x512x1x1xf32, None, 1xf32)
        t729, t730 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                t727, None, t728, False, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t727

        # pd_op.flatten: (128x512xf32) <- (128x512x1x1xf32)
        t731 = paddle._C_ops.flatten(t729, 1, 3)

        # pd_op.flatten: (128x512xf32) <- (128x512xf32)
        t732 = paddle._C_ops.flatten(t731, 1, 1)

        # pd_op.batch_norm_: (128x512xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t733, t734, t735, t736, t737, t738 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t732,
                t266,
                t267,
                t268,
                t269,
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
        del t269, t268, t267, t266

        return (
            t271,
            t273,
            t274,
            t275,
            t276,
            t277,
            t278,
            t279,
            t281,
            t282,
            t283,
            t284,
            t285,
            t286,
            t287,
            t289,
            t290,
            t291,
            t292,
            t293,
            t294,
            t295,
            t297,
            t298,
            t299,
            t300,
            t301,
            t302,
            t303,
            t305,
            t306,
            t307,
            t308,
            t309,
            t310,
            t311,
            t313,
            t314,
            t315,
            t316,
            t317,
            t318,
            t319,
            t321,
            t322,
            t323,
            t324,
            t325,
            t326,
            t327,
            t329,
            t330,
            t331,
            t332,
            t333,
            t334,
            t335,
            t337,
            t338,
            t339,
            t340,
            t341,
            t342,
            t343,
            t345,
            t346,
            t347,
            t348,
            t349,
            t350,
            t351,
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
            t371,
            t372,
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
            t385,
            t387,
            t388,
            t389,
            t391,
            t392,
            t393,
            t395,
            t396,
            t397,
            t398,
            t399,
            t400,
            t401,
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
            t416,
            t417,
            t418,
            t419,
            t420,
            t421,
            t422,
            t423,
            t424,
            t425,
            t426,
            t427,
            t428,
            t429,
            t430,
            t432,
            t433,
            t434,
            t435,
            t437,
            t438,
            t439,
            t441,
            t442,
            t443,
            t445,
            t446,
            t447,
            t448,
            t449,
            t450,
            t451,
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
            t463,
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
            t477,
            t478,
            t479,
            t480,
            t482,
            t483,
            t484,
            t485,
            t487,
            t488,
            t489,
            t491,
            t492,
            t493,
            t495,
            t496,
            t497,
            t498,
            t499,
            t500,
            t501,
            t503,
            t504,
            t505,
            t506,
            t507,
            t508,
            t509,
            t510,
            t511,
            t512,
            t513,
            t514,
            t515,
            t516,
            t517,
            t518,
            t519,
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
            t532,
            t533,
            t534,
            t535,
            t537,
            t538,
            t539,
            t541,
            t542,
            t543,
            t545,
            t546,
            t547,
            t548,
            t549,
            t550,
            t551,
            t553,
            t554,
            t555,
            t556,
            t557,
            t558,
            t559,
            t560,
            t561,
            t562,
            t563,
            t564,
            t565,
            t566,
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
            t582,
            t583,
            t584,
            t585,
            t587,
            t588,
            t589,
            t591,
            t592,
            t593,
            t595,
            t596,
            t597,
            t598,
            t599,
            t600,
            t601,
            t603,
            t604,
            t605,
            t606,
            t607,
            t608,
            t609,
            t610,
            t611,
            t612,
            t613,
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
            t629,
            t630,
            t632,
            t633,
            t634,
            t635,
            t637,
            t638,
            t639,
            t641,
            t642,
            t643,
            t645,
            t646,
            t647,
            t648,
            t649,
            t650,
            t651,
            t653,
            t654,
            t655,
            t656,
            t657,
            t658,
            t659,
            t660,
            t661,
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
            t676,
            t677,
            t678,
            t679,
            t680,
            t682,
            t683,
            t685,
            t686,
            t687,
            t688,
            t689,
            t690,
            t691,
            t692,
            t693,
            t694,
            t695,
            t696,
            t697,
            t698,
            t699,
            t700,
            t701,
            t702,
            t703,
            t704,
            t705,
            t706,
            t707,
            t708,
            t709,
            t710,
            t711,
            t712,
            t714,
            t715,
            t717,
            t718,
            t719,
            t720,
            t721,
            t722,
            t723,
            t724,
            t725,
            t726,
            t728,
            t729,
            t730,
            t731,
            t732,
            t733,
            t734,
            t735,
            t736,
            t737,
            t738,
        )
