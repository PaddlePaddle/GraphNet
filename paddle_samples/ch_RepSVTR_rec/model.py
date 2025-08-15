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
        t270: paddle.Tensor,
        t271: paddle.Tensor,
        t272: paddle.Tensor,
        t273: paddle.Tensor,
        t274: paddle.Tensor,
        t275: paddle.Tensor,
        t276: paddle.Tensor,
        t277: paddle.Tensor,
        t278: paddle.Tensor,
        t279: paddle.Tensor,
        t280: paddle.Tensor,
        t281: paddle.Tensor,
        t282: paddle.Tensor,
        t283: paddle.Tensor,
        t284: paddle.Tensor,
        t285: paddle.Tensor,
        t286: paddle.Tensor,
        t287: paddle.Tensor,
        t288: paddle.Tensor,
        t289: paddle.Tensor,
        t290: paddle.Tensor,
        t291: paddle.Tensor,
        t292: paddle.Tensor,
        t293: paddle.Tensor,
        t294: paddle.Tensor,
        t295: paddle.Tensor,
        t296: paddle.Tensor,
        t297: paddle.Tensor,
        t298: paddle.Tensor,
        t299: paddle.Tensor,
        t300: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x48x24x160xf32) <- (-1x3x48x320xf32, 48x3x3x3xf32)
        t301 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x48x24x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x24x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t302, t303, t304, t305, t306, t307 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t301,
                t1,
                t2,
                t3,
                t4,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t301, t4, t3, t2, t1

        # pd_op.gelu: (-1x48x24x160xf32) <- (-1x48x24x160xf32)
        t308 = paddle._C_ops.gelu(t302, False)
        del t302

        # pd_op.conv2d: (-1x96x12x80xf32) <- (-1x48x24x160xf32, 96x48x3x3xf32)
        t309 = paddle._C_ops.conv2d(
            t308, t5, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t308, t5

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t310, t311, t312, t313, t314, t315 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t309,
                t6,
                t7,
                t8,
                t9,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t309, t9, t8, t7, t6

        # pd_op.depthwise_conv2d: (-1x96x12x80xf32) <- (-1x96x12x80xf32, 96x1x3x3xf32)
        t316 = paddle._C_ops.depthwise_conv2d(
            t310, t10, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t317, t318, t319, t320, t321, t322 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t316,
                t11,
                t12,
                t13,
                t14,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t316, t14, t13, t12, t11

        # pd_op.depthwise_conv2d: (-1x96x12x80xf32) <- (-1x96x12x80xf32, 96x1x1x1xf32)
        t323 = paddle._C_ops.depthwise_conv2d(
            t310, t15, [1, 1], [0, 0], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t15

        # pd_op.full_int_array: (4xi64) <- ()
        t324 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t325 = paddle._C_ops.reshape(t16, t324)
        del t16

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, 1x96x1x1xf32)
        t326 = paddle._C_ops.add(t323, t325)
        del t323, t325

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x12x80xf32)
        t327 = paddle._C_ops.add(t317, t326)
        del t326, t317

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x12x80xf32)
        t328 = paddle._C_ops.add(t327, t310)
        del t327, t310

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t329, t330, t331, t332, t333, t334 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t328,
                t17,
                t18,
                t19,
                t20,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t328, t20, t19, t18, t17

        # pd_op.mean: (-1x96x1x1xf32) <- (-1x96x12x80xf32)
        t335 = paddle._C_ops.mean(t329, [2, 3], True)

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        t336 = paddle._C_ops.conv2d(
            t335, t21, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t335, t21

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t337 = paddle._C_ops.reshape(t22, t324)
        del t22

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        t338 = paddle._C_ops.add(t336, t337)
        del t336, t337

        # pd_op.relu: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        t339 = paddle._C_ops.relu(t338)
        del t338

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        t340 = paddle._C_ops.conv2d(
            t339, t23, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t23, t339

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t341 = paddle._C_ops.reshape(t24, t324)
        del t24

        # pd_op.add: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        t342 = paddle._C_ops.add(t340, t341)
        del t340, t341

        # pd_op.sigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        t343 = paddle._C_ops.sigmoid(t342)
        del t342

        # pd_op.multiply: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x1x1xf32)
        t344 = paddle._C_ops.multiply(t329, t343)
        del t329, t343

        # pd_op.conv2d: (-1x192x12x80xf32) <- (-1x96x12x80xf32, 192x96x1x1xf32)
        t345 = paddle._C_ops.conv2d(
            t344, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x192x12x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x12x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t346, t347, t348, t349, t350, t351 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t345,
                t26,
                t27,
                t28,
                t29,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t345, t29, t28, t27, t26

        # pd_op.gelu: (-1x192x12x80xf32) <- (-1x192x12x80xf32)
        t352 = paddle._C_ops.gelu(t346, False)
        del t346

        # pd_op.conv2d: (-1x96x12x80xf32) <- (-1x192x12x80xf32, 96x192x1x1xf32)
        t353 = paddle._C_ops.conv2d(
            t352, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t352, t30

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t354, t355, t356, t357, t358, t359 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t353,
                t31,
                t32,
                t33,
                t34,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t353, t34, t33, t32, t31

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x12x80xf32)
        t360 = paddle._C_ops.add(t344, t354)
        del t354, t344

        # pd_op.depthwise_conv2d: (-1x96x12x80xf32) <- (-1x96x12x80xf32, 96x1x3x3xf32)
        t361 = paddle._C_ops.depthwise_conv2d(
            t360, t35, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t362, t363, t364, t365, t366, t367 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t361,
                t36,
                t37,
                t38,
                t39,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t361, t39, t38, t37, t36

        # pd_op.depthwise_conv2d: (-1x96x12x80xf32) <- (-1x96x12x80xf32, 96x1x1x1xf32)
        t368 = paddle._C_ops.depthwise_conv2d(
            t360, t40, [1, 1], [0, 0], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t40

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t369 = paddle._C_ops.reshape(t41, t324)
        del t41

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, 1x96x1x1xf32)
        t370 = paddle._C_ops.add(t368, t369)
        del t368, t369

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x12x80xf32)
        t371 = paddle._C_ops.add(t362, t370)
        del t370, t362

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x12x80xf32)
        t372 = paddle._C_ops.add(t371, t360)
        del t360, t371

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t373, t374, t375, t376, t377, t378 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t372,
                t42,
                t43,
                t44,
                t45,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t372, t45, t44, t43, t42

        # pd_op.conv2d: (-1x192x12x80xf32) <- (-1x96x12x80xf32, 192x96x1x1xf32)
        t379 = paddle._C_ops.conv2d(
            t373, t46, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t46

        # pd_op.batch_norm_: (-1x192x12x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x12x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t380, t381, t382, t383, t384, t385 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t379,
                t47,
                t48,
                t49,
                t50,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t379, t50, t49, t48, t47

        # pd_op.gelu: (-1x192x12x80xf32) <- (-1x192x12x80xf32)
        t386 = paddle._C_ops.gelu(t380, False)
        del t380

        # pd_op.conv2d: (-1x96x12x80xf32) <- (-1x192x12x80xf32, 96x192x1x1xf32)
        t387 = paddle._C_ops.conv2d(
            t386, t51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t386, t51

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t388, t389, t390, t391, t392, t393 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t387,
                t52,
                t53,
                t54,
                t55,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t387, t55, t54, t53, t52

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x12x80xf32)
        t394 = paddle._C_ops.add(t373, t388)
        del t373, t388

        # pd_op.depthwise_conv2d: (-1x96x12x80xf32) <- (-1x96x12x80xf32, 96x1x3x3xf32)
        t395 = paddle._C_ops.depthwise_conv2d(
            t394, t56, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t56

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t396, t397, t398, t399, t400, t401 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t395,
                t57,
                t58,
                t59,
                t60,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t395, t60, t59, t58, t57

        # pd_op.depthwise_conv2d: (-1x96x12x80xf32) <- (-1x96x12x80xf32, 96x1x1x1xf32)
        t402 = paddle._C_ops.depthwise_conv2d(
            t394, t61, [1, 1], [0, 0], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t61

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t403 = paddle._C_ops.reshape(t62, t324)
        del t62

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, 1x96x1x1xf32)
        t404 = paddle._C_ops.add(t402, t403)
        del t402, t403

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x12x80xf32)
        t405 = paddle._C_ops.add(t396, t404)
        del t404, t396

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x12x80xf32)
        t406 = paddle._C_ops.add(t405, t394)
        del t405, t394

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t407, t408, t409, t410, t411, t412 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t406,
                t63,
                t64,
                t65,
                t66,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t406, t66, t65, t64, t63

        # pd_op.conv2d: (-1x192x12x80xf32) <- (-1x96x12x80xf32, 192x96x1x1xf32)
        t413 = paddle._C_ops.conv2d(
            t407, t67, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t67

        # pd_op.batch_norm_: (-1x192x12x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x12x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t414, t415, t416, t417, t418, t419 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t413,
                t68,
                t69,
                t70,
                t71,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t413, t71, t70, t69, t68

        # pd_op.gelu: (-1x192x12x80xf32) <- (-1x192x12x80xf32)
        t420 = paddle._C_ops.gelu(t414, False)
        del t414

        # pd_op.conv2d: (-1x96x12x80xf32) <- (-1x192x12x80xf32, 96x192x1x1xf32)
        t421 = paddle._C_ops.conv2d(
            t420, t72, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t420, t72

        # pd_op.batch_norm_: (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x12x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t422, t423, t424, t425, t426, t427 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t421,
                t73,
                t74,
                t75,
                t76,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t421, t76, t75, t74, t73

        # pd_op.add: (-1x96x12x80xf32) <- (-1x96x12x80xf32, -1x96x12x80xf32)
        t428 = paddle._C_ops.add(t407, t422)
        del t407, t422

        # pd_op.depthwise_conv2d: (-1x96x6x80xf32) <- (-1x96x12x80xf32, 96x1x3x3xf32)
        t429 = paddle._C_ops.depthwise_conv2d(
            t428, t77, [2, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t428, t77

        # pd_op.batch_norm_: (-1x96x6x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x6x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t430, t431, t432, t433, t434, t435 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t429,
                t78,
                t79,
                t80,
                t81,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t429, t81, t80, t79, t78

        # pd_op.conv2d: (-1x192x6x80xf32) <- (-1x96x6x80xf32, 192x96x1x1xf32)
        t436 = paddle._C_ops.conv2d(
            t430, t82, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t430, t82

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t437, t438, t439, t440, t441, t442 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t436,
                t83,
                t84,
                t85,
                t86,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t436, t86, t85, t84, t83

        # pd_op.conv2d: (-1x384x6x80xf32) <- (-1x192x6x80xf32, 384x192x1x1xf32)
        t443 = paddle._C_ops.conv2d(
            t437, t87, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t87

        # pd_op.batch_norm_: (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t444, t445, t446, t447, t448, t449 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t443,
                t88,
                t89,
                t90,
                t91,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t443, t91, t90, t89, t88

        # pd_op.gelu: (-1x384x6x80xf32) <- (-1x384x6x80xf32)
        t450 = paddle._C_ops.gelu(t444, False)
        del t444

        # pd_op.conv2d: (-1x192x6x80xf32) <- (-1x384x6x80xf32, 192x384x1x1xf32)
        t451 = paddle._C_ops.conv2d(
            t450, t92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t450, t92

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t452, t453, t454, t455, t456, t457 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t451,
                t93,
                t94,
                t95,
                t96,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t451, t96, t95, t94, t93

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t458 = paddle._C_ops.add(t437, t452)
        del t452, t437

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x3x3xf32)
        t459 = paddle._C_ops.depthwise_conv2d(
            t458, t97, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t97

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t460, t461, t462, t463, t464, t465 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t459,
                t98,
                t99,
                t100,
                t101,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t459, t101, t100, t99, t98

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x1x1xf32)
        t466 = paddle._C_ops.depthwise_conv2d(
            t458, t102, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t102

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t467 = paddle._C_ops.reshape(t103, t324)
        del t103

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 1x192x1x1xf32)
        t468 = paddle._C_ops.add(t466, t467)
        del t466, t467

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t469 = paddle._C_ops.add(t460, t468)
        del t468, t460

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t470 = paddle._C_ops.add(t469, t458)
        del t458, t469

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t471, t472, t473, t474, t475, t476 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t470,
                t104,
                t105,
                t106,
                t107,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t470, t107, t106, t105, t104

        # pd_op.mean: (-1x192x1x1xf32) <- (-1x192x6x80xf32)
        t477 = paddle._C_ops.mean(t471, [2, 3], True)

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x192x1x1xf32, 48x192x1x1xf32)
        t478 = paddle._C_ops.conv2d(
            t477, t108, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t477, t108

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t479 = paddle._C_ops.reshape(t109, t324)
        del t109

        # pd_op.add: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        t480 = paddle._C_ops.add(t478, t479)
        del t478, t479

        # pd_op.relu: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        t481 = paddle._C_ops.relu(t480)
        del t480

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x48x1x1xf32, 192x48x1x1xf32)
        t482 = paddle._C_ops.conv2d(
            t481, t110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110, t481

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t483 = paddle._C_ops.reshape(t111, t324)
        del t111

        # pd_op.add: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        t484 = paddle._C_ops.add(t482, t483)
        del t482, t483

        # pd_op.sigmoid: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        t485 = paddle._C_ops.sigmoid(t484)
        del t484

        # pd_op.multiply: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x1x1xf32)
        t486 = paddle._C_ops.multiply(t471, t485)
        del t471, t485

        # pd_op.conv2d: (-1x384x6x80xf32) <- (-1x192x6x80xf32, 384x192x1x1xf32)
        t487 = paddle._C_ops.conv2d(
            t486, t112, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t112

        # pd_op.batch_norm_: (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t488, t489, t490, t491, t492, t493 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t487,
                t113,
                t114,
                t115,
                t116,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t487, t116, t115, t114, t113

        # pd_op.gelu: (-1x384x6x80xf32) <- (-1x384x6x80xf32)
        t494 = paddle._C_ops.gelu(t488, False)
        del t488

        # pd_op.conv2d: (-1x192x6x80xf32) <- (-1x384x6x80xf32, 192x384x1x1xf32)
        t495 = paddle._C_ops.conv2d(
            t494, t117, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t494, t117

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t496, t497, t498, t499, t500, t501 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t495,
                t118,
                t119,
                t120,
                t121,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t495, t121, t120, t119, t118

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t502 = paddle._C_ops.add(t486, t496)
        del t496, t486

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x3x3xf32)
        t503 = paddle._C_ops.depthwise_conv2d(
            t502, t122, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t122

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t504, t505, t506, t507, t508, t509 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t503,
                t123,
                t124,
                t125,
                t126,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t503, t126, t125, t124, t123

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x1x1xf32)
        t510 = paddle._C_ops.depthwise_conv2d(
            t502, t127, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t127

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t511 = paddle._C_ops.reshape(t128, t324)
        del t128

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 1x192x1x1xf32)
        t512 = paddle._C_ops.add(t510, t511)
        del t510, t511

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t513 = paddle._C_ops.add(t504, t512)
        del t512, t504

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t514 = paddle._C_ops.add(t513, t502)
        del t502, t513

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t515, t516, t517, t518, t519, t520 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t514,
                t129,
                t130,
                t131,
                t132,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t514, t132, t131, t130, t129

        # pd_op.conv2d: (-1x384x6x80xf32) <- (-1x192x6x80xf32, 384x192x1x1xf32)
        t521 = paddle._C_ops.conv2d(
            t515, t133, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t133

        # pd_op.batch_norm_: (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t522, t523, t524, t525, t526, t527 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t521,
                t134,
                t135,
                t136,
                t137,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t521, t137, t136, t135, t134

        # pd_op.gelu: (-1x384x6x80xf32) <- (-1x384x6x80xf32)
        t528 = paddle._C_ops.gelu(t522, False)
        del t522

        # pd_op.conv2d: (-1x192x6x80xf32) <- (-1x384x6x80xf32, 192x384x1x1xf32)
        t529 = paddle._C_ops.conv2d(
            t528, t138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t528, t138

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t530, t531, t532, t533, t534, t535 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t529,
                t139,
                t140,
                t141,
                t142,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t529, t142, t141, t140, t139

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t536 = paddle._C_ops.add(t515, t530)
        del t515, t530

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x3x3xf32)
        t537 = paddle._C_ops.depthwise_conv2d(
            t536, t143, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t143

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t538, t539, t540, t541, t542, t543 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t537,
                t144,
                t145,
                t146,
                t147,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t537, t147, t146, t145, t144

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x1x1xf32)
        t544 = paddle._C_ops.depthwise_conv2d(
            t536, t148, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t148

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t545 = paddle._C_ops.reshape(t149, t324)
        del t149

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 1x192x1x1xf32)
        t546 = paddle._C_ops.add(t544, t545)
        del t544, t545

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t547 = paddle._C_ops.add(t538, t546)
        del t546, t538

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t548 = paddle._C_ops.add(t547, t536)
        del t536, t547

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t549, t550, t551, t552, t553, t554 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t548,
                t150,
                t151,
                t152,
                t153,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t548, t153, t152, t151, t150

        # pd_op.mean: (-1x192x1x1xf32) <- (-1x192x6x80xf32)
        t555 = paddle._C_ops.mean(t549, [2, 3], True)

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x192x1x1xf32, 48x192x1x1xf32)
        t556 = paddle._C_ops.conv2d(
            t555, t154, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t555, t154

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t557 = paddle._C_ops.reshape(t155, t324)
        del t155

        # pd_op.add: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        t558 = paddle._C_ops.add(t556, t557)
        del t556, t557

        # pd_op.relu: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        t559 = paddle._C_ops.relu(t558)
        del t558

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x48x1x1xf32, 192x48x1x1xf32)
        t560 = paddle._C_ops.conv2d(
            t559, t156, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156, t559

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t561 = paddle._C_ops.reshape(t157, t324)
        del t157

        # pd_op.add: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        t562 = paddle._C_ops.add(t560, t561)
        del t560, t561

        # pd_op.sigmoid: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        t563 = paddle._C_ops.sigmoid(t562)
        del t562

        # pd_op.multiply: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x1x1xf32)
        t564 = paddle._C_ops.multiply(t549, t563)
        del t549, t563

        # pd_op.conv2d: (-1x384x6x80xf32) <- (-1x192x6x80xf32, 384x192x1x1xf32)
        t565 = paddle._C_ops.conv2d(
            t564, t158, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t158

        # pd_op.batch_norm_: (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t566, t567, t568, t569, t570, t571 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t565,
                t159,
                t160,
                t161,
                t162,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t565, t162, t161, t160, t159

        # pd_op.gelu: (-1x384x6x80xf32) <- (-1x384x6x80xf32)
        t572 = paddle._C_ops.gelu(t566, False)
        del t566

        # pd_op.conv2d: (-1x192x6x80xf32) <- (-1x384x6x80xf32, 192x384x1x1xf32)
        t573 = paddle._C_ops.conv2d(
            t572, t163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t572, t163

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t574, t575, t576, t577, t578, t579 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t573,
                t164,
                t165,
                t166,
                t167,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t573, t167, t166, t165, t164

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t580 = paddle._C_ops.add(t564, t574)
        del t574, t564

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x3x3xf32)
        t581 = paddle._C_ops.depthwise_conv2d(
            t580, t168, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t168

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t582, t583, t584, t585, t586, t587 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t581,
                t169,
                t170,
                t171,
                t172,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t581, t172, t171, t170, t169

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x1x1xf32)
        t588 = paddle._C_ops.depthwise_conv2d(
            t580, t173, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t173

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t589 = paddle._C_ops.reshape(t174, t324)
        del t174

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 1x192x1x1xf32)
        t590 = paddle._C_ops.add(t588, t589)
        del t588, t589

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t591 = paddle._C_ops.add(t582, t590)
        del t590, t582

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t592 = paddle._C_ops.add(t591, t580)
        del t580, t591

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t593, t594, t595, t596, t597, t598 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t592,
                t175,
                t176,
                t177,
                t178,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t592, t178, t177, t176, t175

        # pd_op.conv2d: (-1x384x6x80xf32) <- (-1x192x6x80xf32, 384x192x1x1xf32)
        t599 = paddle._C_ops.conv2d(
            t593, t179, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t179

        # pd_op.batch_norm_: (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t600, t601, t602, t603, t604, t605 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t599,
                t180,
                t181,
                t182,
                t183,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t599, t183, t182, t181, t180

        # pd_op.gelu: (-1x384x6x80xf32) <- (-1x384x6x80xf32)
        t606 = paddle._C_ops.gelu(t600, False)
        del t600

        # pd_op.conv2d: (-1x192x6x80xf32) <- (-1x384x6x80xf32, 192x384x1x1xf32)
        t607 = paddle._C_ops.conv2d(
            t606, t184, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t606, t184

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t608, t609, t610, t611, t612, t613 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t607,
                t185,
                t186,
                t187,
                t188,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t607, t188, t187, t186, t185

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t614 = paddle._C_ops.add(t593, t608)
        del t593, t608

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x3x3xf32)
        t615 = paddle._C_ops.depthwise_conv2d(
            t614, t189, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t189

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t616, t617, t618, t619, t620, t621 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t615,
                t190,
                t191,
                t192,
                t193,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t615, t193, t192, t191, t190

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x1x1xf32)
        t622 = paddle._C_ops.depthwise_conv2d(
            t614, t194, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t194

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t623 = paddle._C_ops.reshape(t195, t324)
        del t195

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 1x192x1x1xf32)
        t624 = paddle._C_ops.add(t622, t623)
        del t622, t623

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t625 = paddle._C_ops.add(t616, t624)
        del t624, t616

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t626 = paddle._C_ops.add(t625, t614)
        del t614, t625

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t627, t628, t629, t630, t631, t632 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t626,
                t196,
                t197,
                t198,
                t199,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t626, t199, t198, t197, t196

        # pd_op.mean: (-1x192x1x1xf32) <- (-1x192x6x80xf32)
        t633 = paddle._C_ops.mean(t627, [2, 3], True)

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x192x1x1xf32, 48x192x1x1xf32)
        t634 = paddle._C_ops.conv2d(
            t633, t200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t633, t200

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t635 = paddle._C_ops.reshape(t201, t324)
        del t201

        # pd_op.add: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        t636 = paddle._C_ops.add(t634, t635)
        del t634, t635

        # pd_op.relu: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        t637 = paddle._C_ops.relu(t636)
        del t636

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x48x1x1xf32, 192x48x1x1xf32)
        t638 = paddle._C_ops.conv2d(
            t637, t202, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t202, t637

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t639 = paddle._C_ops.reshape(t203, t324)
        del t203

        # pd_op.add: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        t640 = paddle._C_ops.add(t638, t639)
        del t638, t639

        # pd_op.sigmoid: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        t641 = paddle._C_ops.sigmoid(t640)
        del t640

        # pd_op.multiply: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x1x1xf32)
        t642 = paddle._C_ops.multiply(t627, t641)
        del t627, t641

        # pd_op.conv2d: (-1x384x6x80xf32) <- (-1x192x6x80xf32, 384x192x1x1xf32)
        t643 = paddle._C_ops.conv2d(
            t642, t204, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t204

        # pd_op.batch_norm_: (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t644, t645, t646, t647, t648, t649 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t643,
                t205,
                t206,
                t207,
                t208,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t643, t208, t207, t206, t205

        # pd_op.gelu: (-1x384x6x80xf32) <- (-1x384x6x80xf32)
        t650 = paddle._C_ops.gelu(t644, False)
        del t644

        # pd_op.conv2d: (-1x192x6x80xf32) <- (-1x384x6x80xf32, 192x384x1x1xf32)
        t651 = paddle._C_ops.conv2d(
            t650, t209, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t650, t209

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t652, t653, t654, t655, t656, t657 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t651,
                t210,
                t211,
                t212,
                t213,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t651, t213, t212, t211, t210

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t658 = paddle._C_ops.add(t642, t652)
        del t652, t642

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x3x3xf32)
        t659 = paddle._C_ops.depthwise_conv2d(
            t658, t214, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t214

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t660, t661, t662, t663, t664, t665 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t659,
                t215,
                t216,
                t217,
                t218,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t659, t218, t217, t216, t215

        # pd_op.depthwise_conv2d: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 192x1x1x1xf32)
        t666 = paddle._C_ops.depthwise_conv2d(
            t658, t219, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t219

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t667 = paddle._C_ops.reshape(t220, t324)
        del t220

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, 1x192x1x1xf32)
        t668 = paddle._C_ops.add(t666, t667)
        del t666, t667

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t669 = paddle._C_ops.add(t660, t668)
        del t668, t660

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t670 = paddle._C_ops.add(t669, t658)
        del t658, t669

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t671, t672, t673, t674, t675, t676 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t670,
                t221,
                t222,
                t223,
                t224,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t670, t224, t223, t222, t221

        # pd_op.conv2d: (-1x384x6x80xf32) <- (-1x192x6x80xf32, 384x192x1x1xf32)
        t677 = paddle._C_ops.conv2d(
            t671, t225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225

        # pd_op.batch_norm_: (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x6x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t678, t679, t680, t681, t682, t683 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t677,
                t226,
                t227,
                t228,
                t229,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t677, t229, t228, t227, t226

        # pd_op.gelu: (-1x384x6x80xf32) <- (-1x384x6x80xf32)
        t684 = paddle._C_ops.gelu(t678, False)
        del t678

        # pd_op.conv2d: (-1x192x6x80xf32) <- (-1x384x6x80xf32, 192x384x1x1xf32)
        t685 = paddle._C_ops.conv2d(
            t684, t230, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t684, t230

        # pd_op.batch_norm_: (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x6x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t686, t687, t688, t689, t690, t691 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t685,
                t231,
                t232,
                t233,
                t234,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t685, t234, t233, t232, t231

        # pd_op.add: (-1x192x6x80xf32) <- (-1x192x6x80xf32, -1x192x6x80xf32)
        t692 = paddle._C_ops.add(t671, t686)
        del t671, t686

        # pd_op.depthwise_conv2d: (-1x192x3x80xf32) <- (-1x192x6x80xf32, 192x1x3x3xf32)
        t693 = paddle._C_ops.depthwise_conv2d(
            t692, t235, [2, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t692, t235

        # pd_op.batch_norm_: (-1x192x3x80xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x3x80xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t694, t695, t696, t697, t698, t699 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t693,
                t236,
                t237,
                t238,
                t239,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t693, t239, t238, t237, t236

        # pd_op.conv2d: (-1x384x3x80xf32) <- (-1x192x3x80xf32, 384x192x1x1xf32)
        t700 = paddle._C_ops.conv2d(
            t694, t240, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t694, t240

        # pd_op.batch_norm_: (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t701, t702, t703, t704, t705, t706 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t700,
                t241,
                t242,
                t243,
                t244,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t700, t244, t243, t242, t241

        # pd_op.conv2d: (-1x768x3x80xf32) <- (-1x384x3x80xf32, 768x384x1x1xf32)
        t707 = paddle._C_ops.conv2d(
            t701, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (-1x768x3x80xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x3x80xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t708, t709, t710, t711, t712, t713 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t707,
                t246,
                t247,
                t248,
                t249,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t707, t249, t248, t247, t246

        # pd_op.gelu: (-1x768x3x80xf32) <- (-1x768x3x80xf32)
        t714 = paddle._C_ops.gelu(t708, False)
        del t708

        # pd_op.conv2d: (-1x384x3x80xf32) <- (-1x768x3x80xf32, 384x768x1x1xf32)
        t715 = paddle._C_ops.conv2d(
            t714, t250, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t714, t250

        # pd_op.batch_norm_: (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t716, t717, t718, t719, t720, t721 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t715,
                t251,
                t252,
                t253,
                t254,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t715, t254, t253, t252, t251

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, -1x384x3x80xf32)
        t722 = paddle._C_ops.add(t701, t716)
        del t701, t716

        # pd_op.depthwise_conv2d: (-1x384x3x80xf32) <- (-1x384x3x80xf32, 384x1x3x3xf32)
        t723 = paddle._C_ops.depthwise_conv2d(
            t722, t255, [1, 1], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t255

        # pd_op.batch_norm_: (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t724, t725, t726, t727, t728, t729 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t723,
                t256,
                t257,
                t258,
                t259,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t723, t259, t258, t257, t256

        # pd_op.depthwise_conv2d: (-1x384x3x80xf32) <- (-1x384x3x80xf32, 384x1x1x1xf32)
        t730 = paddle._C_ops.depthwise_conv2d(
            t722, t260, [1, 1], [0, 0], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t260

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t731 = paddle._C_ops.reshape(t261, t324)
        del t261

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, 1x384x1x1xf32)
        t732 = paddle._C_ops.add(t730, t731)
        del t730, t731

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, -1x384x3x80xf32)
        t733 = paddle._C_ops.add(t724, t732)
        del t732, t724

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, -1x384x3x80xf32)
        t734 = paddle._C_ops.add(t733, t722)
        del t722, t733

        # pd_op.batch_norm_: (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t735, t736, t737, t738, t739, t740 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t734,
                t262,
                t263,
                t264,
                t265,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t734, t265, t264, t263, t262

        # pd_op.mean: (-1x384x1x1xf32) <- (-1x384x3x80xf32)
        t741 = paddle._C_ops.mean(t735, [2, 3], True)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x384x1x1xf32, 96x384x1x1xf32)
        t742 = paddle._C_ops.conv2d(
            t741, t266, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t741, t266

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t743 = paddle._C_ops.reshape(t267, t324)
        del t267

        # pd_op.add: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        t744 = paddle._C_ops.add(t742, t743)
        del t742, t743

        # pd_op.relu: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        t745 = paddle._C_ops.relu(t744)
        del t744

        # pd_op.conv2d: (-1x384x1x1xf32) <- (-1x96x1x1xf32, 384x96x1x1xf32)
        t746 = paddle._C_ops.conv2d(
            t745, t268, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t268, t745

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t747 = paddle._C_ops.reshape(t269, t324)
        del t269

        # pd_op.add: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 1x384x1x1xf32)
        t748 = paddle._C_ops.add(t746, t747)
        del t746, t747

        # pd_op.sigmoid: (-1x384x1x1xf32) <- (-1x384x1x1xf32)
        t749 = paddle._C_ops.sigmoid(t748)
        del t748

        # pd_op.multiply: (-1x384x3x80xf32) <- (-1x384x3x80xf32, -1x384x1x1xf32)
        t750 = paddle._C_ops.multiply(t735, t749)
        del t735, t749

        # pd_op.conv2d: (-1x768x3x80xf32) <- (-1x384x3x80xf32, 768x384x1x1xf32)
        t751 = paddle._C_ops.conv2d(
            t750, t270, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t270

        # pd_op.batch_norm_: (-1x768x3x80xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x3x80xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t752, t753, t754, t755, t756, t757 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t751,
                t271,
                t272,
                t273,
                t274,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t751, t274, t273, t272, t271

        # pd_op.gelu: (-1x768x3x80xf32) <- (-1x768x3x80xf32)
        t758 = paddle._C_ops.gelu(t752, False)
        del t752

        # pd_op.conv2d: (-1x384x3x80xf32) <- (-1x768x3x80xf32, 384x768x1x1xf32)
        t759 = paddle._C_ops.conv2d(
            t758, t275, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t758, t275

        # pd_op.batch_norm_: (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t760, t761, t762, t763, t764, t765 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t759,
                t276,
                t277,
                t278,
                t279,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t759, t279, t278, t277, t276

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, -1x384x3x80xf32)
        t766 = paddle._C_ops.add(t750, t760)
        del t760, t750

        # pd_op.depthwise_conv2d: (-1x384x3x80xf32) <- (-1x384x3x80xf32, 384x1x3x3xf32)
        t767 = paddle._C_ops.depthwise_conv2d(
            t766, t280, [1, 1], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t280

        # pd_op.batch_norm_: (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t768, t769, t770, t771, t772, t773 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t767,
                t281,
                t282,
                t283,
                t284,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t767, t284, t283, t282, t281

        # pd_op.depthwise_conv2d: (-1x384x3x80xf32) <- (-1x384x3x80xf32, 384x1x1x1xf32)
        t774 = paddle._C_ops.depthwise_conv2d(
            t766, t285, [1, 1], [0, 0], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t285

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t775 = paddle._C_ops.reshape(t286, t324)
        del t324, t286

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, 1x384x1x1xf32)
        t776 = paddle._C_ops.add(t774, t775)
        del t774, t775

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, -1x384x3x80xf32)
        t777 = paddle._C_ops.add(t768, t776)
        del t776, t768

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, -1x384x3x80xf32)
        t778 = paddle._C_ops.add(t777, t766)
        del t766, t777

        # pd_op.batch_norm_: (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t779, t780, t781, t782, t783, t784 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t778,
                t287,
                t288,
                t289,
                t290,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t778, t290, t289, t288, t287

        # pd_op.conv2d: (-1x768x3x80xf32) <- (-1x384x3x80xf32, 768x384x1x1xf32)
        t785 = paddle._C_ops.conv2d(
            t779, t291, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t291

        # pd_op.batch_norm_: (-1x768x3x80xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x3x80xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t786, t787, t788, t789, t790, t791 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t785,
                t292,
                t293,
                t294,
                t295,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t785, t295, t294, t293, t292

        # pd_op.gelu: (-1x768x3x80xf32) <- (-1x768x3x80xf32)
        t792 = paddle._C_ops.gelu(t786, False)
        del t786

        # pd_op.conv2d: (-1x384x3x80xf32) <- (-1x768x3x80xf32, 384x768x1x1xf32)
        t793 = paddle._C_ops.conv2d(
            t792, t296, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t792, t296

        # pd_op.batch_norm_: (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x3x80xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t794, t795, t796, t797, t798, t799 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t793,
                t297,
                t298,
                t299,
                t300,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t793, t300, t299, t298, t297

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, -1x384x3x80xf32)
        t800 = paddle._C_ops.add(t779, t794)
        del t779, t794

        # pd_op.shape64: (4xi64) <- (-1x384x3x80xf32)
        t801 = paddle._C_ops.shape64(t800)

        # pd_op.full_int_array: (1xi64) <- ()
        t802 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        t803 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t804 = paddle._C_ops.slice(t801, [0], t802, t803, [1], [0])
        del t802, t803, t801

        # pd_op.full_int_array: (2xi64) <- ()
        t805 = [3, 2]

        return t800, t805
