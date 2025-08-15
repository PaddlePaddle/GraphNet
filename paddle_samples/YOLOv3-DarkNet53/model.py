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
        t301: paddle.Tensor,
        t302: paddle.Tensor,
        t303: paddle.Tensor,
        t304: paddle.Tensor,
        t305: paddle.Tensor,
        t306: paddle.Tensor,
        t307: paddle.Tensor,
        t308: paddle.Tensor,
        t309: paddle.Tensor,
        t310: paddle.Tensor,
        t311: paddle.Tensor,
        t312: paddle.Tensor,
        t313: paddle.Tensor,
        t314: paddle.Tensor,
        t315: paddle.Tensor,
        t316: paddle.Tensor,
        t317: paddle.Tensor,
        t318: paddle.Tensor,
        t319: paddle.Tensor,
        t320: paddle.Tensor,
        t321: paddle.Tensor,
        t322: paddle.Tensor,
        t323: paddle.Tensor,
        t324: paddle.Tensor,
        t325: paddle.Tensor,
        t326: paddle.Tensor,
        t327: paddle.Tensor,
        t328: paddle.Tensor,
        t329: paddle.Tensor,
        t330: paddle.Tensor,
        t331: paddle.Tensor,
        t332: paddle.Tensor,
        t333: paddle.Tensor,
        t334: paddle.Tensor,
        t335: paddle.Tensor,
        t336: paddle.Tensor,
        t337: paddle.Tensor,
        t338: paddle.Tensor,
        t339: paddle.Tensor,
        t340: paddle.Tensor,
        t341: paddle.Tensor,
        t342: paddle.Tensor,
        t343: paddle.Tensor,
        t344: paddle.Tensor,
        t345: paddle.Tensor,
        t346: paddle.Tensor,
        t347: paddle.Tensor,
        t348: paddle.Tensor,
        t349: paddle.Tensor,
        t350: paddle.Tensor,
        t351: paddle.Tensor,
        t352: paddle.Tensor,
        t353: paddle.Tensor,
        t354: paddle.Tensor,
        t355: paddle.Tensor,
        t356: paddle.Tensor,
        t357: paddle.Tensor,
        t358: paddle.Tensor,
        t359: paddle.Tensor,
        t360: paddle.Tensor,
        t361: paddle.Tensor,
        t362: paddle.Tensor,
        t363: paddle.Tensor,
        t364: paddle.Tensor,
        t365: paddle.Tensor,
        input1: paddle.Tensor,
        input2: paddle.Tensor,
        input3: paddle.Tensor,
        input4: paddle.Tensor,
    ):
        t366 = None
        # pd_op.conv2d: (2x32x480x480xf32) <- (2x3x480x480xf32, 32x3x3x3xf32)
        t367 = paddle._C_ops.conv2d(
            input0, t0, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (2x32x480x480xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (2x32x480x480xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t368, t369, t370, t371, t372, t373 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t367,
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

        # pd_op.leaky_relu: (2x32x480x480xf32) <- (2x32x480x480xf32)
        t374 = paddle._C_ops.leaky_relu(t368, float("0.1"))

        # pd_op.conv2d: (2x64x240x240xf32) <- (2x32x480x480xf32, 64x32x3x3xf32)
        t375 = paddle._C_ops.conv2d(
            t374, t5, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (2x64x240x240xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (2x64x240x240xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t376, t377, t378, t379, t380, t381 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t375,
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

        # pd_op.leaky_relu: (2x64x240x240xf32) <- (2x64x240x240xf32)
        t382 = paddle._C_ops.leaky_relu(t376, float("0.1"))

        # pd_op.conv2d: (2x32x240x240xf32) <- (2x64x240x240xf32, 32x64x1x1xf32)
        t383 = paddle._C_ops.conv2d(
            t382, t10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (2x32x240x240xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (2x32x240x240xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t384, t385, t386, t387, t388, t389 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t383,
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

        # pd_op.leaky_relu: (2x32x240x240xf32) <- (2x32x240x240xf32)
        t390 = paddle._C_ops.leaky_relu(t384, float("0.1"))

        # pd_op.conv2d: (2x64x240x240xf32) <- (2x32x240x240xf32, 64x32x3x3xf32)
        t391 = paddle._C_ops.conv2d(
            t390, t15, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (2x64x240x240xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (2x64x240x240xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t392, t393, t394, t395, t396, t397 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t391,
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

        # pd_op.leaky_relu: (2x64x240x240xf32) <- (2x64x240x240xf32)
        t398 = paddle._C_ops.leaky_relu(t392, float("0.1"))

        # pd_op.add: (2x64x240x240xf32) <- (2x64x240x240xf32, 2x64x240x240xf32)
        t399 = paddle._C_ops.add(t382, t398)

        # pd_op.conv2d: (2x128x120x120xf32) <- (2x64x240x240xf32, 128x64x3x3xf32)
        t400 = paddle._C_ops.conv2d(
            t399, t20, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (2x128x120x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x120x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t401, t402, t403, t404, t405, t406 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t400,
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

        # pd_op.leaky_relu: (2x128x120x120xf32) <- (2x128x120x120xf32)
        t407 = paddle._C_ops.leaky_relu(t401, float("0.1"))

        # pd_op.conv2d: (2x64x120x120xf32) <- (2x128x120x120xf32, 64x128x1x1xf32)
        t408 = paddle._C_ops.conv2d(
            t407, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (2x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (2x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t409, t410, t411, t412, t413, t414 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t408,
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

        # pd_op.leaky_relu: (2x64x120x120xf32) <- (2x64x120x120xf32)
        t415 = paddle._C_ops.leaky_relu(t409, float("0.1"))

        # pd_op.conv2d: (2x128x120x120xf32) <- (2x64x120x120xf32, 128x64x3x3xf32)
        t416 = paddle._C_ops.conv2d(
            t415, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (2x128x120x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x120x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t417, t418, t419, t420, t421, t422 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t416,
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

        # pd_op.leaky_relu: (2x128x120x120xf32) <- (2x128x120x120xf32)
        t423 = paddle._C_ops.leaky_relu(t417, float("0.1"))

        # pd_op.add: (2x128x120x120xf32) <- (2x128x120x120xf32, 2x128x120x120xf32)
        t424 = paddle._C_ops.add(t407, t423)

        # pd_op.conv2d: (2x64x120x120xf32) <- (2x128x120x120xf32, 64x128x1x1xf32)
        t425 = paddle._C_ops.conv2d(
            t424, t35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (2x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (2x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t426, t427, t428, t429, t430, t431 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t425,
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

        # pd_op.leaky_relu: (2x64x120x120xf32) <- (2x64x120x120xf32)
        t432 = paddle._C_ops.leaky_relu(t426, float("0.1"))

        # pd_op.conv2d: (2x128x120x120xf32) <- (2x64x120x120xf32, 128x64x3x3xf32)
        t433 = paddle._C_ops.conv2d(
            t432, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (2x128x120x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x120x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t434, t435, t436, t437, t438, t439 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t433,
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

        # pd_op.leaky_relu: (2x128x120x120xf32) <- (2x128x120x120xf32)
        t440 = paddle._C_ops.leaky_relu(t434, float("0.1"))

        # pd_op.add: (2x128x120x120xf32) <- (2x128x120x120xf32, 2x128x120x120xf32)
        t441 = paddle._C_ops.add(t424, t440)

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x120x120xf32, 256x128x3x3xf32)
        t442 = paddle._C_ops.conv2d(
            t441, t45, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t443, t444, t445, t446, t447, t448 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t442,
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

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t449 = paddle._C_ops.leaky_relu(t443, float("0.1"))

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t450 = paddle._C_ops.conv2d(
            t449, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t451, t452, t453, t454, t455, t456 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t450,
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

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t457 = paddle._C_ops.leaky_relu(t451, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t458 = paddle._C_ops.conv2d(
            t457, t55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t459, t460, t461, t462, t463, t464 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t458,
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

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t465 = paddle._C_ops.leaky_relu(t459, float("0.1"))

        # pd_op.add: (2x256x60x60xf32) <- (2x256x60x60xf32, 2x256x60x60xf32)
        t466 = paddle._C_ops.add(t449, t465)

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t467 = paddle._C_ops.conv2d(
            t466, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t468, t469, t470, t471, t472, t473 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t467,
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

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t474 = paddle._C_ops.leaky_relu(t468, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t475 = paddle._C_ops.conv2d(
            t474, t65, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t476, t477, t478, t479, t480, t481 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t475,
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

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t482 = paddle._C_ops.leaky_relu(t476, float("0.1"))

        # pd_op.add: (2x256x60x60xf32) <- (2x256x60x60xf32, 2x256x60x60xf32)
        t483 = paddle._C_ops.add(t466, t482)

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t484 = paddle._C_ops.conv2d(
            t483, t70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t485, t486, t487, t488, t489, t490 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t484,
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

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t491 = paddle._C_ops.leaky_relu(t485, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t492 = paddle._C_ops.conv2d(
            t491, t75, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t493, t494, t495, t496, t497, t498 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t492,
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

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t499 = paddle._C_ops.leaky_relu(t493, float("0.1"))

        # pd_op.add: (2x256x60x60xf32) <- (2x256x60x60xf32, 2x256x60x60xf32)
        t500 = paddle._C_ops.add(t483, t499)

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t501 = paddle._C_ops.conv2d(
            t500, t80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t502, t503, t504, t505, t506, t507 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t501,
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

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t508 = paddle._C_ops.leaky_relu(t502, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t509 = paddle._C_ops.conv2d(
            t508, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t510, t511, t512, t513, t514, t515 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t509,
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

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t516 = paddle._C_ops.leaky_relu(t510, float("0.1"))

        # pd_op.add: (2x256x60x60xf32) <- (2x256x60x60xf32, 2x256x60x60xf32)
        t517 = paddle._C_ops.add(t500, t516)

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t518 = paddle._C_ops.conv2d(
            t517, t90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t519, t520, t521, t522, t523, t524 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t518,
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

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t525 = paddle._C_ops.leaky_relu(t519, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t526 = paddle._C_ops.conv2d(
            t525, t95, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t527, t528, t529, t530, t531, t532 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t526,
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

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t533 = paddle._C_ops.leaky_relu(t527, float("0.1"))

        # pd_op.add: (2x256x60x60xf32) <- (2x256x60x60xf32, 2x256x60x60xf32)
        t534 = paddle._C_ops.add(t517, t533)

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t535 = paddle._C_ops.conv2d(
            t534, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t536, t537, t538, t539, t540, t541 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t535,
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

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t542 = paddle._C_ops.leaky_relu(t536, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t543 = paddle._C_ops.conv2d(
            t542, t105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t544, t545, t546, t547, t548, t549 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t543,
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

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t550 = paddle._C_ops.leaky_relu(t544, float("0.1"))

        # pd_op.add: (2x256x60x60xf32) <- (2x256x60x60xf32, 2x256x60x60xf32)
        t551 = paddle._C_ops.add(t534, t550)

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t552 = paddle._C_ops.conv2d(
            t551, t110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t553, t554, t555, t556, t557, t558 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t552,
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

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t559 = paddle._C_ops.leaky_relu(t553, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t560 = paddle._C_ops.conv2d(
            t559, t115, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t561, t562, t563, t564, t565, t566 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t560,
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

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t567 = paddle._C_ops.leaky_relu(t561, float("0.1"))

        # pd_op.add: (2x256x60x60xf32) <- (2x256x60x60xf32, 2x256x60x60xf32)
        t568 = paddle._C_ops.add(t551, t567)

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t569 = paddle._C_ops.conv2d(
            t568, t120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t570, t571, t572, t573, t574, t575 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t569,
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

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t576 = paddle._C_ops.leaky_relu(t570, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t577 = paddle._C_ops.conv2d(
            t576, t125, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t578, t579, t580, t581, t582, t583 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t577,
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

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t584 = paddle._C_ops.leaky_relu(t578, float("0.1"))

        # pd_op.add: (2x256x60x60xf32) <- (2x256x60x60xf32, 2x256x60x60xf32)
        t585 = paddle._C_ops.add(t568, t584)

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x60x60xf32, 512x256x3x3xf32)
        t586 = paddle._C_ops.conv2d(
            t585, t130, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t587, t588, t589, t590, t591, t592 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t586,
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

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t593 = paddle._C_ops.leaky_relu(t587, float("0.1"))

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t594 = paddle._C_ops.conv2d(
            t593, t135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t595, t596, t597, t598, t599, t600 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t594,
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

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t601 = paddle._C_ops.leaky_relu(t595, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t602 = paddle._C_ops.conv2d(
            t601, t140, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t603, t604, t605, t606, t607, t608 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t602,
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

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t609 = paddle._C_ops.leaky_relu(t603, float("0.1"))

        # pd_op.add: (2x512x30x30xf32) <- (2x512x30x30xf32, 2x512x30x30xf32)
        t610 = paddle._C_ops.add(t593, t609)

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t611 = paddle._C_ops.conv2d(
            t610, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t612, t613, t614, t615, t616, t617 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t611,
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

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t618 = paddle._C_ops.leaky_relu(t612, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t619 = paddle._C_ops.conv2d(
            t618, t150, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t620, t621, t622, t623, t624, t625 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t619,
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

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t626 = paddle._C_ops.leaky_relu(t620, float("0.1"))

        # pd_op.add: (2x512x30x30xf32) <- (2x512x30x30xf32, 2x512x30x30xf32)
        t627 = paddle._C_ops.add(t610, t626)

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t628 = paddle._C_ops.conv2d(
            t627, t155, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t629, t630, t631, t632, t633, t634 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t628,
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

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t635 = paddle._C_ops.leaky_relu(t629, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t636 = paddle._C_ops.conv2d(
            t635, t160, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t637, t638, t639, t640, t641, t642 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t636,
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

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t643 = paddle._C_ops.leaky_relu(t637, float("0.1"))

        # pd_op.add: (2x512x30x30xf32) <- (2x512x30x30xf32, 2x512x30x30xf32)
        t644 = paddle._C_ops.add(t627, t643)

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t645 = paddle._C_ops.conv2d(
            t644, t165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t646, t647, t648, t649, t650, t651 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t645,
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

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t652 = paddle._C_ops.leaky_relu(t646, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t653 = paddle._C_ops.conv2d(
            t652, t170, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t654, t655, t656, t657, t658, t659 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t653,
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

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t660 = paddle._C_ops.leaky_relu(t654, float("0.1"))

        # pd_op.add: (2x512x30x30xf32) <- (2x512x30x30xf32, 2x512x30x30xf32)
        t661 = paddle._C_ops.add(t644, t660)

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t662 = paddle._C_ops.conv2d(
            t661, t175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t663, t664, t665, t666, t667, t668 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t662,
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

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t669 = paddle._C_ops.leaky_relu(t663, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t670 = paddle._C_ops.conv2d(
            t669, t180, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t671, t672, t673, t674, t675, t676 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t670,
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

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t677 = paddle._C_ops.leaky_relu(t671, float("0.1"))

        # pd_op.add: (2x512x30x30xf32) <- (2x512x30x30xf32, 2x512x30x30xf32)
        t678 = paddle._C_ops.add(t661, t677)

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t679 = paddle._C_ops.conv2d(
            t678, t185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t680, t681, t682, t683, t684, t685 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t679,
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

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t686 = paddle._C_ops.leaky_relu(t680, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t687 = paddle._C_ops.conv2d(
            t686, t190, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t688, t689, t690, t691, t692, t693 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t687,
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

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t694 = paddle._C_ops.leaky_relu(t688, float("0.1"))

        # pd_op.add: (2x512x30x30xf32) <- (2x512x30x30xf32, 2x512x30x30xf32)
        t695 = paddle._C_ops.add(t678, t694)

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t696 = paddle._C_ops.conv2d(
            t695, t195, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t697, t698, t699, t700, t701, t702 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t696,
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

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t703 = paddle._C_ops.leaky_relu(t697, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t704 = paddle._C_ops.conv2d(
            t703, t200, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t705, t706, t707, t708, t709, t710 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t704,
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

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t711 = paddle._C_ops.leaky_relu(t705, float("0.1"))

        # pd_op.add: (2x512x30x30xf32) <- (2x512x30x30xf32, 2x512x30x30xf32)
        t712 = paddle._C_ops.add(t695, t711)

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t713 = paddle._C_ops.conv2d(
            t712, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t714, t715, t716, t717, t718, t719 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t713,
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

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t720 = paddle._C_ops.leaky_relu(t714, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t721 = paddle._C_ops.conv2d(
            t720, t210, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t722, t723, t724, t725, t726, t727 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t721,
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

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t728 = paddle._C_ops.leaky_relu(t722, float("0.1"))

        # pd_op.add: (2x512x30x30xf32) <- (2x512x30x30xf32, 2x512x30x30xf32)
        t729 = paddle._C_ops.add(t712, t728)

        # pd_op.conv2d: (2x1024x15x15xf32) <- (2x512x30x30xf32, 1024x512x3x3xf32)
        t730 = paddle._C_ops.conv2d(
            t729, t215, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.batch_norm_: (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t731, t732, t733, t734, t735, t736 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t730,
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

        # pd_op.leaky_relu: (2x1024x15x15xf32) <- (2x1024x15x15xf32)
        t737 = paddle._C_ops.leaky_relu(t731, float("0.1"))

        # pd_op.conv2d: (2x512x15x15xf32) <- (2x1024x15x15xf32, 512x1024x1x1xf32)
        t738 = paddle._C_ops.conv2d(
            t737, t220, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t220

        # pd_op.batch_norm_: (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t739, t740, t741, t742, t743, t744 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t738,
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

        # pd_op.leaky_relu: (2x512x15x15xf32) <- (2x512x15x15xf32)
        t745 = paddle._C_ops.leaky_relu(t739, float("0.1"))

        # pd_op.conv2d: (2x1024x15x15xf32) <- (2x512x15x15xf32, 1024x512x3x3xf32)
        t746 = paddle._C_ops.conv2d(
            t745, t225, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225

        # pd_op.batch_norm_: (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t747, t748, t749, t750, t751, t752 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t746,
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

        # pd_op.leaky_relu: (2x1024x15x15xf32) <- (2x1024x15x15xf32)
        t753 = paddle._C_ops.leaky_relu(t747, float("0.1"))

        # pd_op.add: (2x1024x15x15xf32) <- (2x1024x15x15xf32, 2x1024x15x15xf32)
        t754 = paddle._C_ops.add(t737, t753)

        # pd_op.conv2d: (2x512x15x15xf32) <- (2x1024x15x15xf32, 512x1024x1x1xf32)
        t755 = paddle._C_ops.conv2d(
            t754, t230, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t230

        # pd_op.batch_norm_: (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t756, t757, t758, t759, t760, t761 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t755,
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

        # pd_op.leaky_relu: (2x512x15x15xf32) <- (2x512x15x15xf32)
        t762 = paddle._C_ops.leaky_relu(t756, float("0.1"))

        # pd_op.conv2d: (2x1024x15x15xf32) <- (2x512x15x15xf32, 1024x512x3x3xf32)
        t763 = paddle._C_ops.conv2d(
            t762, t235, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t764, t765, t766, t767, t768, t769 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t763,
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

        # pd_op.leaky_relu: (2x1024x15x15xf32) <- (2x1024x15x15xf32)
        t770 = paddle._C_ops.leaky_relu(t764, float("0.1"))

        # pd_op.add: (2x1024x15x15xf32) <- (2x1024x15x15xf32, 2x1024x15x15xf32)
        t771 = paddle._C_ops.add(t754, t770)

        # pd_op.conv2d: (2x512x15x15xf32) <- (2x1024x15x15xf32, 512x1024x1x1xf32)
        t772 = paddle._C_ops.conv2d(
            t771, t240, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t240

        # pd_op.batch_norm_: (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t773, t774, t775, t776, t777, t778 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t772,
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

        # pd_op.leaky_relu: (2x512x15x15xf32) <- (2x512x15x15xf32)
        t779 = paddle._C_ops.leaky_relu(t773, float("0.1"))

        # pd_op.conv2d: (2x1024x15x15xf32) <- (2x512x15x15xf32, 1024x512x3x3xf32)
        t780 = paddle._C_ops.conv2d(
            t779, t245, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t781, t782, t783, t784, t785, t786 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t780,
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

        # pd_op.leaky_relu: (2x1024x15x15xf32) <- (2x1024x15x15xf32)
        t787 = paddle._C_ops.leaky_relu(t781, float("0.1"))

        # pd_op.add: (2x1024x15x15xf32) <- (2x1024x15x15xf32, 2x1024x15x15xf32)
        t788 = paddle._C_ops.add(t771, t787)

        # pd_op.conv2d: (2x512x15x15xf32) <- (2x1024x15x15xf32, 512x1024x1x1xf32)
        t789 = paddle._C_ops.conv2d(
            t788, t250, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t250

        # pd_op.batch_norm_: (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t790, t791, t792, t793, t794, t795 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t789,
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

        # pd_op.leaky_relu: (2x512x15x15xf32) <- (2x512x15x15xf32)
        t796 = paddle._C_ops.leaky_relu(t790, float("0.1"))

        # pd_op.conv2d: (2x1024x15x15xf32) <- (2x512x15x15xf32, 1024x512x3x3xf32)
        t797 = paddle._C_ops.conv2d(
            t796, t255, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255

        # pd_op.batch_norm_: (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t798, t799, t800, t801, t802, t803 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t797,
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
        del t259, t258, t257, t256

        # pd_op.leaky_relu: (2x1024x15x15xf32) <- (2x1024x15x15xf32)
        t804 = paddle._C_ops.leaky_relu(t798, float("0.1"))

        # pd_op.add: (2x1024x15x15xf32) <- (2x1024x15x15xf32, 2x1024x15x15xf32)
        t805 = paddle._C_ops.add(t788, t804)

        # pd_op.conv2d: (2x512x15x15xf32) <- (2x1024x15x15xf32, 512x1024x1x1xf32)
        t806 = paddle._C_ops.conv2d(
            t805, t260, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t260

        # pd_op.batch_norm_: (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t807, t808, t809, t810, t811, t812 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t806,
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

        # pd_op.leaky_relu: (2x512x15x15xf32) <- (2x512x15x15xf32)
        t813 = paddle._C_ops.leaky_relu(t807, float("0.1"))

        # pd_op.conv2d: (2x1024x15x15xf32) <- (2x512x15x15xf32, 1024x512x3x3xf32)
        t814 = paddle._C_ops.conv2d(
            t813, t265, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t265

        # pd_op.batch_norm_: (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t815, t816, t817, t818, t819, t820 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t814,
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

        # pd_op.leaky_relu: (2x1024x15x15xf32) <- (2x1024x15x15xf32)
        t821 = paddle._C_ops.leaky_relu(t815, float("0.1"))

        # pd_op.conv2d: (2x512x15x15xf32) <- (2x1024x15x15xf32, 512x1024x1x1xf32)
        t822 = paddle._C_ops.conv2d(
            t821, t270, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t270

        # pd_op.batch_norm_: (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t823, t824, t825, t826, t827, t828 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t822,
                t271,
                t272,
                t273,
                t274,
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
        del t274, t273, t272, t271

        # pd_op.leaky_relu: (2x512x15x15xf32) <- (2x512x15x15xf32)
        t829 = paddle._C_ops.leaky_relu(t823, float("0.1"))

        # pd_op.conv2d: (2x1024x15x15xf32) <- (2x512x15x15xf32, 1024x512x3x3xf32)
        t830 = paddle._C_ops.conv2d(
            t829, t275, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t275

        # pd_op.batch_norm_: (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t831, t832, t833, t834, t835, t836 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t830,
                t276,
                t277,
                t278,
                t279,
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
        del t279, t278, t277, t276

        # pd_op.leaky_relu: (2x1024x15x15xf32) <- (2x1024x15x15xf32)
        t837 = paddle._C_ops.leaky_relu(t831, float("0.1"))

        # pd_op.conv2d: (2x512x15x15xf32) <- (2x1024x15x15xf32, 512x1024x1x1xf32)
        t838 = paddle._C_ops.conv2d(
            t837, t280, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t280

        # pd_op.batch_norm_: (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x15x15xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t839, t840, t841, t842, t843, t844 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t838,
                t281,
                t282,
                t283,
                t284,
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
        del t284, t283, t282, t281

        # pd_op.leaky_relu: (2x512x15x15xf32) <- (2x512x15x15xf32)
        t845 = paddle._C_ops.leaky_relu(t839, float("0.1"))

        # pd_op.conv2d: (2x1024x15x15xf32) <- (2x512x15x15xf32, 1024x512x3x3xf32)
        t846 = paddle._C_ops.conv2d(
            t845, t285, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t285

        # pd_op.batch_norm_: (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (2x1024x15x15xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t847, t848, t849, t850, t851, t852 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t846,
                t286,
                t287,
                t288,
                t289,
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
        del t289, t288, t287, t286

        # pd_op.leaky_relu: (2x1024x15x15xf32) <- (2x1024x15x15xf32)
        t853 = paddle._C_ops.leaky_relu(t847, float("0.1"))

        # pd_op.conv2d: (2x256x15x15xf32) <- (2x512x15x15xf32, 256x512x1x1xf32)
        t854 = paddle._C_ops.conv2d(
            t845, t290, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t290

        # pd_op.batch_norm_: (2x256x15x15xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x15x15xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t855, t856, t857, t858, t859, t860 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t854,
                t291,
                t292,
                t293,
                t294,
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
        del t294, t293, t292, t291

        # pd_op.leaky_relu: (2x256x15x15xf32) <- (2x256x15x15xf32)
        t861 = paddle._C_ops.leaky_relu(t855, float("0.1"))

        # pd_op.nearest_interp: (2x256x30x30xf32) <- (2x256x15x15xf32, None, None, None)
        t862 = paddle._C_ops.nearest_interp(
            t861,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.full: (1xi32) <- ()
        t863 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t864 = t863

        # builtin.combine: ([2x256x30x30xf32, 2x512x30x30xf32]) <- (2x256x30x30xf32, 2x512x30x30xf32)
        t865 = [t862, t729]

        # pd_op.concat: (2x768x30x30xf32) <- ([2x256x30x30xf32, 2x512x30x30xf32], 1xi32)
        t866 = paddle._C_ops.concat(t865, t863)
        del t865

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x768x30x30xf32, 256x768x1x1xf32)
        t867 = paddle._C_ops.conv2d(
            t866, t295, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t295

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t868, t869, t870, t871, t872, t873 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t867,
                t296,
                t297,
                t298,
                t299,
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
        del t299, t298, t297, t296

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t874 = paddle._C_ops.leaky_relu(t868, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t875 = paddle._C_ops.conv2d(
            t874, t300, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t300

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t876, t877, t878, t879, t880, t881 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t875,
                t301,
                t302,
                t303,
                t304,
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
        del t304, t303, t302, t301

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t882 = paddle._C_ops.leaky_relu(t876, float("0.1"))

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t883 = paddle._C_ops.conv2d(
            t882, t305, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t305

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t884, t885, t886, t887, t888, t889 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t883,
                t306,
                t307,
                t308,
                t309,
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
        del t309, t308, t307, t306

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t890 = paddle._C_ops.leaky_relu(t884, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t891 = paddle._C_ops.conv2d(
            t890, t310, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t310

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t892, t893, t894, t895, t896, t897 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t891,
                t311,
                t312,
                t313,
                t314,
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
        del t314, t313, t312, t311

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t898 = paddle._C_ops.leaky_relu(t892, float("0.1"))

        # pd_op.conv2d: (2x256x30x30xf32) <- (2x512x30x30xf32, 256x512x1x1xf32)
        t899 = paddle._C_ops.conv2d(
            t898, t315, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t315

        # pd_op.batch_norm_: (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x30x30xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t900, t901, t902, t903, t904, t905 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t899,
                t316,
                t317,
                t318,
                t319,
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
        del t319, t318, t317, t316

        # pd_op.leaky_relu: (2x256x30x30xf32) <- (2x256x30x30xf32)
        t906 = paddle._C_ops.leaky_relu(t900, float("0.1"))

        # pd_op.conv2d: (2x512x30x30xf32) <- (2x256x30x30xf32, 512x256x3x3xf32)
        t907 = paddle._C_ops.conv2d(
            t906, t320, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t320

        # pd_op.batch_norm_: (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x30x30xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t908, t909, t910, t911, t912, t913 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t907,
                t321,
                t322,
                t323,
                t324,
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
        del t324, t323, t322, t321

        # pd_op.leaky_relu: (2x512x30x30xf32) <- (2x512x30x30xf32)
        t914 = paddle._C_ops.leaky_relu(t908, float("0.1"))

        # pd_op.conv2d: (2x128x30x30xf32) <- (2x256x30x30xf32, 128x256x1x1xf32)
        t915 = paddle._C_ops.conv2d(
            t906, t325, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t325

        # pd_op.batch_norm_: (2x128x30x30xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x30x30xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t916, t917, t918, t919, t920, t921 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t915,
                t326,
                t327,
                t328,
                t329,
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
        del t329, t328, t327, t326

        # pd_op.leaky_relu: (2x128x30x30xf32) <- (2x128x30x30xf32)
        t922 = paddle._C_ops.leaky_relu(t916, float("0.1"))

        # pd_op.nearest_interp: (2x128x60x60xf32) <- (2x128x30x30xf32, None, None, None)
        t923 = paddle._C_ops.nearest_interp(
            t922,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # builtin.combine: ([2x128x60x60xf32, 2x256x60x60xf32]) <- (2x128x60x60xf32, 2x256x60x60xf32)
        t924 = [t923, t585]

        # pd_op.concat: (2x384x60x60xf32) <- ([2x128x60x60xf32, 2x256x60x60xf32], 1xi32)
        t925 = paddle._C_ops.concat(t924, t863)
        del t924

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x384x60x60xf32, 128x384x1x1xf32)
        t926 = paddle._C_ops.conv2d(
            t925, t330, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t330

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t927, t928, t929, t930, t931, t932 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t926,
                t331,
                t332,
                t333,
                t334,
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
        del t334, t333, t332, t331

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t933 = paddle._C_ops.leaky_relu(t927, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t934 = paddle._C_ops.conv2d(
            t933, t335, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t335

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t935, t936, t937, t938, t939, t940 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t934,
                t336,
                t337,
                t338,
                t339,
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
        del t339, t338, t337, t336

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t941 = paddle._C_ops.leaky_relu(t935, float("0.1"))

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t942 = paddle._C_ops.conv2d(
            t941, t340, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t340

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t943, t944, t945, t946, t947, t948 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t942,
                t341,
                t342,
                t343,
                t344,
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
        del t344, t343, t342, t341

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t949 = paddle._C_ops.leaky_relu(t943, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t950 = paddle._C_ops.conv2d(
            t949, t345, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t345

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t951, t952, t953, t954, t955, t956 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t950,
                t346,
                t347,
                t348,
                t349,
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
        del t349, t348, t347, t346

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t957 = paddle._C_ops.leaky_relu(t951, float("0.1"))

        # pd_op.conv2d: (2x128x60x60xf32) <- (2x256x60x60xf32, 128x256x1x1xf32)
        t958 = paddle._C_ops.conv2d(
            t957, t350, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t350

        # pd_op.batch_norm_: (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x60x60xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t959, t960, t961, t962, t963, t964 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t958,
                t351,
                t352,
                t353,
                t354,
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
        del t354, t353, t352, t351

        # pd_op.leaky_relu: (2x128x60x60xf32) <- (2x128x60x60xf32)
        t965 = paddle._C_ops.leaky_relu(t959, float("0.1"))

        # pd_op.conv2d: (2x256x60x60xf32) <- (2x128x60x60xf32, 256x128x3x3xf32)
        t966 = paddle._C_ops.conv2d(
            t965, t355, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t355

        # pd_op.batch_norm_: (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x60x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t967, t968, t969, t970, t971, t972 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t966,
                t356,
                t357,
                t358,
                t359,
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
        del t359, t358, t357, t356

        # pd_op.leaky_relu: (2x256x60x60xf32) <- (2x256x60x60xf32)
        t973 = paddle._C_ops.leaky_relu(t967, float("0.1"))

        # pd_op.conv2d: (2x27x15x15xf32) <- (2x1024x15x15xf32, 27x1024x1x1xf32)
        t974 = paddle._C_ops.conv2d(
            t853, t360, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t360

        # pd_op.full_int_array: (4xi64) <- ()
        t975 = [1, -1, 1, 1]

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        t976 = paddle._C_ops.reshape(t361, t975)
        del t361

        # pd_op.add: (2x27x15x15xf32) <- (2x27x15x15xf32, 1x27x1x1xf32)
        t977 = paddle._C_ops.add(t974, t976)

        # pd_op.conv2d: (2x27x30x30xf32) <- (2x512x30x30xf32, 27x512x1x1xf32)
        t978 = paddle._C_ops.conv2d(
            t914, t362, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t362

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        t979 = paddle._C_ops.reshape(t363, t975)
        del t363

        # pd_op.add: (2x27x30x30xf32) <- (2x27x30x30xf32, 1x27x1x1xf32)
        t980 = paddle._C_ops.add(t978, t979)

        # pd_op.conv2d: (2x27x60x60xf32) <- (2x256x60x60xf32, 27x256x1x1xf32)
        t981 = paddle._C_ops.conv2d(
            t973, t364, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t364

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        t982 = paddle._C_ops.reshape(t365, t975)
        del t975, t365

        # pd_op.add: (2x27x60x60xf32) <- (2x27x60x60xf32, 1x27x1x1xf32)
        t983 = paddle._C_ops.add(t981, t982)

        # pd_op.full_int_array: (5xi64) <- ()
        t984 = [2, 3, -1, 15, 15]

        # pd_op.reshape: (2x3x9x15x15xf32) <- (2x27x15x15xf32, 5xi64)
        t985 = paddle._C_ops.reshape(t977, t984)
        del t984

        # pd_op.transpose: (2x3x15x15x9xf32) <- (2x3x9x15x15xf32)
        t986 = paddle._C_ops.transpose(t985, [0, 1, 3, 4, 2])
        del t985

        # pd_op.full_int_array: (1xi64) <- ()
        t987 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        t988 = t987

        # pd_op.assign: (1xi64) <- (1xi64)
        t989 = t987

        # pd_op.full_int_array: (1xi64) <- ()
        t990 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t991 = t990

        # pd_op.assign: (1xi64) <- (1xi64)
        t992 = t990

        # pd_op.assign: (1xi64) <- (1xi64)
        t993 = t990

        # pd_op.assign: (1xi64) <- (1xi64)
        t994 = t990

        # pd_op.assign: (1xi64) <- (1xi64)
        t995 = t990

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x9xf32, 1xi64, 1xi64)
        t996 = paddle._C_ops.slice(t986, [4], t987, t990, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t997 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        t998 = t997

        # pd_op.assign: (1xi64) <- (1xi64)
        t999 = t997

        # pd_op.assign: (1xi64) <- (1xi64)
        t1000 = t997

        # pd_op.assign: (1xi64) <- (1xi64)
        t1001 = t997

        # pd_op.assign: (1xi64) <- (1xi64)
        t1002 = t997

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x9xf32, 1xi64, 1xi64)
        t1003 = paddle._C_ops.slice(t986, [4], t990, t997, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1004 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1005 = t1004

        # pd_op.assign: (1xi64) <- (1xi64)
        t1006 = t1004

        # pd_op.assign: (1xi64) <- (1xi64)
        t1007 = t1004

        # pd_op.assign: (1xi64) <- (1xi64)
        t1008 = t1004

        # pd_op.assign: (1xi64) <- (1xi64)
        t1009 = t1004

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x9xf32, 1xi64, 1xi64)
        t1010 = paddle._C_ops.slice(t986, [4], t997, t1004, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1011 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1012 = t1011

        # pd_op.assign: (1xi64) <- (1xi64)
        t1013 = t1011

        # pd_op.assign: (1xi64) <- (1xi64)
        t1014 = t1011

        # pd_op.assign: (1xi64) <- (1xi64)
        t1015 = t1011

        # pd_op.assign: (1xi64) <- (1xi64)
        t1016 = t1011

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x9xf32, 1xi64, 1xi64)
        t1017 = paddle._C_ops.slice(t986, [4], t1004, t1011, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1018 = [5]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1019 = t1018

        # pd_op.assign: (1xi64) <- (1xi64)
        t1020 = t1018

        # pd_op.assign: (1xi64) <- (1xi64)
        t1021 = t1018

        # pd_op.assign: (1xi64) <- (1xi64)
        t1022 = t1018

        # pd_op.assign: (1xi64) <- (1xi64)
        t1023 = t1018

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x9xf32, 1xi64, 1xi64)
        t1024 = paddle._C_ops.slice(t986, [4], t1011, t1018, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1025 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1026 = t1025

        # pd_op.assign: (1xi64) <- (1xi64)
        t1027 = t1025

        # pd_op.slice: (2x3x15x15x4xf32) <- (2x3x15x15x9xf32, 1xi64, 1xi64)
        t1028 = paddle._C_ops.slice(t986, [4], t1018, t1025, [1], [])

        # pd_op.transpose: (2x3x15x15x10xf32) <- (2x3x10x15x15xf32)
        t1029 = paddle._C_ops.transpose(input1, [0, 1, 3, 4, 2])
        del input1

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x10xf32, 1xi64, 1xi64)
        t1030 = paddle._C_ops.slice(t1029, [4], t987, t990, [1], [])

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x10xf32, 1xi64, 1xi64)
        t1031 = paddle._C_ops.slice(t1029, [4], t990, t997, [1], [])

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x10xf32, 1xi64, 1xi64)
        t1032 = paddle._C_ops.slice(t1029, [4], t997, t1004, [1], [])

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x10xf32, 1xi64, 1xi64)
        t1033 = paddle._C_ops.slice(t1029, [4], t1004, t1011, [1], [])

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x10xf32, 1xi64, 1xi64)
        t1034 = paddle._C_ops.slice(t1029, [4], t1011, t1018, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1035 = [6]

        # pd_op.slice: (2x3x15x15x1xf32) <- (2x3x15x15x10xf32, 1xi64, 1xi64)
        t1036 = paddle._C_ops.slice(t1029, [4], t1018, t1035, [1], [])

        # pd_op.slice: (2x3x15x15x4xf32) <- (2x3x15x15x10xf32, 1xi64, 1xi64)
        t1037 = paddle._C_ops.slice(t1029, [4], t1035, t1025, [1], [])
        del t1029

        # pd_op.multiply: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1038 = paddle._C_ops.multiply(t1034, t1036)
        del t1034

        # pd_op.sigmoid: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32)
        t1039 = paddle._C_ops.sigmoid(t996)

        # pd_op.full: (1xf32) <- ()
        t1040 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t1041 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1042 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1043 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1044 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1045 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1046 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1047 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1048 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1049 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1050 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1051 = t1040

        # pd_op.assign: (1xf32) <- (1xf32)
        t1052 = t1040

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1053 = paddle._C_ops.scale(t1039, t1040, float("0"), True)

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1054 = paddle._C_ops.scale(t1053, t1040, float("0"), True)
        del t1053

        # pd_op.sigmoid: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32)
        t1055 = paddle._C_ops.sigmoid(t1003)

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1056 = paddle._C_ops.scale(t1055, t1040, float("0"), True)

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1057 = paddle._C_ops.scale(t1056, t1040, float("0"), True)
        del t1056

        # pd_op.bce_loss: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1058 = paddle._C_ops.bce_loss(t1054, t1030)

        # pd_op.bce_loss: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1059 = paddle._C_ops.bce_loss(t1057, t1031)

        # pd_op.add: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1060 = paddle._C_ops.add(t1058, t1059)

        # pd_op.multiply: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1061 = paddle._C_ops.multiply(t1038, t1060)

        # pd_op.full_int_array: (4xi64) <- ()
        t1062 = [1, 2, 3, 4]

        # pd_op.assign: (4xi64) <- (4xi64)
        t1063 = t1062

        # pd_op.assign: (4xi64) <- (4xi64)
        t1064 = t1062

        # pd_op.assign: (4xi64) <- (4xi64)
        t1065 = t1062

        # pd_op.assign: (4xi64) <- (4xi64)
        t1066 = t1062

        # pd_op.assign: (4xi64) <- (4xi64)
        t1067 = t1062

        # pd_op.assign: (4xi64) <- (4xi64)
        t1068 = t1062

        # pd_op.assign: (4xi64) <- (4xi64)
        t1069 = t1062

        # pd_op.assign: (4xi64) <- (4xi64)
        t1070 = t1062

        # pd_op.sum: (2xf32) <- (2x3x15x15x1xf32, 4xi64)
        t1071 = paddle._C_ops.sum(t1061, t1062, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1072 = paddle._C_ops.mean(t1071, [], False)

        # pd_op.subtract: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1073 = paddle._C_ops.subtract(t1010, t1032)

        # pd_op.abs: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32)
        t1074 = paddle._C_ops.abs(t1073)

        # pd_op.subtract: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1075 = paddle._C_ops.subtract(t1017, t1033)

        # pd_op.abs: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32)
        t1076 = paddle._C_ops.abs(t1075)

        # pd_op.add: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1077 = paddle._C_ops.add(t1074, t1076)

        # pd_op.multiply: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1078 = paddle._C_ops.multiply(t1038, t1077)

        # pd_op.sum: (2xf32) <- (2x3x15x15x1xf32, 4xi64)
        t1079 = paddle._C_ops.sum(t1078, t1062, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1080 = paddle._C_ops.mean(t1079, [], False)

        # pd_op.full: (1xf64) <- ()
        t1081 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        t1082 = paddle._C_ops.full(
            [1], float("15"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        t1083 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (15xi64) <- (1xf64, 1xf64, 1xf64)
        t1084 = paddle.arange(t1081, t1082, t1083, dtype="int64")
        del t1082

        # builtin.combine: ([15xi64, 15xi64]) <- (15xi64, 15xi64)
        t1085 = [t1084, t1084]
        del t1084

        # pd_op.meshgrid: ([15x15xi64, 15x15xi64]) <- ([15xi64, 15xi64])
        t1086 = paddle._C_ops.meshgrid(t1085)
        del t1085

        # builtin.split: (15x15xi64, 15x15xi64) <- ([15x15xi64, 15x15xi64])
        (
            t1087,
            t1088,
        ) = t1086
        del t1086

        # builtin.combine: ([15x15xi64, 15x15xi64]) <- (15x15xi64, 15x15xi64)
        t1089 = [t1088, t1087]
        del t1087, t1088

        # pd_op.stack: (15x15x2xi64) <- ([15x15xi64, 15x15xi64])
        t1090 = paddle._C_ops.stack(t1089, 2)
        del t1089

        # pd_op.cast: (15x15x2xf32) <- (15x15x2xi64)
        t1091 = paddle._C_ops.cast(t1090, paddle.float32)
        del t1090

        # pd_op.full_int_array: (5xi64) <- ()
        t1092 = [1, 1, 15, 15, 2]

        # pd_op.reshape: (1x1x15x15x2xf32) <- (15x15x2xf32, 5xi64)
        t1093 = paddle._C_ops.reshape(t1091, t1092)
        del t1091, t1092

        # pd_op.slice: (1x1x15x15x1xf32) <- (1x1x15x15x2xf32, 1xi64, 1xi64)
        t1094 = paddle._C_ops.slice(t1093, [4], t987, t990, [1], [])

        # pd_op.add: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1x1x15x15x1xf32)
        t1095 = paddle._C_ops.add(t1054, t1094)
        del t1094

        # pd_op.full: (1xf32) <- ()
        t1096 = paddle._C_ops.full(
            [1], float("0.0666667"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1097 = paddle._C_ops.scale(t1095, t1096, float("0"), True)
        del t1095

        # pd_op.slice: (1x1x15x15x1xf32) <- (1x1x15x15x2xf32, 1xi64, 1xi64)
        t1098 = paddle._C_ops.slice(t1093, [4], t990, t1025, [1], [])
        del t1093

        # pd_op.add: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1x1x15x15x1xf32)
        t1099 = paddle._C_ops.add(t1057, t1098)
        del t1098

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1100 = paddle._C_ops.scale(t1099, t1096, float("0"), True)
        del t1099, t1096

        # pd_op.full: (3x2xi64) <- ()
        t1101 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        t1102 = paddle._C_ops.assign_value_(
            t1101,
            [3, 2],
            paddle.int64,
            [
                float("116"),
                float("90"),
                float("156"),
                float("198"),
                float("373"),
                float("326"),
            ],
            paddle.framework._current_expected_place(),
        )
        del t1101

        # pd_op.cast: (3x2xf32) <- (3x2xi64)
        t1103 = paddle._C_ops.cast(t1102, paddle.float32)
        del t1102

        # pd_op.full_int_array: (5xi64) <- ()
        t1104 = [1, 3, 1, 1, 2]

        # pd_op.reshape: (1x3x1x1x2xf32) <- (3x2xf32, 5xi64)
        t1105 = paddle._C_ops.reshape(t1103, t1104)
        del t1103

        # pd_op.exp: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32)
        t1106 = paddle._C_ops.exp(t1010)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1107 = paddle._C_ops.slice(t1105, [4], t987, t990, [1], [])

        # pd_op.multiply: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1x3x1x1x1xf32)
        t1108 = paddle._C_ops.multiply(t1106, t1107)
        del t1106, t1107

        # pd_op.full: (1xf32) <- ()
        t1109 = paddle._C_ops.full(
            [1], float("0.00208333"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1110 = paddle._C_ops.scale(t1108, t1109, float("0"), True)
        del t1108

        # pd_op.exp: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32)
        t1111 = paddle._C_ops.exp(t1017)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1112 = paddle._C_ops.slice(t1105, [4], t990, t1025, [1], [])
        del t1105

        # pd_op.multiply: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1x3x1x1x1xf32)
        t1113 = paddle._C_ops.multiply(t1111, t1112)
        del t1111, t1112

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1114 = paddle._C_ops.scale(t1113, t1109, float("0"), True)
        del t1113

        # pd_op.full: (1xf32) <- ()
        t1115 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1116 = paddle._C_ops.scale(t1110, t1115, float("0"), True)

        # pd_op.subtract: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1117 = paddle._C_ops.subtract(t1097, t1116)
        del t1116

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1118 = paddle._C_ops.scale(t1114, t1115, float("0"), True)

        # pd_op.subtract: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1119 = paddle._C_ops.subtract(t1100, t1118)
        del t1118

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1120 = paddle._C_ops.scale(t1110, t1115, float("0"), True)
        del t1110

        # pd_op.add: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1121 = paddle._C_ops.add(t1097, t1120)
        del t1097, t1120

        # pd_op.scale: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 1xf32)
        t1122 = paddle._C_ops.scale(t1114, t1115, float("0"), True)
        del t1114

        # pd_op.add: (2x3x15x15x1xf32) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1123 = paddle._C_ops.add(t1100, t1122)
        del t1100, t1122

        # pd_op.full: (1xi32) <- ()
        t1124 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x3x15x15x1xf32, 2x3x15x15x1xf32, 2x3x15x15x1xf32, 2x3x15x15x1xf32]) <- (2x3x15x15x1xf32, 2x3x15x15x1xf32, 2x3x15x15x1xf32, 2x3x15x15x1xf32)
        t1125 = [t1117, t1119, t1121, t1123]
        del t1121, t1123, t1117, t1119

        # pd_op.concat: (2x3x15x15x4xf32) <- ([2x3x15x15x1xf32, 2x3x15x15x1xf32, 2x3x15x15x1xf32, 2x3x15x15x1xf32], 1xi32)
        t1126 = paddle._C_ops.concat(t1125, t1124)
        del t1125

        # pd_op.full_int_array: (3xi64) <- ()
        t1127 = [2, -1, 4]

        # pd_op.reshape: (2x675x4xf32) <- (2x3x15x15x4xf32, 3xi64)
        t1128 = paddle._C_ops.reshape(t1126, t1127)
        del t1126

        # pd_op.slice: (2x50x2xf32) <- (2x50x4xf32, 1xi64, 1xi64)
        t1129 = paddle._C_ops.slice(input2, [2], t987, t997, [1], [])

        # pd_op.slice: (2x50x2xf32) <- (2x50x4xf32, 1xi64, 1xi64)
        t1130 = paddle._C_ops.slice(input2, [2], t997, t1025, [1], [])
        del input2

        # pd_op.scale: (2x50x2xf32) <- (2x50x2xf32, 1xf32)
        t1131 = paddle._C_ops.scale(t1130, t1115, float("0"), True)
        del t1130

        # pd_op.subtract: (2x50x2xf32) <- (2x50x2xf32, 2x50x2xf32)
        t1132 = paddle._C_ops.subtract(t1129, t1131)

        # pd_op.add: (2x50x2xf32) <- (2x50x2xf32, 2x50x2xf32)
        t1133 = paddle._C_ops.add(t1129, t1131)
        del t1131, t1129

        # builtin.combine: ([2x50x2xf32, 2x50x2xf32]) <- (2x50x2xf32, 2x50x2xf32)
        t1134 = [t1132, t1133]
        del t1133, t1132

        # pd_op.concat: (2x50x4xf32) <- ([2x50x2xf32, 2x50x2xf32], 1xi32)
        t1135 = paddle._C_ops.concat(t1134, t1124)
        del t1134

        # pd_op.unsqueeze: (2x675x1x4xf32) <- (2x675x4xf32, 1xi64)
        t1136 = paddle._C_ops.unsqueeze(t1128, t997)
        del t1128

        # pd_op.unsqueeze: (2x1x50x4xf32) <- (2x50x4xf32, 1xi64)
        t1137 = paddle._C_ops.unsqueeze(t1135, t990)
        del t1135

        # pd_op.slice: (2x675x1x2xf32) <- (2x675x1x4xf32, 1xi64, 1xi64)
        t1138 = paddle._C_ops.slice(t1136, [3], t987, t997, [1], [])

        # pd_op.slice: (2x675x1x2xf32) <- (2x675x1x4xf32, 1xi64, 1xi64)
        t1139 = paddle._C_ops.slice(t1136, [3], t997, t1025, [1], [])
        del t1136

        # pd_op.slice: (2x1x50x2xf32) <- (2x1x50x4xf32, 1xi64, 1xi64)
        t1140 = paddle._C_ops.slice(t1137, [3], t987, t997, [1], [])

        # pd_op.slice: (2x1x50x2xf32) <- (2x1x50x4xf32, 1xi64, 1xi64)
        t1141 = paddle._C_ops.slice(t1137, [3], t997, t1025, [1], [])
        del t1137

        # pd_op.maximum: (2x675x50x2xf32) <- (2x675x1x2xf32, 2x1x50x2xf32)
        t1142 = paddle._C_ops.maximum(t1138, t1140)

        # pd_op.minimum: (2x675x50x2xf32) <- (2x675x1x2xf32, 2x1x50x2xf32)
        t1143 = paddle._C_ops.minimum(t1139, t1141)

        # pd_op.subtract: (2x675x50x2xf32) <- (2x675x50x2xf32, 2x675x50x2xf32)
        t1144 = paddle._C_ops.subtract(t1143, t1142)
        del t1142, t1143

        # pd_op.full: (1xf32) <- ()
        t1145 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t1146 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (2x675x50x2xf32) <- (2x675x50x2xf32, 1xf32, 1xf32)
        t1147 = paddle._C_ops.clip(t1144, t1145, t1146)
        del t1144

        # pd_op.full_int_array: (1xi64) <- ()
        t1148 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1149 = t1148

        # pd_op.assign: (1xi64) <- (1xi64)
        t1150 = t1148

        # pd_op.assign: (1xi64) <- (1xi64)
        t1151 = t1148

        # pd_op.prod: (2x675x50xf32) <- (2x675x50x2xf32, 1xi64)
        t1152 = paddle._C_ops.prod(t1147, t1148, False, False)
        del t1147

        # pd_op.subtract: (2x675x1x2xf32) <- (2x675x1x2xf32, 2x675x1x2xf32)
        t1153 = paddle._C_ops.subtract(t1139, t1138)
        del t1138, t1139

        # pd_op.clip: (2x675x1x2xf32) <- (2x675x1x2xf32, 1xf32, 1xf32)
        t1154 = paddle._C_ops.clip(t1153, t1145, t1146)
        del t1153

        # pd_op.prod: (2x675x1xf32) <- (2x675x1x2xf32, 1xi64)
        t1155 = paddle._C_ops.prod(t1154, t1148, False, False)
        del t1154

        # pd_op.subtract: (2x1x50x2xf32) <- (2x1x50x2xf32, 2x1x50x2xf32)
        t1156 = paddle._C_ops.subtract(t1141, t1140)

        # pd_op.clip: (2x1x50x2xf32) <- (2x1x50x2xf32, 1xf32, 1xf32)
        t1157 = paddle._C_ops.clip(t1156, t1145, t1146)
        del t1156

        # pd_op.prod: (2x1x50xf32) <- (2x1x50x2xf32, 1xi64)
        t1158 = paddle._C_ops.prod(t1157, t1148, False, False)
        del t1157

        # pd_op.add: (2x675x50xf32) <- (2x675x1xf32, 2x1x50xf32)
        t1159 = paddle._C_ops.add(t1155, t1158)
        del t1155

        # pd_op.subtract: (2x675x50xf32) <- (2x675x50xf32, 2x675x50xf32)
        t1160 = paddle._C_ops.subtract(t1159, t1152)
        del t1159

        # pd_op.scale: (2x675x50xf32) <- (2x675x50xf32, 1xf32)
        t1161 = paddle._C_ops.scale(t1160, t1040, float("1e-09"), True)
        del t1160

        # pd_op.divide: (2x675x50xf32) <- (2x675x50xf32, 2x675x50xf32)
        t1162 = paddle._C_ops.divide(t1152, t1161)
        del t1152, t1161

        # pd_op.max: (2x675xf32) <- (2x675x50xf32, 1xi64)
        t1163 = paddle._C_ops.max(t1162, t997, False)
        del t1162

        # pd_op.full: (xf32) <- ()
        t1164 = paddle._C_ops.full(
            [], float("0.5"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.less_equal: (2x675xb) <- (2x675xf32, xf32)
        t1165 = paddle._C_ops.less_equal(t1163, t1164)
        del t1163

        # pd_op.cast: (2x675xf32) <- (2x675xb)
        t1166 = paddle._C_ops.cast(t1165, paddle.float32)
        del t1165

        # pd_op.full_int_array: (2xi64) <- ()
        t1167 = [2, -1]

        # pd_op.reshape: (2x675xf32) <- (2x3x15x15x1xf32, 2xi64)
        t1168 = paddle._C_ops.reshape(t1024, t1167)

        # pd_op.reshape: (2x675xf32) <- (2x3x15x15x1xf32, 2xi64)
        t1169 = paddle._C_ops.reshape(t1036, t1167)

        # pd_op.full: (xf32) <- ()
        t1170 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (2x675xb) <- (2x675xf32, xf32)
        t1171 = paddle._C_ops.greater_than(t1169, t1170)

        # pd_op.cast: (2x675xf32) <- (2x675xb)
        t1172 = paddle._C_ops.cast(t1171, paddle.float32)
        del t1171

        # pd_op.full: (1xf32) <- ()
        t1173 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.sigmoid_cross_entropy_with_logits: (2x675xf32) <- (2x675xf32, 2x675xf32, None)
        t1174 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1168, t1172, None, False, -100
        )

        # pd_op.multiply: (2x675xf32) <- (2x675xf32, 2x675xf32)
        t1175 = paddle._C_ops.multiply(t1174, t1169)

        # pd_op.full: (1xf32) <- ()
        t1176 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x675xf32) <- (2x675xf32, 1xf32)
        t1177 = paddle._C_ops.scale(t1172, t1176, float("1"), True)

        # pd_op.multiply: (2x675xf32) <- (2x675xf32, 2x675xf32)
        t1178 = paddle._C_ops.multiply(t1174, t1177)

        # pd_op.multiply: (2x675xf32) <- (2x675xf32, 2x675xf32)
        t1179 = paddle._C_ops.multiply(t1178, t1166)

        # pd_op.add: (2x675xf32) <- (2x675xf32, 2x675xf32)
        t1180 = paddle._C_ops.add(t1175, t1179)

        # pd_op.sum: (2xf32) <- (2x675xf32, 1xi64)
        t1181 = paddle._C_ops.sum(t1180, t1148, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1182 = paddle._C_ops.mean(t1181, [], False)

        # pd_op.sigmoid_cross_entropy_with_logits: (2x3x15x15x4xf32) <- (2x3x15x15x4xf32, 2x3x15x15x4xf32, None)
        t1183 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1028, t1037, None, False, -100
        )

        # pd_op.multiply: (2x3x15x15x4xf32) <- (2x3x15x15x4xf32, 2x3x15x15x1xf32)
        t1184 = paddle._C_ops.multiply(t1183, t1036)

        # pd_op.sum: (2xf32) <- (2x3x15x15x4xf32, 4xi64)
        t1185 = paddle._C_ops.sum(t1184, t1062, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1186 = paddle._C_ops.mean(t1185, [], False)

        # pd_op.full_int_array: (5xi64) <- ()
        t1187 = [2, 3, -1, 30, 30]

        # pd_op.reshape: (2x3x9x30x30xf32) <- (2x27x30x30xf32, 5xi64)
        t1188 = paddle._C_ops.reshape(t980, t1187)
        del t1187

        # pd_op.transpose: (2x3x30x30x9xf32) <- (2x3x9x30x30xf32)
        t1189 = paddle._C_ops.transpose(t1188, [0, 1, 3, 4, 2])
        del t1188

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x9xf32, 1xi64, 1xi64)
        t1190 = paddle._C_ops.slice(t1189, [4], t987, t990, [1], [])

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x9xf32, 1xi64, 1xi64)
        t1191 = paddle._C_ops.slice(t1189, [4], t990, t997, [1], [])

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x9xf32, 1xi64, 1xi64)
        t1192 = paddle._C_ops.slice(t1189, [4], t997, t1004, [1], [])

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x9xf32, 1xi64, 1xi64)
        t1193 = paddle._C_ops.slice(t1189, [4], t1004, t1011, [1], [])

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x9xf32, 1xi64, 1xi64)
        t1194 = paddle._C_ops.slice(t1189, [4], t1011, t1018, [1], [])

        # pd_op.slice: (2x3x30x30x4xf32) <- (2x3x30x30x9xf32, 1xi64, 1xi64)
        t1195 = paddle._C_ops.slice(t1189, [4], t1018, t1025, [1], [])

        # pd_op.transpose: (2x3x30x30x10xf32) <- (2x3x10x30x30xf32)
        t1196 = paddle._C_ops.transpose(input3, [0, 1, 3, 4, 2])
        del input3

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x10xf32, 1xi64, 1xi64)
        t1197 = paddle._C_ops.slice(t1196, [4], t987, t990, [1], [])

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x10xf32, 1xi64, 1xi64)
        t1198 = paddle._C_ops.slice(t1196, [4], t990, t997, [1], [])

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x10xf32, 1xi64, 1xi64)
        t1199 = paddle._C_ops.slice(t1196, [4], t997, t1004, [1], [])

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x10xf32, 1xi64, 1xi64)
        t1200 = paddle._C_ops.slice(t1196, [4], t1004, t1011, [1], [])

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x10xf32, 1xi64, 1xi64)
        t1201 = paddle._C_ops.slice(t1196, [4], t1011, t1018, [1], [])

        # pd_op.slice: (2x3x30x30x1xf32) <- (2x3x30x30x10xf32, 1xi64, 1xi64)
        t1202 = paddle._C_ops.slice(t1196, [4], t1018, t1035, [1], [])

        # pd_op.slice: (2x3x30x30x4xf32) <- (2x3x30x30x10xf32, 1xi64, 1xi64)
        t1203 = paddle._C_ops.slice(t1196, [4], t1035, t1025, [1], [])
        del t1196

        # pd_op.multiply: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1204 = paddle._C_ops.multiply(t1201, t1202)
        del t1201

        # pd_op.sigmoid: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32)
        t1205 = paddle._C_ops.sigmoid(t1190)

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1206 = paddle._C_ops.scale(t1205, t1040, float("0"), True)

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1207 = paddle._C_ops.scale(t1206, t1040, float("0"), True)
        del t1206

        # pd_op.sigmoid: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32)
        t1208 = paddle._C_ops.sigmoid(t1191)

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1209 = paddle._C_ops.scale(t1208, t1040, float("0"), True)

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1210 = paddle._C_ops.scale(t1209, t1040, float("0"), True)
        del t1209

        # pd_op.bce_loss: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1211 = paddle._C_ops.bce_loss(t1207, t1197)

        # pd_op.bce_loss: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1212 = paddle._C_ops.bce_loss(t1210, t1198)

        # pd_op.add: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1213 = paddle._C_ops.add(t1211, t1212)

        # pd_op.multiply: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1214 = paddle._C_ops.multiply(t1204, t1213)

        # pd_op.sum: (2xf32) <- (2x3x30x30x1xf32, 4xi64)
        t1215 = paddle._C_ops.sum(t1214, t1062, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1216 = paddle._C_ops.mean(t1215, [], False)

        # pd_op.subtract: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1217 = paddle._C_ops.subtract(t1192, t1199)

        # pd_op.abs: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32)
        t1218 = paddle._C_ops.abs(t1217)

        # pd_op.subtract: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1219 = paddle._C_ops.subtract(t1193, t1200)

        # pd_op.abs: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32)
        t1220 = paddle._C_ops.abs(t1219)

        # pd_op.add: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1221 = paddle._C_ops.add(t1218, t1220)

        # pd_op.multiply: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1222 = paddle._C_ops.multiply(t1204, t1221)

        # pd_op.sum: (2xf32) <- (2x3x30x30x1xf32, 4xi64)
        t1223 = paddle._C_ops.sum(t1222, t1062, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1224 = paddle._C_ops.mean(t1223, [], False)

        # pd_op.full: (1xf64) <- ()
        t1225 = paddle._C_ops.full(
            [1], float("30"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (30xi64) <- (1xf64, 1xf64, 1xf64)
        t1226 = paddle.arange(t1081, t1225, t1083, dtype="int64")
        del t1225

        # builtin.combine: ([30xi64, 30xi64]) <- (30xi64, 30xi64)
        t1227 = [t1226, t1226]
        del t1226

        # pd_op.meshgrid: ([30x30xi64, 30x30xi64]) <- ([30xi64, 30xi64])
        t1228 = paddle._C_ops.meshgrid(t1227)
        del t1227

        # builtin.split: (30x30xi64, 30x30xi64) <- ([30x30xi64, 30x30xi64])
        (
            t1229,
            t1230,
        ) = t1228
        del t1228

        # builtin.combine: ([30x30xi64, 30x30xi64]) <- (30x30xi64, 30x30xi64)
        t1231 = [t1230, t1229]
        del t1229, t1230

        # pd_op.stack: (30x30x2xi64) <- ([30x30xi64, 30x30xi64])
        t1232 = paddle._C_ops.stack(t1231, 2)
        del t1231

        # pd_op.cast: (30x30x2xf32) <- (30x30x2xi64)
        t1233 = paddle._C_ops.cast(t1232, paddle.float32)
        del t1232

        # pd_op.full_int_array: (5xi64) <- ()
        t1234 = [1, 1, 30, 30, 2]

        # pd_op.reshape: (1x1x30x30x2xf32) <- (30x30x2xf32, 5xi64)
        t1235 = paddle._C_ops.reshape(t1233, t1234)
        del t1233, t1234

        # pd_op.slice: (1x1x30x30x1xf32) <- (1x1x30x30x2xf32, 1xi64, 1xi64)
        t1236 = paddle._C_ops.slice(t1235, [4], t987, t990, [1], [])

        # pd_op.add: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1x1x30x30x1xf32)
        t1237 = paddle._C_ops.add(t1207, t1236)
        del t1236

        # pd_op.full: (1xf32) <- ()
        t1238 = paddle._C_ops.full(
            [1], float("0.0333333"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1239 = paddle._C_ops.scale(t1237, t1238, float("0"), True)
        del t1237

        # pd_op.slice: (1x1x30x30x1xf32) <- (1x1x30x30x2xf32, 1xi64, 1xi64)
        t1240 = paddle._C_ops.slice(t1235, [4], t990, t1025, [1], [])
        del t1235

        # pd_op.add: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1x1x30x30x1xf32)
        t1241 = paddle._C_ops.add(t1210, t1240)
        del t1240

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1242 = paddle._C_ops.scale(t1241, t1238, float("0"), True)
        del t1241, t1238

        # pd_op.full: (3x2xi64) <- ()
        t1243 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        t1244 = paddle._C_ops.assign_value_(
            t1243,
            [3, 2],
            paddle.int64,
            [
                float("30"),
                float("61"),
                float("62"),
                float("45"),
                float("59"),
                float("119"),
            ],
            paddle.framework._current_expected_place(),
        )
        del t1243

        # pd_op.cast: (3x2xf32) <- (3x2xi64)
        t1245 = paddle._C_ops.cast(t1244, paddle.float32)
        del t1244

        # pd_op.reshape: (1x3x1x1x2xf32) <- (3x2xf32, 5xi64)
        t1246 = paddle._C_ops.reshape(t1245, t1104)
        del t1245

        # pd_op.exp: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32)
        t1247 = paddle._C_ops.exp(t1192)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1248 = paddle._C_ops.slice(t1246, [4], t987, t990, [1], [])

        # pd_op.multiply: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1x3x1x1x1xf32)
        t1249 = paddle._C_ops.multiply(t1247, t1248)
        del t1247, t1248

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1250 = paddle._C_ops.scale(t1249, t1109, float("0"), True)
        del t1249

        # pd_op.exp: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32)
        t1251 = paddle._C_ops.exp(t1193)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1252 = paddle._C_ops.slice(t1246, [4], t990, t1025, [1], [])
        del t1246

        # pd_op.multiply: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1x3x1x1x1xf32)
        t1253 = paddle._C_ops.multiply(t1251, t1252)
        del t1251, t1252

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1254 = paddle._C_ops.scale(t1253, t1109, float("0"), True)
        del t1253

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1255 = paddle._C_ops.scale(t1250, t1115, float("0"), True)

        # pd_op.subtract: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1256 = paddle._C_ops.subtract(t1239, t1255)
        del t1255

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1257 = paddle._C_ops.scale(t1254, t1115, float("0"), True)

        # pd_op.subtract: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1258 = paddle._C_ops.subtract(t1242, t1257)
        del t1257

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1259 = paddle._C_ops.scale(t1250, t1115, float("0"), True)
        del t1250

        # pd_op.add: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1260 = paddle._C_ops.add(t1239, t1259)
        del t1239, t1259

        # pd_op.scale: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 1xf32)
        t1261 = paddle._C_ops.scale(t1254, t1115, float("0"), True)
        del t1254

        # pd_op.add: (2x3x30x30x1xf32) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1262 = paddle._C_ops.add(t1242, t1261)
        del t1242, t1261

        # builtin.combine: ([2x3x30x30x1xf32, 2x3x30x30x1xf32, 2x3x30x30x1xf32, 2x3x30x30x1xf32]) <- (2x3x30x30x1xf32, 2x3x30x30x1xf32, 2x3x30x30x1xf32, 2x3x30x30x1xf32)
        t1263 = [t1256, t1258, t1260, t1262]
        del t1260, t1262, t1256, t1258

        # pd_op.concat: (2x3x30x30x4xf32) <- ([2x3x30x30x1xf32, 2x3x30x30x1xf32, 2x3x30x30x1xf32, 2x3x30x30x1xf32], 1xi32)
        t1264 = paddle._C_ops.concat(t1263, t1124)
        del t1263

        # pd_op.reshape: (2x2700x4xf32) <- (2x3x30x30x4xf32, 3xi64)
        t1265 = paddle._C_ops.reshape(t1264, t1127)
        del t1264

        # pd_op.unsqueeze: (2x2700x1x4xf32) <- (2x2700x4xf32, 1xi64)
        t1266 = paddle._C_ops.unsqueeze(t1265, t997)
        del t1265

        # pd_op.slice: (2x2700x1x2xf32) <- (2x2700x1x4xf32, 1xi64, 1xi64)
        t1267 = paddle._C_ops.slice(t1266, [3], t987, t997, [1], [])

        # pd_op.slice: (2x2700x1x2xf32) <- (2x2700x1x4xf32, 1xi64, 1xi64)
        t1268 = paddle._C_ops.slice(t1266, [3], t997, t1025, [1], [])
        del t1266

        # pd_op.maximum: (2x2700x50x2xf32) <- (2x2700x1x2xf32, 2x1x50x2xf32)
        t1269 = paddle._C_ops.maximum(t1267, t1140)

        # pd_op.minimum: (2x2700x50x2xf32) <- (2x2700x1x2xf32, 2x1x50x2xf32)
        t1270 = paddle._C_ops.minimum(t1268, t1141)

        # pd_op.subtract: (2x2700x50x2xf32) <- (2x2700x50x2xf32, 2x2700x50x2xf32)
        t1271 = paddle._C_ops.subtract(t1270, t1269)
        del t1269, t1270

        # pd_op.clip: (2x2700x50x2xf32) <- (2x2700x50x2xf32, 1xf32, 1xf32)
        t1272 = paddle._C_ops.clip(t1271, t1145, t1146)
        del t1271

        # pd_op.prod: (2x2700x50xf32) <- (2x2700x50x2xf32, 1xi64)
        t1273 = paddle._C_ops.prod(t1272, t1148, False, False)
        del t1272

        # pd_op.subtract: (2x2700x1x2xf32) <- (2x2700x1x2xf32, 2x2700x1x2xf32)
        t1274 = paddle._C_ops.subtract(t1268, t1267)
        del t1267, t1268

        # pd_op.clip: (2x2700x1x2xf32) <- (2x2700x1x2xf32, 1xf32, 1xf32)
        t1275 = paddle._C_ops.clip(t1274, t1145, t1146)
        del t1274

        # pd_op.prod: (2x2700x1xf32) <- (2x2700x1x2xf32, 1xi64)
        t1276 = paddle._C_ops.prod(t1275, t1148, False, False)
        del t1275

        # pd_op.add: (2x2700x50xf32) <- (2x2700x1xf32, 2x1x50xf32)
        t1277 = paddle._C_ops.add(t1276, t1158)
        del t1276

        # pd_op.subtract: (2x2700x50xf32) <- (2x2700x50xf32, 2x2700x50xf32)
        t1278 = paddle._C_ops.subtract(t1277, t1273)
        del t1277

        # pd_op.scale: (2x2700x50xf32) <- (2x2700x50xf32, 1xf32)
        t1279 = paddle._C_ops.scale(t1278, t1040, float("1e-09"), True)
        del t1278

        # pd_op.divide: (2x2700x50xf32) <- (2x2700x50xf32, 2x2700x50xf32)
        t1280 = paddle._C_ops.divide(t1273, t1279)
        del t1273, t1279

        # pd_op.max: (2x2700xf32) <- (2x2700x50xf32, 1xi64)
        t1281 = paddle._C_ops.max(t1280, t997, False)
        del t1280

        # pd_op.less_equal: (2x2700xb) <- (2x2700xf32, xf32)
        t1282 = paddle._C_ops.less_equal(t1281, t1164)
        del t1281

        # pd_op.cast: (2x2700xf32) <- (2x2700xb)
        t1283 = paddle._C_ops.cast(t1282, paddle.float32)
        del t1282

        # pd_op.reshape: (2x2700xf32) <- (2x3x30x30x1xf32, 2xi64)
        t1284 = paddle._C_ops.reshape(t1194, t1167)

        # pd_op.reshape: (2x2700xf32) <- (2x3x30x30x1xf32, 2xi64)
        t1285 = paddle._C_ops.reshape(t1202, t1167)

        # pd_op.greater_than: (2x2700xb) <- (2x2700xf32, xf32)
        t1286 = paddle._C_ops.greater_than(t1285, t1170)

        # pd_op.cast: (2x2700xf32) <- (2x2700xb)
        t1287 = paddle._C_ops.cast(t1286, paddle.float32)
        del t1286

        # pd_op.sigmoid_cross_entropy_with_logits: (2x2700xf32) <- (2x2700xf32, 2x2700xf32, None)
        t1288 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1284, t1287, None, False, -100
        )

        # pd_op.multiply: (2x2700xf32) <- (2x2700xf32, 2x2700xf32)
        t1289 = paddle._C_ops.multiply(t1288, t1285)

        # pd_op.scale: (2x2700xf32) <- (2x2700xf32, 1xf32)
        t1290 = paddle._C_ops.scale(t1287, t1176, float("1"), True)

        # pd_op.multiply: (2x2700xf32) <- (2x2700xf32, 2x2700xf32)
        t1291 = paddle._C_ops.multiply(t1288, t1290)

        # pd_op.multiply: (2x2700xf32) <- (2x2700xf32, 2x2700xf32)
        t1292 = paddle._C_ops.multiply(t1291, t1283)

        # pd_op.add: (2x2700xf32) <- (2x2700xf32, 2x2700xf32)
        t1293 = paddle._C_ops.add(t1289, t1292)

        # pd_op.sum: (2xf32) <- (2x2700xf32, 1xi64)
        t1294 = paddle._C_ops.sum(t1293, t1148, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1295 = paddle._C_ops.mean(t1294, [], False)

        # pd_op.sigmoid_cross_entropy_with_logits: (2x3x30x30x4xf32) <- (2x3x30x30x4xf32, 2x3x30x30x4xf32, None)
        t1296 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1195, t1203, None, False, -100
        )

        # pd_op.multiply: (2x3x30x30x4xf32) <- (2x3x30x30x4xf32, 2x3x30x30x1xf32)
        t1297 = paddle._C_ops.multiply(t1296, t1202)

        # pd_op.sum: (2xf32) <- (2x3x30x30x4xf32, 4xi64)
        t1298 = paddle._C_ops.sum(t1297, t1062, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1299 = paddle._C_ops.mean(t1298, [], False)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1300 = paddle._C_ops.add(t1072, t1216)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1301 = paddle._C_ops.add(t1080, t1224)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1302 = paddle._C_ops.add(t1182, t1295)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1303 = paddle._C_ops.add(t1186, t1299)

        # pd_op.full_int_array: (5xi64) <- ()
        t1304 = [2, 3, -1, 60, 60]

        # pd_op.reshape: (2x3x9x60x60xf32) <- (2x27x60x60xf32, 5xi64)
        t1305 = paddle._C_ops.reshape(t983, t1304)
        del t1304

        # pd_op.transpose: (2x3x60x60x9xf32) <- (2x3x9x60x60xf32)
        t1306 = paddle._C_ops.transpose(t1305, [0, 1, 3, 4, 2])
        del t1305

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x9xf32, 1xi64, 1xi64)
        t1307 = paddle._C_ops.slice(t1306, [4], t987, t990, [1], [])

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x9xf32, 1xi64, 1xi64)
        t1308 = paddle._C_ops.slice(t1306, [4], t990, t997, [1], [])

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x9xf32, 1xi64, 1xi64)
        t1309 = paddle._C_ops.slice(t1306, [4], t997, t1004, [1], [])

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x9xf32, 1xi64, 1xi64)
        t1310 = paddle._C_ops.slice(t1306, [4], t1004, t1011, [1], [])

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x9xf32, 1xi64, 1xi64)
        t1311 = paddle._C_ops.slice(t1306, [4], t1011, t1018, [1], [])

        # pd_op.slice: (2x3x60x60x4xf32) <- (2x3x60x60x9xf32, 1xi64, 1xi64)
        t1312 = paddle._C_ops.slice(t1306, [4], t1018, t1025, [1], [])

        # pd_op.transpose: (2x3x60x60x10xf32) <- (2x3x10x60x60xf32)
        t1313 = paddle._C_ops.transpose(input4, [0, 1, 3, 4, 2])
        del input4

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x10xf32, 1xi64, 1xi64)
        t1314 = paddle._C_ops.slice(t1313, [4], t987, t990, [1], [])

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x10xf32, 1xi64, 1xi64)
        t1315 = paddle._C_ops.slice(t1313, [4], t990, t997, [1], [])

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x10xf32, 1xi64, 1xi64)
        t1316 = paddle._C_ops.slice(t1313, [4], t997, t1004, [1], [])

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x10xf32, 1xi64, 1xi64)
        t1317 = paddle._C_ops.slice(t1313, [4], t1004, t1011, [1], [])

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x10xf32, 1xi64, 1xi64)
        t1318 = paddle._C_ops.slice(t1313, [4], t1011, t1018, [1], [])

        # pd_op.slice: (2x3x60x60x1xf32) <- (2x3x60x60x10xf32, 1xi64, 1xi64)
        t1319 = paddle._C_ops.slice(t1313, [4], t1018, t1035, [1], [])

        # pd_op.slice: (2x3x60x60x4xf32) <- (2x3x60x60x10xf32, 1xi64, 1xi64)
        t1320 = paddle._C_ops.slice(t1313, [4], t1035, t1025, [1], [])
        del t1035, t1313

        # pd_op.multiply: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1321 = paddle._C_ops.multiply(t1318, t1319)
        del t1318

        # pd_op.sigmoid: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32)
        t1322 = paddle._C_ops.sigmoid(t1307)

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1323 = paddle._C_ops.scale(t1322, t1040, float("0"), True)

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1324 = paddle._C_ops.scale(t1323, t1040, float("0"), True)
        del t1323

        # pd_op.sigmoid: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32)
        t1325 = paddle._C_ops.sigmoid(t1308)

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1326 = paddle._C_ops.scale(t1325, t1040, float("0"), True)

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1327 = paddle._C_ops.scale(t1326, t1040, float("0"), True)
        del t1326

        # pd_op.bce_loss: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1328 = paddle._C_ops.bce_loss(t1324, t1314)

        # pd_op.bce_loss: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1329 = paddle._C_ops.bce_loss(t1327, t1315)

        # pd_op.add: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1330 = paddle._C_ops.add(t1328, t1329)

        # pd_op.multiply: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1331 = paddle._C_ops.multiply(t1321, t1330)

        # pd_op.sum: (2xf32) <- (2x3x60x60x1xf32, 4xi64)
        t1332 = paddle._C_ops.sum(t1331, t1062, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1333 = paddle._C_ops.mean(t1332, [], False)

        # pd_op.subtract: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1334 = paddle._C_ops.subtract(t1309, t1316)

        # pd_op.abs: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32)
        t1335 = paddle._C_ops.abs(t1334)

        # pd_op.subtract: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1336 = paddle._C_ops.subtract(t1310, t1317)

        # pd_op.abs: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32)
        t1337 = paddle._C_ops.abs(t1336)

        # pd_op.add: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1338 = paddle._C_ops.add(t1335, t1337)

        # pd_op.multiply: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1339 = paddle._C_ops.multiply(t1321, t1338)

        # pd_op.sum: (2xf32) <- (2x3x60x60x1xf32, 4xi64)
        t1340 = paddle._C_ops.sum(t1339, t1062, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1341 = paddle._C_ops.mean(t1340, [], False)

        # pd_op.full: (1xf64) <- ()
        t1342 = paddle._C_ops.full(
            [1], float("60"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (60xi64) <- (1xf64, 1xf64, 1xf64)
        t1343 = paddle.arange(t1081, t1342, t1083, dtype="int64")
        del t1342, t1081, t1083

        # builtin.combine: ([60xi64, 60xi64]) <- (60xi64, 60xi64)
        t1344 = [t1343, t1343]
        del t1343

        # pd_op.meshgrid: ([60x60xi64, 60x60xi64]) <- ([60xi64, 60xi64])
        t1345 = paddle._C_ops.meshgrid(t1344)
        del t1344

        # builtin.split: (60x60xi64, 60x60xi64) <- ([60x60xi64, 60x60xi64])
        (
            t1346,
            t1347,
        ) = t1345
        del t1345

        # builtin.combine: ([60x60xi64, 60x60xi64]) <- (60x60xi64, 60x60xi64)
        t1348 = [t1347, t1346]
        del t1346, t1347

        # pd_op.stack: (60x60x2xi64) <- ([60x60xi64, 60x60xi64])
        t1349 = paddle._C_ops.stack(t1348, 2)
        del t1348

        # pd_op.cast: (60x60x2xf32) <- (60x60x2xi64)
        t1350 = paddle._C_ops.cast(t1349, paddle.float32)
        del t1349

        # pd_op.full_int_array: (5xi64) <- ()
        t1351 = [1, 1, 60, 60, 2]

        # pd_op.reshape: (1x1x60x60x2xf32) <- (60x60x2xf32, 5xi64)
        t1352 = paddle._C_ops.reshape(t1350, t1351)
        del t1350, t1351

        # pd_op.slice: (1x1x60x60x1xf32) <- (1x1x60x60x2xf32, 1xi64, 1xi64)
        t1353 = paddle._C_ops.slice(t1352, [4], t987, t990, [1], [])

        # pd_op.add: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1x1x60x60x1xf32)
        t1354 = paddle._C_ops.add(t1324, t1353)
        del t1353

        # pd_op.full: (1xf32) <- ()
        t1355 = paddle._C_ops.full(
            [1], float("0.0166667"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1356 = paddle._C_ops.scale(t1354, t1355, float("0"), True)
        del t1354

        # pd_op.slice: (1x1x60x60x1xf32) <- (1x1x60x60x2xf32, 1xi64, 1xi64)
        t1357 = paddle._C_ops.slice(t1352, [4], t990, t1025, [1], [])
        del t1352

        # pd_op.add: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1x1x60x60x1xf32)
        t1358 = paddle._C_ops.add(t1327, t1357)
        del t1357

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1359 = paddle._C_ops.scale(t1358, t1355, float("0"), True)
        del t1358, t1355

        # pd_op.full: (3x2xi64) <- ()
        t1360 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        t1361 = paddle._C_ops.assign_value_(
            t1360,
            [3, 2],
            paddle.int64,
            [
                float("10"),
                float("13"),
                float("16"),
                float("30"),
                float("33"),
                float("23"),
            ],
            paddle.framework._current_expected_place(),
        )
        del t1360

        # pd_op.cast: (3x2xf32) <- (3x2xi64)
        t1362 = paddle._C_ops.cast(t1361, paddle.float32)
        del t1361

        # pd_op.reshape: (1x3x1x1x2xf32) <- (3x2xf32, 5xi64)
        t1363 = paddle._C_ops.reshape(t1362, t1104)
        del t1362, t1104

        # pd_op.exp: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32)
        t1364 = paddle._C_ops.exp(t1309)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1365 = paddle._C_ops.slice(t1363, [4], t987, t990, [1], [])

        # pd_op.multiply: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1x3x1x1x1xf32)
        t1366 = paddle._C_ops.multiply(t1364, t1365)
        del t1364, t1365

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1367 = paddle._C_ops.scale(t1366, t1109, float("0"), True)
        del t1366

        # pd_op.exp: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32)
        t1368 = paddle._C_ops.exp(t1310)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1369 = paddle._C_ops.slice(t1363, [4], t990, t1025, [1], [])
        del t1363

        # pd_op.multiply: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1x3x1x1x1xf32)
        t1370 = paddle._C_ops.multiply(t1368, t1369)
        del t1368, t1369

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1371 = paddle._C_ops.scale(t1370, t1109, float("0"), True)
        del t1109, t1370

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1372 = paddle._C_ops.scale(t1367, t1115, float("0"), True)

        # pd_op.subtract: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1373 = paddle._C_ops.subtract(t1356, t1372)
        del t1372

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1374 = paddle._C_ops.scale(t1371, t1115, float("0"), True)

        # pd_op.subtract: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1375 = paddle._C_ops.subtract(t1359, t1374)
        del t1374

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1376 = paddle._C_ops.scale(t1367, t1115, float("0"), True)
        del t1367

        # pd_op.add: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1377 = paddle._C_ops.add(t1356, t1376)
        del t1356, t1376

        # pd_op.scale: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 1xf32)
        t1378 = paddle._C_ops.scale(t1371, t1115, float("0"), True)
        del t1115, t1371

        # pd_op.add: (2x3x60x60x1xf32) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1379 = paddle._C_ops.add(t1359, t1378)
        del t1359, t1378

        # builtin.combine: ([2x3x60x60x1xf32, 2x3x60x60x1xf32, 2x3x60x60x1xf32, 2x3x60x60x1xf32]) <- (2x3x60x60x1xf32, 2x3x60x60x1xf32, 2x3x60x60x1xf32, 2x3x60x60x1xf32)
        t1380 = [t1373, t1375, t1377, t1379]
        del t1377, t1379, t1373, t1375

        # pd_op.concat: (2x3x60x60x4xf32) <- ([2x3x60x60x1xf32, 2x3x60x60x1xf32, 2x3x60x60x1xf32, 2x3x60x60x1xf32], 1xi32)
        t1381 = paddle._C_ops.concat(t1380, t1124)
        del t1380, t1124

        # pd_op.reshape: (2x10800x4xf32) <- (2x3x60x60x4xf32, 3xi64)
        t1382 = paddle._C_ops.reshape(t1381, t1127)
        del t1381, t1127

        # pd_op.unsqueeze: (2x10800x1x4xf32) <- (2x10800x4xf32, 1xi64)
        t1383 = paddle._C_ops.unsqueeze(t1382, t997)
        del t1382

        # pd_op.slice: (2x10800x1x2xf32) <- (2x10800x1x4xf32, 1xi64, 1xi64)
        t1384 = paddle._C_ops.slice(t1383, [3], t987, t997, [1], [])

        # pd_op.slice: (2x10800x1x2xf32) <- (2x10800x1x4xf32, 1xi64, 1xi64)
        t1385 = paddle._C_ops.slice(t1383, [3], t997, t1025, [1], [])
        del t1383

        # pd_op.maximum: (2x10800x50x2xf32) <- (2x10800x1x2xf32, 2x1x50x2xf32)
        t1386 = paddle._C_ops.maximum(t1384, t1140)
        del t1140

        # pd_op.minimum: (2x10800x50x2xf32) <- (2x10800x1x2xf32, 2x1x50x2xf32)
        t1387 = paddle._C_ops.minimum(t1385, t1141)
        del t1141

        # pd_op.subtract: (2x10800x50x2xf32) <- (2x10800x50x2xf32, 2x10800x50x2xf32)
        t1388 = paddle._C_ops.subtract(t1387, t1386)
        del t1386, t1387

        # pd_op.clip: (2x10800x50x2xf32) <- (2x10800x50x2xf32, 1xf32, 1xf32)
        t1389 = paddle._C_ops.clip(t1388, t1145, t1146)
        del t1388

        # pd_op.prod: (2x10800x50xf32) <- (2x10800x50x2xf32, 1xi64)
        t1390 = paddle._C_ops.prod(t1389, t1148, False, False)
        del t1389

        # pd_op.subtract: (2x10800x1x2xf32) <- (2x10800x1x2xf32, 2x10800x1x2xf32)
        t1391 = paddle._C_ops.subtract(t1385, t1384)
        del t1384, t1385

        # pd_op.clip: (2x10800x1x2xf32) <- (2x10800x1x2xf32, 1xf32, 1xf32)
        t1392 = paddle._C_ops.clip(t1391, t1145, t1146)
        del t1145, t1146, t1391

        # pd_op.prod: (2x10800x1xf32) <- (2x10800x1x2xf32, 1xi64)
        t1393 = paddle._C_ops.prod(t1392, t1148, False, False)
        del t1392

        # pd_op.add: (2x10800x50xf32) <- (2x10800x1xf32, 2x1x50xf32)
        t1394 = paddle._C_ops.add(t1393, t1158)
        del t1158, t1393

        # pd_op.subtract: (2x10800x50xf32) <- (2x10800x50xf32, 2x10800x50xf32)
        t1395 = paddle._C_ops.subtract(t1394, t1390)
        del t1394

        # pd_op.scale: (2x10800x50xf32) <- (2x10800x50xf32, 1xf32)
        t1396 = paddle._C_ops.scale(t1395, t1040, float("1e-09"), True)
        del t1395

        # pd_op.divide: (2x10800x50xf32) <- (2x10800x50xf32, 2x10800x50xf32)
        t1397 = paddle._C_ops.divide(t1390, t1396)
        del t1390, t1396

        # pd_op.max: (2x10800xf32) <- (2x10800x50xf32, 1xi64)
        t1398 = paddle._C_ops.max(t1397, t997, False)
        del t1397

        # pd_op.less_equal: (2x10800xb) <- (2x10800xf32, xf32)
        t1399 = paddle._C_ops.less_equal(t1398, t1164)
        del t1164, t1398

        # pd_op.cast: (2x10800xf32) <- (2x10800xb)
        t1400 = paddle._C_ops.cast(t1399, paddle.float32)
        del t1399

        # pd_op.reshape: (2x10800xf32) <- (2x3x60x60x1xf32, 2xi64)
        t1401 = paddle._C_ops.reshape(t1311, t1167)

        # pd_op.reshape: (2x10800xf32) <- (2x3x60x60x1xf32, 2xi64)
        t1402 = paddle._C_ops.reshape(t1319, t1167)
        del t1167

        # pd_op.greater_than: (2x10800xb) <- (2x10800xf32, xf32)
        t1403 = paddle._C_ops.greater_than(t1402, t1170)
        del t1170

        # pd_op.cast: (2x10800xf32) <- (2x10800xb)
        t1404 = paddle._C_ops.cast(t1403, paddle.float32)
        del t1403

        # pd_op.sigmoid_cross_entropy_with_logits: (2x10800xf32) <- (2x10800xf32, 2x10800xf32, None)
        t1405 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1401, t1404, None, False, -100
        )

        # pd_op.multiply: (2x10800xf32) <- (2x10800xf32, 2x10800xf32)
        t1406 = paddle._C_ops.multiply(t1405, t1402)

        # pd_op.scale: (2x10800xf32) <- (2x10800xf32, 1xf32)
        t1407 = paddle._C_ops.scale(t1404, t1176, float("1"), True)
        del t1176

        # pd_op.multiply: (2x10800xf32) <- (2x10800xf32, 2x10800xf32)
        t1408 = paddle._C_ops.multiply(t1405, t1407)

        # pd_op.multiply: (2x10800xf32) <- (2x10800xf32, 2x10800xf32)
        t1409 = paddle._C_ops.multiply(t1408, t1400)

        # pd_op.add: (2x10800xf32) <- (2x10800xf32, 2x10800xf32)
        t1410 = paddle._C_ops.add(t1406, t1409)

        # pd_op.sum: (2xf32) <- (2x10800xf32, 1xi64)
        t1411 = paddle._C_ops.sum(t1410, t1148, None, False)
        del t1148

        # pd_op.mean: (xf32) <- (2xf32)
        t1412 = paddle._C_ops.mean(t1411, [], False)

        # pd_op.sigmoid_cross_entropy_with_logits: (2x3x60x60x4xf32) <- (2x3x60x60x4xf32, 2x3x60x60x4xf32, None)
        t1413 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1312, t1320, None, False, -100
        )

        # pd_op.multiply: (2x3x60x60x4xf32) <- (2x3x60x60x4xf32, 2x3x60x60x1xf32)
        t1414 = paddle._C_ops.multiply(t1413, t1319)

        # pd_op.sum: (2xf32) <- (2x3x60x60x4xf32, 4xi64)
        t1415 = paddle._C_ops.sum(t1414, t1062, None, False)

        # pd_op.mean: (xf32) <- (2xf32)
        t1416 = paddle._C_ops.mean(t1415, [], False)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1417 = paddle._C_ops.add(t1300, t1333)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1418 = paddle._C_ops.add(t1301, t1341)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1419 = paddle._C_ops.add(t1302, t1412)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1420 = paddle._C_ops.add(t1303, t1416)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t1421 = paddle._C_ops.scale(t1417, t1040, float("0"), True)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1422 = paddle._C_ops.add(t1421, t1418)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1423 = paddle._C_ops.add(t1422, t1419)

        return (
            t367,
            t368,
            t369,
            t370,
            t371,
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
            t385,
            t386,
            t387,
            t388,
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
            t431,
            t432,
            t433,
            t434,
            t435,
            t436,
            t437,
            t438,
            t439,
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
            t494,
            t495,
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
            t531,
            t532,
            t533,
            t534,
            t535,
            t536,
            t537,
            t538,
            t539,
            t540,
            t541,
            t542,
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
            t581,
            t582,
            t583,
            t584,
            t585,
            t586,
            t587,
            t588,
            t589,
            t590,
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
            t631,
            t632,
            t633,
            t634,
            t635,
            t636,
            t637,
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
            t681,
            t682,
            t683,
            t684,
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
            t713,
            t714,
            t715,
            t716,
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
            t727,
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
            t739,
            t740,
            t741,
            t742,
            t743,
            t744,
            t745,
            t746,
            t747,
            t748,
            t749,
            t750,
            t751,
            t752,
            t753,
            t754,
            t755,
            t756,
            t757,
            t758,
            t759,
            t760,
            t761,
            t762,
            t763,
            t764,
            t765,
            t766,
            t767,
            t768,
            t769,
            t770,
            t771,
            t772,
            t773,
            t774,
            t775,
            t776,
            t777,
            t778,
            t779,
            t780,
            t781,
            t782,
            t783,
            t784,
            t785,
            t786,
            t787,
            t788,
            t789,
            t790,
            t791,
            t792,
            t793,
            t794,
            t795,
            t796,
            t797,
            t798,
            t799,
            t800,
            t801,
            t802,
            t803,
            t804,
            t805,
            t806,
            t807,
            t808,
            t809,
            t810,
            t811,
            t812,
            t813,
            t814,
            t815,
            t816,
            t817,
            t818,
            t819,
            t820,
            t821,
            t822,
            t823,
            t824,
            t825,
            t826,
            t827,
            t828,
            t829,
            t830,
            t831,
            t832,
            t833,
            t834,
            t835,
            t836,
            t837,
            t838,
            t839,
            t840,
            t841,
            t842,
            t843,
            t844,
            t845,
            t846,
            t847,
            t848,
            t849,
            t850,
            t851,
            t852,
            t853,
            t854,
            t855,
            t856,
            t857,
            t858,
            t859,
            t860,
            t861,
            t862,
            t863,
            t864,
            t866,
            t867,
            t868,
            t869,
            t870,
            t871,
            t872,
            t873,
            t874,
            t875,
            t876,
            t877,
            t878,
            t879,
            t880,
            t881,
            t882,
            t883,
            t884,
            t885,
            t886,
            t887,
            t888,
            t889,
            t890,
            t891,
            t892,
            t893,
            t894,
            t895,
            t896,
            t897,
            t898,
            t899,
            t900,
            t901,
            t902,
            t903,
            t904,
            t905,
            t906,
            t907,
            t908,
            t909,
            t910,
            t911,
            t912,
            t913,
            t914,
            t915,
            t916,
            t917,
            t918,
            t919,
            t920,
            t921,
            t922,
            t923,
            t925,
            t926,
            t927,
            t928,
            t929,
            t930,
            t931,
            t932,
            t933,
            t934,
            t935,
            t936,
            t937,
            t938,
            t939,
            t940,
            t941,
            t942,
            t943,
            t944,
            t945,
            t946,
            t947,
            t948,
            t949,
            t950,
            t951,
            t952,
            t953,
            t954,
            t955,
            t956,
            t957,
            t958,
            t959,
            t960,
            t961,
            t962,
            t963,
            t964,
            t965,
            t966,
            t967,
            t968,
            t969,
            t970,
            t971,
            t972,
            t973,
            t974,
            t976,
            t977,
            t978,
            t979,
            t980,
            t981,
            t982,
            t983,
            t986,
            t987,
            t988,
            t989,
            t990,
            t991,
            t992,
            t993,
            t994,
            t995,
            t996,
            t997,
            t998,
            t999,
            t1000,
            t1001,
            t1002,
            t1003,
            t1004,
            t1005,
            t1006,
            t1007,
            t1008,
            t1009,
            t1010,
            t1011,
            t1012,
            t1013,
            t1014,
            t1015,
            t1016,
            t1017,
            t1018,
            t1019,
            t1020,
            t1021,
            t1022,
            t1023,
            t1024,
            t1025,
            t1026,
            t1027,
            t1028,
            t1030,
            t1031,
            t1032,
            t1033,
            t1036,
            t1037,
            t1038,
            t1039,
            t1040,
            t1041,
            t1042,
            t1043,
            t1044,
            t1045,
            t1046,
            t1047,
            t1048,
            t1049,
            t1050,
            t1051,
            t1052,
            t1054,
            t1055,
            t1057,
            t1058,
            t1059,
            t1060,
            t1061,
            t1062,
            t1063,
            t1064,
            t1065,
            t1066,
            t1067,
            t1068,
            t1069,
            t1070,
            t1071,
            t1072,
            t1073,
            t1074,
            t1075,
            t1076,
            t1077,
            t1078,
            t1079,
            t1080,
            t1149,
            t1150,
            t1151,
            t1166,
            t1168,
            t1169,
            t1172,
            t1174,
            t1175,
            t1177,
            t1178,
            t1179,
            t1180,
            t1181,
            t1182,
            t1183,
            t1184,
            t1185,
            t1186,
            t1189,
            t1190,
            t1191,
            t1192,
            t1193,
            t1194,
            t1195,
            t1197,
            t1198,
            t1199,
            t1200,
            t1202,
            t1203,
            t1204,
            t1205,
            t1207,
            t1208,
            t1210,
            t1211,
            t1212,
            t1213,
            t1214,
            t1215,
            t1216,
            t1217,
            t1218,
            t1219,
            t1220,
            t1221,
            t1222,
            t1223,
            t1224,
            t1283,
            t1284,
            t1285,
            t1287,
            t1288,
            t1289,
            t1290,
            t1291,
            t1292,
            t1293,
            t1294,
            t1295,
            t1296,
            t1297,
            t1298,
            t1299,
            t1300,
            t1301,
            t1302,
            t1303,
            t1306,
            t1307,
            t1308,
            t1309,
            t1310,
            t1311,
            t1312,
            t1314,
            t1315,
            t1316,
            t1317,
            t1319,
            t1320,
            t1321,
            t1322,
            t1324,
            t1325,
            t1327,
            t1328,
            t1329,
            t1330,
            t1331,
            t1332,
            t1333,
            t1334,
            t1335,
            t1336,
            t1337,
            t1338,
            t1339,
            t1340,
            t1341,
            t1400,
            t1401,
            t1402,
            t1404,
            t1405,
            t1406,
            t1407,
            t1408,
            t1409,
            t1410,
            t1411,
            t1412,
            t1413,
            t1414,
            t1415,
            t1416,
            t1417,
            t1418,
            t1419,
            t1420,
            t1421,
            t1422,
            t1423,
        )
