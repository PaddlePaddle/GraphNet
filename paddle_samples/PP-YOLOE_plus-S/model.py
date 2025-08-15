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
        t366: paddle.Tensor,
        t367: paddle.Tensor,
    ):
        t368 = None
        # pd_op.conv2d: (8x16x352x352xf32) <- (8x3x704x704xf32, 16x3x3x3xf32)
        t369 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (8x16x352x352xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x352x352xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t370, t371, t372, t373, t374, t375 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t369,
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

        # pd_op.swish: (8x16x352x352xf32) <- (8x16x352x352xf32)
        t376 = paddle._C_ops.swish(t370)

        # pd_op.conv2d: (8x16x352x352xf32) <- (8x16x352x352xf32, 16x16x3x3xf32)
        t377 = paddle._C_ops.conv2d(
            t376, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (8x16x352x352xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x352x352xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t378, t379, t380, t381, t382, t383 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t377,
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

        # pd_op.swish: (8x16x352x352xf32) <- (8x16x352x352xf32)
        t384 = paddle._C_ops.swish(t378)

        # pd_op.conv2d: (8x32x352x352xf32) <- (8x16x352x352xf32, 32x16x3x3xf32)
        t385 = paddle._C_ops.conv2d(
            t384, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (8x32x352x352xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x352x352xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t386, t387, t388, t389, t390, t391 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t385,
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

        # pd_op.swish: (8x32x352x352xf32) <- (8x32x352x352xf32)
        t392 = paddle._C_ops.swish(t386)

        # pd_op.conv2d: (8x48x176x176xf32) <- (8x32x352x352xf32, 48x32x3x3xf32)
        t393 = paddle._C_ops.conv2d(
            t392, t15, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (8x48x176x176xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x176x176xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t394, t395, t396, t397, t398, t399 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t393,
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

        # pd_op.swish: (8x48x176x176xf32) <- (8x48x176x176xf32)
        t400 = paddle._C_ops.swish(t394)

        # pd_op.conv2d: (8x24x176x176xf32) <- (8x48x176x176xf32, 24x48x1x1xf32)
        t401 = paddle._C_ops.conv2d(
            t400, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t402, t403, t404, t405, t406, t407 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t401,
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

        # pd_op.swish: (8x24x176x176xf32) <- (8x24x176x176xf32)
        t408 = paddle._C_ops.swish(t402)

        # pd_op.conv2d: (8x24x176x176xf32) <- (8x48x176x176xf32, 24x48x1x1xf32)
        t409 = paddle._C_ops.conv2d(
            t400, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t410, t411, t412, t413, t414, t415 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t409,
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

        # pd_op.swish: (8x24x176x176xf32) <- (8x24x176x176xf32)
        t416 = paddle._C_ops.swish(t410)

        # pd_op.conv2d: (8x24x176x176xf32) <- (8x24x176x176xf32, 24x24x3x3xf32)
        t417 = paddle._C_ops.conv2d(
            t416, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t418, t419, t420, t421, t422, t423 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t417,
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

        # pd_op.swish: (8x24x176x176xf32) <- (8x24x176x176xf32)
        t424 = paddle._C_ops.swish(t418)

        # pd_op.conv2d: (8x24x176x176xf32) <- (8x24x176x176xf32, 24x24x3x3xf32)
        t425 = paddle._C_ops.conv2d(
            t424, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.conv2d: (8x24x176x176xf32) <- (8x24x176x176xf32, 24x24x1x1xf32)
        t432 = paddle._C_ops.conv2d(
            t424, t40, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x176x176xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t433, t434, t435, t436, t437, t438 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t432,
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

        # pd_op.add: (8x24x176x176xf32) <- (8x24x176x176xf32, 8x24x176x176xf32)
        t439 = paddle._C_ops.add(t426, t433)

        # pd_op.swish: (8x24x176x176xf32) <- (8x24x176x176xf32)
        t440 = paddle._C_ops.swish(t439)

        # pd_op.add: (8x24x176x176xf32) <- (8x24x176x176xf32, 8x24x176x176xf32)
        t441 = paddle._C_ops.add(t416, t440)

        # pd_op.full: (1xi32) <- ()
        t442 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t443 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t444 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t445 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t446 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t447 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t448 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t449 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t450 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t451 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t452 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t453 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t454 = t442

        # pd_op.assign: (1xi32) <- (1xi32)
        t455 = t442

        # builtin.combine: ([8x24x176x176xf32, 8x24x176x176xf32]) <- (8x24x176x176xf32, 8x24x176x176xf32)
        t456 = [t408, t441]

        # pd_op.concat: (8x48x176x176xf32) <- ([8x24x176x176xf32, 8x24x176x176xf32], 1xi32)
        t457 = paddle._C_ops.concat(t456, t442)
        del t456

        # pd_op.mean: (8x48x1x1xf32) <- (8x48x176x176xf32)
        t458 = paddle._C_ops.mean(t457, [2, 3], True)

        # pd_op.conv2d: (8x48x1x1xf32) <- (8x48x1x1xf32, 48x48x1x1xf32)
        t459 = paddle._C_ops.conv2d(
            t458, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.full_int_array: (4xi64) <- ()
        t460 = [1, -1, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t461 = paddle._C_ops.reshape(t46, t460)
        del t46

        # pd_op.add: (8x48x1x1xf32) <- (8x48x1x1xf32, 1x48x1x1xf32)
        t462 = paddle._C_ops.add(t459, t461)

        # pd_op.hardsigmoid: (8x48x1x1xf32) <- (8x48x1x1xf32)
        t463 = paddle._C_ops.hardsigmoid(t462, float("0.166667"), float("0.5"))
        del t462

        # pd_op.multiply: (8x48x176x176xf32) <- (8x48x176x176xf32, 8x48x1x1xf32)
        t464 = paddle._C_ops.multiply(t457, t463)

        # pd_op.conv2d: (8x64x176x176xf32) <- (8x48x176x176xf32, 64x48x1x1xf32)
        t465 = paddle._C_ops.conv2d(
            t464, t47, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t47

        # pd_op.batch_norm_: (8x64x176x176xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x176x176xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t466, t467, t468, t469, t470, t471 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t465,
                t48,
                t49,
                t50,
                t51,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t51, t50, t49, t48

        # pd_op.swish: (8x64x176x176xf32) <- (8x64x176x176xf32)
        t472 = paddle._C_ops.swish(t466)

        # pd_op.conv2d: (8x96x88x88xf32) <- (8x64x176x176xf32, 96x64x3x3xf32)
        t473 = paddle._C_ops.conv2d(
            t472, t52, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t52

        # pd_op.batch_norm_: (8x96x88x88xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x88x88xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t474, t475, t476, t477, t478, t479 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t473,
                t53,
                t54,
                t55,
                t56,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t56, t55, t54, t53

        # pd_op.swish: (8x96x88x88xf32) <- (8x96x88x88xf32)
        t480 = paddle._C_ops.swish(t474)

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x96x88x88xf32, 48x96x1x1xf32)
        t481 = paddle._C_ops.conv2d(
            t480, t57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t57

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t482, t483, t484, t485, t486, t487 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t481,
                t58,
                t59,
                t60,
                t61,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t61, t60, t59, t58

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t488 = paddle._C_ops.swish(t482)

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x96x88x88xf32, 48x96x1x1xf32)
        t489 = paddle._C_ops.conv2d(
            t480, t62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t62

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t490, t491, t492, t493, t494, t495 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t489,
                t63,
                t64,
                t65,
                t66,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t66, t65, t64, t63

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t496 = paddle._C_ops.swish(t490)

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x48x88x88xf32, 48x48x3x3xf32)
        t497 = paddle._C_ops.conv2d(
            t496, t67, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t67

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t498, t499, t500, t501, t502, t503 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t497,
                t68,
                t69,
                t70,
                t71,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t71, t70, t69, t68

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t504 = paddle._C_ops.swish(t498)

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x48x88x88xf32, 48x48x3x3xf32)
        t505 = paddle._C_ops.conv2d(
            t504, t72, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t72

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t506, t507, t508, t509, t510, t511 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t505,
                t73,
                t74,
                t75,
                t76,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t76, t75, t74, t73

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x48x88x88xf32, 48x48x1x1xf32)
        t512 = paddle._C_ops.conv2d(
            t504, t77, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t77

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t513, t514, t515, t516, t517, t518 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t512,
                t78,
                t79,
                t80,
                t81,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t81, t80, t79, t78

        # pd_op.add: (8x48x88x88xf32) <- (8x48x88x88xf32, 8x48x88x88xf32)
        t519 = paddle._C_ops.add(t506, t513)

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t520 = paddle._C_ops.swish(t519)

        # pd_op.add: (8x48x88x88xf32) <- (8x48x88x88xf32, 8x48x88x88xf32)
        t521 = paddle._C_ops.add(t496, t520)

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x48x88x88xf32, 48x48x3x3xf32)
        t522 = paddle._C_ops.conv2d(
            t521, t82, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t82

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t523, t524, t525, t526, t527, t528 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t522,
                t83,
                t84,
                t85,
                t86,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t86, t85, t84, t83

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t529 = paddle._C_ops.swish(t523)

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x48x88x88xf32, 48x48x3x3xf32)
        t530 = paddle._C_ops.conv2d(
            t529, t87, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t87

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t531, t532, t533, t534, t535, t536 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t530,
                t88,
                t89,
                t90,
                t91,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t91, t90, t89, t88

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x48x88x88xf32, 48x48x1x1xf32)
        t537 = paddle._C_ops.conv2d(
            t529, t92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t92

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t538, t539, t540, t541, t542, t543 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t537,
                t93,
                t94,
                t95,
                t96,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t96, t95, t94, t93

        # pd_op.add: (8x48x88x88xf32) <- (8x48x88x88xf32, 8x48x88x88xf32)
        t544 = paddle._C_ops.add(t531, t538)

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t545 = paddle._C_ops.swish(t544)

        # pd_op.add: (8x48x88x88xf32) <- (8x48x88x88xf32, 8x48x88x88xf32)
        t546 = paddle._C_ops.add(t521, t545)

        # builtin.combine: ([8x48x88x88xf32, 8x48x88x88xf32]) <- (8x48x88x88xf32, 8x48x88x88xf32)
        t547 = [t488, t546]

        # pd_op.concat: (8x96x88x88xf32) <- ([8x48x88x88xf32, 8x48x88x88xf32], 1xi32)
        t548 = paddle._C_ops.concat(t547, t442)
        del t547

        # pd_op.mean: (8x96x1x1xf32) <- (8x96x88x88xf32)
        t549 = paddle._C_ops.mean(t548, [2, 3], True)

        # pd_op.conv2d: (8x96x1x1xf32) <- (8x96x1x1xf32, 96x96x1x1xf32)
        t550 = paddle._C_ops.conv2d(
            t549, t97, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t97

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t551 = paddle._C_ops.reshape(t98, t460)
        del t98

        # pd_op.add: (8x96x1x1xf32) <- (8x96x1x1xf32, 1x96x1x1xf32)
        t552 = paddle._C_ops.add(t550, t551)

        # pd_op.hardsigmoid: (8x96x1x1xf32) <- (8x96x1x1xf32)
        t553 = paddle._C_ops.hardsigmoid(t552, float("0.166667"), float("0.5"))
        del t552

        # pd_op.multiply: (8x96x88x88xf32) <- (8x96x88x88xf32, 8x96x1x1xf32)
        t554 = paddle._C_ops.multiply(t548, t553)

        # pd_op.conv2d: (8x128x88x88xf32) <- (8x96x88x88xf32, 128x96x1x1xf32)
        t555 = paddle._C_ops.conv2d(
            t554, t99, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t99

        # pd_op.batch_norm_: (8x128x88x88xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (8x128x88x88xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t556, t557, t558, t559, t560, t561 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t555,
                t100,
                t101,
                t102,
                t103,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t103, t102, t101, t100

        # pd_op.swish: (8x128x88x88xf32) <- (8x128x88x88xf32)
        t562 = paddle._C_ops.swish(t556)

        # pd_op.conv2d: (8x192x44x44xf32) <- (8x128x88x88xf32, 192x128x3x3xf32)
        t563 = paddle._C_ops.conv2d(
            t562, t104, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t104

        # pd_op.batch_norm_: (8x192x44x44xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x44x44xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t564, t565, t566, t567, t568, t569 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t563,
                t105,
                t106,
                t107,
                t108,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t108, t107, t106, t105

        # pd_op.swish: (8x192x44x44xf32) <- (8x192x44x44xf32)
        t570 = paddle._C_ops.swish(t564)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x192x44x44xf32, 96x192x1x1xf32)
        t571 = paddle._C_ops.conv2d(
            t570, t109, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t109

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t572, t573, t574, t575, t576, t577 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t571,
                t110,
                t111,
                t112,
                t113,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t113, t112, t111, t110

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t578 = paddle._C_ops.swish(t572)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x192x44x44xf32, 96x192x1x1xf32)
        t579 = paddle._C_ops.conv2d(
            t570, t114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t114

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t580, t581, t582, t583, t584, t585 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t579,
                t115,
                t116,
                t117,
                t118,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t118, t117, t116, t115

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t586 = paddle._C_ops.swish(t580)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x3x3xf32)
        t587 = paddle._C_ops.conv2d(
            t586, t119, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t119

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t588, t589, t590, t591, t592, t593 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t587,
                t120,
                t121,
                t122,
                t123,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t123, t122, t121, t120

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t594 = paddle._C_ops.swish(t588)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x3x3xf32)
        t595 = paddle._C_ops.conv2d(
            t594, t124, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t124

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t596, t597, t598, t599, t600, t601 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t595,
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

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x1x1xf32)
        t602 = paddle._C_ops.conv2d(
            t594, t129, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t129

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t603, t604, t605, t606, t607, t608 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t602,
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

        # pd_op.add: (8x96x44x44xf32) <- (8x96x44x44xf32, 8x96x44x44xf32)
        t609 = paddle._C_ops.add(t596, t603)

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t610 = paddle._C_ops.swish(t609)

        # pd_op.add: (8x96x44x44xf32) <- (8x96x44x44xf32, 8x96x44x44xf32)
        t611 = paddle._C_ops.add(t586, t610)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x3x3xf32)
        t612 = paddle._C_ops.conv2d(
            t611, t134, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t134

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t613, t614, t615, t616, t617, t618 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t612,
                t135,
                t136,
                t137,
                t138,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t138, t137, t136, t135

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t619 = paddle._C_ops.swish(t613)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x3x3xf32)
        t620 = paddle._C_ops.conv2d(
            t619, t139, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t139

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t621, t622, t623, t624, t625, t626 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t620,
                t140,
                t141,
                t142,
                t143,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t143, t142, t141, t140

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x1x1xf32)
        t627 = paddle._C_ops.conv2d(
            t619, t144, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t144

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t628, t629, t630, t631, t632, t633 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t627,
                t145,
                t146,
                t147,
                t148,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t148, t147, t146, t145

        # pd_op.add: (8x96x44x44xf32) <- (8x96x44x44xf32, 8x96x44x44xf32)
        t634 = paddle._C_ops.add(t621, t628)

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t635 = paddle._C_ops.swish(t634)

        # pd_op.add: (8x96x44x44xf32) <- (8x96x44x44xf32, 8x96x44x44xf32)
        t636 = paddle._C_ops.add(t611, t635)

        # builtin.combine: ([8x96x44x44xf32, 8x96x44x44xf32]) <- (8x96x44x44xf32, 8x96x44x44xf32)
        t637 = [t578, t636]

        # pd_op.concat: (8x192x44x44xf32) <- ([8x96x44x44xf32, 8x96x44x44xf32], 1xi32)
        t638 = paddle._C_ops.concat(t637, t442)
        del t637

        # pd_op.mean: (8x192x1x1xf32) <- (8x192x44x44xf32)
        t639 = paddle._C_ops.mean(t638, [2, 3], True)

        # pd_op.conv2d: (8x192x1x1xf32) <- (8x192x1x1xf32, 192x192x1x1xf32)
        t640 = paddle._C_ops.conv2d(
            t639, t149, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t149

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t641 = paddle._C_ops.reshape(t150, t460)
        del t150

        # pd_op.add: (8x192x1x1xf32) <- (8x192x1x1xf32, 1x192x1x1xf32)
        t642 = paddle._C_ops.add(t640, t641)

        # pd_op.hardsigmoid: (8x192x1x1xf32) <- (8x192x1x1xf32)
        t643 = paddle._C_ops.hardsigmoid(t642, float("0.166667"), float("0.5"))
        del t642

        # pd_op.multiply: (8x192x44x44xf32) <- (8x192x44x44xf32, 8x192x1x1xf32)
        t644 = paddle._C_ops.multiply(t638, t643)

        # pd_op.conv2d: (8x256x44x44xf32) <- (8x192x44x44xf32, 256x192x1x1xf32)
        t645 = paddle._C_ops.conv2d(
            t644, t151, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t151

        # pd_op.batch_norm_: (8x256x44x44xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x44x44xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t646, t647, t648, t649, t650, t651 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t645,
                t152,
                t153,
                t154,
                t155,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t155, t154, t153, t152

        # pd_op.swish: (8x256x44x44xf32) <- (8x256x44x44xf32)
        t652 = paddle._C_ops.swish(t646)

        # pd_op.conv2d: (8x384x22x22xf32) <- (8x256x44x44xf32, 384x256x3x3xf32)
        t653 = paddle._C_ops.conv2d(
            t652, t156, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156

        # pd_op.batch_norm_: (8x384x22x22xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x22x22xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t654, t655, t656, t657, t658, t659 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t653,
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

        # pd_op.swish: (8x384x22x22xf32) <- (8x384x22x22xf32)
        t660 = paddle._C_ops.swish(t654)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x384x22x22xf32, 192x384x1x1xf32)
        t661 = paddle._C_ops.conv2d(
            t660, t161, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t161

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t662, t663, t664, t665, t666, t667 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t661,
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

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t668 = paddle._C_ops.swish(t662)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x384x22x22xf32, 192x384x1x1xf32)
        t669 = paddle._C_ops.conv2d(
            t660, t166, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t166

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t670, t671, t672, t673, t674, t675 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t669,
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

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t676 = paddle._C_ops.swish(t670)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 192x192x3x3xf32)
        t677 = paddle._C_ops.conv2d(
            t676, t171, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t171

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t678, t679, t680, t681, t682, t683 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t677,
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

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t684 = paddle._C_ops.swish(t678)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 192x192x3x3xf32)
        t685 = paddle._C_ops.conv2d(
            t684, t176, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t176

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t686, t687, t688, t689, t690, t691 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t685,
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

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 192x192x1x1xf32)
        t692 = paddle._C_ops.conv2d(
            t684, t181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t181

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t693, t694, t695, t696, t697, t698 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t692,
                t182,
                t183,
                t184,
                t185,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t185, t184, t183, t182

        # pd_op.add: (8x192x22x22xf32) <- (8x192x22x22xf32, 8x192x22x22xf32)
        t699 = paddle._C_ops.add(t686, t693)

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t700 = paddle._C_ops.swish(t699)

        # pd_op.add: (8x192x22x22xf32) <- (8x192x22x22xf32, 8x192x22x22xf32)
        t701 = paddle._C_ops.add(t676, t700)

        # builtin.combine: ([8x192x22x22xf32, 8x192x22x22xf32]) <- (8x192x22x22xf32, 8x192x22x22xf32)
        t702 = [t668, t701]

        # pd_op.concat: (8x384x22x22xf32) <- ([8x192x22x22xf32, 8x192x22x22xf32], 1xi32)
        t703 = paddle._C_ops.concat(t702, t442)
        del t702

        # pd_op.mean: (8x384x1x1xf32) <- (8x384x22x22xf32)
        t704 = paddle._C_ops.mean(t703, [2, 3], True)

        # pd_op.conv2d: (8x384x1x1xf32) <- (8x384x1x1xf32, 384x384x1x1xf32)
        t705 = paddle._C_ops.conv2d(
            t704, t186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t186

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t706 = paddle._C_ops.reshape(t187, t460)
        del t460, t187

        # pd_op.add: (8x384x1x1xf32) <- (8x384x1x1xf32, 1x384x1x1xf32)
        t707 = paddle._C_ops.add(t705, t706)

        # pd_op.hardsigmoid: (8x384x1x1xf32) <- (8x384x1x1xf32)
        t708 = paddle._C_ops.hardsigmoid(t707, float("0.166667"), float("0.5"))
        del t707

        # pd_op.multiply: (8x384x22x22xf32) <- (8x384x22x22xf32, 8x384x1x1xf32)
        t709 = paddle._C_ops.multiply(t703, t708)

        # pd_op.conv2d: (8x512x22x22xf32) <- (8x384x22x22xf32, 512x384x1x1xf32)
        t710 = paddle._C_ops.conv2d(
            t709, t188, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t188

        # pd_op.batch_norm_: (8x512x22x22xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x22x22xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t711, t712, t713, t714, t715, t716 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t710,
                t189,
                t190,
                t191,
                t192,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t192, t191, t190, t189

        # pd_op.swish: (8x512x22x22xf32) <- (8x512x22x22xf32)
        t717 = paddle._C_ops.swish(t711)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x512x22x22xf32, 192x512x1x1xf32)
        t718 = paddle._C_ops.conv2d(
            t717, t193, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t193

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t719, t720, t721, t722, t723, t724 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t718,
                t194,
                t195,
                t196,
                t197,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t197, t196, t195, t194

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t725 = paddle._C_ops.swish(t719)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x512x22x22xf32, 192x512x1x1xf32)
        t726 = paddle._C_ops.conv2d(
            t717, t198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t198

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t727, t728, t729, t730, t731, t732 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t726,
                t199,
                t200,
                t201,
                t202,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t202, t201, t200, t199

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t733 = paddle._C_ops.swish(t727)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 192x192x3x3xf32)
        t734 = paddle._C_ops.conv2d(
            t733, t203, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t203

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t735, t736, t737, t738, t739, t740 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t734,
                t204,
                t205,
                t206,
                t207,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t207, t206, t205, t204

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t741 = paddle._C_ops.swish(t735)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 192x192x3x3xf32)
        t742 = paddle._C_ops.conv2d(
            t741, t208, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t208

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t743, t744, t745, t746, t747, t748 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t742,
                t209,
                t210,
                t211,
                t212,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t212, t211, t210, t209

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 192x192x1x1xf32)
        t749 = paddle._C_ops.conv2d(
            t741, t213, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t213

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t750, t751, t752, t753, t754, t755 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t749,
                t214,
                t215,
                t216,
                t217,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t217, t216, t215, t214

        # pd_op.add: (8x192x22x22xf32) <- (8x192x22x22xf32, 8x192x22x22xf32)
        t756 = paddle._C_ops.add(t743, t750)

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t757 = paddle._C_ops.swish(t756)

        # pd_op.full_int_array: (2xi64) <- ()
        t758 = [5, 5]

        # pd_op.pool2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 2xi64)
        t759 = paddle._C_ops.pool2d(
            t757,
            t758,
            [1, 1],
            [2, 2],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.full_int_array: (2xi64) <- ()
        t760 = [9, 9]

        # pd_op.pool2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 2xi64)
        t761 = paddle._C_ops.pool2d(
            t757,
            t760,
            [1, 1],
            [4, 4],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.full_int_array: (2xi64) <- ()
        t762 = [13, 13]

        # pd_op.pool2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 2xi64)
        t763 = paddle._C_ops.pool2d(
            t757,
            t762,
            [1, 1],
            [6, 6],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # builtin.combine: ([8x192x22x22xf32, 8x192x22x22xf32, 8x192x22x22xf32, 8x192x22x22xf32]) <- (8x192x22x22xf32, 8x192x22x22xf32, 8x192x22x22xf32, 8x192x22x22xf32)
        t764 = [t757, t759, t761, t763]

        # pd_op.concat: (8x768x22x22xf32) <- ([8x192x22x22xf32, 8x192x22x22xf32, 8x192x22x22xf32, 8x192x22x22xf32], 1xi32)
        t765 = paddle._C_ops.concat(t764, t442)
        del t764

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x768x22x22xf32, 192x768x1x1xf32)
        t766 = paddle._C_ops.conv2d(
            t765, t218, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t218

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t767, t768, t769, t770, t771, t772 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t766,
                t219,
                t220,
                t221,
                t222,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t222, t221, t220, t219

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t773 = paddle._C_ops.swish(t767)

        # builtin.combine: ([8x192x22x22xf32, 8x192x22x22xf32]) <- (8x192x22x22xf32, 8x192x22x22xf32)
        t774 = [t725, t773]

        # pd_op.concat: (8x384x22x22xf32) <- ([8x192x22x22xf32, 8x192x22x22xf32], 1xi32)
        t775 = paddle._C_ops.concat(t774, t442)
        del t774

        # pd_op.conv2d: (8x384x22x22xf32) <- (8x384x22x22xf32, 384x384x1x1xf32)
        t776 = paddle._C_ops.conv2d(
            t775, t223, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t223

        # pd_op.batch_norm_: (8x384x22x22xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x22x22xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t777, t778, t779, t780, t781, t782 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t776,
                t224,
                t225,
                t226,
                t227,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t227, t226, t225, t224

        # pd_op.swish: (8x384x22x22xf32) <- (8x384x22x22xf32)
        t783 = paddle._C_ops.swish(t777)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x384x22x22xf32, 192x384x1x1xf32)
        t784 = paddle._C_ops.conv2d(
            t783, t228, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t228

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t785, t786, t787, t788, t789, t790 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t784,
                t229,
                t230,
                t231,
                t232,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t232, t231, t230, t229

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t791 = paddle._C_ops.swish(t785)

        # pd_op.nearest_interp: (8x192x44x44xf32) <- (8x192x22x22xf32, None, None, None)
        t792 = paddle._C_ops.nearest_interp(
            t791,
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

        # builtin.combine: ([8x192x44x44xf32, 8x256x44x44xf32]) <- (8x192x44x44xf32, 8x256x44x44xf32)
        t793 = [t792, t652]

        # pd_op.concat: (8x448x44x44xf32) <- ([8x192x44x44xf32, 8x256x44x44xf32], 1xi32)
        t794 = paddle._C_ops.concat(t793, t442)
        del t793

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x448x44x44xf32, 96x448x1x1xf32)
        t795 = paddle._C_ops.conv2d(
            t794, t233, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t233

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t796, t797, t798, t799, t800, t801 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t795,
                t234,
                t235,
                t236,
                t237,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t237, t236, t235, t234

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t802 = paddle._C_ops.swish(t796)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x448x44x44xf32, 96x448x1x1xf32)
        t803 = paddle._C_ops.conv2d(
            t794, t238, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t238

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t804, t805, t806, t807, t808, t809 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t803,
                t239,
                t240,
                t241,
                t242,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t242, t241, t240, t239

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t810 = paddle._C_ops.swish(t804)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x3x3xf32)
        t811 = paddle._C_ops.conv2d(
            t810, t243, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t243

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t812, t813, t814, t815, t816, t817 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t811,
                t244,
                t245,
                t246,
                t247,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t247, t246, t245, t244

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t818 = paddle._C_ops.swish(t812)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x3x3xf32)
        t819 = paddle._C_ops.conv2d(
            t818, t248, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t248

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t820, t821, t822, t823, t824, t825 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t819,
                t249,
                t250,
                t251,
                t252,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t252, t251, t250, t249

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x1x1xf32)
        t826 = paddle._C_ops.conv2d(
            t818, t253, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t253

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t827, t828, t829, t830, t831, t832 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t826,
                t254,
                t255,
                t256,
                t257,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t257, t256, t255, t254

        # pd_op.add: (8x96x44x44xf32) <- (8x96x44x44xf32, 8x96x44x44xf32)
        t833 = paddle._C_ops.add(t820, t827)

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t834 = paddle._C_ops.swish(t833)

        # builtin.combine: ([8x96x44x44xf32, 8x96x44x44xf32]) <- (8x96x44x44xf32, 8x96x44x44xf32)
        t835 = [t802, t834]

        # pd_op.concat: (8x192x44x44xf32) <- ([8x96x44x44xf32, 8x96x44x44xf32], 1xi32)
        t836 = paddle._C_ops.concat(t835, t442)
        del t835

        # pd_op.conv2d: (8x192x44x44xf32) <- (8x192x44x44xf32, 192x192x1x1xf32)
        t837 = paddle._C_ops.conv2d(
            t836, t258, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t258

        # pd_op.batch_norm_: (8x192x44x44xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x44x44xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t838, t839, t840, t841, t842, t843 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t837,
                t259,
                t260,
                t261,
                t262,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t262, t261, t260, t259

        # pd_op.swish: (8x192x44x44xf32) <- (8x192x44x44xf32)
        t844 = paddle._C_ops.swish(t838)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x192x44x44xf32, 96x192x1x1xf32)
        t845 = paddle._C_ops.conv2d(
            t844, t263, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t263

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t846, t847, t848, t849, t850, t851 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t845,
                t264,
                t265,
                t266,
                t267,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t267, t266, t265, t264

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t852 = paddle._C_ops.swish(t846)

        # pd_op.nearest_interp: (8x96x88x88xf32) <- (8x96x44x44xf32, None, None, None)
        t853 = paddle._C_ops.nearest_interp(
            t852,
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

        # builtin.combine: ([8x96x88x88xf32, 8x128x88x88xf32]) <- (8x96x88x88xf32, 8x128x88x88xf32)
        t854 = [t853, t562]

        # pd_op.concat: (8x224x88x88xf32) <- ([8x96x88x88xf32, 8x128x88x88xf32], 1xi32)
        t855 = paddle._C_ops.concat(t854, t442)
        del t854

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x224x88x88xf32, 48x224x1x1xf32)
        t856 = paddle._C_ops.conv2d(
            t855, t268, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t268

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t857, t858, t859, t860, t861, t862 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t856,
                t269,
                t270,
                t271,
                t272,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t272, t271, t270, t269

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t863 = paddle._C_ops.swish(t857)

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x224x88x88xf32, 48x224x1x1xf32)
        t864 = paddle._C_ops.conv2d(
            t855, t273, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t273

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t865, t866, t867, t868, t869, t870 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t864,
                t274,
                t275,
                t276,
                t277,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t277, t276, t275, t274

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t871 = paddle._C_ops.swish(t865)

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x48x88x88xf32, 48x48x3x3xf32)
        t872 = paddle._C_ops.conv2d(
            t871, t278, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t278

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t873, t874, t875, t876, t877, t878 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t872,
                t279,
                t280,
                t281,
                t282,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t282, t281, t280, t279

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t879 = paddle._C_ops.swish(t873)

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x48x88x88xf32, 48x48x3x3xf32)
        t880 = paddle._C_ops.conv2d(
            t879, t283, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t283

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t881, t882, t883, t884, t885, t886 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t880,
                t284,
                t285,
                t286,
                t287,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t287, t286, t285, t284

        # pd_op.conv2d: (8x48x88x88xf32) <- (8x48x88x88xf32, 48x48x1x1xf32)
        t887 = paddle._C_ops.conv2d(
            t879, t288, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t288

        # pd_op.batch_norm_: (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x88x88xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t888, t889, t890, t891, t892, t893 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t887,
                t289,
                t290,
                t291,
                t292,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t292, t291, t290, t289

        # pd_op.add: (8x48x88x88xf32) <- (8x48x88x88xf32, 8x48x88x88xf32)
        t894 = paddle._C_ops.add(t881, t888)

        # pd_op.swish: (8x48x88x88xf32) <- (8x48x88x88xf32)
        t895 = paddle._C_ops.swish(t894)

        # builtin.combine: ([8x48x88x88xf32, 8x48x88x88xf32]) <- (8x48x88x88xf32, 8x48x88x88xf32)
        t896 = [t863, t895]

        # pd_op.concat: (8x96x88x88xf32) <- ([8x48x88x88xf32, 8x48x88x88xf32], 1xi32)
        t897 = paddle._C_ops.concat(t896, t442)
        del t896

        # pd_op.conv2d: (8x96x88x88xf32) <- (8x96x88x88xf32, 96x96x1x1xf32)
        t898 = paddle._C_ops.conv2d(
            t897, t293, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t293

        # pd_op.batch_norm_: (8x96x88x88xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x88x88xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t899, t900, t901, t902, t903, t904 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t898,
                t294,
                t295,
                t296,
                t297,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t297, t296, t295, t294

        # pd_op.swish: (8x96x88x88xf32) <- (8x96x88x88xf32)
        t905 = paddle._C_ops.swish(t899)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x88x88xf32, 96x96x3x3xf32)
        t906 = paddle._C_ops.conv2d(
            t905, t298, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t298

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t907, t908, t909, t910, t911, t912 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t906,
                t299,
                t300,
                t301,
                t302,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t302, t301, t300, t299

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t913 = paddle._C_ops.swish(t907)

        # builtin.combine: ([8x96x44x44xf32, 8x192x44x44xf32]) <- (8x96x44x44xf32, 8x192x44x44xf32)
        t914 = [t913, t844]

        # pd_op.concat: (8x288x44x44xf32) <- ([8x96x44x44xf32, 8x192x44x44xf32], 1xi32)
        t915 = paddle._C_ops.concat(t914, t442)
        del t914

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x288x44x44xf32, 96x288x1x1xf32)
        t916 = paddle._C_ops.conv2d(
            t915, t303, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t303

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t917, t918, t919, t920, t921, t922 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t916,
                t304,
                t305,
                t306,
                t307,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t307, t306, t305, t304

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t923 = paddle._C_ops.swish(t917)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x288x44x44xf32, 96x288x1x1xf32)
        t924 = paddle._C_ops.conv2d(
            t915, t308, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t308

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t925, t926, t927, t928, t929, t930 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t924,
                t309,
                t310,
                t311,
                t312,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t312, t311, t310, t309

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t931 = paddle._C_ops.swish(t925)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x3x3xf32)
        t932 = paddle._C_ops.conv2d(
            t931, t313, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t313

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t933, t934, t935, t936, t937, t938 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t932,
                t314,
                t315,
                t316,
                t317,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t317, t316, t315, t314

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t939 = paddle._C_ops.swish(t933)

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x3x3xf32)
        t940 = paddle._C_ops.conv2d(
            t939, t318, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t318

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t941, t942, t943, t944, t945, t946 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t940,
                t319,
                t320,
                t321,
                t322,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t322, t321, t320, t319

        # pd_op.conv2d: (8x96x44x44xf32) <- (8x96x44x44xf32, 96x96x1x1xf32)
        t947 = paddle._C_ops.conv2d(
            t939, t323, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t323

        # pd_op.batch_norm_: (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x44x44xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t948, t949, t950, t951, t952, t953 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t947,
                t324,
                t325,
                t326,
                t327,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t327, t326, t325, t324

        # pd_op.add: (8x96x44x44xf32) <- (8x96x44x44xf32, 8x96x44x44xf32)
        t954 = paddle._C_ops.add(t941, t948)

        # pd_op.swish: (8x96x44x44xf32) <- (8x96x44x44xf32)
        t955 = paddle._C_ops.swish(t954)

        # builtin.combine: ([8x96x44x44xf32, 8x96x44x44xf32]) <- (8x96x44x44xf32, 8x96x44x44xf32)
        t956 = [t923, t955]

        # pd_op.concat: (8x192x44x44xf32) <- ([8x96x44x44xf32, 8x96x44x44xf32], 1xi32)
        t957 = paddle._C_ops.concat(t956, t442)
        del t956

        # pd_op.conv2d: (8x192x44x44xf32) <- (8x192x44x44xf32, 192x192x1x1xf32)
        t958 = paddle._C_ops.conv2d(
            t957, t328, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t328

        # pd_op.batch_norm_: (8x192x44x44xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x44x44xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t959, t960, t961, t962, t963, t964 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t958,
                t329,
                t330,
                t331,
                t332,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t332, t331, t330, t329

        # pd_op.swish: (8x192x44x44xf32) <- (8x192x44x44xf32)
        t965 = paddle._C_ops.swish(t959)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x44x44xf32, 192x192x3x3xf32)
        t966 = paddle._C_ops.conv2d(
            t965, t333, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t333

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t967, t968, t969, t970, t971, t972 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t966,
                t334,
                t335,
                t336,
                t337,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t337, t336, t335, t334

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t973 = paddle._C_ops.swish(t967)

        # builtin.combine: ([8x192x22x22xf32, 8x384x22x22xf32]) <- (8x192x22x22xf32, 8x384x22x22xf32)
        t974 = [t973, t783]

        # pd_op.concat: (8x576x22x22xf32) <- ([8x192x22x22xf32, 8x384x22x22xf32], 1xi32)
        t975 = paddle._C_ops.concat(t974, t442)
        del t974

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x576x22x22xf32, 192x576x1x1xf32)
        t976 = paddle._C_ops.conv2d(
            t975, t338, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t338

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t977, t978, t979, t980, t981, t982 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t976,
                t339,
                t340,
                t341,
                t342,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t342, t341, t340, t339

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t983 = paddle._C_ops.swish(t977)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x576x22x22xf32, 192x576x1x1xf32)
        t984 = paddle._C_ops.conv2d(
            t975, t343, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t343

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t985, t986, t987, t988, t989, t990 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t984,
                t344,
                t345,
                t346,
                t347,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t347, t346, t345, t344

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t991 = paddle._C_ops.swish(t985)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 192x192x3x3xf32)
        t992 = paddle._C_ops.conv2d(
            t991, t348, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t348

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t993, t994, t995, t996, t997, t998 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t992,
                t349,
                t350,
                t351,
                t352,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t352, t351, t350, t349

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t999 = paddle._C_ops.swish(t993)

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 192x192x3x3xf32)
        t1000 = paddle._C_ops.conv2d(
            t999, t353, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t353

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1001, t1002, t1003, t1004, t1005, t1006 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1000,
                t354,
                t355,
                t356,
                t357,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t357, t356, t355, t354

        # pd_op.conv2d: (8x192x22x22xf32) <- (8x192x22x22xf32, 192x192x1x1xf32)
        t1007 = paddle._C_ops.conv2d(
            t999, t358, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t358

        # pd_op.batch_norm_: (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x22x22xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1008, t1009, t1010, t1011, t1012, t1013 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1007,
                t359,
                t360,
                t361,
                t362,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t362, t361, t360, t359

        # pd_op.add: (8x192x22x22xf32) <- (8x192x22x22xf32, 8x192x22x22xf32)
        t1014 = paddle._C_ops.add(t1001, t1008)

        # pd_op.swish: (8x192x22x22xf32) <- (8x192x22x22xf32)
        t1015 = paddle._C_ops.swish(t1014)

        # builtin.combine: ([8x192x22x22xf32, 8x192x22x22xf32]) <- (8x192x22x22xf32, 8x192x22x22xf32)
        t1016 = [t983, t1015]

        # pd_op.concat: (8x384x22x22xf32) <- ([8x192x22x22xf32, 8x192x22x22xf32], 1xi32)
        t1017 = paddle._C_ops.concat(t1016, t442)
        del t1016

        # pd_op.conv2d: (8x384x22x22xf32) <- (8x384x22x22xf32, 384x384x1x1xf32)
        t1018 = paddle._C_ops.conv2d(
            t1017, t363, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t363

        # pd_op.batch_norm_: (8x384x22x22xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x22x22xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1019, t1020, t1021, t1022, t1023, t1024 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1018,
                t364,
                t365,
                t366,
                t367,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t367, t366, t365, t364

        return (
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
            t457,
            t458,
            t459,
            t461,
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
            t638,
            t639,
            t640,
            t641,
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
            t703,
            t704,
            t705,
            t706,
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
            t765,
            t766,
            t767,
            t768,
            t769,
            t770,
            t771,
            t772,
            t773,
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
            t865,
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
            t915,
            t916,
            t917,
            t918,
            t919,
            t920,
            t921,
            t922,
            t923,
            t924,
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
            t975,
            t976,
            t977,
            t978,
            t979,
            t980,
            t981,
            t982,
            t983,
            t984,
            t985,
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
            t1017,
            t1018,
            t1019,
            t1020,
            t1021,
            t1022,
            t1023,
            t1024,
        )
