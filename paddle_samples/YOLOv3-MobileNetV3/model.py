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
        input1: paddle.Tensor,
        input2: paddle.Tensor,
        input3: paddle.Tensor,
        input4: paddle.Tensor,
    ):
        t368 = None
        # pd_op.conv2d: (8x16x176x176xf32) <- (8x3x352x352xf32, 16x3x3x3xf32)
        t369 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (8x16x176x176xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x176x176xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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

        # pd_op.hardswish: (8x16x176x176xf32) <- (8x16x176x176xf32)
        t376 = paddle._C_ops.hardswish(t370)

        # pd_op.conv2d: (8x16x176x176xf32) <- (8x16x176x176xf32, 16x16x1x1xf32)
        t377 = paddle._C_ops.conv2d(
            t376, t5, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (8x16x176x176xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x176x176xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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

        # pd_op.relu: (8x16x176x176xf32) <- (8x16x176x176xf32)
        t384 = paddle._C_ops.relu(t378)
        del t378

        # pd_op.depthwise_conv2d: (8x16x176x176xf32) <- (8x16x176x176xf32, 16x1x3x3xf32)
        t385 = paddle._C_ops.depthwise_conv2d(
            t384, t10, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (8x16x176x176xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x176x176xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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

        # pd_op.relu: (8x16x176x176xf32) <- (8x16x176x176xf32)
        t392 = paddle._C_ops.relu(t386)
        del t386

        # pd_op.conv2d: (8x16x176x176xf32) <- (8x16x176x176xf32, 16x16x1x1xf32)
        t393 = paddle._C_ops.conv2d(
            t392, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (8x16x176x176xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x176x176xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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

        # pd_op.add: (8x16x176x176xf32) <- (8x16x176x176xf32, 8x16x176x176xf32)
        t400 = paddle._C_ops.add(t376, t394)

        # pd_op.conv2d: (8x64x176x176xf32) <- (8x16x176x176xf32, 64x16x1x1xf32)
        t401 = paddle._C_ops.conv2d(
            t400, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (8x64x176x176xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x176x176xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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

        # pd_op.relu: (8x64x176x176xf32) <- (8x64x176x176xf32)
        t408 = paddle._C_ops.relu(t402)
        del t402

        # pd_op.depthwise_conv2d: (8x64x88x88xf32) <- (8x64x176x176xf32, 64x1x3x3xf32)
        t409 = paddle._C_ops.depthwise_conv2d(
            t408, t25, [2, 2], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (8x64x88x88xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x88x88xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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

        # pd_op.relu: (8x64x88x88xf32) <- (8x64x88x88xf32)
        t416 = paddle._C_ops.relu(t410)
        del t410

        # pd_op.conv2d: (8x24x88x88xf32) <- (8x64x88x88xf32, 24x64x1x1xf32)
        t417 = paddle._C_ops.conv2d(
            t416, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (8x24x88x88xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x88x88xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.conv2d: (8x72x88x88xf32) <- (8x24x88x88xf32, 72x24x1x1xf32)
        t424 = paddle._C_ops.conv2d(
            t418, t35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (8x72x88x88xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (8x72x88x88xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        t425, t426, t427, t428, t429, t430 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t424,
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

        # pd_op.relu: (8x72x88x88xf32) <- (8x72x88x88xf32)
        t431 = paddle._C_ops.relu(t425)
        del t425

        # pd_op.depthwise_conv2d: (8x72x88x88xf32) <- (8x72x88x88xf32, 72x1x3x3xf32)
        t432 = paddle._C_ops.depthwise_conv2d(
            t431, t40, [1, 1], [1, 1], "EXPLICIT", 72, [1, 1], "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (8x72x88x88xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (8x72x88x88xf32, 72xf32, 72xf32, 72xf32, 72xf32)
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

        # pd_op.relu: (8x72x88x88xf32) <- (8x72x88x88xf32)
        t439 = paddle._C_ops.relu(t433)
        del t433

        # pd_op.conv2d: (8x24x88x88xf32) <- (8x72x88x88xf32, 24x72x1x1xf32)
        t440 = paddle._C_ops.conv2d(
            t439, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (8x24x88x88xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x88x88xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t441, t442, t443, t444, t445, t446 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t440,
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

        # pd_op.add: (8x24x88x88xf32) <- (8x24x88x88xf32, 8x24x88x88xf32)
        t447 = paddle._C_ops.add(t418, t441)

        # pd_op.conv2d: (8x72x88x88xf32) <- (8x24x88x88xf32, 72x24x1x1xf32)
        t448 = paddle._C_ops.conv2d(
            t447, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (8x72x88x88xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (8x72x88x88xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        t449, t450, t451, t452, t453, t454 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t448,
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

        # pd_op.relu: (8x72x88x88xf32) <- (8x72x88x88xf32)
        t455 = paddle._C_ops.relu(t449)
        del t449

        # pd_op.depthwise_conv2d: (8x72x44x44xf32) <- (8x72x88x88xf32, 72x1x5x5xf32)
        t456 = paddle._C_ops.depthwise_conv2d(
            t455, t55, [2, 2], [2, 2], "EXPLICIT", 72, [1, 1], "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (8x72x44x44xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (8x72x44x44xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        t457, t458, t459, t460, t461, t462 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t456,
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

        # pd_op.relu: (8x72x44x44xf32) <- (8x72x44x44xf32)
        t463 = paddle._C_ops.relu(t457)
        del t457

        # pd_op.full_int_array: (2xi64) <- ()
        t464 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        t465 = t464

        # pd_op.assign: (2xi64) <- (2xi64)
        t466 = t464

        # pd_op.assign: (2xi64) <- (2xi64)
        t467 = t464

        # pd_op.assign: (2xi64) <- (2xi64)
        t468 = t464

        # pd_op.assign: (2xi64) <- (2xi64)
        t469 = t464

        # pd_op.assign: (2xi64) <- (2xi64)
        t470 = t464

        # pd_op.assign: (2xi64) <- (2xi64)
        t471 = t464

        # pd_op.pool2d: (8x72x1x1xf32) <- (8x72x44x44xf32, 2xi64)
        t472 = paddle._C_ops.pool2d(
            t463,
            t464,
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

        # pd_op.conv2d: (8x18x1x1xf32) <- (8x72x1x1xf32, 18x72x1x1xf32)
        t473 = paddle._C_ops.conv2d(
            t472, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.full_int_array: (4xi64) <- ()
        t474 = [1, -1, 1, 1]

        # pd_op.reshape: (1x18x1x1xf32) <- (18xf32, 4xi64)
        t475 = paddle._C_ops.reshape(t61, t474)
        del t61

        # pd_op.add: (8x18x1x1xf32) <- (8x18x1x1xf32, 1x18x1x1xf32)
        t476 = paddle._C_ops.add(t473, t475)

        # pd_op.relu: (8x18x1x1xf32) <- (8x18x1x1xf32)
        t477 = paddle._C_ops.relu(t476)
        del t476

        # pd_op.conv2d: (8x72x1x1xf32) <- (8x18x1x1xf32, 72x18x1x1xf32)
        t478 = paddle._C_ops.conv2d(
            t477, t62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t62

        # pd_op.reshape: (1x72x1x1xf32) <- (72xf32, 4xi64)
        t479 = paddle._C_ops.reshape(t63, t474)
        del t63

        # pd_op.add: (8x72x1x1xf32) <- (8x72x1x1xf32, 1x72x1x1xf32)
        t480 = paddle._C_ops.add(t478, t479)

        # pd_op.hardsigmoid: (8x72x1x1xf32) <- (8x72x1x1xf32)
        t481 = paddle._C_ops.hardsigmoid(t480, float("0.2"), float("0.5"))
        del t480

        # pd_op.multiply: (8x72x44x44xf32) <- (8x72x44x44xf32, 8x72x1x1xf32)
        t482 = paddle._C_ops.multiply(t463, t481)

        # pd_op.conv2d: (8x40x44x44xf32) <- (8x72x44x44xf32, 40x72x1x1xf32)
        t483 = paddle._C_ops.conv2d(
            t482, t64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t64

        # pd_op.batch_norm_: (8x40x44x44xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (8x40x44x44xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        t484, t485, t486, t487, t488, t489 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t483,
                t65,
                t66,
                t67,
                t68,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t68, t67, t66, t65

        # pd_op.conv2d: (8x120x44x44xf32) <- (8x40x44x44xf32, 120x40x1x1xf32)
        t490 = paddle._C_ops.conv2d(
            t484, t69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t69

        # pd_op.batch_norm_: (8x120x44x44xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (8x120x44x44xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t491, t492, t493, t494, t495, t496 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t490,
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

        # pd_op.relu: (8x120x44x44xf32) <- (8x120x44x44xf32)
        t497 = paddle._C_ops.relu(t491)
        del t491

        # pd_op.depthwise_conv2d: (8x120x44x44xf32) <- (8x120x44x44xf32, 120x1x5x5xf32)
        t498 = paddle._C_ops.depthwise_conv2d(
            t497, t74, [1, 1], [2, 2], "EXPLICIT", 120, [1, 1], "NCHW"
        )
        del t74

        # pd_op.batch_norm_: (8x120x44x44xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (8x120x44x44xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t499, t500, t501, t502, t503, t504 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t498,
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

        # pd_op.relu: (8x120x44x44xf32) <- (8x120x44x44xf32)
        t505 = paddle._C_ops.relu(t499)
        del t499

        # pd_op.pool2d: (8x120x1x1xf32) <- (8x120x44x44xf32, 2xi64)
        t506 = paddle._C_ops.pool2d(
            t505,
            t464,
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

        # pd_op.conv2d: (8x30x1x1xf32) <- (8x120x1x1xf32, 30x120x1x1xf32)
        t507 = paddle._C_ops.conv2d(
            t506, t79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t79

        # pd_op.reshape: (1x30x1x1xf32) <- (30xf32, 4xi64)
        t508 = paddle._C_ops.reshape(t80, t474)
        del t80

        # pd_op.add: (8x30x1x1xf32) <- (8x30x1x1xf32, 1x30x1x1xf32)
        t509 = paddle._C_ops.add(t507, t508)

        # pd_op.relu: (8x30x1x1xf32) <- (8x30x1x1xf32)
        t510 = paddle._C_ops.relu(t509)
        del t509

        # pd_op.conv2d: (8x120x1x1xf32) <- (8x30x1x1xf32, 120x30x1x1xf32)
        t511 = paddle._C_ops.conv2d(
            t510, t81, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t81

        # pd_op.reshape: (1x120x1x1xf32) <- (120xf32, 4xi64)
        t512 = paddle._C_ops.reshape(t82, t474)
        del t82

        # pd_op.add: (8x120x1x1xf32) <- (8x120x1x1xf32, 1x120x1x1xf32)
        t513 = paddle._C_ops.add(t511, t512)

        # pd_op.hardsigmoid: (8x120x1x1xf32) <- (8x120x1x1xf32)
        t514 = paddle._C_ops.hardsigmoid(t513, float("0.2"), float("0.5"))
        del t513

        # pd_op.multiply: (8x120x44x44xf32) <- (8x120x44x44xf32, 8x120x1x1xf32)
        t515 = paddle._C_ops.multiply(t505, t514)

        # pd_op.conv2d: (8x40x44x44xf32) <- (8x120x44x44xf32, 40x120x1x1xf32)
        t516 = paddle._C_ops.conv2d(
            t515, t83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t83

        # pd_op.batch_norm_: (8x40x44x44xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (8x40x44x44xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        t517, t518, t519, t520, t521, t522 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t516,
                t84,
                t85,
                t86,
                t87,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t87, t86, t85, t84

        # pd_op.add: (8x40x44x44xf32) <- (8x40x44x44xf32, 8x40x44x44xf32)
        t523 = paddle._C_ops.add(t484, t517)

        # pd_op.conv2d: (8x120x44x44xf32) <- (8x40x44x44xf32, 120x40x1x1xf32)
        t524 = paddle._C_ops.conv2d(
            t523, t88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t88

        # pd_op.batch_norm_: (8x120x44x44xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (8x120x44x44xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t525, t526, t527, t528, t529, t530 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t524,
                t89,
                t90,
                t91,
                t92,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t92, t91, t90, t89

        # pd_op.relu: (8x120x44x44xf32) <- (8x120x44x44xf32)
        t531 = paddle._C_ops.relu(t525)
        del t525

        # pd_op.depthwise_conv2d: (8x120x44x44xf32) <- (8x120x44x44xf32, 120x1x5x5xf32)
        t532 = paddle._C_ops.depthwise_conv2d(
            t531, t93, [1, 1], [2, 2], "EXPLICIT", 120, [1, 1], "NCHW"
        )
        del t93

        # pd_op.batch_norm_: (8x120x44x44xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (8x120x44x44xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t533, t534, t535, t536, t537, t538 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t532,
                t94,
                t95,
                t96,
                t97,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t97, t96, t95, t94

        # pd_op.relu: (8x120x44x44xf32) <- (8x120x44x44xf32)
        t539 = paddle._C_ops.relu(t533)
        del t533

        # pd_op.pool2d: (8x120x1x1xf32) <- (8x120x44x44xf32, 2xi64)
        t540 = paddle._C_ops.pool2d(
            t539,
            t464,
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

        # pd_op.conv2d: (8x30x1x1xf32) <- (8x120x1x1xf32, 30x120x1x1xf32)
        t541 = paddle._C_ops.conv2d(
            t540, t98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t98

        # pd_op.reshape: (1x30x1x1xf32) <- (30xf32, 4xi64)
        t542 = paddle._C_ops.reshape(t99, t474)
        del t99

        # pd_op.add: (8x30x1x1xf32) <- (8x30x1x1xf32, 1x30x1x1xf32)
        t543 = paddle._C_ops.add(t541, t542)

        # pd_op.relu: (8x30x1x1xf32) <- (8x30x1x1xf32)
        t544 = paddle._C_ops.relu(t543)
        del t543

        # pd_op.conv2d: (8x120x1x1xf32) <- (8x30x1x1xf32, 120x30x1x1xf32)
        t545 = paddle._C_ops.conv2d(
            t544, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.reshape: (1x120x1x1xf32) <- (120xf32, 4xi64)
        t546 = paddle._C_ops.reshape(t101, t474)
        del t101

        # pd_op.add: (8x120x1x1xf32) <- (8x120x1x1xf32, 1x120x1x1xf32)
        t547 = paddle._C_ops.add(t545, t546)

        # pd_op.hardsigmoid: (8x120x1x1xf32) <- (8x120x1x1xf32)
        t548 = paddle._C_ops.hardsigmoid(t547, float("0.2"), float("0.5"))
        del t547

        # pd_op.multiply: (8x120x44x44xf32) <- (8x120x44x44xf32, 8x120x1x1xf32)
        t549 = paddle._C_ops.multiply(t539, t548)

        # pd_op.conv2d: (8x40x44x44xf32) <- (8x120x44x44xf32, 40x120x1x1xf32)
        t550 = paddle._C_ops.conv2d(
            t549, t102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t102

        # pd_op.batch_norm_: (8x40x44x44xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (8x40x44x44xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        t551, t552, t553, t554, t555, t556 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t550,
                t103,
                t104,
                t105,
                t106,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t106, t105, t104, t103

        # pd_op.add: (8x40x44x44xf32) <- (8x40x44x44xf32, 8x40x44x44xf32)
        t557 = paddle._C_ops.add(t523, t551)

        # pd_op.conv2d: (8x240x44x44xf32) <- (8x40x44x44xf32, 240x40x1x1xf32)
        t558 = paddle._C_ops.conv2d(
            t557, t107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t107

        # pd_op.batch_norm_: (8x240x44x44xf32, 240xf32, 240xf32, 240xf32, 240xf32, -1xui8) <- (8x240x44x44xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        t559, t560, t561, t562, t563, t564 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t558,
                t108,
                t109,
                t110,
                t111,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t111, t110, t109, t108

        # pd_op.hardswish: (8x240x44x44xf32) <- (8x240x44x44xf32)
        t565 = paddle._C_ops.hardswish(t559)

        # pd_op.depthwise_conv2d: (8x240x22x22xf32) <- (8x240x44x44xf32, 240x1x3x3xf32)
        t566 = paddle._C_ops.depthwise_conv2d(
            t565, t112, [2, 2], [1, 1], "EXPLICIT", 240, [1, 1], "NCHW"
        )
        del t112

        # pd_op.batch_norm_: (8x240x22x22xf32, 240xf32, 240xf32, 240xf32, 240xf32, -1xui8) <- (8x240x22x22xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        t567, t568, t569, t570, t571, t572 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t566,
                t113,
                t114,
                t115,
                t116,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t116, t115, t114, t113

        # pd_op.hardswish: (8x240x22x22xf32) <- (8x240x22x22xf32)
        t573 = paddle._C_ops.hardswish(t567)

        # pd_op.conv2d: (8x80x22x22xf32) <- (8x240x22x22xf32, 80x240x1x1xf32)
        t574 = paddle._C_ops.conv2d(
            t573, t117, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t117

        # pd_op.batch_norm_: (8x80x22x22xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (8x80x22x22xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        t575, t576, t577, t578, t579, t580 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t574,
                t118,
                t119,
                t120,
                t121,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t121, t120, t119, t118

        # pd_op.conv2d: (8x200x22x22xf32) <- (8x80x22x22xf32, 200x80x1x1xf32)
        t581 = paddle._C_ops.conv2d(
            t575, t122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t122

        # pd_op.batch_norm_: (8x200x22x22xf32, 200xf32, 200xf32, 200xf32, 200xf32, -1xui8) <- (8x200x22x22xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        t582, t583, t584, t585, t586, t587 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t581,
                t123,
                t124,
                t125,
                t126,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t126, t125, t124, t123

        # pd_op.hardswish: (8x200x22x22xf32) <- (8x200x22x22xf32)
        t588 = paddle._C_ops.hardswish(t582)

        # pd_op.depthwise_conv2d: (8x200x22x22xf32) <- (8x200x22x22xf32, 200x1x3x3xf32)
        t589 = paddle._C_ops.depthwise_conv2d(
            t588, t127, [1, 1], [1, 1], "EXPLICIT", 200, [1, 1], "NCHW"
        )
        del t127

        # pd_op.batch_norm_: (8x200x22x22xf32, 200xf32, 200xf32, 200xf32, 200xf32, -1xui8) <- (8x200x22x22xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        t590, t591, t592, t593, t594, t595 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t589,
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

        # pd_op.hardswish: (8x200x22x22xf32) <- (8x200x22x22xf32)
        t596 = paddle._C_ops.hardswish(t590)

        # pd_op.conv2d: (8x80x22x22xf32) <- (8x200x22x22xf32, 80x200x1x1xf32)
        t597 = paddle._C_ops.conv2d(
            t596, t132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t132

        # pd_op.batch_norm_: (8x80x22x22xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (8x80x22x22xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        t598, t599, t600, t601, t602, t603 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t597,
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

        # pd_op.add: (8x80x22x22xf32) <- (8x80x22x22xf32, 8x80x22x22xf32)
        t604 = paddle._C_ops.add(t575, t598)

        # pd_op.conv2d: (8x184x22x22xf32) <- (8x80x22x22xf32, 184x80x1x1xf32)
        t605 = paddle._C_ops.conv2d(
            t604, t137, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t137

        # pd_op.batch_norm_: (8x184x22x22xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (8x184x22x22xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        t606, t607, t608, t609, t610, t611 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t605,
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

        # pd_op.hardswish: (8x184x22x22xf32) <- (8x184x22x22xf32)
        t612 = paddle._C_ops.hardswish(t606)

        # pd_op.depthwise_conv2d: (8x184x22x22xf32) <- (8x184x22x22xf32, 184x1x3x3xf32)
        t613 = paddle._C_ops.depthwise_conv2d(
            t612, t142, [1, 1], [1, 1], "EXPLICIT", 184, [1, 1], "NCHW"
        )
        del t142

        # pd_op.batch_norm_: (8x184x22x22xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (8x184x22x22xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        t614, t615, t616, t617, t618, t619 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t613,
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

        # pd_op.hardswish: (8x184x22x22xf32) <- (8x184x22x22xf32)
        t620 = paddle._C_ops.hardswish(t614)

        # pd_op.conv2d: (8x80x22x22xf32) <- (8x184x22x22xf32, 80x184x1x1xf32)
        t621 = paddle._C_ops.conv2d(
            t620, t147, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t147

        # pd_op.batch_norm_: (8x80x22x22xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (8x80x22x22xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        t622, t623, t624, t625, t626, t627 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t621,
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

        # pd_op.add: (8x80x22x22xf32) <- (8x80x22x22xf32, 8x80x22x22xf32)
        t628 = paddle._C_ops.add(t604, t622)

        # pd_op.conv2d: (8x184x22x22xf32) <- (8x80x22x22xf32, 184x80x1x1xf32)
        t629 = paddle._C_ops.conv2d(
            t628, t152, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t152

        # pd_op.batch_norm_: (8x184x22x22xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (8x184x22x22xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        t630, t631, t632, t633, t634, t635 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t629,
                t153,
                t154,
                t155,
                t156,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t156, t155, t154, t153

        # pd_op.hardswish: (8x184x22x22xf32) <- (8x184x22x22xf32)
        t636 = paddle._C_ops.hardswish(t630)

        # pd_op.depthwise_conv2d: (8x184x22x22xf32) <- (8x184x22x22xf32, 184x1x3x3xf32)
        t637 = paddle._C_ops.depthwise_conv2d(
            t636, t157, [1, 1], [1, 1], "EXPLICIT", 184, [1, 1], "NCHW"
        )
        del t157

        # pd_op.batch_norm_: (8x184x22x22xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (8x184x22x22xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        t638, t639, t640, t641, t642, t643 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t637,
                t158,
                t159,
                t160,
                t161,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t161, t160, t159, t158

        # pd_op.hardswish: (8x184x22x22xf32) <- (8x184x22x22xf32)
        t644 = paddle._C_ops.hardswish(t638)

        # pd_op.conv2d: (8x80x22x22xf32) <- (8x184x22x22xf32, 80x184x1x1xf32)
        t645 = paddle._C_ops.conv2d(
            t644, t162, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t162

        # pd_op.batch_norm_: (8x80x22x22xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (8x80x22x22xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        t646, t647, t648, t649, t650, t651 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t645,
                t163,
                t164,
                t165,
                t166,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t166, t165, t164, t163

        # pd_op.add: (8x80x22x22xf32) <- (8x80x22x22xf32, 8x80x22x22xf32)
        t652 = paddle._C_ops.add(t628, t646)

        # pd_op.conv2d: (8x480x22x22xf32) <- (8x80x22x22xf32, 480x80x1x1xf32)
        t653 = paddle._C_ops.conv2d(
            t652, t167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t167

        # pd_op.batch_norm_: (8x480x22x22xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (8x480x22x22xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        t654, t655, t656, t657, t658, t659 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t653,
                t168,
                t169,
                t170,
                t171,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t171, t170, t169, t168

        # pd_op.hardswish: (8x480x22x22xf32) <- (8x480x22x22xf32)
        t660 = paddle._C_ops.hardswish(t654)

        # pd_op.depthwise_conv2d: (8x480x22x22xf32) <- (8x480x22x22xf32, 480x1x3x3xf32)
        t661 = paddle._C_ops.depthwise_conv2d(
            t660, t172, [1, 1], [1, 1], "EXPLICIT", 480, [1, 1], "NCHW"
        )
        del t172

        # pd_op.batch_norm_: (8x480x22x22xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (8x480x22x22xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        t662, t663, t664, t665, t666, t667 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t661,
                t173,
                t174,
                t175,
                t176,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t176, t175, t174, t173

        # pd_op.hardswish: (8x480x22x22xf32) <- (8x480x22x22xf32)
        t668 = paddle._C_ops.hardswish(t662)

        # pd_op.pool2d: (8x480x1x1xf32) <- (8x480x22x22xf32, 2xi64)
        t669 = paddle._C_ops.pool2d(
            t668,
            t464,
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

        # pd_op.conv2d: (8x120x1x1xf32) <- (8x480x1x1xf32, 120x480x1x1xf32)
        t670 = paddle._C_ops.conv2d(
            t669, t177, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t177

        # pd_op.reshape: (1x120x1x1xf32) <- (120xf32, 4xi64)
        t671 = paddle._C_ops.reshape(t178, t474)
        del t178

        # pd_op.add: (8x120x1x1xf32) <- (8x120x1x1xf32, 1x120x1x1xf32)
        t672 = paddle._C_ops.add(t670, t671)

        # pd_op.relu: (8x120x1x1xf32) <- (8x120x1x1xf32)
        t673 = paddle._C_ops.relu(t672)
        del t672

        # pd_op.conv2d: (8x480x1x1xf32) <- (8x120x1x1xf32, 480x120x1x1xf32)
        t674 = paddle._C_ops.conv2d(
            t673, t179, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t179

        # pd_op.reshape: (1x480x1x1xf32) <- (480xf32, 4xi64)
        t675 = paddle._C_ops.reshape(t180, t474)
        del t180

        # pd_op.add: (8x480x1x1xf32) <- (8x480x1x1xf32, 1x480x1x1xf32)
        t676 = paddle._C_ops.add(t674, t675)

        # pd_op.hardsigmoid: (8x480x1x1xf32) <- (8x480x1x1xf32)
        t677 = paddle._C_ops.hardsigmoid(t676, float("0.2"), float("0.5"))
        del t676

        # pd_op.multiply: (8x480x22x22xf32) <- (8x480x22x22xf32, 8x480x1x1xf32)
        t678 = paddle._C_ops.multiply(t668, t677)

        # pd_op.conv2d: (8x112x22x22xf32) <- (8x480x22x22xf32, 112x480x1x1xf32)
        t679 = paddle._C_ops.conv2d(
            t678, t181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t181

        # pd_op.batch_norm_: (8x112x22x22xf32, 112xf32, 112xf32, 112xf32, 112xf32, -1xui8) <- (8x112x22x22xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        t680, t681, t682, t683, t684, t685 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t679,
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

        # pd_op.conv2d: (8x672x22x22xf32) <- (8x112x22x22xf32, 672x112x1x1xf32)
        t686 = paddle._C_ops.conv2d(
            t680, t186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t186

        # pd_op.batch_norm_: (8x672x22x22xf32, 672xf32, 672xf32, 672xf32, 672xf32, -1xui8) <- (8x672x22x22xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        t687, t688, t689, t690, t691, t692 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t686,
                t187,
                t188,
                t189,
                t190,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t190, t189, t188, t187

        # pd_op.hardswish: (8x672x22x22xf32) <- (8x672x22x22xf32)
        t693 = paddle._C_ops.hardswish(t687)

        # pd_op.depthwise_conv2d: (8x672x22x22xf32) <- (8x672x22x22xf32, 672x1x3x3xf32)
        t694 = paddle._C_ops.depthwise_conv2d(
            t693, t191, [1, 1], [1, 1], "EXPLICIT", 672, [1, 1], "NCHW"
        )
        del t191

        # pd_op.batch_norm_: (8x672x22x22xf32, 672xf32, 672xf32, 672xf32, 672xf32, -1xui8) <- (8x672x22x22xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        t695, t696, t697, t698, t699, t700 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t694,
                t192,
                t193,
                t194,
                t195,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t195, t194, t193, t192

        # pd_op.hardswish: (8x672x22x22xf32) <- (8x672x22x22xf32)
        t701 = paddle._C_ops.hardswish(t695)

        # pd_op.pool2d: (8x672x1x1xf32) <- (8x672x22x22xf32, 2xi64)
        t702 = paddle._C_ops.pool2d(
            t701,
            t464,
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

        # pd_op.conv2d: (8x168x1x1xf32) <- (8x672x1x1xf32, 168x672x1x1xf32)
        t703 = paddle._C_ops.conv2d(
            t702, t196, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t196

        # pd_op.reshape: (1x168x1x1xf32) <- (168xf32, 4xi64)
        t704 = paddle._C_ops.reshape(t197, t474)
        del t197

        # pd_op.add: (8x168x1x1xf32) <- (8x168x1x1xf32, 1x168x1x1xf32)
        t705 = paddle._C_ops.add(t703, t704)

        # pd_op.relu: (8x168x1x1xf32) <- (8x168x1x1xf32)
        t706 = paddle._C_ops.relu(t705)
        del t705

        # pd_op.conv2d: (8x672x1x1xf32) <- (8x168x1x1xf32, 672x168x1x1xf32)
        t707 = paddle._C_ops.conv2d(
            t706, t198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t198

        # pd_op.reshape: (1x672x1x1xf32) <- (672xf32, 4xi64)
        t708 = paddle._C_ops.reshape(t199, t474)
        del t199

        # pd_op.add: (8x672x1x1xf32) <- (8x672x1x1xf32, 1x672x1x1xf32)
        t709 = paddle._C_ops.add(t707, t708)

        # pd_op.hardsigmoid: (8x672x1x1xf32) <- (8x672x1x1xf32)
        t710 = paddle._C_ops.hardsigmoid(t709, float("0.2"), float("0.5"))
        del t709

        # pd_op.multiply: (8x672x22x22xf32) <- (8x672x22x22xf32, 8x672x1x1xf32)
        t711 = paddle._C_ops.multiply(t701, t710)

        # pd_op.conv2d: (8x112x22x22xf32) <- (8x672x22x22xf32, 112x672x1x1xf32)
        t712 = paddle._C_ops.conv2d(
            t711, t200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (8x112x22x22xf32, 112xf32, 112xf32, 112xf32, 112xf32, -1xui8) <- (8x112x22x22xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        t713, t714, t715, t716, t717, t718 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t712,
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

        # pd_op.add: (8x112x22x22xf32) <- (8x112x22x22xf32, 8x112x22x22xf32)
        t719 = paddle._C_ops.add(t680, t713)

        # pd_op.conv2d: (8x672x22x22xf32) <- (8x112x22x22xf32, 672x112x1x1xf32)
        t720 = paddle._C_ops.conv2d(
            t719, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (8x672x22x22xf32, 672xf32, 672xf32, 672xf32, 672xf32, -1xui8) <- (8x672x22x22xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        t721, t722, t723, t724, t725, t726 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t720,
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

        # pd_op.hardswish: (8x672x22x22xf32) <- (8x672x22x22xf32)
        t727 = paddle._C_ops.hardswish(t721)

        # pd_op.depthwise_conv2d: (8x672x11x11xf32) <- (8x672x22x22xf32, 672x1x5x5xf32)
        t728 = paddle._C_ops.depthwise_conv2d(
            t727, t210, [2, 2], [2, 2], "EXPLICIT", 672, [1, 1], "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (8x672x11x11xf32, 672xf32, 672xf32, 672xf32, 672xf32, -1xui8) <- (8x672x11x11xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        t729, t730, t731, t732, t733, t734 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t728,
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

        # pd_op.hardswish: (8x672x11x11xf32) <- (8x672x11x11xf32)
        t735 = paddle._C_ops.hardswish(t729)

        # pd_op.pool2d: (8x672x1x1xf32) <- (8x672x11x11xf32, 2xi64)
        t736 = paddle._C_ops.pool2d(
            t735,
            t464,
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

        # pd_op.conv2d: (8x168x1x1xf32) <- (8x672x1x1xf32, 168x672x1x1xf32)
        t737 = paddle._C_ops.conv2d(
            t736, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.reshape: (1x168x1x1xf32) <- (168xf32, 4xi64)
        t738 = paddle._C_ops.reshape(t216, t474)
        del t216

        # pd_op.add: (8x168x1x1xf32) <- (8x168x1x1xf32, 1x168x1x1xf32)
        t739 = paddle._C_ops.add(t737, t738)

        # pd_op.relu: (8x168x1x1xf32) <- (8x168x1x1xf32)
        t740 = paddle._C_ops.relu(t739)
        del t739

        # pd_op.conv2d: (8x672x1x1xf32) <- (8x168x1x1xf32, 672x168x1x1xf32)
        t741 = paddle._C_ops.conv2d(
            t740, t217, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t217

        # pd_op.reshape: (1x672x1x1xf32) <- (672xf32, 4xi64)
        t742 = paddle._C_ops.reshape(t218, t474)
        del t218

        # pd_op.add: (8x672x1x1xf32) <- (8x672x1x1xf32, 1x672x1x1xf32)
        t743 = paddle._C_ops.add(t741, t742)

        # pd_op.hardsigmoid: (8x672x1x1xf32) <- (8x672x1x1xf32)
        t744 = paddle._C_ops.hardsigmoid(t743, float("0.2"), float("0.5"))
        del t743

        # pd_op.multiply: (8x672x11x11xf32) <- (8x672x11x11xf32, 8x672x1x1xf32)
        t745 = paddle._C_ops.multiply(t735, t744)

        # pd_op.conv2d: (8x160x11x11xf32) <- (8x672x11x11xf32, 160x672x1x1xf32)
        t746 = paddle._C_ops.conv2d(
            t745, t219, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t219

        # pd_op.batch_norm_: (8x160x11x11xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (8x160x11x11xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t747, t748, t749, t750, t751, t752 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t746,
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

        # pd_op.conv2d: (8x960x11x11xf32) <- (8x160x11x11xf32, 960x160x1x1xf32)
        t753 = paddle._C_ops.conv2d(
            t747, t224, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t224

        # pd_op.batch_norm_: (8x960x11x11xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (8x960x11x11xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t754, t755, t756, t757, t758, t759 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t753,
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

        # pd_op.hardswish: (8x960x11x11xf32) <- (8x960x11x11xf32)
        t760 = paddle._C_ops.hardswish(t754)

        # pd_op.depthwise_conv2d: (8x960x11x11xf32) <- (8x960x11x11xf32, 960x1x5x5xf32)
        t761 = paddle._C_ops.depthwise_conv2d(
            t760, t229, [1, 1], [2, 2], "EXPLICIT", 960, [1, 1], "NCHW"
        )
        del t229

        # pd_op.batch_norm_: (8x960x11x11xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (8x960x11x11xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t762, t763, t764, t765, t766, t767 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t761,
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

        # pd_op.hardswish: (8x960x11x11xf32) <- (8x960x11x11xf32)
        t768 = paddle._C_ops.hardswish(t762)

        # pd_op.pool2d: (8x960x1x1xf32) <- (8x960x11x11xf32, 2xi64)
        t769 = paddle._C_ops.pool2d(
            t768,
            t464,
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

        # pd_op.conv2d: (8x240x1x1xf32) <- (8x960x1x1xf32, 240x960x1x1xf32)
        t770 = paddle._C_ops.conv2d(
            t769, t234, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t234

        # pd_op.reshape: (1x240x1x1xf32) <- (240xf32, 4xi64)
        t771 = paddle._C_ops.reshape(t235, t474)
        del t235

        # pd_op.add: (8x240x1x1xf32) <- (8x240x1x1xf32, 1x240x1x1xf32)
        t772 = paddle._C_ops.add(t770, t771)

        # pd_op.relu: (8x240x1x1xf32) <- (8x240x1x1xf32)
        t773 = paddle._C_ops.relu(t772)
        del t772

        # pd_op.conv2d: (8x960x1x1xf32) <- (8x240x1x1xf32, 960x240x1x1xf32)
        t774 = paddle._C_ops.conv2d(
            t773, t236, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t236

        # pd_op.reshape: (1x960x1x1xf32) <- (960xf32, 4xi64)
        t775 = paddle._C_ops.reshape(t237, t474)
        del t237

        # pd_op.add: (8x960x1x1xf32) <- (8x960x1x1xf32, 1x960x1x1xf32)
        t776 = paddle._C_ops.add(t774, t775)

        # pd_op.hardsigmoid: (8x960x1x1xf32) <- (8x960x1x1xf32)
        t777 = paddle._C_ops.hardsigmoid(t776, float("0.2"), float("0.5"))
        del t776

        # pd_op.multiply: (8x960x11x11xf32) <- (8x960x11x11xf32, 8x960x1x1xf32)
        t778 = paddle._C_ops.multiply(t768, t777)

        # pd_op.conv2d: (8x160x11x11xf32) <- (8x960x11x11xf32, 160x960x1x1xf32)
        t779 = paddle._C_ops.conv2d(
            t778, t238, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t238

        # pd_op.batch_norm_: (8x160x11x11xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (8x160x11x11xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t780, t781, t782, t783, t784, t785 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t779,
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

        # pd_op.add: (8x160x11x11xf32) <- (8x160x11x11xf32, 8x160x11x11xf32)
        t786 = paddle._C_ops.add(t747, t780)

        # pd_op.conv2d: (8x960x11x11xf32) <- (8x160x11x11xf32, 960x160x1x1xf32)
        t787 = paddle._C_ops.conv2d(
            t786, t243, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t243

        # pd_op.batch_norm_: (8x960x11x11xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (8x960x11x11xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t788, t789, t790, t791, t792, t793 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t787,
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

        # pd_op.hardswish: (8x960x11x11xf32) <- (8x960x11x11xf32)
        t794 = paddle._C_ops.hardswish(t788)

        # pd_op.depthwise_conv2d: (8x960x11x11xf32) <- (8x960x11x11xf32, 960x1x5x5xf32)
        t795 = paddle._C_ops.depthwise_conv2d(
            t794, t248, [1, 1], [2, 2], "EXPLICIT", 960, [1, 1], "NCHW"
        )
        del t248

        # pd_op.batch_norm_: (8x960x11x11xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (8x960x11x11xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t796, t797, t798, t799, t800, t801 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t795,
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

        # pd_op.hardswish: (8x960x11x11xf32) <- (8x960x11x11xf32)
        t802 = paddle._C_ops.hardswish(t796)

        # pd_op.pool2d: (8x960x1x1xf32) <- (8x960x11x11xf32, 2xi64)
        t803 = paddle._C_ops.pool2d(
            t802,
            t464,
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

        # pd_op.conv2d: (8x240x1x1xf32) <- (8x960x1x1xf32, 240x960x1x1xf32)
        t804 = paddle._C_ops.conv2d(
            t803, t253, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t253

        # pd_op.reshape: (1x240x1x1xf32) <- (240xf32, 4xi64)
        t805 = paddle._C_ops.reshape(t254, t474)
        del t254

        # pd_op.add: (8x240x1x1xf32) <- (8x240x1x1xf32, 1x240x1x1xf32)
        t806 = paddle._C_ops.add(t804, t805)

        # pd_op.relu: (8x240x1x1xf32) <- (8x240x1x1xf32)
        t807 = paddle._C_ops.relu(t806)
        del t806

        # pd_op.conv2d: (8x960x1x1xf32) <- (8x240x1x1xf32, 960x240x1x1xf32)
        t808 = paddle._C_ops.conv2d(
            t807, t255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255

        # pd_op.reshape: (1x960x1x1xf32) <- (960xf32, 4xi64)
        t809 = paddle._C_ops.reshape(t256, t474)
        del t256

        # pd_op.add: (8x960x1x1xf32) <- (8x960x1x1xf32, 1x960x1x1xf32)
        t810 = paddle._C_ops.add(t808, t809)

        # pd_op.hardsigmoid: (8x960x1x1xf32) <- (8x960x1x1xf32)
        t811 = paddle._C_ops.hardsigmoid(t810, float("0.2"), float("0.5"))
        del t810

        # pd_op.multiply: (8x960x11x11xf32) <- (8x960x11x11xf32, 8x960x1x1xf32)
        t812 = paddle._C_ops.multiply(t802, t811)

        # pd_op.conv2d: (8x160x11x11xf32) <- (8x960x11x11xf32, 160x960x1x1xf32)
        t813 = paddle._C_ops.conv2d(
            t812, t257, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t257

        # pd_op.batch_norm_: (8x160x11x11xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (8x160x11x11xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t814, t815, t816, t817, t818, t819 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t813,
                t258,
                t259,
                t260,
                t261,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t261, t260, t259, t258

        # pd_op.add: (8x160x11x11xf32) <- (8x160x11x11xf32, 8x160x11x11xf32)
        t820 = paddle._C_ops.add(t786, t814)

        # pd_op.conv2d: (8x512x11x11xf32) <- (8x160x11x11xf32, 512x160x1x1xf32)
        t821 = paddle._C_ops.conv2d(
            t820, t262, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t262

        # pd_op.batch_norm_: (8x512x11x11xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x11x11xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t822, t823, t824, t825, t826, t827 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t821,
                t263,
                t264,
                t265,
                t266,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t266, t265, t264, t263

        # pd_op.leaky_relu: (8x512x11x11xf32) <- (8x512x11x11xf32)
        t828 = paddle._C_ops.leaky_relu(t822, float("0.1"))

        # pd_op.conv2d: (8x1024x11x11xf32) <- (8x512x11x11xf32, 1024x512x3x3xf32)
        t829 = paddle._C_ops.conv2d(
            t828, t267, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t267

        # pd_op.batch_norm_: (8x1024x11x11xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (8x1024x11x11xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t830, t831, t832, t833, t834, t835 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t829,
                t268,
                t269,
                t270,
                t271,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t271, t270, t269, t268

        # pd_op.leaky_relu: (8x1024x11x11xf32) <- (8x1024x11x11xf32)
        t836 = paddle._C_ops.leaky_relu(t830, float("0.1"))

        # pd_op.conv2d: (8x512x11x11xf32) <- (8x1024x11x11xf32, 512x1024x1x1xf32)
        t837 = paddle._C_ops.conv2d(
            t836, t272, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t272

        # pd_op.batch_norm_: (8x512x11x11xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x11x11xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t838, t839, t840, t841, t842, t843 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t837,
                t273,
                t274,
                t275,
                t276,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t276, t275, t274, t273

        # pd_op.leaky_relu: (8x512x11x11xf32) <- (8x512x11x11xf32)
        t844 = paddle._C_ops.leaky_relu(t838, float("0.1"))

        # pd_op.conv2d: (8x1024x11x11xf32) <- (8x512x11x11xf32, 1024x512x3x3xf32)
        t845 = paddle._C_ops.conv2d(
            t844, t277, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t277

        # pd_op.batch_norm_: (8x1024x11x11xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (8x1024x11x11xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t846, t847, t848, t849, t850, t851 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t845,
                t278,
                t279,
                t280,
                t281,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t281, t280, t279, t278

        # pd_op.leaky_relu: (8x1024x11x11xf32) <- (8x1024x11x11xf32)
        t852 = paddle._C_ops.leaky_relu(t846, float("0.1"))

        # pd_op.conv2d: (8x512x11x11xf32) <- (8x1024x11x11xf32, 512x1024x1x1xf32)
        t853 = paddle._C_ops.conv2d(
            t852, t282, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t282

        # pd_op.batch_norm_: (8x512x11x11xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x11x11xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t854, t855, t856, t857, t858, t859 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t853,
                t283,
                t284,
                t285,
                t286,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t286, t285, t284, t283

        # pd_op.leaky_relu: (8x512x11x11xf32) <- (8x512x11x11xf32)
        t860 = paddle._C_ops.leaky_relu(t854, float("0.1"))

        # pd_op.conv2d: (8x1024x11x11xf32) <- (8x512x11x11xf32, 1024x512x3x3xf32)
        t861 = paddle._C_ops.conv2d(
            t860, t287, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t287

        # pd_op.batch_norm_: (8x1024x11x11xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (8x1024x11x11xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t862, t863, t864, t865, t866, t867 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t861,
                t288,
                t289,
                t290,
                t291,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t291, t290, t289, t288

        # pd_op.leaky_relu: (8x1024x11x11xf32) <- (8x1024x11x11xf32)
        t868 = paddle._C_ops.leaky_relu(t862, float("0.1"))

        # pd_op.conv2d: (8x256x11x11xf32) <- (8x512x11x11xf32, 256x512x1x1xf32)
        t869 = paddle._C_ops.conv2d(
            t860, t292, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t292

        # pd_op.batch_norm_: (8x256x11x11xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x11x11xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t870, t871, t872, t873, t874, t875 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t869,
                t293,
                t294,
                t295,
                t296,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t296, t295, t294, t293

        # pd_op.leaky_relu: (8x256x11x11xf32) <- (8x256x11x11xf32)
        t876 = paddle._C_ops.leaky_relu(t870, float("0.1"))

        # pd_op.nearest_interp: (8x256x22x22xf32) <- (8x256x11x11xf32, None, None, None)
        t877 = paddle._C_ops.nearest_interp(
            t876,
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
        t878 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t879 = t878

        # builtin.combine: ([8x256x22x22xf32, 8x112x22x22xf32]) <- (8x256x22x22xf32, 8x112x22x22xf32)
        t880 = [t877, t719]

        # pd_op.concat: (8x368x22x22xf32) <- ([8x256x22x22xf32, 8x112x22x22xf32], 1xi32)
        t881 = paddle._C_ops.concat(t880, t878)
        del t880

        # pd_op.conv2d: (8x256x22x22xf32) <- (8x368x22x22xf32, 256x368x1x1xf32)
        t882 = paddle._C_ops.conv2d(
            t881, t297, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t297

        # pd_op.batch_norm_: (8x256x22x22xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x22x22xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t883, t884, t885, t886, t887, t888 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t882,
                t298,
                t299,
                t300,
                t301,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t301, t300, t299, t298

        # pd_op.leaky_relu: (8x256x22x22xf32) <- (8x256x22x22xf32)
        t889 = paddle._C_ops.leaky_relu(t883, float("0.1"))

        # pd_op.conv2d: (8x512x22x22xf32) <- (8x256x22x22xf32, 512x256x3x3xf32)
        t890 = paddle._C_ops.conv2d(
            t889, t302, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t302

        # pd_op.batch_norm_: (8x512x22x22xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x22x22xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t891, t892, t893, t894, t895, t896 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t890,
                t303,
                t304,
                t305,
                t306,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t306, t305, t304, t303

        # pd_op.leaky_relu: (8x512x22x22xf32) <- (8x512x22x22xf32)
        t897 = paddle._C_ops.leaky_relu(t891, float("0.1"))

        # pd_op.conv2d: (8x256x22x22xf32) <- (8x512x22x22xf32, 256x512x1x1xf32)
        t898 = paddle._C_ops.conv2d(
            t897, t307, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t307

        # pd_op.batch_norm_: (8x256x22x22xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x22x22xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t899, t900, t901, t902, t903, t904 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t898,
                t308,
                t309,
                t310,
                t311,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t311, t310, t309, t308

        # pd_op.leaky_relu: (8x256x22x22xf32) <- (8x256x22x22xf32)
        t905 = paddle._C_ops.leaky_relu(t899, float("0.1"))

        # pd_op.conv2d: (8x512x22x22xf32) <- (8x256x22x22xf32, 512x256x3x3xf32)
        t906 = paddle._C_ops.conv2d(
            t905, t312, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t312

        # pd_op.batch_norm_: (8x512x22x22xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x22x22xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t907, t908, t909, t910, t911, t912 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t906,
                t313,
                t314,
                t315,
                t316,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t316, t315, t314, t313

        # pd_op.leaky_relu: (8x512x22x22xf32) <- (8x512x22x22xf32)
        t913 = paddle._C_ops.leaky_relu(t907, float("0.1"))

        # pd_op.conv2d: (8x256x22x22xf32) <- (8x512x22x22xf32, 256x512x1x1xf32)
        t914 = paddle._C_ops.conv2d(
            t913, t317, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t317

        # pd_op.batch_norm_: (8x256x22x22xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x22x22xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t915, t916, t917, t918, t919, t920 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t914,
                t318,
                t319,
                t320,
                t321,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t321, t320, t319, t318

        # pd_op.leaky_relu: (8x256x22x22xf32) <- (8x256x22x22xf32)
        t921 = paddle._C_ops.leaky_relu(t915, float("0.1"))

        # pd_op.conv2d: (8x512x22x22xf32) <- (8x256x22x22xf32, 512x256x3x3xf32)
        t922 = paddle._C_ops.conv2d(
            t921, t322, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t322

        # pd_op.batch_norm_: (8x512x22x22xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x22x22xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t923, t924, t925, t926, t927, t928 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t922,
                t323,
                t324,
                t325,
                t326,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t326, t325, t324, t323

        # pd_op.leaky_relu: (8x512x22x22xf32) <- (8x512x22x22xf32)
        t929 = paddle._C_ops.leaky_relu(t923, float("0.1"))

        # pd_op.conv2d: (8x128x22x22xf32) <- (8x256x22x22xf32, 128x256x1x1xf32)
        t930 = paddle._C_ops.conv2d(
            t921, t327, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t327

        # pd_op.batch_norm_: (8x128x22x22xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (8x128x22x22xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t931, t932, t933, t934, t935, t936 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t930,
                t328,
                t329,
                t330,
                t331,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t331, t330, t329, t328

        # pd_op.leaky_relu: (8x128x22x22xf32) <- (8x128x22x22xf32)
        t937 = paddle._C_ops.leaky_relu(t931, float("0.1"))

        # pd_op.nearest_interp: (8x128x44x44xf32) <- (8x128x22x22xf32, None, None, None)
        t938 = paddle._C_ops.nearest_interp(
            t937,
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

        # builtin.combine: ([8x128x44x44xf32, 8x40x44x44xf32]) <- (8x128x44x44xf32, 8x40x44x44xf32)
        t939 = [t938, t557]

        # pd_op.concat: (8x168x44x44xf32) <- ([8x128x44x44xf32, 8x40x44x44xf32], 1xi32)
        t940 = paddle._C_ops.concat(t939, t878)
        del t939

        # pd_op.conv2d: (8x128x44x44xf32) <- (8x168x44x44xf32, 128x168x1x1xf32)
        t941 = paddle._C_ops.conv2d(
            t940, t332, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t332

        # pd_op.batch_norm_: (8x128x44x44xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (8x128x44x44xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t942, t943, t944, t945, t946, t947 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t941,
                t333,
                t334,
                t335,
                t336,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t336, t335, t334, t333

        # pd_op.leaky_relu: (8x128x44x44xf32) <- (8x128x44x44xf32)
        t948 = paddle._C_ops.leaky_relu(t942, float("0.1"))

        # pd_op.conv2d: (8x256x44x44xf32) <- (8x128x44x44xf32, 256x128x3x3xf32)
        t949 = paddle._C_ops.conv2d(
            t948, t337, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t337

        # pd_op.batch_norm_: (8x256x44x44xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x44x44xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t950, t951, t952, t953, t954, t955 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t949,
                t338,
                t339,
                t340,
                t341,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t341, t340, t339, t338

        # pd_op.leaky_relu: (8x256x44x44xf32) <- (8x256x44x44xf32)
        t956 = paddle._C_ops.leaky_relu(t950, float("0.1"))

        # pd_op.conv2d: (8x128x44x44xf32) <- (8x256x44x44xf32, 128x256x1x1xf32)
        t957 = paddle._C_ops.conv2d(
            t956, t342, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t342

        # pd_op.batch_norm_: (8x128x44x44xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (8x128x44x44xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t958, t959, t960, t961, t962, t963 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t957,
                t343,
                t344,
                t345,
                t346,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t346, t345, t344, t343

        # pd_op.leaky_relu: (8x128x44x44xf32) <- (8x128x44x44xf32)
        t964 = paddle._C_ops.leaky_relu(t958, float("0.1"))

        # pd_op.conv2d: (8x256x44x44xf32) <- (8x128x44x44xf32, 256x128x3x3xf32)
        t965 = paddle._C_ops.conv2d(
            t964, t347, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t347

        # pd_op.batch_norm_: (8x256x44x44xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x44x44xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t966, t967, t968, t969, t970, t971 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t965,
                t348,
                t349,
                t350,
                t351,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t351, t350, t349, t348

        # pd_op.leaky_relu: (8x256x44x44xf32) <- (8x256x44x44xf32)
        t972 = paddle._C_ops.leaky_relu(t966, float("0.1"))

        # pd_op.conv2d: (8x128x44x44xf32) <- (8x256x44x44xf32, 128x256x1x1xf32)
        t973 = paddle._C_ops.conv2d(
            t972, t352, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t352

        # pd_op.batch_norm_: (8x128x44x44xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (8x128x44x44xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t974, t975, t976, t977, t978, t979 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t973,
                t353,
                t354,
                t355,
                t356,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t356, t355, t354, t353

        # pd_op.leaky_relu: (8x128x44x44xf32) <- (8x128x44x44xf32)
        t980 = paddle._C_ops.leaky_relu(t974, float("0.1"))

        # pd_op.conv2d: (8x256x44x44xf32) <- (8x128x44x44xf32, 256x128x3x3xf32)
        t981 = paddle._C_ops.conv2d(
            t980, t357, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t357

        # pd_op.batch_norm_: (8x256x44x44xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x44x44xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t982, t983, t984, t985, t986, t987 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t981,
                t358,
                t359,
                t360,
                t361,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t361, t360, t359, t358

        # pd_op.leaky_relu: (8x256x44x44xf32) <- (8x256x44x44xf32)
        t988 = paddle._C_ops.leaky_relu(t982, float("0.1"))

        # pd_op.conv2d: (8x27x11x11xf32) <- (8x1024x11x11xf32, 27x1024x1x1xf32)
        t989 = paddle._C_ops.conv2d(
            t868, t362, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t362

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        t990 = paddle._C_ops.reshape(t363, t474)
        del t363

        # pd_op.add: (8x27x11x11xf32) <- (8x27x11x11xf32, 1x27x1x1xf32)
        t991 = paddle._C_ops.add(t989, t990)

        # pd_op.conv2d: (8x27x22x22xf32) <- (8x512x22x22xf32, 27x512x1x1xf32)
        t992 = paddle._C_ops.conv2d(
            t929, t364, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t364

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        t993 = paddle._C_ops.reshape(t365, t474)
        del t365

        # pd_op.add: (8x27x22x22xf32) <- (8x27x22x22xf32, 1x27x1x1xf32)
        t994 = paddle._C_ops.add(t992, t993)

        # pd_op.conv2d: (8x27x44x44xf32) <- (8x256x44x44xf32, 27x256x1x1xf32)
        t995 = paddle._C_ops.conv2d(
            t988, t366, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t366

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        t996 = paddle._C_ops.reshape(t367, t474)
        del t474, t367

        # pd_op.add: (8x27x44x44xf32) <- (8x27x44x44xf32, 1x27x1x1xf32)
        t997 = paddle._C_ops.add(t995, t996)

        # pd_op.full_int_array: (5xi64) <- ()
        t998 = [8, 3, -1, 11, 11]

        # pd_op.reshape: (8x3x9x11x11xf32) <- (8x27x11x11xf32, 5xi64)
        t999 = paddle._C_ops.reshape(t991, t998)
        del t998

        # pd_op.transpose: (8x3x11x11x9xf32) <- (8x3x9x11x11xf32)
        t1000 = paddle._C_ops.transpose(t999, [0, 1, 3, 4, 2])
        del t999

        # pd_op.full_int_array: (1xi64) <- ()
        t1001 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1002 = t1001

        # pd_op.assign: (1xi64) <- (1xi64)
        t1003 = t1001

        # pd_op.full_int_array: (1xi64) <- ()
        t1004 = [1]

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

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x9xf32, 1xi64, 1xi64)
        t1010 = paddle._C_ops.slice(t1000, [4], t1001, t1004, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1011 = [2]

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

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x9xf32, 1xi64, 1xi64)
        t1017 = paddle._C_ops.slice(t1000, [4], t1004, t1011, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1018 = [3]

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

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x9xf32, 1xi64, 1xi64)
        t1024 = paddle._C_ops.slice(t1000, [4], t1011, t1018, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1025 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1026 = t1025

        # pd_op.assign: (1xi64) <- (1xi64)
        t1027 = t1025

        # pd_op.assign: (1xi64) <- (1xi64)
        t1028 = t1025

        # pd_op.assign: (1xi64) <- (1xi64)
        t1029 = t1025

        # pd_op.assign: (1xi64) <- (1xi64)
        t1030 = t1025

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x9xf32, 1xi64, 1xi64)
        t1031 = paddle._C_ops.slice(t1000, [4], t1018, t1025, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1032 = [5]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1033 = t1032

        # pd_op.assign: (1xi64) <- (1xi64)
        t1034 = t1032

        # pd_op.assign: (1xi64) <- (1xi64)
        t1035 = t1032

        # pd_op.assign: (1xi64) <- (1xi64)
        t1036 = t1032

        # pd_op.assign: (1xi64) <- (1xi64)
        t1037 = t1032

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x9xf32, 1xi64, 1xi64)
        t1038 = paddle._C_ops.slice(t1000, [4], t1025, t1032, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1039 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1040 = t1039

        # pd_op.assign: (1xi64) <- (1xi64)
        t1041 = t1039

        # pd_op.slice: (8x3x11x11x4xf32) <- (8x3x11x11x9xf32, 1xi64, 1xi64)
        t1042 = paddle._C_ops.slice(t1000, [4], t1032, t1039, [1], [])

        # pd_op.transpose: (8x3x11x11x10xf32) <- (8x3x10x11x11xf32)
        t1043 = paddle._C_ops.transpose(input1, [0, 1, 3, 4, 2])
        del input1

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x10xf32, 1xi64, 1xi64)
        t1044 = paddle._C_ops.slice(t1043, [4], t1001, t1004, [1], [])

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x10xf32, 1xi64, 1xi64)
        t1045 = paddle._C_ops.slice(t1043, [4], t1004, t1011, [1], [])

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x10xf32, 1xi64, 1xi64)
        t1046 = paddle._C_ops.slice(t1043, [4], t1011, t1018, [1], [])

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x10xf32, 1xi64, 1xi64)
        t1047 = paddle._C_ops.slice(t1043, [4], t1018, t1025, [1], [])

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x10xf32, 1xi64, 1xi64)
        t1048 = paddle._C_ops.slice(t1043, [4], t1025, t1032, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t1049 = [6]

        # pd_op.slice: (8x3x11x11x1xf32) <- (8x3x11x11x10xf32, 1xi64, 1xi64)
        t1050 = paddle._C_ops.slice(t1043, [4], t1032, t1049, [1], [])

        # pd_op.slice: (8x3x11x11x4xf32) <- (8x3x11x11x10xf32, 1xi64, 1xi64)
        t1051 = paddle._C_ops.slice(t1043, [4], t1049, t1039, [1], [])
        del t1043

        # pd_op.multiply: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1052 = paddle._C_ops.multiply(t1048, t1050)
        del t1048

        # pd_op.sigmoid: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32)
        t1053 = paddle._C_ops.sigmoid(t1010)

        # pd_op.full: (1xf32) <- ()
        t1054 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t1055 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1056 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1057 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1058 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1059 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1060 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1061 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1062 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1063 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1064 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1065 = t1054

        # pd_op.assign: (1xf32) <- (1xf32)
        t1066 = t1054

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1067 = paddle._C_ops.scale(t1053, t1054, float("0"), True)

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1068 = paddle._C_ops.scale(t1067, t1054, float("0"), True)
        del t1067

        # pd_op.sigmoid: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32)
        t1069 = paddle._C_ops.sigmoid(t1017)

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1070 = paddle._C_ops.scale(t1069, t1054, float("0"), True)

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1071 = paddle._C_ops.scale(t1070, t1054, float("0"), True)
        del t1070

        # pd_op.bce_loss: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1072 = paddle._C_ops.bce_loss(t1068, t1044)

        # pd_op.bce_loss: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1073 = paddle._C_ops.bce_loss(t1071, t1045)

        # pd_op.add: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1074 = paddle._C_ops.add(t1072, t1073)

        # pd_op.multiply: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1075 = paddle._C_ops.multiply(t1052, t1074)

        # pd_op.full_int_array: (4xi64) <- ()
        t1076 = [1, 2, 3, 4]

        # pd_op.assign: (4xi64) <- (4xi64)
        t1077 = t1076

        # pd_op.assign: (4xi64) <- (4xi64)
        t1078 = t1076

        # pd_op.assign: (4xi64) <- (4xi64)
        t1079 = t1076

        # pd_op.assign: (4xi64) <- (4xi64)
        t1080 = t1076

        # pd_op.assign: (4xi64) <- (4xi64)
        t1081 = t1076

        # pd_op.assign: (4xi64) <- (4xi64)
        t1082 = t1076

        # pd_op.assign: (4xi64) <- (4xi64)
        t1083 = t1076

        # pd_op.assign: (4xi64) <- (4xi64)
        t1084 = t1076

        # pd_op.sum: (8xf32) <- (8x3x11x11x1xf32, 4xi64)
        t1085 = paddle._C_ops.sum(t1075, t1076, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1086 = paddle._C_ops.mean(t1085, [], False)

        # pd_op.subtract: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1087 = paddle._C_ops.subtract(t1024, t1046)

        # pd_op.abs: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32)
        t1088 = paddle._C_ops.abs(t1087)

        # pd_op.subtract: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1089 = paddle._C_ops.subtract(t1031, t1047)

        # pd_op.abs: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32)
        t1090 = paddle._C_ops.abs(t1089)

        # pd_op.add: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1091 = paddle._C_ops.add(t1088, t1090)

        # pd_op.multiply: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1092 = paddle._C_ops.multiply(t1052, t1091)

        # pd_op.sum: (8xf32) <- (8x3x11x11x1xf32, 4xi64)
        t1093 = paddle._C_ops.sum(t1092, t1076, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1094 = paddle._C_ops.mean(t1093, [], False)

        # pd_op.full: (1xf64) <- ()
        t1095 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        t1096 = paddle._C_ops.full(
            [1], float("11"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        t1097 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (11xi64) <- (1xf64, 1xf64, 1xf64)
        t1098 = paddle.arange(t1095, t1096, t1097, dtype="int64")
        del t1096

        # builtin.combine: ([11xi64, 11xi64]) <- (11xi64, 11xi64)
        t1099 = [t1098, t1098]
        del t1098

        # pd_op.meshgrid: ([11x11xi64, 11x11xi64]) <- ([11xi64, 11xi64])
        t1100 = paddle._C_ops.meshgrid(t1099)
        del t1099

        # builtin.split: (11x11xi64, 11x11xi64) <- ([11x11xi64, 11x11xi64])
        (
            t1101,
            t1102,
        ) = t1100
        del t1100

        # builtin.combine: ([11x11xi64, 11x11xi64]) <- (11x11xi64, 11x11xi64)
        t1103 = [t1102, t1101]
        del t1101, t1102

        # pd_op.stack: (11x11x2xi64) <- ([11x11xi64, 11x11xi64])
        t1104 = paddle._C_ops.stack(t1103, 2)
        del t1103

        # pd_op.cast: (11x11x2xf32) <- (11x11x2xi64)
        t1105 = paddle._C_ops.cast(t1104, paddle.float32)
        del t1104

        # pd_op.full_int_array: (5xi64) <- ()
        t1106 = [1, 1, 11, 11, 2]

        # pd_op.reshape: (1x1x11x11x2xf32) <- (11x11x2xf32, 5xi64)
        t1107 = paddle._C_ops.reshape(t1105, t1106)
        del t1105, t1106

        # pd_op.slice: (1x1x11x11x1xf32) <- (1x1x11x11x2xf32, 1xi64, 1xi64)
        t1108 = paddle._C_ops.slice(t1107, [4], t1001, t1004, [1], [])

        # pd_op.add: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1x1x11x11x1xf32)
        t1109 = paddle._C_ops.add(t1068, t1108)
        del t1108

        # pd_op.full: (1xf32) <- ()
        t1110 = paddle._C_ops.full(
            [1], float("0.0909091"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1111 = paddle._C_ops.scale(t1109, t1110, float("0"), True)
        del t1109

        # pd_op.slice: (1x1x11x11x1xf32) <- (1x1x11x11x2xf32, 1xi64, 1xi64)
        t1112 = paddle._C_ops.slice(t1107, [4], t1004, t1039, [1], [])
        del t1107

        # pd_op.add: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1x1x11x11x1xf32)
        t1113 = paddle._C_ops.add(t1071, t1112)
        del t1112

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1114 = paddle._C_ops.scale(t1113, t1110, float("0"), True)
        del t1113, t1110

        # pd_op.full: (3x2xi64) <- ()
        t1115 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        t1116 = paddle._C_ops.assign_value_(
            t1115,
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
        del t1115

        # pd_op.cast: (3x2xf32) <- (3x2xi64)
        t1117 = paddle._C_ops.cast(t1116, paddle.float32)
        del t1116

        # pd_op.full_int_array: (5xi64) <- ()
        t1118 = [1, 3, 1, 1, 2]

        # pd_op.reshape: (1x3x1x1x2xf32) <- (3x2xf32, 5xi64)
        t1119 = paddle._C_ops.reshape(t1117, t1118)
        del t1117

        # pd_op.exp: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32)
        t1120 = paddle._C_ops.exp(t1024)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1121 = paddle._C_ops.slice(t1119, [4], t1001, t1004, [1], [])

        # pd_op.multiply: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1x3x1x1x1xf32)
        t1122 = paddle._C_ops.multiply(t1120, t1121)
        del t1120, t1121

        # pd_op.full: (1xf32) <- ()
        t1123 = paddle._C_ops.full(
            [1], float("0.00284091"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1124 = paddle._C_ops.scale(t1122, t1123, float("0"), True)
        del t1122

        # pd_op.exp: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32)
        t1125 = paddle._C_ops.exp(t1031)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1126 = paddle._C_ops.slice(t1119, [4], t1004, t1039, [1], [])
        del t1119

        # pd_op.multiply: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1x3x1x1x1xf32)
        t1127 = paddle._C_ops.multiply(t1125, t1126)
        del t1125, t1126

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1128 = paddle._C_ops.scale(t1127, t1123, float("0"), True)
        del t1127

        # pd_op.full: (1xf32) <- ()
        t1129 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1130 = paddle._C_ops.scale(t1124, t1129, float("0"), True)

        # pd_op.subtract: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1131 = paddle._C_ops.subtract(t1111, t1130)
        del t1130

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1132 = paddle._C_ops.scale(t1128, t1129, float("0"), True)

        # pd_op.subtract: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1133 = paddle._C_ops.subtract(t1114, t1132)
        del t1132

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1134 = paddle._C_ops.scale(t1124, t1129, float("0"), True)
        del t1124

        # pd_op.add: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1135 = paddle._C_ops.add(t1111, t1134)
        del t1111, t1134

        # pd_op.scale: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 1xf32)
        t1136 = paddle._C_ops.scale(t1128, t1129, float("0"), True)
        del t1128

        # pd_op.add: (8x3x11x11x1xf32) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1137 = paddle._C_ops.add(t1114, t1136)
        del t1114, t1136

        # pd_op.full: (1xi32) <- ()
        t1138 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([8x3x11x11x1xf32, 8x3x11x11x1xf32, 8x3x11x11x1xf32, 8x3x11x11x1xf32]) <- (8x3x11x11x1xf32, 8x3x11x11x1xf32, 8x3x11x11x1xf32, 8x3x11x11x1xf32)
        t1139 = [t1131, t1133, t1135, t1137]
        del t1135, t1137, t1131, t1133

        # pd_op.concat: (8x3x11x11x4xf32) <- ([8x3x11x11x1xf32, 8x3x11x11x1xf32, 8x3x11x11x1xf32, 8x3x11x11x1xf32], 1xi32)
        t1140 = paddle._C_ops.concat(t1139, t1138)
        del t1139

        # pd_op.full_int_array: (3xi64) <- ()
        t1141 = [8, -1, 4]

        # pd_op.reshape: (8x363x4xf32) <- (8x3x11x11x4xf32, 3xi64)
        t1142 = paddle._C_ops.reshape(t1140, t1141)
        del t1140

        # pd_op.slice: (8x50x2xf32) <- (8x50x4xf32, 1xi64, 1xi64)
        t1143 = paddle._C_ops.slice(input2, [2], t1001, t1011, [1], [])

        # pd_op.slice: (8x50x2xf32) <- (8x50x4xf32, 1xi64, 1xi64)
        t1144 = paddle._C_ops.slice(input2, [2], t1011, t1039, [1], [])
        del input2

        # pd_op.scale: (8x50x2xf32) <- (8x50x2xf32, 1xf32)
        t1145 = paddle._C_ops.scale(t1144, t1129, float("0"), True)
        del t1144

        # pd_op.subtract: (8x50x2xf32) <- (8x50x2xf32, 8x50x2xf32)
        t1146 = paddle._C_ops.subtract(t1143, t1145)

        # pd_op.add: (8x50x2xf32) <- (8x50x2xf32, 8x50x2xf32)
        t1147 = paddle._C_ops.add(t1143, t1145)
        del t1145, t1143

        # builtin.combine: ([8x50x2xf32, 8x50x2xf32]) <- (8x50x2xf32, 8x50x2xf32)
        t1148 = [t1146, t1147]
        del t1147, t1146

        # pd_op.concat: (8x50x4xf32) <- ([8x50x2xf32, 8x50x2xf32], 1xi32)
        t1149 = paddle._C_ops.concat(t1148, t1138)
        del t1148

        # pd_op.unsqueeze: (8x363x1x4xf32) <- (8x363x4xf32, 1xi64)
        t1150 = paddle._C_ops.unsqueeze(t1142, t1011)
        del t1142

        # pd_op.unsqueeze: (8x1x50x4xf32) <- (8x50x4xf32, 1xi64)
        t1151 = paddle._C_ops.unsqueeze(t1149, t1004)
        del t1149

        # pd_op.slice: (8x363x1x2xf32) <- (8x363x1x4xf32, 1xi64, 1xi64)
        t1152 = paddle._C_ops.slice(t1150, [3], t1001, t1011, [1], [])

        # pd_op.slice: (8x363x1x2xf32) <- (8x363x1x4xf32, 1xi64, 1xi64)
        t1153 = paddle._C_ops.slice(t1150, [3], t1011, t1039, [1], [])
        del t1150

        # pd_op.slice: (8x1x50x2xf32) <- (8x1x50x4xf32, 1xi64, 1xi64)
        t1154 = paddle._C_ops.slice(t1151, [3], t1001, t1011, [1], [])

        # pd_op.slice: (8x1x50x2xf32) <- (8x1x50x4xf32, 1xi64, 1xi64)
        t1155 = paddle._C_ops.slice(t1151, [3], t1011, t1039, [1], [])
        del t1151

        # pd_op.maximum: (8x363x50x2xf32) <- (8x363x1x2xf32, 8x1x50x2xf32)
        t1156 = paddle._C_ops.maximum(t1152, t1154)

        # pd_op.minimum: (8x363x50x2xf32) <- (8x363x1x2xf32, 8x1x50x2xf32)
        t1157 = paddle._C_ops.minimum(t1153, t1155)

        # pd_op.subtract: (8x363x50x2xf32) <- (8x363x50x2xf32, 8x363x50x2xf32)
        t1158 = paddle._C_ops.subtract(t1157, t1156)
        del t1156, t1157

        # pd_op.full: (1xf32) <- ()
        t1159 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t1160 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (8x363x50x2xf32) <- (8x363x50x2xf32, 1xf32, 1xf32)
        t1161 = paddle._C_ops.clip(t1158, t1159, t1160)
        del t1158

        # pd_op.full_int_array: (1xi64) <- ()
        t1162 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1163 = t1162

        # pd_op.assign: (1xi64) <- (1xi64)
        t1164 = t1162

        # pd_op.assign: (1xi64) <- (1xi64)
        t1165 = t1162

        # pd_op.prod: (8x363x50xf32) <- (8x363x50x2xf32, 1xi64)
        t1166 = paddle._C_ops.prod(t1161, t1162, False, False)
        del t1161

        # pd_op.subtract: (8x363x1x2xf32) <- (8x363x1x2xf32, 8x363x1x2xf32)
        t1167 = paddle._C_ops.subtract(t1153, t1152)
        del t1152, t1153

        # pd_op.clip: (8x363x1x2xf32) <- (8x363x1x2xf32, 1xf32, 1xf32)
        t1168 = paddle._C_ops.clip(t1167, t1159, t1160)
        del t1167

        # pd_op.prod: (8x363x1xf32) <- (8x363x1x2xf32, 1xi64)
        t1169 = paddle._C_ops.prod(t1168, t1162, False, False)
        del t1168

        # pd_op.subtract: (8x1x50x2xf32) <- (8x1x50x2xf32, 8x1x50x2xf32)
        t1170 = paddle._C_ops.subtract(t1155, t1154)

        # pd_op.clip: (8x1x50x2xf32) <- (8x1x50x2xf32, 1xf32, 1xf32)
        t1171 = paddle._C_ops.clip(t1170, t1159, t1160)
        del t1170

        # pd_op.prod: (8x1x50xf32) <- (8x1x50x2xf32, 1xi64)
        t1172 = paddle._C_ops.prod(t1171, t1162, False, False)
        del t1171

        # pd_op.add: (8x363x50xf32) <- (8x363x1xf32, 8x1x50xf32)
        t1173 = paddle._C_ops.add(t1169, t1172)
        del t1169

        # pd_op.subtract: (8x363x50xf32) <- (8x363x50xf32, 8x363x50xf32)
        t1174 = paddle._C_ops.subtract(t1173, t1166)
        del t1173

        # pd_op.scale: (8x363x50xf32) <- (8x363x50xf32, 1xf32)
        t1175 = paddle._C_ops.scale(t1174, t1054, float("1e-09"), True)
        del t1174

        # pd_op.divide: (8x363x50xf32) <- (8x363x50xf32, 8x363x50xf32)
        t1176 = paddle._C_ops.divide(t1166, t1175)
        del t1166, t1175

        # pd_op.max: (8x363xf32) <- (8x363x50xf32, 1xi64)
        t1177 = paddle._C_ops.max(t1176, t1011, False)
        del t1176

        # pd_op.full: (xf32) <- ()
        t1178 = paddle._C_ops.full(
            [], float("0.7"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.less_equal: (8x363xb) <- (8x363xf32, xf32)
        t1179 = paddle._C_ops.less_equal(t1177, t1178)
        del t1177

        # pd_op.cast: (8x363xf32) <- (8x363xb)
        t1180 = paddle._C_ops.cast(t1179, paddle.float32)
        del t1179

        # pd_op.full_int_array: (2xi64) <- ()
        t1181 = [8, -1]

        # pd_op.reshape: (8x363xf32) <- (8x3x11x11x1xf32, 2xi64)
        t1182 = paddle._C_ops.reshape(t1038, t1181)

        # pd_op.reshape: (8x363xf32) <- (8x3x11x11x1xf32, 2xi64)
        t1183 = paddle._C_ops.reshape(t1050, t1181)

        # pd_op.full: (xf32) <- ()
        t1184 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (8x363xb) <- (8x363xf32, xf32)
        t1185 = paddle._C_ops.greater_than(t1183, t1184)

        # pd_op.cast: (8x363xf32) <- (8x363xb)
        t1186 = paddle._C_ops.cast(t1185, paddle.float32)
        del t1185

        # pd_op.full: (1xf32) <- ()
        t1187 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.sigmoid_cross_entropy_with_logits: (8x363xf32) <- (8x363xf32, 8x363xf32, None)
        t1188 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1182, t1186, None, False, -100
        )

        # pd_op.multiply: (8x363xf32) <- (8x363xf32, 8x363xf32)
        t1189 = paddle._C_ops.multiply(t1188, t1183)

        # pd_op.full: (1xf32) <- ()
        t1190 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x363xf32) <- (8x363xf32, 1xf32)
        t1191 = paddle._C_ops.scale(t1186, t1190, float("1"), True)

        # pd_op.multiply: (8x363xf32) <- (8x363xf32, 8x363xf32)
        t1192 = paddle._C_ops.multiply(t1188, t1191)

        # pd_op.multiply: (8x363xf32) <- (8x363xf32, 8x363xf32)
        t1193 = paddle._C_ops.multiply(t1192, t1180)

        # pd_op.add: (8x363xf32) <- (8x363xf32, 8x363xf32)
        t1194 = paddle._C_ops.add(t1189, t1193)

        # pd_op.sum: (8xf32) <- (8x363xf32, 1xi64)
        t1195 = paddle._C_ops.sum(t1194, t1162, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1196 = paddle._C_ops.mean(t1195, [], False)

        # pd_op.sigmoid_cross_entropy_with_logits: (8x3x11x11x4xf32) <- (8x3x11x11x4xf32, 8x3x11x11x4xf32, None)
        t1197 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1042, t1051, None, False, -100
        )

        # pd_op.multiply: (8x3x11x11x4xf32) <- (8x3x11x11x4xf32, 8x3x11x11x1xf32)
        t1198 = paddle._C_ops.multiply(t1197, t1050)

        # pd_op.sum: (8xf32) <- (8x3x11x11x4xf32, 4xi64)
        t1199 = paddle._C_ops.sum(t1198, t1076, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1200 = paddle._C_ops.mean(t1199, [], False)

        # pd_op.full_int_array: (5xi64) <- ()
        t1201 = [8, 3, -1, 22, 22]

        # pd_op.reshape: (8x3x9x22x22xf32) <- (8x27x22x22xf32, 5xi64)
        t1202 = paddle._C_ops.reshape(t994, t1201)
        del t1201

        # pd_op.transpose: (8x3x22x22x9xf32) <- (8x3x9x22x22xf32)
        t1203 = paddle._C_ops.transpose(t1202, [0, 1, 3, 4, 2])
        del t1202

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x9xf32, 1xi64, 1xi64)
        t1204 = paddle._C_ops.slice(t1203, [4], t1001, t1004, [1], [])

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x9xf32, 1xi64, 1xi64)
        t1205 = paddle._C_ops.slice(t1203, [4], t1004, t1011, [1], [])

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x9xf32, 1xi64, 1xi64)
        t1206 = paddle._C_ops.slice(t1203, [4], t1011, t1018, [1], [])

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x9xf32, 1xi64, 1xi64)
        t1207 = paddle._C_ops.slice(t1203, [4], t1018, t1025, [1], [])

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x9xf32, 1xi64, 1xi64)
        t1208 = paddle._C_ops.slice(t1203, [4], t1025, t1032, [1], [])

        # pd_op.slice: (8x3x22x22x4xf32) <- (8x3x22x22x9xf32, 1xi64, 1xi64)
        t1209 = paddle._C_ops.slice(t1203, [4], t1032, t1039, [1], [])

        # pd_op.transpose: (8x3x22x22x10xf32) <- (8x3x10x22x22xf32)
        t1210 = paddle._C_ops.transpose(input3, [0, 1, 3, 4, 2])
        del input3

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x10xf32, 1xi64, 1xi64)
        t1211 = paddle._C_ops.slice(t1210, [4], t1001, t1004, [1], [])

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x10xf32, 1xi64, 1xi64)
        t1212 = paddle._C_ops.slice(t1210, [4], t1004, t1011, [1], [])

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x10xf32, 1xi64, 1xi64)
        t1213 = paddle._C_ops.slice(t1210, [4], t1011, t1018, [1], [])

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x10xf32, 1xi64, 1xi64)
        t1214 = paddle._C_ops.slice(t1210, [4], t1018, t1025, [1], [])

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x10xf32, 1xi64, 1xi64)
        t1215 = paddle._C_ops.slice(t1210, [4], t1025, t1032, [1], [])

        # pd_op.slice: (8x3x22x22x1xf32) <- (8x3x22x22x10xf32, 1xi64, 1xi64)
        t1216 = paddle._C_ops.slice(t1210, [4], t1032, t1049, [1], [])

        # pd_op.slice: (8x3x22x22x4xf32) <- (8x3x22x22x10xf32, 1xi64, 1xi64)
        t1217 = paddle._C_ops.slice(t1210, [4], t1049, t1039, [1], [])
        del t1210

        # pd_op.multiply: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1218 = paddle._C_ops.multiply(t1215, t1216)
        del t1215

        # pd_op.sigmoid: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32)
        t1219 = paddle._C_ops.sigmoid(t1204)

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1220 = paddle._C_ops.scale(t1219, t1054, float("0"), True)

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1221 = paddle._C_ops.scale(t1220, t1054, float("0"), True)
        del t1220

        # pd_op.sigmoid: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32)
        t1222 = paddle._C_ops.sigmoid(t1205)

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1223 = paddle._C_ops.scale(t1222, t1054, float("0"), True)

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1224 = paddle._C_ops.scale(t1223, t1054, float("0"), True)
        del t1223

        # pd_op.bce_loss: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1225 = paddle._C_ops.bce_loss(t1221, t1211)

        # pd_op.bce_loss: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1226 = paddle._C_ops.bce_loss(t1224, t1212)

        # pd_op.add: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1227 = paddle._C_ops.add(t1225, t1226)

        # pd_op.multiply: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1228 = paddle._C_ops.multiply(t1218, t1227)

        # pd_op.sum: (8xf32) <- (8x3x22x22x1xf32, 4xi64)
        t1229 = paddle._C_ops.sum(t1228, t1076, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1230 = paddle._C_ops.mean(t1229, [], False)

        # pd_op.subtract: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1231 = paddle._C_ops.subtract(t1206, t1213)

        # pd_op.abs: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32)
        t1232 = paddle._C_ops.abs(t1231)

        # pd_op.subtract: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1233 = paddle._C_ops.subtract(t1207, t1214)

        # pd_op.abs: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32)
        t1234 = paddle._C_ops.abs(t1233)

        # pd_op.add: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1235 = paddle._C_ops.add(t1232, t1234)

        # pd_op.multiply: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1236 = paddle._C_ops.multiply(t1218, t1235)

        # pd_op.sum: (8xf32) <- (8x3x22x22x1xf32, 4xi64)
        t1237 = paddle._C_ops.sum(t1236, t1076, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1238 = paddle._C_ops.mean(t1237, [], False)

        # pd_op.full: (1xf64) <- ()
        t1239 = paddle._C_ops.full(
            [1], float("22"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (22xi64) <- (1xf64, 1xf64, 1xf64)
        t1240 = paddle.arange(t1095, t1239, t1097, dtype="int64")
        del t1239

        # builtin.combine: ([22xi64, 22xi64]) <- (22xi64, 22xi64)
        t1241 = [t1240, t1240]
        del t1240

        # pd_op.meshgrid: ([22x22xi64, 22x22xi64]) <- ([22xi64, 22xi64])
        t1242 = paddle._C_ops.meshgrid(t1241)
        del t1241

        # builtin.split: (22x22xi64, 22x22xi64) <- ([22x22xi64, 22x22xi64])
        (
            t1243,
            t1244,
        ) = t1242
        del t1242

        # builtin.combine: ([22x22xi64, 22x22xi64]) <- (22x22xi64, 22x22xi64)
        t1245 = [t1244, t1243]
        del t1243, t1244

        # pd_op.stack: (22x22x2xi64) <- ([22x22xi64, 22x22xi64])
        t1246 = paddle._C_ops.stack(t1245, 2)
        del t1245

        # pd_op.cast: (22x22x2xf32) <- (22x22x2xi64)
        t1247 = paddle._C_ops.cast(t1246, paddle.float32)
        del t1246

        # pd_op.full_int_array: (5xi64) <- ()
        t1248 = [1, 1, 22, 22, 2]

        # pd_op.reshape: (1x1x22x22x2xf32) <- (22x22x2xf32, 5xi64)
        t1249 = paddle._C_ops.reshape(t1247, t1248)
        del t1247, t1248

        # pd_op.slice: (1x1x22x22x1xf32) <- (1x1x22x22x2xf32, 1xi64, 1xi64)
        t1250 = paddle._C_ops.slice(t1249, [4], t1001, t1004, [1], [])

        # pd_op.add: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1x1x22x22x1xf32)
        t1251 = paddle._C_ops.add(t1221, t1250)
        del t1250

        # pd_op.full: (1xf32) <- ()
        t1252 = paddle._C_ops.full(
            [1], float("0.0454545"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1253 = paddle._C_ops.scale(t1251, t1252, float("0"), True)
        del t1251

        # pd_op.slice: (1x1x22x22x1xf32) <- (1x1x22x22x2xf32, 1xi64, 1xi64)
        t1254 = paddle._C_ops.slice(t1249, [4], t1004, t1039, [1], [])
        del t1249

        # pd_op.add: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1x1x22x22x1xf32)
        t1255 = paddle._C_ops.add(t1224, t1254)
        del t1254

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1256 = paddle._C_ops.scale(t1255, t1252, float("0"), True)
        del t1255, t1252

        # pd_op.full: (3x2xi64) <- ()
        t1257 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        t1258 = paddle._C_ops.assign_value_(
            t1257,
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
        del t1257

        # pd_op.cast: (3x2xf32) <- (3x2xi64)
        t1259 = paddle._C_ops.cast(t1258, paddle.float32)
        del t1258

        # pd_op.reshape: (1x3x1x1x2xf32) <- (3x2xf32, 5xi64)
        t1260 = paddle._C_ops.reshape(t1259, t1118)
        del t1259

        # pd_op.exp: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32)
        t1261 = paddle._C_ops.exp(t1206)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1262 = paddle._C_ops.slice(t1260, [4], t1001, t1004, [1], [])

        # pd_op.multiply: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1x3x1x1x1xf32)
        t1263 = paddle._C_ops.multiply(t1261, t1262)
        del t1261, t1262

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1264 = paddle._C_ops.scale(t1263, t1123, float("0"), True)
        del t1263

        # pd_op.exp: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32)
        t1265 = paddle._C_ops.exp(t1207)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1266 = paddle._C_ops.slice(t1260, [4], t1004, t1039, [1], [])
        del t1260

        # pd_op.multiply: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1x3x1x1x1xf32)
        t1267 = paddle._C_ops.multiply(t1265, t1266)
        del t1265, t1266

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1268 = paddle._C_ops.scale(t1267, t1123, float("0"), True)
        del t1267

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1269 = paddle._C_ops.scale(t1264, t1129, float("0"), True)

        # pd_op.subtract: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1270 = paddle._C_ops.subtract(t1253, t1269)
        del t1269

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1271 = paddle._C_ops.scale(t1268, t1129, float("0"), True)

        # pd_op.subtract: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1272 = paddle._C_ops.subtract(t1256, t1271)
        del t1271

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1273 = paddle._C_ops.scale(t1264, t1129, float("0"), True)
        del t1264

        # pd_op.add: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1274 = paddle._C_ops.add(t1253, t1273)
        del t1253, t1273

        # pd_op.scale: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 1xf32)
        t1275 = paddle._C_ops.scale(t1268, t1129, float("0"), True)
        del t1268

        # pd_op.add: (8x3x22x22x1xf32) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1276 = paddle._C_ops.add(t1256, t1275)
        del t1256, t1275

        # builtin.combine: ([8x3x22x22x1xf32, 8x3x22x22x1xf32, 8x3x22x22x1xf32, 8x3x22x22x1xf32]) <- (8x3x22x22x1xf32, 8x3x22x22x1xf32, 8x3x22x22x1xf32, 8x3x22x22x1xf32)
        t1277 = [t1270, t1272, t1274, t1276]
        del t1274, t1276, t1270, t1272

        # pd_op.concat: (8x3x22x22x4xf32) <- ([8x3x22x22x1xf32, 8x3x22x22x1xf32, 8x3x22x22x1xf32, 8x3x22x22x1xf32], 1xi32)
        t1278 = paddle._C_ops.concat(t1277, t1138)
        del t1277

        # pd_op.reshape: (8x1452x4xf32) <- (8x3x22x22x4xf32, 3xi64)
        t1279 = paddle._C_ops.reshape(t1278, t1141)
        del t1278

        # pd_op.unsqueeze: (8x1452x1x4xf32) <- (8x1452x4xf32, 1xi64)
        t1280 = paddle._C_ops.unsqueeze(t1279, t1011)
        del t1279

        # pd_op.slice: (8x1452x1x2xf32) <- (8x1452x1x4xf32, 1xi64, 1xi64)
        t1281 = paddle._C_ops.slice(t1280, [3], t1001, t1011, [1], [])

        # pd_op.slice: (8x1452x1x2xf32) <- (8x1452x1x4xf32, 1xi64, 1xi64)
        t1282 = paddle._C_ops.slice(t1280, [3], t1011, t1039, [1], [])
        del t1280

        # pd_op.maximum: (8x1452x50x2xf32) <- (8x1452x1x2xf32, 8x1x50x2xf32)
        t1283 = paddle._C_ops.maximum(t1281, t1154)

        # pd_op.minimum: (8x1452x50x2xf32) <- (8x1452x1x2xf32, 8x1x50x2xf32)
        t1284 = paddle._C_ops.minimum(t1282, t1155)

        # pd_op.subtract: (8x1452x50x2xf32) <- (8x1452x50x2xf32, 8x1452x50x2xf32)
        t1285 = paddle._C_ops.subtract(t1284, t1283)
        del t1283, t1284

        # pd_op.clip: (8x1452x50x2xf32) <- (8x1452x50x2xf32, 1xf32, 1xf32)
        t1286 = paddle._C_ops.clip(t1285, t1159, t1160)
        del t1285

        # pd_op.prod: (8x1452x50xf32) <- (8x1452x50x2xf32, 1xi64)
        t1287 = paddle._C_ops.prod(t1286, t1162, False, False)
        del t1286

        # pd_op.subtract: (8x1452x1x2xf32) <- (8x1452x1x2xf32, 8x1452x1x2xf32)
        t1288 = paddle._C_ops.subtract(t1282, t1281)
        del t1281, t1282

        # pd_op.clip: (8x1452x1x2xf32) <- (8x1452x1x2xf32, 1xf32, 1xf32)
        t1289 = paddle._C_ops.clip(t1288, t1159, t1160)
        del t1288

        # pd_op.prod: (8x1452x1xf32) <- (8x1452x1x2xf32, 1xi64)
        t1290 = paddle._C_ops.prod(t1289, t1162, False, False)
        del t1289

        # pd_op.add: (8x1452x50xf32) <- (8x1452x1xf32, 8x1x50xf32)
        t1291 = paddle._C_ops.add(t1290, t1172)
        del t1290

        # pd_op.subtract: (8x1452x50xf32) <- (8x1452x50xf32, 8x1452x50xf32)
        t1292 = paddle._C_ops.subtract(t1291, t1287)
        del t1291

        # pd_op.scale: (8x1452x50xf32) <- (8x1452x50xf32, 1xf32)
        t1293 = paddle._C_ops.scale(t1292, t1054, float("1e-09"), True)
        del t1292

        # pd_op.divide: (8x1452x50xf32) <- (8x1452x50xf32, 8x1452x50xf32)
        t1294 = paddle._C_ops.divide(t1287, t1293)
        del t1287, t1293

        # pd_op.max: (8x1452xf32) <- (8x1452x50xf32, 1xi64)
        t1295 = paddle._C_ops.max(t1294, t1011, False)
        del t1294

        # pd_op.less_equal: (8x1452xb) <- (8x1452xf32, xf32)
        t1296 = paddle._C_ops.less_equal(t1295, t1178)
        del t1295

        # pd_op.cast: (8x1452xf32) <- (8x1452xb)
        t1297 = paddle._C_ops.cast(t1296, paddle.float32)
        del t1296

        # pd_op.reshape: (8x1452xf32) <- (8x3x22x22x1xf32, 2xi64)
        t1298 = paddle._C_ops.reshape(t1208, t1181)

        # pd_op.reshape: (8x1452xf32) <- (8x3x22x22x1xf32, 2xi64)
        t1299 = paddle._C_ops.reshape(t1216, t1181)

        # pd_op.greater_than: (8x1452xb) <- (8x1452xf32, xf32)
        t1300 = paddle._C_ops.greater_than(t1299, t1184)

        # pd_op.cast: (8x1452xf32) <- (8x1452xb)
        t1301 = paddle._C_ops.cast(t1300, paddle.float32)
        del t1300

        # pd_op.sigmoid_cross_entropy_with_logits: (8x1452xf32) <- (8x1452xf32, 8x1452xf32, None)
        t1302 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1298, t1301, None, False, -100
        )

        # pd_op.multiply: (8x1452xf32) <- (8x1452xf32, 8x1452xf32)
        t1303 = paddle._C_ops.multiply(t1302, t1299)

        # pd_op.scale: (8x1452xf32) <- (8x1452xf32, 1xf32)
        t1304 = paddle._C_ops.scale(t1301, t1190, float("1"), True)

        # pd_op.multiply: (8x1452xf32) <- (8x1452xf32, 8x1452xf32)
        t1305 = paddle._C_ops.multiply(t1302, t1304)

        # pd_op.multiply: (8x1452xf32) <- (8x1452xf32, 8x1452xf32)
        t1306 = paddle._C_ops.multiply(t1305, t1297)

        # pd_op.add: (8x1452xf32) <- (8x1452xf32, 8x1452xf32)
        t1307 = paddle._C_ops.add(t1303, t1306)

        # pd_op.sum: (8xf32) <- (8x1452xf32, 1xi64)
        t1308 = paddle._C_ops.sum(t1307, t1162, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1309 = paddle._C_ops.mean(t1308, [], False)

        # pd_op.sigmoid_cross_entropy_with_logits: (8x3x22x22x4xf32) <- (8x3x22x22x4xf32, 8x3x22x22x4xf32, None)
        t1310 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1209, t1217, None, False, -100
        )

        # pd_op.multiply: (8x3x22x22x4xf32) <- (8x3x22x22x4xf32, 8x3x22x22x1xf32)
        t1311 = paddle._C_ops.multiply(t1310, t1216)

        # pd_op.sum: (8xf32) <- (8x3x22x22x4xf32, 4xi64)
        t1312 = paddle._C_ops.sum(t1311, t1076, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1313 = paddle._C_ops.mean(t1312, [], False)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1314 = paddle._C_ops.add(t1086, t1230)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1315 = paddle._C_ops.add(t1094, t1238)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1316 = paddle._C_ops.add(t1196, t1309)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1317 = paddle._C_ops.add(t1200, t1313)

        # pd_op.full_int_array: (5xi64) <- ()
        t1318 = [8, 3, -1, 44, 44]

        # pd_op.reshape: (8x3x9x44x44xf32) <- (8x27x44x44xf32, 5xi64)
        t1319 = paddle._C_ops.reshape(t997, t1318)
        del t1318

        # pd_op.transpose: (8x3x44x44x9xf32) <- (8x3x9x44x44xf32)
        t1320 = paddle._C_ops.transpose(t1319, [0, 1, 3, 4, 2])
        del t1319

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x9xf32, 1xi64, 1xi64)
        t1321 = paddle._C_ops.slice(t1320, [4], t1001, t1004, [1], [])

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x9xf32, 1xi64, 1xi64)
        t1322 = paddle._C_ops.slice(t1320, [4], t1004, t1011, [1], [])

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x9xf32, 1xi64, 1xi64)
        t1323 = paddle._C_ops.slice(t1320, [4], t1011, t1018, [1], [])

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x9xf32, 1xi64, 1xi64)
        t1324 = paddle._C_ops.slice(t1320, [4], t1018, t1025, [1], [])

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x9xf32, 1xi64, 1xi64)
        t1325 = paddle._C_ops.slice(t1320, [4], t1025, t1032, [1], [])

        # pd_op.slice: (8x3x44x44x4xf32) <- (8x3x44x44x9xf32, 1xi64, 1xi64)
        t1326 = paddle._C_ops.slice(t1320, [4], t1032, t1039, [1], [])

        # pd_op.transpose: (8x3x44x44x10xf32) <- (8x3x10x44x44xf32)
        t1327 = paddle._C_ops.transpose(input4, [0, 1, 3, 4, 2])
        del input4

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x10xf32, 1xi64, 1xi64)
        t1328 = paddle._C_ops.slice(t1327, [4], t1001, t1004, [1], [])

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x10xf32, 1xi64, 1xi64)
        t1329 = paddle._C_ops.slice(t1327, [4], t1004, t1011, [1], [])

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x10xf32, 1xi64, 1xi64)
        t1330 = paddle._C_ops.slice(t1327, [4], t1011, t1018, [1], [])

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x10xf32, 1xi64, 1xi64)
        t1331 = paddle._C_ops.slice(t1327, [4], t1018, t1025, [1], [])

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x10xf32, 1xi64, 1xi64)
        t1332 = paddle._C_ops.slice(t1327, [4], t1025, t1032, [1], [])

        # pd_op.slice: (8x3x44x44x1xf32) <- (8x3x44x44x10xf32, 1xi64, 1xi64)
        t1333 = paddle._C_ops.slice(t1327, [4], t1032, t1049, [1], [])

        # pd_op.slice: (8x3x44x44x4xf32) <- (8x3x44x44x10xf32, 1xi64, 1xi64)
        t1334 = paddle._C_ops.slice(t1327, [4], t1049, t1039, [1], [])
        del t1049, t1327

        # pd_op.multiply: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1335 = paddle._C_ops.multiply(t1332, t1333)
        del t1332

        # pd_op.sigmoid: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32)
        t1336 = paddle._C_ops.sigmoid(t1321)

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1337 = paddle._C_ops.scale(t1336, t1054, float("0"), True)

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1338 = paddle._C_ops.scale(t1337, t1054, float("0"), True)
        del t1337

        # pd_op.sigmoid: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32)
        t1339 = paddle._C_ops.sigmoid(t1322)

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1340 = paddle._C_ops.scale(t1339, t1054, float("0"), True)

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1341 = paddle._C_ops.scale(t1340, t1054, float("0"), True)
        del t1340

        # pd_op.bce_loss: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1342 = paddle._C_ops.bce_loss(t1338, t1328)

        # pd_op.bce_loss: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1343 = paddle._C_ops.bce_loss(t1341, t1329)

        # pd_op.add: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1344 = paddle._C_ops.add(t1342, t1343)

        # pd_op.multiply: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1345 = paddle._C_ops.multiply(t1335, t1344)

        # pd_op.sum: (8xf32) <- (8x3x44x44x1xf32, 4xi64)
        t1346 = paddle._C_ops.sum(t1345, t1076, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1347 = paddle._C_ops.mean(t1346, [], False)

        # pd_op.subtract: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1348 = paddle._C_ops.subtract(t1323, t1330)

        # pd_op.abs: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32)
        t1349 = paddle._C_ops.abs(t1348)

        # pd_op.subtract: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1350 = paddle._C_ops.subtract(t1324, t1331)

        # pd_op.abs: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32)
        t1351 = paddle._C_ops.abs(t1350)

        # pd_op.add: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1352 = paddle._C_ops.add(t1349, t1351)

        # pd_op.multiply: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1353 = paddle._C_ops.multiply(t1335, t1352)

        # pd_op.sum: (8xf32) <- (8x3x44x44x1xf32, 4xi64)
        t1354 = paddle._C_ops.sum(t1353, t1076, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1355 = paddle._C_ops.mean(t1354, [], False)

        # pd_op.full: (1xf64) <- ()
        t1356 = paddle._C_ops.full(
            [1], float("44"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (44xi64) <- (1xf64, 1xf64, 1xf64)
        t1357 = paddle.arange(t1095, t1356, t1097, dtype="int64")
        del t1356, t1095, t1097

        # builtin.combine: ([44xi64, 44xi64]) <- (44xi64, 44xi64)
        t1358 = [t1357, t1357]
        del t1357

        # pd_op.meshgrid: ([44x44xi64, 44x44xi64]) <- ([44xi64, 44xi64])
        t1359 = paddle._C_ops.meshgrid(t1358)
        del t1358

        # builtin.split: (44x44xi64, 44x44xi64) <- ([44x44xi64, 44x44xi64])
        (
            t1360,
            t1361,
        ) = t1359
        del t1359

        # builtin.combine: ([44x44xi64, 44x44xi64]) <- (44x44xi64, 44x44xi64)
        t1362 = [t1361, t1360]
        del t1360, t1361

        # pd_op.stack: (44x44x2xi64) <- ([44x44xi64, 44x44xi64])
        t1363 = paddle._C_ops.stack(t1362, 2)
        del t1362

        # pd_op.cast: (44x44x2xf32) <- (44x44x2xi64)
        t1364 = paddle._C_ops.cast(t1363, paddle.float32)
        del t1363

        # pd_op.full_int_array: (5xi64) <- ()
        t1365 = [1, 1, 44, 44, 2]

        # pd_op.reshape: (1x1x44x44x2xf32) <- (44x44x2xf32, 5xi64)
        t1366 = paddle._C_ops.reshape(t1364, t1365)
        del t1364, t1365

        # pd_op.slice: (1x1x44x44x1xf32) <- (1x1x44x44x2xf32, 1xi64, 1xi64)
        t1367 = paddle._C_ops.slice(t1366, [4], t1001, t1004, [1], [])

        # pd_op.add: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1x1x44x44x1xf32)
        t1368 = paddle._C_ops.add(t1338, t1367)
        del t1367

        # pd_op.full: (1xf32) <- ()
        t1369 = paddle._C_ops.full(
            [1], float("0.0227273"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1370 = paddle._C_ops.scale(t1368, t1369, float("0"), True)
        del t1368

        # pd_op.slice: (1x1x44x44x1xf32) <- (1x1x44x44x2xf32, 1xi64, 1xi64)
        t1371 = paddle._C_ops.slice(t1366, [4], t1004, t1039, [1], [])
        del t1366

        # pd_op.add: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1x1x44x44x1xf32)
        t1372 = paddle._C_ops.add(t1341, t1371)
        del t1371

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1373 = paddle._C_ops.scale(t1372, t1369, float("0"), True)
        del t1372, t1369

        # pd_op.full: (3x2xi64) <- ()
        t1374 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        t1375 = paddle._C_ops.assign_value_(
            t1374,
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
        del t1374

        # pd_op.cast: (3x2xf32) <- (3x2xi64)
        t1376 = paddle._C_ops.cast(t1375, paddle.float32)
        del t1375

        # pd_op.reshape: (1x3x1x1x2xf32) <- (3x2xf32, 5xi64)
        t1377 = paddle._C_ops.reshape(t1376, t1118)
        del t1376, t1118

        # pd_op.exp: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32)
        t1378 = paddle._C_ops.exp(t1323)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1379 = paddle._C_ops.slice(t1377, [4], t1001, t1004, [1], [])

        # pd_op.multiply: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1x3x1x1x1xf32)
        t1380 = paddle._C_ops.multiply(t1378, t1379)
        del t1378, t1379

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1381 = paddle._C_ops.scale(t1380, t1123, float("0"), True)
        del t1380

        # pd_op.exp: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32)
        t1382 = paddle._C_ops.exp(t1324)

        # pd_op.slice: (1x3x1x1x1xf32) <- (1x3x1x1x2xf32, 1xi64, 1xi64)
        t1383 = paddle._C_ops.slice(t1377, [4], t1004, t1039, [1], [])
        del t1377

        # pd_op.multiply: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1x3x1x1x1xf32)
        t1384 = paddle._C_ops.multiply(t1382, t1383)
        del t1382, t1383

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1385 = paddle._C_ops.scale(t1384, t1123, float("0"), True)
        del t1123, t1384

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1386 = paddle._C_ops.scale(t1381, t1129, float("0"), True)

        # pd_op.subtract: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1387 = paddle._C_ops.subtract(t1370, t1386)
        del t1386

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1388 = paddle._C_ops.scale(t1385, t1129, float("0"), True)

        # pd_op.subtract: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1389 = paddle._C_ops.subtract(t1373, t1388)
        del t1388

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1390 = paddle._C_ops.scale(t1381, t1129, float("0"), True)
        del t1381

        # pd_op.add: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1391 = paddle._C_ops.add(t1370, t1390)
        del t1370, t1390

        # pd_op.scale: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 1xf32)
        t1392 = paddle._C_ops.scale(t1385, t1129, float("0"), True)
        del t1129, t1385

        # pd_op.add: (8x3x44x44x1xf32) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1393 = paddle._C_ops.add(t1373, t1392)
        del t1373, t1392

        # builtin.combine: ([8x3x44x44x1xf32, 8x3x44x44x1xf32, 8x3x44x44x1xf32, 8x3x44x44x1xf32]) <- (8x3x44x44x1xf32, 8x3x44x44x1xf32, 8x3x44x44x1xf32, 8x3x44x44x1xf32)
        t1394 = [t1387, t1389, t1391, t1393]
        del t1391, t1393, t1387, t1389

        # pd_op.concat: (8x3x44x44x4xf32) <- ([8x3x44x44x1xf32, 8x3x44x44x1xf32, 8x3x44x44x1xf32, 8x3x44x44x1xf32], 1xi32)
        t1395 = paddle._C_ops.concat(t1394, t1138)
        del t1394, t1138

        # pd_op.reshape: (8x5808x4xf32) <- (8x3x44x44x4xf32, 3xi64)
        t1396 = paddle._C_ops.reshape(t1395, t1141)
        del t1395, t1141

        # pd_op.unsqueeze: (8x5808x1x4xf32) <- (8x5808x4xf32, 1xi64)
        t1397 = paddle._C_ops.unsqueeze(t1396, t1011)
        del t1396

        # pd_op.slice: (8x5808x1x2xf32) <- (8x5808x1x4xf32, 1xi64, 1xi64)
        t1398 = paddle._C_ops.slice(t1397, [3], t1001, t1011, [1], [])

        # pd_op.slice: (8x5808x1x2xf32) <- (8x5808x1x4xf32, 1xi64, 1xi64)
        t1399 = paddle._C_ops.slice(t1397, [3], t1011, t1039, [1], [])
        del t1397

        # pd_op.maximum: (8x5808x50x2xf32) <- (8x5808x1x2xf32, 8x1x50x2xf32)
        t1400 = paddle._C_ops.maximum(t1398, t1154)
        del t1154

        # pd_op.minimum: (8x5808x50x2xf32) <- (8x5808x1x2xf32, 8x1x50x2xf32)
        t1401 = paddle._C_ops.minimum(t1399, t1155)
        del t1155

        # pd_op.subtract: (8x5808x50x2xf32) <- (8x5808x50x2xf32, 8x5808x50x2xf32)
        t1402 = paddle._C_ops.subtract(t1401, t1400)
        del t1400, t1401

        # pd_op.clip: (8x5808x50x2xf32) <- (8x5808x50x2xf32, 1xf32, 1xf32)
        t1403 = paddle._C_ops.clip(t1402, t1159, t1160)
        del t1402

        # pd_op.prod: (8x5808x50xf32) <- (8x5808x50x2xf32, 1xi64)
        t1404 = paddle._C_ops.prod(t1403, t1162, False, False)
        del t1403

        # pd_op.subtract: (8x5808x1x2xf32) <- (8x5808x1x2xf32, 8x5808x1x2xf32)
        t1405 = paddle._C_ops.subtract(t1399, t1398)
        del t1398, t1399

        # pd_op.clip: (8x5808x1x2xf32) <- (8x5808x1x2xf32, 1xf32, 1xf32)
        t1406 = paddle._C_ops.clip(t1405, t1159, t1160)
        del t1159, t1160, t1405

        # pd_op.prod: (8x5808x1xf32) <- (8x5808x1x2xf32, 1xi64)
        t1407 = paddle._C_ops.prod(t1406, t1162, False, False)
        del t1406

        # pd_op.add: (8x5808x50xf32) <- (8x5808x1xf32, 8x1x50xf32)
        t1408 = paddle._C_ops.add(t1407, t1172)
        del t1172, t1407

        # pd_op.subtract: (8x5808x50xf32) <- (8x5808x50xf32, 8x5808x50xf32)
        t1409 = paddle._C_ops.subtract(t1408, t1404)
        del t1408

        # pd_op.scale: (8x5808x50xf32) <- (8x5808x50xf32, 1xf32)
        t1410 = paddle._C_ops.scale(t1409, t1054, float("1e-09"), True)
        del t1409

        # pd_op.divide: (8x5808x50xf32) <- (8x5808x50xf32, 8x5808x50xf32)
        t1411 = paddle._C_ops.divide(t1404, t1410)
        del t1404, t1410

        # pd_op.max: (8x5808xf32) <- (8x5808x50xf32, 1xi64)
        t1412 = paddle._C_ops.max(t1411, t1011, False)
        del t1411

        # pd_op.less_equal: (8x5808xb) <- (8x5808xf32, xf32)
        t1413 = paddle._C_ops.less_equal(t1412, t1178)
        del t1178, t1412

        # pd_op.cast: (8x5808xf32) <- (8x5808xb)
        t1414 = paddle._C_ops.cast(t1413, paddle.float32)
        del t1413

        # pd_op.reshape: (8x5808xf32) <- (8x3x44x44x1xf32, 2xi64)
        t1415 = paddle._C_ops.reshape(t1325, t1181)

        # pd_op.reshape: (8x5808xf32) <- (8x3x44x44x1xf32, 2xi64)
        t1416 = paddle._C_ops.reshape(t1333, t1181)
        del t1181

        # pd_op.greater_than: (8x5808xb) <- (8x5808xf32, xf32)
        t1417 = paddle._C_ops.greater_than(t1416, t1184)
        del t1184

        # pd_op.cast: (8x5808xf32) <- (8x5808xb)
        t1418 = paddle._C_ops.cast(t1417, paddle.float32)
        del t1417

        # pd_op.sigmoid_cross_entropy_with_logits: (8x5808xf32) <- (8x5808xf32, 8x5808xf32, None)
        t1419 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1415, t1418, None, False, -100
        )

        # pd_op.multiply: (8x5808xf32) <- (8x5808xf32, 8x5808xf32)
        t1420 = paddle._C_ops.multiply(t1419, t1416)

        # pd_op.scale: (8x5808xf32) <- (8x5808xf32, 1xf32)
        t1421 = paddle._C_ops.scale(t1418, t1190, float("1"), True)
        del t1190

        # pd_op.multiply: (8x5808xf32) <- (8x5808xf32, 8x5808xf32)
        t1422 = paddle._C_ops.multiply(t1419, t1421)

        # pd_op.multiply: (8x5808xf32) <- (8x5808xf32, 8x5808xf32)
        t1423 = paddle._C_ops.multiply(t1422, t1414)

        # pd_op.add: (8x5808xf32) <- (8x5808xf32, 8x5808xf32)
        t1424 = paddle._C_ops.add(t1420, t1423)

        # pd_op.sum: (8xf32) <- (8x5808xf32, 1xi64)
        t1425 = paddle._C_ops.sum(t1424, t1162, None, False)
        del t1162

        # pd_op.mean: (xf32) <- (8xf32)
        t1426 = paddle._C_ops.mean(t1425, [], False)

        # pd_op.sigmoid_cross_entropy_with_logits: (8x3x44x44x4xf32) <- (8x3x44x44x4xf32, 8x3x44x44x4xf32, None)
        t1427 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            t1326, t1334, None, False, -100
        )

        # pd_op.multiply: (8x3x44x44x4xf32) <- (8x3x44x44x4xf32, 8x3x44x44x1xf32)
        t1428 = paddle._C_ops.multiply(t1427, t1333)

        # pd_op.sum: (8xf32) <- (8x3x44x44x4xf32, 4xi64)
        t1429 = paddle._C_ops.sum(t1428, t1076, None, False)

        # pd_op.mean: (xf32) <- (8xf32)
        t1430 = paddle._C_ops.mean(t1429, [], False)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1431 = paddle._C_ops.add(t1314, t1347)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1432 = paddle._C_ops.add(t1315, t1355)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1433 = paddle._C_ops.add(t1316, t1426)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1434 = paddle._C_ops.add(t1317, t1430)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t1435 = paddle._C_ops.scale(t1431, t1054, float("0"), True)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1436 = paddle._C_ops.add(t1435, t1432)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1437 = paddle._C_ops.add(t1436, t1433)

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
            t379,
            t380,
            t381,
            t382,
            t383,
            t384,
            t385,
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
            t403,
            t404,
            t405,
            t406,
            t407,
            t408,
            t409,
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
            t426,
            t427,
            t428,
            t429,
            t430,
            t431,
            t432,
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
            t450,
            t451,
            t452,
            t453,
            t454,
            t455,
            t456,
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
            t475,
            t477,
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
            t492,
            t493,
            t494,
            t495,
            t496,
            t497,
            t498,
            t500,
            t501,
            t502,
            t503,
            t504,
            t505,
            t506,
            t507,
            t508,
            t510,
            t511,
            t512,
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
            t526,
            t527,
            t528,
            t529,
            t530,
            t531,
            t532,
            t534,
            t535,
            t536,
            t537,
            t538,
            t539,
            t540,
            t541,
            t542,
            t544,
            t545,
            t546,
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
            t706,
            t707,
            t708,
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
            t740,
            t741,
            t742,
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
            t773,
            t774,
            t775,
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
            t807,
            t808,
            t809,
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
            t1029,
            t1030,
            t1031,
            t1032,
            t1033,
            t1034,
            t1035,
            t1036,
            t1037,
            t1038,
            t1039,
            t1040,
            t1041,
            t1042,
            t1044,
            t1045,
            t1046,
            t1047,
            t1050,
            t1051,
            t1052,
            t1053,
            t1054,
            t1055,
            t1056,
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
            t1068,
            t1069,
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
            t1081,
            t1082,
            t1083,
            t1084,
            t1085,
            t1086,
            t1087,
            t1088,
            t1089,
            t1090,
            t1091,
            t1092,
            t1093,
            t1094,
            t1163,
            t1164,
            t1165,
            t1180,
            t1182,
            t1183,
            t1186,
            t1188,
            t1189,
            t1191,
            t1192,
            t1193,
            t1194,
            t1195,
            t1196,
            t1197,
            t1198,
            t1199,
            t1200,
            t1203,
            t1204,
            t1205,
            t1206,
            t1207,
            t1208,
            t1209,
            t1211,
            t1212,
            t1213,
            t1214,
            t1216,
            t1217,
            t1218,
            t1219,
            t1221,
            t1222,
            t1224,
            t1225,
            t1226,
            t1227,
            t1228,
            t1229,
            t1230,
            t1231,
            t1232,
            t1233,
            t1234,
            t1235,
            t1236,
            t1237,
            t1238,
            t1297,
            t1298,
            t1299,
            t1301,
            t1302,
            t1303,
            t1304,
            t1305,
            t1306,
            t1307,
            t1308,
            t1309,
            t1310,
            t1311,
            t1312,
            t1313,
            t1314,
            t1315,
            t1316,
            t1317,
            t1320,
            t1321,
            t1322,
            t1323,
            t1324,
            t1325,
            t1326,
            t1328,
            t1329,
            t1330,
            t1331,
            t1333,
            t1334,
            t1335,
            t1336,
            t1338,
            t1339,
            t1341,
            t1342,
            t1343,
            t1344,
            t1345,
            t1346,
            t1347,
            t1348,
            t1349,
            t1350,
            t1351,
            t1352,
            t1353,
            t1354,
            t1355,
            t1414,
            t1415,
            t1416,
            t1418,
            t1419,
            t1420,
            t1421,
            t1422,
            t1423,
            t1424,
            t1425,
            t1426,
            t1427,
            t1428,
            t1429,
            t1430,
            t1431,
            t1432,
            t1433,
            t1434,
            t1435,
            t1436,
            t1437,
        )
