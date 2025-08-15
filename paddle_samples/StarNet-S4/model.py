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
        t368: paddle.Tensor,
        t369: paddle.Tensor,
        t370: paddle.Tensor,
        t371: paddle.Tensor,
        t372: paddle.Tensor,
        t373: paddle.Tensor,
        t374: paddle.Tensor,
        t375: paddle.Tensor,
        t376: paddle.Tensor,
        t377: paddle.Tensor,
        t378: paddle.Tensor,
        t379: paddle.Tensor,
        t380: paddle.Tensor,
        t381: paddle.Tensor,
        t382: paddle.Tensor,
        t383: paddle.Tensor,
        t384: paddle.Tensor,
        t385: paddle.Tensor,
        t386: paddle.Tensor,
        t387: paddle.Tensor,
        t388: paddle.Tensor,
        t389: paddle.Tensor,
        t390: paddle.Tensor,
        t391: paddle.Tensor,
        t392: paddle.Tensor,
        t393: paddle.Tensor,
        t394: paddle.Tensor,
        t395: paddle.Tensor,
        t396: paddle.Tensor,
        t397: paddle.Tensor,
        t398: paddle.Tensor,
        t399: paddle.Tensor,
        t400: paddle.Tensor,
        t401: paddle.Tensor,
        t402: paddle.Tensor,
        t403: paddle.Tensor,
        t404: paddle.Tensor,
        t405: paddle.Tensor,
        t406: paddle.Tensor,
        t407: paddle.Tensor,
        t408: paddle.Tensor,
        t409: paddle.Tensor,
        t410: paddle.Tensor,
        t411: paddle.Tensor,
        t412: paddle.Tensor,
        t413: paddle.Tensor,
        t414: paddle.Tensor,
        t415: paddle.Tensor,
        t416: paddle.Tensor,
        t417: paddle.Tensor,
        t418: paddle.Tensor,
        t419: paddle.Tensor,
        t420: paddle.Tensor,
        t421: paddle.Tensor,
        t422: paddle.Tensor,
        t423: paddle.Tensor,
        t424: paddle.Tensor,
        t425: paddle.Tensor,
        t426: paddle.Tensor,
        t427: paddle.Tensor,
        t428: paddle.Tensor,
        t429: paddle.Tensor,
        t430: paddle.Tensor,
        t431: paddle.Tensor,
        t432: paddle.Tensor,
        t433: paddle.Tensor,
        t434: paddle.Tensor,
        t435: paddle.Tensor,
        t436: paddle.Tensor,
        t437: paddle.Tensor,
        t438: paddle.Tensor,
        t439: paddle.Tensor,
        t440: paddle.Tensor,
        t441: paddle.Tensor,
        t442: paddle.Tensor,
        t443: paddle.Tensor,
        t444: paddle.Tensor,
        t445: paddle.Tensor,
        t446: paddle.Tensor,
        t447: paddle.Tensor,
        t448: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        t449 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t450 = [1, -1, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t451 = paddle._C_ops.reshape(t1, t450)
        del t1

        # pd_op.add: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 1x32x1x1xf32)
        t452 = paddle._C_ops.add(t449, t451)
        del t449, t451

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t453, t454, t455, t456, t457, t458 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t452,
                t2,
                t3,
                t4,
                t5,
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
        del t452, t5, t4, t3, t2

        # pd_op.relu6: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t459 = paddle._C_ops.relu6(t453)
        del t453

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x112x112xf32, 32x32x3x3xf32)
        t460 = paddle._C_ops.conv2d(
            t459, t6, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t6, t459

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t461 = paddle._C_ops.reshape(t7, t450)
        del t7

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t462 = paddle._C_ops.add(t460, t461)
        del t460, t461

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t463, t464, t465, t466, t467, t468 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t462,
                t8,
                t9,
                t10,
                t11,
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
        del t462, t11, t10, t9, t8

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        t469 = paddle._C_ops.depthwise_conv2d(
            t463, t12, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t12

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t470 = paddle._C_ops.reshape(t13, t450)
        del t13

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t471 = paddle._C_ops.add(t469, t470)
        del t469, t470

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t472, t473, t474, t475, t476, t477 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t471,
                t14,
                t15,
                t16,
                t17,
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
        del t471, t17, t16, t15, t14

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        t478 = paddle._C_ops.conv2d(
            t472, t18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t18

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t479 = paddle._C_ops.reshape(t19, t450)
        del t19

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        t480 = paddle._C_ops.add(t478, t479)
        del t478, t479

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        t481 = paddle._C_ops.conv2d(
            t472, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t472, t20

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t482 = paddle._C_ops.reshape(t21, t450)
        del t21

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        t483 = paddle._C_ops.add(t481, t482)
        del t481, t482

        # pd_op.relu6: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        t484 = paddle._C_ops.relu6(t480)
        del t480

        # pd_op.multiply: (-1x128x56x56xf32) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        t485 = paddle._C_ops.multiply(t484, t483)
        del t483, t484

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x128x56x56xf32, 32x128x1x1xf32)
        t486 = paddle._C_ops.conv2d(
            t485, t22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t485, t22

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t487 = paddle._C_ops.reshape(t23, t450)
        del t23

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t488 = paddle._C_ops.add(t486, t487)
        del t486, t487

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t489, t490, t491, t492, t493, t494 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t488,
                t24,
                t25,
                t26,
                t27,
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
        del t488, t27, t26, t25, t24

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        t495 = paddle._C_ops.depthwise_conv2d(
            t489, t28, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t489, t28

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t496 = paddle._C_ops.reshape(t29, t450)
        del t29

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t497 = paddle._C_ops.add(t495, t496)
        del t495, t496

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        t498 = paddle._C_ops.add(t463, t497)
        del t497, t463

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        t499 = paddle._C_ops.depthwise_conv2d(
            t498, t30, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t30

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t500 = paddle._C_ops.reshape(t31, t450)
        del t31

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t501 = paddle._C_ops.add(t499, t500)
        del t499, t500

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t502, t503, t504, t505, t506, t507 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t501,
                t32,
                t33,
                t34,
                t35,
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
        del t501, t35, t34, t33, t32

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        t508 = paddle._C_ops.conv2d(
            t502, t36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t36

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t509 = paddle._C_ops.reshape(t37, t450)
        del t37

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        t510 = paddle._C_ops.add(t508, t509)
        del t508, t509

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        t511 = paddle._C_ops.conv2d(
            t502, t38, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t502, t38

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t512 = paddle._C_ops.reshape(t39, t450)
        del t39

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        t513 = paddle._C_ops.add(t511, t512)
        del t511, t512

        # pd_op.relu6: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        t514 = paddle._C_ops.relu6(t510)
        del t510

        # pd_op.multiply: (-1x128x56x56xf32) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        t515 = paddle._C_ops.multiply(t514, t513)
        del t513, t514

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x128x56x56xf32, 32x128x1x1xf32)
        t516 = paddle._C_ops.conv2d(
            t515, t40, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t515, t40

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t517 = paddle._C_ops.reshape(t41, t450)
        del t41

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t518 = paddle._C_ops.add(t516, t517)
        del t516, t517

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t519, t520, t521, t522, t523, t524 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t518,
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
        del t518, t45, t44, t43, t42

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        t525 = paddle._C_ops.depthwise_conv2d(
            t519, t46, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t519, t46

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t526 = paddle._C_ops.reshape(t47, t450)
        del t47

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t527 = paddle._C_ops.add(t525, t526)
        del t525, t526

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        t528 = paddle._C_ops.add(t498, t527)
        del t527, t498

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        t529 = paddle._C_ops.depthwise_conv2d(
            t528, t48, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t48

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t530 = paddle._C_ops.reshape(t49, t450)
        del t49

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t531 = paddle._C_ops.add(t529, t530)
        del t529, t530

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t532, t533, t534, t535, t536, t537 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t531,
                t50,
                t51,
                t52,
                t53,
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
        del t531, t53, t52, t51, t50

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        t538 = paddle._C_ops.conv2d(
            t532, t54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t54

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t539 = paddle._C_ops.reshape(t55, t450)
        del t55

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        t540 = paddle._C_ops.add(t538, t539)
        del t538, t539

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        t541 = paddle._C_ops.conv2d(
            t532, t56, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t532, t56

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t542 = paddle._C_ops.reshape(t57, t450)
        del t57

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        t543 = paddle._C_ops.add(t541, t542)
        del t541, t542

        # pd_op.relu6: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        t544 = paddle._C_ops.relu6(t540)
        del t540

        # pd_op.multiply: (-1x128x56x56xf32) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        t545 = paddle._C_ops.multiply(t544, t543)
        del t543, t544

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x128x56x56xf32, 32x128x1x1xf32)
        t546 = paddle._C_ops.conv2d(
            t545, t58, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t545, t58

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t547 = paddle._C_ops.reshape(t59, t450)
        del t59

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t548 = paddle._C_ops.add(t546, t547)
        del t546, t547

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t549, t550, t551, t552, t553, t554 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t548,
                t60,
                t61,
                t62,
                t63,
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
        del t548, t63, t62, t61, t60

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        t555 = paddle._C_ops.depthwise_conv2d(
            t549, t64, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t549, t64

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t556 = paddle._C_ops.reshape(t65, t450)
        del t65

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t557 = paddle._C_ops.add(t555, t556)
        del t555, t556

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        t558 = paddle._C_ops.add(t528, t557)
        del t528, t557

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x32x56x56xf32, 64x32x3x3xf32)
        t559 = paddle._C_ops.conv2d(
            t558, t66, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t558, t66

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t560 = paddle._C_ops.reshape(t67, t450)
        del t67

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t561 = paddle._C_ops.add(t559, t560)
        del t559, t560

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t562, t563, t564, t565, t566, t567 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t561,
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
        del t561, t71, t70, t69, t68

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t568 = paddle._C_ops.depthwise_conv2d(
            t562, t72, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t72

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t569 = paddle._C_ops.reshape(t73, t450)
        del t73

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t570 = paddle._C_ops.add(t568, t569)
        del t568, t569

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t571, t572, t573, t574, t575, t576 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t570,
                t74,
                t75,
                t76,
                t77,
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
        del t570, t77, t76, t75, t74

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t577 = paddle._C_ops.conv2d(
            t571, t78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t78

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t578 = paddle._C_ops.reshape(t79, t450)
        del t79

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t579 = paddle._C_ops.add(t577, t578)
        del t577, t578

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t580 = paddle._C_ops.conv2d(
            t571, t80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t571, t80

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t581 = paddle._C_ops.reshape(t81, t450)
        del t81

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t582 = paddle._C_ops.add(t580, t581)
        del t580, t581

        # pd_op.relu6: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t583 = paddle._C_ops.relu6(t579)
        del t579

        # pd_op.multiply: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        t584 = paddle._C_ops.multiply(t583, t582)
        del t582, t583

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x256x28x28xf32, 64x256x1x1xf32)
        t585 = paddle._C_ops.conv2d(
            t584, t82, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t584, t82

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t586 = paddle._C_ops.reshape(t83, t450)
        del t83

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t587 = paddle._C_ops.add(t585, t586)
        del t585, t586

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t588, t589, t590, t591, t592, t593 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t587,
                t84,
                t85,
                t86,
                t87,
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
        del t587, t87, t86, t85, t84

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t594 = paddle._C_ops.depthwise_conv2d(
            t588, t88, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t588, t88

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t595 = paddle._C_ops.reshape(t89, t450)
        del t89

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t596 = paddle._C_ops.add(t594, t595)
        del t594, t595

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        t597 = paddle._C_ops.add(t562, t596)
        del t596, t562

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t598 = paddle._C_ops.depthwise_conv2d(
            t597, t90, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t90

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t599 = paddle._C_ops.reshape(t91, t450)
        del t91

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t600 = paddle._C_ops.add(t598, t599)
        del t598, t599

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t601, t602, t603, t604, t605, t606 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t600,
                t92,
                t93,
                t94,
                t95,
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
        del t600, t95, t94, t93, t92

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t607 = paddle._C_ops.conv2d(
            t601, t96, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t96

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t608 = paddle._C_ops.reshape(t97, t450)
        del t97

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t609 = paddle._C_ops.add(t607, t608)
        del t607, t608

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t610 = paddle._C_ops.conv2d(
            t601, t98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t601, t98

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t611 = paddle._C_ops.reshape(t99, t450)
        del t99

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t612 = paddle._C_ops.add(t610, t611)
        del t610, t611

        # pd_op.relu6: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t613 = paddle._C_ops.relu6(t609)
        del t609

        # pd_op.multiply: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        t614 = paddle._C_ops.multiply(t613, t612)
        del t612, t613

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x256x28x28xf32, 64x256x1x1xf32)
        t615 = paddle._C_ops.conv2d(
            t614, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t614, t100

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t616 = paddle._C_ops.reshape(t101, t450)
        del t101

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t617 = paddle._C_ops.add(t615, t616)
        del t615, t616

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t618, t619, t620, t621, t622, t623 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t617,
                t102,
                t103,
                t104,
                t105,
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
        del t617, t105, t104, t103, t102

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t624 = paddle._C_ops.depthwise_conv2d(
            t618, t106, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t618, t106

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t625 = paddle._C_ops.reshape(t107, t450)
        del t107

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t626 = paddle._C_ops.add(t624, t625)
        del t624, t625

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        t627 = paddle._C_ops.add(t597, t626)
        del t597, t626

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t628 = paddle._C_ops.depthwise_conv2d(
            t627, t108, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t108

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t629 = paddle._C_ops.reshape(t109, t450)
        del t109

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t630 = paddle._C_ops.add(t628, t629)
        del t628, t629

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t631, t632, t633, t634, t635, t636 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t630,
                t110,
                t111,
                t112,
                t113,
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
        del t630, t113, t112, t111, t110

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t637 = paddle._C_ops.conv2d(
            t631, t114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t114

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t638 = paddle._C_ops.reshape(t115, t450)
        del t115

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t639 = paddle._C_ops.add(t637, t638)
        del t637, t638

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t640 = paddle._C_ops.conv2d(
            t631, t116, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t631, t116

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t641 = paddle._C_ops.reshape(t117, t450)
        del t117

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t642 = paddle._C_ops.add(t640, t641)
        del t640, t641

        # pd_op.relu6: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t643 = paddle._C_ops.relu6(t639)
        del t639

        # pd_op.multiply: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        t644 = paddle._C_ops.multiply(t643, t642)
        del t642, t643

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x256x28x28xf32, 64x256x1x1xf32)
        t645 = paddle._C_ops.conv2d(
            t644, t118, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t644, t118

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t646 = paddle._C_ops.reshape(t119, t450)
        del t119

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t647 = paddle._C_ops.add(t645, t646)
        del t645, t646

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t648, t649, t650, t651, t652, t653 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t647,
                t120,
                t121,
                t122,
                t123,
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
        del t647, t123, t122, t121, t120

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t654 = paddle._C_ops.depthwise_conv2d(
            t648, t124, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t648, t124

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t655 = paddle._C_ops.reshape(t125, t450)
        del t125

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t656 = paddle._C_ops.add(t654, t655)
        del t654, t655

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        t657 = paddle._C_ops.add(t627, t656)
        del t627, t656

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x64x28x28xf32, 128x64x3x3xf32)
        t658 = paddle._C_ops.conv2d(
            t657, t126, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t657, t126

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t659 = paddle._C_ops.reshape(t127, t450)
        del t127

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t660 = paddle._C_ops.add(t658, t659)
        del t658, t659

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t661, t662, t663, t664, t665, t666 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t660,
                t128,
                t129,
                t130,
                t131,
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
        del t660, t131, t130, t129, t128

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t667 = paddle._C_ops.depthwise_conv2d(
            t661, t132, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t132

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t668 = paddle._C_ops.reshape(t133, t450)
        del t133

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t669 = paddle._C_ops.add(t667, t668)
        del t667, t668

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t670, t671, t672, t673, t674, t675 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t669,
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
        del t669, t137, t136, t135, t134

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t676 = paddle._C_ops.conv2d(
            t670, t138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t138

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t677 = paddle._C_ops.reshape(t139, t450)
        del t139

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t678 = paddle._C_ops.add(t676, t677)
        del t676, t677

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t679 = paddle._C_ops.conv2d(
            t670, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t670, t140

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t680 = paddle._C_ops.reshape(t141, t450)
        del t141

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t681 = paddle._C_ops.add(t679, t680)
        del t679, t680

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t682 = paddle._C_ops.relu6(t678)
        del t678

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t683 = paddle._C_ops.multiply(t682, t681)
        del t681, t682

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t684 = paddle._C_ops.conv2d(
            t683, t142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t683, t142

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t685 = paddle._C_ops.reshape(t143, t450)
        del t143

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t686 = paddle._C_ops.add(t684, t685)
        del t684, t685

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t687, t688, t689, t690, t691, t692 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t686,
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
        del t686, t147, t146, t145, t144

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t693 = paddle._C_ops.depthwise_conv2d(
            t687, t148, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t687, t148

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t694 = paddle._C_ops.reshape(t149, t450)
        del t149

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t695 = paddle._C_ops.add(t693, t694)
        del t693, t694

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t696 = paddle._C_ops.add(t661, t695)
        del t695, t661

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t697 = paddle._C_ops.depthwise_conv2d(
            t696, t150, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t150

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t698 = paddle._C_ops.reshape(t151, t450)
        del t151

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t699 = paddle._C_ops.add(t697, t698)
        del t697, t698

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t700, t701, t702, t703, t704, t705 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t699,
                t152,
                t153,
                t154,
                t155,
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
        del t699, t155, t154, t153, t152

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t706 = paddle._C_ops.conv2d(
            t700, t156, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t707 = paddle._C_ops.reshape(t157, t450)
        del t157

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t708 = paddle._C_ops.add(t706, t707)
        del t706, t707

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t709 = paddle._C_ops.conv2d(
            t700, t158, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t700, t158

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t710 = paddle._C_ops.reshape(t159, t450)
        del t159

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t711 = paddle._C_ops.add(t709, t710)
        del t709, t710

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t712 = paddle._C_ops.relu6(t708)
        del t708

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t713 = paddle._C_ops.multiply(t712, t711)
        del t711, t712

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t714 = paddle._C_ops.conv2d(
            t713, t160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t713, t160

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t715 = paddle._C_ops.reshape(t161, t450)
        del t161

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t716 = paddle._C_ops.add(t714, t715)
        del t714, t715

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t717, t718, t719, t720, t721, t722 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t716,
                t162,
                t163,
                t164,
                t165,
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
        del t716, t165, t164, t163, t162

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t723 = paddle._C_ops.depthwise_conv2d(
            t717, t166, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t717, t166

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t724 = paddle._C_ops.reshape(t167, t450)
        del t167

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t725 = paddle._C_ops.add(t723, t724)
        del t723, t724

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t726 = paddle._C_ops.add(t696, t725)
        del t696, t725

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t727 = paddle._C_ops.depthwise_conv2d(
            t726, t168, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t168

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t728 = paddle._C_ops.reshape(t169, t450)
        del t169

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t729 = paddle._C_ops.add(t727, t728)
        del t727, t728

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t730, t731, t732, t733, t734, t735 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t729,
                t170,
                t171,
                t172,
                t173,
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
        del t729, t173, t172, t171, t170

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t736 = paddle._C_ops.conv2d(
            t730, t174, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t174

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t737 = paddle._C_ops.reshape(t175, t450)
        del t175

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t738 = paddle._C_ops.add(t736, t737)
        del t736, t737

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t739 = paddle._C_ops.conv2d(
            t730, t176, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t730, t176

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t740 = paddle._C_ops.reshape(t177, t450)
        del t177

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t741 = paddle._C_ops.add(t739, t740)
        del t739, t740

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t742 = paddle._C_ops.relu6(t738)
        del t738

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t743 = paddle._C_ops.multiply(t742, t741)
        del t741, t742

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t744 = paddle._C_ops.conv2d(
            t743, t178, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t743, t178

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t745 = paddle._C_ops.reshape(t179, t450)
        del t179

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t746 = paddle._C_ops.add(t744, t745)
        del t744, t745

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t747, t748, t749, t750, t751, t752 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t746,
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
        del t746, t183, t182, t181, t180

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t753 = paddle._C_ops.depthwise_conv2d(
            t747, t184, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t747, t184

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t754 = paddle._C_ops.reshape(t185, t450)
        del t185

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t755 = paddle._C_ops.add(t753, t754)
        del t753, t754

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t756 = paddle._C_ops.add(t726, t755)
        del t726, t755

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t757 = paddle._C_ops.depthwise_conv2d(
            t756, t186, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t186

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t758 = paddle._C_ops.reshape(t187, t450)
        del t187

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t759 = paddle._C_ops.add(t757, t758)
        del t757, t758

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t760, t761, t762, t763, t764, t765 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t759,
                t188,
                t189,
                t190,
                t191,
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
        del t759, t191, t190, t189, t188

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t766 = paddle._C_ops.conv2d(
            t760, t192, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t192

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t767 = paddle._C_ops.reshape(t193, t450)
        del t193

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t768 = paddle._C_ops.add(t766, t767)
        del t766, t767

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t769 = paddle._C_ops.conv2d(
            t760, t194, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t760, t194

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t770 = paddle._C_ops.reshape(t195, t450)
        del t195

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t771 = paddle._C_ops.add(t769, t770)
        del t769, t770

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t772 = paddle._C_ops.relu6(t768)
        del t768

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t773 = paddle._C_ops.multiply(t772, t771)
        del t771, t772

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t774 = paddle._C_ops.conv2d(
            t773, t196, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t773, t196

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t775 = paddle._C_ops.reshape(t197, t450)
        del t197

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t776 = paddle._C_ops.add(t774, t775)
        del t774, t775

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t777, t778, t779, t780, t781, t782 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t776,
                t198,
                t199,
                t200,
                t201,
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
        del t776, t201, t200, t199, t198

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t783 = paddle._C_ops.depthwise_conv2d(
            t777, t202, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t777, t202

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t784 = paddle._C_ops.reshape(t203, t450)
        del t203

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t785 = paddle._C_ops.add(t783, t784)
        del t783, t784

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t786 = paddle._C_ops.add(t756, t785)
        del t756, t785

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t787 = paddle._C_ops.depthwise_conv2d(
            t786, t204, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t204

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t788 = paddle._C_ops.reshape(t205, t450)
        del t205

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t789 = paddle._C_ops.add(t787, t788)
        del t787, t788

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t790, t791, t792, t793, t794, t795 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t789,
                t206,
                t207,
                t208,
                t209,
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
        del t789, t209, t208, t207, t206

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t796 = paddle._C_ops.conv2d(
            t790, t210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t797 = paddle._C_ops.reshape(t211, t450)
        del t211

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t798 = paddle._C_ops.add(t796, t797)
        del t796, t797

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t799 = paddle._C_ops.conv2d(
            t790, t212, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t790, t212

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t800 = paddle._C_ops.reshape(t213, t450)
        del t213

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t801 = paddle._C_ops.add(t799, t800)
        del t799, t800

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t802 = paddle._C_ops.relu6(t798)
        del t798

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t803 = paddle._C_ops.multiply(t802, t801)
        del t801, t802

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t804 = paddle._C_ops.conv2d(
            t803, t214, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t803, t214

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t805 = paddle._C_ops.reshape(t215, t450)
        del t215

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t806 = paddle._C_ops.add(t804, t805)
        del t804, t805

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t807, t808, t809, t810, t811, t812 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t806,
                t216,
                t217,
                t218,
                t219,
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
        del t806, t219, t218, t217, t216

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t813 = paddle._C_ops.depthwise_conv2d(
            t807, t220, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t807, t220

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t814 = paddle._C_ops.reshape(t221, t450)
        del t221

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t815 = paddle._C_ops.add(t813, t814)
        del t813, t814

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t816 = paddle._C_ops.add(t786, t815)
        del t786, t815

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t817 = paddle._C_ops.depthwise_conv2d(
            t816, t222, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t222

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t818 = paddle._C_ops.reshape(t223, t450)
        del t223

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t819 = paddle._C_ops.add(t817, t818)
        del t817, t818

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t820, t821, t822, t823, t824, t825 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t819,
                t224,
                t225,
                t226,
                t227,
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
        del t819, t227, t226, t225, t224

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t826 = paddle._C_ops.conv2d(
            t820, t228, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t228

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t827 = paddle._C_ops.reshape(t229, t450)
        del t229

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t828 = paddle._C_ops.add(t826, t827)
        del t826, t827

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t829 = paddle._C_ops.conv2d(
            t820, t230, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t820, t230

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t830 = paddle._C_ops.reshape(t231, t450)
        del t231

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t831 = paddle._C_ops.add(t829, t830)
        del t829, t830

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t832 = paddle._C_ops.relu6(t828)
        del t828

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t833 = paddle._C_ops.multiply(t832, t831)
        del t831, t832

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t834 = paddle._C_ops.conv2d(
            t833, t232, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t833, t232

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t835 = paddle._C_ops.reshape(t233, t450)
        del t233

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t836 = paddle._C_ops.add(t834, t835)
        del t834, t835

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t837, t838, t839, t840, t841, t842 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t836,
                t234,
                t235,
                t236,
                t237,
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
        del t836, t237, t236, t235, t234

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t843 = paddle._C_ops.depthwise_conv2d(
            t837, t238, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t837, t238

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t844 = paddle._C_ops.reshape(t239, t450)
        del t239

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t845 = paddle._C_ops.add(t843, t844)
        del t843, t844

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t846 = paddle._C_ops.add(t816, t845)
        del t816, t845

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t847 = paddle._C_ops.depthwise_conv2d(
            t846, t240, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t240

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t848 = paddle._C_ops.reshape(t241, t450)
        del t241

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t849 = paddle._C_ops.add(t847, t848)
        del t847, t848

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t850, t851, t852, t853, t854, t855 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t849,
                t242,
                t243,
                t244,
                t245,
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
        del t849, t245, t244, t243, t242

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t856 = paddle._C_ops.conv2d(
            t850, t246, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t246

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t857 = paddle._C_ops.reshape(t247, t450)
        del t247

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t858 = paddle._C_ops.add(t856, t857)
        del t856, t857

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t859 = paddle._C_ops.conv2d(
            t850, t248, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t850, t248

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t860 = paddle._C_ops.reshape(t249, t450)
        del t249

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t861 = paddle._C_ops.add(t859, t860)
        del t859, t860

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t862 = paddle._C_ops.relu6(t858)
        del t858

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t863 = paddle._C_ops.multiply(t862, t861)
        del t861, t862

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t864 = paddle._C_ops.conv2d(
            t863, t250, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t863, t250

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t865 = paddle._C_ops.reshape(t251, t450)
        del t251

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t866 = paddle._C_ops.add(t864, t865)
        del t864, t865

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t867, t868, t869, t870, t871, t872 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t866,
                t252,
                t253,
                t254,
                t255,
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
        del t866, t255, t254, t253, t252

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t873 = paddle._C_ops.depthwise_conv2d(
            t867, t256, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t867, t256

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t874 = paddle._C_ops.reshape(t257, t450)
        del t257

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t875 = paddle._C_ops.add(t873, t874)
        del t873, t874

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t876 = paddle._C_ops.add(t846, t875)
        del t846, t875

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t877 = paddle._C_ops.depthwise_conv2d(
            t876, t258, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t258

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t878 = paddle._C_ops.reshape(t259, t450)
        del t259

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t879 = paddle._C_ops.add(t877, t878)
        del t877, t878

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t880, t881, t882, t883, t884, t885 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t879,
                t260,
                t261,
                t262,
                t263,
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
        del t879, t263, t262, t261, t260

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t886 = paddle._C_ops.conv2d(
            t880, t264, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t264

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t887 = paddle._C_ops.reshape(t265, t450)
        del t265

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t888 = paddle._C_ops.add(t886, t887)
        del t886, t887

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t889 = paddle._C_ops.conv2d(
            t880, t266, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t880, t266

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t890 = paddle._C_ops.reshape(t267, t450)
        del t267

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t891 = paddle._C_ops.add(t889, t890)
        del t889, t890

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t892 = paddle._C_ops.relu6(t888)
        del t888

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t893 = paddle._C_ops.multiply(t892, t891)
        del t891, t892

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t894 = paddle._C_ops.conv2d(
            t893, t268, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t893, t268

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t895 = paddle._C_ops.reshape(t269, t450)
        del t269

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t896 = paddle._C_ops.add(t894, t895)
        del t894, t895

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t897, t898, t899, t900, t901, t902 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t896,
                t270,
                t271,
                t272,
                t273,
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
        del t896, t273, t272, t271, t270

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t903 = paddle._C_ops.depthwise_conv2d(
            t897, t274, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t897, t274

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t904 = paddle._C_ops.reshape(t275, t450)
        del t275

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t905 = paddle._C_ops.add(t903, t904)
        del t903, t904

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t906 = paddle._C_ops.add(t876, t905)
        del t876, t905

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t907 = paddle._C_ops.depthwise_conv2d(
            t906, t276, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t276

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t908 = paddle._C_ops.reshape(t277, t450)
        del t277

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t909 = paddle._C_ops.add(t907, t908)
        del t907, t908

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t910, t911, t912, t913, t914, t915 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t909,
                t278,
                t279,
                t280,
                t281,
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
        del t909, t281, t280, t279, t278

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t916 = paddle._C_ops.conv2d(
            t910, t282, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t282

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t917 = paddle._C_ops.reshape(t283, t450)
        del t283

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t918 = paddle._C_ops.add(t916, t917)
        del t916, t917

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t919 = paddle._C_ops.conv2d(
            t910, t284, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t910, t284

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t920 = paddle._C_ops.reshape(t285, t450)
        del t285

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t921 = paddle._C_ops.add(t919, t920)
        del t919, t920

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t922 = paddle._C_ops.relu6(t918)
        del t918

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t923 = paddle._C_ops.multiply(t922, t921)
        del t921, t922

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t924 = paddle._C_ops.conv2d(
            t923, t286, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t923, t286

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t925 = paddle._C_ops.reshape(t287, t450)
        del t287

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t926 = paddle._C_ops.add(t924, t925)
        del t924, t925

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t927, t928, t929, t930, t931, t932 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t926,
                t288,
                t289,
                t290,
                t291,
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
        del t926, t291, t290, t289, t288

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t933 = paddle._C_ops.depthwise_conv2d(
            t927, t292, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t927, t292

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t934 = paddle._C_ops.reshape(t293, t450)
        del t293

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t935 = paddle._C_ops.add(t933, t934)
        del t933, t934

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t936 = paddle._C_ops.add(t906, t935)
        del t906, t935

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t937 = paddle._C_ops.depthwise_conv2d(
            t936, t294, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t294

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t938 = paddle._C_ops.reshape(t295, t450)
        del t295

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t939 = paddle._C_ops.add(t937, t938)
        del t937, t938

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t940, t941, t942, t943, t944, t945 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t939,
                t296,
                t297,
                t298,
                t299,
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
        del t939, t299, t298, t297, t296

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t946 = paddle._C_ops.conv2d(
            t940, t300, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t300

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t947 = paddle._C_ops.reshape(t301, t450)
        del t301

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t948 = paddle._C_ops.add(t946, t947)
        del t946, t947

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t949 = paddle._C_ops.conv2d(
            t940, t302, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t940, t302

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t950 = paddle._C_ops.reshape(t303, t450)
        del t303

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t951 = paddle._C_ops.add(t949, t950)
        del t949, t950

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t952 = paddle._C_ops.relu6(t948)
        del t948

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t953 = paddle._C_ops.multiply(t952, t951)
        del t951, t952

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t954 = paddle._C_ops.conv2d(
            t953, t304, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t953, t304

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t955 = paddle._C_ops.reshape(t305, t450)
        del t305

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t956 = paddle._C_ops.add(t954, t955)
        del t954, t955

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t957, t958, t959, t960, t961, t962 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t956,
                t306,
                t307,
                t308,
                t309,
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
        del t956, t309, t308, t307, t306

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t963 = paddle._C_ops.depthwise_conv2d(
            t957, t310, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t957, t310

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t964 = paddle._C_ops.reshape(t311, t450)
        del t311

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t965 = paddle._C_ops.add(t963, t964)
        del t963, t964

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t966 = paddle._C_ops.add(t936, t965)
        del t936, t965

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t967 = paddle._C_ops.depthwise_conv2d(
            t966, t312, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t312

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t968 = paddle._C_ops.reshape(t313, t450)
        del t313

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t969 = paddle._C_ops.add(t967, t968)
        del t967, t968

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t970, t971, t972, t973, t974, t975 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t969,
                t314,
                t315,
                t316,
                t317,
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
        del t969, t317, t316, t315, t314

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t976 = paddle._C_ops.conv2d(
            t970, t318, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t318

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t977 = paddle._C_ops.reshape(t319, t450)
        del t319

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t978 = paddle._C_ops.add(t976, t977)
        del t976, t977

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t979 = paddle._C_ops.conv2d(
            t970, t320, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t970, t320

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t980 = paddle._C_ops.reshape(t321, t450)
        del t321

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t981 = paddle._C_ops.add(t979, t980)
        del t979, t980

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t982 = paddle._C_ops.relu6(t978)
        del t978

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t983 = paddle._C_ops.multiply(t982, t981)
        del t981, t982

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t984 = paddle._C_ops.conv2d(
            t983, t322, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t983, t322

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t985 = paddle._C_ops.reshape(t323, t450)
        del t323

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t986 = paddle._C_ops.add(t984, t985)
        del t984, t985

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t987, t988, t989, t990, t991, t992 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t986,
                t324,
                t325,
                t326,
                t327,
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
        del t986, t327, t326, t325, t324

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t993 = paddle._C_ops.depthwise_conv2d(
            t987, t328, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t987, t328

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t994 = paddle._C_ops.reshape(t329, t450)
        del t329

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t995 = paddle._C_ops.add(t993, t994)
        del t993, t994

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t996 = paddle._C_ops.add(t966, t995)
        del t966, t995

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t997 = paddle._C_ops.depthwise_conv2d(
            t996, t330, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t330

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t998 = paddle._C_ops.reshape(t331, t450)
        del t331

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t999 = paddle._C_ops.add(t997, t998)
        del t997, t998

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1000, t1001, t1002, t1003, t1004, t1005 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t999,
                t332,
                t333,
                t334,
                t335,
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
        del t999, t335, t334, t333, t332

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t1006 = paddle._C_ops.conv2d(
            t1000, t336, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t336

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t1007 = paddle._C_ops.reshape(t337, t450)
        del t337

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t1008 = paddle._C_ops.add(t1006, t1007)
        del t1006, t1007

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t1009 = paddle._C_ops.conv2d(
            t1000, t338, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1000, t338

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t1010 = paddle._C_ops.reshape(t339, t450)
        del t339

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t1011 = paddle._C_ops.add(t1009, t1010)
        del t1009, t1010

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1012 = paddle._C_ops.relu6(t1008)
        del t1008

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t1013 = paddle._C_ops.multiply(t1012, t1011)
        del t1011, t1012

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t1014 = paddle._C_ops.conv2d(
            t1013, t340, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1013, t340

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t1015 = paddle._C_ops.reshape(t341, t450)
        del t341

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t1016 = paddle._C_ops.add(t1014, t1015)
        del t1014, t1015

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1017, t1018, t1019, t1020, t1021, t1022 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1016,
                t342,
                t343,
                t344,
                t345,
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
        del t1016, t345, t344, t343, t342

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t1023 = paddle._C_ops.depthwise_conv2d(
            t1017, t346, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t1017, t346

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t1024 = paddle._C_ops.reshape(t347, t450)
        del t347

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t1025 = paddle._C_ops.add(t1023, t1024)
        del t1023, t1024

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t1026 = paddle._C_ops.add(t996, t1025)
        del t996, t1025

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x128x14x14xf32, 256x128x3x3xf32)
        t1027 = paddle._C_ops.conv2d(
            t1026, t348, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1026, t348

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1028 = paddle._C_ops.reshape(t349, t450)
        del t349

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1029 = paddle._C_ops.add(t1027, t1028)
        del t1027, t1028

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1030, t1031, t1032, t1033, t1034, t1035 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1029,
                t350,
                t351,
                t352,
                t353,
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
        del t1029, t353, t352, t351, t350

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1036 = paddle._C_ops.depthwise_conv2d(
            t1030, t354, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t354

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1037 = paddle._C_ops.reshape(t355, t450)
        del t355

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1038 = paddle._C_ops.add(t1036, t1037)
        del t1036, t1037

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1039, t1040, t1041, t1042, t1043, t1044 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1038,
                t356,
                t357,
                t358,
                t359,
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
        del t1038, t359, t358, t357, t356

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1045 = paddle._C_ops.conv2d(
            t1039, t360, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t360

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1046 = paddle._C_ops.reshape(t361, t450)
        del t361

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1047 = paddle._C_ops.add(t1045, t1046)
        del t1045, t1046

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1048 = paddle._C_ops.conv2d(
            t1039, t362, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1039, t362

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1049 = paddle._C_ops.reshape(t363, t450)
        del t363

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1050 = paddle._C_ops.add(t1048, t1049)
        del t1048, t1049

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t1051 = paddle._C_ops.relu6(t1047)
        del t1047

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        t1052 = paddle._C_ops.multiply(t1051, t1050)
        del t1050, t1051

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        t1053 = paddle._C_ops.conv2d(
            t1052, t364, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1052, t364

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1054 = paddle._C_ops.reshape(t365, t450)
        del t365

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1055 = paddle._C_ops.add(t1053, t1054)
        del t1053, t1054

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1056, t1057, t1058, t1059, t1060, t1061 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1055,
                t366,
                t367,
                t368,
                t369,
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
        del t1055, t369, t368, t367, t366

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1062 = paddle._C_ops.depthwise_conv2d(
            t1056, t370, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t1056, t370

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1063 = paddle._C_ops.reshape(t371, t450)
        del t371

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1064 = paddle._C_ops.add(t1062, t1063)
        del t1062, t1063

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        t1065 = paddle._C_ops.add(t1030, t1064)
        del t1064, t1030

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1066 = paddle._C_ops.depthwise_conv2d(
            t1065, t372, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t372

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1067 = paddle._C_ops.reshape(t373, t450)
        del t373

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1068 = paddle._C_ops.add(t1066, t1067)
        del t1066, t1067

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1069, t1070, t1071, t1072, t1073, t1074 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1068,
                t374,
                t375,
                t376,
                t377,
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
        del t1068, t377, t376, t375, t374

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1075 = paddle._C_ops.conv2d(
            t1069, t378, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t378

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1076 = paddle._C_ops.reshape(t379, t450)
        del t379

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1077 = paddle._C_ops.add(t1075, t1076)
        del t1075, t1076

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1078 = paddle._C_ops.conv2d(
            t1069, t380, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1069, t380

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1079 = paddle._C_ops.reshape(t381, t450)
        del t381

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1080 = paddle._C_ops.add(t1078, t1079)
        del t1078, t1079

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t1081 = paddle._C_ops.relu6(t1077)
        del t1077

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        t1082 = paddle._C_ops.multiply(t1081, t1080)
        del t1080, t1081

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        t1083 = paddle._C_ops.conv2d(
            t1082, t382, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1082, t382

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1084 = paddle._C_ops.reshape(t383, t450)
        del t383

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1085 = paddle._C_ops.add(t1083, t1084)
        del t1083, t1084

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1086, t1087, t1088, t1089, t1090, t1091 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1085,
                t384,
                t385,
                t386,
                t387,
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
        del t1085, t387, t386, t385, t384

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1092 = paddle._C_ops.depthwise_conv2d(
            t1086, t388, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t1086, t388

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1093 = paddle._C_ops.reshape(t389, t450)
        del t389

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1094 = paddle._C_ops.add(t1092, t1093)
        del t1092, t1093

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        t1095 = paddle._C_ops.add(t1065, t1094)
        del t1065, t1094

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1096 = paddle._C_ops.depthwise_conv2d(
            t1095, t390, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t390

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1097 = paddle._C_ops.reshape(t391, t450)
        del t391

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1098 = paddle._C_ops.add(t1096, t1097)
        del t1096, t1097

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1099, t1100, t1101, t1102, t1103, t1104 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1098,
                t392,
                t393,
                t394,
                t395,
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
        del t1098, t395, t394, t393, t392

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1105 = paddle._C_ops.conv2d(
            t1099, t396, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t396

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1106 = paddle._C_ops.reshape(t397, t450)
        del t397

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1107 = paddle._C_ops.add(t1105, t1106)
        del t1105, t1106

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1108 = paddle._C_ops.conv2d(
            t1099, t398, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1099, t398

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1109 = paddle._C_ops.reshape(t399, t450)
        del t399

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1110 = paddle._C_ops.add(t1108, t1109)
        del t1108, t1109

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t1111 = paddle._C_ops.relu6(t1107)
        del t1107

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        t1112 = paddle._C_ops.multiply(t1111, t1110)
        del t1110, t1111

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        t1113 = paddle._C_ops.conv2d(
            t1112, t400, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1112, t400

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1114 = paddle._C_ops.reshape(t401, t450)
        del t401

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1115 = paddle._C_ops.add(t1113, t1114)
        del t1113, t1114

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1116, t1117, t1118, t1119, t1120, t1121 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1115,
                t402,
                t403,
                t404,
                t405,
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
        del t1115, t405, t404, t403, t402

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1122 = paddle._C_ops.depthwise_conv2d(
            t1116, t406, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t1116, t406

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1123 = paddle._C_ops.reshape(t407, t450)
        del t407

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1124 = paddle._C_ops.add(t1122, t1123)
        del t1122, t1123

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        t1125 = paddle._C_ops.add(t1095, t1124)
        del t1095, t1124

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1126 = paddle._C_ops.depthwise_conv2d(
            t1125, t408, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t408

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1127 = paddle._C_ops.reshape(t409, t450)
        del t409

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1128 = paddle._C_ops.add(t1126, t1127)
        del t1126, t1127

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1129, t1130, t1131, t1132, t1133, t1134 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1128,
                t410,
                t411,
                t412,
                t413,
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
        del t1128, t413, t412, t411, t410

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1135 = paddle._C_ops.conv2d(
            t1129, t414, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t414

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1136 = paddle._C_ops.reshape(t415, t450)
        del t415

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1137 = paddle._C_ops.add(t1135, t1136)
        del t1135, t1136

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1138 = paddle._C_ops.conv2d(
            t1129, t416, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1129, t416

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1139 = paddle._C_ops.reshape(t417, t450)
        del t417

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1140 = paddle._C_ops.add(t1138, t1139)
        del t1138, t1139

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t1141 = paddle._C_ops.relu6(t1137)
        del t1137

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        t1142 = paddle._C_ops.multiply(t1141, t1140)
        del t1140, t1141

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        t1143 = paddle._C_ops.conv2d(
            t1142, t418, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1142, t418

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1144 = paddle._C_ops.reshape(t419, t450)
        del t419

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1145 = paddle._C_ops.add(t1143, t1144)
        del t1143, t1144

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1146, t1147, t1148, t1149, t1150, t1151 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1145,
                t420,
                t421,
                t422,
                t423,
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
        del t1145, t423, t422, t421, t420

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1152 = paddle._C_ops.depthwise_conv2d(
            t1146, t424, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t1146, t424

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1153 = paddle._C_ops.reshape(t425, t450)
        del t425

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1154 = paddle._C_ops.add(t1152, t1153)
        del t1152, t1153

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        t1155 = paddle._C_ops.add(t1125, t1154)
        del t1125, t1154

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1156 = paddle._C_ops.depthwise_conv2d(
            t1155, t426, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t426

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1157 = paddle._C_ops.reshape(t427, t450)
        del t427

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1158 = paddle._C_ops.add(t1156, t1157)
        del t1156, t1157

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1159, t1160, t1161, t1162, t1163, t1164 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1158,
                t428,
                t429,
                t430,
                t431,
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
        del t1158, t431, t430, t429, t428

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1165 = paddle._C_ops.conv2d(
            t1159, t432, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t432

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1166 = paddle._C_ops.reshape(t433, t450)
        del t433

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1167 = paddle._C_ops.add(t1165, t1166)
        del t1165, t1166

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t1168 = paddle._C_ops.conv2d(
            t1159, t434, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1159, t434

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t1169 = paddle._C_ops.reshape(t435, t450)
        del t435

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t1170 = paddle._C_ops.add(t1168, t1169)
        del t1168, t1169

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t1171 = paddle._C_ops.relu6(t1167)
        del t1167

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        t1172 = paddle._C_ops.multiply(t1171, t1170)
        del t1170, t1171

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        t1173 = paddle._C_ops.conv2d(
            t1172, t436, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1172, t436

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1174 = paddle._C_ops.reshape(t437, t450)
        del t437

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1175 = paddle._C_ops.add(t1173, t1174)
        del t1173, t1174

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1176, t1177, t1178, t1179, t1180, t1181 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1175,
                t438,
                t439,
                t440,
                t441,
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
        del t1175, t439, t438, t441, t440

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t1182 = paddle._C_ops.depthwise_conv2d(
            t1176, t442, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t1176, t442

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1183 = paddle._C_ops.reshape(t443, t450)
        del t450, t443

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t1184 = paddle._C_ops.add(t1182, t1183)
        del t1182, t1183

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        t1185 = paddle._C_ops.add(t1155, t1184)
        del t1155, t1184

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1186, t1187, t1188, t1189, t1190, t1191 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1185,
                t444,
                t445,
                t446,
                t447,
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
        del t1185, t447, t446, t445, t444

        # pd_op.full_int_array: (2xi64) <- ()
        t1192 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x7x7xf32, 2xi64)
        t1193 = paddle._C_ops.pool2d(
            t1186,
            t1192,
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
        del t1186, t1192

        # pd_op.flatten: (-1x256xf32) <- (-1x256x1x1xf32)
        t1194 = paddle._C_ops.flatten(t1193, 1, 3)
        del t1193

        # pd_op.matmul: (-1x102xf32) <- (-1x256xf32, 256x102xf32)
        t1195 = paddle._C_ops.matmul(t1194, t448, False, False)
        del t1194, t448

        return t1195
