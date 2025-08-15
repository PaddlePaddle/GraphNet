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
    ):
        # pd_op.conv2d: (64x32x112x112xf32) <- (64x3x224x224xf32, 32x3x3x3xf32)
        t402 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (64x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t403, t404, t405, t406, t407, t408 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t402,
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

        # pd_op.relu: (64x32x112x112xf32) <- (64x32x112x112xf32)
        t409 = paddle._C_ops.relu(t403)
        del t403

        # pd_op.conv2d: (64x16x112x112xf32) <- (64x32x112x112xf32, 16x32x2x2xf32)
        t410 = paddle._C_ops.conv2d(t409, t5, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW")
        del t5

        # pd_op.batch_norm_: (64x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (64x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t411, t412, t413, t414, t415, t416 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t410,
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

        # pd_op.relu: (64x16x112x112xf32) <- (64x16x112x112xf32)
        t417 = paddle._C_ops.relu(t411)
        del t411

        # pd_op.conv2d: (64x32x112x112xf32) <- (64x16x112x112xf32, 32x16x2x2xf32)
        t418 = paddle._C_ops.conv2d(
            t417, t10, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (64x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t419, t420, t421, t422, t423, t424 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t418,
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

        # pd_op.relu: (64x32x112x112xf32) <- (64x32x112x112xf32)
        t425 = paddle._C_ops.relu(t419)
        del t419

        # pd_op.full_int_array: (2xi64) <- ()
        t426 = [2, 2]

        # pd_op.pool2d: (64x32x112x112xf32) <- (64x32x112x112xf32, 2xi64)
        t427 = paddle._C_ops.pool2d(
            t409, t426, [1, 1], [0, 0], True, True, "NCHW", "max", False, False, "SAME"
        )

        # pd_op.full: (1xi32) <- ()
        t428 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t429 = t428

        # pd_op.assign: (1xi32) <- (1xi32)
        t430 = t428

        # pd_op.assign: (1xi32) <- (1xi32)
        t431 = t428

        # pd_op.assign: (1xi32) <- (1xi32)
        t432 = t428

        # pd_op.assign: (1xi32) <- (1xi32)
        t433 = t428

        # pd_op.assign: (1xi32) <- (1xi32)
        t434 = t428

        # builtin.combine: ([64x32x112x112xf32, 64x32x112x112xf32]) <- (64x32x112x112xf32, 64x32x112x112xf32)
        t435 = [t427, t425]

        # pd_op.concat: (64x64x112x112xf32) <- ([64x32x112x112xf32, 64x32x112x112xf32], 1xi32)
        t436 = paddle._C_ops.concat(t435, t428)
        del t435

        # pd_op.conv2d: (64x32x56x56xf32) <- (64x64x112x112xf32, 32x64x3x3xf32)
        t437 = paddle._C_ops.conv2d(
            t436, t15, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t438, t439, t440, t441, t442, t443 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t437,
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

        # pd_op.relu: (64x32x56x56xf32) <- (64x32x56x56xf32)
        t444 = paddle._C_ops.relu(t438)
        del t438

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x32x56x56xf32, 48x32x1x1xf32)
        t445 = paddle._C_ops.conv2d(
            t444, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t446, t447, t448, t449, t450, t451 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t445,
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

        # pd_op.relu: (64x48x56x56xf32) <- (64x48x56x56xf32)
        t452 = paddle._C_ops.relu(t446)
        del t446

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x48x56x56xf32, 48x48x3x3xf32)
        t453 = paddle._C_ops.conv2d(
            t452, t25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t454, t455, t456, t457, t458, t459 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t453,
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

        # pd_op.relu: (64x48x56x56xf32) <- (64x48x56x56xf32)
        t460 = paddle._C_ops.relu(t454)
        del t454

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x48x56x56xf32, 48x48x3x3xf32)
        t461 = paddle._C_ops.conv2d(
            t460, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t462, t463, t464, t465, t466, t467 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t461,
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

        # pd_op.relu: (64x48x56x56xf32) <- (64x48x56x56xf32)
        t468 = paddle._C_ops.relu(t462)
        del t462

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x48x56x56xf32, 48x48x3x3xf32)
        t469 = paddle._C_ops.conv2d(
            t468, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t470, t471, t472, t473, t474, t475 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t469,
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

        # pd_op.relu: (64x48x56x56xf32) <- (64x48x56x56xf32)
        t476 = paddle._C_ops.relu(t470)
        del t470

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x48x56x56xf32, 48x48x3x3xf32)
        t477 = paddle._C_ops.conv2d(
            t476, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t478, t479, t480, t481, t482, t483 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t477,
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

        # pd_op.relu: (64x48x56x56xf32) <- (64x48x56x56xf32)
        t484 = paddle._C_ops.relu(t478)
        del t478

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x48x56x56xf32, 48x48x3x3xf32)
        t485 = paddle._C_ops.conv2d(
            t484, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t486, t487, t488, t489, t490, t491 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t485,
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

        # pd_op.relu: (64x48x56x56xf32) <- (64x48x56x56xf32)
        t492 = paddle._C_ops.relu(t486)
        del t486

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x48x56x56xf32, 48x48x3x3xf32)
        t493 = paddle._C_ops.conv2d(
            t492, t50, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t494, t495, t496, t497, t498, t499 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t493,
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

        # pd_op.relu: (64x48x56x56xf32) <- (64x48x56x56xf32)
        t500 = paddle._C_ops.relu(t494)
        del t494

        # builtin.combine: ([64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32]) <- (64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32)
        t501 = [t452, t460, t468, t476, t484, t492, t500]

        # pd_op.concat: (64x336x56x56xf32) <- ([64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32, 64x48x56x56xf32], 1xi32)
        t502 = paddle._C_ops.concat(t501, t428)
        del t501

        # pd_op.conv2d: (64x64x56x56xf32) <- (64x336x56x56xf32, 64x336x1x1xf32)
        t503 = paddle._C_ops.conv2d(
            t502, t55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (64x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (64x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t504, t505, t506, t507, t508, t509 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t503,
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

        # pd_op.relu: (64x64x56x56xf32) <- (64x64x56x56xf32)
        t510 = paddle._C_ops.relu(t504)
        del t504

        # pd_op.conv2d: (64x128x56x56xf32) <- (64x64x56x56xf32, 128x64x1x1xf32)
        t511 = paddle._C_ops.conv2d(
            t510, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (64x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t512, t513, t514, t515, t516, t517 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t511,
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

        # pd_op.relu: (64x128x56x56xf32) <- (64x128x56x56xf32)
        t518 = paddle._C_ops.relu(t512)
        del t512

        # pd_op.depthwise_conv2d: (64x128x28x28xf32) <- (64x128x56x56xf32, 128x1x3x3xf32)
        t519 = paddle._C_ops.depthwise_conv2d(
            t518, t65, [2, 2], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (64x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t520, t521, t522, t523, t524, t525 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t519,
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

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x128x28x28xf32, 96x128x3x3xf32)
        t526 = paddle._C_ops.conv2d(
            t520, t70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t527, t528, t529, t530, t531, t532 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t526,
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

        # pd_op.relu: (64x96x28x28xf32) <- (64x96x28x28xf32)
        t533 = paddle._C_ops.relu(t527)
        del t527

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x96x28x28xf32, 96x96x3x3xf32)
        t534 = paddle._C_ops.conv2d(
            t533, t75, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t535, t536, t537, t538, t539, t540 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t534,
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

        # pd_op.relu: (64x96x28x28xf32) <- (64x96x28x28xf32)
        t541 = paddle._C_ops.relu(t535)
        del t535

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x96x28x28xf32, 96x96x3x3xf32)
        t542 = paddle._C_ops.conv2d(
            t541, t80, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t543, t544, t545, t546, t547, t548 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t542,
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

        # pd_op.relu: (64x96x28x28xf32) <- (64x96x28x28xf32)
        t549 = paddle._C_ops.relu(t543)
        del t543

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x96x28x28xf32, 96x96x3x3xf32)
        t550 = paddle._C_ops.conv2d(
            t549, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t551, t552, t553, t554, t555, t556 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t550,
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

        # pd_op.relu: (64x96x28x28xf32) <- (64x96x28x28xf32)
        t557 = paddle._C_ops.relu(t551)
        del t551

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x96x28x28xf32, 96x96x3x3xf32)
        t558 = paddle._C_ops.conv2d(
            t557, t90, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t559, t560, t561, t562, t563, t564 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t558,
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

        # pd_op.relu: (64x96x28x28xf32) <- (64x96x28x28xf32)
        t565 = paddle._C_ops.relu(t559)
        del t559

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x96x28x28xf32, 96x96x3x3xf32)
        t566 = paddle._C_ops.conv2d(
            t565, t95, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t567, t568, t569, t570, t571, t572 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t566,
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

        # pd_op.relu: (64x96x28x28xf32) <- (64x96x28x28xf32)
        t573 = paddle._C_ops.relu(t567)
        del t567

        # builtin.combine: ([64x128x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32]) <- (64x128x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32)
        t574 = [t520, t533, t541, t549, t557, t565, t573]

        # pd_op.concat: (64x704x28x28xf32) <- ([64x128x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32, 64x96x28x28xf32], 1xi32)
        t575 = paddle._C_ops.concat(t574, t428)
        del t574

        # pd_op.conv2d: (64x256x28x28xf32) <- (64x704x28x28xf32, 256x704x1x1xf32)
        t576 = paddle._C_ops.conv2d(
            t575, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (64x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (64x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t577, t578, t579, t580, t581, t582 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t576,
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

        # pd_op.relu: (64x256x28x28xf32) <- (64x256x28x28xf32)
        t583 = paddle._C_ops.relu(t577)
        del t577

        # pd_op.conv2d: (64x512x28x28xf32) <- (64x256x28x28xf32, 512x256x1x1xf32)
        t584 = paddle._C_ops.conv2d(
            t583, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (64x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (64x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t585, t586, t587, t588, t589, t590 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t584,
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

        # pd_op.relu: (64x512x28x28xf32) <- (64x512x28x28xf32)
        t591 = paddle._C_ops.relu(t585)
        del t585

        # pd_op.depthwise_conv2d: (64x512x14x14xf32) <- (64x512x28x28xf32, 512x1x3x3xf32)
        t592 = paddle._C_ops.depthwise_conv2d(
            t591, t110, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t593, t594, t595, t596, t597, t598 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t592,
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

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x512x14x14xf32, 192x512x1x1xf32)
        t599 = paddle._C_ops.conv2d(
            t593, t115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t600, t601, t602, t603, t604, t605 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t599,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t606 = paddle._C_ops.depthwise_conv2d(
            t600, t120, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t607, t608, t609, t610, t611, t612 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t606,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t613 = paddle._C_ops.relu(t607)
        del t607

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t614 = paddle._C_ops.conv2d(
            t613, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t615, t616, t617, t618, t619, t620 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t614,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t621 = paddle._C_ops.depthwise_conv2d(
            t615, t130, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t622, t623, t624, t625, t626, t627 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t621,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t628 = paddle._C_ops.relu(t622)
        del t622

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t629 = paddle._C_ops.conv2d(
            t628, t135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t630, t631, t632, t633, t634, t635 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t629,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t636 = paddle._C_ops.depthwise_conv2d(
            t630, t140, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t637, t638, t639, t640, t641, t642 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t636,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t643 = paddle._C_ops.relu(t637)
        del t637

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t644 = paddle._C_ops.conv2d(
            t643, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t645, t646, t647, t648, t649, t650 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t644,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t651 = paddle._C_ops.depthwise_conv2d(
            t645, t150, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t652, t653, t654, t655, t656, t657 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t651,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t658 = paddle._C_ops.relu(t652)
        del t652

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t659 = paddle._C_ops.conv2d(
            t658, t155, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t660, t661, t662, t663, t664, t665 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t659,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t666 = paddle._C_ops.depthwise_conv2d(
            t660, t160, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t667, t668, t669, t670, t671, t672 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t666,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t673 = paddle._C_ops.relu(t667)
        del t667

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t674 = paddle._C_ops.conv2d(
            t673, t165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t675, t676, t677, t678, t679, t680 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t674,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t681 = paddle._C_ops.depthwise_conv2d(
            t675, t170, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t682, t683, t684, t685, t686, t687 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t681,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t688 = paddle._C_ops.relu(t682)
        del t682

        # builtin.combine: ([64x512x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32]) <- (64x512x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32)
        t689 = [t593, t613, t628, t643, t658, t673, t688]

        # pd_op.concat: (64x1664x14x14xf32) <- ([64x512x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32], 1xi32)
        t690 = paddle._C_ops.concat(t689, t428)
        del t689

        # pd_op.conv2d: (64x512x14x14xf32) <- (64x1664x14x14xf32, 512x1664x1x1xf32)
        t691 = paddle._C_ops.conv2d(
            t690, t175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t692, t693, t694, t695, t696, t697 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t691,
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

        # pd_op.relu: (64x512x14x14xf32) <- (64x512x14x14xf32)
        t698 = paddle._C_ops.relu(t692)
        del t692

        # pd_op.conv2d: (64x1024x14x14xf32) <- (64x512x14x14xf32, 1024x512x1x1xf32)
        t699 = paddle._C_ops.conv2d(
            t698, t180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180

        # pd_op.batch_norm_: (64x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (64x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t700, t701, t702, t703, t704, t705 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t699,
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

        # pd_op.relu: (64x1024x14x14xf32) <- (64x1024x14x14xf32)
        t706 = paddle._C_ops.relu(t700)
        del t700

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x1024x14x14xf32, 192x1024x1x1xf32)
        t707 = paddle._C_ops.conv2d(
            t706, t185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t708, t709, t710, t711, t712, t713 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t707,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t714 = paddle._C_ops.depthwise_conv2d(
            t708, t190, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t715, t716, t717, t718, t719, t720 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t714,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t721 = paddle._C_ops.relu(t715)
        del t715

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t722 = paddle._C_ops.conv2d(
            t721, t195, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t723, t724, t725, t726, t727, t728 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t722,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t729 = paddle._C_ops.depthwise_conv2d(
            t723, t200, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t730, t731, t732, t733, t734, t735 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t729,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t736 = paddle._C_ops.relu(t730)
        del t730

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t737 = paddle._C_ops.conv2d(
            t736, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t738, t739, t740, t741, t742, t743 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t737,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t744 = paddle._C_ops.depthwise_conv2d(
            t738, t210, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t745, t746, t747, t748, t749, t750 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t744,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t751 = paddle._C_ops.relu(t745)
        del t745

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t752 = paddle._C_ops.conv2d(
            t751, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t753, t754, t755, t756, t757, t758 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t752,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t759 = paddle._C_ops.depthwise_conv2d(
            t753, t220, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t220

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t760, t761, t762, t763, t764, t765 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t759,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t766 = paddle._C_ops.relu(t760)
        del t760

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t767 = paddle._C_ops.conv2d(
            t766, t225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t768, t769, t770, t771, t772, t773 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t767,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t774 = paddle._C_ops.depthwise_conv2d(
            t768, t230, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t230

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t775, t776, t777, t778, t779, t780 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t774,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t781 = paddle._C_ops.relu(t775)
        del t775

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t782 = paddle._C_ops.conv2d(
            t781, t235, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t783, t784, t785, t786, t787, t788 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t782,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t789 = paddle._C_ops.depthwise_conv2d(
            t783, t240, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t240

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t790, t791, t792, t793, t794, t795 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t789,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t796 = paddle._C_ops.relu(t790)
        del t790

        # builtin.combine: ([64x1024x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32]) <- (64x1024x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32)
        t797 = [t706, t721, t736, t751, t766, t781, t796]

        # pd_op.concat: (64x2176x14x14xf32) <- ([64x1024x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32], 1xi32)
        t798 = paddle._C_ops.concat(t797, t428)
        del t797

        # pd_op.conv2d: (64x512x14x14xf32) <- (64x2176x14x14xf32, 512x2176x1x1xf32)
        t799 = paddle._C_ops.conv2d(
            t798, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t800, t801, t802, t803, t804, t805 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t799,
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

        # pd_op.relu: (64x512x14x14xf32) <- (64x512x14x14xf32)
        t806 = paddle._C_ops.relu(t800)
        del t800

        # pd_op.conv2d: (64x1024x14x14xf32) <- (64x512x14x14xf32, 1024x512x1x1xf32)
        t807 = paddle._C_ops.conv2d(
            t806, t250, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t250

        # pd_op.batch_norm_: (64x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (64x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t808, t809, t810, t811, t812, t813 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t807,
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

        # pd_op.relu: (64x1024x14x14xf32) <- (64x1024x14x14xf32)
        t814 = paddle._C_ops.relu(t808)
        del t808

        # pd_op.add: (64x1024x14x14xf32) <- (64x1024x14x14xf32, 64x1024x14x14xf32)
        t815 = paddle._C_ops.add(t814, t706)

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x1024x14x14xf32, 192x1024x1x1xf32)
        t816 = paddle._C_ops.conv2d(
            t815, t255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t817, t818, t819, t820, t821, t822 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t816,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t823 = paddle._C_ops.depthwise_conv2d(
            t817, t260, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t260

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t824, t825, t826, t827, t828, t829 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t823,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t830 = paddle._C_ops.relu(t824)
        del t824

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t831 = paddle._C_ops.conv2d(
            t830, t265, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t265

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t832, t833, t834, t835, t836, t837 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t831,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t838 = paddle._C_ops.depthwise_conv2d(
            t832, t270, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t270

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t839, t840, t841, t842, t843, t844 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t838,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t845 = paddle._C_ops.relu(t839)
        del t839

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t846 = paddle._C_ops.conv2d(
            t845, t275, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t275

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t847, t848, t849, t850, t851, t852 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t846,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t853 = paddle._C_ops.depthwise_conv2d(
            t847, t280, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t280

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t854, t855, t856, t857, t858, t859 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t853,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t860 = paddle._C_ops.relu(t854)
        del t854

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t861 = paddle._C_ops.conv2d(
            t860, t285, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t285

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t862, t863, t864, t865, t866, t867 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t861,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t868 = paddle._C_ops.depthwise_conv2d(
            t862, t290, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t290

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t869, t870, t871, t872, t873, t874 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t868,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t875 = paddle._C_ops.relu(t869)
        del t869

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t876 = paddle._C_ops.conv2d(
            t875, t295, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t295

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t877, t878, t879, t880, t881, t882 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t876,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t883 = paddle._C_ops.depthwise_conv2d(
            t877, t300, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t300

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t884, t885, t886, t887, t888, t889 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t883,
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
        del t302, t301, t304, t303

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t890 = paddle._C_ops.relu(t884)
        del t884

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x1x1xf32)
        t891 = paddle._C_ops.conv2d(
            t890, t305, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t305

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t892, t893, t894, t895, t896, t897 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t891,
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

        # pd_op.depthwise_conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x1x5x5xf32)
        t898 = paddle._C_ops.depthwise_conv2d(
            t892, t310, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t310

        # pd_op.batch_norm_: (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t899, t900, t901, t902, t903, t904 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t898,
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

        # pd_op.relu: (64x192x14x14xf32) <- (64x192x14x14xf32)
        t905 = paddle._C_ops.relu(t899)
        del t899

        # builtin.combine: ([64x1024x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32]) <- (64x1024x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32)
        t906 = [t815, t830, t845, t860, t875, t890, t905]

        # pd_op.concat: (64x2176x14x14xf32) <- ([64x1024x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32, 64x192x14x14xf32], 1xi32)
        t907 = paddle._C_ops.concat(t906, t428)
        del t906

        # pd_op.conv2d: (64x512x14x14xf32) <- (64x2176x14x14xf32, 512x2176x1x1xf32)
        t908 = paddle._C_ops.conv2d(
            t907, t315, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t315

        # pd_op.batch_norm_: (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t909, t910, t911, t912, t913, t914 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t908,
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

        # pd_op.relu: (64x512x14x14xf32) <- (64x512x14x14xf32)
        t915 = paddle._C_ops.relu(t909)
        del t909

        # pd_op.conv2d: (64x1024x14x14xf32) <- (64x512x14x14xf32, 1024x512x1x1xf32)
        t916 = paddle._C_ops.conv2d(
            t915, t320, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t320

        # pd_op.batch_norm_: (64x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (64x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t917, t918, t919, t920, t921, t922 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t916,
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

        # pd_op.relu: (64x1024x14x14xf32) <- (64x1024x14x14xf32)
        t923 = paddle._C_ops.relu(t917)
        del t917

        # pd_op.add: (64x1024x14x14xf32) <- (64x1024x14x14xf32, 64x1024x14x14xf32)
        t924 = paddle._C_ops.add(t923, t815)

        # pd_op.depthwise_conv2d: (64x1024x7x7xf32) <- (64x1024x14x14xf32, 1024x1x3x3xf32)
        t925 = paddle._C_ops.depthwise_conv2d(
            t924, t325, [2, 2], [1, 1], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del t325

        # pd_op.batch_norm_: (64x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (64x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t926, t927, t928, t929, t930, t931 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t925,
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

        # pd_op.conv2d: (64x384x7x7xf32) <- (64x1024x7x7xf32, 384x1024x1x1xf32)
        t932 = paddle._C_ops.conv2d(
            t926, t330, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t330

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t933, t934, t935, t936, t937, t938 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t932,
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

        # pd_op.depthwise_conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x1x5x5xf32)
        t939 = paddle._C_ops.depthwise_conv2d(
            t933, t335, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t335

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t940, t941, t942, t943, t944, t945 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t939,
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

        # pd_op.relu: (64x384x7x7xf32) <- (64x384x7x7xf32)
        t946 = paddle._C_ops.relu(t940)
        del t940

        # pd_op.conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x384x1x1xf32)
        t947 = paddle._C_ops.conv2d(
            t946, t340, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t340

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t948, t949, t950, t951, t952, t953 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t947,
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

        # pd_op.depthwise_conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x1x5x5xf32)
        t954 = paddle._C_ops.depthwise_conv2d(
            t948, t345, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t345

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t955, t956, t957, t958, t959, t960 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t954,
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

        # pd_op.relu: (64x384x7x7xf32) <- (64x384x7x7xf32)
        t961 = paddle._C_ops.relu(t955)
        del t955

        # pd_op.conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x384x1x1xf32)
        t962 = paddle._C_ops.conv2d(
            t961, t350, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t350

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t963, t964, t965, t966, t967, t968 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t962,
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

        # pd_op.depthwise_conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x1x5x5xf32)
        t969 = paddle._C_ops.depthwise_conv2d(
            t963, t355, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t355

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t970, t971, t972, t973, t974, t975 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t969,
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

        # pd_op.relu: (64x384x7x7xf32) <- (64x384x7x7xf32)
        t976 = paddle._C_ops.relu(t970)
        del t970

        # pd_op.conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x384x1x1xf32)
        t977 = paddle._C_ops.conv2d(
            t976, t360, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t360

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t978, t979, t980, t981, t982, t983 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t977,
                t361,
                t362,
                t363,
                t364,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t364, t363, t362, t361

        # pd_op.depthwise_conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x1x5x5xf32)
        t984 = paddle._C_ops.depthwise_conv2d(
            t978, t365, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t365

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t985, t986, t987, t988, t989, t990 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t984,
                t366,
                t367,
                t368,
                t369,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t369, t368, t367, t366

        # pd_op.relu: (64x384x7x7xf32) <- (64x384x7x7xf32)
        t991 = paddle._C_ops.relu(t985)
        del t985

        # pd_op.conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x384x1x1xf32)
        t992 = paddle._C_ops.conv2d(
            t991, t370, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t370

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t993, t994, t995, t996, t997, t998 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t992,
                t371,
                t372,
                t373,
                t374,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t374, t373, t372, t371

        # pd_op.depthwise_conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x1x5x5xf32)
        t999 = paddle._C_ops.depthwise_conv2d(
            t993, t375, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t375

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1000, t1001, t1002, t1003, t1004, t1005 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t999,
                t376,
                t377,
                t378,
                t379,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t379, t378, t377, t376

        # pd_op.relu: (64x384x7x7xf32) <- (64x384x7x7xf32)
        t1006 = paddle._C_ops.relu(t1000)
        del t1000

        # pd_op.conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x384x1x1xf32)
        t1007 = paddle._C_ops.conv2d(
            t1006, t380, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t380

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1008, t1009, t1010, t1011, t1012, t1013 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1007,
                t381,
                t382,
                t383,
                t384,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t384, t383, t382, t381

        # pd_op.depthwise_conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x1x5x5xf32)
        t1014 = paddle._C_ops.depthwise_conv2d(
            t1008, t385, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t385

        # pd_op.batch_norm_: (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1015, t1016, t1017, t1018, t1019, t1020 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1014,
                t386,
                t387,
                t388,
                t389,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t389, t388, t387, t386

        # pd_op.relu: (64x384x7x7xf32) <- (64x384x7x7xf32)
        t1021 = paddle._C_ops.relu(t1015)
        del t1015

        # builtin.combine: ([64x1024x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32]) <- (64x1024x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32)
        t1022 = [t926, t946, t961, t976, t991, t1006, t1021]

        # pd_op.concat: (64x3328x7x7xf32) <- ([64x1024x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32, 64x384x7x7xf32], 1xi32)
        t1023 = paddle._C_ops.concat(t1022, t428)
        del t1022

        # pd_op.conv2d: (64x1024x7x7xf32) <- (64x3328x7x7xf32, 1024x3328x1x1xf32)
        t1024 = paddle._C_ops.conv2d(
            t1023, t390, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t390

        # pd_op.batch_norm_: (64x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (64x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1025, t1026, t1027, t1028, t1029, t1030 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1024,
                t391,
                t392,
                t393,
                t394,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t392, t391, t394, t393

        # pd_op.relu: (64x1024x7x7xf32) <- (64x1024x7x7xf32)
        t1031 = paddle._C_ops.relu(t1025)
        del t1025

        # pd_op.conv2d: (64x2048x7x7xf32) <- (64x1024x7x7xf32, 2048x1024x1x1xf32)
        t1032 = paddle._C_ops.conv2d(
            t1031, t395, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t395

        # pd_op.batch_norm_: (64x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (64x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t1033, t1034, t1035, t1036, t1037, t1038 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1032,
                t396,
                t397,
                t398,
                t399,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t399, t398, t397, t396

        # pd_op.relu: (64x2048x7x7xf32) <- (64x2048x7x7xf32)
        t1039 = paddle._C_ops.relu(t1033)
        del t1033

        # pd_op.full_int_array: (2xi64) <- ()
        t1040 = [1, 1]

        # pd_op.pool2d: (64x2048x1x1xf32) <- (64x2048x7x7xf32, 2xi64)
        t1041 = paddle._C_ops.pool2d(
            t1039,
            t1040,
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

        # pd_op.conv2d: (64x2048x1x1xf32) <- (64x2048x1x1xf32, 2048x2048x1x1xf32)
        t1042 = paddle._C_ops.conv2d(
            t1041, t400, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t400

        # pd_op.relu: (64x2048x1x1xf32) <- (64x2048x1x1xf32)
        t1043 = paddle._C_ops.relu(t1042)
        del t1042

        # pd_op.flatten: (64x2048xf32) <- (64x2048x1x1xf32)
        t1044 = paddle._C_ops.flatten(t1043, 1, 3)

        # pd_op.matmul: (64x102xf32) <- (64x2048xf32, 2048x102xf32)
        t1045 = paddle._C_ops.matmul(t1044, t401, False, False)
        del t401

        return (
            t402,
            t404,
            t405,
            t406,
            t407,
            t408,
            t409,
            t410,
            t412,
            t413,
            t414,
            t415,
            t416,
            t417,
            t418,
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
            t436,
            t437,
            t439,
            t440,
            t441,
            t442,
            t443,
            t444,
            t445,
            t447,
            t448,
            t449,
            t450,
            t451,
            t452,
            t453,
            t455,
            t456,
            t457,
            t458,
            t459,
            t460,
            t461,
            t463,
            t464,
            t465,
            t466,
            t467,
            t468,
            t469,
            t471,
            t472,
            t473,
            t474,
            t475,
            t476,
            t477,
            t479,
            t480,
            t481,
            t482,
            t483,
            t484,
            t485,
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
            t500,
            t502,
            t503,
            t505,
            t506,
            t507,
            t508,
            t509,
            t510,
            t511,
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
            t528,
            t529,
            t530,
            t531,
            t532,
            t533,
            t534,
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
            t547,
            t548,
            t549,
            t550,
            t552,
            t553,
            t554,
            t555,
            t556,
            t557,
            t558,
            t560,
            t561,
            t562,
            t563,
            t564,
            t565,
            t566,
            t568,
            t569,
            t570,
            t571,
            t572,
            t573,
            t575,
            t576,
            t578,
            t579,
            t580,
            t581,
            t582,
            t583,
            t584,
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
            t683,
            t684,
            t685,
            t686,
            t687,
            t688,
            t690,
            t691,
            t693,
            t694,
            t695,
            t696,
            t697,
            t698,
            t699,
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
            t791,
            t792,
            t793,
            t794,
            t795,
            t796,
            t798,
            t799,
            t801,
            t802,
            t803,
            t804,
            t805,
            t806,
            t807,
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
            t900,
            t901,
            t902,
            t903,
            t904,
            t905,
            t907,
            t908,
            t910,
            t911,
            t912,
            t913,
            t914,
            t915,
            t916,
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
            t1016,
            t1017,
            t1018,
            t1019,
            t1020,
            t1021,
            t1023,
            t1024,
            t1026,
            t1027,
            t1028,
            t1029,
            t1030,
            t1031,
            t1032,
            t1034,
            t1035,
            t1036,
            t1037,
            t1038,
            t1039,
            t1040,
            t1041,
            t1043,
            t1044,
            t1045,
        )
