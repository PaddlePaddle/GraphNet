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
        input1: paddle.Tensor,
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
        input2: paddle.Tensor,
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
        input3: paddle.Tensor,
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
        input4: paddle.Tensor,
        input5: paddle.Tensor,
        input6: paddle.Tensor,
        input7: paddle.Tensor,
        input8: paddle.Tensor,
    ):
        # pd_op.conv2d: (4x32x256x256xf32) <- (4x3x512x512xf32, 32x3x3x3xf32)
        t389 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (4x32x256x256xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (4x32x256x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t390, t391, t392, t393, t394, t395 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t389,
                t1,
                t2,
                t3,
                t4,
                False,
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
        del t4, t3, t2, t1

        # pd_op.relu: (4x32x256x256xf32) <- (4x32x256x256xf32)
        t396 = paddle._C_ops.relu(t390)
        del t390

        # pd_op.conv2d: (4x32x256x256xf32) <- (4x32x256x256xf32, 32x32x3x3xf32)
        t397 = paddle._C_ops.conv2d(
            t396, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (4x32x256x256xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (4x32x256x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t398, t399, t400, t401, t402, t403 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t397,
                t6,
                t7,
                t8,
                t9,
                False,
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
        del t9, t8, t7, t6

        # pd_op.relu: (4x32x256x256xf32) <- (4x32x256x256xf32)
        t404 = paddle._C_ops.relu(t398)
        del t398

        # pd_op.conv2d: (4x64x256x256xf32) <- (4x32x256x256xf32, 64x32x3x3xf32)
        t405 = paddle._C_ops.conv2d(
            t404, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (4x64x256x256xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x256x256xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t406, t407, t408, t409, t410, t411 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t405,
                t11,
                t12,
                t13,
                t14,
                False,
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
        del t14, t13, t12, t11

        # pd_op.relu: (4x64x256x256xf32) <- (4x64x256x256xf32)
        t412 = paddle._C_ops.relu(t406)
        del t406

        # pd_op.full_int_array: (2xi64) <- ()
        t413 = [3, 3]

        # pd_op.pool2d: (4x64x128x128xf32) <- (4x64x256x256xf32, 2xi64)
        t414 = paddle._C_ops.pool2d(
            t412,
            t413,
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

        # pd_op.conv2d: (4x64x128x128xf32) <- (4x64x128x128xf32, 64x64x1x1xf32)
        t415 = paddle._C_ops.conv2d(
            t414, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t416, t417, t418, t419, t420, t421 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t415,
                t16,
                t17,
                t18,
                t19,
                False,
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
        del t19, t18, t17, t16

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        t422 = paddle._C_ops.relu(t416)
        del t416

        # pd_op.conv2d: (4x64x128x128xf32) <- (4x64x128x128xf32, 64x64x3x3xf32)
        t423 = paddle._C_ops.conv2d(
            t422, t20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t424, t425, t426, t427, t428, t429 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t423,
                t21,
                t22,
                t23,
                t24,
                False,
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
        del t24, t23, t22, t21

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        t430 = paddle._C_ops.relu(t424)
        del t424

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x64x128x128xf32, 256x64x1x1xf32)
        t431 = paddle._C_ops.conv2d(
            t430, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t432, t433, t434, t435, t436, t437 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t431,
                t26,
                t27,
                t28,
                t29,
                False,
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
        del t29, t28, t27, t26

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x64x128x128xf32, 256x64x1x1xf32)
        t438 = paddle._C_ops.conv2d(
            t414, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t439, t440, t441, t442, t443, t444 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t438,
                t31,
                t32,
                t33,
                t34,
                False,
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
        del t34, t33, t32, t31

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 4x256x128x128xf32)
        t445 = paddle._C_ops.add(t432, t439)

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t446 = paddle._C_ops.relu(t445)
        del t445

        # pd_op.conv2d: (4x64x128x128xf32) <- (4x256x128x128xf32, 64x256x1x1xf32)
        t447 = paddle._C_ops.conv2d(
            t446, t35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t448, t449, t450, t451, t452, t453 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t447,
                t36,
                t37,
                t38,
                t39,
                False,
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
        del t39, t38, t37, t36

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        t454 = paddle._C_ops.relu(t448)
        del t448

        # pd_op.conv2d: (4x64x128x128xf32) <- (4x64x128x128xf32, 64x64x3x3xf32)
        t455 = paddle._C_ops.conv2d(
            t454, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t456, t457, t458, t459, t460, t461 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t455,
                t41,
                t42,
                t43,
                t44,
                False,
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
        del t44, t43, t42, t41

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        t462 = paddle._C_ops.relu(t456)
        del t456

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x64x128x128xf32, 256x64x1x1xf32)
        t463 = paddle._C_ops.conv2d(
            t462, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t464, t465, t466, t467, t468, t469 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t463,
                t46,
                t47,
                t48,
                t49,
                False,
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
        del t49, t48, t47, t46

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 4x256x128x128xf32)
        t470 = paddle._C_ops.add(t464, t446)

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t471 = paddle._C_ops.relu(t470)
        del t470

        # pd_op.conv2d: (4x64x128x128xf32) <- (4x256x128x128xf32, 64x256x1x1xf32)
        t472 = paddle._C_ops.conv2d(
            t471, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t473, t474, t475, t476, t477, t478 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t472,
                t51,
                t52,
                t53,
                t54,
                False,
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
        del t54, t53, t52, t51

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        t479 = paddle._C_ops.relu(t473)
        del t473

        # pd_op.conv2d: (4x64x128x128xf32) <- (4x64x128x128xf32, 64x64x3x3xf32)
        t480 = paddle._C_ops.conv2d(
            t479, t55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t481, t482, t483, t484, t485, t486 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t480,
                t56,
                t57,
                t58,
                t59,
                False,
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
        del t59, t58, t57, t56

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        t487 = paddle._C_ops.relu(t481)
        del t481

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x64x128x128xf32, 256x64x1x1xf32)
        t488 = paddle._C_ops.conv2d(
            t487, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t489, t490, t491, t492, t493, t494 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t488,
                t61,
                t62,
                t63,
                t64,
                False,
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
        del t64, t63, t62, t61

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 4x256x128x128xf32)
        t495 = paddle._C_ops.add(t489, t471)

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t496 = paddle._C_ops.relu(t495)
        del t495

        # pd_op.conv2d: (4x128x128x128xf32) <- (4x256x128x128xf32, 128x256x1x1xf32)
        t497 = paddle._C_ops.conv2d(
            t496, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (4x128x128x128xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x128x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t498, t499, t500, t501, t502, t503 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t497,
                t66,
                t67,
                t68,
                t69,
                False,
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
        del t69, t68, t67, t66

        # pd_op.relu: (4x128x128x128xf32) <- (4x128x128x128xf32)
        t504 = paddle._C_ops.relu(t498)
        del t498

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x128x128x128xf32, 128x128x3x3xf32)
        t505 = paddle._C_ops.conv2d(
            t504, t70, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t506, t507, t508, t509, t510, t511 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t505,
                t71,
                t72,
                t73,
                t74,
                False,
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
        del t74, t73, t72, t71

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        t512 = paddle._C_ops.relu(t506)
        del t506

        # pd_op.conv2d: (4x512x64x64xf32) <- (4x128x64x64xf32, 512x128x1x1xf32)
        t513 = paddle._C_ops.conv2d(
            t512, t75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t514, t515, t516, t517, t518, t519 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t513,
                t76,
                t77,
                t78,
                t79,
                False,
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
        del t79, t78, t77, t76

        # pd_op.full_int_array: (2xi64) <- ()
        t520 = [2, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        t521 = t520

        # pd_op.assign: (2xi64) <- (2xi64)
        t522 = t520

        # pd_op.pool2d: (4x256x64x64xf32) <- (4x256x128x128xf32, 2xi64)
        t523 = paddle._C_ops.pool2d(
            t496,
            t520,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.conv2d: (4x512x64x64xf32) <- (4x256x64x64xf32, 512x256x1x1xf32)
        t524 = paddle._C_ops.conv2d(
            t523, t80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t525, t526, t527, t528, t529, t530 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t524,
                t81,
                t82,
                t83,
                t84,
                False,
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
        del t84, t83, t82, t81

        # pd_op.add: (4x512x64x64xf32) <- (4x512x64x64xf32, 4x512x64x64xf32)
        t531 = paddle._C_ops.add(t514, t525)

        # pd_op.relu: (4x512x64x64xf32) <- (4x512x64x64xf32)
        t532 = paddle._C_ops.relu(t531)
        del t531

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x512x64x64xf32, 128x512x1x1xf32)
        t533 = paddle._C_ops.conv2d(
            t532, t85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t534, t535, t536, t537, t538, t539 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t533,
                t86,
                t87,
                t88,
                t89,
                False,
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
        del t89, t88, t87, t86

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        t540 = paddle._C_ops.relu(t534)
        del t534

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x128x64x64xf32, 128x128x3x3xf32)
        t541 = paddle._C_ops.conv2d(
            t540, t90, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t542, t543, t544, t545, t546, t547 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t541,
                t91,
                t92,
                t93,
                t94,
                False,
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
        del t94, t93, t92, t91

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        t548 = paddle._C_ops.relu(t542)
        del t542

        # pd_op.conv2d: (4x512x64x64xf32) <- (4x128x64x64xf32, 512x128x1x1xf32)
        t549 = paddle._C_ops.conv2d(
            t548, t95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t550, t551, t552, t553, t554, t555 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t549,
                t96,
                t97,
                t98,
                t99,
                False,
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
        del t99, t98, t97, t96

        # pd_op.add: (4x512x64x64xf32) <- (4x512x64x64xf32, 4x512x64x64xf32)
        t556 = paddle._C_ops.add(t550, t532)

        # pd_op.relu: (4x512x64x64xf32) <- (4x512x64x64xf32)
        t557 = paddle._C_ops.relu(t556)
        del t556

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x512x64x64xf32, 128x512x1x1xf32)
        t558 = paddle._C_ops.conv2d(
            t557, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t559, t560, t561, t562, t563, t564 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t558,
                t101,
                t102,
                t103,
                t104,
                False,
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
        del t104, t103, t102, t101

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        t565 = paddle._C_ops.relu(t559)
        del t559

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x128x64x64xf32, 128x128x3x3xf32)
        t566 = paddle._C_ops.conv2d(
            t565, t105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t567, t568, t569, t570, t571, t572 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t566,
                t106,
                t107,
                t108,
                t109,
                False,
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
        del t109, t108, t107, t106

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        t573 = paddle._C_ops.relu(t567)
        del t567

        # pd_op.conv2d: (4x512x64x64xf32) <- (4x128x64x64xf32, 512x128x1x1xf32)
        t574 = paddle._C_ops.conv2d(
            t573, t110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t575, t576, t577, t578, t579, t580 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t574,
                t111,
                t112,
                t113,
                t114,
                False,
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
        del t114, t113, t112, t111

        # pd_op.add: (4x512x64x64xf32) <- (4x512x64x64xf32, 4x512x64x64xf32)
        t581 = paddle._C_ops.add(t575, t557)

        # pd_op.relu: (4x512x64x64xf32) <- (4x512x64x64xf32)
        t582 = paddle._C_ops.relu(t581)
        del t581

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x512x64x64xf32, 128x512x1x1xf32)
        t583 = paddle._C_ops.conv2d(
            t582, t115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t584, t585, t586, t587, t588, t589 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t583,
                t116,
                t117,
                t118,
                t119,
                False,
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
        del t119, t118, t117, t116

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        t590 = paddle._C_ops.relu(t584)
        del t584

        # pd_op.conv2d: (4x128x64x64xf32) <- (4x128x64x64xf32, 128x128x3x3xf32)
        t591 = paddle._C_ops.conv2d(
            t590, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t592, t593, t594, t595, t596, t597 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t591,
                t121,
                t122,
                t123,
                t124,
                False,
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
        del t124, t123, t122, t121

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        t598 = paddle._C_ops.relu(t592)
        del t592

        # pd_op.conv2d: (4x512x64x64xf32) <- (4x128x64x64xf32, 512x128x1x1xf32)
        t599 = paddle._C_ops.conv2d(
            t598, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t600, t601, t602, t603, t604, t605 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t599,
                t126,
                t127,
                t128,
                t129,
                False,
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
        del t129, t128, t127, t126

        # pd_op.add: (4x512x64x64xf32) <- (4x512x64x64xf32, 4x512x64x64xf32)
        t606 = paddle._C_ops.add(t600, t582)

        # pd_op.relu: (4x512x64x64xf32) <- (4x512x64x64xf32)
        t607 = paddle._C_ops.relu(t606)
        del t606

        # pd_op.conv2d: (4x256x64x64xf32) <- (4x512x64x64xf32, 256x512x1x1xf32)
        t608 = paddle._C_ops.conv2d(
            t607, t130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t609, t610, t611, t612, t613, t614 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t608,
                t131,
                t132,
                t133,
                t134,
                False,
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
        del t134, t133, t132, t131

        # pd_op.relu: (4x256x64x64xf32) <- (4x256x64x64xf32)
        t615 = paddle._C_ops.relu(t609)
        del t609

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x256x64x64xf32, 256x256x3x3xf32)
        t616 = paddle._C_ops.conv2d(
            t615, t135, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t617, t618, t619, t620, t621, t622 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t616,
                t136,
                t137,
                t138,
                t139,
                False,
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
        del t139, t138, t137, t136

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t623 = paddle._C_ops.relu(t617)
        del t617

        # pd_op.conv2d: (4x1024x32x32xf32) <- (4x256x32x32xf32, 1024x256x1x1xf32)
        t624 = paddle._C_ops.conv2d(
            t623, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t625, t626, t627, t628, t629, t630 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t624,
                t141,
                t142,
                t143,
                t144,
                False,
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
        del t144, t143, t142, t141

        # pd_op.pool2d: (4x512x32x32xf32) <- (4x512x64x64xf32, 2xi64)
        t631 = paddle._C_ops.pool2d(
            t607,
            t520,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.conv2d: (4x1024x32x32xf32) <- (4x512x32x32xf32, 1024x512x1x1xf32)
        t632 = paddle._C_ops.conv2d(
            t631, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145

        # pd_op.batch_norm_: (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t633, t634, t635, t636, t637, t638 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t632,
                t146,
                t147,
                t148,
                t149,
                False,
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
        del t149, t148, t147, t146

        # pd_op.add: (4x1024x32x32xf32) <- (4x1024x32x32xf32, 4x1024x32x32xf32)
        t639 = paddle._C_ops.add(t625, t633)

        # pd_op.relu: (4x1024x32x32xf32) <- (4x1024x32x32xf32)
        t640 = paddle._C_ops.relu(t639)
        del t639

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x1024x32x32xf32, 256x1024x1x1xf32)
        t641 = paddle._C_ops.conv2d(
            t640, t150, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t642, t643, t644, t645, t646, t647 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t641,
                t151,
                t152,
                t153,
                t154,
                False,
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
        del t154, t153, t152, t151

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t648 = paddle._C_ops.relu(t642)
        del t642

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x256x32x32xf32, 256x256x3x3xf32)
        t649 = paddle._C_ops.conv2d(
            t648, t155, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t650, t651, t652, t653, t654, t655 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t649,
                t156,
                t157,
                t158,
                t159,
                False,
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
        del t159, t158, t157, t156

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t656 = paddle._C_ops.relu(t650)
        del t650

        # pd_op.conv2d: (4x1024x32x32xf32) <- (4x256x32x32xf32, 1024x256x1x1xf32)
        t657 = paddle._C_ops.conv2d(
            t656, t160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t658, t659, t660, t661, t662, t663 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t657,
                t161,
                t162,
                t163,
                t164,
                False,
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
        del t164, t163, t162, t161

        # pd_op.add: (4x1024x32x32xf32) <- (4x1024x32x32xf32, 4x1024x32x32xf32)
        t664 = paddle._C_ops.add(t658, t640)

        # pd_op.relu: (4x1024x32x32xf32) <- (4x1024x32x32xf32)
        t665 = paddle._C_ops.relu(t664)
        del t664

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x1024x32x32xf32, 256x1024x1x1xf32)
        t666 = paddle._C_ops.conv2d(
            t665, t165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t667, t668, t669, t670, t671, t672 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t666,
                t166,
                t167,
                t168,
                t169,
                False,
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
        del t169, t168, t167, t166

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t673 = paddle._C_ops.relu(t667)
        del t667

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x256x32x32xf32, 256x256x3x3xf32)
        t674 = paddle._C_ops.conv2d(
            t673, t170, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t675, t676, t677, t678, t679, t680 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t674,
                t171,
                t172,
                t173,
                t174,
                False,
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
        del t174, t173, t172, t171

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t681 = paddle._C_ops.relu(t675)
        del t675

        # pd_op.conv2d: (4x1024x32x32xf32) <- (4x256x32x32xf32, 1024x256x1x1xf32)
        t682 = paddle._C_ops.conv2d(
            t681, t175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t683, t684, t685, t686, t687, t688 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t682,
                t176,
                t177,
                t178,
                t179,
                False,
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
        del t179, t178, t177, t176

        # pd_op.add: (4x1024x32x32xf32) <- (4x1024x32x32xf32, 4x1024x32x32xf32)
        t689 = paddle._C_ops.add(t683, t665)

        # pd_op.relu: (4x1024x32x32xf32) <- (4x1024x32x32xf32)
        t690 = paddle._C_ops.relu(t689)
        del t689

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x1024x32x32xf32, 256x1024x1x1xf32)
        t691 = paddle._C_ops.conv2d(
            t690, t180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t692, t693, t694, t695, t696, t697 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t691,
                t181,
                t182,
                t183,
                t184,
                False,
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
        del t184, t183, t182, t181

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t698 = paddle._C_ops.relu(t692)
        del t692

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x256x32x32xf32, 256x256x3x3xf32)
        t699 = paddle._C_ops.conv2d(
            t698, t185, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t700, t701, t702, t703, t704, t705 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t699,
                t186,
                t187,
                t188,
                t189,
                False,
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
        del t189, t188, t187, t186

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t706 = paddle._C_ops.relu(t700)
        del t700

        # pd_op.conv2d: (4x1024x32x32xf32) <- (4x256x32x32xf32, 1024x256x1x1xf32)
        t707 = paddle._C_ops.conv2d(
            t706, t190, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t708, t709, t710, t711, t712, t713 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t707,
                t191,
                t192,
                t193,
                t194,
                False,
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
        del t194, t193, t192, t191

        # pd_op.add: (4x1024x32x32xf32) <- (4x1024x32x32xf32, 4x1024x32x32xf32)
        t714 = paddle._C_ops.add(t708, t690)

        # pd_op.relu: (4x1024x32x32xf32) <- (4x1024x32x32xf32)
        t715 = paddle._C_ops.relu(t714)
        del t714

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x1024x32x32xf32, 256x1024x1x1xf32)
        t716 = paddle._C_ops.conv2d(
            t715, t195, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t717, t718, t719, t720, t721, t722 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t716,
                t196,
                t197,
                t198,
                t199,
                False,
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
        del t199, t198, t197, t196

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t723 = paddle._C_ops.relu(t717)
        del t717

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x256x32x32xf32, 256x256x3x3xf32)
        t724 = paddle._C_ops.conv2d(
            t723, t200, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t725, t726, t727, t728, t729, t730 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t724,
                t201,
                t202,
                t203,
                t204,
                False,
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
        del t204, t203, t202, t201

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t731 = paddle._C_ops.relu(t725)
        del t725

        # pd_op.conv2d: (4x1024x32x32xf32) <- (4x256x32x32xf32, 1024x256x1x1xf32)
        t732 = paddle._C_ops.conv2d(
            t731, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t733, t734, t735, t736, t737, t738 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t732,
                t206,
                t207,
                t208,
                t209,
                False,
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
        del t209, t208, t207, t206

        # pd_op.add: (4x1024x32x32xf32) <- (4x1024x32x32xf32, 4x1024x32x32xf32)
        t739 = paddle._C_ops.add(t733, t715)

        # pd_op.relu: (4x1024x32x32xf32) <- (4x1024x32x32xf32)
        t740 = paddle._C_ops.relu(t739)
        del t739

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x1024x32x32xf32, 256x1024x1x1xf32)
        t741 = paddle._C_ops.conv2d(
            t740, t210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t742, t743, t744, t745, t746, t747 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t741,
                t211,
                t212,
                t213,
                t214,
                False,
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
        del t214, t213, t212, t211

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t748 = paddle._C_ops.relu(t742)
        del t742

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x256x32x32xf32, 256x256x3x3xf32)
        t749 = paddle._C_ops.conv2d(
            t748, t215, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t750, t751, t752, t753, t754, t755 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t749,
                t216,
                t217,
                t218,
                t219,
                False,
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
        del t219, t218, t217, t216

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t756 = paddle._C_ops.relu(t750)
        del t750

        # pd_op.conv2d: (4x1024x32x32xf32) <- (4x256x32x32xf32, 1024x256x1x1xf32)
        t757 = paddle._C_ops.conv2d(
            t756, t220, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t220

        # pd_op.batch_norm_: (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t758, t759, t760, t761, t762, t763 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t757,
                t221,
                t222,
                t223,
                t224,
                False,
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
        del t224, t223, t222, t221

        # pd_op.add: (4x1024x32x32xf32) <- (4x1024x32x32xf32, 4x1024x32x32xf32)
        t764 = paddle._C_ops.add(t758, t740)

        # pd_op.relu: (4x1024x32x32xf32) <- (4x1024x32x32xf32)
        t765 = paddle._C_ops.relu(t764)
        del t764

        # pd_op.conv2d: (4x512x32x32xf32) <- (4x1024x32x32xf32, 512x1024x1x1xf32)
        t766 = paddle._C_ops.conv2d(
            t765, t225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225

        # pd_op.batch_norm_: (4x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t767, t768, t769, t770, t771, t772 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t766,
                t226,
                t227,
                t228,
                t229,
                False,
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
        del t229, t228, t227, t226

        # pd_op.relu: (4x512x32x32xf32) <- (4x512x32x32xf32)
        t773 = paddle._C_ops.relu(t767)
        del t767

        # pd_op.conv2d: (4x27x16x16xf32) <- (4x512x32x32xf32, 27x512x3x3xf32)
        t774 = paddle._C_ops.conv2d(
            t773, t230, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t230

        # pd_op.full_int_array: (4xi64) <- ()
        t775 = [1, -1, 1, 1]

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        t776 = paddle._C_ops.reshape(t231, t775)
        del t231

        # pd_op.add: (4x27x16x16xf32) <- (4x27x16x16xf32, 1x27x1x1xf32)
        t777 = paddle._C_ops.add(t774, t776)

        # pd_op.full_int_array: (2xi64) <- ()
        t778 = [18, 9]

        # pd_op.full: (1xi32) <- ()
        t779 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t780 = t779

        # pd_op.assign: (1xi32) <- (1xi32)
        t781 = t779

        # pd_op.split: ([4x18x16x16xf32, 4x9x16x16xf32]) <- (4x27x16x16xf32, 2xi64, 1xi32)
        t782 = paddle._C_ops.split(t777, t778, t779)
        del t777

        # builtin.split: (4x18x16x16xf32, 4x9x16x16xf32) <- ([4x18x16x16xf32, 4x9x16x16xf32])
        (
            t783,
            t784,
        ) = t782
        del t782

        # pd_op.sigmoid: (4x9x16x16xf32) <- (4x9x16x16xf32)
        t785 = paddle._C_ops.sigmoid(t784)
        del t784

        # pd_op.deformable_conv: (4x512x16x16xf32) <- (4x512x32x32xf32, 4x18x16x16xf32, 512x512x3x3xf32, 4x9x16x16xf32)
        t786 = paddle._C_ops.deformable_conv(
            t773, t783, input1, t785, [2, 2], [1, 1], [1, 1], 1, 1, 1
        )
        del input1

        # pd_op.batch_norm_: (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t787, t788, t789, t790, t791, t792 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t786,
                t232,
                t233,
                t234,
                t235,
                False,
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
        del t235, t234, t233, t232

        # pd_op.relu: (4x512x16x16xf32) <- (4x512x16x16xf32)
        t793 = paddle._C_ops.relu(t787)
        del t787

        # pd_op.conv2d: (4x2048x16x16xf32) <- (4x512x16x16xf32, 2048x512x1x1xf32)
        t794 = paddle._C_ops.conv2d(
            t793, t236, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t236

        # pd_op.batch_norm_: (4x2048x16x16xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (4x2048x16x16xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t795, t796, t797, t798, t799, t800 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t794,
                t237,
                t238,
                t239,
                t240,
                False,
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
        del t240, t239, t238, t237

        # pd_op.pool2d: (4x1024x16x16xf32) <- (4x1024x32x32xf32, 2xi64)
        t801 = paddle._C_ops.pool2d(
            t765,
            t520,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.conv2d: (4x2048x16x16xf32) <- (4x1024x16x16xf32, 2048x1024x1x1xf32)
        t802 = paddle._C_ops.conv2d(
            t801, t241, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t241

        # pd_op.batch_norm_: (4x2048x16x16xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (4x2048x16x16xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t803, t804, t805, t806, t807, t808 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t802,
                t242,
                t243,
                t244,
                t245,
                False,
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
        del t245, t244, t243, t242

        # pd_op.add: (4x2048x16x16xf32) <- (4x2048x16x16xf32, 4x2048x16x16xf32)
        t809 = paddle._C_ops.add(t795, t803)

        # pd_op.relu: (4x2048x16x16xf32) <- (4x2048x16x16xf32)
        t810 = paddle._C_ops.relu(t809)
        del t809

        # pd_op.conv2d: (4x512x16x16xf32) <- (4x2048x16x16xf32, 512x2048x1x1xf32)
        t811 = paddle._C_ops.conv2d(
            t810, t246, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t246

        # pd_op.batch_norm_: (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t812, t813, t814, t815, t816, t817 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t811,
                t247,
                t248,
                t249,
                t250,
                False,
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
        del t250, t249, t248, t247

        # pd_op.relu: (4x512x16x16xf32) <- (4x512x16x16xf32)
        t818 = paddle._C_ops.relu(t812)
        del t812

        # pd_op.conv2d: (4x27x16x16xf32) <- (4x512x16x16xf32, 27x512x3x3xf32)
        t819 = paddle._C_ops.conv2d(
            t818, t251, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t251

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        t820 = paddle._C_ops.reshape(t252, t775)
        del t252

        # pd_op.add: (4x27x16x16xf32) <- (4x27x16x16xf32, 1x27x1x1xf32)
        t821 = paddle._C_ops.add(t819, t820)

        # pd_op.split: ([4x18x16x16xf32, 4x9x16x16xf32]) <- (4x27x16x16xf32, 2xi64, 1xi32)
        t822 = paddle._C_ops.split(t821, t778, t779)
        del t821

        # builtin.split: (4x18x16x16xf32, 4x9x16x16xf32) <- ([4x18x16x16xf32, 4x9x16x16xf32])
        (
            t823,
            t824,
        ) = t822
        del t822

        # pd_op.sigmoid: (4x9x16x16xf32) <- (4x9x16x16xf32)
        t825 = paddle._C_ops.sigmoid(t824)
        del t824

        # pd_op.deformable_conv: (4x512x16x16xf32) <- (4x512x16x16xf32, 4x18x16x16xf32, 512x512x3x3xf32, 4x9x16x16xf32)
        t826 = paddle._C_ops.deformable_conv(
            t818, t823, input2, t825, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )
        del input2

        # pd_op.batch_norm_: (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t827, t828, t829, t830, t831, t832 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t826,
                t253,
                t254,
                t255,
                t256,
                False,
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
        del t256, t255, t254, t253

        # pd_op.relu: (4x512x16x16xf32) <- (4x512x16x16xf32)
        t833 = paddle._C_ops.relu(t827)
        del t827

        # pd_op.conv2d: (4x2048x16x16xf32) <- (4x512x16x16xf32, 2048x512x1x1xf32)
        t834 = paddle._C_ops.conv2d(
            t833, t257, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t257

        # pd_op.batch_norm_: (4x2048x16x16xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (4x2048x16x16xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t835, t836, t837, t838, t839, t840 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t834,
                t258,
                t259,
                t260,
                t261,
                False,
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
        del t261, t260, t259, t258

        # pd_op.add: (4x2048x16x16xf32) <- (4x2048x16x16xf32, 4x2048x16x16xf32)
        t841 = paddle._C_ops.add(t835, t810)

        # pd_op.relu: (4x2048x16x16xf32) <- (4x2048x16x16xf32)
        t842 = paddle._C_ops.relu(t841)
        del t841

        # pd_op.conv2d: (4x512x16x16xf32) <- (4x2048x16x16xf32, 512x2048x1x1xf32)
        t843 = paddle._C_ops.conv2d(
            t842, t262, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t262

        # pd_op.batch_norm_: (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t844, t845, t846, t847, t848, t849 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t843,
                t263,
                t264,
                t265,
                t266,
                False,
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
        del t266, t265, t264, t263

        # pd_op.relu: (4x512x16x16xf32) <- (4x512x16x16xf32)
        t850 = paddle._C_ops.relu(t844)
        del t844

        # pd_op.conv2d: (4x27x16x16xf32) <- (4x512x16x16xf32, 27x512x3x3xf32)
        t851 = paddle._C_ops.conv2d(
            t850, t267, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t267

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        t852 = paddle._C_ops.reshape(t268, t775)
        del t268

        # pd_op.add: (4x27x16x16xf32) <- (4x27x16x16xf32, 1x27x1x1xf32)
        t853 = paddle._C_ops.add(t851, t852)

        # pd_op.split: ([4x18x16x16xf32, 4x9x16x16xf32]) <- (4x27x16x16xf32, 2xi64, 1xi32)
        t854 = paddle._C_ops.split(t853, t778, t779)
        del t853, t778

        # builtin.split: (4x18x16x16xf32, 4x9x16x16xf32) <- ([4x18x16x16xf32, 4x9x16x16xf32])
        (
            t855,
            t856,
        ) = t854
        del t854

        # pd_op.sigmoid: (4x9x16x16xf32) <- (4x9x16x16xf32)
        t857 = paddle._C_ops.sigmoid(t856)
        del t856

        # pd_op.deformable_conv: (4x512x16x16xf32) <- (4x512x16x16xf32, 4x18x16x16xf32, 512x512x3x3xf32, 4x9x16x16xf32)
        t858 = paddle._C_ops.deformable_conv(
            t850, t855, input3, t857, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )
        del input3

        # pd_op.batch_norm_: (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t859, t860, t861, t862, t863, t864 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t858,
                t269,
                t270,
                t271,
                t272,
                False,
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
        del t272, t271, t270, t269

        # pd_op.relu: (4x512x16x16xf32) <- (4x512x16x16xf32)
        t865 = paddle._C_ops.relu(t859)
        del t859

        # pd_op.conv2d: (4x2048x16x16xf32) <- (4x512x16x16xf32, 2048x512x1x1xf32)
        t866 = paddle._C_ops.conv2d(
            t865, t273, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t273

        # pd_op.batch_norm_: (4x2048x16x16xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (4x2048x16x16xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t867, t868, t869, t870, t871, t872 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t866,
                t274,
                t275,
                t276,
                t277,
                False,
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
        del t277, t276, t275, t274

        # pd_op.add: (4x2048x16x16xf32) <- (4x2048x16x16xf32, 4x2048x16x16xf32)
        t873 = paddle._C_ops.add(t867, t842)

        # pd_op.relu: (4x2048x16x16xf32) <- (4x2048x16x16xf32)
        t874 = paddle._C_ops.relu(t873)
        del t873

        # pd_op.conv2d: (4x1024x16x16xf32) <- (4x2048x16x16xf32, 1024x2048x3x3xf32)
        t875 = paddle._C_ops.conv2d(
            t874, t278, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t278

        # pd_op.batch_norm_: (4x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t876, t877, t878, t879, t880, t881 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t875,
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

        # pd_op.relu: (4x1024x16x16xf32) <- (4x1024x16x16xf32)
        t882 = paddle._C_ops.relu(t876)
        del t876

        # pd_op.full_int_array: (0xi64) <- ()
        t883 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        t884 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t885 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t886 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t887 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t888 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t889 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t890 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t891 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t892 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t893 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t894 = t883

        # pd_op.assign: (0xi64) <- (0xi64)
        t895 = t883

        # pd_op.depthwise_conv2d_transpose: (4x1024x32x32xf32) <- (4x1024x16x16xf32, 1024x1x4x4xf32, 0xi64)
        t896 = paddle._C_ops.depthwise_conv2d_transpose(
            t882, t283, [2, 2], [1, 1], [], t883, "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del t283

        # pd_op.add: (4x1024x32x32xf32) <- (4x1024x32x32xf32, 4x1024x32x32xf32)
        t897 = paddle._C_ops.add(t896, t765)

        # pd_op.conv2d: (4x1024x32x32xf32) <- (4x1024x32x32xf32, 1024x1024x3x3xf32)
        t898 = paddle._C_ops.conv2d(
            t897, t284, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t284

        # pd_op.batch_norm_: (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x32x32xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t899, t900, t901, t902, t903, t904 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t898,
                t285,
                t286,
                t287,
                t288,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t288, t287, t286, t285

        # pd_op.relu: (4x1024x32x32xf32) <- (4x1024x32x32xf32)
        t905 = paddle._C_ops.relu(t899)
        del t899

        # pd_op.conv2d: (4x512x32x32xf32) <- (4x1024x32x32xf32, 512x1024x3x3xf32)
        t906 = paddle._C_ops.conv2d(
            t765, t289, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t289

        # pd_op.batch_norm_: (4x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t907, t908, t909, t910, t911, t912 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t906,
                t290,
                t291,
                t292,
                t293,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t293, t292, t291, t290

        # pd_op.relu: (4x512x32x32xf32) <- (4x512x32x32xf32)
        t913 = paddle._C_ops.relu(t907)
        del t907

        # pd_op.depthwise_conv2d_transpose: (4x512x64x64xf32) <- (4x512x32x32xf32, 512x1x4x4xf32, 0xi64)
        t914 = paddle._C_ops.depthwise_conv2d_transpose(
            t913, t294, [2, 2], [1, 1], [], t883, "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t294

        # pd_op.add: (4x512x64x64xf32) <- (4x512x64x64xf32, 4x512x64x64xf32)
        t915 = paddle._C_ops.add(t914, t607)

        # pd_op.conv2d: (4x512x64x64xf32) <- (4x512x64x64xf32, 512x512x3x3xf32)
        t916 = paddle._C_ops.conv2d(
            t915, t295, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t295

        # pd_op.batch_norm_: (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t917, t918, t919, t920, t921, t922 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t916,
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

        # pd_op.relu: (4x512x64x64xf32) <- (4x512x64x64xf32)
        t923 = paddle._C_ops.relu(t917)
        del t917

        # pd_op.conv2d: (4x512x32x32xf32) <- (4x1024x32x32xf32, 512x1024x3x3xf32)
        t924 = paddle._C_ops.conv2d(
            t905, t300, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t300

        # pd_op.batch_norm_: (4x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t925, t926, t927, t928, t929, t930 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t924,
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

        # pd_op.relu: (4x512x32x32xf32) <- (4x512x32x32xf32)
        t931 = paddle._C_ops.relu(t925)
        del t925

        # pd_op.depthwise_conv2d_transpose: (4x512x64x64xf32) <- (4x512x32x32xf32, 512x1x4x4xf32, 0xi64)
        t932 = paddle._C_ops.depthwise_conv2d_transpose(
            t931, t305, [2, 2], [1, 1], [], t883, "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t305

        # pd_op.add: (4x512x64x64xf32) <- (4x512x64x64xf32, 4x512x64x64xf32)
        t933 = paddle._C_ops.add(t932, t923)

        # pd_op.conv2d: (4x512x64x64xf32) <- (4x512x64x64xf32, 512x512x3x3xf32)
        t934 = paddle._C_ops.conv2d(
            t933, t306, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t306

        # pd_op.batch_norm_: (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x64x64xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t935, t936, t937, t938, t939, t940 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t934,
                t307,
                t308,
                t309,
                t310,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t310, t309, t308, t307

        # pd_op.relu: (4x512x64x64xf32) <- (4x512x64x64xf32)
        t941 = paddle._C_ops.relu(t935)
        del t935

        # pd_op.conv2d: (4x256x64x64xf32) <- (4x512x64x64xf32, 256x512x3x3xf32)
        t942 = paddle._C_ops.conv2d(
            t607, t311, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t311

        # pd_op.batch_norm_: (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t943, t944, t945, t946, t947, t948 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t942,
                t312,
                t313,
                t314,
                t315,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t315, t314, t313, t312

        # pd_op.relu: (4x256x64x64xf32) <- (4x256x64x64xf32)
        t949 = paddle._C_ops.relu(t943)
        del t943

        # pd_op.depthwise_conv2d_transpose: (4x256x128x128xf32) <- (4x256x64x64xf32, 256x1x4x4xf32, 0xi64)
        t950 = paddle._C_ops.depthwise_conv2d_transpose(
            t949, t316, [2, 2], [1, 1], [], t883, "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t316

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 4x256x128x128xf32)
        t951 = paddle._C_ops.add(t950, t496)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x256x128x128xf32, 256x256x3x3xf32)
        t952 = paddle._C_ops.conv2d(
            t951, t317, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t317

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t953, t954, t955, t956, t957, t958 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t952,
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

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t959 = paddle._C_ops.relu(t953)
        del t953

        # pd_op.conv2d: (4x256x64x64xf32) <- (4x512x64x64xf32, 256x512x3x3xf32)
        t960 = paddle._C_ops.conv2d(
            t923, t322, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t322

        # pd_op.batch_norm_: (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t961, t962, t963, t964, t965, t966 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t960,
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

        # pd_op.relu: (4x256x64x64xf32) <- (4x256x64x64xf32)
        t967 = paddle._C_ops.relu(t961)
        del t961

        # pd_op.depthwise_conv2d_transpose: (4x256x128x128xf32) <- (4x256x64x64xf32, 256x1x4x4xf32, 0xi64)
        t968 = paddle._C_ops.depthwise_conv2d_transpose(
            t967, t327, [2, 2], [1, 1], [], t883, "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t327

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 4x256x128x128xf32)
        t969 = paddle._C_ops.add(t968, t959)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x256x128x128xf32, 256x256x3x3xf32)
        t970 = paddle._C_ops.conv2d(
            t969, t328, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t328

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t971, t972, t973, t974, t975, t976 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t970,
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

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t977 = paddle._C_ops.relu(t971)
        del t971

        # pd_op.conv2d: (4x256x64x64xf32) <- (4x512x64x64xf32, 256x512x3x3xf32)
        t978 = paddle._C_ops.conv2d(
            t941, t333, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t333

        # pd_op.batch_norm_: (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t979, t980, t981, t982, t983, t984 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t978,
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

        # pd_op.relu: (4x256x64x64xf32) <- (4x256x64x64xf32)
        t985 = paddle._C_ops.relu(t979)
        del t979

        # pd_op.depthwise_conv2d_transpose: (4x256x128x128xf32) <- (4x256x64x64xf32, 256x1x4x4xf32, 0xi64)
        t986 = paddle._C_ops.depthwise_conv2d_transpose(
            t985, t338, [2, 2], [1, 1], [], t883, "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t338

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 4x256x128x128xf32)
        t987 = paddle._C_ops.add(t986, t977)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x256x128x128xf32, 256x256x3x3xf32)
        t988 = paddle._C_ops.conv2d(
            t987, t339, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t339

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t989, t990, t991, t992, t993, t994 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t988,
                t340,
                t341,
                t342,
                t343,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t343, t342, t341, t340

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t995 = paddle._C_ops.relu(t989)
        del t989

        # pd_op.assign: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t996 = t995

        # pd_op.assign: (4x512x64x64xf32) <- (4x512x64x64xf32)
        t997 = t941

        # pd_op.assign: (4x1024x32x32xf32) <- (4x1024x32x32xf32)
        t998 = t905

        # pd_op.assign: (4x2048x16x16xf32) <- (4x2048x16x16xf32)
        t999 = t874

        # pd_op.conv2d: (4x256x64x64xf32) <- (4x512x64x64xf32, 256x512x3x3xf32)
        t1000 = paddle._C_ops.conv2d(
            t997, t344, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t344

        # pd_op.batch_norm_: (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1001, t1002, t1003, t1004, t1005, t1006 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1000,
                t345,
                t346,
                t347,
                t348,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t348, t347, t346, t345

        # pd_op.relu: (4x256x64x64xf32) <- (4x256x64x64xf32)
        t1007 = paddle._C_ops.relu(t1001)
        del t1001

        # pd_op.depthwise_conv2d_transpose: (4x256x128x128xf32) <- (4x256x64x64xf32, 256x1x4x4xf32, 0xi64)
        t1008 = paddle._C_ops.depthwise_conv2d_transpose(
            t1007, t349, [2, 2], [1, 1], [], t883, "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t349

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 4x256x128x128xf32)
        t1009 = paddle._C_ops.add(t1008, t996)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x256x128x128xf32, 256x256x3x3xf32)
        t1010 = paddle._C_ops.conv2d(
            t1009, t350, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t350

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1011, t1012, t1013, t1014, t1015, t1016 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1010,
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

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t1017 = paddle._C_ops.relu(t1011)
        del t1011

        # pd_op.conv2d: (4x256x32x32xf32) <- (4x1024x32x32xf32, 256x1024x3x3xf32)
        t1018 = paddle._C_ops.conv2d(
            t998, t355, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t355

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1019, t1020, t1021, t1022, t1023, t1024 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1018,
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

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        t1025 = paddle._C_ops.relu(t1019)
        del t1019

        # pd_op.depthwise_conv2d_transpose: (4x256x128x128xf32) <- (4x256x32x32xf32, 256x1x8x8xf32, 0xi64)
        t1026 = paddle._C_ops.depthwise_conv2d_transpose(
            t1025, t360, [4, 4], [2, 2], [], t883, "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t360

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 4x256x128x128xf32)
        t1027 = paddle._C_ops.add(t1026, t1017)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x256x128x128xf32, 256x256x3x3xf32)
        t1028 = paddle._C_ops.conv2d(
            t1027, t361, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t361

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1029, t1030, t1031, t1032, t1033, t1034 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1028,
                t362,
                t363,
                t364,
                t365,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t365, t364, t363, t362

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t1035 = paddle._C_ops.relu(t1029)
        del t1029

        # pd_op.conv2d: (4x256x16x16xf32) <- (4x2048x16x16xf32, 256x2048x3x3xf32)
        t1036 = paddle._C_ops.conv2d(
            t999, t366, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t366

        # pd_op.batch_norm_: (4x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1037, t1038, t1039, t1040, t1041, t1042 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1036,
                t367,
                t368,
                t369,
                t370,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t370, t369, t368, t367

        # pd_op.relu: (4x256x16x16xf32) <- (4x256x16x16xf32)
        t1043 = paddle._C_ops.relu(t1037)
        del t1037

        # pd_op.depthwise_conv2d_transpose: (4x256x128x128xf32) <- (4x256x16x16xf32, 256x1x16x16xf32, 0xi64)
        t1044 = paddle._C_ops.depthwise_conv2d_transpose(
            t1043, t371, [8, 8], [4, 4], [], t883, "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t371

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 4x256x128x128xf32)
        t1045 = paddle._C_ops.add(t1044, t1035)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x256x128x128xf32, 256x256x3x3xf32)
        t1046 = paddle._C_ops.conv2d(
            t1045, t372, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t372

        # pd_op.batch_norm_: (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x128x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1047, t1048, t1049, t1050, t1051, t1052 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1046,
                t373,
                t374,
                t375,
                t376,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t376, t375, t374, t373

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t1053 = paddle._C_ops.relu(t1047)
        del t1047

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x256x128x128xf32, 256x256x3x3xf32)
        t1054 = paddle._C_ops.conv2d(
            t1053, t377, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t377

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1055 = paddle._C_ops.reshape(t378, t775)
        del t378

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 1x256x1x1xf32)
        t1056 = paddle._C_ops.add(t1054, t1055)

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t1057 = paddle._C_ops.relu(t1056)
        del t1056

        # pd_op.conv2d: (4x4x128x128xf32) <- (4x256x128x128xf32, 4x256x1x1xf32)
        t1058 = paddle._C_ops.conv2d(
            t1057, t379, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t379

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        t1059 = paddle._C_ops.reshape(t380, t775)
        del t380

        # pd_op.add: (4x4x128x128xf32) <- (4x4x128x128xf32, 1x4x1x1xf32)
        t1060 = paddle._C_ops.add(t1058, t1059)

        # pd_op.sigmoid: (4x4x128x128xf32) <- (4x4x128x128xf32)
        t1061 = paddle._C_ops.sigmoid(t1060)
        del t1060

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x256x128x128xf32, 256x256x3x3xf32)
        t1062 = paddle._C_ops.conv2d(
            t1053, t381, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t381

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1063 = paddle._C_ops.reshape(t382, t775)
        del t382

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 1x256x1x1xf32)
        t1064 = paddle._C_ops.add(t1062, t1063)

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t1065 = paddle._C_ops.relu(t1064)
        del t1064

        # pd_op.conv2d: (4x2x128x128xf32) <- (4x256x128x128xf32, 2x256x1x1xf32)
        t1066 = paddle._C_ops.conv2d(
            t1065, t383, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t383

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        t1067 = paddle._C_ops.reshape(t384, t775)
        del t384

        # pd_op.add: (4x2x128x128xf32) <- (4x2x128x128xf32, 1x2x1x1xf32)
        t1068 = paddle._C_ops.add(t1066, t1067)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x256x128x128xf32, 256x256x3x3xf32)
        t1069 = paddle._C_ops.conv2d(
            t1053, t385, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t385

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t1070 = paddle._C_ops.reshape(t386, t775)
        del t386

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 1x256x1x1xf32)
        t1071 = paddle._C_ops.add(t1069, t1070)

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        t1072 = paddle._C_ops.relu(t1071)
        del t1071

        # pd_op.conv2d: (4x2x128x128xf32) <- (4x256x128x128xf32, 2x256x1x1xf32)
        t1073 = paddle._C_ops.conv2d(
            t1072, t387, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t387

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        t1074 = paddle._C_ops.reshape(t388, t775)
        del t775, t388

        # pd_op.add: (4x2x128x128xf32) <- (4x2x128x128xf32, 1x2x1x1xf32)
        t1075 = paddle._C_ops.add(t1073, t1074)

        # pd_op.full: (1xf32) <- ()
        t1076 = paddle._C_ops.full(
            [1], float("0.0001"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t1077 = paddle._C_ops.full(
            [1], float("0.9999"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32, 1xf32)
        t1078 = paddle._C_ops.clip(t1061, t1076, t1077)

        # pd_op.full: (xf32) <- ()
        t1079 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (4x4x128x128xb) <- (4x4x128x128xf32, xf32)
        t1080 = paddle._C_ops.equal(input4, t1079)

        # pd_op.cast: (4x4x128x128xf32) <- (4x4x128x128xb)
        t1081 = paddle._C_ops.cast(t1080, paddle.float32)
        del t1080

        # pd_op.less_than: (4x4x128x128xb) <- (4x4x128x128xf32, xf32)
        t1082 = paddle._C_ops.less_than(input4, t1079)
        del t1079

        # pd_op.cast: (4x4x128x128xf32) <- (4x4x128x128xb)
        t1083 = paddle._C_ops.cast(t1082, paddle.float32)
        del t1082

        # pd_op.full: (1xf32) <- ()
        t1084 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t1085 = t1084

        # pd_op.assign: (1xf32) <- (1xf32)
        t1086 = t1084

        # pd_op.assign: (1xf32) <- (1xf32)
        t1087 = t1084

        # pd_op.assign: (1xf32) <- (1xf32)
        t1088 = t1084

        # pd_op.scale: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32)
        t1089 = paddle._C_ops.scale(input4, t1084, float("1"), True)
        del input4

        # pd_op.pow: (4x4x128x128xf32) <- (4x4x128x128xf32)
        t1090 = paddle._C_ops.pow(t1089, float("4"))
        del t1089

        # pd_op.log: (4x4x128x128xf32) <- (4x4x128x128xf32)
        t1091 = paddle._C_ops.log(t1078)

        # pd_op.scale: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32)
        t1092 = paddle._C_ops.scale(t1078, t1084, float("1"), True)

        # pd_op.assign: (4x4x128x128xf32) <- (4x4x128x128xf32)
        t1093 = t1092

        # pd_op.pow: (4x4x128x128xf32) <- (4x4x128x128xf32)
        t1094 = paddle._C_ops.pow(t1092, float("2"))

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        t1095 = paddle._C_ops.multiply(t1091, t1094)

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        t1096 = paddle._C_ops.multiply(t1095, t1081)

        # pd_op.scale: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32)
        t1097 = paddle._C_ops.scale(t1096, t1084, float("0"), True)
        del t1096

        # pd_op.log: (4x4x128x128xf32) <- (4x4x128x128xf32)
        t1098 = paddle._C_ops.log(t1092)

        # pd_op.pow: (4x4x128x128xf32) <- (4x4x128x128xf32)
        t1099 = paddle._C_ops.pow(t1078, float("2"))

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        t1100 = paddle._C_ops.multiply(t1098, t1099)

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        t1101 = paddle._C_ops.multiply(t1100, t1090)

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        t1102 = paddle._C_ops.multiply(t1101, t1083)

        # pd_op.scale: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32)
        t1103 = paddle._C_ops.scale(t1102, t1084, float("0"), True)
        del t1084, t1102

        # pd_op.sum: (xf32) <- (4x4x128x128xf32, 0xi64)
        t1104 = paddle._C_ops.sum(t1097, t883, None, False)

        # pd_op.sum: (xf32) <- (4x4x128x128xf32, 0xi64)
        t1105 = paddle._C_ops.sum(t1103, t883, None, False)

        # pd_op.sum: (xf32) <- (4x4x128x128xf32, 0xi64)
        t1106 = paddle._C_ops.sum(t1081, t883, None, False)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1107 = paddle._C_ops.add(t1104, t1105)

        # pd_op.full: (xf32) <- ()
        t1108 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xf32, xf32)
        t1109 = paddle._C_ops.equal(t1106, t1108)
        del t1108

        # pd_op.cast: (xf32) <- (xb)
        t1110 = paddle._C_ops.cast(t1109, paddle.float32)
        del t1109

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1111 = paddle._C_ops.add(t1106, t1110)
        del t1110, t1106

        # pd_op.divide: (xf32) <- (xf32, xf32)
        t1112 = paddle._C_ops.divide(t1107, t1111)

        # pd_op.full: (1xf32) <- ()
        t1113 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t1114 = t1113

        # pd_op.assign: (1xf32) <- (1xf32)
        t1115 = t1113

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t1116 = paddle._C_ops.scale(t1112, t1113, float("0"), True)

        # pd_op.transpose: (4x128x128x2xf32) <- (4x2x128x128xf32)
        t1117 = paddle._C_ops.transpose(t1068, [0, 2, 3, 1])
        del t1068

        # pd_op.full_int_array: (3xi64) <- ()
        t1118 = [4, -1, 2]

        # pd_op.reshape: (4x16384x2xf32) <- (4x128x128x2xf32, 3xi64)
        t1119 = paddle._C_ops.reshape(t1117, t1118)

        # pd_op.full_int_array: (1xi64) <- ()
        t1120 = [2]

        # pd_op.unsqueeze: (4x128x1xi64) <- (4x128xi64, 1xi64)
        t1121 = paddle._C_ops.unsqueeze(input5, t1120)
        del input5

        # pd_op.full: (1x128x1xi64) <- ()
        t1122 = paddle._C_ops.full(
            [1, 128, 1],
            float("0"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1x128x1xi64) <- ()
        t1123 = paddle._C_ops.full(
            [1, 128, 1],
            float("1"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1x128x1xi64) <- ()
        t1124 = paddle._C_ops.full(
            [1, 128, 1],
            float("2"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1x128x1xi64) <- ()
        t1125 = paddle._C_ops.full(
            [1, 128, 1],
            float("3"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        t1126 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x128x1xi64, 1x128x1xi64, 1x128x1xi64, 1x128x1xi64]) <- (1x128x1xi64, 1x128x1xi64, 1x128x1xi64, 1x128x1xi64)
        t1127 = [t1122, t1123, t1124, t1125]
        del t1124, t1125, t1122, t1123

        # pd_op.concat: (4x128x1xi64) <- ([1x128x1xi64, 1x128x1xi64, 1x128x1xi64, 1x128x1xi64], 1xi32)
        t1128 = paddle._C_ops.concat(t1127, t1126)
        del t1127, t1126

        # pd_op.full: (1xi32) <- ()
        t1129 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([4x128x1xi64, 4x128x1xi64]) <- (4x128x1xi64, 4x128x1xi64)
        t1130 = [t1128, t1121]
        del t1128, t1121

        # pd_op.concat: (4x128x2xi64) <- ([4x128x1xi64, 4x128x1xi64], 1xi32)
        t1131 = paddle._C_ops.concat(t1130, t1129)
        del t1130, t1129

        # pd_op.gather_nd: (4x128x2xf32) <- (4x16384x2xf32, 4x128x2xi64)
        t1132 = paddle._C_ops.gather_nd(t1119, t1131)

        # pd_op.unsqueeze: (4x128x1xi32) <- (4x128xi32, 1xi64)
        t1133 = paddle._C_ops.unsqueeze(input6, t1120)
        del input6, t1120

        # pd_op.expand_as: (4x128x2xi32) <- (4x128x1xi32, 4x128x2xf32)
        t1134 = paddle._C_ops.expand_as(t1133, t1132, [4, 128, 2])

        # pd_op.cast: (4x128x2xf32) <- (4x128x2xi32)
        t1135 = paddle._C_ops.cast(t1134, paddle.float32)
        del t1134

        # pd_op.sum: (xf32) <- (4x128x2xf32, 0xi64)
        t1136 = paddle._C_ops.sum(t1135, t883, None, False)

        # pd_op.multiply: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        t1137 = paddle._C_ops.multiply(t1132, t1135)

        # pd_op.multiply: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        t1138 = paddle._C_ops.multiply(input7, t1135)
        del input7

        # pd_op.subtract: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        t1139 = paddle._C_ops.subtract(t1137, t1138)

        # pd_op.abs: (4x128x2xf32) <- (4x128x2xf32)
        t1140 = paddle._C_ops.abs(t1139)

        # pd_op.sum: (xf32) <- (4x128x2xf32, 0xi64)
        t1141 = paddle._C_ops.sum(t1140, t883, None, False)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t1142 = paddle._C_ops.scale(t1136, t1113, float("0.0001"), True)
        del t1136

        # pd_op.divide: (xf32) <- (xf32, xf32)
        t1143 = paddle._C_ops.divide(t1141, t1142)

        # pd_op.transpose: (4x128x128x2xf32) <- (4x2x128x128xf32)
        t1144 = paddle._C_ops.transpose(t1075, [0, 2, 3, 1])
        del t1075

        # pd_op.reshape: (4x16384x2xf32) <- (4x128x128x2xf32, 3xi64)
        t1145 = paddle._C_ops.reshape(t1144, t1118)
        del t1118

        # pd_op.gather_nd: (4x128x2xf32) <- (4x16384x2xf32, 4x128x2xi64)
        t1146 = paddle._C_ops.gather_nd(t1145, t1131)

        # pd_op.expand_as: (4x128x2xi32) <- (4x128x1xi32, 4x128x2xf32)
        t1147 = paddle._C_ops.expand_as(t1133, t1146, [4, 128, 2])
        del t1133

        # pd_op.cast: (4x128x2xf32) <- (4x128x2xi32)
        t1148 = paddle._C_ops.cast(t1147, paddle.float32)
        del t1147

        # pd_op.sum: (xf32) <- (4x128x2xf32, 0xi64)
        t1149 = paddle._C_ops.sum(t1148, t883, None, False)

        # pd_op.multiply: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        t1150 = paddle._C_ops.multiply(t1146, t1148)

        # pd_op.multiply: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        t1151 = paddle._C_ops.multiply(input8, t1148)
        del input8

        # pd_op.subtract: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        t1152 = paddle._C_ops.subtract(t1150, t1151)

        # pd_op.abs: (4x128x2xf32) <- (4x128x2xf32)
        t1153 = paddle._C_ops.abs(t1152)

        # pd_op.sum: (xf32) <- (4x128x2xf32, 0xi64)
        t1154 = paddle._C_ops.sum(t1153, t883, None, False)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t1155 = paddle._C_ops.scale(t1149, t1113, float("0.0001"), True)
        del t1149

        # pd_op.divide: (xf32) <- (xf32, xf32)
        t1156 = paddle._C_ops.divide(t1154, t1155)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t1157 = paddle._C_ops.scale(t1116, t1113, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        t1158 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t1159 = paddle._C_ops.scale(t1143, t1158, float("0"), True)

        # pd_op.add: (xf32) <- (xf32, xf32)
        t1160 = paddle._C_ops.add(t1157, t1159)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t1161 = paddle._C_ops.scale(t1156, t1113, float("0"), True)

        return (
            t389,
            t391,
            t392,
            t393,
            t394,
            t395,
            t396,
            t397,
            t399,
            t400,
            t401,
            t402,
            t403,
            t404,
            t405,
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
            t439,
            t440,
            t441,
            t442,
            t443,
            t444,
            t446,
            t447,
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
            t471,
            t472,
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
            t486,
            t487,
            t488,
            t489,
            t490,
            t491,
            t492,
            t493,
            t494,
            t496,
            t497,
            t499,
            t500,
            t501,
            t502,
            t503,
            t504,
            t505,
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
            t574,
            t575,
            t576,
            t577,
            t578,
            t579,
            t580,
            t582,
            t583,
            t585,
            t586,
            t587,
            t588,
            t589,
            t590,
            t591,
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
            t607,
            t608,
            t610,
            t611,
            t612,
            t613,
            t614,
            t615,
            t616,
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
            t640,
            t641,
            t643,
            t644,
            t645,
            t646,
            t647,
            t648,
            t649,
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
            t665,
            t666,
            t668,
            t669,
            t670,
            t671,
            t672,
            t673,
            t674,
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
            t715,
            t716,
            t718,
            t719,
            t720,
            t721,
            t722,
            t723,
            t724,
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
            t743,
            t744,
            t745,
            t746,
            t747,
            t748,
            t749,
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
            t768,
            t769,
            t770,
            t771,
            t772,
            t773,
            t774,
            t776,
            t779,
            t780,
            t781,
            t783,
            t785,
            t786,
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
            t810,
            t811,
            t813,
            t814,
            t815,
            t816,
            t817,
            t818,
            t819,
            t820,
            t823,
            t825,
            t826,
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
            t842,
            t843,
            t845,
            t846,
            t847,
            t848,
            t849,
            t850,
            t851,
            t852,
            t855,
            t857,
            t858,
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
            t874,
            t875,
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
            t900,
            t901,
            t902,
            t903,
            t904,
            t905,
            t906,
            t908,
            t909,
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
            t926,
            t927,
            t928,
            t929,
            t930,
            t931,
            t932,
            t933,
            t934,
            t936,
            t937,
            t938,
            t939,
            t940,
            t941,
            t942,
            t944,
            t945,
            t946,
            t947,
            t948,
            t949,
            t950,
            t951,
            t952,
            t954,
            t955,
            t956,
            t957,
            t958,
            t959,
            t960,
            t962,
            t963,
            t964,
            t965,
            t966,
            t967,
            t968,
            t969,
            t970,
            t972,
            t973,
            t974,
            t975,
            t976,
            t977,
            t978,
            t980,
            t981,
            t982,
            t983,
            t984,
            t985,
            t986,
            t987,
            t988,
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
            t1002,
            t1003,
            t1004,
            t1005,
            t1006,
            t1007,
            t1008,
            t1009,
            t1010,
            t1012,
            t1013,
            t1014,
            t1015,
            t1016,
            t1017,
            t1018,
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
            t1034,
            t1035,
            t1036,
            t1038,
            t1039,
            t1040,
            t1041,
            t1042,
            t1043,
            t1044,
            t1045,
            t1046,
            t1048,
            t1049,
            t1050,
            t1051,
            t1052,
            t1053,
            t1054,
            t1055,
            t1057,
            t1058,
            t1059,
            t1061,
            t1062,
            t1063,
            t1065,
            t1066,
            t1067,
            t1069,
            t1070,
            t1072,
            t1073,
            t1074,
            t1076,
            t1077,
            t1078,
            t1081,
            t1083,
            t1085,
            t1086,
            t1087,
            t1088,
            t1090,
            t1091,
            t1092,
            t1093,
            t1094,
            t1095,
            t1097,
            t1098,
            t1099,
            t1100,
            t1101,
            t1103,
            t1104,
            t1105,
            t1107,
            t1111,
            t1112,
            t1113,
            t1114,
            t1115,
            t1116,
            t1117,
            t1119,
            t1131,
            t1132,
            t1135,
            t1137,
            t1138,
            t1139,
            t1140,
            t1141,
            t1142,
            t1143,
            t1144,
            t1145,
            t1146,
            t1148,
            t1150,
            t1151,
            t1152,
            t1153,
            t1154,
            t1155,
            t1156,
            t1157,
            t1158,
            t1159,
            t1160,
            t1161,
        )
