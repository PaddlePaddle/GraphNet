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
    ):
        t367 = None
        # pd_op.conv2d: (4x64x320x320xf32) <- (4x3x640x640xf32, 64x3x3x3xf32)
        t368 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (4x64x320x320xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x320x320xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t369, t370, t371, t372, t373, t374 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t368,
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

        # pd_op.relu: (4x64x320x320xf32) <- (4x64x320x320xf32)
        t375 = paddle._C_ops.relu(t369)
        del t369

        # pd_op.conv2d: (4x64x320x320xf32) <- (4x64x320x320xf32, 64x64x3x3xf32)
        t376 = paddle._C_ops.conv2d(
            t375, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (4x64x320x320xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x320x320xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t377, t378, t379, t380, t381, t382 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t376,
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

        # pd_op.relu: (4x64x320x320xf32) <- (4x64x320x320xf32)
        t383 = paddle._C_ops.relu(t377)
        del t377

        # pd_op.conv2d: (4x128x320x320xf32) <- (4x64x320x320xf32, 128x64x3x3xf32)
        t384 = paddle._C_ops.conv2d(
            t383, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (4x128x320x320xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x320x320xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t385, t386, t387, t388, t389, t390 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t384,
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

        # pd_op.relu: (4x128x320x320xf32) <- (4x128x320x320xf32)
        t391 = paddle._C_ops.relu(t385)
        del t385

        # pd_op.full_int_array: (2xi64) <- ()
        t392 = [3, 3]

        # pd_op.pool2d: (4x128x160x160xf32) <- (4x128x320x320xf32, 2xi64)
        t393 = paddle._C_ops.pool2d(
            t391,
            t392,
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

        # pd_op.conv2d: (4x128x160x160xf32) <- (4x128x160x160xf32, 128x128x3x3xf32)
        t394 = paddle._C_ops.conv2d(
            t393, t15, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t395, t396, t397, t398, t399, t400 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t394,
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

        # pd_op.relu: (4x128x160x160xf32) <- (4x128x160x160xf32)
        t401 = paddle._C_ops.relu(t395)
        del t395

        # pd_op.conv2d: (4x128x160x160xf32) <- (4x128x160x160xf32, 128x128x3x3xf32)
        t402 = paddle._C_ops.conv2d(
            t401, t20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t403, t404, t405, t406, t407, t408 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t402,
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

        # pd_op.relu: (4x128x160x160xf32) <- (4x128x160x160xf32)
        t409 = paddle._C_ops.relu(t403)
        del t403

        # pd_op.conv2d: (4x128x160x160xf32) <- (4x128x160x160xf32, 128x128x3x3xf32)
        t410 = paddle._C_ops.conv2d(
            t409, t25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t411, t412, t413, t414, t415, t416 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t410,
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

        # pd_op.relu: (4x128x160x160xf32) <- (4x128x160x160xf32)
        t417 = paddle._C_ops.relu(t411)
        del t411

        # pd_op.conv2d: (4x128x160x160xf32) <- (4x128x160x160xf32, 128x128x3x3xf32)
        t418 = paddle._C_ops.conv2d(
            t417, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t419, t420, t421, t422, t423, t424 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t418,
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

        # pd_op.relu: (4x128x160x160xf32) <- (4x128x160x160xf32)
        t425 = paddle._C_ops.relu(t419)
        del t419

        # pd_op.conv2d: (4x128x160x160xf32) <- (4x128x160x160xf32, 128x128x3x3xf32)
        t426 = paddle._C_ops.conv2d(
            t425, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t427, t428, t429, t430, t431, t432 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t426,
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

        # pd_op.relu: (4x128x160x160xf32) <- (4x128x160x160xf32)
        t433 = paddle._C_ops.relu(t427)
        del t427

        # pd_op.conv2d: (4x128x160x160xf32) <- (4x128x160x160xf32, 128x128x3x3xf32)
        t434 = paddle._C_ops.conv2d(
            t433, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x160x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t435, t436, t437, t438, t439, t440 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t434,
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

        # pd_op.relu: (4x128x160x160xf32) <- (4x128x160x160xf32)
        t441 = paddle._C_ops.relu(t435)
        del t435

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

        # builtin.combine: ([4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32]) <- (4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32)
        t450 = [t393, t401, t409, t417, t425, t433, t441]

        # pd_op.concat: (4x896x160x160xf32) <- ([4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32, 4x128x160x160xf32], 1xi32)
        t451 = paddle._C_ops.concat(t450, t442)
        del t450

        # pd_op.conv2d: (4x256x160x160xf32) <- (4x896x160x160xf32, 256x896x1x1xf32)
        t452 = paddle._C_ops.conv2d(
            t451, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (4x256x160x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x160x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t453, t454, t455, t456, t457, t458 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t452,
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

        # pd_op.relu: (4x256x160x160xf32) <- (4x256x160x160xf32)
        t459 = paddle._C_ops.relu(t453)
        del t453

        # pd_op.full_int_array: (2xi64) <- ()
        t460 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        t461 = t460

        # pd_op.assign: (2xi64) <- (2xi64)
        t462 = t460

        # pd_op.assign: (2xi64) <- (2xi64)
        t463 = t460

        # pd_op.assign: (2xi64) <- (2xi64)
        t464 = t460

        # pd_op.pool2d: (4x256x1x1xf32) <- (4x256x160x160xf32, 2xi64)
        t465 = paddle._C_ops.pool2d(
            t459,
            t460,
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

        # pd_op.conv2d: (4x256x1x1xf32) <- (4x256x1x1xf32, 256x256x1x1xf32)
        t466 = paddle._C_ops.conv2d(
            t465, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.full_int_array: (4xi64) <- ()
        t467 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t468 = paddle._C_ops.reshape(t51, t467)
        del t51

        # pd_op.add: (4x256x1x1xf32) <- (4x256x1x1xf32, 1x256x1x1xf32)
        t469 = paddle._C_ops.add(t466, t468)

        # pd_op.sigmoid: (4x256x1x1xf32) <- (4x256x1x1xf32)
        t470 = paddle._C_ops.sigmoid(t469)
        del t469

        # pd_op.multiply: (4x256x160x160xf32) <- (4x256x160x160xf32, 4x256x1x1xf32)
        t471 = paddle._C_ops.multiply(t459, t470)

        # pd_op.depthwise_conv2d: (4x256x80x80xf32) <- (4x256x160x160xf32, 256x1x3x3xf32)
        t472 = paddle._C_ops.depthwise_conv2d(
            t471, t52, [2, 2], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t52

        # pd_op.batch_norm_: (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x80x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t473, t474, t475, t476, t477, t478 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t472,
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

        # pd_op.conv2d: (4x160x80x80xf32) <- (4x256x80x80xf32, 160x256x3x3xf32)
        t479 = paddle._C_ops.conv2d(
            t473, t57, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t57

        # pd_op.batch_norm_: (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t480, t481, t482, t483, t484, t485 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t479,
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

        # pd_op.relu: (4x160x80x80xf32) <- (4x160x80x80xf32)
        t486 = paddle._C_ops.relu(t480)
        del t480

        # pd_op.conv2d: (4x160x80x80xf32) <- (4x160x80x80xf32, 160x160x3x3xf32)
        t487 = paddle._C_ops.conv2d(
            t486, t62, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t62

        # pd_op.batch_norm_: (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t488, t489, t490, t491, t492, t493 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t487,
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

        # pd_op.relu: (4x160x80x80xf32) <- (4x160x80x80xf32)
        t494 = paddle._C_ops.relu(t488)
        del t488

        # pd_op.conv2d: (4x160x80x80xf32) <- (4x160x80x80xf32, 160x160x3x3xf32)
        t495 = paddle._C_ops.conv2d(
            t494, t67, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t67

        # pd_op.batch_norm_: (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t496, t497, t498, t499, t500, t501 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t495,
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

        # pd_op.relu: (4x160x80x80xf32) <- (4x160x80x80xf32)
        t502 = paddle._C_ops.relu(t496)
        del t496

        # pd_op.conv2d: (4x160x80x80xf32) <- (4x160x80x80xf32, 160x160x3x3xf32)
        t503 = paddle._C_ops.conv2d(
            t502, t72, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t72

        # pd_op.batch_norm_: (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t504, t505, t506, t507, t508, t509 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t503,
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

        # pd_op.relu: (4x160x80x80xf32) <- (4x160x80x80xf32)
        t510 = paddle._C_ops.relu(t504)
        del t504

        # pd_op.conv2d: (4x160x80x80xf32) <- (4x160x80x80xf32, 160x160x3x3xf32)
        t511 = paddle._C_ops.conv2d(
            t510, t77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t77

        # pd_op.batch_norm_: (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t512, t513, t514, t515, t516, t517 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t511,
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

        # pd_op.relu: (4x160x80x80xf32) <- (4x160x80x80xf32)
        t518 = paddle._C_ops.relu(t512)
        del t512

        # pd_op.conv2d: (4x160x80x80xf32) <- (4x160x80x80xf32, 160x160x3x3xf32)
        t519 = paddle._C_ops.conv2d(
            t518, t82, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t82

        # pd_op.batch_norm_: (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (4x160x80x80xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t520, t521, t522, t523, t524, t525 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t519,
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

        # pd_op.relu: (4x160x80x80xf32) <- (4x160x80x80xf32)
        t526 = paddle._C_ops.relu(t520)
        del t520

        # builtin.combine: ([4x256x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32]) <- (4x256x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32)
        t527 = [t473, t486, t494, t502, t510, t518, t526]

        # pd_op.concat: (4x1216x80x80xf32) <- ([4x256x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32, 4x160x80x80xf32], 1xi32)
        t528 = paddle._C_ops.concat(t527, t442)
        del t527

        # pd_op.conv2d: (4x512x80x80xf32) <- (4x1216x80x80xf32, 512x1216x1x1xf32)
        t529 = paddle._C_ops.conv2d(
            t528, t87, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t87

        # pd_op.batch_norm_: (4x512x80x80xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x80x80xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t530, t531, t532, t533, t534, t535 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t529,
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

        # pd_op.relu: (4x512x80x80xf32) <- (4x512x80x80xf32)
        t536 = paddle._C_ops.relu(t530)
        del t530

        # pd_op.pool2d: (4x512x1x1xf32) <- (4x512x80x80xf32, 2xi64)
        t537 = paddle._C_ops.pool2d(
            t536,
            t460,
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

        # pd_op.conv2d: (4x512x1x1xf32) <- (4x512x1x1xf32, 512x512x1x1xf32)
        t538 = paddle._C_ops.conv2d(
            t537, t92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t92

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t539 = paddle._C_ops.reshape(t93, t467)
        del t93

        # pd_op.add: (4x512x1x1xf32) <- (4x512x1x1xf32, 1x512x1x1xf32)
        t540 = paddle._C_ops.add(t538, t539)

        # pd_op.sigmoid: (4x512x1x1xf32) <- (4x512x1x1xf32)
        t541 = paddle._C_ops.sigmoid(t540)
        del t540

        # pd_op.multiply: (4x512x80x80xf32) <- (4x512x80x80xf32, 4x512x1x1xf32)
        t542 = paddle._C_ops.multiply(t536, t541)

        # pd_op.depthwise_conv2d: (4x512x40x40xf32) <- (4x512x80x80xf32, 512x1x3x3xf32)
        t543 = paddle._C_ops.depthwise_conv2d(
            t542, t94, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t94

        # pd_op.batch_norm_: (4x512x40x40xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x40x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t544, t545, t546, t547, t548, t549 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t543,
                t95,
                t96,
                t97,
                t98,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t98, t97, t96, t95

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x512x40x40xf32, 192x512x3x3xf32)
        t550 = paddle._C_ops.conv2d(
            t544, t99, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t99

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t551, t552, t553, t554, t555, t556 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t550,
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

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t557 = paddle._C_ops.relu(t551)
        del t551

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t558 = paddle._C_ops.conv2d(
            t557, t104, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t104

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t559, t560, t561, t562, t563, t564 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t558,
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

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t565 = paddle._C_ops.relu(t559)
        del t559

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t566 = paddle._C_ops.conv2d(
            t565, t109, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t109

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t567, t568, t569, t570, t571, t572 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t566,
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

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t573 = paddle._C_ops.relu(t567)
        del t567

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t574 = paddle._C_ops.conv2d(
            t573, t114, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t114

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t575, t576, t577, t578, t579, t580 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t574,
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

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t581 = paddle._C_ops.relu(t575)
        del t575

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t582 = paddle._C_ops.conv2d(
            t581, t119, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t119

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t583, t584, t585, t586, t587, t588 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t582,
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

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t589 = paddle._C_ops.relu(t583)
        del t583

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t590 = paddle._C_ops.conv2d(
            t589, t124, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t124

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t591, t592, t593, t594, t595, t596 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t590,
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

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t597 = paddle._C_ops.relu(t591)
        del t591

        # builtin.combine: ([4x512x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32]) <- (4x512x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32)
        t598 = [t544, t557, t565, t573, t581, t589, t597]

        # pd_op.concat: (4x1664x40x40xf32) <- ([4x512x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32], 1xi32)
        t599 = paddle._C_ops.concat(t598, t442)
        del t598

        # pd_op.conv2d: (4x768x40x40xf32) <- (4x1664x40x40xf32, 768x1664x1x1xf32)
        t600 = paddle._C_ops.conv2d(
            t599, t129, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t129

        # pd_op.batch_norm_: (4x768x40x40xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (4x768x40x40xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t601, t602, t603, t604, t605, t606 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t600,
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

        # pd_op.relu: (4x768x40x40xf32) <- (4x768x40x40xf32)
        t607 = paddle._C_ops.relu(t601)
        del t601

        # pd_op.pool2d: (4x768x1x1xf32) <- (4x768x40x40xf32, 2xi64)
        t608 = paddle._C_ops.pool2d(
            t607,
            t460,
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

        # pd_op.conv2d: (4x768x1x1xf32) <- (4x768x1x1xf32, 768x768x1x1xf32)
        t609 = paddle._C_ops.conv2d(
            t608, t134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t134

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t610 = paddle._C_ops.reshape(t135, t467)
        del t135

        # pd_op.add: (4x768x1x1xf32) <- (4x768x1x1xf32, 1x768x1x1xf32)
        t611 = paddle._C_ops.add(t609, t610)

        # pd_op.sigmoid: (4x768x1x1xf32) <- (4x768x1x1xf32)
        t612 = paddle._C_ops.sigmoid(t611)
        del t611

        # pd_op.multiply: (4x768x40x40xf32) <- (4x768x40x40xf32, 4x768x1x1xf32)
        t613 = paddle._C_ops.multiply(t607, t612)

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x768x40x40xf32, 192x768x3x3xf32)
        t614 = paddle._C_ops.conv2d(
            t613, t136, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t136

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t615, t616, t617, t618, t619, t620 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t614,
                t137,
                t138,
                t139,
                t140,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t140, t139, t138, t137

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t621 = paddle._C_ops.relu(t615)
        del t615

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t622 = paddle._C_ops.conv2d(
            t621, t141, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t141

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t623, t624, t625, t626, t627, t628 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t622,
                t142,
                t143,
                t144,
                t145,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t145, t144, t143, t142

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t629 = paddle._C_ops.relu(t623)
        del t623

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t630 = paddle._C_ops.conv2d(
            t629, t146, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t146

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t631, t632, t633, t634, t635, t636 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t630,
                t147,
                t148,
                t149,
                t150,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t150, t149, t148, t147

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t637 = paddle._C_ops.relu(t631)
        del t631

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t638 = paddle._C_ops.conv2d(
            t637, t151, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t151

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t639, t640, t641, t642, t643, t644 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t638,
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

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t645 = paddle._C_ops.relu(t639)
        del t639

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t646 = paddle._C_ops.conv2d(
            t645, t156, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t647, t648, t649, t650, t651, t652 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t646,
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

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t653 = paddle._C_ops.relu(t647)
        del t647

        # pd_op.conv2d: (4x192x40x40xf32) <- (4x192x40x40xf32, 192x192x3x3xf32)
        t654 = paddle._C_ops.conv2d(
            t653, t161, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t161

        # pd_op.batch_norm_: (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (4x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t655, t656, t657, t658, t659, t660 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t654,
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

        # pd_op.relu: (4x192x40x40xf32) <- (4x192x40x40xf32)
        t661 = paddle._C_ops.relu(t655)
        del t655

        # builtin.combine: ([4x768x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32]) <- (4x768x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32)
        t662 = [t613, t621, t629, t637, t645, t653, t661]

        # pd_op.concat: (4x1920x40x40xf32) <- ([4x768x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32, 4x192x40x40xf32], 1xi32)
        t663 = paddle._C_ops.concat(t662, t442)
        del t662

        # pd_op.conv2d: (4x768x40x40xf32) <- (4x1920x40x40xf32, 768x1920x1x1xf32)
        t664 = paddle._C_ops.conv2d(
            t663, t166, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t166

        # pd_op.batch_norm_: (4x768x40x40xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (4x768x40x40xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t665, t666, t667, t668, t669, t670 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t664,
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

        # pd_op.relu: (4x768x40x40xf32) <- (4x768x40x40xf32)
        t671 = paddle._C_ops.relu(t665)
        del t665

        # pd_op.pool2d: (4x768x1x1xf32) <- (4x768x40x40xf32, 2xi64)
        t672 = paddle._C_ops.pool2d(
            t671,
            t460,
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

        # pd_op.conv2d: (4x768x1x1xf32) <- (4x768x1x1xf32, 768x768x1x1xf32)
        t673 = paddle._C_ops.conv2d(
            t672, t171, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t171

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t674 = paddle._C_ops.reshape(t172, t467)
        del t172

        # pd_op.add: (4x768x1x1xf32) <- (4x768x1x1xf32, 1x768x1x1xf32)
        t675 = paddle._C_ops.add(t673, t674)

        # pd_op.sigmoid: (4x768x1x1xf32) <- (4x768x1x1xf32)
        t676 = paddle._C_ops.sigmoid(t675)
        del t675

        # pd_op.multiply: (4x768x40x40xf32) <- (4x768x40x40xf32, 4x768x1x1xf32)
        t677 = paddle._C_ops.multiply(t671, t676)

        # pd_op.add: (4x768x40x40xf32) <- (4x768x40x40xf32, 4x768x40x40xf32)
        t678 = paddle._C_ops.add(t677, t613)

        # pd_op.depthwise_conv2d: (4x768x20x20xf32) <- (4x768x40x40xf32, 768x1x3x3xf32)
        t679 = paddle._C_ops.depthwise_conv2d(
            t678, t173, [2, 2], [1, 1], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t173

        # pd_op.batch_norm_: (4x768x20x20xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (4x768x20x20xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t680, t681, t682, t683, t684, t685 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t679,
                t174,
                t175,
                t176,
                t177,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t177, t176, t175, t174

        # pd_op.conv2d: (4x224x20x20xf32) <- (4x768x20x20xf32, 224x768x3x3xf32)
        t686 = paddle._C_ops.conv2d(
            t680, t178, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t178

        # pd_op.batch_norm_: (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t687, t688, t689, t690, t691, t692 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t686,
                t179,
                t180,
                t181,
                t182,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t182, t181, t180, t179

        # pd_op.relu: (4x224x20x20xf32) <- (4x224x20x20xf32)
        t693 = paddle._C_ops.relu(t687)
        del t687

        # pd_op.conv2d: (4x224x20x20xf32) <- (4x224x20x20xf32, 224x224x3x3xf32)
        t694 = paddle._C_ops.conv2d(
            t693, t183, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t183

        # pd_op.batch_norm_: (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t695, t696, t697, t698, t699, t700 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t694,
                t184,
                t185,
                t186,
                t187,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t187, t186, t185, t184

        # pd_op.relu: (4x224x20x20xf32) <- (4x224x20x20xf32)
        t701 = paddle._C_ops.relu(t695)
        del t695

        # pd_op.conv2d: (4x224x20x20xf32) <- (4x224x20x20xf32, 224x224x3x3xf32)
        t702 = paddle._C_ops.conv2d(
            t701, t188, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t188

        # pd_op.batch_norm_: (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t703, t704, t705, t706, t707, t708 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t702,
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

        # pd_op.relu: (4x224x20x20xf32) <- (4x224x20x20xf32)
        t709 = paddle._C_ops.relu(t703)
        del t703

        # pd_op.conv2d: (4x224x20x20xf32) <- (4x224x20x20xf32, 224x224x3x3xf32)
        t710 = paddle._C_ops.conv2d(
            t709, t193, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t193

        # pd_op.batch_norm_: (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t711, t712, t713, t714, t715, t716 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t710,
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

        # pd_op.relu: (4x224x20x20xf32) <- (4x224x20x20xf32)
        t717 = paddle._C_ops.relu(t711)
        del t711

        # pd_op.conv2d: (4x224x20x20xf32) <- (4x224x20x20xf32, 224x224x3x3xf32)
        t718 = paddle._C_ops.conv2d(
            t717, t198, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t198

        # pd_op.batch_norm_: (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t719, t720, t721, t722, t723, t724 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t718,
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

        # pd_op.relu: (4x224x20x20xf32) <- (4x224x20x20xf32)
        t725 = paddle._C_ops.relu(t719)
        del t719

        # pd_op.conv2d: (4x224x20x20xf32) <- (4x224x20x20xf32, 224x224x3x3xf32)
        t726 = paddle._C_ops.conv2d(
            t725, t203, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t203

        # pd_op.batch_norm_: (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (4x224x20x20xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t727, t728, t729, t730, t731, t732 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t726,
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

        # pd_op.relu: (4x224x20x20xf32) <- (4x224x20x20xf32)
        t733 = paddle._C_ops.relu(t727)
        del t727

        # builtin.combine: ([4x768x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32]) <- (4x768x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32)
        t734 = [t680, t693, t701, t709, t717, t725, t733]

        # pd_op.concat: (4x2112x20x20xf32) <- ([4x768x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32, 4x224x20x20xf32], 1xi32)
        t735 = paddle._C_ops.concat(t734, t442)
        del t734

        # pd_op.conv2d: (4x1024x20x20xf32) <- (4x2112x20x20xf32, 1024x2112x1x1xf32)
        t736 = paddle._C_ops.conv2d(
            t735, t208, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t208

        # pd_op.batch_norm_: (4x1024x20x20xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (4x1024x20x20xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t737, t738, t739, t740, t741, t742 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t736,
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

        # pd_op.relu: (4x1024x20x20xf32) <- (4x1024x20x20xf32)
        t743 = paddle._C_ops.relu(t737)
        del t737

        # pd_op.pool2d: (4x1024x1x1xf32) <- (4x1024x20x20xf32, 2xi64)
        t744 = paddle._C_ops.pool2d(
            t743,
            t460,
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

        # pd_op.conv2d: (4x1024x1x1xf32) <- (4x1024x1x1xf32, 1024x1024x1x1xf32)
        t745 = paddle._C_ops.conv2d(
            t744, t213, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t213

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t746 = paddle._C_ops.reshape(t214, t467)
        del t214

        # pd_op.add: (4x1024x1x1xf32) <- (4x1024x1x1xf32, 1x1024x1x1xf32)
        t747 = paddle._C_ops.add(t745, t746)

        # pd_op.sigmoid: (4x1024x1x1xf32) <- (4x1024x1x1xf32)
        t748 = paddle._C_ops.sigmoid(t747)
        del t747

        # pd_op.multiply: (4x1024x20x20xf32) <- (4x1024x20x20xf32, 4x1024x1x1xf32)
        t749 = paddle._C_ops.multiply(t743, t748)

        # pd_op.conv2d: (4x256x20x20xf32) <- (4x1024x20x20xf32, 256x1024x1x1xf32)
        t750 = paddle._C_ops.conv2d(
            t749, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.conv2d: (4x256x40x40xf32) <- (4x768x40x40xf32, 256x768x1x1xf32)
        t751 = paddle._C_ops.conv2d(
            t678, t216, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t216

        # pd_op.conv2d: (4x256x80x80xf32) <- (4x512x80x80xf32, 256x512x1x1xf32)
        t752 = paddle._C_ops.conv2d(
            t542, t217, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t217

        # pd_op.conv2d: (4x256x160x160xf32) <- (4x256x160x160xf32, 256x256x1x1xf32)
        t753 = paddle._C_ops.conv2d(
            t471, t218, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t218

        # pd_op.nearest_interp: (4x256x40x40xf32) <- (4x256x20x20xf32, None, None, None)
        t754 = paddle._C_ops.nearest_interp(
            t750,
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
            1,
        )

        # pd_op.add: (4x256x40x40xf32) <- (4x256x40x40xf32, 4x256x40x40xf32)
        t755 = paddle._C_ops.add(t751, t754)

        # pd_op.nearest_interp: (4x256x80x80xf32) <- (4x256x40x40xf32, None, None, None)
        t756 = paddle._C_ops.nearest_interp(
            t755,
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
            1,
        )

        # pd_op.add: (4x256x80x80xf32) <- (4x256x80x80xf32, 4x256x80x80xf32)
        t757 = paddle._C_ops.add(t752, t756)

        # pd_op.nearest_interp: (4x256x160x160xf32) <- (4x256x80x80xf32, None, None, None)
        t758 = paddle._C_ops.nearest_interp(
            t757,
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
            1,
        )

        # pd_op.add: (4x256x160x160xf32) <- (4x256x160x160xf32, 4x256x160x160xf32)
        t759 = paddle._C_ops.add(t753, t758)

        # pd_op.conv2d: (4x64x20x20xf32) <- (4x256x20x20xf32, 64x256x9x9xf32)
        t760 = paddle._C_ops.conv2d(
            t750, t219, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t219

        # pd_op.conv2d: (4x64x40x40xf32) <- (4x256x40x40xf32, 64x256x9x9xf32)
        t761 = paddle._C_ops.conv2d(
            t755, t220, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t220

        # pd_op.conv2d: (4x64x80x80xf32) <- (4x256x80x80xf32, 64x256x9x9xf32)
        t762 = paddle._C_ops.conv2d(
            t757, t221, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t221

        # pd_op.conv2d: (4x64x160x160xf32) <- (4x256x160x160xf32, 64x256x9x9xf32)
        t763 = paddle._C_ops.conv2d(
            t759, t222, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t222

        # pd_op.conv2d: (4x64x80x80xf32) <- (4x64x160x160xf32, 64x64x3x3xf32)
        t764 = paddle._C_ops.conv2d(
            t763, t223, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t223

        # pd_op.add: (4x64x80x80xf32) <- (4x64x80x80xf32, 4x64x80x80xf32)
        t765 = paddle._C_ops.add(t762, t764)

        # pd_op.conv2d: (4x64x40x40xf32) <- (4x64x80x80xf32, 64x64x3x3xf32)
        t766 = paddle._C_ops.conv2d(
            t765, t224, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t224

        # pd_op.add: (4x64x40x40xf32) <- (4x64x40x40xf32, 4x64x40x40xf32)
        t767 = paddle._C_ops.add(t761, t766)

        # pd_op.conv2d: (4x64x20x20xf32) <- (4x64x40x40xf32, 64x64x3x3xf32)
        t768 = paddle._C_ops.conv2d(
            t767, t225, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225

        # pd_op.add: (4x64x20x20xf32) <- (4x64x20x20xf32, 4x64x20x20xf32)
        t769 = paddle._C_ops.add(t760, t768)

        # pd_op.conv2d: (4x64x160x160xf32) <- (4x64x160x160xf32, 64x64x9x9xf32)
        t770 = paddle._C_ops.conv2d(
            t763, t226, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t226

        # pd_op.conv2d: (4x64x80x80xf32) <- (4x64x80x80xf32, 64x64x9x9xf32)
        t771 = paddle._C_ops.conv2d(
            t765, t227, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t227

        # pd_op.conv2d: (4x64x40x40xf32) <- (4x64x40x40xf32, 64x64x9x9xf32)
        t772 = paddle._C_ops.conv2d(
            t767, t228, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t228

        # pd_op.conv2d: (4x64x20x20xf32) <- (4x64x20x20xf32, 64x64x9x9xf32)
        t773 = paddle._C_ops.conv2d(
            t769, t229, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t229

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x64x20x20xf32, 32x64x1x1xf32)
        t774 = paddle._C_ops.conv2d(
            t773, t230, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t230

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t775 = paddle._C_ops.reshape(t231, t467)
        del t231

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t776 = paddle._C_ops.add(t774, t775)

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x32x20x20xf32, 32x32x7x7xf32)
        t777 = paddle._C_ops.conv2d(
            t776, t232, [1, 1], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t232

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t778 = paddle._C_ops.reshape(t233, t467)
        del t233

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t779 = paddle._C_ops.add(t777, t778)

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x32x20x20xf32, 32x32x7x1xf32)
        t780 = paddle._C_ops.conv2d(
            t776, t234, [1, 1], [3, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t234

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t781 = paddle._C_ops.reshape(t235, t467)
        del t235

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t782 = paddle._C_ops.add(t780, t781)

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x32x20x20xf32, 32x32x1x7xf32)
        t783 = paddle._C_ops.conv2d(
            t776, t236, [1, 1], [0, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t236

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t784 = paddle._C_ops.reshape(t237, t467)
        del t237

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t785 = paddle._C_ops.add(t783, t784)

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 4x32x20x20xf32)
        t786 = paddle._C_ops.add(t779, t782)

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 4x32x20x20xf32)
        t787 = paddle._C_ops.add(t786, t785)

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x32x20x20xf32, 32x32x5x5xf32)
        t788 = paddle._C_ops.conv2d(
            t787, t238, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t238

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t789 = paddle._C_ops.reshape(t239, t467)
        del t239

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t790 = paddle._C_ops.add(t788, t789)

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x32x20x20xf32, 32x32x5x1xf32)
        t791 = paddle._C_ops.conv2d(
            t787, t240, [1, 1], [2, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t240

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t792 = paddle._C_ops.reshape(t241, t467)
        del t241

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t793 = paddle._C_ops.add(t791, t792)

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x32x20x20xf32, 32x32x1x5xf32)
        t794 = paddle._C_ops.conv2d(
            t787, t242, [1, 1], [0, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t242

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t795 = paddle._C_ops.reshape(t243, t467)
        del t243

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t796 = paddle._C_ops.add(t794, t795)

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 4x32x20x20xf32)
        t797 = paddle._C_ops.add(t790, t793)

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 4x32x20x20xf32)
        t798 = paddle._C_ops.add(t797, t796)

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x32x20x20xf32, 32x32x3x3xf32)
        t799 = paddle._C_ops.conv2d(
            t798, t244, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t244

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t800 = paddle._C_ops.reshape(t245, t467)
        del t245

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t801 = paddle._C_ops.add(t799, t800)

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x32x20x20xf32, 32x32x3x1xf32)
        t802 = paddle._C_ops.conv2d(
            t798, t246, [1, 1], [1, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t246

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t803 = paddle._C_ops.reshape(t247, t467)
        del t247

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t804 = paddle._C_ops.add(t802, t803)

        # pd_op.conv2d: (4x32x20x20xf32) <- (4x32x20x20xf32, 32x32x1x3xf32)
        t805 = paddle._C_ops.conv2d(
            t798, t248, [1, 1], [0, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t248

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t806 = paddle._C_ops.reshape(t249, t467)
        del t249

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 1x32x1x1xf32)
        t807 = paddle._C_ops.add(t805, t806)

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 4x32x20x20xf32)
        t808 = paddle._C_ops.add(t801, t804)

        # pd_op.add: (4x32x20x20xf32) <- (4x32x20x20xf32, 4x32x20x20xf32)
        t809 = paddle._C_ops.add(t808, t807)

        # pd_op.conv2d: (4x64x20x20xf32) <- (4x32x20x20xf32, 64x32x1x1xf32)
        t810 = paddle._C_ops.conv2d(
            t809, t250, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t250

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t811 = paddle._C_ops.reshape(t251, t467)
        del t251

        # pd_op.add: (4x64x20x20xf32) <- (4x64x20x20xf32, 1x64x1x1xf32)
        t812 = paddle._C_ops.add(t810, t811)

        # pd_op.batch_norm_: (4x64x20x20xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x20x20xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t813, t814, t815, t816, t817, t818 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t812,
                t252,
                t253,
                t254,
                t255,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t255, t254, t253, t252

        # pd_op.relu: (4x64x20x20xf32) <- (4x64x20x20xf32)
        t819 = paddle._C_ops.relu(t813)
        del t813

        # pd_op.add: (4x64x20x20xf32) <- (4x64x20x20xf32, 4x64x20x20xf32)
        t820 = paddle._C_ops.add(t773, t819)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x64x40x40xf32, 32x64x1x1xf32)
        t821 = paddle._C_ops.conv2d(
            t772, t256, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t256

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t822 = paddle._C_ops.reshape(t257, t467)
        del t257

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t823 = paddle._C_ops.add(t821, t822)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x32x40x40xf32, 32x32x7x7xf32)
        t824 = paddle._C_ops.conv2d(
            t823, t258, [1, 1], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t258

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t825 = paddle._C_ops.reshape(t259, t467)
        del t259

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t826 = paddle._C_ops.add(t824, t825)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x32x40x40xf32, 32x32x7x1xf32)
        t827 = paddle._C_ops.conv2d(
            t823, t260, [1, 1], [3, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t260

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t828 = paddle._C_ops.reshape(t261, t467)
        del t261

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t829 = paddle._C_ops.add(t827, t828)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x32x40x40xf32, 32x32x1x7xf32)
        t830 = paddle._C_ops.conv2d(
            t823, t262, [1, 1], [0, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t262

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t831 = paddle._C_ops.reshape(t263, t467)
        del t263

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t832 = paddle._C_ops.add(t830, t831)

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 4x32x40x40xf32)
        t833 = paddle._C_ops.add(t826, t829)

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 4x32x40x40xf32)
        t834 = paddle._C_ops.add(t833, t832)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x32x40x40xf32, 32x32x5x5xf32)
        t835 = paddle._C_ops.conv2d(
            t834, t264, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t264

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t836 = paddle._C_ops.reshape(t265, t467)
        del t265

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t837 = paddle._C_ops.add(t835, t836)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x32x40x40xf32, 32x32x5x1xf32)
        t838 = paddle._C_ops.conv2d(
            t834, t266, [1, 1], [2, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t266

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t839 = paddle._C_ops.reshape(t267, t467)
        del t267

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t840 = paddle._C_ops.add(t838, t839)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x32x40x40xf32, 32x32x1x5xf32)
        t841 = paddle._C_ops.conv2d(
            t834, t268, [1, 1], [0, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t268

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t842 = paddle._C_ops.reshape(t269, t467)
        del t269

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t843 = paddle._C_ops.add(t841, t842)

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 4x32x40x40xf32)
        t844 = paddle._C_ops.add(t837, t840)

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 4x32x40x40xf32)
        t845 = paddle._C_ops.add(t844, t843)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x32x40x40xf32, 32x32x3x3xf32)
        t846 = paddle._C_ops.conv2d(
            t845, t270, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t270

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t847 = paddle._C_ops.reshape(t271, t467)
        del t271

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t848 = paddle._C_ops.add(t846, t847)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x32x40x40xf32, 32x32x3x1xf32)
        t849 = paddle._C_ops.conv2d(
            t845, t272, [1, 1], [1, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t272

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t850 = paddle._C_ops.reshape(t273, t467)
        del t273

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t851 = paddle._C_ops.add(t849, t850)

        # pd_op.conv2d: (4x32x40x40xf32) <- (4x32x40x40xf32, 32x32x1x3xf32)
        t852 = paddle._C_ops.conv2d(
            t845, t274, [1, 1], [0, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t274

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t853 = paddle._C_ops.reshape(t275, t467)
        del t275

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 1x32x1x1xf32)
        t854 = paddle._C_ops.add(t852, t853)

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 4x32x40x40xf32)
        t855 = paddle._C_ops.add(t848, t851)

        # pd_op.add: (4x32x40x40xf32) <- (4x32x40x40xf32, 4x32x40x40xf32)
        t856 = paddle._C_ops.add(t855, t854)

        # pd_op.conv2d: (4x64x40x40xf32) <- (4x32x40x40xf32, 64x32x1x1xf32)
        t857 = paddle._C_ops.conv2d(
            t856, t276, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t276

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t858 = paddle._C_ops.reshape(t277, t467)
        del t277

        # pd_op.add: (4x64x40x40xf32) <- (4x64x40x40xf32, 1x64x1x1xf32)
        t859 = paddle._C_ops.add(t857, t858)

        # pd_op.batch_norm_: (4x64x40x40xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x40x40xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t860, t861, t862, t863, t864, t865 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t859,
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

        # pd_op.relu: (4x64x40x40xf32) <- (4x64x40x40xf32)
        t866 = paddle._C_ops.relu(t860)
        del t860

        # pd_op.add: (4x64x40x40xf32) <- (4x64x40x40xf32, 4x64x40x40xf32)
        t867 = paddle._C_ops.add(t772, t866)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x64x80x80xf32, 32x64x1x1xf32)
        t868 = paddle._C_ops.conv2d(
            t771, t282, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t282

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t869 = paddle._C_ops.reshape(t283, t467)
        del t283

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t870 = paddle._C_ops.add(t868, t869)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x32x80x80xf32, 32x32x7x7xf32)
        t871 = paddle._C_ops.conv2d(
            t870, t284, [1, 1], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t284

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t872 = paddle._C_ops.reshape(t285, t467)
        del t285

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t873 = paddle._C_ops.add(t871, t872)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x32x80x80xf32, 32x32x7x1xf32)
        t874 = paddle._C_ops.conv2d(
            t870, t286, [1, 1], [3, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t286

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t875 = paddle._C_ops.reshape(t287, t467)
        del t287

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t876 = paddle._C_ops.add(t874, t875)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x32x80x80xf32, 32x32x1x7xf32)
        t877 = paddle._C_ops.conv2d(
            t870, t288, [1, 1], [0, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t288

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t878 = paddle._C_ops.reshape(t289, t467)
        del t289

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t879 = paddle._C_ops.add(t877, t878)

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 4x32x80x80xf32)
        t880 = paddle._C_ops.add(t873, t876)

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 4x32x80x80xf32)
        t881 = paddle._C_ops.add(t880, t879)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x32x80x80xf32, 32x32x5x5xf32)
        t882 = paddle._C_ops.conv2d(
            t881, t290, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t290

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t883 = paddle._C_ops.reshape(t291, t467)
        del t291

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t884 = paddle._C_ops.add(t882, t883)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x32x80x80xf32, 32x32x5x1xf32)
        t885 = paddle._C_ops.conv2d(
            t881, t292, [1, 1], [2, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t292

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t886 = paddle._C_ops.reshape(t293, t467)
        del t293

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t887 = paddle._C_ops.add(t885, t886)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x32x80x80xf32, 32x32x1x5xf32)
        t888 = paddle._C_ops.conv2d(
            t881, t294, [1, 1], [0, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t294

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t889 = paddle._C_ops.reshape(t295, t467)
        del t295

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t890 = paddle._C_ops.add(t888, t889)

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 4x32x80x80xf32)
        t891 = paddle._C_ops.add(t884, t887)

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 4x32x80x80xf32)
        t892 = paddle._C_ops.add(t891, t890)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x32x80x80xf32, 32x32x3x3xf32)
        t893 = paddle._C_ops.conv2d(
            t892, t296, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t296

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t894 = paddle._C_ops.reshape(t297, t467)
        del t297

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t895 = paddle._C_ops.add(t893, t894)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x32x80x80xf32, 32x32x3x1xf32)
        t896 = paddle._C_ops.conv2d(
            t892, t298, [1, 1], [1, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t298

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t897 = paddle._C_ops.reshape(t299, t467)
        del t299

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t898 = paddle._C_ops.add(t896, t897)

        # pd_op.conv2d: (4x32x80x80xf32) <- (4x32x80x80xf32, 32x32x1x3xf32)
        t899 = paddle._C_ops.conv2d(
            t892, t300, [1, 1], [0, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t300

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t900 = paddle._C_ops.reshape(t301, t467)
        del t301

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 1x32x1x1xf32)
        t901 = paddle._C_ops.add(t899, t900)

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 4x32x80x80xf32)
        t902 = paddle._C_ops.add(t895, t898)

        # pd_op.add: (4x32x80x80xf32) <- (4x32x80x80xf32, 4x32x80x80xf32)
        t903 = paddle._C_ops.add(t902, t901)

        # pd_op.conv2d: (4x64x80x80xf32) <- (4x32x80x80xf32, 64x32x1x1xf32)
        t904 = paddle._C_ops.conv2d(
            t903, t302, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t302

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t905 = paddle._C_ops.reshape(t303, t467)
        del t303

        # pd_op.add: (4x64x80x80xf32) <- (4x64x80x80xf32, 1x64x1x1xf32)
        t906 = paddle._C_ops.add(t904, t905)

        # pd_op.batch_norm_: (4x64x80x80xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x80x80xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t907, t908, t909, t910, t911, t912 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t906,
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

        # pd_op.relu: (4x64x80x80xf32) <- (4x64x80x80xf32)
        t913 = paddle._C_ops.relu(t907)
        del t907

        # pd_op.add: (4x64x80x80xf32) <- (4x64x80x80xf32, 4x64x80x80xf32)
        t914 = paddle._C_ops.add(t771, t913)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x64x160x160xf32, 32x64x1x1xf32)
        t915 = paddle._C_ops.conv2d(
            t770, t308, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t308

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t916 = paddle._C_ops.reshape(t309, t467)
        del t309

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t917 = paddle._C_ops.add(t915, t916)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x32x160x160xf32, 32x32x7x7xf32)
        t918 = paddle._C_ops.conv2d(
            t917, t310, [1, 1], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t310

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t919 = paddle._C_ops.reshape(t311, t467)
        del t311

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t920 = paddle._C_ops.add(t918, t919)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x32x160x160xf32, 32x32x7x1xf32)
        t921 = paddle._C_ops.conv2d(
            t917, t312, [1, 1], [3, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t312

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t922 = paddle._C_ops.reshape(t313, t467)
        del t313

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t923 = paddle._C_ops.add(t921, t922)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x32x160x160xf32, 32x32x1x7xf32)
        t924 = paddle._C_ops.conv2d(
            t917, t314, [1, 1], [0, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t314

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t925 = paddle._C_ops.reshape(t315, t467)
        del t315

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t926 = paddle._C_ops.add(t924, t925)

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 4x32x160x160xf32)
        t927 = paddle._C_ops.add(t920, t923)

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 4x32x160x160xf32)
        t928 = paddle._C_ops.add(t927, t926)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x32x160x160xf32, 32x32x5x5xf32)
        t929 = paddle._C_ops.conv2d(
            t928, t316, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t316

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t930 = paddle._C_ops.reshape(t317, t467)
        del t317

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t931 = paddle._C_ops.add(t929, t930)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x32x160x160xf32, 32x32x5x1xf32)
        t932 = paddle._C_ops.conv2d(
            t928, t318, [1, 1], [2, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t318

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t933 = paddle._C_ops.reshape(t319, t467)
        del t319

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t934 = paddle._C_ops.add(t932, t933)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x32x160x160xf32, 32x32x1x5xf32)
        t935 = paddle._C_ops.conv2d(
            t928, t320, [1, 1], [0, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t320

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t936 = paddle._C_ops.reshape(t321, t467)
        del t321

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t937 = paddle._C_ops.add(t935, t936)

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 4x32x160x160xf32)
        t938 = paddle._C_ops.add(t931, t934)

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 4x32x160x160xf32)
        t939 = paddle._C_ops.add(t938, t937)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x32x160x160xf32, 32x32x3x3xf32)
        t940 = paddle._C_ops.conv2d(
            t939, t322, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t322

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t941 = paddle._C_ops.reshape(t323, t467)
        del t323

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t942 = paddle._C_ops.add(t940, t941)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x32x160x160xf32, 32x32x3x1xf32)
        t943 = paddle._C_ops.conv2d(
            t939, t324, [1, 1], [1, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t324

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t944 = paddle._C_ops.reshape(t325, t467)
        del t325

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t945 = paddle._C_ops.add(t943, t944)

        # pd_op.conv2d: (4x32x160x160xf32) <- (4x32x160x160xf32, 32x32x1x3xf32)
        t946 = paddle._C_ops.conv2d(
            t939, t326, [1, 1], [0, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t326

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t947 = paddle._C_ops.reshape(t327, t467)
        del t327

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 1x32x1x1xf32)
        t948 = paddle._C_ops.add(t946, t947)

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 4x32x160x160xf32)
        t949 = paddle._C_ops.add(t942, t945)

        # pd_op.add: (4x32x160x160xf32) <- (4x32x160x160xf32, 4x32x160x160xf32)
        t950 = paddle._C_ops.add(t949, t948)

        # pd_op.conv2d: (4x64x160x160xf32) <- (4x32x160x160xf32, 64x32x1x1xf32)
        t951 = paddle._C_ops.conv2d(
            t950, t328, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t328

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t952 = paddle._C_ops.reshape(t329, t467)
        del t329

        # pd_op.add: (4x64x160x160xf32) <- (4x64x160x160xf32, 1x64x1x1xf32)
        t953 = paddle._C_ops.add(t951, t952)

        # pd_op.batch_norm_: (4x64x160x160xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x160x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t954, t955, t956, t957, t958, t959 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t953,
                t330,
                t331,
                t332,
                t333,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t333, t332, t331, t330

        # pd_op.relu: (4x64x160x160xf32) <- (4x64x160x160xf32)
        t960 = paddle._C_ops.relu(t954)
        del t954

        # pd_op.add: (4x64x160x160xf32) <- (4x64x160x160xf32, 4x64x160x160xf32)
        t961 = paddle._C_ops.add(t770, t960)

        # pd_op.nearest_interp: (4x64x160x160xf32) <- (4x64x20x20xf32, None, None, None)
        t962 = paddle._C_ops.nearest_interp(
            t820,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("8"), float("8")],
            "nearest",
            False,
            1,
        )

        # pd_op.nearest_interp: (4x64x160x160xf32) <- (4x64x40x40xf32, None, None, None)
        t963 = paddle._C_ops.nearest_interp(
            t867,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("4"), float("4")],
            "nearest",
            False,
            1,
        )

        # pd_op.nearest_interp: (4x64x160x160xf32) <- (4x64x80x80xf32, None, None, None)
        t964 = paddle._C_ops.nearest_interp(
            t914,
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
            1,
        )

        # builtin.combine: ([4x64x160x160xf32, 4x64x160x160xf32, 4x64x160x160xf32, 4x64x160x160xf32]) <- (4x64x160x160xf32, 4x64x160x160xf32, 4x64x160x160xf32, 4x64x160x160xf32)
        t965 = [t962, t963, t964, t961]

        # pd_op.concat: (4x256x160x160xf32) <- ([4x64x160x160xf32, 4x64x160x160xf32, 4x64x160x160xf32, 4x64x160x160xf32], 1xi32)
        t966 = paddle._C_ops.concat(t965, t442)
        del t965

        # pd_op.conv2d: (4x64x160x160xf32) <- (4x256x160x160xf32, 64x256x3x3xf32)
        t967 = paddle._C_ops.conv2d(
            t966, t334, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t334

        # pd_op.batch_norm_: (4x64x160x160xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x160x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t968, t969, t970, t971, t972, t973 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t967,
                t335,
                t336,
                t337,
                t338,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t338, t337, t336, t335

        # pd_op.relu: (4x64x160x160xf32) <- (4x64x160x160xf32)
        t974 = paddle._C_ops.relu(t968)
        del t968

        # pd_op.isnan: (4x64x160x160xb) <- (4x64x160x160xf32)
        t975 = paddle._C_ops.isnan(t974)

        # pd_op.full: (1xf32) <- ()
        t976 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (4x64x160x160xf32) <- (4x64x160x160xf32, 1xf32)
        t977 = paddle._C_ops.full_like(
            t974, t976, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (4x64x160x160xf32) <- (4x64x160x160xb, 4x64x160x160xf32, 4x64x160x160xf32)
        t978 = paddle._C_ops.where(t975, t977, t974)

        # pd_op.full_int_array: (0xi64) <- ()
        t979 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        t980 = t979

        # pd_op.assign: (0xi64) <- (0xi64)
        t981 = t979

        # pd_op.assign: (0xi64) <- (0xi64)
        t982 = t979

        # pd_op.conv2d_transpose: (4x64x320x320xf32) <- (4x64x160x160xf32, 64x64x2x2xf32, 0xi64)
        t983 = paddle._C_ops.conv2d_transpose(
            t978, t339, [2, 2], [0, 0], [], t979, "EXPLICIT", 1, [1, 1], "NCHW"
        )
        del t339

        # pd_op.full_int_array: (4xi64) <- ()
        t984 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t985 = paddle._C_ops.reshape(t340, t984)
        del t340

        # pd_op.add: (4x64x320x320xf32) <- (4x64x320x320xf32, 1x64x1x1xf32)
        t986 = paddle._C_ops.add(t983, t985)

        # pd_op.batch_norm_: (4x64x320x320xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x320x320xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t987, t988, t989, t990, t991, t992 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t986,
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

        # pd_op.relu: (4x64x320x320xf32) <- (4x64x320x320xf32)
        t993 = paddle._C_ops.relu(t987)
        del t987

        # pd_op.isnan: (4x64x320x320xb) <- (4x64x320x320xf32)
        t994 = paddle._C_ops.isnan(t993)

        # pd_op.full_like: (4x64x320x320xf32) <- (4x64x320x320xf32, 1xf32)
        t995 = paddle._C_ops.full_like(
            t993, t976, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (4x64x320x320xf32) <- (4x64x320x320xb, 4x64x320x320xf32, 4x64x320x320xf32)
        t996 = paddle._C_ops.where(t994, t995, t993)

        # pd_op.conv2d_transpose: (4x1x640x640xf32) <- (4x64x320x320xf32, 64x1x2x2xf32, 0xi64)
        t997 = paddle._C_ops.conv2d_transpose(
            t996, t345, [2, 2], [0, 0], [], t979, "EXPLICIT", 1, [1, 1], "NCHW"
        )
        del t345

        # pd_op.full_int_array: (4xi64) <- ()
        t998 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        t999 = paddle._C_ops.reshape(t346, t998)
        del t346

        # pd_op.add: (4x1x640x640xf32) <- (4x1x640x640xf32, 1x1x1x1xf32)
        t1000 = paddle._C_ops.add(t997, t999)

        # pd_op.sigmoid: (4x1x640x640xf32) <- (4x1x640x640xf32)
        t1001 = paddle._C_ops.sigmoid(t1000)
        del t1000

        # pd_op.nearest_interp: (4x64x640x640xf32) <- (4x64x320x320xf32, None, None, None)
        t1002 = paddle._C_ops.nearest_interp(
            t996,
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
            1,
        )

        # builtin.combine: ([4x1x640x640xf32, 4x64x640x640xf32]) <- (4x1x640x640xf32, 4x64x640x640xf32)
        t1003 = [t1001, t1002]

        # pd_op.concat: (4x65x640x640xf32) <- ([4x1x640x640xf32, 4x64x640x640xf32], 1xi32)
        t1004 = paddle._C_ops.concat(t1003, t442)
        del t1003

        # pd_op.conv2d: (4x64x640x640xf32) <- (4x65x640x640xf32, 64x65x3x3xf32)
        t1005 = paddle._C_ops.conv2d(
            t1004, t347, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t347

        # pd_op.batch_norm_: (4x64x640x640xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x640x640xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1006, t1007, t1008, t1009, t1010, t1011 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1005,
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

        # pd_op.relu: (4x64x640x640xf32) <- (4x64x640x640xf32)
        t1012 = paddle._C_ops.relu(t1006)
        del t1006

        # pd_op.conv2d: (4x1x640x640xf32) <- (4x64x640x640xf32, 1x64x1x1xf32)
        t1013 = paddle._C_ops.conv2d(
            t1012, t352, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t352

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        t1014 = paddle._C_ops.reshape(t353, t467)
        del t467, t353

        # pd_op.add: (4x1x640x640xf32) <- (4x1x640x640xf32, 1x1x1x1xf32)
        t1015 = paddle._C_ops.add(t1013, t1014)

        # pd_op.sigmoid: (4x1x640x640xf32) <- (4x1x640x640xf32)
        t1016 = paddle._C_ops.sigmoid(t1015)
        del t1015

        # pd_op.conv2d: (4x64x160x160xf32) <- (4x256x160x160xf32, 64x256x3x3xf32)
        t1017 = paddle._C_ops.conv2d(
            t966, t354, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t354

        # pd_op.batch_norm_: (4x64x160x160xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x160x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1018, t1019, t1020, t1021, t1022, t1023 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1017,
                t355,
                t356,
                t357,
                t358,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t356, t355, t358, t357

        # pd_op.relu: (4x64x160x160xf32) <- (4x64x160x160xf32)
        t1024 = paddle._C_ops.relu(t1018)
        del t1018

        # pd_op.isnan: (4x64x160x160xb) <- (4x64x160x160xf32)
        t1025 = paddle._C_ops.isnan(t1024)

        # pd_op.full_like: (4x64x160x160xf32) <- (4x64x160x160xf32, 1xf32)
        t1026 = paddle._C_ops.full_like(
            t1024, t976, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (4x64x160x160xf32) <- (4x64x160x160xb, 4x64x160x160xf32, 4x64x160x160xf32)
        t1027 = paddle._C_ops.where(t1025, t1026, t1024)

        # pd_op.conv2d_transpose: (4x64x320x320xf32) <- (4x64x160x160xf32, 64x64x2x2xf32, 0xi64)
        t1028 = paddle._C_ops.conv2d_transpose(
            t1027, t359, [2, 2], [0, 0], [], t979, "EXPLICIT", 1, [1, 1], "NCHW"
        )
        del t359

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t1029 = paddle._C_ops.reshape(t360, t984)
        del t984, t360

        # pd_op.add: (4x64x320x320xf32) <- (4x64x320x320xf32, 1x64x1x1xf32)
        t1030 = paddle._C_ops.add(t1028, t1029)

        # pd_op.batch_norm_: (4x64x320x320xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x320x320xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1031, t1032, t1033, t1034, t1035, t1036 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1030,
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

        # pd_op.relu: (4x64x320x320xf32) <- (4x64x320x320xf32)
        t1037 = paddle._C_ops.relu(t1031)
        del t1031

        # pd_op.isnan: (4x64x320x320xb) <- (4x64x320x320xf32)
        t1038 = paddle._C_ops.isnan(t1037)

        # pd_op.full_like: (4x64x320x320xf32) <- (4x64x320x320xf32, 1xf32)
        t1039 = paddle._C_ops.full_like(
            t1037, t976, paddle.float32, paddle.framework._current_expected_place()
        )
        del t976

        # pd_op.where: (4x64x320x320xf32) <- (4x64x320x320xb, 4x64x320x320xf32, 4x64x320x320xf32)
        t1040 = paddle._C_ops.where(t1038, t1039, t1037)

        # pd_op.conv2d_transpose: (4x1x640x640xf32) <- (4x64x320x320xf32, 64x1x2x2xf32, 0xi64)
        t1041 = paddle._C_ops.conv2d_transpose(
            t1040, t365, [2, 2], [0, 0], [], t979, "EXPLICIT", 1, [1, 1], "NCHW"
        )
        del t365

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        t1042 = paddle._C_ops.reshape(t366, t998)
        del t998, t366

        # pd_op.add: (4x1x640x640xf32) <- (4x1x640x640xf32, 1x1x1x1xf32)
        t1043 = paddle._C_ops.add(t1041, t1042)

        # pd_op.sigmoid: (4x1x640x640xf32) <- (4x1x640x640xf32)
        t1044 = paddle._C_ops.sigmoid(t1043)
        del t1043

        # pd_op.subtract: (4x1x640x640xf32) <- (4x1x640x640xf32, 4x1x640x640xf32)
        t1045 = paddle._C_ops.subtract(t1001, t1044)

        # pd_op.full: (1xf32) <- ()
        t1046 = paddle._C_ops.full(
            [1], float("-50"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x1x640x640xf32) <- (4x1x640x640xf32, 1xf32)
        t1047 = paddle._C_ops.scale(t1045, t1046, float("0"), True)
        del t1045

        # pd_op.exp: (4x1x640x640xf32) <- (4x1x640x640xf32)
        t1048 = paddle._C_ops.exp(t1047)
        del t1047

        # pd_op.full: (1xf32) <- ()
        t1049 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x1x640x640xf32) <- (4x1x640x640xf32, 1xf32)
        t1050 = paddle._C_ops.scale(t1048, t1049, float("1"), True)

        # pd_op.reciprocal: (4x1x640x640xf32) <- (4x1x640x640xf32)
        t1051 = paddle._C_ops.reciprocal(t1050)
        del t1050

        # builtin.combine: ([4x1x640x640xf32, 4x1x640x640xf32, 4x1x640x640xf32]) <- (4x1x640x640xf32, 4x1x640x640xf32, 4x1x640x640xf32)
        t1052 = [t1016, t1044, t1051]

        return (
            t368,
            t370,
            t371,
            t372,
            t373,
            t374,
            t375,
            t376,
            t378,
            t379,
            t380,
            t381,
            t382,
            t383,
            t384,
            t386,
            t387,
            t388,
            t389,
            t390,
            t391,
            t392,
            t393,
            t394,
            t396,
            t397,
            t398,
            t399,
            t400,
            t401,
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
            t428,
            t429,
            t430,
            t431,
            t432,
            t433,
            t434,
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
            t451,
            t452,
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
            t468,
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
            t481,
            t482,
            t483,
            t484,
            t485,
            t486,
            t487,
            t489,
            t490,
            t491,
            t492,
            t493,
            t494,
            t495,
            t497,
            t498,
            t499,
            t500,
            t501,
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
            t521,
            t522,
            t523,
            t524,
            t525,
            t526,
            t528,
            t529,
            t531,
            t532,
            t533,
            t534,
            t535,
            t536,
            t537,
            t538,
            t539,
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
            t574,
            t576,
            t577,
            t578,
            t579,
            t580,
            t581,
            t582,
            t584,
            t585,
            t586,
            t587,
            t588,
            t589,
            t590,
            t592,
            t593,
            t594,
            t595,
            t596,
            t597,
            t599,
            t600,
            t602,
            t603,
            t604,
            t605,
            t606,
            t607,
            t608,
            t609,
            t610,
            t612,
            t613,
            t614,
            t616,
            t617,
            t618,
            t619,
            t620,
            t621,
            t622,
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
            t636,
            t637,
            t638,
            t640,
            t641,
            t642,
            t643,
            t644,
            t645,
            t646,
            t648,
            t649,
            t650,
            t651,
            t652,
            t653,
            t654,
            t656,
            t657,
            t658,
            t659,
            t660,
            t661,
            t663,
            t664,
            t666,
            t667,
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
            t688,
            t689,
            t690,
            t691,
            t692,
            t693,
            t694,
            t696,
            t697,
            t698,
            t699,
            t700,
            t701,
            t702,
            t704,
            t705,
            t706,
            t707,
            t708,
            t709,
            t710,
            t712,
            t713,
            t714,
            t715,
            t716,
            t717,
            t718,
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
            t735,
            t736,
            t738,
            t739,
            t740,
            t741,
            t742,
            t743,
            t744,
            t745,
            t746,
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
            t966,
            t967,
            t969,
            t970,
            t971,
            t972,
            t973,
            t974,
            t975,
            t977,
            t978,
            t979,
            t980,
            t981,
            t982,
            t983,
            t985,
            t986,
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
            t999,
            t1001,
            t1002,
            t1004,
            t1005,
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
            t1046,
            t1048,
            t1049,
            t1051,
            t1052,
        )
