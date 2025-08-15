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
        # pd_op.conv2d: (-1x96x112x112xf32) <- (-1x3x224x224xf32, 96x3x3x3xf32)
        t369 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x96x112x112xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x112x112xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t370, t371, t372, t373, t374, t375 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t369,
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
        del t369, t4, t3, t2, t1

        # pd_op.relu: (-1x96x112x112xf32) <- (-1x96x112x112xf32)
        t376 = paddle._C_ops.relu(t370)
        del t370

        # pd_op.conv2d: (-1x96x112x112xf32) <- (-1x96x112x112xf32, 96x96x3x3xf32)
        t377 = paddle._C_ops.conv2d(
            t376, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5, t376

        # pd_op.batch_norm_: (-1x96x112x112xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x112x112xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t378, t379, t380, t381, t382, t383 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t377,
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
        del t377, t9, t8, t7, t6

        # pd_op.relu: (-1x96x112x112xf32) <- (-1x96x112x112xf32)
        t384 = paddle._C_ops.relu(t378)
        del t378

        # pd_op.conv2d: (-1x160x112x112xf32) <- (-1x96x112x112xf32, 160x96x3x3xf32)
        t385 = paddle._C_ops.conv2d(
            t384, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10, t384

        # pd_op.batch_norm_: (-1x160x112x112xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x112x112xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        t386, t387, t388, t389, t390, t391 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t385,
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
        del t385, t14, t13, t12, t11

        # pd_op.relu: (-1x160x112x112xf32) <- (-1x160x112x112xf32)
        t392 = paddle._C_ops.relu(t386)
        del t386

        # pd_op.full_int_array: (2xi64) <- ()
        t393 = [3, 3]

        # pd_op.pool2d: (-1x160x56x56xf32) <- (-1x160x112x112xf32, 2xi64)
        t394 = paddle._C_ops.pool2d(
            t392,
            t393,
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
        del t393, t392

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x160x56x56xf32, 192x160x3x3xf32)
        t395 = paddle._C_ops.conv2d(
            t394, t15, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t396, t397, t398, t399, t400, t401 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t395,
                t16,
                t17,
                t18,
                t19,
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
        del t395, t19, t18, t17, t16

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t402 = paddle._C_ops.relu(t396)
        del t396

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x192x3x3xf32)
        t403 = paddle._C_ops.conv2d(
            t402, t20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t404, t405, t406, t407, t408, t409 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t403,
                t21,
                t22,
                t23,
                t24,
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
        del t403, t24, t23, t22, t21

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t410 = paddle._C_ops.relu(t404)
        del t404

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x192x3x3xf32)
        t411 = paddle._C_ops.conv2d(
            t410, t25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t412, t413, t414, t415, t416, t417 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t411,
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
        del t411, t29, t28, t27, t26

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t418 = paddle._C_ops.relu(t412)
        del t412

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x192x3x3xf32)
        t419 = paddle._C_ops.conv2d(
            t418, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t420, t421, t422, t423, t424, t425 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t419,
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
        del t419, t34, t33, t32, t31

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t426 = paddle._C_ops.relu(t420)
        del t420

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x192x3x3xf32)
        t427 = paddle._C_ops.conv2d(
            t426, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t428, t429, t430, t431, t432, t433 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t427,
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
        del t427, t39, t38, t37, t36

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t434 = paddle._C_ops.relu(t428)
        del t428

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x192x3x3xf32)
        t435 = paddle._C_ops.conv2d(
            t434, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t436, t437, t438, t439, t440, t441 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t435,
                t41,
                t42,
                t43,
                t44,
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
        del t435, t44, t43, t42, t41

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t442 = paddle._C_ops.relu(t436)
        del t436

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x192x3x3xf32)
        t443 = paddle._C_ops.conv2d(
            t442, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t444, t445, t446, t447, t448, t449 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t443,
                t46,
                t47,
                t48,
                t49,
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
        del t443, t49, t48, t47, t46

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t450 = paddle._C_ops.relu(t444)
        del t444

        # pd_op.full: (1xi32) <- ()
        t451 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x160x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32]) <- (-1x160x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32)
        t452 = [t394, t402, t410, t418, t426, t434, t442, t450]
        del t394, t402, t410, t418, t426, t434, t442, t450

        # pd_op.concat: (-1x1504x56x56xf32) <- ([-1x160x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32, -1x192x56x56xf32], 1xi32)
        t453 = paddle._C_ops.concat(t452, t451)
        del t452

        # pd_op.conv2d: (-1x320x56x56xf32) <- (-1x1504x56x56xf32, 320x1504x1x1xf32)
        t454 = paddle._C_ops.conv2d(
            t453, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t453, t50

        # pd_op.batch_norm_: (-1x320x56x56xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (-1x320x56x56xf32, 320xf32, 320xf32, 320xf32, 320xf32)
        t455, t456, t457, t458, t459, t460 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t454,
                t51,
                t52,
                t53,
                t54,
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
        del t454, t54, t53, t52, t51

        # pd_op.relu: (-1x320x56x56xf32) <- (-1x320x56x56xf32)
        t461 = paddle._C_ops.relu(t455)
        del t455

        # pd_op.full_int_array: (2xi64) <- ()
        t462 = [1, 1]

        # pd_op.pool2d: (-1x320x1x1xf32) <- (-1x320x56x56xf32, 2xi64)
        t463 = paddle._C_ops.pool2d(
            t461,
            t462,
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

        # pd_op.conv2d: (-1x320x1x1xf32) <- (-1x320x1x1xf32, 320x320x1x1xf32)
        t464 = paddle._C_ops.conv2d(
            t463, t55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55, t463

        # pd_op.full_int_array: (4xi64) <- ()
        t465 = [1, -1, 1, 1]

        # pd_op.reshape: (1x320x1x1xf32) <- (320xf32, 4xi64)
        t466 = paddle._C_ops.reshape(t56, t465)
        del t56

        # pd_op.add: (-1x320x1x1xf32) <- (-1x320x1x1xf32, 1x320x1x1xf32)
        t467 = paddle._C_ops.add(t464, t466)
        del t464, t466

        # pd_op.sigmoid: (-1x320x1x1xf32) <- (-1x320x1x1xf32)
        t468 = paddle._C_ops.sigmoid(t467)
        del t467

        # pd_op.multiply: (-1x320x56x56xf32) <- (-1x320x56x56xf32, -1x320x1x1xf32)
        t469 = paddle._C_ops.multiply(t461, t468)
        del t461, t468

        # pd_op.depthwise_conv2d: (-1x320x28x28xf32) <- (-1x320x56x56xf32, 320x1x3x3xf32)
        t470 = paddle._C_ops.depthwise_conv2d(
            t469, t57, [2, 2], [1, 1], "EXPLICIT", 320, [1, 1], "NCHW"
        )
        del t469, t57

        # pd_op.batch_norm_: (-1x320x28x28xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (-1x320x28x28xf32, 320xf32, 320xf32, 320xf32, 320xf32)
        t471, t472, t473, t474, t475, t476 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t470,
                t58,
                t59,
                t60,
                t61,
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
        del t470, t61, t60, t59, t58

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x320x28x28xf32, 224x320x3x3xf32)
        t477 = paddle._C_ops.conv2d(
            t471, t62, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t62

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t478, t479, t480, t481, t482, t483 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t477,
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
        del t477, t66, t65, t64, t63

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t484 = paddle._C_ops.relu(t478)
        del t478

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t485 = paddle._C_ops.conv2d(
            t484, t67, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t67

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t486, t487, t488, t489, t490, t491 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t485,
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
        del t485, t71, t70, t69, t68

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t492 = paddle._C_ops.relu(t486)
        del t486

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t493 = paddle._C_ops.conv2d(
            t492, t72, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t72

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t494, t495, t496, t497, t498, t499 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t493,
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
        del t493, t76, t75, t74, t73

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t500 = paddle._C_ops.relu(t494)
        del t494

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t501 = paddle._C_ops.conv2d(
            t500, t77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t77

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t502, t503, t504, t505, t506, t507 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t501,
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
        del t501, t81, t80, t79, t78

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t508 = paddle._C_ops.relu(t502)
        del t502

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t509 = paddle._C_ops.conv2d(
            t508, t82, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t82

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t510, t511, t512, t513, t514, t515 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t509,
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
        del t509, t86, t85, t84, t83

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t516 = paddle._C_ops.relu(t510)
        del t510

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t517 = paddle._C_ops.conv2d(
            t516, t87, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t87

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t518, t519, t520, t521, t522, t523 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t517,
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
        del t517, t91, t90, t89, t88

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t524 = paddle._C_ops.relu(t518)
        del t518

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t525 = paddle._C_ops.conv2d(
            t524, t92, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t92

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t526, t527, t528, t529, t530, t531 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t525,
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
        del t525, t96, t95, t94, t93

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t532 = paddle._C_ops.relu(t526)
        del t526

        # builtin.combine: ([-1x320x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32]) <- (-1x320x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32)
        t533 = [t471, t484, t492, t500, t508, t516, t524, t532]
        del t471, t484, t492, t500, t508, t516, t524, t532

        # pd_op.concat: (-1x1888x28x28xf32) <- ([-1x320x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32], 1xi32)
        t534 = paddle._C_ops.concat(t533, t451)
        del t533

        # pd_op.conv2d: (-1x640x28x28xf32) <- (-1x1888x28x28xf32, 640x1888x1x1xf32)
        t535 = paddle._C_ops.conv2d(
            t534, t97, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t534, t97

        # pd_op.batch_norm_: (-1x640x28x28xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x28x28xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        t536, t537, t538, t539, t540, t541 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t535,
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
        del t535, t101, t100, t99, t98

        # pd_op.relu: (-1x640x28x28xf32) <- (-1x640x28x28xf32)
        t542 = paddle._C_ops.relu(t536)
        del t536

        # pd_op.pool2d: (-1x640x1x1xf32) <- (-1x640x28x28xf32, 2xi64)
        t543 = paddle._C_ops.pool2d(
            t542,
            t462,
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

        # pd_op.conv2d: (-1x640x1x1xf32) <- (-1x640x1x1xf32, 640x640x1x1xf32)
        t544 = paddle._C_ops.conv2d(
            t543, t102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t102, t543

        # pd_op.reshape: (1x640x1x1xf32) <- (640xf32, 4xi64)
        t545 = paddle._C_ops.reshape(t103, t465)
        del t103

        # pd_op.add: (-1x640x1x1xf32) <- (-1x640x1x1xf32, 1x640x1x1xf32)
        t546 = paddle._C_ops.add(t544, t545)
        del t544, t545

        # pd_op.sigmoid: (-1x640x1x1xf32) <- (-1x640x1x1xf32)
        t547 = paddle._C_ops.sigmoid(t546)
        del t546

        # pd_op.multiply: (-1x640x28x28xf32) <- (-1x640x28x28xf32, -1x640x1x1xf32)
        t548 = paddle._C_ops.multiply(t542, t547)
        del t542, t547

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x640x28x28xf32, 224x640x3x3xf32)
        t549 = paddle._C_ops.conv2d(
            t548, t104, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t104

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t550, t551, t552, t553, t554, t555 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t549,
                t105,
                t106,
                t107,
                t108,
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
        del t549, t108, t107, t106, t105

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t556 = paddle._C_ops.relu(t550)
        del t550

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t557 = paddle._C_ops.conv2d(
            t556, t109, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t109

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t558, t559, t560, t561, t562, t563 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t557,
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
        del t557, t113, t112, t111, t110

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t564 = paddle._C_ops.relu(t558)
        del t558

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t565 = paddle._C_ops.conv2d(
            t564, t114, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t114

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t566, t567, t568, t569, t570, t571 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t565,
                t115,
                t116,
                t117,
                t118,
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
        del t565, t118, t117, t116, t115

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t572 = paddle._C_ops.relu(t566)
        del t566

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t573 = paddle._C_ops.conv2d(
            t572, t119, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t119

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t574, t575, t576, t577, t578, t579 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t573,
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
        del t573, t123, t122, t121, t120

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t580 = paddle._C_ops.relu(t574)
        del t574

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t581 = paddle._C_ops.conv2d(
            t580, t124, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t124

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t582, t583, t584, t585, t586, t587 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t581,
                t125,
                t126,
                t127,
                t128,
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
        del t581, t128, t127, t126, t125

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t588 = paddle._C_ops.relu(t582)
        del t582

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t589 = paddle._C_ops.conv2d(
            t588, t129, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t129

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t590, t591, t592, t593, t594, t595 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t589,
                t130,
                t131,
                t132,
                t133,
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
        del t589, t133, t132, t131, t130

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t596 = paddle._C_ops.relu(t590)
        del t590

        # pd_op.conv2d: (-1x224x28x28xf32) <- (-1x224x28x28xf32, 224x224x3x3xf32)
        t597 = paddle._C_ops.conv2d(
            t596, t134, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t134

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        t598, t599, t600, t601, t602, t603 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t597,
                t135,
                t136,
                t137,
                t138,
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
        del t597, t138, t137, t136, t135

        # pd_op.relu: (-1x224x28x28xf32) <- (-1x224x28x28xf32)
        t604 = paddle._C_ops.relu(t598)
        del t598

        # builtin.combine: ([-1x640x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32]) <- (-1x640x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32)
        t605 = [t548, t556, t564, t572, t580, t588, t596, t604]
        del t556, t564, t572, t580, t588, t596, t604

        # pd_op.concat: (-1x2208x28x28xf32) <- ([-1x640x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32, -1x224x28x28xf32], 1xi32)
        t606 = paddle._C_ops.concat(t605, t451)
        del t605

        # pd_op.conv2d: (-1x640x28x28xf32) <- (-1x2208x28x28xf32, 640x2208x1x1xf32)
        t607 = paddle._C_ops.conv2d(
            t606, t139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t606, t139

        # pd_op.batch_norm_: (-1x640x28x28xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x28x28xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        t608, t609, t610, t611, t612, t613 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t607,
                t140,
                t141,
                t142,
                t143,
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
        del t607, t143, t142, t141, t140

        # pd_op.relu: (-1x640x28x28xf32) <- (-1x640x28x28xf32)
        t614 = paddle._C_ops.relu(t608)
        del t608

        # pd_op.pool2d: (-1x640x1x1xf32) <- (-1x640x28x28xf32, 2xi64)
        t615 = paddle._C_ops.pool2d(
            t614,
            t462,
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

        # pd_op.conv2d: (-1x640x1x1xf32) <- (-1x640x1x1xf32, 640x640x1x1xf32)
        t616 = paddle._C_ops.conv2d(
            t615, t144, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t144, t615

        # pd_op.reshape: (1x640x1x1xf32) <- (640xf32, 4xi64)
        t617 = paddle._C_ops.reshape(t145, t465)
        del t145

        # pd_op.add: (-1x640x1x1xf32) <- (-1x640x1x1xf32, 1x640x1x1xf32)
        t618 = paddle._C_ops.add(t616, t617)
        del t616, t617

        # pd_op.sigmoid: (-1x640x1x1xf32) <- (-1x640x1x1xf32)
        t619 = paddle._C_ops.sigmoid(t618)
        del t618

        # pd_op.multiply: (-1x640x28x28xf32) <- (-1x640x28x28xf32, -1x640x1x1xf32)
        t620 = paddle._C_ops.multiply(t614, t619)
        del t614, t619

        # pd_op.add: (-1x640x28x28xf32) <- (-1x640x28x28xf32, -1x640x28x28xf32)
        t621 = paddle._C_ops.add(t620, t548)
        del t548, t620

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x28x28xf32, 640x1x3x3xf32)
        t622 = paddle._C_ops.depthwise_conv2d(
            t621, t146, [2, 2], [1, 1], "EXPLICIT", 640, [1, 1], "NCHW"
        )
        del t621, t146

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        t623, t624, t625, t626, t627, t628 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t622,
                t147,
                t148,
                t149,
                t150,
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
        del t622, t150, t149, t148, t147

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x640x14x14xf32, 256x640x3x3xf32)
        t629 = paddle._C_ops.conv2d(
            t623, t151, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t151

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t630, t631, t632, t633, t634, t635 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t629,
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
        del t629, t155, t154, t153, t152

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t636 = paddle._C_ops.relu(t630)
        del t630

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t637 = paddle._C_ops.conv2d(
            t636, t156, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t638, t639, t640, t641, t642, t643 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t637,
                t157,
                t158,
                t159,
                t160,
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
        del t637, t160, t159, t158, t157

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t644 = paddle._C_ops.relu(t638)
        del t638

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t645 = paddle._C_ops.conv2d(
            t644, t161, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t161

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t646, t647, t648, t649, t650, t651 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t645,
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
        del t645, t165, t164, t163, t162

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t652 = paddle._C_ops.relu(t646)
        del t646

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t653 = paddle._C_ops.conv2d(
            t652, t166, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t166

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t654, t655, t656, t657, t658, t659 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t653,
                t167,
                t168,
                t169,
                t170,
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
        del t653, t170, t169, t168, t167

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t660 = paddle._C_ops.relu(t654)
        del t654

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t661 = paddle._C_ops.conv2d(
            t660, t171, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t171

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t662, t663, t664, t665, t666, t667 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t661,
                t172,
                t173,
                t174,
                t175,
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
        del t661, t175, t174, t173, t172

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t668 = paddle._C_ops.relu(t662)
        del t662

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t669 = paddle._C_ops.conv2d(
            t668, t176, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t176

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t670, t671, t672, t673, t674, t675 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t669,
                t177,
                t178,
                t179,
                t180,
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
        del t669, t180, t179, t178, t177

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t676 = paddle._C_ops.relu(t670)
        del t670

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t677 = paddle._C_ops.conv2d(
            t676, t181, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t181

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t678, t679, t680, t681, t682, t683 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t677,
                t182,
                t183,
                t184,
                t185,
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
        del t677, t185, t184, t183, t182

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t684 = paddle._C_ops.relu(t678)
        del t678

        # builtin.combine: ([-1x640x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x640x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32)
        t685 = [t623, t636, t644, t652, t660, t668, t676, t684]
        del t623, t636, t644, t652, t660, t668, t676, t684

        # pd_op.concat: (-1x2432x14x14xf32) <- ([-1x640x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32], 1xi32)
        t686 = paddle._C_ops.concat(t685, t451)
        del t685

        # pd_op.conv2d: (-1x960x14x14xf32) <- (-1x2432x14x14xf32, 960x2432x1x1xf32)
        t687 = paddle._C_ops.conv2d(
            t686, t186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t686, t186

        # pd_op.batch_norm_: (-1x960x14x14xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (-1x960x14x14xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t688, t689, t690, t691, t692, t693 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t687,
                t187,
                t188,
                t189,
                t190,
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
        del t687, t190, t189, t188, t187

        # pd_op.relu: (-1x960x14x14xf32) <- (-1x960x14x14xf32)
        t694 = paddle._C_ops.relu(t688)
        del t688

        # pd_op.pool2d: (-1x960x1x1xf32) <- (-1x960x14x14xf32, 2xi64)
        t695 = paddle._C_ops.pool2d(
            t694,
            t462,
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

        # pd_op.conv2d: (-1x960x1x1xf32) <- (-1x960x1x1xf32, 960x960x1x1xf32)
        t696 = paddle._C_ops.conv2d(
            t695, t191, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t191, t695

        # pd_op.reshape: (1x960x1x1xf32) <- (960xf32, 4xi64)
        t697 = paddle._C_ops.reshape(t192, t465)
        del t192

        # pd_op.add: (-1x960x1x1xf32) <- (-1x960x1x1xf32, 1x960x1x1xf32)
        t698 = paddle._C_ops.add(t696, t697)
        del t696, t697

        # pd_op.sigmoid: (-1x960x1x1xf32) <- (-1x960x1x1xf32)
        t699 = paddle._C_ops.sigmoid(t698)
        del t698

        # pd_op.multiply: (-1x960x14x14xf32) <- (-1x960x14x14xf32, -1x960x1x1xf32)
        t700 = paddle._C_ops.multiply(t694, t699)
        del t694, t699

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x960x14x14xf32, 256x960x3x3xf32)
        t701 = paddle._C_ops.conv2d(
            t700, t193, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t193

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t702, t703, t704, t705, t706, t707 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t701,
                t194,
                t195,
                t196,
                t197,
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
        del t701, t197, t196, t195, t194

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t708 = paddle._C_ops.relu(t702)
        del t702

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t709 = paddle._C_ops.conv2d(
            t708, t198, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t198

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t710, t711, t712, t713, t714, t715 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t709,
                t199,
                t200,
                t201,
                t202,
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
        del t709, t202, t201, t200, t199

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t716 = paddle._C_ops.relu(t710)
        del t710

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t717 = paddle._C_ops.conv2d(
            t716, t203, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t203

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t718, t719, t720, t721, t722, t723 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t717,
                t204,
                t205,
                t206,
                t207,
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
        del t717, t207, t206, t205, t204

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t724 = paddle._C_ops.relu(t718)
        del t718

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t725 = paddle._C_ops.conv2d(
            t724, t208, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t208

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t726, t727, t728, t729, t730, t731 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t725,
                t209,
                t210,
                t211,
                t212,
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
        del t725, t212, t211, t210, t209

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t732 = paddle._C_ops.relu(t726)
        del t726

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t733 = paddle._C_ops.conv2d(
            t732, t213, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t213

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t734, t735, t736, t737, t738, t739 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t733,
                t214,
                t215,
                t216,
                t217,
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
        del t733, t217, t216, t215, t214

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t740 = paddle._C_ops.relu(t734)
        del t734

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t741 = paddle._C_ops.conv2d(
            t740, t218, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t218

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t742, t743, t744, t745, t746, t747 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t741,
                t219,
                t220,
                t221,
                t222,
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
        del t741, t222, t221, t220, t219

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t748 = paddle._C_ops.relu(t742)
        del t742

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t749 = paddle._C_ops.conv2d(
            t748, t223, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t223

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t750, t751, t752, t753, t754, t755 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t749,
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
        del t749, t227, t226, t225, t224

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t756 = paddle._C_ops.relu(t750)
        del t750

        # builtin.combine: ([-1x960x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x960x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32)
        t757 = [t700, t708, t716, t724, t732, t740, t748, t756]
        del t708, t716, t724, t732, t740, t748, t756

        # pd_op.concat: (-1x2752x14x14xf32) <- ([-1x960x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32], 1xi32)
        t758 = paddle._C_ops.concat(t757, t451)
        del t757

        # pd_op.conv2d: (-1x960x14x14xf32) <- (-1x2752x14x14xf32, 960x2752x1x1xf32)
        t759 = paddle._C_ops.conv2d(
            t758, t228, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t758, t228

        # pd_op.batch_norm_: (-1x960x14x14xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (-1x960x14x14xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t760, t761, t762, t763, t764, t765 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t759,
                t229,
                t230,
                t231,
                t232,
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
        del t759, t232, t231, t230, t229

        # pd_op.relu: (-1x960x14x14xf32) <- (-1x960x14x14xf32)
        t766 = paddle._C_ops.relu(t760)
        del t760

        # pd_op.pool2d: (-1x960x1x1xf32) <- (-1x960x14x14xf32, 2xi64)
        t767 = paddle._C_ops.pool2d(
            t766,
            t462,
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

        # pd_op.conv2d: (-1x960x1x1xf32) <- (-1x960x1x1xf32, 960x960x1x1xf32)
        t768 = paddle._C_ops.conv2d(
            t767, t233, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t233, t767

        # pd_op.reshape: (1x960x1x1xf32) <- (960xf32, 4xi64)
        t769 = paddle._C_ops.reshape(t234, t465)
        del t234

        # pd_op.add: (-1x960x1x1xf32) <- (-1x960x1x1xf32, 1x960x1x1xf32)
        t770 = paddle._C_ops.add(t768, t769)
        del t768, t769

        # pd_op.sigmoid: (-1x960x1x1xf32) <- (-1x960x1x1xf32)
        t771 = paddle._C_ops.sigmoid(t770)
        del t770

        # pd_op.multiply: (-1x960x14x14xf32) <- (-1x960x14x14xf32, -1x960x1x1xf32)
        t772 = paddle._C_ops.multiply(t766, t771)
        del t766, t771

        # pd_op.add: (-1x960x14x14xf32) <- (-1x960x14x14xf32, -1x960x14x14xf32)
        t773 = paddle._C_ops.add(t772, t700)
        del t700, t772

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x960x14x14xf32, 256x960x3x3xf32)
        t774 = paddle._C_ops.conv2d(
            t773, t235, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t775, t776, t777, t778, t779, t780 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t774,
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
        del t774, t239, t238, t237, t236

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t781 = paddle._C_ops.relu(t775)
        del t775

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t782 = paddle._C_ops.conv2d(
            t781, t240, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t240

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t783, t784, t785, t786, t787, t788 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t782,
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
        del t782, t244, t243, t242, t241

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t789 = paddle._C_ops.relu(t783)
        del t783

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t790 = paddle._C_ops.conv2d(
            t789, t245, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t791, t792, t793, t794, t795, t796 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t790,
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
        del t790, t249, t248, t247, t246

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t797 = paddle._C_ops.relu(t791)
        del t791

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t798 = paddle._C_ops.conv2d(
            t797, t250, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t250

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t799, t800, t801, t802, t803, t804 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t798,
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
        del t798, t254, t253, t252, t251

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t805 = paddle._C_ops.relu(t799)
        del t799

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t806 = paddle._C_ops.conv2d(
            t805, t255, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t807, t808, t809, t810, t811, t812 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t806,
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
        del t806, t259, t258, t257, t256

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t813 = paddle._C_ops.relu(t807)
        del t807

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t814 = paddle._C_ops.conv2d(
            t813, t260, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t260

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t815, t816, t817, t818, t819, t820 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t814,
                t261,
                t262,
                t263,
                t264,
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
        del t814, t264, t263, t262, t261

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t821 = paddle._C_ops.relu(t815)
        del t815

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t822 = paddle._C_ops.conv2d(
            t821, t265, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t265

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t823, t824, t825, t826, t827, t828 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t822,
                t266,
                t267,
                t268,
                t269,
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
        del t822, t268, t267, t266, t269

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t829 = paddle._C_ops.relu(t823)
        del t823

        # builtin.combine: ([-1x960x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x960x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32)
        t830 = [t773, t781, t789, t797, t805, t813, t821, t829]
        del t781, t789, t797, t805, t813, t821, t829

        # pd_op.concat: (-1x2752x14x14xf32) <- ([-1x960x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32], 1xi32)
        t831 = paddle._C_ops.concat(t830, t451)
        del t830

        # pd_op.conv2d: (-1x960x14x14xf32) <- (-1x2752x14x14xf32, 960x2752x1x1xf32)
        t832 = paddle._C_ops.conv2d(
            t831, t270, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t831, t270

        # pd_op.batch_norm_: (-1x960x14x14xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (-1x960x14x14xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t833, t834, t835, t836, t837, t838 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t832,
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
        del t832, t274, t273, t272, t271

        # pd_op.relu: (-1x960x14x14xf32) <- (-1x960x14x14xf32)
        t839 = paddle._C_ops.relu(t833)
        del t833

        # pd_op.pool2d: (-1x960x1x1xf32) <- (-1x960x14x14xf32, 2xi64)
        t840 = paddle._C_ops.pool2d(
            t839,
            t462,
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

        # pd_op.conv2d: (-1x960x1x1xf32) <- (-1x960x1x1xf32, 960x960x1x1xf32)
        t841 = paddle._C_ops.conv2d(
            t840, t275, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t275, t840

        # pd_op.reshape: (1x960x1x1xf32) <- (960xf32, 4xi64)
        t842 = paddle._C_ops.reshape(t276, t465)
        del t276

        # pd_op.add: (-1x960x1x1xf32) <- (-1x960x1x1xf32, 1x960x1x1xf32)
        t843 = paddle._C_ops.add(t841, t842)
        del t841, t842

        # pd_op.sigmoid: (-1x960x1x1xf32) <- (-1x960x1x1xf32)
        t844 = paddle._C_ops.sigmoid(t843)
        del t843

        # pd_op.multiply: (-1x960x14x14xf32) <- (-1x960x14x14xf32, -1x960x1x1xf32)
        t845 = paddle._C_ops.multiply(t839, t844)
        del t839, t844

        # pd_op.add: (-1x960x14x14xf32) <- (-1x960x14x14xf32, -1x960x14x14xf32)
        t846 = paddle._C_ops.add(t845, t773)
        del t773, t845

        # pd_op.depthwise_conv2d: (-1x960x7x7xf32) <- (-1x960x14x14xf32, 960x1x3x3xf32)
        t847 = paddle._C_ops.depthwise_conv2d(
            t846, t277, [2, 2], [1, 1], "EXPLICIT", 960, [1, 1], "NCHW"
        )
        del t846, t277

        # pd_op.batch_norm_: (-1x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (-1x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        t848, t849, t850, t851, t852, t853 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t847,
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
        del t847, t281, t280, t279, t278

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x960x7x7xf32, 288x960x3x3xf32)
        t854 = paddle._C_ops.conv2d(
            t848, t282, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t282

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t855, t856, t857, t858, t859, t860 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t854,
                t283,
                t284,
                t285,
                t286,
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
        del t854, t286, t285, t284, t283

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t861 = paddle._C_ops.relu(t855)
        del t855

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t862 = paddle._C_ops.conv2d(
            t861, t287, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t287

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t863, t864, t865, t866, t867, t868 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t862,
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
        del t862, t291, t290, t289, t288

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t869 = paddle._C_ops.relu(t863)
        del t863

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t870 = paddle._C_ops.conv2d(
            t869, t292, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t292

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t871, t872, t873, t874, t875, t876 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t870,
                t293,
                t294,
                t295,
                t296,
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
        del t870, t296, t295, t294, t293

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t877 = paddle._C_ops.relu(t871)
        del t871

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t878 = paddle._C_ops.conv2d(
            t877, t297, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t297

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t879, t880, t881, t882, t883, t884 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t878,
                t298,
                t299,
                t300,
                t301,
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
        del t878, t301, t300, t299, t298

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t885 = paddle._C_ops.relu(t879)
        del t879

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t886 = paddle._C_ops.conv2d(
            t885, t302, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t302

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t887, t888, t889, t890, t891, t892 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t886,
                t303,
                t304,
                t305,
                t306,
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
        del t886, t306, t305, t304, t303

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t893 = paddle._C_ops.relu(t887)
        del t887

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t894 = paddle._C_ops.conv2d(
            t893, t307, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t307

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t895, t896, t897, t898, t899, t900 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t894,
                t308,
                t309,
                t310,
                t311,
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
        del t894, t311, t310, t309, t308

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t901 = paddle._C_ops.relu(t895)
        del t895

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t902 = paddle._C_ops.conv2d(
            t901, t312, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t312

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t903, t904, t905, t906, t907, t908 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t902,
                t313,
                t314,
                t315,
                t316,
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
        del t902, t316, t315, t314, t313

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t909 = paddle._C_ops.relu(t903)
        del t903

        # builtin.combine: ([-1x960x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32]) <- (-1x960x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32)
        t910 = [t848, t861, t869, t877, t885, t893, t901, t909]
        del t848, t861, t869, t877, t885, t893, t901, t909

        # pd_op.concat: (-1x2976x7x7xf32) <- ([-1x960x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32], 1xi32)
        t911 = paddle._C_ops.concat(t910, t451)
        del t910

        # pd_op.conv2d: (-1x1280x7x7xf32) <- (-1x2976x7x7xf32, 1280x2976x1x1xf32)
        t912 = paddle._C_ops.conv2d(
            t911, t317, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t911, t317

        # pd_op.batch_norm_: (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32, -1xui8) <- (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        t913, t914, t915, t916, t917, t918 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t912,
                t318,
                t319,
                t320,
                t321,
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
        del t912, t321, t320, t319, t318

        # pd_op.relu: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32)
        t919 = paddle._C_ops.relu(t913)
        del t913

        # pd_op.pool2d: (-1x1280x1x1xf32) <- (-1x1280x7x7xf32, 2xi64)
        t920 = paddle._C_ops.pool2d(
            t919,
            t462,
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

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32, 1280x1280x1x1xf32)
        t921 = paddle._C_ops.conv2d(
            t920, t322, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t322, t920

        # pd_op.reshape: (1x1280x1x1xf32) <- (1280xf32, 4xi64)
        t922 = paddle._C_ops.reshape(t323, t465)
        del t323

        # pd_op.add: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32, 1x1280x1x1xf32)
        t923 = paddle._C_ops.add(t921, t922)
        del t921, t922

        # pd_op.sigmoid: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        t924 = paddle._C_ops.sigmoid(t923)
        del t923

        # pd_op.multiply: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32, -1x1280x1x1xf32)
        t925 = paddle._C_ops.multiply(t919, t924)
        del t919, t924

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x1280x7x7xf32, 288x1280x3x3xf32)
        t926 = paddle._C_ops.conv2d(
            t925, t324, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t324

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t927, t928, t929, t930, t931, t932 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t926,
                t325,
                t326,
                t327,
                t328,
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
        del t926, t328, t327, t326, t325

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t933 = paddle._C_ops.relu(t927)
        del t927

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t934 = paddle._C_ops.conv2d(
            t933, t329, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t329

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t935, t936, t937, t938, t939, t940 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t934,
                t330,
                t331,
                t332,
                t333,
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
        del t934, t333, t332, t331, t330

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t941 = paddle._C_ops.relu(t935)
        del t935

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t942 = paddle._C_ops.conv2d(
            t941, t334, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t334

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t943, t944, t945, t946, t947, t948 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t942,
                t335,
                t336,
                t337,
                t338,
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
        del t942, t338, t337, t336, t335

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t949 = paddle._C_ops.relu(t943)
        del t943

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t950 = paddle._C_ops.conv2d(
            t949, t339, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t339

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t951, t952, t953, t954, t955, t956 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t950,
                t340,
                t341,
                t342,
                t343,
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
        del t950, t343, t342, t341, t340

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t957 = paddle._C_ops.relu(t951)
        del t951

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t958 = paddle._C_ops.conv2d(
            t957, t344, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t344

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t959, t960, t961, t962, t963, t964 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t958,
                t345,
                t346,
                t347,
                t348,
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
        del t958, t348, t347, t346, t345

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t965 = paddle._C_ops.relu(t959)
        del t959

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t966 = paddle._C_ops.conv2d(
            t965, t349, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t349

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t967, t968, t969, t970, t971, t972 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t966,
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
        del t966, t353, t352, t351, t350

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t973 = paddle._C_ops.relu(t967)
        del t967

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t974 = paddle._C_ops.conv2d(
            t973, t354, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t354

        # pd_op.batch_norm_: (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t975, t976, t977, t978, t979, t980 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t974,
                t355,
                t356,
                t357,
                t358,
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
        del t974, t358, t357, t356, t355

        # pd_op.relu: (-1x288x7x7xf32) <- (-1x288x7x7xf32)
        t981 = paddle._C_ops.relu(t975)
        del t975

        # builtin.combine: ([-1x1280x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32]) <- (-1x1280x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32)
        t982 = [t925, t933, t941, t949, t957, t965, t973, t981]
        del t933, t941, t949, t957, t965, t973, t981

        # pd_op.concat: (-1x3296x7x7xf32) <- ([-1x1280x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32, -1x288x7x7xf32], 1xi32)
        t983 = paddle._C_ops.concat(t982, t451)
        del t982, t451

        # pd_op.conv2d: (-1x1280x7x7xf32) <- (-1x3296x7x7xf32, 1280x3296x1x1xf32)
        t984 = paddle._C_ops.conv2d(
            t983, t359, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t983, t359

        # pd_op.batch_norm_: (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32, -1xui8) <- (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        t985, t986, t987, t988, t989, t990 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t984,
                t360,
                t361,
                t362,
                t363,
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
        del t984, t363, t362, t361, t360

        # pd_op.relu: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32)
        t991 = paddle._C_ops.relu(t985)
        del t985

        # pd_op.pool2d: (-1x1280x1x1xf32) <- (-1x1280x7x7xf32, 2xi64)
        t992 = paddle._C_ops.pool2d(
            t991,
            t462,
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

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32, 1280x1280x1x1xf32)
        t993 = paddle._C_ops.conv2d(
            t992, t364, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t364, t992

        # pd_op.reshape: (1x1280x1x1xf32) <- (1280xf32, 4xi64)
        t994 = paddle._C_ops.reshape(t365, t465)
        del t465, t365

        # pd_op.add: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32, 1x1280x1x1xf32)
        t995 = paddle._C_ops.add(t993, t994)
        del t993, t994

        # pd_op.sigmoid: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        t996 = paddle._C_ops.sigmoid(t995)
        del t995

        # pd_op.multiply: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32, -1x1280x1x1xf32)
        t997 = paddle._C_ops.multiply(t991, t996)
        del t991, t996

        # pd_op.add: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32, -1x1280x7x7xf32)
        t998 = paddle._C_ops.add(t997, t925)
        del t925, t997

        # pd_op.pool2d: (-1x1280x1x1xf32) <- (-1x1280x7x7xf32, 2xi64)
        t999 = paddle._C_ops.pool2d(
            t998,
            t462,
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
        del t998, t462

        # pd_op.conv2d: (-1x2048x1x1xf32) <- (-1x1280x1x1xf32, 2048x1280x1x1xf32)
        t1000 = paddle._C_ops.conv2d(
            t999, t366, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t366, t999

        # pd_op.relu: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32)
        t1001 = paddle._C_ops.relu(t1000)
        del t1000

        # pd_op.full: (1xf32) <- ()
        t1002 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x2048x1x1xf32, -1x2048x1x1xui8) <- (-1x2048x1x1xf32, None, 1xf32)
        t1003, t1004 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                t1001, None, t1002, True, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t1002, t1001

        # pd_op.flatten: (-1x2048xf32) <- (-1x2048x1x1xf32)
        t1005 = paddle._C_ops.flatten(t1003, 1, 3)
        del t1003

        # pd_op.matmul: (-1x102xf32) <- (-1x2048xf32, 2048x102xf32)
        t1006 = paddle._C_ops.matmul(t1005, t367, False, False)
        del t1005, t367

        return t1006
