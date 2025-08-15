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
    ):
        t269 = None
        # pd_op.conv2d: (-1x16x112x112xf32) <- (-1x3x224x224xf32, 16x3x3x3xf32)
        t270 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t271, t272, t273, t274, t275, t276 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t270,
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

        # pd_op.hardswish: (-1x16x112x112xf32) <- (-1x16x112x112xf32)
        t277 = paddle._C_ops.hardswish(t271)

        # pd_op.conv2d: (-1x16x112x112xf32) <- (-1x16x112x112xf32, 16x16x1x1xf32)
        t278 = paddle._C_ops.conv2d(
            t277, t5, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t279, t280, t281, t282, t283, t284 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t278,
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

        # pd_op.relu: (-1x16x112x112xf32) <- (-1x16x112x112xf32)
        t285 = paddle._C_ops.relu(t279)
        del t279

        # pd_op.depthwise_conv2d: (-1x16x112x112xf32) <- (-1x16x112x112xf32, 16x1x3x3xf32)
        t286 = paddle._C_ops.depthwise_conv2d(
            t285, t10, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t287, t288, t289, t290, t291, t292 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t286,
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

        # pd_op.relu: (-1x16x112x112xf32) <- (-1x16x112x112xf32)
        t293 = paddle._C_ops.relu(t287)
        del t287

        # pd_op.conv2d: (-1x16x112x112xf32) <- (-1x16x112x112xf32, 16x16x1x1xf32)
        t294 = paddle._C_ops.conv2d(
            t293, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t295, t296, t297, t298, t299, t300 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t294,
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

        # pd_op.add: (-1x16x112x112xf32) <- (-1x16x112x112xf32, -1x16x112x112xf32)
        t301 = paddle._C_ops.add(t277, t295)

        # pd_op.conv2d: (-1x48x112x112xf32) <- (-1x16x112x112xf32, 48x16x1x1xf32)
        t302 = paddle._C_ops.conv2d(
            t301, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t303, t304, t305, t306, t307, t308 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t302,
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

        # pd_op.relu: (-1x48x112x112xf32) <- (-1x48x112x112xf32)
        t309 = paddle._C_ops.relu(t303)
        del t303

        # pd_op.depthwise_conv2d: (-1x48x56x56xf32) <- (-1x48x112x112xf32, 48x1x3x3xf32)
        t310 = paddle._C_ops.depthwise_conv2d(
            t309, t25, [2, 2], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t311, t312, t313, t314, t315, t316 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t310,
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

        # pd_op.relu: (-1x48x56x56xf32) <- (-1x48x56x56xf32)
        t317 = paddle._C_ops.relu(t311)
        del t311

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x48x56x56xf32, 24x48x1x1xf32)
        t318 = paddle._C_ops.conv2d(
            t317, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t319, t320, t321, t322, t323, t324 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t318,
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

        # pd_op.conv2d: (-1x56x56x56xf32) <- (-1x24x56x56xf32, 56x24x1x1xf32)
        t325 = paddle._C_ops.conv2d(
            t319, t35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x56x56x56xf32, 56xf32, 56xf32, 56xf32, 56xf32, -1xui8) <- (-1x56x56x56xf32, 56xf32, 56xf32, 56xf32, 56xf32)
        t326, t327, t328, t329, t330, t331 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t325,
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

        # pd_op.relu: (-1x56x56x56xf32) <- (-1x56x56x56xf32)
        t332 = paddle._C_ops.relu(t326)
        del t326

        # pd_op.depthwise_conv2d: (-1x56x56x56xf32) <- (-1x56x56x56xf32, 56x1x3x3xf32)
        t333 = paddle._C_ops.depthwise_conv2d(
            t332, t40, [1, 1], [1, 1], "EXPLICIT", 56, [1, 1], "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (-1x56x56x56xf32, 56xf32, 56xf32, 56xf32, 56xf32, -1xui8) <- (-1x56x56x56xf32, 56xf32, 56xf32, 56xf32, 56xf32)
        t334, t335, t336, t337, t338, t339 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t333,
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

        # pd_op.relu: (-1x56x56x56xf32) <- (-1x56x56x56xf32)
        t340 = paddle._C_ops.relu(t334)
        del t334

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x56x56x56xf32, 24x56x1x1xf32)
        t341 = paddle._C_ops.conv2d(
            t340, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t342, t343, t344, t345, t346, t347 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t341,
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

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, -1x24x56x56xf32)
        t348 = paddle._C_ops.add(t319, t342)

        # pd_op.conv2d: (-1x56x56x56xf32) <- (-1x24x56x56xf32, 56x24x1x1xf32)
        t349 = paddle._C_ops.conv2d(
            t348, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (-1x56x56x56xf32, 56xf32, 56xf32, 56xf32, 56xf32, -1xui8) <- (-1x56x56x56xf32, 56xf32, 56xf32, 56xf32, 56xf32)
        t350, t351, t352, t353, t354, t355 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t349,
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

        # pd_op.relu: (-1x56x56x56xf32) <- (-1x56x56x56xf32)
        t356 = paddle._C_ops.relu(t350)
        del t350

        # pd_op.depthwise_conv2d: (-1x56x28x28xf32) <- (-1x56x56x56xf32, 56x1x5x5xf32)
        t357 = paddle._C_ops.depthwise_conv2d(
            t356, t55, [2, 2], [2, 2], "EXPLICIT", 56, [1, 1], "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (-1x56x28x28xf32, 56xf32, 56xf32, 56xf32, 56xf32, -1xui8) <- (-1x56x28x28xf32, 56xf32, 56xf32, 56xf32, 56xf32)
        t358, t359, t360, t361, t362, t363 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t357,
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

        # pd_op.relu: (-1x56x28x28xf32) <- (-1x56x28x28xf32)
        t364 = paddle._C_ops.relu(t358)
        del t358

        # pd_op.full_int_array: (2xi64) <- ()
        t365 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        t366 = t365

        # pd_op.assign: (2xi64) <- (2xi64)
        t367 = t365

        # pd_op.assign: (2xi64) <- (2xi64)
        t368 = t365

        # pd_op.assign: (2xi64) <- (2xi64)
        t369 = t365

        # pd_op.assign: (2xi64) <- (2xi64)
        t370 = t365

        # pd_op.assign: (2xi64) <- (2xi64)
        t371 = t365

        # pd_op.assign: (2xi64) <- (2xi64)
        t372 = t365

        # pd_op.assign: (2xi64) <- (2xi64)
        t373 = t365

        # pd_op.pool2d: (-1x56x1x1xf32) <- (-1x56x28x28xf32, 2xi64)
        t374 = paddle._C_ops.pool2d(
            t364,
            t365,
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

        # pd_op.conv2d: (-1x14x1x1xf32) <- (-1x56x1x1xf32, 14x56x1x1xf32)
        t375 = paddle._C_ops.conv2d(
            t374, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.full_int_array: (4xi64) <- ()
        t376 = [1, -1, 1, 1]

        # pd_op.reshape: (1x14x1x1xf32) <- (14xf32, 4xi64)
        t377 = paddle._C_ops.reshape(t61, t376)
        del t61

        # pd_op.add: (-1x14x1x1xf32) <- (-1x14x1x1xf32, 1x14x1x1xf32)
        t378 = paddle._C_ops.add(t375, t377)

        # pd_op.relu: (-1x14x1x1xf32) <- (-1x14x1x1xf32)
        t379 = paddle._C_ops.relu(t378)
        del t378

        # pd_op.conv2d: (-1x56x1x1xf32) <- (-1x14x1x1xf32, 56x14x1x1xf32)
        t380 = paddle._C_ops.conv2d(
            t379, t62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t62

        # pd_op.reshape: (1x56x1x1xf32) <- (56xf32, 4xi64)
        t381 = paddle._C_ops.reshape(t63, t376)
        del t63

        # pd_op.add: (-1x56x1x1xf32) <- (-1x56x1x1xf32, 1x56x1x1xf32)
        t382 = paddle._C_ops.add(t380, t381)

        # pd_op.hardsigmoid: (-1x56x1x1xf32) <- (-1x56x1x1xf32)
        t383 = paddle._C_ops.hardsigmoid(t382, float("0.2"), float("0.5"))
        del t382

        # pd_op.multiply: (-1x56x28x28xf32) <- (-1x56x28x28xf32, -1x56x1x1xf32)
        t384 = paddle._C_ops.multiply(t364, t383)

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x56x28x28xf32, 32x56x1x1xf32)
        t385 = paddle._C_ops.conv2d(
            t384, t64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t64

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t386, t387, t388, t389, t390, t391 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t385,
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

        # pd_op.conv2d: (-1x88x28x28xf32) <- (-1x32x28x28xf32, 88x32x1x1xf32)
        t392 = paddle._C_ops.conv2d(
            t386, t69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t69

        # pd_op.batch_norm_: (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t393, t394, t395, t396, t397, t398 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t392,
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

        # pd_op.relu: (-1x88x28x28xf32) <- (-1x88x28x28xf32)
        t399 = paddle._C_ops.relu(t393)
        del t393

        # pd_op.depthwise_conv2d: (-1x88x28x28xf32) <- (-1x88x28x28xf32, 88x1x5x5xf32)
        t400 = paddle._C_ops.depthwise_conv2d(
            t399, t74, [1, 1], [2, 2], "EXPLICIT", 88, [1, 1], "NCHW"
        )
        del t74

        # pd_op.batch_norm_: (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t401, t402, t403, t404, t405, t406 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t400,
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

        # pd_op.relu: (-1x88x28x28xf32) <- (-1x88x28x28xf32)
        t407 = paddle._C_ops.relu(t401)
        del t401

        # pd_op.pool2d: (-1x88x1x1xf32) <- (-1x88x28x28xf32, 2xi64)
        t408 = paddle._C_ops.pool2d(
            t407,
            t365,
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

        # pd_op.conv2d: (-1x22x1x1xf32) <- (-1x88x1x1xf32, 22x88x1x1xf32)
        t409 = paddle._C_ops.conv2d(
            t408, t79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t79

        # pd_op.reshape: (1x22x1x1xf32) <- (22xf32, 4xi64)
        t410 = paddle._C_ops.reshape(t80, t376)
        del t80

        # pd_op.add: (-1x22x1x1xf32) <- (-1x22x1x1xf32, 1x22x1x1xf32)
        t411 = paddle._C_ops.add(t409, t410)

        # pd_op.relu: (-1x22x1x1xf32) <- (-1x22x1x1xf32)
        t412 = paddle._C_ops.relu(t411)
        del t411

        # pd_op.conv2d: (-1x88x1x1xf32) <- (-1x22x1x1xf32, 88x22x1x1xf32)
        t413 = paddle._C_ops.conv2d(
            t412, t81, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t81

        # pd_op.reshape: (1x88x1x1xf32) <- (88xf32, 4xi64)
        t414 = paddle._C_ops.reshape(t82, t376)
        del t82

        # pd_op.add: (-1x88x1x1xf32) <- (-1x88x1x1xf32, 1x88x1x1xf32)
        t415 = paddle._C_ops.add(t413, t414)

        # pd_op.hardsigmoid: (-1x88x1x1xf32) <- (-1x88x1x1xf32)
        t416 = paddle._C_ops.hardsigmoid(t415, float("0.2"), float("0.5"))
        del t415

        # pd_op.multiply: (-1x88x28x28xf32) <- (-1x88x28x28xf32, -1x88x1x1xf32)
        t417 = paddle._C_ops.multiply(t407, t416)

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x88x28x28xf32, 32x88x1x1xf32)
        t418 = paddle._C_ops.conv2d(
            t417, t83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t83

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t419, t420, t421, t422, t423, t424 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t418,
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

        # pd_op.add: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x28x28xf32)
        t425 = paddle._C_ops.add(t386, t419)

        # pd_op.conv2d: (-1x88x28x28xf32) <- (-1x32x28x28xf32, 88x32x1x1xf32)
        t426 = paddle._C_ops.conv2d(
            t425, t88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t88

        # pd_op.batch_norm_: (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t427, t428, t429, t430, t431, t432 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t426,
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

        # pd_op.relu: (-1x88x28x28xf32) <- (-1x88x28x28xf32)
        t433 = paddle._C_ops.relu(t427)
        del t427

        # pd_op.depthwise_conv2d: (-1x88x28x28xf32) <- (-1x88x28x28xf32, 88x1x5x5xf32)
        t434 = paddle._C_ops.depthwise_conv2d(
            t433, t93, [1, 1], [2, 2], "EXPLICIT", 88, [1, 1], "NCHW"
        )
        del t93

        # pd_op.batch_norm_: (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t435, t436, t437, t438, t439, t440 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t434,
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

        # pd_op.relu: (-1x88x28x28xf32) <- (-1x88x28x28xf32)
        t441 = paddle._C_ops.relu(t435)
        del t435

        # pd_op.pool2d: (-1x88x1x1xf32) <- (-1x88x28x28xf32, 2xi64)
        t442 = paddle._C_ops.pool2d(
            t441,
            t365,
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

        # pd_op.conv2d: (-1x22x1x1xf32) <- (-1x88x1x1xf32, 22x88x1x1xf32)
        t443 = paddle._C_ops.conv2d(
            t442, t98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t98

        # pd_op.reshape: (1x22x1x1xf32) <- (22xf32, 4xi64)
        t444 = paddle._C_ops.reshape(t99, t376)
        del t99

        # pd_op.add: (-1x22x1x1xf32) <- (-1x22x1x1xf32, 1x22x1x1xf32)
        t445 = paddle._C_ops.add(t443, t444)

        # pd_op.relu: (-1x22x1x1xf32) <- (-1x22x1x1xf32)
        t446 = paddle._C_ops.relu(t445)
        del t445

        # pd_op.conv2d: (-1x88x1x1xf32) <- (-1x22x1x1xf32, 88x22x1x1xf32)
        t447 = paddle._C_ops.conv2d(
            t446, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.reshape: (1x88x1x1xf32) <- (88xf32, 4xi64)
        t448 = paddle._C_ops.reshape(t101, t376)
        del t101

        # pd_op.add: (-1x88x1x1xf32) <- (-1x88x1x1xf32, 1x88x1x1xf32)
        t449 = paddle._C_ops.add(t447, t448)

        # pd_op.hardsigmoid: (-1x88x1x1xf32) <- (-1x88x1x1xf32)
        t450 = paddle._C_ops.hardsigmoid(t449, float("0.2"), float("0.5"))
        del t449

        # pd_op.multiply: (-1x88x28x28xf32) <- (-1x88x28x28xf32, -1x88x1x1xf32)
        t451 = paddle._C_ops.multiply(t441, t450)

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x88x28x28xf32, 32x88x1x1xf32)
        t452 = paddle._C_ops.conv2d(
            t451, t102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t102

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t453, t454, t455, t456, t457, t458 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t452,
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

        # pd_op.add: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x28x28xf32)
        t459 = paddle._C_ops.add(t425, t453)

        # pd_op.conv2d: (-1x184x28x28xf32) <- (-1x32x28x28xf32, 184x32x1x1xf32)
        t460 = paddle._C_ops.conv2d(
            t459, t107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t107

        # pd_op.batch_norm_: (-1x184x28x28xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (-1x184x28x28xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        t461, t462, t463, t464, t465, t466 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t460,
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

        # pd_op.hardswish: (-1x184x28x28xf32) <- (-1x184x28x28xf32)
        t467 = paddle._C_ops.hardswish(t461)

        # pd_op.depthwise_conv2d: (-1x184x14x14xf32) <- (-1x184x28x28xf32, 184x1x3x3xf32)
        t468 = paddle._C_ops.depthwise_conv2d(
            t467, t112, [2, 2], [1, 1], "EXPLICIT", 184, [1, 1], "NCHW"
        )
        del t112

        # pd_op.batch_norm_: (-1x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (-1x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        t469, t470, t471, t472, t473, t474 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t468,
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

        # pd_op.hardswish: (-1x184x14x14xf32) <- (-1x184x14x14xf32)
        t475 = paddle._C_ops.hardswish(t469)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x184x14x14xf32, 64x184x1x1xf32)
        t476 = paddle._C_ops.conv2d(
            t475, t117, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t117

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t477, t478, t479, t480, t481, t482 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t476,
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

        # pd_op.conv2d: (-1x152x14x14xf32) <- (-1x64x14x14xf32, 152x64x1x1xf32)
        t483 = paddle._C_ops.conv2d(
            t477, t122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t122

        # pd_op.batch_norm_: (-1x152x14x14xf32, 152xf32, 152xf32, 152xf32, 152xf32, -1xui8) <- (-1x152x14x14xf32, 152xf32, 152xf32, 152xf32, 152xf32)
        t484, t485, t486, t487, t488, t489 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t483,
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

        # pd_op.hardswish: (-1x152x14x14xf32) <- (-1x152x14x14xf32)
        t490 = paddle._C_ops.hardswish(t484)

        # pd_op.depthwise_conv2d: (-1x152x14x14xf32) <- (-1x152x14x14xf32, 152x1x3x3xf32)
        t491 = paddle._C_ops.depthwise_conv2d(
            t490, t127, [1, 1], [1, 1], "EXPLICIT", 152, [1, 1], "NCHW"
        )
        del t127

        # pd_op.batch_norm_: (-1x152x14x14xf32, 152xf32, 152xf32, 152xf32, 152xf32, -1xui8) <- (-1x152x14x14xf32, 152xf32, 152xf32, 152xf32, 152xf32)
        t492, t493, t494, t495, t496, t497 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t491,
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

        # pd_op.hardswish: (-1x152x14x14xf32) <- (-1x152x14x14xf32)
        t498 = paddle._C_ops.hardswish(t492)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x152x14x14xf32, 64x152x1x1xf32)
        t499 = paddle._C_ops.conv2d(
            t498, t132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t132

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t500, t501, t502, t503, t504, t505 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t499,
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

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        t506 = paddle._C_ops.add(t477, t500)

        # pd_op.conv2d: (-1x136x14x14xf32) <- (-1x64x14x14xf32, 136x64x1x1xf32)
        t507 = paddle._C_ops.conv2d(
            t506, t137, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t137

        # pd_op.batch_norm_: (-1x136x14x14xf32, 136xf32, 136xf32, 136xf32, 136xf32, -1xui8) <- (-1x136x14x14xf32, 136xf32, 136xf32, 136xf32, 136xf32)
        t508, t509, t510, t511, t512, t513 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t507,
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

        # pd_op.hardswish: (-1x136x14x14xf32) <- (-1x136x14x14xf32)
        t514 = paddle._C_ops.hardswish(t508)

        # pd_op.depthwise_conv2d: (-1x136x14x14xf32) <- (-1x136x14x14xf32, 136x1x3x3xf32)
        t515 = paddle._C_ops.depthwise_conv2d(
            t514, t142, [1, 1], [1, 1], "EXPLICIT", 136, [1, 1], "NCHW"
        )
        del t142

        # pd_op.batch_norm_: (-1x136x14x14xf32, 136xf32, 136xf32, 136xf32, 136xf32, -1xui8) <- (-1x136x14x14xf32, 136xf32, 136xf32, 136xf32, 136xf32)
        t516, t517, t518, t519, t520, t521 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t515,
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

        # pd_op.hardswish: (-1x136x14x14xf32) <- (-1x136x14x14xf32)
        t522 = paddle._C_ops.hardswish(t516)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x136x14x14xf32, 64x136x1x1xf32)
        t523 = paddle._C_ops.conv2d(
            t522, t147, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t147

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t524, t525, t526, t527, t528, t529 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t523,
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

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        t530 = paddle._C_ops.add(t506, t524)

        # pd_op.conv2d: (-1x136x14x14xf32) <- (-1x64x14x14xf32, 136x64x1x1xf32)
        t531 = paddle._C_ops.conv2d(
            t530, t152, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t152

        # pd_op.batch_norm_: (-1x136x14x14xf32, 136xf32, 136xf32, 136xf32, 136xf32, -1xui8) <- (-1x136x14x14xf32, 136xf32, 136xf32, 136xf32, 136xf32)
        t532, t533, t534, t535, t536, t537 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t531,
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

        # pd_op.hardswish: (-1x136x14x14xf32) <- (-1x136x14x14xf32)
        t538 = paddle._C_ops.hardswish(t532)

        # pd_op.depthwise_conv2d: (-1x136x14x14xf32) <- (-1x136x14x14xf32, 136x1x3x3xf32)
        t539 = paddle._C_ops.depthwise_conv2d(
            t538, t157, [1, 1], [1, 1], "EXPLICIT", 136, [1, 1], "NCHW"
        )
        del t157

        # pd_op.batch_norm_: (-1x136x14x14xf32, 136xf32, 136xf32, 136xf32, 136xf32, -1xui8) <- (-1x136x14x14xf32, 136xf32, 136xf32, 136xf32, 136xf32)
        t540, t541, t542, t543, t544, t545 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t539,
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

        # pd_op.hardswish: (-1x136x14x14xf32) <- (-1x136x14x14xf32)
        t546 = paddle._C_ops.hardswish(t540)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x136x14x14xf32, 64x136x1x1xf32)
        t547 = paddle._C_ops.conv2d(
            t546, t162, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t162

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t548, t549, t550, t551, t552, t553 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t547,
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

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        t554 = paddle._C_ops.add(t530, t548)

        # pd_op.conv2d: (-1x360x14x14xf32) <- (-1x64x14x14xf32, 360x64x1x1xf32)
        t555 = paddle._C_ops.conv2d(
            t554, t167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t167

        # pd_op.batch_norm_: (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32, -1xui8) <- (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        t556, t557, t558, t559, t560, t561 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t555,
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
        del t169, t168, t171, t170

        # pd_op.hardswish: (-1x360x14x14xf32) <- (-1x360x14x14xf32)
        t562 = paddle._C_ops.hardswish(t556)

        # pd_op.depthwise_conv2d: (-1x360x14x14xf32) <- (-1x360x14x14xf32, 360x1x3x3xf32)
        t563 = paddle._C_ops.depthwise_conv2d(
            t562, t172, [1, 1], [1, 1], "EXPLICIT", 360, [1, 1], "NCHW"
        )
        del t172

        # pd_op.batch_norm_: (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32, -1xui8) <- (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        t564, t565, t566, t567, t568, t569 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t563,
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

        # pd_op.hardswish: (-1x360x14x14xf32) <- (-1x360x14x14xf32)
        t570 = paddle._C_ops.hardswish(t564)

        # pd_op.pool2d: (-1x360x1x1xf32) <- (-1x360x14x14xf32, 2xi64)
        t571 = paddle._C_ops.pool2d(
            t570,
            t365,
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

        # pd_op.conv2d: (-1x90x1x1xf32) <- (-1x360x1x1xf32, 90x360x1x1xf32)
        t572 = paddle._C_ops.conv2d(
            t571, t177, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t177

        # pd_op.reshape: (1x90x1x1xf32) <- (90xf32, 4xi64)
        t573 = paddle._C_ops.reshape(t178, t376)
        del t178

        # pd_op.add: (-1x90x1x1xf32) <- (-1x90x1x1xf32, 1x90x1x1xf32)
        t574 = paddle._C_ops.add(t572, t573)

        # pd_op.relu: (-1x90x1x1xf32) <- (-1x90x1x1xf32)
        t575 = paddle._C_ops.relu(t574)
        del t574

        # pd_op.conv2d: (-1x360x1x1xf32) <- (-1x90x1x1xf32, 360x90x1x1xf32)
        t576 = paddle._C_ops.conv2d(
            t575, t179, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t179

        # pd_op.reshape: (1x360x1x1xf32) <- (360xf32, 4xi64)
        t577 = paddle._C_ops.reshape(t180, t376)
        del t180

        # pd_op.add: (-1x360x1x1xf32) <- (-1x360x1x1xf32, 1x360x1x1xf32)
        t578 = paddle._C_ops.add(t576, t577)

        # pd_op.hardsigmoid: (-1x360x1x1xf32) <- (-1x360x1x1xf32)
        t579 = paddle._C_ops.hardsigmoid(t578, float("0.2"), float("0.5"))
        del t578

        # pd_op.multiply: (-1x360x14x14xf32) <- (-1x360x14x14xf32, -1x360x1x1xf32)
        t580 = paddle._C_ops.multiply(t570, t579)

        # pd_op.conv2d: (-1x88x14x14xf32) <- (-1x360x14x14xf32, 88x360x1x1xf32)
        t581 = paddle._C_ops.conv2d(
            t580, t181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t181

        # pd_op.batch_norm_: (-1x88x14x14xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x14x14xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t582, t583, t584, t585, t586, t587 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t581,
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

        # pd_op.conv2d: (-1x504x14x14xf32) <- (-1x88x14x14xf32, 504x88x1x1xf32)
        t588 = paddle._C_ops.conv2d(
            t582, t186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t186

        # pd_op.batch_norm_: (-1x504x14x14xf32, 504xf32, 504xf32, 504xf32, 504xf32, -1xui8) <- (-1x504x14x14xf32, 504xf32, 504xf32, 504xf32, 504xf32)
        t589, t590, t591, t592, t593, t594 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t588,
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

        # pd_op.hardswish: (-1x504x14x14xf32) <- (-1x504x14x14xf32)
        t595 = paddle._C_ops.hardswish(t589)

        # pd_op.depthwise_conv2d: (-1x504x14x14xf32) <- (-1x504x14x14xf32, 504x1x3x3xf32)
        t596 = paddle._C_ops.depthwise_conv2d(
            t595, t191, [1, 1], [1, 1], "EXPLICIT", 504, [1, 1], "NCHW"
        )
        del t191

        # pd_op.batch_norm_: (-1x504x14x14xf32, 504xf32, 504xf32, 504xf32, 504xf32, -1xui8) <- (-1x504x14x14xf32, 504xf32, 504xf32, 504xf32, 504xf32)
        t597, t598, t599, t600, t601, t602 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t596,
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

        # pd_op.hardswish: (-1x504x14x14xf32) <- (-1x504x14x14xf32)
        t603 = paddle._C_ops.hardswish(t597)

        # pd_op.pool2d: (-1x504x1x1xf32) <- (-1x504x14x14xf32, 2xi64)
        t604 = paddle._C_ops.pool2d(
            t603,
            t365,
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

        # pd_op.conv2d: (-1x126x1x1xf32) <- (-1x504x1x1xf32, 126x504x1x1xf32)
        t605 = paddle._C_ops.conv2d(
            t604, t196, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t196

        # pd_op.reshape: (1x126x1x1xf32) <- (126xf32, 4xi64)
        t606 = paddle._C_ops.reshape(t197, t376)
        del t197

        # pd_op.add: (-1x126x1x1xf32) <- (-1x126x1x1xf32, 1x126x1x1xf32)
        t607 = paddle._C_ops.add(t605, t606)

        # pd_op.relu: (-1x126x1x1xf32) <- (-1x126x1x1xf32)
        t608 = paddle._C_ops.relu(t607)
        del t607

        # pd_op.conv2d: (-1x504x1x1xf32) <- (-1x126x1x1xf32, 504x126x1x1xf32)
        t609 = paddle._C_ops.conv2d(
            t608, t198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t198

        # pd_op.reshape: (1x504x1x1xf32) <- (504xf32, 4xi64)
        t610 = paddle._C_ops.reshape(t199, t376)
        del t199

        # pd_op.add: (-1x504x1x1xf32) <- (-1x504x1x1xf32, 1x504x1x1xf32)
        t611 = paddle._C_ops.add(t609, t610)

        # pd_op.hardsigmoid: (-1x504x1x1xf32) <- (-1x504x1x1xf32)
        t612 = paddle._C_ops.hardsigmoid(t611, float("0.2"), float("0.5"))
        del t611

        # pd_op.multiply: (-1x504x14x14xf32) <- (-1x504x14x14xf32, -1x504x1x1xf32)
        t613 = paddle._C_ops.multiply(t603, t612)

        # pd_op.conv2d: (-1x88x14x14xf32) <- (-1x504x14x14xf32, 88x504x1x1xf32)
        t614 = paddle._C_ops.conv2d(
            t613, t200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (-1x88x14x14xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x14x14xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t615, t616, t617, t618, t619, t620 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t614,
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

        # pd_op.add: (-1x88x14x14xf32) <- (-1x88x14x14xf32, -1x88x14x14xf32)
        t621 = paddle._C_ops.add(t582, t615)

        # pd_op.conv2d: (-1x504x14x14xf32) <- (-1x88x14x14xf32, 504x88x1x1xf32)
        t622 = paddle._C_ops.conv2d(
            t621, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (-1x504x14x14xf32, 504xf32, 504xf32, 504xf32, 504xf32, -1xui8) <- (-1x504x14x14xf32, 504xf32, 504xf32, 504xf32, 504xf32)
        t623, t624, t625, t626, t627, t628 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t622,
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

        # pd_op.hardswish: (-1x504x14x14xf32) <- (-1x504x14x14xf32)
        t629 = paddle._C_ops.hardswish(t623)

        # pd_op.depthwise_conv2d: (-1x504x7x7xf32) <- (-1x504x14x14xf32, 504x1x5x5xf32)
        t630 = paddle._C_ops.depthwise_conv2d(
            t629, t210, [2, 2], [2, 2], "EXPLICIT", 504, [1, 1], "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (-1x504x7x7xf32, 504xf32, 504xf32, 504xf32, 504xf32, -1xui8) <- (-1x504x7x7xf32, 504xf32, 504xf32, 504xf32, 504xf32)
        t631, t632, t633, t634, t635, t636 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t630,
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

        # pd_op.hardswish: (-1x504x7x7xf32) <- (-1x504x7x7xf32)
        t637 = paddle._C_ops.hardswish(t631)

        # pd_op.pool2d: (-1x504x1x1xf32) <- (-1x504x7x7xf32, 2xi64)
        t638 = paddle._C_ops.pool2d(
            t637,
            t365,
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

        # pd_op.conv2d: (-1x126x1x1xf32) <- (-1x504x1x1xf32, 126x504x1x1xf32)
        t639 = paddle._C_ops.conv2d(
            t638, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.reshape: (1x126x1x1xf32) <- (126xf32, 4xi64)
        t640 = paddle._C_ops.reshape(t216, t376)
        del t216

        # pd_op.add: (-1x126x1x1xf32) <- (-1x126x1x1xf32, 1x126x1x1xf32)
        t641 = paddle._C_ops.add(t639, t640)

        # pd_op.relu: (-1x126x1x1xf32) <- (-1x126x1x1xf32)
        t642 = paddle._C_ops.relu(t641)
        del t641

        # pd_op.conv2d: (-1x504x1x1xf32) <- (-1x126x1x1xf32, 504x126x1x1xf32)
        t643 = paddle._C_ops.conv2d(
            t642, t217, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t217

        # pd_op.reshape: (1x504x1x1xf32) <- (504xf32, 4xi64)
        t644 = paddle._C_ops.reshape(t218, t376)
        del t218

        # pd_op.add: (-1x504x1x1xf32) <- (-1x504x1x1xf32, 1x504x1x1xf32)
        t645 = paddle._C_ops.add(t643, t644)

        # pd_op.hardsigmoid: (-1x504x1x1xf32) <- (-1x504x1x1xf32)
        t646 = paddle._C_ops.hardsigmoid(t645, float("0.2"), float("0.5"))
        del t645

        # pd_op.multiply: (-1x504x7x7xf32) <- (-1x504x7x7xf32, -1x504x1x1xf32)
        t647 = paddle._C_ops.multiply(t637, t646)

        # pd_op.conv2d: (-1x120x7x7xf32) <- (-1x504x7x7xf32, 120x504x1x1xf32)
        t648 = paddle._C_ops.conv2d(
            t647, t219, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t219

        # pd_op.batch_norm_: (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t649, t650, t651, t652, t653, t654 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t648,
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

        # pd_op.conv2d: (-1x720x7x7xf32) <- (-1x120x7x7xf32, 720x120x1x1xf32)
        t655 = paddle._C_ops.conv2d(
            t649, t224, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t224

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t656, t657, t658, t659, t660, t661 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t655,
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

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t662 = paddle._C_ops.hardswish(t656)

        # pd_op.depthwise_conv2d: (-1x720x7x7xf32) <- (-1x720x7x7xf32, 720x1x5x5xf32)
        t663 = paddle._C_ops.depthwise_conv2d(
            t662, t229, [1, 1], [2, 2], "EXPLICIT", 720, [1, 1], "NCHW"
        )
        del t229

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t664, t665, t666, t667, t668, t669 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t663,
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

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t670 = paddle._C_ops.hardswish(t664)

        # pd_op.pool2d: (-1x720x1x1xf32) <- (-1x720x7x7xf32, 2xi64)
        t671 = paddle._C_ops.pool2d(
            t670,
            t365,
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

        # pd_op.conv2d: (-1x180x1x1xf32) <- (-1x720x1x1xf32, 180x720x1x1xf32)
        t672 = paddle._C_ops.conv2d(
            t671, t234, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t234

        # pd_op.reshape: (1x180x1x1xf32) <- (180xf32, 4xi64)
        t673 = paddle._C_ops.reshape(t235, t376)
        del t235

        # pd_op.add: (-1x180x1x1xf32) <- (-1x180x1x1xf32, 1x180x1x1xf32)
        t674 = paddle._C_ops.add(t672, t673)

        # pd_op.relu: (-1x180x1x1xf32) <- (-1x180x1x1xf32)
        t675 = paddle._C_ops.relu(t674)
        del t674

        # pd_op.conv2d: (-1x720x1x1xf32) <- (-1x180x1x1xf32, 720x180x1x1xf32)
        t676 = paddle._C_ops.conv2d(
            t675, t236, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t236

        # pd_op.reshape: (1x720x1x1xf32) <- (720xf32, 4xi64)
        t677 = paddle._C_ops.reshape(t237, t376)
        del t237

        # pd_op.add: (-1x720x1x1xf32) <- (-1x720x1x1xf32, 1x720x1x1xf32)
        t678 = paddle._C_ops.add(t676, t677)

        # pd_op.hardsigmoid: (-1x720x1x1xf32) <- (-1x720x1x1xf32)
        t679 = paddle._C_ops.hardsigmoid(t678, float("0.2"), float("0.5"))
        del t678

        # pd_op.multiply: (-1x720x7x7xf32) <- (-1x720x7x7xf32, -1x720x1x1xf32)
        t680 = paddle._C_ops.multiply(t670, t679)

        # pd_op.conv2d: (-1x120x7x7xf32) <- (-1x720x7x7xf32, 120x720x1x1xf32)
        t681 = paddle._C_ops.conv2d(
            t680, t238, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t238

        # pd_op.batch_norm_: (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t682, t683, t684, t685, t686, t687 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t681,
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

        # pd_op.add: (-1x120x7x7xf32) <- (-1x120x7x7xf32, -1x120x7x7xf32)
        t688 = paddle._C_ops.add(t649, t682)

        # pd_op.conv2d: (-1x720x7x7xf32) <- (-1x120x7x7xf32, 720x120x1x1xf32)
        t689 = paddle._C_ops.conv2d(
            t688, t243, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t243

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t690, t691, t692, t693, t694, t695 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t689,
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

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t696 = paddle._C_ops.hardswish(t690)

        # pd_op.depthwise_conv2d: (-1x720x7x7xf32) <- (-1x720x7x7xf32, 720x1x5x5xf32)
        t697 = paddle._C_ops.depthwise_conv2d(
            t696, t248, [1, 1], [2, 2], "EXPLICIT", 720, [1, 1], "NCHW"
        )
        del t248

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t698, t699, t700, t701, t702, t703 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t697,
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

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t704 = paddle._C_ops.hardswish(t698)

        # pd_op.pool2d: (-1x720x1x1xf32) <- (-1x720x7x7xf32, 2xi64)
        t705 = paddle._C_ops.pool2d(
            t704,
            t365,
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

        # pd_op.conv2d: (-1x180x1x1xf32) <- (-1x720x1x1xf32, 180x720x1x1xf32)
        t706 = paddle._C_ops.conv2d(
            t705, t253, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t253

        # pd_op.reshape: (1x180x1x1xf32) <- (180xf32, 4xi64)
        t707 = paddle._C_ops.reshape(t254, t376)
        del t254

        # pd_op.add: (-1x180x1x1xf32) <- (-1x180x1x1xf32, 1x180x1x1xf32)
        t708 = paddle._C_ops.add(t706, t707)

        # pd_op.relu: (-1x180x1x1xf32) <- (-1x180x1x1xf32)
        t709 = paddle._C_ops.relu(t708)
        del t708

        # pd_op.conv2d: (-1x720x1x1xf32) <- (-1x180x1x1xf32, 720x180x1x1xf32)
        t710 = paddle._C_ops.conv2d(
            t709, t255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255

        # pd_op.reshape: (1x720x1x1xf32) <- (720xf32, 4xi64)
        t711 = paddle._C_ops.reshape(t256, t376)
        del t376, t256

        # pd_op.add: (-1x720x1x1xf32) <- (-1x720x1x1xf32, 1x720x1x1xf32)
        t712 = paddle._C_ops.add(t710, t711)

        # pd_op.hardsigmoid: (-1x720x1x1xf32) <- (-1x720x1x1xf32)
        t713 = paddle._C_ops.hardsigmoid(t712, float("0.2"), float("0.5"))
        del t712

        # pd_op.multiply: (-1x720x7x7xf32) <- (-1x720x7x7xf32, -1x720x1x1xf32)
        t714 = paddle._C_ops.multiply(t704, t713)

        # pd_op.conv2d: (-1x120x7x7xf32) <- (-1x720x7x7xf32, 120x720x1x1xf32)
        t715 = paddle._C_ops.conv2d(
            t714, t257, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t257

        # pd_op.batch_norm_: (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (-1x120x7x7xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t716, t717, t718, t719, t720, t721 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t715,
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
        del t259, t258, t261, t260

        # pd_op.add: (-1x120x7x7xf32) <- (-1x120x7x7xf32, -1x120x7x7xf32)
        t722 = paddle._C_ops.add(t688, t716)

        # pd_op.conv2d: (-1x720x7x7xf32) <- (-1x120x7x7xf32, 720x120x1x1xf32)
        t723 = paddle._C_ops.conv2d(
            t722, t262, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t262

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32, -1xui8) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        t724, t725, t726, t727, t728, t729 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t723,
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

        # pd_op.hardswish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        t730 = paddle._C_ops.hardswish(t724)

        # pd_op.pool2d: (-1x720x1x1xf32) <- (-1x720x7x7xf32, 2xi64)
        t731 = paddle._C_ops.pool2d(
            t730,
            t365,
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

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x720x1x1xf32, 1280x720x1x1xf32)
        t732 = paddle._C_ops.conv2d(
            t731, t267, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t267

        # pd_op.hardswish: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        t733 = paddle._C_ops.hardswish(t732)

        # pd_op.full: (1xf32) <- ()
        t734 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x1280x1x1xf32, -1x1280x1x1xui8) <- (-1x1280x1x1xf32, None, 1xf32)
        t735, t736 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                t733, None, t734, False, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t733

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        t737 = paddle._C_ops.flatten(t735, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        t738 = paddle._C_ops.matmul(t737, t268, False, False)
        del t268

        return (
            t270,
            t271,
            t272,
            t273,
            t274,
            t275,
            t276,
            t277,
            t278,
            t280,
            t281,
            t282,
            t283,
            t284,
            t285,
            t286,
            t288,
            t289,
            t290,
            t291,
            t292,
            t293,
            t294,
            t295,
            t296,
            t297,
            t298,
            t299,
            t300,
            t301,
            t302,
            t304,
            t305,
            t306,
            t307,
            t308,
            t309,
            t310,
            t312,
            t313,
            t314,
            t315,
            t316,
            t317,
            t318,
            t319,
            t320,
            t321,
            t322,
            t323,
            t324,
            t325,
            t327,
            t328,
            t329,
            t330,
            t331,
            t332,
            t333,
            t335,
            t336,
            t337,
            t338,
            t339,
            t340,
            t341,
            t342,
            t343,
            t344,
            t345,
            t346,
            t347,
            t348,
            t349,
            t351,
            t352,
            t353,
            t354,
            t355,
            t356,
            t357,
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
            t373,
            t374,
            t375,
            t377,
            t379,
            t380,
            t381,
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
            t394,
            t395,
            t396,
            t397,
            t398,
            t399,
            t400,
            t402,
            t403,
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
            t575,
            t576,
            t577,
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
            t608,
            t609,
            t610,
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
            t642,
            t643,
            t644,
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
            t675,
            t676,
            t677,
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
            t709,
            t710,
            t711,
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
            t734,
            t735,
            t736,
            t737,
            t738,
        )
