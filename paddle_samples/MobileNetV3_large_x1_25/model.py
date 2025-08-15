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
        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x3x224x224xf32, 24x3x3x3xf32)
        t270 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t271, t272, t273, t274, t275, t276 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t270,
                t1,
                t2,
                t3,
                t4,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t270, t4, t3, t2, t1

        # pd_op.hardswish: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        t277 = paddle._C_ops.hardswish(t271)
        del t271

        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x24x112x112xf32, 24x24x1x1xf32)
        t278 = paddle._C_ops.conv2d(
            t277, t5, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t279, t280, t281, t282, t283, t284 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t278,
                t6,
                t7,
                t8,
                t9,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t278, t9, t8, t7, t6

        # pd_op.relu: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        t285 = paddle._C_ops.relu(t279)
        del t279

        # pd_op.depthwise_conv2d: (-1x24x112x112xf32) <- (-1x24x112x112xf32, 24x1x3x3xf32)
        t286 = paddle._C_ops.depthwise_conv2d(
            t285, t10, [1, 1], [1, 1], "EXPLICIT", 24, [1, 1], "NCHW"
        )
        del t10, t285

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t287, t288, t289, t290, t291, t292 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t286,
                t11,
                t12,
                t13,
                t14,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t286, t14, t13, t12, t11

        # pd_op.relu: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        t293 = paddle._C_ops.relu(t287)
        del t287

        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x24x112x112xf32, 24x24x1x1xf32)
        t294 = paddle._C_ops.conv2d(
            t293, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15, t293

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t295, t296, t297, t298, t299, t300 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t294,
                t16,
                t17,
                t18,
                t19,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t294, t19, t18, t17, t16

        # pd_op.add: (-1x24x112x112xf32) <- (-1x24x112x112xf32, -1x24x112x112xf32)
        t301 = paddle._C_ops.add(t277, t295)
        del t295, t277

        # pd_op.conv2d: (-1x80x112x112xf32) <- (-1x24x112x112xf32, 80x24x1x1xf32)
        t302 = paddle._C_ops.conv2d(
            t301, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t301, t20

        # pd_op.batch_norm_: (-1x80x112x112xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (-1x80x112x112xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        t303, t304, t305, t306, t307, t308 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t302,
                t21,
                t22,
                t23,
                t24,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t302, t24, t23, t22, t21

        # pd_op.relu: (-1x80x112x112xf32) <- (-1x80x112x112xf32)
        t309 = paddle._C_ops.relu(t303)
        del t303

        # pd_op.depthwise_conv2d: (-1x80x56x56xf32) <- (-1x80x112x112xf32, 80x1x3x3xf32)
        t310 = paddle._C_ops.depthwise_conv2d(
            t309, t25, [2, 2], [1, 1], "EXPLICIT", 80, [1, 1], "NCHW"
        )
        del t25, t309

        # pd_op.batch_norm_: (-1x80x56x56xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (-1x80x56x56xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        t311, t312, t313, t314, t315, t316 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t310,
                t26,
                t27,
                t28,
                t29,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t310, t29, t28, t27, t26

        # pd_op.relu: (-1x80x56x56xf32) <- (-1x80x56x56xf32)
        t317 = paddle._C_ops.relu(t311)
        del t311

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x80x56x56xf32, 32x80x1x1xf32)
        t318 = paddle._C_ops.conv2d(
            t317, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30, t317

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t319, t320, t321, t322, t323, t324 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t318,
                t31,
                t32,
                t33,
                t34,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t318, t34, t33, t32, t31

        # pd_op.conv2d: (-1x88x56x56xf32) <- (-1x32x56x56xf32, 88x32x1x1xf32)
        t325 = paddle._C_ops.conv2d(
            t319, t35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x88x56x56xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x56x56xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t326, t327, t328, t329, t330, t331 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t325,
                t36,
                t37,
                t38,
                t39,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t325, t39, t38, t37, t36

        # pd_op.relu: (-1x88x56x56xf32) <- (-1x88x56x56xf32)
        t332 = paddle._C_ops.relu(t326)
        del t326

        # pd_op.depthwise_conv2d: (-1x88x56x56xf32) <- (-1x88x56x56xf32, 88x1x3x3xf32)
        t333 = paddle._C_ops.depthwise_conv2d(
            t332, t40, [1, 1], [1, 1], "EXPLICIT", 88, [1, 1], "NCHW"
        )
        del t40, t332

        # pd_op.batch_norm_: (-1x88x56x56xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x56x56xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t334, t335, t336, t337, t338, t339 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t333,
                t41,
                t42,
                t43,
                t44,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t333, t44, t43, t42, t41

        # pd_op.relu: (-1x88x56x56xf32) <- (-1x88x56x56xf32)
        t340 = paddle._C_ops.relu(t334)
        del t334

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x88x56x56xf32, 32x88x1x1xf32)
        t341 = paddle._C_ops.conv2d(
            t340, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45, t340

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t342, t343, t344, t345, t346, t347 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t341,
                t46,
                t47,
                t48,
                t49,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t341, t49, t48, t47, t46

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        t348 = paddle._C_ops.add(t319, t342)
        del t319, t342

        # pd_op.conv2d: (-1x88x56x56xf32) <- (-1x32x56x56xf32, 88x32x1x1xf32)
        t349 = paddle._C_ops.conv2d(
            t348, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t348, t50

        # pd_op.batch_norm_: (-1x88x56x56xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x56x56xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t350, t351, t352, t353, t354, t355 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t349,
                t51,
                t52,
                t53,
                t54,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t349, t54, t53, t52, t51

        # pd_op.relu: (-1x88x56x56xf32) <- (-1x88x56x56xf32)
        t356 = paddle._C_ops.relu(t350)
        del t350

        # pd_op.depthwise_conv2d: (-1x88x28x28xf32) <- (-1x88x56x56xf32, 88x1x5x5xf32)
        t357 = paddle._C_ops.depthwise_conv2d(
            t356, t55, [2, 2], [2, 2], "EXPLICIT", 88, [1, 1], "NCHW"
        )
        del t55, t356

        # pd_op.batch_norm_: (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32, -1xui8) <- (-1x88x28x28xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        t358, t359, t360, t361, t362, t363 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t357,
                t56,
                t57,
                t58,
                t59,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t357, t59, t58, t57, t56

        # pd_op.relu: (-1x88x28x28xf32) <- (-1x88x28x28xf32)
        t364 = paddle._C_ops.relu(t358)
        del t358

        # pd_op.full_int_array: (2xi64) <- ()
        t365 = [1, 1]

        # pd_op.pool2d: (-1x88x1x1xf32) <- (-1x88x28x28xf32, 2xi64)
        t366 = paddle._C_ops.pool2d(
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

        # pd_op.conv2d: (-1x22x1x1xf32) <- (-1x88x1x1xf32, 22x88x1x1xf32)
        t367 = paddle._C_ops.conv2d(
            t366, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60, t366

        # pd_op.full_int_array: (4xi64) <- ()
        t368 = [1, -1, 1, 1]

        # pd_op.reshape: (1x22x1x1xf32) <- (22xf32, 4xi64)
        t369 = paddle._C_ops.reshape(t61, t368)
        del t61

        # pd_op.add: (-1x22x1x1xf32) <- (-1x22x1x1xf32, 1x22x1x1xf32)
        t370 = paddle._C_ops.add(t367, t369)
        del t367, t369

        # pd_op.relu: (-1x22x1x1xf32) <- (-1x22x1x1xf32)
        t371 = paddle._C_ops.relu(t370)
        del t370

        # pd_op.conv2d: (-1x88x1x1xf32) <- (-1x22x1x1xf32, 88x22x1x1xf32)
        t372 = paddle._C_ops.conv2d(
            t371, t62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t62, t371

        # pd_op.reshape: (1x88x1x1xf32) <- (88xf32, 4xi64)
        t373 = paddle._C_ops.reshape(t63, t368)
        del t63

        # pd_op.add: (-1x88x1x1xf32) <- (-1x88x1x1xf32, 1x88x1x1xf32)
        t374 = paddle._C_ops.add(t372, t373)
        del t372, t373

        # pd_op.hardsigmoid: (-1x88x1x1xf32) <- (-1x88x1x1xf32)
        t375 = paddle._C_ops.hardsigmoid(t374, float("0.2"), float("0.5"))
        del t374

        # pd_op.multiply: (-1x88x28x28xf32) <- (-1x88x28x28xf32, -1x88x1x1xf32)
        t376 = paddle._C_ops.multiply(t364, t375)
        del t375, t364

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x88x28x28xf32, 48x88x1x1xf32)
        t377 = paddle._C_ops.conv2d(
            t376, t64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t376, t64

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t378, t379, t380, t381, t382, t383 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t377,
                t65,
                t66,
                t67,
                t68,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t377, t68, t67, t66, t65

        # pd_op.conv2d: (-1x152x28x28xf32) <- (-1x48x28x28xf32, 152x48x1x1xf32)
        t384 = paddle._C_ops.conv2d(
            t378, t69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t69

        # pd_op.batch_norm_: (-1x152x28x28xf32, 152xf32, 152xf32, 152xf32, 152xf32, -1xui8) <- (-1x152x28x28xf32, 152xf32, 152xf32, 152xf32, 152xf32)
        t385, t386, t387, t388, t389, t390 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t384,
                t70,
                t71,
                t72,
                t73,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t384, t73, t72, t71, t70

        # pd_op.relu: (-1x152x28x28xf32) <- (-1x152x28x28xf32)
        t391 = paddle._C_ops.relu(t385)
        del t385

        # pd_op.depthwise_conv2d: (-1x152x28x28xf32) <- (-1x152x28x28xf32, 152x1x5x5xf32)
        t392 = paddle._C_ops.depthwise_conv2d(
            t391, t74, [1, 1], [2, 2], "EXPLICIT", 152, [1, 1], "NCHW"
        )
        del t74, t391

        # pd_op.batch_norm_: (-1x152x28x28xf32, 152xf32, 152xf32, 152xf32, 152xf32, -1xui8) <- (-1x152x28x28xf32, 152xf32, 152xf32, 152xf32, 152xf32)
        t393, t394, t395, t396, t397, t398 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t392,
                t75,
                t76,
                t77,
                t78,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t392, t78, t77, t76, t75

        # pd_op.relu: (-1x152x28x28xf32) <- (-1x152x28x28xf32)
        t399 = paddle._C_ops.relu(t393)
        del t393

        # pd_op.pool2d: (-1x152x1x1xf32) <- (-1x152x28x28xf32, 2xi64)
        t400 = paddle._C_ops.pool2d(
            t399,
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

        # pd_op.conv2d: (-1x38x1x1xf32) <- (-1x152x1x1xf32, 38x152x1x1xf32)
        t401 = paddle._C_ops.conv2d(
            t400, t79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t79, t400

        # pd_op.reshape: (1x38x1x1xf32) <- (38xf32, 4xi64)
        t402 = paddle._C_ops.reshape(t80, t368)
        del t80

        # pd_op.add: (-1x38x1x1xf32) <- (-1x38x1x1xf32, 1x38x1x1xf32)
        t403 = paddle._C_ops.add(t401, t402)
        del t401, t402

        # pd_op.relu: (-1x38x1x1xf32) <- (-1x38x1x1xf32)
        t404 = paddle._C_ops.relu(t403)
        del t403

        # pd_op.conv2d: (-1x152x1x1xf32) <- (-1x38x1x1xf32, 152x38x1x1xf32)
        t405 = paddle._C_ops.conv2d(
            t404, t81, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t81, t404

        # pd_op.reshape: (1x152x1x1xf32) <- (152xf32, 4xi64)
        t406 = paddle._C_ops.reshape(t82, t368)
        del t82

        # pd_op.add: (-1x152x1x1xf32) <- (-1x152x1x1xf32, 1x152x1x1xf32)
        t407 = paddle._C_ops.add(t405, t406)
        del t405, t406

        # pd_op.hardsigmoid: (-1x152x1x1xf32) <- (-1x152x1x1xf32)
        t408 = paddle._C_ops.hardsigmoid(t407, float("0.2"), float("0.5"))
        del t407

        # pd_op.multiply: (-1x152x28x28xf32) <- (-1x152x28x28xf32, -1x152x1x1xf32)
        t409 = paddle._C_ops.multiply(t399, t408)
        del t408, t399

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x152x28x28xf32, 48x152x1x1xf32)
        t410 = paddle._C_ops.conv2d(
            t409, t83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t409, t83

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t411, t412, t413, t414, t415, t416 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t410,
                t84,
                t85,
                t86,
                t87,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t410, t87, t86, t85, t84

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, -1x48x28x28xf32)
        t417 = paddle._C_ops.add(t378, t411)
        del t378, t411

        # pd_op.conv2d: (-1x152x28x28xf32) <- (-1x48x28x28xf32, 152x48x1x1xf32)
        t418 = paddle._C_ops.conv2d(
            t417, t88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t88

        # pd_op.batch_norm_: (-1x152x28x28xf32, 152xf32, 152xf32, 152xf32, 152xf32, -1xui8) <- (-1x152x28x28xf32, 152xf32, 152xf32, 152xf32, 152xf32)
        t419, t420, t421, t422, t423, t424 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t418,
                t89,
                t90,
                t91,
                t92,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t418, t92, t91, t90, t89

        # pd_op.relu: (-1x152x28x28xf32) <- (-1x152x28x28xf32)
        t425 = paddle._C_ops.relu(t419)
        del t419

        # pd_op.depthwise_conv2d: (-1x152x28x28xf32) <- (-1x152x28x28xf32, 152x1x5x5xf32)
        t426 = paddle._C_ops.depthwise_conv2d(
            t425, t93, [1, 1], [2, 2], "EXPLICIT", 152, [1, 1], "NCHW"
        )
        del t93, t425

        # pd_op.batch_norm_: (-1x152x28x28xf32, 152xf32, 152xf32, 152xf32, 152xf32, -1xui8) <- (-1x152x28x28xf32, 152xf32, 152xf32, 152xf32, 152xf32)
        t427, t428, t429, t430, t431, t432 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t426,
                t94,
                t95,
                t96,
                t97,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t426, t97, t96, t95, t94

        # pd_op.relu: (-1x152x28x28xf32) <- (-1x152x28x28xf32)
        t433 = paddle._C_ops.relu(t427)
        del t427

        # pd_op.pool2d: (-1x152x1x1xf32) <- (-1x152x28x28xf32, 2xi64)
        t434 = paddle._C_ops.pool2d(
            t433,
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

        # pd_op.conv2d: (-1x38x1x1xf32) <- (-1x152x1x1xf32, 38x152x1x1xf32)
        t435 = paddle._C_ops.conv2d(
            t434, t98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t98, t434

        # pd_op.reshape: (1x38x1x1xf32) <- (38xf32, 4xi64)
        t436 = paddle._C_ops.reshape(t99, t368)
        del t99

        # pd_op.add: (-1x38x1x1xf32) <- (-1x38x1x1xf32, 1x38x1x1xf32)
        t437 = paddle._C_ops.add(t435, t436)
        del t435, t436

        # pd_op.relu: (-1x38x1x1xf32) <- (-1x38x1x1xf32)
        t438 = paddle._C_ops.relu(t437)
        del t437

        # pd_op.conv2d: (-1x152x1x1xf32) <- (-1x38x1x1xf32, 152x38x1x1xf32)
        t439 = paddle._C_ops.conv2d(
            t438, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100, t438

        # pd_op.reshape: (1x152x1x1xf32) <- (152xf32, 4xi64)
        t440 = paddle._C_ops.reshape(t101, t368)
        del t101

        # pd_op.add: (-1x152x1x1xf32) <- (-1x152x1x1xf32, 1x152x1x1xf32)
        t441 = paddle._C_ops.add(t439, t440)
        del t439, t440

        # pd_op.hardsigmoid: (-1x152x1x1xf32) <- (-1x152x1x1xf32)
        t442 = paddle._C_ops.hardsigmoid(t441, float("0.2"), float("0.5"))
        del t441

        # pd_op.multiply: (-1x152x28x28xf32) <- (-1x152x28x28xf32, -1x152x1x1xf32)
        t443 = paddle._C_ops.multiply(t433, t442)
        del t442, t433

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x152x28x28xf32, 48x152x1x1xf32)
        t444 = paddle._C_ops.conv2d(
            t443, t102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t443, t102

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t445, t446, t447, t448, t449, t450 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t444,
                t103,
                t104,
                t105,
                t106,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t444, t106, t105, t104, t103

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, -1x48x28x28xf32)
        t451 = paddle._C_ops.add(t417, t445)
        del t417, t445

        # pd_op.conv2d: (-1x304x28x28xf32) <- (-1x48x28x28xf32, 304x48x1x1xf32)
        t452 = paddle._C_ops.conv2d(
            t451, t107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t451, t107

        # pd_op.batch_norm_: (-1x304x28x28xf32, 304xf32, 304xf32, 304xf32, 304xf32, -1xui8) <- (-1x304x28x28xf32, 304xf32, 304xf32, 304xf32, 304xf32)
        t453, t454, t455, t456, t457, t458 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t452,
                t108,
                t109,
                t110,
                t111,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t452, t111, t110, t109, t108

        # pd_op.hardswish: (-1x304x28x28xf32) <- (-1x304x28x28xf32)
        t459 = paddle._C_ops.hardswish(t453)
        del t453

        # pd_op.depthwise_conv2d: (-1x304x14x14xf32) <- (-1x304x28x28xf32, 304x1x3x3xf32)
        t460 = paddle._C_ops.depthwise_conv2d(
            t459, t112, [2, 2], [1, 1], "EXPLICIT", 304, [1, 1], "NCHW"
        )
        del t459, t112

        # pd_op.batch_norm_: (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32, -1xui8) <- (-1x304x14x14xf32, 304xf32, 304xf32, 304xf32, 304xf32)
        t461, t462, t463, t464, t465, t466 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t460,
                t113,
                t114,
                t115,
                t116,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t460, t116, t115, t114, t113

        # pd_op.hardswish: (-1x304x14x14xf32) <- (-1x304x14x14xf32)
        t467 = paddle._C_ops.hardswish(t461)
        del t461

        # pd_op.conv2d: (-1x104x14x14xf32) <- (-1x304x14x14xf32, 104x304x1x1xf32)
        t468 = paddle._C_ops.conv2d(
            t467, t117, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t467, t117

        # pd_op.batch_norm_: (-1x104x14x14xf32, 104xf32, 104xf32, 104xf32, 104xf32, -1xui8) <- (-1x104x14x14xf32, 104xf32, 104xf32, 104xf32, 104xf32)
        t469, t470, t471, t472, t473, t474 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t468,
                t118,
                t119,
                t120,
                t121,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t468, t121, t120, t119, t118

        # pd_op.conv2d: (-1x248x14x14xf32) <- (-1x104x14x14xf32, 248x104x1x1xf32)
        t475 = paddle._C_ops.conv2d(
            t469, t122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t122

        # pd_op.batch_norm_: (-1x248x14x14xf32, 248xf32, 248xf32, 248xf32, 248xf32, -1xui8) <- (-1x248x14x14xf32, 248xf32, 248xf32, 248xf32, 248xf32)
        t476, t477, t478, t479, t480, t481 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t475,
                t123,
                t124,
                t125,
                t126,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t475, t126, t125, t124, t123

        # pd_op.hardswish: (-1x248x14x14xf32) <- (-1x248x14x14xf32)
        t482 = paddle._C_ops.hardswish(t476)
        del t476

        # pd_op.depthwise_conv2d: (-1x248x14x14xf32) <- (-1x248x14x14xf32, 248x1x3x3xf32)
        t483 = paddle._C_ops.depthwise_conv2d(
            t482, t127, [1, 1], [1, 1], "EXPLICIT", 248, [1, 1], "NCHW"
        )
        del t482, t127

        # pd_op.batch_norm_: (-1x248x14x14xf32, 248xf32, 248xf32, 248xf32, 248xf32, -1xui8) <- (-1x248x14x14xf32, 248xf32, 248xf32, 248xf32, 248xf32)
        t484, t485, t486, t487, t488, t489 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t483,
                t128,
                t129,
                t130,
                t131,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t483, t131, t130, t129, t128

        # pd_op.hardswish: (-1x248x14x14xf32) <- (-1x248x14x14xf32)
        t490 = paddle._C_ops.hardswish(t484)
        del t484

        # pd_op.conv2d: (-1x104x14x14xf32) <- (-1x248x14x14xf32, 104x248x1x1xf32)
        t491 = paddle._C_ops.conv2d(
            t490, t132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t490, t132

        # pd_op.batch_norm_: (-1x104x14x14xf32, 104xf32, 104xf32, 104xf32, 104xf32, -1xui8) <- (-1x104x14x14xf32, 104xf32, 104xf32, 104xf32, 104xf32)
        t492, t493, t494, t495, t496, t497 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t491,
                t133,
                t134,
                t135,
                t136,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t491, t136, t135, t134, t133

        # pd_op.add: (-1x104x14x14xf32) <- (-1x104x14x14xf32, -1x104x14x14xf32)
        t498 = paddle._C_ops.add(t469, t492)
        del t469, t492

        # pd_op.conv2d: (-1x232x14x14xf32) <- (-1x104x14x14xf32, 232x104x1x1xf32)
        t499 = paddle._C_ops.conv2d(
            t498, t137, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t137

        # pd_op.batch_norm_: (-1x232x14x14xf32, 232xf32, 232xf32, 232xf32, 232xf32, -1xui8) <- (-1x232x14x14xf32, 232xf32, 232xf32, 232xf32, 232xf32)
        t500, t501, t502, t503, t504, t505 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t499,
                t138,
                t139,
                t140,
                t141,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t499, t141, t140, t139, t138

        # pd_op.hardswish: (-1x232x14x14xf32) <- (-1x232x14x14xf32)
        t506 = paddle._C_ops.hardswish(t500)
        del t500

        # pd_op.depthwise_conv2d: (-1x232x14x14xf32) <- (-1x232x14x14xf32, 232x1x3x3xf32)
        t507 = paddle._C_ops.depthwise_conv2d(
            t506, t142, [1, 1], [1, 1], "EXPLICIT", 232, [1, 1], "NCHW"
        )
        del t506, t142

        # pd_op.batch_norm_: (-1x232x14x14xf32, 232xf32, 232xf32, 232xf32, 232xf32, -1xui8) <- (-1x232x14x14xf32, 232xf32, 232xf32, 232xf32, 232xf32)
        t508, t509, t510, t511, t512, t513 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t507,
                t143,
                t144,
                t145,
                t146,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t507, t146, t145, t144, t143

        # pd_op.hardswish: (-1x232x14x14xf32) <- (-1x232x14x14xf32)
        t514 = paddle._C_ops.hardswish(t508)
        del t508

        # pd_op.conv2d: (-1x104x14x14xf32) <- (-1x232x14x14xf32, 104x232x1x1xf32)
        t515 = paddle._C_ops.conv2d(
            t514, t147, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t514, t147

        # pd_op.batch_norm_: (-1x104x14x14xf32, 104xf32, 104xf32, 104xf32, 104xf32, -1xui8) <- (-1x104x14x14xf32, 104xf32, 104xf32, 104xf32, 104xf32)
        t516, t517, t518, t519, t520, t521 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t515,
                t148,
                t149,
                t150,
                t151,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t515, t151, t150, t149, t148

        # pd_op.add: (-1x104x14x14xf32) <- (-1x104x14x14xf32, -1x104x14x14xf32)
        t522 = paddle._C_ops.add(t498, t516)
        del t498, t516

        # pd_op.conv2d: (-1x232x14x14xf32) <- (-1x104x14x14xf32, 232x104x1x1xf32)
        t523 = paddle._C_ops.conv2d(
            t522, t152, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t152

        # pd_op.batch_norm_: (-1x232x14x14xf32, 232xf32, 232xf32, 232xf32, 232xf32, -1xui8) <- (-1x232x14x14xf32, 232xf32, 232xf32, 232xf32, 232xf32)
        t524, t525, t526, t527, t528, t529 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t523,
                t153,
                t154,
                t155,
                t156,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t523, t156, t155, t154, t153

        # pd_op.hardswish: (-1x232x14x14xf32) <- (-1x232x14x14xf32)
        t530 = paddle._C_ops.hardswish(t524)
        del t524

        # pd_op.depthwise_conv2d: (-1x232x14x14xf32) <- (-1x232x14x14xf32, 232x1x3x3xf32)
        t531 = paddle._C_ops.depthwise_conv2d(
            t530, t157, [1, 1], [1, 1], "EXPLICIT", 232, [1, 1], "NCHW"
        )
        del t530, t157

        # pd_op.batch_norm_: (-1x232x14x14xf32, 232xf32, 232xf32, 232xf32, 232xf32, -1xui8) <- (-1x232x14x14xf32, 232xf32, 232xf32, 232xf32, 232xf32)
        t532, t533, t534, t535, t536, t537 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t531,
                t158,
                t159,
                t160,
                t161,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t531, t161, t160, t159, t158

        # pd_op.hardswish: (-1x232x14x14xf32) <- (-1x232x14x14xf32)
        t538 = paddle._C_ops.hardswish(t532)
        del t532

        # pd_op.conv2d: (-1x104x14x14xf32) <- (-1x232x14x14xf32, 104x232x1x1xf32)
        t539 = paddle._C_ops.conv2d(
            t538, t162, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t538, t162

        # pd_op.batch_norm_: (-1x104x14x14xf32, 104xf32, 104xf32, 104xf32, 104xf32, -1xui8) <- (-1x104x14x14xf32, 104xf32, 104xf32, 104xf32, 104xf32)
        t540, t541, t542, t543, t544, t545 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t539,
                t163,
                t164,
                t165,
                t166,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t539, t166, t165, t164, t163

        # pd_op.add: (-1x104x14x14xf32) <- (-1x104x14x14xf32, -1x104x14x14xf32)
        t546 = paddle._C_ops.add(t522, t540)
        del t522, t540

        # pd_op.conv2d: (-1x600x14x14xf32) <- (-1x104x14x14xf32, 600x104x1x1xf32)
        t547 = paddle._C_ops.conv2d(
            t546, t167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t546, t167

        # pd_op.batch_norm_: (-1x600x14x14xf32, 600xf32, 600xf32, 600xf32, 600xf32, -1xui8) <- (-1x600x14x14xf32, 600xf32, 600xf32, 600xf32, 600xf32)
        t548, t549, t550, t551, t552, t553 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t547,
                t168,
                t169,
                t170,
                t171,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t547, t169, t168, t171, t170

        # pd_op.hardswish: (-1x600x14x14xf32) <- (-1x600x14x14xf32)
        t554 = paddle._C_ops.hardswish(t548)
        del t548

        # pd_op.depthwise_conv2d: (-1x600x14x14xf32) <- (-1x600x14x14xf32, 600x1x3x3xf32)
        t555 = paddle._C_ops.depthwise_conv2d(
            t554, t172, [1, 1], [1, 1], "EXPLICIT", 600, [1, 1], "NCHW"
        )
        del t554, t172

        # pd_op.batch_norm_: (-1x600x14x14xf32, 600xf32, 600xf32, 600xf32, 600xf32, -1xui8) <- (-1x600x14x14xf32, 600xf32, 600xf32, 600xf32, 600xf32)
        t556, t557, t558, t559, t560, t561 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t555,
                t173,
                t174,
                t175,
                t176,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t555, t176, t175, t174, t173

        # pd_op.hardswish: (-1x600x14x14xf32) <- (-1x600x14x14xf32)
        t562 = paddle._C_ops.hardswish(t556)
        del t556

        # pd_op.pool2d: (-1x600x1x1xf32) <- (-1x600x14x14xf32, 2xi64)
        t563 = paddle._C_ops.pool2d(
            t562,
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

        # pd_op.conv2d: (-1x150x1x1xf32) <- (-1x600x1x1xf32, 150x600x1x1xf32)
        t564 = paddle._C_ops.conv2d(
            t563, t177, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t177, t563

        # pd_op.reshape: (1x150x1x1xf32) <- (150xf32, 4xi64)
        t565 = paddle._C_ops.reshape(t178, t368)
        del t178

        # pd_op.add: (-1x150x1x1xf32) <- (-1x150x1x1xf32, 1x150x1x1xf32)
        t566 = paddle._C_ops.add(t564, t565)
        del t564, t565

        # pd_op.relu: (-1x150x1x1xf32) <- (-1x150x1x1xf32)
        t567 = paddle._C_ops.relu(t566)
        del t566

        # pd_op.conv2d: (-1x600x1x1xf32) <- (-1x150x1x1xf32, 600x150x1x1xf32)
        t568 = paddle._C_ops.conv2d(
            t567, t179, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t179, t567

        # pd_op.reshape: (1x600x1x1xf32) <- (600xf32, 4xi64)
        t569 = paddle._C_ops.reshape(t180, t368)
        del t180

        # pd_op.add: (-1x600x1x1xf32) <- (-1x600x1x1xf32, 1x600x1x1xf32)
        t570 = paddle._C_ops.add(t568, t569)
        del t568, t569

        # pd_op.hardsigmoid: (-1x600x1x1xf32) <- (-1x600x1x1xf32)
        t571 = paddle._C_ops.hardsigmoid(t570, float("0.2"), float("0.5"))
        del t570

        # pd_op.multiply: (-1x600x14x14xf32) <- (-1x600x14x14xf32, -1x600x1x1xf32)
        t572 = paddle._C_ops.multiply(t562, t571)
        del t571, t562

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x600x14x14xf32, 144x600x1x1xf32)
        t573 = paddle._C_ops.conv2d(
            t572, t181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t572, t181

        # pd_op.batch_norm_: (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        t574, t575, t576, t577, t578, t579 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t573,
                t182,
                t183,
                t184,
                t185,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t573, t185, t184, t183, t182

        # pd_op.conv2d: (-1x840x14x14xf32) <- (-1x144x14x14xf32, 840x144x1x1xf32)
        t580 = paddle._C_ops.conv2d(
            t574, t186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t186

        # pd_op.batch_norm_: (-1x840x14x14xf32, 840xf32, 840xf32, 840xf32, 840xf32, -1xui8) <- (-1x840x14x14xf32, 840xf32, 840xf32, 840xf32, 840xf32)
        t581, t582, t583, t584, t585, t586 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t580,
                t187,
                t188,
                t189,
                t190,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t580, t190, t189, t188, t187

        # pd_op.hardswish: (-1x840x14x14xf32) <- (-1x840x14x14xf32)
        t587 = paddle._C_ops.hardswish(t581)
        del t581

        # pd_op.depthwise_conv2d: (-1x840x14x14xf32) <- (-1x840x14x14xf32, 840x1x3x3xf32)
        t588 = paddle._C_ops.depthwise_conv2d(
            t587, t191, [1, 1], [1, 1], "EXPLICIT", 840, [1, 1], "NCHW"
        )
        del t587, t191

        # pd_op.batch_norm_: (-1x840x14x14xf32, 840xf32, 840xf32, 840xf32, 840xf32, -1xui8) <- (-1x840x14x14xf32, 840xf32, 840xf32, 840xf32, 840xf32)
        t589, t590, t591, t592, t593, t594 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t588,
                t192,
                t193,
                t194,
                t195,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t588, t195, t194, t193, t192

        # pd_op.hardswish: (-1x840x14x14xf32) <- (-1x840x14x14xf32)
        t595 = paddle._C_ops.hardswish(t589)
        del t589

        # pd_op.pool2d: (-1x840x1x1xf32) <- (-1x840x14x14xf32, 2xi64)
        t596 = paddle._C_ops.pool2d(
            t595,
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

        # pd_op.conv2d: (-1x210x1x1xf32) <- (-1x840x1x1xf32, 210x840x1x1xf32)
        t597 = paddle._C_ops.conv2d(
            t596, t196, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t196, t596

        # pd_op.reshape: (1x210x1x1xf32) <- (210xf32, 4xi64)
        t598 = paddle._C_ops.reshape(t197, t368)
        del t197

        # pd_op.add: (-1x210x1x1xf32) <- (-1x210x1x1xf32, 1x210x1x1xf32)
        t599 = paddle._C_ops.add(t597, t598)
        del t597, t598

        # pd_op.relu: (-1x210x1x1xf32) <- (-1x210x1x1xf32)
        t600 = paddle._C_ops.relu(t599)
        del t599

        # pd_op.conv2d: (-1x840x1x1xf32) <- (-1x210x1x1xf32, 840x210x1x1xf32)
        t601 = paddle._C_ops.conv2d(
            t600, t198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t198, t600

        # pd_op.reshape: (1x840x1x1xf32) <- (840xf32, 4xi64)
        t602 = paddle._C_ops.reshape(t199, t368)
        del t199

        # pd_op.add: (-1x840x1x1xf32) <- (-1x840x1x1xf32, 1x840x1x1xf32)
        t603 = paddle._C_ops.add(t601, t602)
        del t601, t602

        # pd_op.hardsigmoid: (-1x840x1x1xf32) <- (-1x840x1x1xf32)
        t604 = paddle._C_ops.hardsigmoid(t603, float("0.2"), float("0.5"))
        del t603

        # pd_op.multiply: (-1x840x14x14xf32) <- (-1x840x14x14xf32, -1x840x1x1xf32)
        t605 = paddle._C_ops.multiply(t595, t604)
        del t604, t595

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x840x14x14xf32, 144x840x1x1xf32)
        t606 = paddle._C_ops.conv2d(
            t605, t200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t605, t200

        # pd_op.batch_norm_: (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        t607, t608, t609, t610, t611, t612 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t606,
                t201,
                t202,
                t203,
                t204,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t606, t204, t203, t202, t201

        # pd_op.add: (-1x144x14x14xf32) <- (-1x144x14x14xf32, -1x144x14x14xf32)
        t613 = paddle._C_ops.add(t574, t607)
        del t574, t607

        # pd_op.conv2d: (-1x840x14x14xf32) <- (-1x144x14x14xf32, 840x144x1x1xf32)
        t614 = paddle._C_ops.conv2d(
            t613, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t613, t205

        # pd_op.batch_norm_: (-1x840x14x14xf32, 840xf32, 840xf32, 840xf32, 840xf32, -1xui8) <- (-1x840x14x14xf32, 840xf32, 840xf32, 840xf32, 840xf32)
        t615, t616, t617, t618, t619, t620 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t614,
                t206,
                t207,
                t208,
                t209,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t614, t209, t208, t207, t206

        # pd_op.hardswish: (-1x840x14x14xf32) <- (-1x840x14x14xf32)
        t621 = paddle._C_ops.hardswish(t615)
        del t615

        # pd_op.depthwise_conv2d: (-1x840x7x7xf32) <- (-1x840x14x14xf32, 840x1x5x5xf32)
        t622 = paddle._C_ops.depthwise_conv2d(
            t621, t210, [2, 2], [2, 2], "EXPLICIT", 840, [1, 1], "NCHW"
        )
        del t621, t210

        # pd_op.batch_norm_: (-1x840x7x7xf32, 840xf32, 840xf32, 840xf32, 840xf32, -1xui8) <- (-1x840x7x7xf32, 840xf32, 840xf32, 840xf32, 840xf32)
        t623, t624, t625, t626, t627, t628 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t622,
                t211,
                t212,
                t213,
                t214,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t622, t214, t213, t212, t211

        # pd_op.hardswish: (-1x840x7x7xf32) <- (-1x840x7x7xf32)
        t629 = paddle._C_ops.hardswish(t623)
        del t623

        # pd_op.pool2d: (-1x840x1x1xf32) <- (-1x840x7x7xf32, 2xi64)
        t630 = paddle._C_ops.pool2d(
            t629,
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

        # pd_op.conv2d: (-1x210x1x1xf32) <- (-1x840x1x1xf32, 210x840x1x1xf32)
        t631 = paddle._C_ops.conv2d(
            t630, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215, t630

        # pd_op.reshape: (1x210x1x1xf32) <- (210xf32, 4xi64)
        t632 = paddle._C_ops.reshape(t216, t368)
        del t216

        # pd_op.add: (-1x210x1x1xf32) <- (-1x210x1x1xf32, 1x210x1x1xf32)
        t633 = paddle._C_ops.add(t631, t632)
        del t631, t632

        # pd_op.relu: (-1x210x1x1xf32) <- (-1x210x1x1xf32)
        t634 = paddle._C_ops.relu(t633)
        del t633

        # pd_op.conv2d: (-1x840x1x1xf32) <- (-1x210x1x1xf32, 840x210x1x1xf32)
        t635 = paddle._C_ops.conv2d(
            t634, t217, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t217, t634

        # pd_op.reshape: (1x840x1x1xf32) <- (840xf32, 4xi64)
        t636 = paddle._C_ops.reshape(t218, t368)
        del t218

        # pd_op.add: (-1x840x1x1xf32) <- (-1x840x1x1xf32, 1x840x1x1xf32)
        t637 = paddle._C_ops.add(t635, t636)
        del t635, t636

        # pd_op.hardsigmoid: (-1x840x1x1xf32) <- (-1x840x1x1xf32)
        t638 = paddle._C_ops.hardsigmoid(t637, float("0.2"), float("0.5"))
        del t637

        # pd_op.multiply: (-1x840x7x7xf32) <- (-1x840x7x7xf32, -1x840x1x1xf32)
        t639 = paddle._C_ops.multiply(t629, t638)
        del t638, t629

        # pd_op.conv2d: (-1x200x7x7xf32) <- (-1x840x7x7xf32, 200x840x1x1xf32)
        t640 = paddle._C_ops.conv2d(
            t639, t219, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t639, t219

        # pd_op.batch_norm_: (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32, -1xui8) <- (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        t641, t642, t643, t644, t645, t646 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t640,
                t220,
                t221,
                t222,
                t223,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t640, t223, t222, t221, t220

        # pd_op.conv2d: (-1x1200x7x7xf32) <- (-1x200x7x7xf32, 1200x200x1x1xf32)
        t647 = paddle._C_ops.conv2d(
            t641, t224, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t224

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32, -1xui8) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        t648, t649, t650, t651, t652, t653 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t647,
                t225,
                t226,
                t227,
                t228,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t647, t228, t227, t226, t225

        # pd_op.hardswish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        t654 = paddle._C_ops.hardswish(t648)
        del t648

        # pd_op.depthwise_conv2d: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32, 1200x1x5x5xf32)
        t655 = paddle._C_ops.depthwise_conv2d(
            t654, t229, [1, 1], [2, 2], "EXPLICIT", 1200, [1, 1], "NCHW"
        )
        del t654, t229

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32, -1xui8) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        t656, t657, t658, t659, t660, t661 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t655,
                t230,
                t231,
                t232,
                t233,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t655, t233, t232, t231, t230

        # pd_op.hardswish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        t662 = paddle._C_ops.hardswish(t656)
        del t656

        # pd_op.pool2d: (-1x1200x1x1xf32) <- (-1x1200x7x7xf32, 2xi64)
        t663 = paddle._C_ops.pool2d(
            t662,
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

        # pd_op.conv2d: (-1x300x1x1xf32) <- (-1x1200x1x1xf32, 300x1200x1x1xf32)
        t664 = paddle._C_ops.conv2d(
            t663, t234, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t234, t663

        # pd_op.reshape: (1x300x1x1xf32) <- (300xf32, 4xi64)
        t665 = paddle._C_ops.reshape(t235, t368)
        del t235

        # pd_op.add: (-1x300x1x1xf32) <- (-1x300x1x1xf32, 1x300x1x1xf32)
        t666 = paddle._C_ops.add(t664, t665)
        del t664, t665

        # pd_op.relu: (-1x300x1x1xf32) <- (-1x300x1x1xf32)
        t667 = paddle._C_ops.relu(t666)
        del t666

        # pd_op.conv2d: (-1x1200x1x1xf32) <- (-1x300x1x1xf32, 1200x300x1x1xf32)
        t668 = paddle._C_ops.conv2d(
            t667, t236, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t236, t667

        # pd_op.reshape: (1x1200x1x1xf32) <- (1200xf32, 4xi64)
        t669 = paddle._C_ops.reshape(t237, t368)
        del t237

        # pd_op.add: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32, 1x1200x1x1xf32)
        t670 = paddle._C_ops.add(t668, t669)
        del t668, t669

        # pd_op.hardsigmoid: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32)
        t671 = paddle._C_ops.hardsigmoid(t670, float("0.2"), float("0.5"))
        del t670

        # pd_op.multiply: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32, -1x1200x1x1xf32)
        t672 = paddle._C_ops.multiply(t662, t671)
        del t671, t662

        # pd_op.conv2d: (-1x200x7x7xf32) <- (-1x1200x7x7xf32, 200x1200x1x1xf32)
        t673 = paddle._C_ops.conv2d(
            t672, t238, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t672, t238

        # pd_op.batch_norm_: (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32, -1xui8) <- (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        t674, t675, t676, t677, t678, t679 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t673,
                t239,
                t240,
                t241,
                t242,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t673, t242, t241, t240, t239

        # pd_op.add: (-1x200x7x7xf32) <- (-1x200x7x7xf32, -1x200x7x7xf32)
        t680 = paddle._C_ops.add(t641, t674)
        del t641, t674

        # pd_op.conv2d: (-1x1200x7x7xf32) <- (-1x200x7x7xf32, 1200x200x1x1xf32)
        t681 = paddle._C_ops.conv2d(
            t680, t243, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t243

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32, -1xui8) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        t682, t683, t684, t685, t686, t687 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t681,
                t244,
                t245,
                t246,
                t247,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t681, t247, t246, t245, t244

        # pd_op.hardswish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        t688 = paddle._C_ops.hardswish(t682)
        del t682

        # pd_op.depthwise_conv2d: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32, 1200x1x5x5xf32)
        t689 = paddle._C_ops.depthwise_conv2d(
            t688, t248, [1, 1], [2, 2], "EXPLICIT", 1200, [1, 1], "NCHW"
        )
        del t688, t248

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32, -1xui8) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        t690, t691, t692, t693, t694, t695 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t689,
                t249,
                t250,
                t251,
                t252,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t689, t252, t251, t250, t249

        # pd_op.hardswish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        t696 = paddle._C_ops.hardswish(t690)
        del t690

        # pd_op.pool2d: (-1x1200x1x1xf32) <- (-1x1200x7x7xf32, 2xi64)
        t697 = paddle._C_ops.pool2d(
            t696,
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

        # pd_op.conv2d: (-1x300x1x1xf32) <- (-1x1200x1x1xf32, 300x1200x1x1xf32)
        t698 = paddle._C_ops.conv2d(
            t697, t253, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t253, t697

        # pd_op.reshape: (1x300x1x1xf32) <- (300xf32, 4xi64)
        t699 = paddle._C_ops.reshape(t254, t368)
        del t254

        # pd_op.add: (-1x300x1x1xf32) <- (-1x300x1x1xf32, 1x300x1x1xf32)
        t700 = paddle._C_ops.add(t698, t699)
        del t698, t699

        # pd_op.relu: (-1x300x1x1xf32) <- (-1x300x1x1xf32)
        t701 = paddle._C_ops.relu(t700)
        del t700

        # pd_op.conv2d: (-1x1200x1x1xf32) <- (-1x300x1x1xf32, 1200x300x1x1xf32)
        t702 = paddle._C_ops.conv2d(
            t701, t255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255, t701

        # pd_op.reshape: (1x1200x1x1xf32) <- (1200xf32, 4xi64)
        t703 = paddle._C_ops.reshape(t256, t368)
        del t368, t256

        # pd_op.add: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32, 1x1200x1x1xf32)
        t704 = paddle._C_ops.add(t702, t703)
        del t702, t703

        # pd_op.hardsigmoid: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32)
        t705 = paddle._C_ops.hardsigmoid(t704, float("0.2"), float("0.5"))
        del t704

        # pd_op.multiply: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32, -1x1200x1x1xf32)
        t706 = paddle._C_ops.multiply(t696, t705)
        del t705, t696

        # pd_op.conv2d: (-1x200x7x7xf32) <- (-1x1200x7x7xf32, 200x1200x1x1xf32)
        t707 = paddle._C_ops.conv2d(
            t706, t257, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t706, t257

        # pd_op.batch_norm_: (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32, -1xui8) <- (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        t708, t709, t710, t711, t712, t713 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t707,
                t258,
                t259,
                t260,
                t261,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t707, t259, t258, t261, t260

        # pd_op.add: (-1x200x7x7xf32) <- (-1x200x7x7xf32, -1x200x7x7xf32)
        t714 = paddle._C_ops.add(t680, t708)
        del t680, t708

        # pd_op.conv2d: (-1x1200x7x7xf32) <- (-1x200x7x7xf32, 1200x200x1x1xf32)
        t715 = paddle._C_ops.conv2d(
            t714, t262, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t714, t262

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32, -1xui8) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        t716, t717, t718, t719, t720, t721 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t715,
                t263,
                t264,
                t265,
                t266,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t715, t266, t265, t264, t263

        # pd_op.hardswish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        t722 = paddle._C_ops.hardswish(t716)
        del t716

        # pd_op.pool2d: (-1x1200x1x1xf32) <- (-1x1200x7x7xf32, 2xi64)
        t723 = paddle._C_ops.pool2d(
            t722,
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
        del t365, t722

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x1200x1x1xf32, 1280x1200x1x1xf32)
        t724 = paddle._C_ops.conv2d(
            t723, t267, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t267, t723

        # pd_op.hardswish: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        t725 = paddle._C_ops.hardswish(t724)
        del t724

        # pd_op.full: (1xf32) <- ()
        t726 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x1280x1x1xf32, -1x1280x1x1xui8) <- (-1x1280x1x1xf32, None, 1xf32)
        t727, t728 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                t725, None, t726, True, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t726, t725

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        t729 = paddle._C_ops.flatten(t727, 1, 3)
        del t727

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        t730 = paddle._C_ops.matmul(t729, t268, False, False)
        del t729, t268

        return t730
