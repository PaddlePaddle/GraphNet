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
    ):
        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x3x224x224xf32, 64x3x7x7xf32)
        t266 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t267, t268, t269, t270, t271, t272 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t266,
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
        del t266, t4, t3, t2, t1

        # pd_op.relu: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        t273 = paddle._C_ops.relu(t267)
        del t267

        # pd_op.full_int_array: (2xi64) <- ()
        t274 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        t275 = paddle._C_ops.pool2d(
            t273,
            t274,
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
        del t274, t273

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        t276 = paddle._C_ops.conv2d(
            t275, t5, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t277, t278, t279, t280, t281, t282 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t276,
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
        del t276, t9, t8, t7, t6

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t283 = paddle._C_ops.relu(t277)
        del t277

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t284 = paddle._C_ops.conv2d(
            t283, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10, t283

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t285, t286, t287, t288, t289, t290 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t284,
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
        del t284, t14, t13, t12, t11

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t291 = paddle._C_ops.relu(t285)
        del t285

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t292 = paddle._C_ops.conv2d(
            t291, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15, t291

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t293, t294, t295, t296, t297, t298 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t292,
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
        del t292, t19, t18, t17, t16

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t299 = paddle._C_ops.conv2d(
            t275, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20, t275

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t300, t301, t302, t303, t304, t305 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t299,
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
        del t299, t24, t23, t22, t21

        # pd_op.add: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        t306 = paddle._C_ops.add(t293, t300)
        del t293, t300

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        t307 = paddle._C_ops.relu(t306)
        del t306

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        t308 = paddle._C_ops.conv2d(
            t307, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t309, t310, t311, t312, t313, t314 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t308,
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
        del t308, t29, t28, t27, t26

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t315 = paddle._C_ops.relu(t309)
        del t309

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t316 = paddle._C_ops.conv2d(
            t315, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30, t315

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t317, t318, t319, t320, t321, t322 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t316,
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
        del t316, t34, t33, t32, t31

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t323 = paddle._C_ops.relu(t317)
        del t317

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t324 = paddle._C_ops.conv2d(
            t323, t35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35, t323

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t325, t326, t327, t328, t329, t330 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t324,
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
        del t324, t39, t38, t37, t36

        # pd_op.add: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        t331 = paddle._C_ops.add(t325, t307)
        del t325, t307

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        t332 = paddle._C_ops.relu(t331)
        del t331

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        t333 = paddle._C_ops.conv2d(
            t332, t40, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t340 = paddle._C_ops.relu(t334)
        del t334

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t341 = paddle._C_ops.conv2d(
            t340, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45, t340

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t348 = paddle._C_ops.relu(t342)
        del t342

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t349 = paddle._C_ops.conv2d(
            t348, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50, t348

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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

        # pd_op.add: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        t356 = paddle._C_ops.add(t350, t332)
        del t350, t332

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        t357 = paddle._C_ops.relu(t356)
        del t356

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x256x56x56xf32, 128x256x1x1xf32)
        t358 = paddle._C_ops.conv2d(
            t357, t55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t359, t360, t361, t362, t363, t364 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t358,
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
        del t358, t59, t58, t57, t56

        # pd_op.relu: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        t365 = paddle._C_ops.relu(t359)
        del t359

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x56x56xf32, 128x128x3x3xf32)
        t366 = paddle._C_ops.conv2d(
            t365, t60, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60, t365

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t367, t368, t369, t370, t371, t372 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t366,
                t61,
                t62,
                t63,
                t64,
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
        del t366, t64, t63, t62, t61

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t373 = paddle._C_ops.relu(t367)
        del t367

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t374 = paddle._C_ops.conv2d(
            t373, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65, t373

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t375, t376, t377, t378, t379, t380 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t374,
                t66,
                t67,
                t68,
                t69,
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
        del t374, t69, t68, t67, t66

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x56x56xf32, 512x256x1x1xf32)
        t381 = paddle._C_ops.conv2d(
            t357, t70, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70, t357

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t382, t383, t384, t385, t386, t387 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t381,
                t71,
                t72,
                t73,
                t74,
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
        del t381, t74, t73, t72, t71

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t388 = paddle._C_ops.add(t375, t382)
        del t375, t382

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t389 = paddle._C_ops.relu(t388)
        del t388

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t390 = paddle._C_ops.conv2d(
            t389, t75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t391, t392, t393, t394, t395, t396 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t390,
                t76,
                t77,
                t78,
                t79,
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
        del t390, t79, t78, t77, t76

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t397 = paddle._C_ops.relu(t391)
        del t391

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t398 = paddle._C_ops.conv2d(
            t397, t80, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80, t397

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t399, t400, t401, t402, t403, t404 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t398,
                t81,
                t82,
                t83,
                t84,
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
        del t398, t84, t83, t82, t81

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t405 = paddle._C_ops.relu(t399)
        del t399

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t406 = paddle._C_ops.conv2d(
            t405, t85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85, t405

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t407, t408, t409, t410, t411, t412 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t406,
                t86,
                t87,
                t88,
                t89,
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
        del t406, t89, t88, t87, t86

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t413 = paddle._C_ops.add(t407, t389)
        del t407, t389

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t414 = paddle._C_ops.relu(t413)
        del t413

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t415 = paddle._C_ops.conv2d(
            t414, t90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t416, t417, t418, t419, t420, t421 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t415,
                t91,
                t92,
                t93,
                t94,
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
        del t415, t94, t93, t92, t91

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t422 = paddle._C_ops.relu(t416)
        del t416

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t423 = paddle._C_ops.conv2d(
            t422, t95, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95, t422

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t424, t425, t426, t427, t428, t429 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t423,
                t96,
                t97,
                t98,
                t99,
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
        del t423, t99, t98, t97, t96

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t430 = paddle._C_ops.relu(t424)
        del t424

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t431 = paddle._C_ops.conv2d(
            t430, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100, t430

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t432, t433, t434, t435, t436, t437 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t431,
                t101,
                t102,
                t103,
                t104,
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
        del t431, t104, t103, t102, t101

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t438 = paddle._C_ops.add(t432, t414)
        del t432, t414

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t439 = paddle._C_ops.relu(t438)
        del t438

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t440 = paddle._C_ops.conv2d(
            t439, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t441, t442, t443, t444, t445, t446 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t440,
                t106,
                t107,
                t108,
                t109,
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
        del t440, t109, t108, t107, t106

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t447 = paddle._C_ops.relu(t441)
        del t441

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t448 = paddle._C_ops.conv2d(
            t447, t110, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110, t447

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t449, t450, t451, t452, t453, t454 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t448,
                t111,
                t112,
                t113,
                t114,
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
        del t448, t114, t113, t112, t111

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t455 = paddle._C_ops.relu(t449)
        del t449

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t456 = paddle._C_ops.conv2d(
            t455, t115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115, t455

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t457, t458, t459, t460, t461, t462 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t456,
                t116,
                t117,
                t118,
                t119,
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
        del t456, t119, t118, t117, t116

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t463 = paddle._C_ops.add(t457, t439)
        del t457, t439

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t464 = paddle._C_ops.relu(t463)
        del t463

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x512x28x28xf32, 256x512x1x1xf32)
        t465 = paddle._C_ops.conv2d(
            t464, t120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t466, t467, t468, t469, t470, t471 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t465,
                t121,
                t122,
                t123,
                t124,
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
        del t465, t124, t123, t122, t121

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t472 = paddle._C_ops.relu(t466)
        del t466

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x28x28xf32, 256x256x3x3xf32)
        t473 = paddle._C_ops.conv2d(
            t472, t125, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125, t472

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t474, t475, t476, t477, t478, t479 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t473,
                t126,
                t127,
                t128,
                t129,
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
        del t473, t129, t128, t127, t126

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t480 = paddle._C_ops.relu(t474)
        del t474

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t481 = paddle._C_ops.conv2d(
            t480, t130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130, t480

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t482, t483, t484, t485, t486, t487 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t481,
                t131,
                t132,
                t133,
                t134,
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
        del t481, t134, t133, t132, t131

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x28x28xf32, 1024x512x1x1xf32)
        t488 = paddle._C_ops.conv2d(
            t464, t135, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135, t464

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t489, t490, t491, t492, t493, t494 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t488,
                t136,
                t137,
                t138,
                t139,
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
        del t488, t139, t138, t137, t136

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t495 = paddle._C_ops.add(t482, t489)
        del t482, t489

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t496 = paddle._C_ops.relu(t495)
        del t495

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t497 = paddle._C_ops.conv2d(
            t496, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t498, t499, t500, t501, t502, t503 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t497,
                t141,
                t142,
                t143,
                t144,
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
        del t497, t144, t143, t142, t141

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t504 = paddle._C_ops.relu(t498)
        del t498

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t505 = paddle._C_ops.conv2d(
            t504, t145, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145, t504

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t506, t507, t508, t509, t510, t511 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t505,
                t146,
                t147,
                t148,
                t149,
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
        del t505, t149, t148, t147, t146

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t512 = paddle._C_ops.relu(t506)
        del t506

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t513 = paddle._C_ops.conv2d(
            t512, t150, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t150, t512

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t514, t515, t516, t517, t518, t519 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t513,
                t151,
                t152,
                t153,
                t154,
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
        del t513, t154, t153, t152, t151

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t520 = paddle._C_ops.add(t514, t496)
        del t514, t496

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t521 = paddle._C_ops.relu(t520)
        del t520

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t522 = paddle._C_ops.conv2d(
            t521, t155, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t523, t524, t525, t526, t527, t528 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t522,
                t156,
                t157,
                t158,
                t159,
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
        del t522, t159, t158, t157, t156

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t529 = paddle._C_ops.relu(t523)
        del t523

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t530 = paddle._C_ops.conv2d(
            t529, t160, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160, t529

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t531, t532, t533, t534, t535, t536 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t530,
                t161,
                t162,
                t163,
                t164,
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
        del t530, t164, t163, t162, t161

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t537 = paddle._C_ops.relu(t531)
        del t531

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t538 = paddle._C_ops.conv2d(
            t537, t165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165, t537

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t539, t540, t541, t542, t543, t544 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t538,
                t166,
                t167,
                t168,
                t169,
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
        del t538, t166, t169, t168, t167

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t545 = paddle._C_ops.add(t539, t521)
        del t539, t521

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t546 = paddle._C_ops.relu(t545)
        del t545

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t547 = paddle._C_ops.conv2d(
            t546, t170, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t548, t549, t550, t551, t552, t553 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t547,
                t171,
                t172,
                t173,
                t174,
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
        del t547, t174, t173, t172, t171

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t554 = paddle._C_ops.relu(t548)
        del t548

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t555 = paddle._C_ops.conv2d(
            t554, t175, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175, t554

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t556, t557, t558, t559, t560, t561 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t555,
                t176,
                t177,
                t178,
                t179,
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
        del t555, t179, t178, t177, t176

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t562 = paddle._C_ops.relu(t556)
        del t556

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t563 = paddle._C_ops.conv2d(
            t562, t180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180, t562

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t564, t565, t566, t567, t568, t569 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t563,
                t181,
                t182,
                t183,
                t184,
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
        del t563, t184, t183, t182, t181

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t570 = paddle._C_ops.add(t564, t546)
        del t564, t546

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t571 = paddle._C_ops.relu(t570)
        del t570

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t572 = paddle._C_ops.conv2d(
            t571, t185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t573, t574, t575, t576, t577, t578 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t572,
                t186,
                t187,
                t188,
                t189,
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
        del t572, t189, t188, t187, t186

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t579 = paddle._C_ops.relu(t573)
        del t573

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t580 = paddle._C_ops.conv2d(
            t579, t190, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190, t579

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t581, t582, t583, t584, t585, t586 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t580,
                t191,
                t192,
                t193,
                t194,
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
        del t580, t194, t193, t192, t191

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t587 = paddle._C_ops.relu(t581)
        del t581

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t588 = paddle._C_ops.conv2d(
            t587, t195, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195, t587

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t589, t590, t591, t592, t593, t594 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t588,
                t196,
                t197,
                t198,
                t199,
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
        del t588, t199, t198, t197, t196

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t595 = paddle._C_ops.add(t589, t571)
        del t589, t571

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t596 = paddle._C_ops.relu(t595)
        del t595

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t597 = paddle._C_ops.conv2d(
            t596, t200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t598, t599, t600, t601, t602, t603 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t597,
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
        del t597, t204, t203, t202, t201

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t604 = paddle._C_ops.relu(t598)
        del t598

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t605 = paddle._C_ops.conv2d(
            t604, t205, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205, t604

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t606, t607, t608, t609, t610, t611 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t605,
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
        del t605, t209, t208, t207, t206

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t612 = paddle._C_ops.relu(t606)
        del t606

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t613 = paddle._C_ops.conv2d(
            t612, t210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210, t612

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t614, t615, t616, t617, t618, t619 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t613,
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
        del t613, t214, t213, t212, t211

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t620 = paddle._C_ops.add(t614, t596)
        del t614, t596

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t621 = paddle._C_ops.relu(t620)
        del t620

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        t622 = paddle._C_ops.conv2d(
            t621, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t623, t624, t625, t626, t627, t628 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t622,
                t216,
                t217,
                t218,
                t219,
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
        del t622, t219, t218, t217, t216

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t629 = paddle._C_ops.relu(t623)
        del t623

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x14x14xf32, 512x512x3x3xf32)
        t630 = paddle._C_ops.conv2d(
            t629, t220, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t220, t629

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t631, t632, t633, t634, t635, t636 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t630,
                t221,
                t222,
                t223,
                t224,
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
        del t630, t224, t223, t222, t221

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t637 = paddle._C_ops.relu(t631)
        del t631

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        t638 = paddle._C_ops.conv2d(
            t637, t225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225, t637

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t639, t640, t641, t642, t643, t644 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t638,
                t226,
                t227,
                t228,
                t229,
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
        del t638, t229, t228, t227, t226

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x14x14xf32, 2048x1024x1x1xf32)
        t645 = paddle._C_ops.conv2d(
            t621, t230, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t230, t621

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t646, t647, t648, t649, t650, t651 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t645,
                t231,
                t232,
                t233,
                t234,
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
        del t645, t234, t233, t232, t231

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t652 = paddle._C_ops.add(t639, t646)
        del t639, t646

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t653 = paddle._C_ops.relu(t652)
        del t652

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x2048x7x7xf32, 512x2048x1x1xf32)
        t654 = paddle._C_ops.conv2d(
            t653, t235, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t655, t656, t657, t658, t659, t660 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t654,
                t236,
                t237,
                t238,
                t239,
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
        del t654, t239, t238, t237, t236

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t661 = paddle._C_ops.relu(t655)
        del t655

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t662 = paddle._C_ops.conv2d(
            t661, t240, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t240, t661

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t663, t664, t665, t666, t667, t668 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t662,
                t241,
                t242,
                t243,
                t244,
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
        del t662, t244, t243, t242, t241

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t669 = paddle._C_ops.relu(t663)
        del t663

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        t670 = paddle._C_ops.conv2d(
            t669, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245, t669

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t671, t672, t673, t674, t675, t676 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t670,
                t246,
                t247,
                t248,
                t249,
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
        del t670, t249, t248, t247, t246

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t677 = paddle._C_ops.add(t671, t653)
        del t671, t653

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t678 = paddle._C_ops.relu(t677)
        del t677

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x2048x7x7xf32, 512x2048x1x1xf32)
        t679 = paddle._C_ops.conv2d(
            t678, t250, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t250

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t680, t681, t682, t683, t684, t685 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t679,
                t251,
                t252,
                t253,
                t254,
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
        del t679, t254, t253, t252, t251

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t686 = paddle._C_ops.relu(t680)
        del t680

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t687 = paddle._C_ops.conv2d(
            t686, t255, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255, t686

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t688, t689, t690, t691, t692, t693 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t687,
                t256,
                t257,
                t258,
                t259,
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
        del t687, t256, t259, t258, t257

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t694 = paddle._C_ops.relu(t688)
        del t688

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        t695 = paddle._C_ops.conv2d(
            t694, t260, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t260, t694

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t696, t697, t698, t699, t700, t701 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t695,
                t261,
                t262,
                t263,
                t264,
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
        del t695, t264, t263, t262, t261

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t702 = paddle._C_ops.add(t696, t678)
        del t696, t678

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t703 = paddle._C_ops.relu(t702)
        del t702

        # pd_op.full_int_array: (2xi64) <- ()
        t704 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        t705 = paddle._C_ops.pool2d(
            t703,
            t704,
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
        del t704, t703

        # pd_op.flatten: (-1x2048xf32) <- (-1x2048x1x1xf32)
        t706 = paddle._C_ops.flatten(t705, 1, 3)
        del t705

        # pd_op.matmul: (-1x102xf32) <- (-1x2048xf32, 2048x102xf32)
        t707 = paddle._C_ops.matmul(t706, t265, False, False)
        del t706, t265

        return t707
