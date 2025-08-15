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
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        t276 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t277, t278, t279, t280, t281, t282 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t276,
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
        del t276, t4, t3, t2, t1

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t283 = paddle._C_ops.relu(t277)
        del t277

        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 32x32x3x3xf32)
        t284 = paddle._C_ops.conv2d(
            t283, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5, t283

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t285, t286, t287, t288, t289, t290 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t284,
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
        del t284, t9, t8, t7, t6

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t291 = paddle._C_ops.relu(t285)
        del t285

        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x32x112x112xf32, 64x32x3x3xf32)
        t292 = paddle._C_ops.conv2d(
            t291, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10, t291

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t293, t294, t295, t296, t297, t298 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t292,
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
        del t292, t14, t13, t12, t11

        # pd_op.relu: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        t299 = paddle._C_ops.relu(t293)
        del t293

        # pd_op.full_int_array: (2xi64) <- ()
        t300 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        t301 = paddle._C_ops.pool2d(
            t299,
            t300,
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
        del t300, t299

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        t302 = paddle._C_ops.conv2d(
            t301, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t303, t304, t305, t306, t307, t308 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t302,
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
        del t302, t19, t18, t17, t16

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t309 = paddle._C_ops.relu(t303)
        del t303

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t310 = paddle._C_ops.conv2d(
            t309, t20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20, t309

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t311, t312, t313, t314, t315, t316 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t310,
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
        del t310, t24, t23, t22, t21

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t317 = paddle._C_ops.relu(t311)
        del t311

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t318 = paddle._C_ops.conv2d(
            t317, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25, t317

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t319, t320, t321, t322, t323, t324 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t318,
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
        del t318, t29, t28, t27, t26

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t325 = paddle._C_ops.conv2d(
            t301, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30, t301

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t326, t327, t328, t329, t330, t331 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t325,
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
        del t325, t34, t33, t32, t31

        # pd_op.add: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        t332 = paddle._C_ops.add(t319, t326)
        del t319, t326

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        t333 = paddle._C_ops.relu(t332)
        del t332

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        t334 = paddle._C_ops.conv2d(
            t333, t35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t335, t336, t337, t338, t339, t340 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t334,
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
        del t334, t39, t38, t37, t36

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t341 = paddle._C_ops.relu(t335)
        del t335

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t342 = paddle._C_ops.conv2d(
            t341, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40, t341

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t343, t344, t345, t346, t347, t348 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t342,
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
        del t342, t44, t43, t42, t41

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t349 = paddle._C_ops.relu(t343)
        del t343

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t350 = paddle._C_ops.conv2d(
            t349, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45, t349

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t351, t352, t353, t354, t355, t356 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t350,
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
        del t350, t49, t48, t47, t46

        # pd_op.add: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        t357 = paddle._C_ops.add(t351, t333)
        del t351, t333

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        t358 = paddle._C_ops.relu(t357)
        del t357

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        t359 = paddle._C_ops.conv2d(
            t358, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t360, t361, t362, t363, t364, t365 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t359,
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
        del t359, t54, t53, t52, t51

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t366 = paddle._C_ops.relu(t360)
        del t360

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t367 = paddle._C_ops.conv2d(
            t366, t55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55, t366

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t368, t369, t370, t371, t372, t373 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t367,
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
        del t367, t59, t58, t57, t56

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t374 = paddle._C_ops.relu(t368)
        del t368

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t375 = paddle._C_ops.conv2d(
            t374, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60, t374

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t376, t377, t378, t379, t380, t381 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t375,
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
        del t375, t64, t63, t62, t61

        # pd_op.add: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        t382 = paddle._C_ops.add(t376, t358)
        del t376, t358

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        t383 = paddle._C_ops.relu(t382)
        del t382

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x256x56x56xf32, 128x256x1x1xf32)
        t384 = paddle._C_ops.conv2d(
            t383, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t385, t386, t387, t388, t389, t390 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t384,
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
        del t384, t69, t68, t67, t66

        # pd_op.relu: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        t391 = paddle._C_ops.relu(t385)
        del t385

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x56x56xf32, 128x128x3x3xf32)
        t392 = paddle._C_ops.conv2d(
            t391, t70, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70, t391

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t393, t394, t395, t396, t397, t398 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t392,
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
        del t392, t74, t73, t72, t71

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t399 = paddle._C_ops.relu(t393)
        del t393

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t400 = paddle._C_ops.conv2d(
            t399, t75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75, t399

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t401, t402, t403, t404, t405, t406 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t400,
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
        del t400, t79, t78, t77, t76

        # pd_op.full_int_array: (2xi64) <- ()
        t407 = [2, 2]

        # pd_op.pool2d: (-1x256x28x28xf32) <- (-1x256x56x56xf32, 2xi64)
        t408 = paddle._C_ops.pool2d(
            t383, t407, [2, 2], [0, 0], True, True, "NCHW", "avg", False, False, "SAME"
        )
        del t383

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        t409 = paddle._C_ops.conv2d(
            t408, t80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80, t408

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t410, t411, t412, t413, t414, t415 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t409,
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
        del t409, t84, t83, t82, t81

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t416 = paddle._C_ops.add(t401, t410)
        del t401, t410

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t417 = paddle._C_ops.relu(t416)
        del t416

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t418 = paddle._C_ops.conv2d(
            t417, t85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t419, t420, t421, t422, t423, t424 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t418,
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
        del t418, t89, t88, t87, t86

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t425 = paddle._C_ops.relu(t419)
        del t419

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t426 = paddle._C_ops.conv2d(
            t425, t90, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90, t425

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t427, t428, t429, t430, t431, t432 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t426,
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
        del t426, t94, t93, t92, t91

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t433 = paddle._C_ops.relu(t427)
        del t427

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t434 = paddle._C_ops.conv2d(
            t433, t95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95, t433

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t435, t436, t437, t438, t439, t440 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t434,
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
        del t434, t99, t98, t97, t96

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t441 = paddle._C_ops.add(t435, t417)
        del t435, t417

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t442 = paddle._C_ops.relu(t441)
        del t441

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t443 = paddle._C_ops.conv2d(
            t442, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t444, t445, t446, t447, t448, t449 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t443,
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
        del t443, t104, t103, t102, t101

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t450 = paddle._C_ops.relu(t444)
        del t444

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t451 = paddle._C_ops.conv2d(
            t450, t105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105, t450

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t452, t453, t454, t455, t456, t457 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t451,
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
        del t451, t109, t108, t107, t106

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t458 = paddle._C_ops.relu(t452)
        del t452

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t459 = paddle._C_ops.conv2d(
            t458, t110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110, t458

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t460, t461, t462, t463, t464, t465 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t459,
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
        del t459, t114, t113, t112, t111

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t466 = paddle._C_ops.add(t460, t442)
        del t460, t442

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t467 = paddle._C_ops.relu(t466)
        del t466

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t468 = paddle._C_ops.conv2d(
            t467, t115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t469, t470, t471, t472, t473, t474 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t468,
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
        del t468, t119, t118, t117, t116

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t475 = paddle._C_ops.relu(t469)
        del t469

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t476 = paddle._C_ops.conv2d(
            t475, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120, t475

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t477, t478, t479, t480, t481, t482 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t476,
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
        del t476, t124, t123, t122, t121

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t483 = paddle._C_ops.relu(t477)
        del t477

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t484 = paddle._C_ops.conv2d(
            t483, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125, t483

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t485, t486, t487, t488, t489, t490 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t484,
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
        del t484, t129, t128, t127, t126

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t491 = paddle._C_ops.add(t485, t467)
        del t485, t467

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t492 = paddle._C_ops.relu(t491)
        del t491

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x512x28x28xf32, 256x512x1x1xf32)
        t493 = paddle._C_ops.conv2d(
            t492, t130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t494, t495, t496, t497, t498, t499 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t493,
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
        del t493, t134, t133, t132, t131

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t500 = paddle._C_ops.relu(t494)
        del t494

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x28x28xf32, 256x256x3x3xf32)
        t501 = paddle._C_ops.conv2d(
            t500, t135, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135, t500

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t502, t503, t504, t505, t506, t507 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t501,
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
        del t501, t139, t138, t137, t136

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t508 = paddle._C_ops.relu(t502)
        del t502

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t509 = paddle._C_ops.conv2d(
            t508, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140, t508

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t510, t511, t512, t513, t514, t515 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t509,
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
        del t509, t144, t143, t142, t141

        # pd_op.pool2d: (-1x512x14x14xf32) <- (-1x512x28x28xf32, 2xi64)
        t516 = paddle._C_ops.pool2d(
            t492, t407, [2, 2], [0, 0], True, True, "NCHW", "avg", False, False, "SAME"
        )
        del t492

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t517 = paddle._C_ops.conv2d(
            t516, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145, t516

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t518, t519, t520, t521, t522, t523 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t517,
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
        del t517, t149, t148, t147, t146

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t524 = paddle._C_ops.add(t510, t518)
        del t510, t518

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t525 = paddle._C_ops.relu(t524)
        del t524

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t526 = paddle._C_ops.conv2d(
            t525, t150, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t527, t528, t529, t530, t531, t532 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t526,
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
        del t526, t154, t153, t152, t151

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t533 = paddle._C_ops.relu(t527)
        del t527

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t534 = paddle._C_ops.conv2d(
            t533, t155, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155, t533

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t535, t536, t537, t538, t539, t540 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t534,
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
        del t534, t159, t158, t157, t156

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t541 = paddle._C_ops.relu(t535)
        del t535

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t542 = paddle._C_ops.conv2d(
            t541, t160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160, t541

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t543, t544, t545, t546, t547, t548 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t542,
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
        del t542, t164, t163, t162, t161

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t549 = paddle._C_ops.add(t543, t525)
        del t543, t525

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t550 = paddle._C_ops.relu(t549)
        del t549

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t551 = paddle._C_ops.conv2d(
            t550, t165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t552, t553, t554, t555, t556, t557 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t551,
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
        del t551, t169, t168, t167, t166

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t558 = paddle._C_ops.relu(t552)
        del t552

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t559 = paddle._C_ops.conv2d(
            t558, t170, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170, t558

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t560, t561, t562, t563, t564, t565 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t559,
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
        del t559, t174, t173, t172, t171

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t566 = paddle._C_ops.relu(t560)
        del t560

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t567 = paddle._C_ops.conv2d(
            t566, t175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175, t566

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t568, t569, t570, t571, t572, t573 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t567,
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
        del t567, t176, t179, t178, t177

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t574 = paddle._C_ops.add(t568, t550)
        del t568, t550

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t575 = paddle._C_ops.relu(t574)
        del t574

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t576 = paddle._C_ops.conv2d(
            t575, t180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t577, t578, t579, t580, t581, t582 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t576,
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
        del t576, t184, t183, t182, t181

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t583 = paddle._C_ops.relu(t577)
        del t577

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t584 = paddle._C_ops.conv2d(
            t583, t185, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185, t583

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t585, t586, t587, t588, t589, t590 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t584,
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
        del t584, t189, t188, t187, t186

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t591 = paddle._C_ops.relu(t585)
        del t585

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t592 = paddle._C_ops.conv2d(
            t591, t190, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190, t591

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t593, t594, t595, t596, t597, t598 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t592,
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
        del t592, t194, t193, t192, t191

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t599 = paddle._C_ops.add(t593, t575)
        del t593, t575

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t600 = paddle._C_ops.relu(t599)
        del t599

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t601 = paddle._C_ops.conv2d(
            t600, t195, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t602, t603, t604, t605, t606, t607 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t601,
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
        del t601, t199, t198, t197, t196

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t608 = paddle._C_ops.relu(t602)
        del t602

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t609 = paddle._C_ops.conv2d(
            t608, t200, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200, t608

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t610, t611, t612, t613, t614, t615 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t609,
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
        del t609, t204, t203, t202, t201

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t616 = paddle._C_ops.relu(t610)
        del t610

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t617 = paddle._C_ops.conv2d(
            t616, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205, t616

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t618, t619, t620, t621, t622, t623 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t617,
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
        del t617, t209, t208, t207, t206

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t624 = paddle._C_ops.add(t618, t600)
        del t618, t600

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t625 = paddle._C_ops.relu(t624)
        del t624

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t626 = paddle._C_ops.conv2d(
            t625, t210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t627, t628, t629, t630, t631, t632 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t626,
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
        del t626, t214, t213, t212, t211

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t633 = paddle._C_ops.relu(t627)
        del t627

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t634 = paddle._C_ops.conv2d(
            t633, t215, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215, t633

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t635, t636, t637, t638, t639, t640 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t634,
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
        del t634, t219, t218, t217, t216

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t641 = paddle._C_ops.relu(t635)
        del t635

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t642 = paddle._C_ops.conv2d(
            t641, t220, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t220, t641

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t643, t644, t645, t646, t647, t648 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t642,
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
        del t642, t224, t223, t222, t221

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t649 = paddle._C_ops.add(t643, t625)
        del t643, t625

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t650 = paddle._C_ops.relu(t649)
        del t649

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        t651 = paddle._C_ops.conv2d(
            t650, t225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t652, t653, t654, t655, t656, t657 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t651,
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
        del t651, t229, t228, t227, t226

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t658 = paddle._C_ops.relu(t652)
        del t652

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x14x14xf32, 512x512x3x3xf32)
        t659 = paddle._C_ops.conv2d(
            t658, t230, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t230, t658

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t660, t661, t662, t663, t664, t665 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t659,
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
        del t659, t234, t233, t232, t231

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t666 = paddle._C_ops.relu(t660)
        del t660

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        t667 = paddle._C_ops.conv2d(
            t666, t235, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235, t666

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t668, t669, t670, t671, t672, t673 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t667,
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
        del t667, t239, t238, t237, t236

        # pd_op.pool2d: (-1x1024x7x7xf32) <- (-1x1024x14x14xf32, 2xi64)
        t674 = paddle._C_ops.pool2d(
            t650, t407, [2, 2], [0, 0], True, True, "NCHW", "avg", False, False, "SAME"
        )
        del t407, t650

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        t675 = paddle._C_ops.conv2d(
            t674, t240, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t240, t674

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t676, t677, t678, t679, t680, t681 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t675,
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
        del t675, t244, t243, t242, t241

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t682 = paddle._C_ops.add(t668, t676)
        del t668, t676

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t683 = paddle._C_ops.relu(t682)
        del t682

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x2048x7x7xf32, 512x2048x1x1xf32)
        t684 = paddle._C_ops.conv2d(
            t683, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t685, t686, t687, t688, t689, t690 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t684,
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
        del t684, t249, t248, t247, t246

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t691 = paddle._C_ops.relu(t685)
        del t685

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t692 = paddle._C_ops.conv2d(
            t691, t250, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t250, t691

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t693, t694, t695, t696, t697, t698 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t692,
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
        del t692, t254, t253, t252, t251

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t699 = paddle._C_ops.relu(t693)
        del t693

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        t700 = paddle._C_ops.conv2d(
            t699, t255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255, t699

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t701, t702, t703, t704, t705, t706 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t700,
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
        del t700, t259, t258, t257, t256

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t707 = paddle._C_ops.add(t701, t683)
        del t701, t683

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t708 = paddle._C_ops.relu(t707)
        del t707

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x2048x7x7xf32, 512x2048x1x1xf32)
        t709 = paddle._C_ops.conv2d(
            t708, t260, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t260

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t710, t711, t712, t713, t714, t715 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t709,
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
        del t709, t264, t263, t262, t261

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t716 = paddle._C_ops.relu(t710)
        del t710

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t717 = paddle._C_ops.conv2d(
            t716, t265, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t265, t716

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t718, t719, t720, t721, t722, t723 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t717,
                t266,
                t267,
                t268,
                t269,
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
        del t717, t266, t269, t268, t267

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t724 = paddle._C_ops.relu(t718)
        del t718

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        t725 = paddle._C_ops.conv2d(
            t724, t270, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t270, t724

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t726, t727, t728, t729, t730, t731 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t725,
                t271,
                t272,
                t273,
                t274,
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
        del t725, t274, t273, t272, t271

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t732 = paddle._C_ops.add(t726, t708)
        del t726, t708

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t733 = paddle._C_ops.relu(t732)
        del t732

        # pd_op.full_int_array: (2xi64) <- ()
        t734 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        t735 = paddle._C_ops.pool2d(
            t733,
            t734,
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
        del t734, t733

        # pd_op.flatten: (-1x2048xf32) <- (-1x2048x1x1xf32)
        t736 = paddle._C_ops.flatten(t735, 1, 3)
        del t735

        # pd_op.matmul: (-1x102xf32) <- (-1x2048xf32, 2048x102xf32)
        t737 = paddle._C_ops.matmul(t736, t275, False, False)
        del t736, t275

        return t737
