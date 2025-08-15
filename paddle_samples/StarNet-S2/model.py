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
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        t233 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t234 = [1, -1, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t235 = paddle._C_ops.reshape(t1, t234)
        del t1

        # pd_op.add: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 1x32x1x1xf32)
        t236 = paddle._C_ops.add(t233, t235)

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t237, t238, t239, t240, t241, t242 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t236,
                t2,
                t3,
                t4,
                t5,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t5, t4, t3, t2

        # pd_op.relu6: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t243 = paddle._C_ops.relu6(t237)
        del t237

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x112x112xf32, 32x32x3x3xf32)
        t244 = paddle._C_ops.conv2d(
            t243, t6, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t6

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t245 = paddle._C_ops.reshape(t7, t234)
        del t7

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t246 = paddle._C_ops.add(t244, t245)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t247, t248, t249, t250, t251, t252 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t246,
                t8,
                t9,
                t10,
                t11,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t11, t10, t9, t8

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        t253 = paddle._C_ops.depthwise_conv2d(
            t247, t12, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t12

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t254 = paddle._C_ops.reshape(t13, t234)
        del t13

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t255 = paddle._C_ops.add(t253, t254)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t256, t257, t258, t259, t260, t261 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t255,
                t14,
                t15,
                t16,
                t17,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t17, t16, t15, t14

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        t262 = paddle._C_ops.conv2d(
            t256, t18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t18

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t263 = paddle._C_ops.reshape(t19, t234)
        del t19

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        t264 = paddle._C_ops.add(t262, t263)

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        t265 = paddle._C_ops.conv2d(
            t256, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t266 = paddle._C_ops.reshape(t21, t234)
        del t21

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        t267 = paddle._C_ops.add(t265, t266)

        # pd_op.relu6: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        t268 = paddle._C_ops.relu6(t264)
        del t264

        # pd_op.multiply: (-1x128x56x56xf32) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        t269 = paddle._C_ops.multiply(t268, t267)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x128x56x56xf32, 32x128x1x1xf32)
        t270 = paddle._C_ops.conv2d(
            t269, t22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t22

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t271 = paddle._C_ops.reshape(t23, t234)
        del t23

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t272 = paddle._C_ops.add(t270, t271)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t273, t274, t275, t276, t277, t278 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t272,
                t24,
                t25,
                t26,
                t27,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t27, t26, t25, t24

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        t279 = paddle._C_ops.depthwise_conv2d(
            t273, t28, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t28

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t280 = paddle._C_ops.reshape(t29, t234)
        del t29

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        t281 = paddle._C_ops.add(t279, t280)

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        t282 = paddle._C_ops.add(t247, t281)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x32x56x56xf32, 64x32x3x3xf32)
        t283 = paddle._C_ops.conv2d(
            t282, t30, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t284 = paddle._C_ops.reshape(t31, t234)
        del t31

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t285 = paddle._C_ops.add(t283, t284)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t286, t287, t288, t289, t290, t291 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t285,
                t32,
                t33,
                t34,
                t35,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t35, t34, t33, t32

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t292 = paddle._C_ops.depthwise_conv2d(
            t286, t36, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t36

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t293 = paddle._C_ops.reshape(t37, t234)
        del t37

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t294 = paddle._C_ops.add(t292, t293)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t295, t296, t297, t298, t299, t300 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t294,
                t38,
                t39,
                t40,
                t41,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t41, t40, t39, t38

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t301 = paddle._C_ops.conv2d(
            t295, t42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t42

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t302 = paddle._C_ops.reshape(t43, t234)
        del t43

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t303 = paddle._C_ops.add(t301, t302)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t304 = paddle._C_ops.conv2d(
            t295, t44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t44

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t305 = paddle._C_ops.reshape(t45, t234)
        del t45

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t306 = paddle._C_ops.add(t304, t305)

        # pd_op.relu6: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t307 = paddle._C_ops.relu6(t303)
        del t303

        # pd_op.multiply: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        t308 = paddle._C_ops.multiply(t307, t306)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x256x28x28xf32, 64x256x1x1xf32)
        t309 = paddle._C_ops.conv2d(
            t308, t46, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t46

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t310 = paddle._C_ops.reshape(t47, t234)
        del t47

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t311 = paddle._C_ops.add(t309, t310)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t312, t313, t314, t315, t316, t317 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t311,
                t48,
                t49,
                t50,
                t51,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t51, t50, t49, t48

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t318 = paddle._C_ops.depthwise_conv2d(
            t312, t52, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t52

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t319 = paddle._C_ops.reshape(t53, t234)
        del t53

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t320 = paddle._C_ops.add(t318, t319)

        # pd_op.full: (xf32) <- ()
        t321 = paddle._C_ops.full(
            [],
            float("0.999"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x64x28x28xf32)
        t322 = paddle._C_ops.shape64(t320)

        # pd_op.full_int_array: (1xi64) <- ()
        t323 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        t324 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t325 = paddle._C_ops.slice(t322, [0], t323, t324, [1], [0])
        del t322

        # pd_op.full: (xi64) <- ()
        t326 = paddle._C_ops.full([], float("1"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t327 = [t325, t326, t326, t326]
        del t325

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t328 = paddle._C_ops.stack(t327, 0)
        del t327

        # pd_op.full: (1xf32) <- ()
        t329 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t330 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t331 = paddle._C_ops.uniform(
            t328,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t328

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t332 = paddle._C_ops.add(t321, t331)
        del t331

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t333 = paddle._C_ops.floor(t332)
        del t332

        # pd_op.divide: (-1x64x28x28xf32) <- (-1x64x28x28xf32, xf32)
        t334 = paddle._C_ops.divide(t320, t321)

        # pd_op.multiply: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x1x1x1xf32)
        t335 = paddle._C_ops.multiply(t334, t333)

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        t336 = paddle._C_ops.add(t286, t335)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t337 = paddle._C_ops.depthwise_conv2d(
            t336, t54, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t54

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t338 = paddle._C_ops.reshape(t55, t234)
        del t55

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t339 = paddle._C_ops.add(t337, t338)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t340, t341, t342, t343, t344, t345 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t339,
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

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t346 = paddle._C_ops.conv2d(
            t340, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t347 = paddle._C_ops.reshape(t61, t234)
        del t61

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t348 = paddle._C_ops.add(t346, t347)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        t349 = paddle._C_ops.conv2d(
            t340, t62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t62

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t350 = paddle._C_ops.reshape(t63, t234)
        del t63

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        t351 = paddle._C_ops.add(t349, t350)

        # pd_op.relu6: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t352 = paddle._C_ops.relu6(t348)
        del t348

        # pd_op.multiply: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        t353 = paddle._C_ops.multiply(t352, t351)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x256x28x28xf32, 64x256x1x1xf32)
        t354 = paddle._C_ops.conv2d(
            t353, t64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t64

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t355 = paddle._C_ops.reshape(t65, t234)
        del t65

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t356 = paddle._C_ops.add(t354, t355)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t357, t358, t359, t360, t361, t362 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t356,
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

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        t363 = paddle._C_ops.depthwise_conv2d(
            t357, t70, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t70

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t364 = paddle._C_ops.reshape(t71, t234)
        del t71

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        t365 = paddle._C_ops.add(t363, t364)

        # pd_op.full: (xf32) <- ()
        t366 = paddle._C_ops.full(
            [],
            float("0.998"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x64x28x28xf32)
        t367 = paddle._C_ops.shape64(t365)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t368 = paddle._C_ops.slice(t367, [0], t323, t324, [1], [0])
        del t367

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t369 = [t368, t326, t326, t326]
        del t368

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t370 = paddle._C_ops.stack(t369, 0)
        del t369

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t371 = paddle._C_ops.uniform(
            t370,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t370

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t372 = paddle._C_ops.add(t366, t371)
        del t371

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t373 = paddle._C_ops.floor(t372)
        del t372

        # pd_op.divide: (-1x64x28x28xf32) <- (-1x64x28x28xf32, xf32)
        t374 = paddle._C_ops.divide(t365, t366)

        # pd_op.multiply: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x1x1x1xf32)
        t375 = paddle._C_ops.multiply(t374, t373)

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        t376 = paddle._C_ops.add(t336, t375)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x64x28x28xf32, 128x64x3x3xf32)
        t377 = paddle._C_ops.conv2d(
            t376, t72, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t72

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t378 = paddle._C_ops.reshape(t73, t234)
        del t73

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t379 = paddle._C_ops.add(t377, t378)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t380, t381, t382, t383, t384, t385 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t379,
                t74,
                t75,
                t76,
                t77,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t77, t76, t75, t74

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t386 = paddle._C_ops.depthwise_conv2d(
            t380, t78, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t78

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t387 = paddle._C_ops.reshape(t79, t234)
        del t79

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t388 = paddle._C_ops.add(t386, t387)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t389, t390, t391, t392, t393, t394 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t388,
                t80,
                t81,
                t82,
                t83,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t83, t82, t81, t80

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t395 = paddle._C_ops.conv2d(
            t389, t84, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t84

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t396 = paddle._C_ops.reshape(t85, t234)
        del t85

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t397 = paddle._C_ops.add(t395, t396)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t398 = paddle._C_ops.conv2d(
            t389, t86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t86

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t399 = paddle._C_ops.reshape(t87, t234)
        del t87

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t400 = paddle._C_ops.add(t398, t399)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t401 = paddle._C_ops.relu6(t397)
        del t397

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t402 = paddle._C_ops.multiply(t401, t400)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t403 = paddle._C_ops.conv2d(
            t402, t88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t88

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t404 = paddle._C_ops.reshape(t89, t234)
        del t89

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t405 = paddle._C_ops.add(t403, t404)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t406, t407, t408, t409, t410, t411 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t405,
                t90,
                t91,
                t92,
                t93,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t93, t92, t91, t90

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t412 = paddle._C_ops.depthwise_conv2d(
            t406, t94, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t94

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t413 = paddle._C_ops.reshape(t95, t234)
        del t95

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t414 = paddle._C_ops.add(t412, t413)

        # pd_op.full: (xf32) <- ()
        t415 = paddle._C_ops.full(
            [],
            float("0.997"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        t416 = paddle._C_ops.shape64(t414)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t417 = paddle._C_ops.slice(t416, [0], t323, t324, [1], [0])
        del t416

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t418 = [t417, t326, t326, t326]
        del t417

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t419 = paddle._C_ops.stack(t418, 0)
        del t418

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t420 = paddle._C_ops.uniform(
            t419,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t419

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t421 = paddle._C_ops.add(t415, t420)
        del t420

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t422 = paddle._C_ops.floor(t421)
        del t421

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        t423 = paddle._C_ops.divide(t414, t415)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        t424 = paddle._C_ops.multiply(t423, t422)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t425 = paddle._C_ops.add(t380, t424)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t426 = paddle._C_ops.depthwise_conv2d(
            t425, t96, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t96

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t427 = paddle._C_ops.reshape(t97, t234)
        del t97

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t428 = paddle._C_ops.add(t426, t427)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t429, t430, t431, t432, t433, t434 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t428,
                t98,
                t99,
                t100,
                t101,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t101, t100, t99, t98

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t435 = paddle._C_ops.conv2d(
            t429, t102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t102

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t436 = paddle._C_ops.reshape(t103, t234)
        del t103

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t437 = paddle._C_ops.add(t435, t436)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t438 = paddle._C_ops.conv2d(
            t429, t104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t104

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t439 = paddle._C_ops.reshape(t105, t234)
        del t105

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t440 = paddle._C_ops.add(t438, t439)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t441 = paddle._C_ops.relu6(t437)
        del t437

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t442 = paddle._C_ops.multiply(t441, t440)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t443 = paddle._C_ops.conv2d(
            t442, t106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t106

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t444 = paddle._C_ops.reshape(t107, t234)
        del t107

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t445 = paddle._C_ops.add(t443, t444)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t446, t447, t448, t449, t450, t451 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t445,
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

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t452 = paddle._C_ops.depthwise_conv2d(
            t446, t112, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t112

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t453 = paddle._C_ops.reshape(t113, t234)
        del t113

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t454 = paddle._C_ops.add(t452, t453)

        # pd_op.full: (xf32) <- ()
        t455 = paddle._C_ops.full(
            [],
            float("0.996"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        t456 = paddle._C_ops.shape64(t454)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t457 = paddle._C_ops.slice(t456, [0], t323, t324, [1], [0])
        del t456

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t458 = [t457, t326, t326, t326]
        del t457

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t459 = paddle._C_ops.stack(t458, 0)
        del t458

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t460 = paddle._C_ops.uniform(
            t459,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t459

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t461 = paddle._C_ops.add(t455, t460)
        del t460

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t462 = paddle._C_ops.floor(t461)
        del t461

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        t463 = paddle._C_ops.divide(t454, t455)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        t464 = paddle._C_ops.multiply(t463, t462)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t465 = paddle._C_ops.add(t425, t464)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t466 = paddle._C_ops.depthwise_conv2d(
            t465, t114, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t114

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t467 = paddle._C_ops.reshape(t115, t234)
        del t115

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t468 = paddle._C_ops.add(t466, t467)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t469, t470, t471, t472, t473, t474 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t468,
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

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t475 = paddle._C_ops.conv2d(
            t469, t120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t476 = paddle._C_ops.reshape(t121, t234)
        del t121

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t477 = paddle._C_ops.add(t475, t476)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t478 = paddle._C_ops.conv2d(
            t469, t122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t122

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t479 = paddle._C_ops.reshape(t123, t234)
        del t123

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t480 = paddle._C_ops.add(t478, t479)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t481 = paddle._C_ops.relu6(t477)
        del t477

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t482 = paddle._C_ops.multiply(t481, t480)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t483 = paddle._C_ops.conv2d(
            t482, t124, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t124

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t484 = paddle._C_ops.reshape(t125, t234)
        del t125

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t485 = paddle._C_ops.add(t483, t484)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t486, t487, t488, t489, t490, t491 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t485,
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

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t492 = paddle._C_ops.depthwise_conv2d(
            t486, t130, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t130

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t493 = paddle._C_ops.reshape(t131, t234)
        del t131

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t494 = paddle._C_ops.add(t492, t493)

        # pd_op.full: (xf32) <- ()
        t495 = paddle._C_ops.full(
            [],
            float("0.995"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        t496 = paddle._C_ops.shape64(t494)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t497 = paddle._C_ops.slice(t496, [0], t323, t324, [1], [0])
        del t496

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t498 = [t497, t326, t326, t326]
        del t497

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t499 = paddle._C_ops.stack(t498, 0)
        del t498

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t500 = paddle._C_ops.uniform(
            t499,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t499

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t501 = paddle._C_ops.add(t495, t500)
        del t500

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t502 = paddle._C_ops.floor(t501)
        del t501

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        t503 = paddle._C_ops.divide(t494, t495)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        t504 = paddle._C_ops.multiply(t503, t502)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t505 = paddle._C_ops.add(t465, t504)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t506 = paddle._C_ops.depthwise_conv2d(
            t505, t132, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t132

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t507 = paddle._C_ops.reshape(t133, t234)
        del t133

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t508 = paddle._C_ops.add(t506, t507)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t509, t510, t511, t512, t513, t514 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t508,
                t134,
                t135,
                t136,
                t137,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t137, t136, t135, t134

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t515 = paddle._C_ops.conv2d(
            t509, t138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t138

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t516 = paddle._C_ops.reshape(t139, t234)
        del t139

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t517 = paddle._C_ops.add(t515, t516)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t518 = paddle._C_ops.conv2d(
            t509, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t519 = paddle._C_ops.reshape(t141, t234)
        del t141

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t520 = paddle._C_ops.add(t518, t519)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t521 = paddle._C_ops.relu6(t517)
        del t517

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t522 = paddle._C_ops.multiply(t521, t520)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t523 = paddle._C_ops.conv2d(
            t522, t142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t142

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t524 = paddle._C_ops.reshape(t143, t234)
        del t143

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t525 = paddle._C_ops.add(t523, t524)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t526, t527, t528, t529, t530, t531 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t525,
                t144,
                t145,
                t146,
                t147,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t147, t146, t145, t144

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t532 = paddle._C_ops.depthwise_conv2d(
            t526, t148, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t148

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t533 = paddle._C_ops.reshape(t149, t234)
        del t149

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t534 = paddle._C_ops.add(t532, t533)

        # pd_op.full: (xf32) <- ()
        t535 = paddle._C_ops.full(
            [],
            float("0.994"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        t536 = paddle._C_ops.shape64(t534)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t537 = paddle._C_ops.slice(t536, [0], t323, t324, [1], [0])
        del t536

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t538 = [t537, t326, t326, t326]
        del t537

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t539 = paddle._C_ops.stack(t538, 0)
        del t538

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t540 = paddle._C_ops.uniform(
            t539,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t539

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t541 = paddle._C_ops.add(t535, t540)
        del t540

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t542 = paddle._C_ops.floor(t541)
        del t541

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        t543 = paddle._C_ops.divide(t534, t535)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        t544 = paddle._C_ops.multiply(t543, t542)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t545 = paddle._C_ops.add(t505, t544)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t546 = paddle._C_ops.depthwise_conv2d(
            t545, t150, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t150

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t547 = paddle._C_ops.reshape(t151, t234)
        del t151

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t548 = paddle._C_ops.add(t546, t547)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t549, t550, t551, t552, t553, t554 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t548,
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

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t555 = paddle._C_ops.conv2d(
            t549, t156, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t556 = paddle._C_ops.reshape(t157, t234)
        del t157

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t557 = paddle._C_ops.add(t555, t556)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t558 = paddle._C_ops.conv2d(
            t549, t158, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t158

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t559 = paddle._C_ops.reshape(t159, t234)
        del t159

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t560 = paddle._C_ops.add(t558, t559)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t561 = paddle._C_ops.relu6(t557)
        del t557

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t562 = paddle._C_ops.multiply(t561, t560)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t563 = paddle._C_ops.conv2d(
            t562, t160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t564 = paddle._C_ops.reshape(t161, t234)
        del t161

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t565 = paddle._C_ops.add(t563, t564)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t566, t567, t568, t569, t570, t571 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t565,
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

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t572 = paddle._C_ops.depthwise_conv2d(
            t566, t166, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t166

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t573 = paddle._C_ops.reshape(t167, t234)
        del t167

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t574 = paddle._C_ops.add(t572, t573)

        # pd_op.full: (xf32) <- ()
        t575 = paddle._C_ops.full(
            [],
            float("0.993"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        t576 = paddle._C_ops.shape64(t574)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t577 = paddle._C_ops.slice(t576, [0], t323, t324, [1], [0])
        del t576

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t578 = [t577, t326, t326, t326]
        del t577

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t579 = paddle._C_ops.stack(t578, 0)
        del t578

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t580 = paddle._C_ops.uniform(
            t579,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t579

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t581 = paddle._C_ops.add(t575, t580)
        del t580

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t582 = paddle._C_ops.floor(t581)
        del t581

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        t583 = paddle._C_ops.divide(t574, t575)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        t584 = paddle._C_ops.multiply(t583, t582)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t585 = paddle._C_ops.add(t545, t584)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t586 = paddle._C_ops.depthwise_conv2d(
            t585, t168, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t168

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t587 = paddle._C_ops.reshape(t169, t234)
        del t169

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t588 = paddle._C_ops.add(t586, t587)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t589, t590, t591, t592, t593, t594 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t588,
                t170,
                t171,
                t172,
                t173,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t173, t172, t171, t170

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t595 = paddle._C_ops.conv2d(
            t589, t174, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t174

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t596 = paddle._C_ops.reshape(t175, t234)
        del t175

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t597 = paddle._C_ops.add(t595, t596)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        t598 = paddle._C_ops.conv2d(
            t589, t176, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t176

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        t599 = paddle._C_ops.reshape(t177, t234)
        del t177

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        t600 = paddle._C_ops.add(t598, t599)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t601 = paddle._C_ops.relu6(t597)
        del t597

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t602 = paddle._C_ops.multiply(t601, t600)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t603 = paddle._C_ops.conv2d(
            t602, t178, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t178

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t604 = paddle._C_ops.reshape(t179, t234)
        del t179

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t605 = paddle._C_ops.add(t603, t604)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t606, t607, t608, t609, t610, t611 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t605,
                t180,
                t181,
                t182,
                t183,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t183, t182, t181, t180

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        t612 = paddle._C_ops.depthwise_conv2d(
            t606, t184, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t184

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t613 = paddle._C_ops.reshape(t185, t234)
        del t185

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        t614 = paddle._C_ops.add(t612, t613)

        # pd_op.full: (xf32) <- ()
        t615 = paddle._C_ops.full(
            [],
            float("0.992"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        t616 = paddle._C_ops.shape64(t614)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t617 = paddle._C_ops.slice(t616, [0], t323, t324, [1], [0])
        del t616

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t618 = [t617, t326, t326, t326]
        del t617

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t619 = paddle._C_ops.stack(t618, 0)
        del t618

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t620 = paddle._C_ops.uniform(
            t619,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t619

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t621 = paddle._C_ops.add(t615, t620)
        del t620

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t622 = paddle._C_ops.floor(t621)
        del t621

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        t623 = paddle._C_ops.divide(t614, t615)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        t624 = paddle._C_ops.multiply(t623, t622)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        t625 = paddle._C_ops.add(t585, t624)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x128x14x14xf32, 256x128x3x3xf32)
        t626 = paddle._C_ops.conv2d(
            t625, t186, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t186

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t627 = paddle._C_ops.reshape(t187, t234)
        del t187

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t628 = paddle._C_ops.add(t626, t627)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t629, t630, t631, t632, t633, t634 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t628,
                t188,
                t189,
                t190,
                t191,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t191, t190, t189, t188

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t635 = paddle._C_ops.depthwise_conv2d(
            t629, t192, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t192

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t636 = paddle._C_ops.reshape(t193, t234)
        del t193

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t637 = paddle._C_ops.add(t635, t636)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t638, t639, t640, t641, t642, t643 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t637,
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

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t644 = paddle._C_ops.conv2d(
            t638, t198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t198

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t645 = paddle._C_ops.reshape(t199, t234)
        del t199

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t646 = paddle._C_ops.add(t644, t645)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t647 = paddle._C_ops.conv2d(
            t638, t200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t648 = paddle._C_ops.reshape(t201, t234)
        del t201

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t649 = paddle._C_ops.add(t647, t648)

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t650 = paddle._C_ops.relu6(t646)
        del t646

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        t651 = paddle._C_ops.multiply(t650, t649)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        t652 = paddle._C_ops.conv2d(
            t651, t202, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t202

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t653 = paddle._C_ops.reshape(t203, t234)
        del t203

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t654 = paddle._C_ops.add(t652, t653)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t655, t656, t657, t658, t659, t660 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t654,
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

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t661 = paddle._C_ops.depthwise_conv2d(
            t655, t208, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t208

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t662 = paddle._C_ops.reshape(t209, t234)
        del t209

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t663 = paddle._C_ops.add(t661, t662)

        # pd_op.full: (xf32) <- ()
        t664 = paddle._C_ops.full(
            [],
            float("0.991"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x7x7xf32)
        t665 = paddle._C_ops.shape64(t663)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t666 = paddle._C_ops.slice(t665, [0], t323, t324, [1], [0])
        del t665

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t667 = [t666, t326, t326, t326]
        del t666

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t668 = paddle._C_ops.stack(t667, 0)
        del t667

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t669 = paddle._C_ops.uniform(
            t668,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t668

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t670 = paddle._C_ops.add(t664, t669)
        del t669

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t671 = paddle._C_ops.floor(t670)
        del t670

        # pd_op.divide: (-1x256x7x7xf32) <- (-1x256x7x7xf32, xf32)
        t672 = paddle._C_ops.divide(t663, t664)

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1x1x1xf32)
        t673 = paddle._C_ops.multiply(t672, t671)

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        t674 = paddle._C_ops.add(t629, t673)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t675 = paddle._C_ops.depthwise_conv2d(
            t674, t210, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t210

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t676 = paddle._C_ops.reshape(t211, t234)
        del t211

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t677 = paddle._C_ops.add(t675, t676)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t678, t679, t680, t681, t682, t683 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t677,
                t212,
                t213,
                t214,
                t215,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t215, t214, t213, t212

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t684 = paddle._C_ops.conv2d(
            t678, t216, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t216

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t685 = paddle._C_ops.reshape(t217, t234)
        del t217

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t686 = paddle._C_ops.add(t684, t685)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        t687 = paddle._C_ops.conv2d(
            t678, t218, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t218

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        t688 = paddle._C_ops.reshape(t219, t234)
        del t219

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        t689 = paddle._C_ops.add(t687, t688)

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t690 = paddle._C_ops.relu6(t686)
        del t686

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        t691 = paddle._C_ops.multiply(t690, t689)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        t692 = paddle._C_ops.conv2d(
            t691, t220, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t220

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t693 = paddle._C_ops.reshape(t221, t234)
        del t221

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t694 = paddle._C_ops.add(t692, t693)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t695, t696, t697, t698, t699, t700 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t694,
                t222,
                t223,
                t224,
                t225,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t223, t222, t225, t224

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        t701 = paddle._C_ops.depthwise_conv2d(
            t695, t226, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t226

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t702 = paddle._C_ops.reshape(t227, t234)
        del t234, t227

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        t703 = paddle._C_ops.add(t701, t702)

        # pd_op.full: (xf32) <- ()
        t704 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x7x7xf32)
        t705 = paddle._C_ops.shape64(t703)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t706 = paddle._C_ops.slice(t705, [0], t323, t324, [1], [0])
        del t323, t324, t705

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t707 = [t706, t326, t326, t326]
        del t326, t706

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t708 = paddle._C_ops.stack(t707, 0)
        del t707

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t709 = paddle._C_ops.uniform(
            t708,
            paddle.float32,
            t329,
            t330,
            0,
            paddle.framework._current_expected_place(),
        )
        del t329, t330, t708

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t710 = paddle._C_ops.add(t704, t709)
        del t709

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t711 = paddle._C_ops.floor(t710)
        del t710

        # pd_op.divide: (-1x256x7x7xf32) <- (-1x256x7x7xf32, xf32)
        t712 = paddle._C_ops.divide(t703, t704)

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1x1x1xf32)
        t713 = paddle._C_ops.multiply(t712, t711)

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        t714 = paddle._C_ops.add(t674, t713)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t715, t716, t717, t718, t719, t720 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t714,
                t228,
                t229,
                t230,
                t231,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t231, t230, t229, t228

        # pd_op.full_int_array: (2xi64) <- ()
        t721 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x7x7xf32, 2xi64)
        t722 = paddle._C_ops.pool2d(
            t715,
            t721,
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

        # pd_op.flatten: (-1x256xf32) <- (-1x256x1x1xf32)
        t723 = paddle._C_ops.flatten(t722, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x256xf32, 256x102xf32)
        t724 = paddle._C_ops.matmul(t723, t232, False, False)
        del t232

        return (
            t233,
            t235,
            t236,
            t238,
            t239,
            t240,
            t241,
            t242,
            t243,
            t244,
            t245,
            t246,
            t247,
            t248,
            t249,
            t250,
            t251,
            t252,
            t253,
            t254,
            t255,
            t256,
            t257,
            t258,
            t259,
            t260,
            t261,
            t262,
            t263,
            t265,
            t266,
            t267,
            t268,
            t269,
            t270,
            t271,
            t272,
            t273,
            t274,
            t275,
            t276,
            t277,
            t278,
            t279,
            t280,
            t281,
            t282,
            t283,
            t284,
            t285,
            t286,
            t287,
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
            t311,
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
            t333,
            t334,
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
            t349,
            t350,
            t351,
            t352,
            t353,
            t354,
            t355,
            t356,
            t357,
            t358,
            t359,
            t360,
            t361,
            t362,
            t363,
            t364,
            t365,
            t366,
            t373,
            t374,
            t375,
            t376,
            t377,
            t378,
            t379,
            t380,
            t381,
            t382,
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
            t393,
            t394,
            t395,
            t396,
            t398,
            t399,
            t400,
            t401,
            t402,
            t403,
            t404,
            t405,
            t406,
            t407,
            t408,
            t409,
            t410,
            t411,
            t412,
            t413,
            t414,
            t415,
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
            t435,
            t436,
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
            t450,
            t451,
            t452,
            t453,
            t454,
            t455,
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
            t682,
            t683,
            t684,
            t685,
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
        )
