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
    ):
        t213 = None
        # pd_op.conv2d: (256x8x112x112xf32) <- (256x3x224x224xf32, 8x3x3x3xf32)
        t214 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t215, t216, t217, t218, t219, t220 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t214,
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

        # pd_op.hardswish: (256x8x112x112xf32) <- (256x8x112x112xf32)
        t221 = paddle._C_ops.hardswish(t215)

        # pd_op.conv2d: (256x8x112x112xf32) <- (256x8x112x112xf32, 8x8x1x1xf32)
        t222 = paddle._C_ops.conv2d(
            t221, t5, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (256x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t223, t224, t225, t226, t227, t228 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t222,
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

        # pd_op.relu: (256x8x112x112xf32) <- (256x8x112x112xf32)
        t229 = paddle._C_ops.relu(t223)
        del t223

        # pd_op.depthwise_conv2d: (256x8x56x56xf32) <- (256x8x112x112xf32, 8x1x3x3xf32)
        t230 = paddle._C_ops.depthwise_conv2d(
            t229, t10, [2, 2], [1, 1], "EXPLICIT", 8, [1, 1], "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (256x8x56x56xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (256x8x56x56xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t231, t232, t233, t234, t235, t236 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t230,
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

        # pd_op.relu: (256x8x56x56xf32) <- (256x8x56x56xf32)
        t237 = paddle._C_ops.relu(t231)
        del t231

        # pd_op.full_int_array: (2xi64) <- ()
        t238 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        t239 = t238

        # pd_op.assign: (2xi64) <- (2xi64)
        t240 = t238

        # pd_op.assign: (2xi64) <- (2xi64)
        t241 = t238

        # pd_op.assign: (2xi64) <- (2xi64)
        t242 = t238

        # pd_op.assign: (2xi64) <- (2xi64)
        t243 = t238

        # pd_op.assign: (2xi64) <- (2xi64)
        t244 = t238

        # pd_op.assign: (2xi64) <- (2xi64)
        t245 = t238

        # pd_op.assign: (2xi64) <- (2xi64)
        t246 = t238

        # pd_op.assign: (2xi64) <- (2xi64)
        t247 = t238

        # pd_op.pool2d: (256x8x1x1xf32) <- (256x8x56x56xf32, 2xi64)
        t248 = paddle._C_ops.pool2d(
            t237,
            t238,
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

        # pd_op.conv2d: (256x2x1x1xf32) <- (256x8x1x1xf32, 2x8x1x1xf32)
        t249 = paddle._C_ops.conv2d(
            t248, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.full_int_array: (4xi64) <- ()
        t250 = [1, -1, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        t251 = paddle._C_ops.reshape(t16, t250)
        del t16

        # pd_op.add: (256x2x1x1xf32) <- (256x2x1x1xf32, 1x2x1x1xf32)
        t252 = paddle._C_ops.add(t249, t251)

        # pd_op.relu: (256x2x1x1xf32) <- (256x2x1x1xf32)
        t253 = paddle._C_ops.relu(t252)
        del t252

        # pd_op.conv2d: (256x8x1x1xf32) <- (256x2x1x1xf32, 8x2x1x1xf32)
        t254 = paddle._C_ops.conv2d(
            t253, t17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t17

        # pd_op.reshape: (1x8x1x1xf32) <- (8xf32, 4xi64)
        t255 = paddle._C_ops.reshape(t18, t250)
        del t18

        # pd_op.add: (256x8x1x1xf32) <- (256x8x1x1xf32, 1x8x1x1xf32)
        t256 = paddle._C_ops.add(t254, t255)

        # pd_op.hardsigmoid: (256x8x1x1xf32) <- (256x8x1x1xf32)
        t257 = paddle._C_ops.hardsigmoid(t256, float("0.2"), float("0.5"))
        del t256

        # pd_op.multiply: (256x8x56x56xf32) <- (256x8x56x56xf32, 256x8x1x1xf32)
        t258 = paddle._C_ops.multiply(t237, t257)

        # pd_op.conv2d: (256x8x56x56xf32) <- (256x8x56x56xf32, 8x8x1x1xf32)
        t259 = paddle._C_ops.conv2d(
            t258, t19, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t19

        # pd_op.batch_norm_: (256x8x56x56xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (256x8x56x56xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t260, t261, t262, t263, t264, t265 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t259,
                t20,
                t21,
                t22,
                t23,
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
        del t23, t22, t21, t20

        # pd_op.conv2d: (256x40x56x56xf32) <- (256x8x56x56xf32, 40x8x1x1xf32)
        t266 = paddle._C_ops.conv2d(
            t260, t24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t24

        # pd_op.batch_norm_: (256x40x56x56xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (256x40x56x56xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        t267, t268, t269, t270, t271, t272 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t266,
                t25,
                t26,
                t27,
                t28,
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
        del t28, t27, t26, t25

        # pd_op.relu: (256x40x56x56xf32) <- (256x40x56x56xf32)
        t273 = paddle._C_ops.relu(t267)
        del t267

        # pd_op.depthwise_conv2d: (256x40x28x28xf32) <- (256x40x56x56xf32, 40x1x3x3xf32)
        t274 = paddle._C_ops.depthwise_conv2d(
            t273, t29, [2, 2], [1, 1], "EXPLICIT", 40, [1, 1], "NCHW"
        )
        del t29

        # pd_op.batch_norm_: (256x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (256x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        t275, t276, t277, t278, t279, t280 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t274,
                t30,
                t31,
                t32,
                t33,
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
        del t33, t32, t31, t30

        # pd_op.relu: (256x40x28x28xf32) <- (256x40x28x28xf32)
        t281 = paddle._C_ops.relu(t275)
        del t275

        # pd_op.conv2d: (256x16x28x28xf32) <- (256x40x28x28xf32, 16x40x1x1xf32)
        t282 = paddle._C_ops.conv2d(
            t281, t34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t34

        # pd_op.batch_norm_: (256x16x28x28xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (256x16x28x28xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t283, t284, t285, t286, t287, t288 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t282,
                t35,
                t36,
                t37,
                t38,
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
        del t38, t37, t36, t35

        # pd_op.conv2d: (256x48x28x28xf32) <- (256x16x28x28xf32, 48x16x1x1xf32)
        t289 = paddle._C_ops.conv2d(
            t283, t39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t39

        # pd_op.batch_norm_: (256x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (256x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t290, t291, t292, t293, t294, t295 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t289,
                t40,
                t41,
                t42,
                t43,
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
        del t43, t42, t41, t40

        # pd_op.relu: (256x48x28x28xf32) <- (256x48x28x28xf32)
        t296 = paddle._C_ops.relu(t290)
        del t290

        # pd_op.depthwise_conv2d: (256x48x28x28xf32) <- (256x48x28x28xf32, 48x1x3x3xf32)
        t297 = paddle._C_ops.depthwise_conv2d(
            t296, t44, [1, 1], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t44

        # pd_op.batch_norm_: (256x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (256x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t298, t299, t300, t301, t302, t303 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t297,
                t45,
                t46,
                t47,
                t48,
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
        del t48, t47, t46, t45

        # pd_op.relu: (256x48x28x28xf32) <- (256x48x28x28xf32)
        t304 = paddle._C_ops.relu(t298)
        del t298

        # pd_op.conv2d: (256x16x28x28xf32) <- (256x48x28x28xf32, 16x48x1x1xf32)
        t305 = paddle._C_ops.conv2d(
            t304, t49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t49

        # pd_op.batch_norm_: (256x16x28x28xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (256x16x28x28xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t306, t307, t308, t309, t310, t311 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t305,
                t50,
                t51,
                t52,
                t53,
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
        del t53, t52, t51, t50

        # pd_op.add: (256x16x28x28xf32) <- (256x16x28x28xf32, 256x16x28x28xf32)
        t312 = paddle._C_ops.add(t283, t306)

        # pd_op.conv2d: (256x48x28x28xf32) <- (256x16x28x28xf32, 48x16x1x1xf32)
        t313 = paddle._C_ops.conv2d(
            t312, t54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t54

        # pd_op.batch_norm_: (256x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (256x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t314, t315, t316, t317, t318, t319 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t313,
                t55,
                t56,
                t57,
                t58,
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
        del t58, t57, t56, t55

        # pd_op.hardswish: (256x48x28x28xf32) <- (256x48x28x28xf32)
        t320 = paddle._C_ops.hardswish(t314)

        # pd_op.depthwise_conv2d: (256x48x14x14xf32) <- (256x48x28x28xf32, 48x1x5x5xf32)
        t321 = paddle._C_ops.depthwise_conv2d(
            t320, t59, [2, 2], [2, 2], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t59

        # pd_op.batch_norm_: (256x48x14x14xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (256x48x14x14xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t322, t323, t324, t325, t326, t327 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t321,
                t60,
                t61,
                t62,
                t63,
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
        del t63, t62, t61, t60

        # pd_op.hardswish: (256x48x14x14xf32) <- (256x48x14x14xf32)
        t328 = paddle._C_ops.hardswish(t322)

        # pd_op.pool2d: (256x48x1x1xf32) <- (256x48x14x14xf32, 2xi64)
        t329 = paddle._C_ops.pool2d(
            t328,
            t238,
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

        # pd_op.conv2d: (256x12x1x1xf32) <- (256x48x1x1xf32, 12x48x1x1xf32)
        t330 = paddle._C_ops.conv2d(
            t329, t64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t64

        # pd_op.reshape: (1x12x1x1xf32) <- (12xf32, 4xi64)
        t331 = paddle._C_ops.reshape(t65, t250)
        del t65

        # pd_op.add: (256x12x1x1xf32) <- (256x12x1x1xf32, 1x12x1x1xf32)
        t332 = paddle._C_ops.add(t330, t331)

        # pd_op.relu: (256x12x1x1xf32) <- (256x12x1x1xf32)
        t333 = paddle._C_ops.relu(t332)
        del t332

        # pd_op.conv2d: (256x48x1x1xf32) <- (256x12x1x1xf32, 48x12x1x1xf32)
        t334 = paddle._C_ops.conv2d(
            t333, t66, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t66

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t335 = paddle._C_ops.reshape(t67, t250)
        del t67

        # pd_op.add: (256x48x1x1xf32) <- (256x48x1x1xf32, 1x48x1x1xf32)
        t336 = paddle._C_ops.add(t334, t335)

        # pd_op.hardsigmoid: (256x48x1x1xf32) <- (256x48x1x1xf32)
        t337 = paddle._C_ops.hardsigmoid(t336, float("0.2"), float("0.5"))
        del t336

        # pd_op.multiply: (256x48x14x14xf32) <- (256x48x14x14xf32, 256x48x1x1xf32)
        t338 = paddle._C_ops.multiply(t328, t337)

        # pd_op.conv2d: (256x24x14x14xf32) <- (256x48x14x14xf32, 24x48x1x1xf32)
        t339 = paddle._C_ops.conv2d(
            t338, t68, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t68

        # pd_op.batch_norm_: (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t340, t341, t342, t343, t344, t345 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t339,
                t69,
                t70,
                t71,
                t72,
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
        del t72, t71, t70, t69

        # pd_op.conv2d: (256x120x14x14xf32) <- (256x24x14x14xf32, 120x24x1x1xf32)
        t346 = paddle._C_ops.conv2d(
            t340, t73, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t73

        # pd_op.batch_norm_: (256x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (256x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t347, t348, t349, t350, t351, t352 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t346,
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

        # pd_op.hardswish: (256x120x14x14xf32) <- (256x120x14x14xf32)
        t353 = paddle._C_ops.hardswish(t347)

        # pd_op.depthwise_conv2d: (256x120x14x14xf32) <- (256x120x14x14xf32, 120x1x5x5xf32)
        t354 = paddle._C_ops.depthwise_conv2d(
            t353, t78, [1, 1], [2, 2], "EXPLICIT", 120, [1, 1], "NCHW"
        )
        del t78

        # pd_op.batch_norm_: (256x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (256x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t355, t356, t357, t358, t359, t360 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t354,
                t79,
                t80,
                t81,
                t82,
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
        del t82, t81, t80, t79

        # pd_op.hardswish: (256x120x14x14xf32) <- (256x120x14x14xf32)
        t361 = paddle._C_ops.hardswish(t355)

        # pd_op.pool2d: (256x120x1x1xf32) <- (256x120x14x14xf32, 2xi64)
        t362 = paddle._C_ops.pool2d(
            t361,
            t238,
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

        # pd_op.conv2d: (256x30x1x1xf32) <- (256x120x1x1xf32, 30x120x1x1xf32)
        t363 = paddle._C_ops.conv2d(
            t362, t83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t83

        # pd_op.reshape: (1x30x1x1xf32) <- (30xf32, 4xi64)
        t364 = paddle._C_ops.reshape(t84, t250)
        del t84

        # pd_op.add: (256x30x1x1xf32) <- (256x30x1x1xf32, 1x30x1x1xf32)
        t365 = paddle._C_ops.add(t363, t364)

        # pd_op.relu: (256x30x1x1xf32) <- (256x30x1x1xf32)
        t366 = paddle._C_ops.relu(t365)
        del t365

        # pd_op.conv2d: (256x120x1x1xf32) <- (256x30x1x1xf32, 120x30x1x1xf32)
        t367 = paddle._C_ops.conv2d(
            t366, t85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.reshape: (1x120x1x1xf32) <- (120xf32, 4xi64)
        t368 = paddle._C_ops.reshape(t86, t250)
        del t86

        # pd_op.add: (256x120x1x1xf32) <- (256x120x1x1xf32, 1x120x1x1xf32)
        t369 = paddle._C_ops.add(t367, t368)

        # pd_op.hardsigmoid: (256x120x1x1xf32) <- (256x120x1x1xf32)
        t370 = paddle._C_ops.hardsigmoid(t369, float("0.2"), float("0.5"))
        del t369

        # pd_op.multiply: (256x120x14x14xf32) <- (256x120x14x14xf32, 256x120x1x1xf32)
        t371 = paddle._C_ops.multiply(t361, t370)

        # pd_op.conv2d: (256x24x14x14xf32) <- (256x120x14x14xf32, 24x120x1x1xf32)
        t372 = paddle._C_ops.conv2d(
            t371, t87, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t87

        # pd_op.batch_norm_: (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t373, t374, t375, t376, t377, t378 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t372,
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

        # pd_op.add: (256x24x14x14xf32) <- (256x24x14x14xf32, 256x24x14x14xf32)
        t379 = paddle._C_ops.add(t340, t373)

        # pd_op.conv2d: (256x120x14x14xf32) <- (256x24x14x14xf32, 120x24x1x1xf32)
        t380 = paddle._C_ops.conv2d(
            t379, t92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t92

        # pd_op.batch_norm_: (256x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (256x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t381, t382, t383, t384, t385, t386 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t380,
                t93,
                t94,
                t95,
                t96,
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
        del t96, t95, t94, t93

        # pd_op.hardswish: (256x120x14x14xf32) <- (256x120x14x14xf32)
        t387 = paddle._C_ops.hardswish(t381)

        # pd_op.depthwise_conv2d: (256x120x14x14xf32) <- (256x120x14x14xf32, 120x1x5x5xf32)
        t388 = paddle._C_ops.depthwise_conv2d(
            t387, t97, [1, 1], [2, 2], "EXPLICIT", 120, [1, 1], "NCHW"
        )
        del t97

        # pd_op.batch_norm_: (256x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (256x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        t389, t390, t391, t392, t393, t394 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t388,
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

        # pd_op.hardswish: (256x120x14x14xf32) <- (256x120x14x14xf32)
        t395 = paddle._C_ops.hardswish(t389)

        # pd_op.pool2d: (256x120x1x1xf32) <- (256x120x14x14xf32, 2xi64)
        t396 = paddle._C_ops.pool2d(
            t395,
            t238,
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

        # pd_op.conv2d: (256x30x1x1xf32) <- (256x120x1x1xf32, 30x120x1x1xf32)
        t397 = paddle._C_ops.conv2d(
            t396, t102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t102

        # pd_op.reshape: (1x30x1x1xf32) <- (30xf32, 4xi64)
        t398 = paddle._C_ops.reshape(t103, t250)
        del t103

        # pd_op.add: (256x30x1x1xf32) <- (256x30x1x1xf32, 1x30x1x1xf32)
        t399 = paddle._C_ops.add(t397, t398)

        # pd_op.relu: (256x30x1x1xf32) <- (256x30x1x1xf32)
        t400 = paddle._C_ops.relu(t399)
        del t399

        # pd_op.conv2d: (256x120x1x1xf32) <- (256x30x1x1xf32, 120x30x1x1xf32)
        t401 = paddle._C_ops.conv2d(
            t400, t104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t104

        # pd_op.reshape: (1x120x1x1xf32) <- (120xf32, 4xi64)
        t402 = paddle._C_ops.reshape(t105, t250)
        del t105

        # pd_op.add: (256x120x1x1xf32) <- (256x120x1x1xf32, 1x120x1x1xf32)
        t403 = paddle._C_ops.add(t401, t402)

        # pd_op.hardsigmoid: (256x120x1x1xf32) <- (256x120x1x1xf32)
        t404 = paddle._C_ops.hardsigmoid(t403, float("0.2"), float("0.5"))
        del t403

        # pd_op.multiply: (256x120x14x14xf32) <- (256x120x14x14xf32, 256x120x1x1xf32)
        t405 = paddle._C_ops.multiply(t395, t404)

        # pd_op.conv2d: (256x24x14x14xf32) <- (256x120x14x14xf32, 24x120x1x1xf32)
        t406 = paddle._C_ops.conv2d(
            t405, t106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t106

        # pd_op.batch_norm_: (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t407, t408, t409, t410, t411, t412 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t406,
                t107,
                t108,
                t109,
                t110,
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
        del t110, t109, t108, t107

        # pd_op.add: (256x24x14x14xf32) <- (256x24x14x14xf32, 256x24x14x14xf32)
        t413 = paddle._C_ops.add(t379, t407)

        # pd_op.conv2d: (256x64x14x14xf32) <- (256x24x14x14xf32, 64x24x1x1xf32)
        t414 = paddle._C_ops.conv2d(
            t413, t111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t111

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t415, t416, t417, t418, t419, t420 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t414,
                t112,
                t113,
                t114,
                t115,
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
        del t113, t112, t115, t114

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t421 = paddle._C_ops.hardswish(t415)

        # pd_op.depthwise_conv2d: (256x64x14x14xf32) <- (256x64x14x14xf32, 64x1x5x5xf32)
        t422 = paddle._C_ops.depthwise_conv2d(
            t421, t116, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t116

        # pd_op.batch_norm_: (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (256x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t423, t424, t425, t426, t427, t428 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t422,
                t117,
                t118,
                t119,
                t120,
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
        del t120, t119, t118, t117

        # pd_op.hardswish: (256x64x14x14xf32) <- (256x64x14x14xf32)
        t429 = paddle._C_ops.hardswish(t423)

        # pd_op.pool2d: (256x64x1x1xf32) <- (256x64x14x14xf32, 2xi64)
        t430 = paddle._C_ops.pool2d(
            t429,
            t238,
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

        # pd_op.conv2d: (256x16x1x1xf32) <- (256x64x1x1xf32, 16x64x1x1xf32)
        t431 = paddle._C_ops.conv2d(
            t430, t121, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t121

        # pd_op.reshape: (1x16x1x1xf32) <- (16xf32, 4xi64)
        t432 = paddle._C_ops.reshape(t122, t250)
        del t122

        # pd_op.add: (256x16x1x1xf32) <- (256x16x1x1xf32, 1x16x1x1xf32)
        t433 = paddle._C_ops.add(t431, t432)

        # pd_op.relu: (256x16x1x1xf32) <- (256x16x1x1xf32)
        t434 = paddle._C_ops.relu(t433)
        del t433

        # pd_op.conv2d: (256x64x1x1xf32) <- (256x16x1x1xf32, 64x16x1x1xf32)
        t435 = paddle._C_ops.conv2d(
            t434, t123, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t123

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        t436 = paddle._C_ops.reshape(t124, t250)
        del t124

        # pd_op.add: (256x64x1x1xf32) <- (256x64x1x1xf32, 1x64x1x1xf32)
        t437 = paddle._C_ops.add(t435, t436)

        # pd_op.hardsigmoid: (256x64x1x1xf32) <- (256x64x1x1xf32)
        t438 = paddle._C_ops.hardsigmoid(t437, float("0.2"), float("0.5"))
        del t437

        # pd_op.multiply: (256x64x14x14xf32) <- (256x64x14x14xf32, 256x64x1x1xf32)
        t439 = paddle._C_ops.multiply(t429, t438)

        # pd_op.conv2d: (256x24x14x14xf32) <- (256x64x14x14xf32, 24x64x1x1xf32)
        t440 = paddle._C_ops.conv2d(
            t439, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t441, t442, t443, t444, t445, t446 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t440,
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

        # pd_op.add: (256x24x14x14xf32) <- (256x24x14x14xf32, 256x24x14x14xf32)
        t447 = paddle._C_ops.add(t413, t441)

        # pd_op.conv2d: (256x72x14x14xf32) <- (256x24x14x14xf32, 72x24x1x1xf32)
        t448 = paddle._C_ops.conv2d(
            t447, t130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (256x72x14x14xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (256x72x14x14xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        t449, t450, t451, t452, t453, t454 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t448,
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

        # pd_op.hardswish: (256x72x14x14xf32) <- (256x72x14x14xf32)
        t455 = paddle._C_ops.hardswish(t449)

        # pd_op.depthwise_conv2d: (256x72x14x14xf32) <- (256x72x14x14xf32, 72x1x5x5xf32)
        t456 = paddle._C_ops.depthwise_conv2d(
            t455, t135, [1, 1], [2, 2], "EXPLICIT", 72, [1, 1], "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (256x72x14x14xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (256x72x14x14xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        t457, t458, t459, t460, t461, t462 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t456,
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

        # pd_op.hardswish: (256x72x14x14xf32) <- (256x72x14x14xf32)
        t463 = paddle._C_ops.hardswish(t457)

        # pd_op.pool2d: (256x72x1x1xf32) <- (256x72x14x14xf32, 2xi64)
        t464 = paddle._C_ops.pool2d(
            t463,
            t238,
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

        # pd_op.conv2d: (256x18x1x1xf32) <- (256x72x1x1xf32, 18x72x1x1xf32)
        t465 = paddle._C_ops.conv2d(
            t464, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.reshape: (1x18x1x1xf32) <- (18xf32, 4xi64)
        t466 = paddle._C_ops.reshape(t141, t250)
        del t141

        # pd_op.add: (256x18x1x1xf32) <- (256x18x1x1xf32, 1x18x1x1xf32)
        t467 = paddle._C_ops.add(t465, t466)

        # pd_op.relu: (256x18x1x1xf32) <- (256x18x1x1xf32)
        t468 = paddle._C_ops.relu(t467)
        del t467

        # pd_op.conv2d: (256x72x1x1xf32) <- (256x18x1x1xf32, 72x18x1x1xf32)
        t469 = paddle._C_ops.conv2d(
            t468, t142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t142

        # pd_op.reshape: (1x72x1x1xf32) <- (72xf32, 4xi64)
        t470 = paddle._C_ops.reshape(t143, t250)
        del t143

        # pd_op.add: (256x72x1x1xf32) <- (256x72x1x1xf32, 1x72x1x1xf32)
        t471 = paddle._C_ops.add(t469, t470)

        # pd_op.hardsigmoid: (256x72x1x1xf32) <- (256x72x1x1xf32)
        t472 = paddle._C_ops.hardsigmoid(t471, float("0.2"), float("0.5"))
        del t471

        # pd_op.multiply: (256x72x14x14xf32) <- (256x72x14x14xf32, 256x72x1x1xf32)
        t473 = paddle._C_ops.multiply(t463, t472)

        # pd_op.conv2d: (256x24x14x14xf32) <- (256x72x14x14xf32, 24x72x1x1xf32)
        t474 = paddle._C_ops.conv2d(
            t473, t144, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t144

        # pd_op.batch_norm_: (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (256x24x14x14xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t475, t476, t477, t478, t479, t480 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t474,
                t145,
                t146,
                t147,
                t148,
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
        del t148, t147, t146, t145

        # pd_op.add: (256x24x14x14xf32) <- (256x24x14x14xf32, 256x24x14x14xf32)
        t481 = paddle._C_ops.add(t447, t475)

        # pd_op.conv2d: (256x144x14x14xf32) <- (256x24x14x14xf32, 144x24x1x1xf32)
        t482 = paddle._C_ops.conv2d(
            t481, t149, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t149

        # pd_op.batch_norm_: (256x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (256x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        t483, t484, t485, t486, t487, t488 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t482,
                t150,
                t151,
                t152,
                t153,
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
        del t153, t152, t151, t150

        # pd_op.hardswish: (256x144x14x14xf32) <- (256x144x14x14xf32)
        t489 = paddle._C_ops.hardswish(t483)

        # pd_op.depthwise_conv2d: (256x144x7x7xf32) <- (256x144x14x14xf32, 144x1x5x5xf32)
        t490 = paddle._C_ops.depthwise_conv2d(
            t489, t154, [2, 2], [2, 2], "EXPLICIT", 144, [1, 1], "NCHW"
        )
        del t154

        # pd_op.batch_norm_: (256x144x7x7xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (256x144x7x7xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        t491, t492, t493, t494, t495, t496 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t490,
                t155,
                t156,
                t157,
                t158,
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
        del t158, t157, t156, t155

        # pd_op.hardswish: (256x144x7x7xf32) <- (256x144x7x7xf32)
        t497 = paddle._C_ops.hardswish(t491)

        # pd_op.pool2d: (256x144x1x1xf32) <- (256x144x7x7xf32, 2xi64)
        t498 = paddle._C_ops.pool2d(
            t497,
            t238,
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

        # pd_op.conv2d: (256x36x1x1xf32) <- (256x144x1x1xf32, 36x144x1x1xf32)
        t499 = paddle._C_ops.conv2d(
            t498, t159, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t159

        # pd_op.reshape: (1x36x1x1xf32) <- (36xf32, 4xi64)
        t500 = paddle._C_ops.reshape(t160, t250)
        del t160

        # pd_op.add: (256x36x1x1xf32) <- (256x36x1x1xf32, 1x36x1x1xf32)
        t501 = paddle._C_ops.add(t499, t500)

        # pd_op.relu: (256x36x1x1xf32) <- (256x36x1x1xf32)
        t502 = paddle._C_ops.relu(t501)
        del t501

        # pd_op.conv2d: (256x144x1x1xf32) <- (256x36x1x1xf32, 144x36x1x1xf32)
        t503 = paddle._C_ops.conv2d(
            t502, t161, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t161

        # pd_op.reshape: (1x144x1x1xf32) <- (144xf32, 4xi64)
        t504 = paddle._C_ops.reshape(t162, t250)
        del t162

        # pd_op.add: (256x144x1x1xf32) <- (256x144x1x1xf32, 1x144x1x1xf32)
        t505 = paddle._C_ops.add(t503, t504)

        # pd_op.hardsigmoid: (256x144x1x1xf32) <- (256x144x1x1xf32)
        t506 = paddle._C_ops.hardsigmoid(t505, float("0.2"), float("0.5"))
        del t505

        # pd_op.multiply: (256x144x7x7xf32) <- (256x144x7x7xf32, 256x144x1x1xf32)
        t507 = paddle._C_ops.multiply(t497, t506)

        # pd_op.conv2d: (256x48x7x7xf32) <- (256x144x7x7xf32, 48x144x1x1xf32)
        t508 = paddle._C_ops.conv2d(
            t507, t163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t163

        # pd_op.batch_norm_: (256x48x7x7xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (256x48x7x7xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t509, t510, t511, t512, t513, t514 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t508,
                t164,
                t165,
                t166,
                t167,
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
        del t167, t166, t165, t164

        # pd_op.conv2d: (256x288x7x7xf32) <- (256x48x7x7xf32, 288x48x1x1xf32)
        t515 = paddle._C_ops.conv2d(
            t509, t168, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t168

        # pd_op.batch_norm_: (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t516, t517, t518, t519, t520, t521 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t515,
                t169,
                t170,
                t171,
                t172,
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
        del t172, t171, t170, t169

        # pd_op.hardswish: (256x288x7x7xf32) <- (256x288x7x7xf32)
        t522 = paddle._C_ops.hardswish(t516)

        # pd_op.depthwise_conv2d: (256x288x7x7xf32) <- (256x288x7x7xf32, 288x1x5x5xf32)
        t523 = paddle._C_ops.depthwise_conv2d(
            t522, t173, [1, 1], [2, 2], "EXPLICIT", 288, [1, 1], "NCHW"
        )
        del t173

        # pd_op.batch_norm_: (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t524, t525, t526, t527, t528, t529 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t523,
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

        # pd_op.hardswish: (256x288x7x7xf32) <- (256x288x7x7xf32)
        t530 = paddle._C_ops.hardswish(t524)

        # pd_op.pool2d: (256x288x1x1xf32) <- (256x288x7x7xf32, 2xi64)
        t531 = paddle._C_ops.pool2d(
            t530,
            t238,
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

        # pd_op.conv2d: (256x72x1x1xf32) <- (256x288x1x1xf32, 72x288x1x1xf32)
        t532 = paddle._C_ops.conv2d(
            t531, t178, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t178

        # pd_op.reshape: (1x72x1x1xf32) <- (72xf32, 4xi64)
        t533 = paddle._C_ops.reshape(t179, t250)
        del t179

        # pd_op.add: (256x72x1x1xf32) <- (256x72x1x1xf32, 1x72x1x1xf32)
        t534 = paddle._C_ops.add(t532, t533)

        # pd_op.relu: (256x72x1x1xf32) <- (256x72x1x1xf32)
        t535 = paddle._C_ops.relu(t534)
        del t534

        # pd_op.conv2d: (256x288x1x1xf32) <- (256x72x1x1xf32, 288x72x1x1xf32)
        t536 = paddle._C_ops.conv2d(
            t535, t180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180

        # pd_op.reshape: (1x288x1x1xf32) <- (288xf32, 4xi64)
        t537 = paddle._C_ops.reshape(t181, t250)
        del t181

        # pd_op.add: (256x288x1x1xf32) <- (256x288x1x1xf32, 1x288x1x1xf32)
        t538 = paddle._C_ops.add(t536, t537)

        # pd_op.hardsigmoid: (256x288x1x1xf32) <- (256x288x1x1xf32)
        t539 = paddle._C_ops.hardsigmoid(t538, float("0.2"), float("0.5"))
        del t538

        # pd_op.multiply: (256x288x7x7xf32) <- (256x288x7x7xf32, 256x288x1x1xf32)
        t540 = paddle._C_ops.multiply(t530, t539)

        # pd_op.conv2d: (256x48x7x7xf32) <- (256x288x7x7xf32, 48x288x1x1xf32)
        t541 = paddle._C_ops.conv2d(
            t540, t182, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t182

        # pd_op.batch_norm_: (256x48x7x7xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (256x48x7x7xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t542, t543, t544, t545, t546, t547 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t541,
                t183,
                t184,
                t185,
                t186,
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
        del t186, t185, t184, t183

        # pd_op.add: (256x48x7x7xf32) <- (256x48x7x7xf32, 256x48x7x7xf32)
        t548 = paddle._C_ops.add(t509, t542)

        # pd_op.conv2d: (256x288x7x7xf32) <- (256x48x7x7xf32, 288x48x1x1xf32)
        t549 = paddle._C_ops.conv2d(
            t548, t187, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t187

        # pd_op.batch_norm_: (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t550, t551, t552, t553, t554, t555 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t549,
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

        # pd_op.hardswish: (256x288x7x7xf32) <- (256x288x7x7xf32)
        t556 = paddle._C_ops.hardswish(t550)

        # pd_op.depthwise_conv2d: (256x288x7x7xf32) <- (256x288x7x7xf32, 288x1x5x5xf32)
        t557 = paddle._C_ops.depthwise_conv2d(
            t556, t192, [1, 1], [2, 2], "EXPLICIT", 288, [1, 1], "NCHW"
        )
        del t192

        # pd_op.batch_norm_: (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t558, t559, t560, t561, t562, t563 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t557,
                t193,
                t194,
                t195,
                t196,
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
        del t196, t195, t194, t193

        # pd_op.hardswish: (256x288x7x7xf32) <- (256x288x7x7xf32)
        t564 = paddle._C_ops.hardswish(t558)

        # pd_op.pool2d: (256x288x1x1xf32) <- (256x288x7x7xf32, 2xi64)
        t565 = paddle._C_ops.pool2d(
            t564,
            t238,
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

        # pd_op.conv2d: (256x72x1x1xf32) <- (256x288x1x1xf32, 72x288x1x1xf32)
        t566 = paddle._C_ops.conv2d(
            t565, t197, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t197

        # pd_op.reshape: (1x72x1x1xf32) <- (72xf32, 4xi64)
        t567 = paddle._C_ops.reshape(t198, t250)
        del t198

        # pd_op.add: (256x72x1x1xf32) <- (256x72x1x1xf32, 1x72x1x1xf32)
        t568 = paddle._C_ops.add(t566, t567)

        # pd_op.relu: (256x72x1x1xf32) <- (256x72x1x1xf32)
        t569 = paddle._C_ops.relu(t568)
        del t568

        # pd_op.conv2d: (256x288x1x1xf32) <- (256x72x1x1xf32, 288x72x1x1xf32)
        t570 = paddle._C_ops.conv2d(
            t569, t199, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t199

        # pd_op.reshape: (1x288x1x1xf32) <- (288xf32, 4xi64)
        t571 = paddle._C_ops.reshape(t200, t250)
        del t250, t200

        # pd_op.add: (256x288x1x1xf32) <- (256x288x1x1xf32, 1x288x1x1xf32)
        t572 = paddle._C_ops.add(t570, t571)

        # pd_op.hardsigmoid: (256x288x1x1xf32) <- (256x288x1x1xf32)
        t573 = paddle._C_ops.hardsigmoid(t572, float("0.2"), float("0.5"))
        del t572

        # pd_op.multiply: (256x288x7x7xf32) <- (256x288x7x7xf32, 256x288x1x1xf32)
        t574 = paddle._C_ops.multiply(t564, t573)

        # pd_op.conv2d: (256x48x7x7xf32) <- (256x288x7x7xf32, 48x288x1x1xf32)
        t575 = paddle._C_ops.conv2d(
            t574, t201, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t201

        # pd_op.batch_norm_: (256x48x7x7xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (256x48x7x7xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t576, t577, t578, t579, t580, t581 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t575,
                t202,
                t203,
                t204,
                t205,
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
        del t203, t202, t205, t204

        # pd_op.add: (256x48x7x7xf32) <- (256x48x7x7xf32, 256x48x7x7xf32)
        t582 = paddle._C_ops.add(t548, t576)

        # pd_op.conv2d: (256x288x7x7xf32) <- (256x48x7x7xf32, 288x48x1x1xf32)
        t583 = paddle._C_ops.conv2d(
            t582, t206, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t206

        # pd_op.batch_norm_: (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (256x288x7x7xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t584, t585, t586, t587, t588, t589 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t583,
                t207,
                t208,
                t209,
                t210,
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
        del t210, t209, t208, t207

        # pd_op.hardswish: (256x288x7x7xf32) <- (256x288x7x7xf32)
        t590 = paddle._C_ops.hardswish(t584)

        # pd_op.pool2d: (256x288x1x1xf32) <- (256x288x7x7xf32, 2xi64)
        t591 = paddle._C_ops.pool2d(
            t590,
            t238,
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

        # pd_op.conv2d: (256x1280x1x1xf32) <- (256x288x1x1xf32, 1280x288x1x1xf32)
        t592 = paddle._C_ops.conv2d(
            t591, t211, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t211

        # pd_op.hardswish: (256x1280x1x1xf32) <- (256x1280x1x1xf32)
        t593 = paddle._C_ops.hardswish(t592)

        # pd_op.full: (1xf32) <- ()
        t594 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (256x1280x1x1xf32, 256x1280x1x1xui8) <- (256x1280x1x1xf32, None, 1xf32)
        t595, t596 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                t593, None, t594, False, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t593

        # pd_op.flatten: (256x1280xf32) <- (256x1280x1x1xf32)
        t597 = paddle._C_ops.flatten(t595, 1, 3)

        # pd_op.matmul: (256x102xf32) <- (256x1280xf32, 1280x102xf32)
        t598 = paddle._C_ops.matmul(t597, t212, False, False)
        del t212

        return (
            t214,
            t215,
            t216,
            t217,
            t218,
            t219,
            t220,
            t221,
            t222,
            t224,
            t225,
            t226,
            t227,
            t228,
            t229,
            t230,
            t232,
            t233,
            t234,
            t235,
            t236,
            t237,
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
            t251,
            t253,
            t254,
            t255,
            t257,
            t258,
            t259,
            t260,
            t261,
            t262,
            t263,
            t264,
            t265,
            t266,
            t268,
            t269,
            t270,
            t271,
            t272,
            t273,
            t274,
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
            t291,
            t292,
            t293,
            t294,
            t295,
            t296,
            t297,
            t299,
            t300,
            t301,
            t302,
            t303,
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
            t322,
            t323,
            t324,
            t325,
            t326,
            t327,
            t328,
            t329,
            t330,
            t331,
            t333,
            t334,
            t335,
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
            t366,
            t367,
            t368,
            t370,
            t371,
            t372,
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
            t397,
            t398,
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
            t411,
            t412,
            t413,
            t414,
            t415,
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
            t427,
            t428,
            t429,
            t430,
            t431,
            t432,
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
            t469,
            t470,
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
            t502,
            t503,
            t504,
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
            t535,
            t536,
            t537,
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
            t569,
            t570,
            t571,
            t573,
            t574,
            t575,
            t576,
            t577,
            t578,
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
            t594,
            t595,
            t596,
            t597,
            t598,
        )
