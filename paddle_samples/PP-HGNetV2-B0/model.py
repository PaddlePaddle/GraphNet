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
        input1: paddle.Tensor,
        input2: paddle.Tensor,
        t5: paddle.Tensor,
        t6: paddle.Tensor,
        t7: paddle.Tensor,
        t8: paddle.Tensor,
        t9: paddle.Tensor,
        input3: paddle.Tensor,
        input4: paddle.Tensor,
        t10: paddle.Tensor,
        t11: paddle.Tensor,
        t12: paddle.Tensor,
        t13: paddle.Tensor,
        t14: paddle.Tensor,
        input5: paddle.Tensor,
        input6: paddle.Tensor,
        t15: paddle.Tensor,
        t16: paddle.Tensor,
        t17: paddle.Tensor,
        t18: paddle.Tensor,
        t19: paddle.Tensor,
        input7: paddle.Tensor,
        input8: paddle.Tensor,
        t20: paddle.Tensor,
        t21: paddle.Tensor,
        t22: paddle.Tensor,
        t23: paddle.Tensor,
        t24: paddle.Tensor,
        input9: paddle.Tensor,
        input10: paddle.Tensor,
        t25: paddle.Tensor,
        t26: paddle.Tensor,
        t27: paddle.Tensor,
        t28: paddle.Tensor,
        t29: paddle.Tensor,
        input11: paddle.Tensor,
        input12: paddle.Tensor,
        t30: paddle.Tensor,
        t31: paddle.Tensor,
        t32: paddle.Tensor,
        t33: paddle.Tensor,
        t34: paddle.Tensor,
        input13: paddle.Tensor,
        input14: paddle.Tensor,
        t35: paddle.Tensor,
        t36: paddle.Tensor,
        t37: paddle.Tensor,
        t38: paddle.Tensor,
        t39: paddle.Tensor,
        input15: paddle.Tensor,
        input16: paddle.Tensor,
        t40: paddle.Tensor,
        t41: paddle.Tensor,
        t42: paddle.Tensor,
        t43: paddle.Tensor,
        t44: paddle.Tensor,
        input17: paddle.Tensor,
        input18: paddle.Tensor,
        t45: paddle.Tensor,
        t46: paddle.Tensor,
        t47: paddle.Tensor,
        t48: paddle.Tensor,
        t49: paddle.Tensor,
        input19: paddle.Tensor,
        input20: paddle.Tensor,
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
        input21: paddle.Tensor,
        input22: paddle.Tensor,
        t60: paddle.Tensor,
        t61: paddle.Tensor,
        t62: paddle.Tensor,
        t63: paddle.Tensor,
        t64: paddle.Tensor,
        input23: paddle.Tensor,
        input24: paddle.Tensor,
        t65: paddle.Tensor,
        t66: paddle.Tensor,
        t67: paddle.Tensor,
        t68: paddle.Tensor,
        t69: paddle.Tensor,
        input25: paddle.Tensor,
        input26: paddle.Tensor,
        t70: paddle.Tensor,
        t71: paddle.Tensor,
        t72: paddle.Tensor,
        t73: paddle.Tensor,
        t74: paddle.Tensor,
        input27: paddle.Tensor,
        input28: paddle.Tensor,
        t75: paddle.Tensor,
        t76: paddle.Tensor,
        t77: paddle.Tensor,
        t78: paddle.Tensor,
        t79: paddle.Tensor,
        input29: paddle.Tensor,
        input30: paddle.Tensor,
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
        input31: paddle.Tensor,
        input32: paddle.Tensor,
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
        input33: paddle.Tensor,
        input34: paddle.Tensor,
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
        input35: paddle.Tensor,
        input36: paddle.Tensor,
        t115: paddle.Tensor,
        t116: paddle.Tensor,
        t117: paddle.Tensor,
        t118: paddle.Tensor,
        t119: paddle.Tensor,
        input37: paddle.Tensor,
        input38: paddle.Tensor,
        t120: paddle.Tensor,
        t121: paddle.Tensor,
        t122: paddle.Tensor,
        t123: paddle.Tensor,
        t124: paddle.Tensor,
        input39: paddle.Tensor,
        input40: paddle.Tensor,
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
        input41: paddle.Tensor,
        input42: paddle.Tensor,
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
        input43: paddle.Tensor,
        input44: paddle.Tensor,
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
        input45: paddle.Tensor,
        input46: paddle.Tensor,
        t155: paddle.Tensor,
        t156: paddle.Tensor,
        t157: paddle.Tensor,
        t158: paddle.Tensor,
        t159: paddle.Tensor,
        input47: paddle.Tensor,
        input48: paddle.Tensor,
        t160: paddle.Tensor,
        t161: paddle.Tensor,
        t162: paddle.Tensor,
        t163: paddle.Tensor,
        t164: paddle.Tensor,
        input49: paddle.Tensor,
        input50: paddle.Tensor,
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
        input51: paddle.Tensor,
        input52: paddle.Tensor,
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
        input53: paddle.Tensor,
        input54: paddle.Tensor,
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
        input55: paddle.Tensor,
        input56: paddle.Tensor,
        t200: paddle.Tensor,
        t201: paddle.Tensor,
        t202: paddle.Tensor,
        t203: paddle.Tensor,
        t204: paddle.Tensor,
        input57: paddle.Tensor,
        input58: paddle.Tensor,
        t205: paddle.Tensor,
        t206: paddle.Tensor,
        t207: paddle.Tensor,
        t208: paddle.Tensor,
        t209: paddle.Tensor,
        input59: paddle.Tensor,
        input60: paddle.Tensor,
        t210: paddle.Tensor,
        input61: paddle.Tensor,
        input62: paddle.Tensor,
        t211: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x16x112x112xf32) <- (-1x3x224x224xf32, 16x3x3x3xf32)
        t212 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t213, t214, t215, t216, t217, t218 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t212,
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
        del t212, t4, t3, t2, t1

        # pd_op.relu: (-1x16x112x112xf32) <- (-1x16x112x112xf32)
        t219 = paddle._C_ops.relu(t213)
        del t213

        # pd_op.multiply: (-1x16x112x112xf32) <- (1xf32, -1x16x112x112xf32)
        t220 = paddle._C_ops.multiply(input1, t219)
        del input1, t219

        # pd_op.add: (-1x16x112x112xf32) <- (-1x16x112x112xf32, 1xf32)
        t221 = paddle._C_ops.add(t220, input2)
        del input2, t220

        # pd_op.conv2d: (-1x8x112x112xf32) <- (-1x16x112x112xf32, 8x16x2x2xf32)
        t222 = paddle._C_ops.conv2d(t221, t5, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW")
        del t5

        # pd_op.batch_norm_: (-1x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (-1x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        t223, t224, t225, t226, t227, t228 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t222,
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
        del t222, t9, t8, t7, t6

        # pd_op.relu: (-1x8x112x112xf32) <- (-1x8x112x112xf32)
        t229 = paddle._C_ops.relu(t223)
        del t223

        # pd_op.multiply: (-1x8x112x112xf32) <- (1xf32, -1x8x112x112xf32)
        t230 = paddle._C_ops.multiply(input3, t229)
        del input3, t229

        # pd_op.add: (-1x8x112x112xf32) <- (-1x8x112x112xf32, 1xf32)
        t231 = paddle._C_ops.add(t230, input4)
        del input4, t230

        # pd_op.conv2d: (-1x16x112x112xf32) <- (-1x8x112x112xf32, 16x8x2x2xf32)
        t232 = paddle._C_ops.conv2d(
            t231, t10, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del t231, t10

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t233, t234, t235, t236, t237, t238 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t232,
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
        del t232, t14, t13, t12, t11

        # pd_op.relu: (-1x16x112x112xf32) <- (-1x16x112x112xf32)
        t239 = paddle._C_ops.relu(t233)
        del t233

        # pd_op.multiply: (-1x16x112x112xf32) <- (1xf32, -1x16x112x112xf32)
        t240 = paddle._C_ops.multiply(input5, t239)
        del input5, t239

        # pd_op.add: (-1x16x112x112xf32) <- (-1x16x112x112xf32, 1xf32)
        t241 = paddle._C_ops.add(t240, input6)
        del input6, t240

        # pd_op.full_int_array: (2xi64) <- ()
        t242 = [2, 2]

        # pd_op.pool2d: (-1x16x112x112xf32) <- (-1x16x112x112xf32, 2xi64)
        t243 = paddle._C_ops.pool2d(
            t221, t242, [1, 1], [0, 0], True, True, "NCHW", "max", False, False, "SAME"
        )
        del t221, t242

        # pd_op.full: (1xi32) <- ()
        t244 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x16x112x112xf32, -1x16x112x112xf32]) <- (-1x16x112x112xf32, -1x16x112x112xf32)
        t245 = [t243, t241]
        del t241, t243

        # pd_op.concat: (-1x32x112x112xf32) <- ([-1x16x112x112xf32, -1x16x112x112xf32], 1xi32)
        t246 = paddle._C_ops.concat(t245, t244)
        del t245

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x32x112x112xf32, 16x32x3x3xf32)
        t247 = paddle._C_ops.conv2d(
            t246, t15, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t246, t15

        # pd_op.batch_norm_: (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t248, t249, t250, t251, t252, t253 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t247,
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
        del t247, t19, t18, t17, t16

        # pd_op.relu: (-1x16x56x56xf32) <- (-1x16x56x56xf32)
        t254 = paddle._C_ops.relu(t248)
        del t248

        # pd_op.multiply: (-1x16x56x56xf32) <- (1xf32, -1x16x56x56xf32)
        t255 = paddle._C_ops.multiply(input7, t254)
        del input7, t254

        # pd_op.add: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 1xf32)
        t256 = paddle._C_ops.add(t255, input8)
        del input8, t255

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 16x16x1x1xf32)
        t257 = paddle._C_ops.conv2d(
            t256, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t256, t20

        # pd_op.batch_norm_: (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t258, t259, t260, t261, t262, t263 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t257,
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
        del t257, t24, t23, t22, t21

        # pd_op.relu: (-1x16x56x56xf32) <- (-1x16x56x56xf32)
        t264 = paddle._C_ops.relu(t258)
        del t258

        # pd_op.multiply: (-1x16x56x56xf32) <- (1xf32, -1x16x56x56xf32)
        t265 = paddle._C_ops.multiply(input9, t264)
        del input9, t264

        # pd_op.add: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 1xf32)
        t266 = paddle._C_ops.add(t265, input10)
        del input10, t265

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 16x16x3x3xf32)
        t267 = paddle._C_ops.conv2d(
            t266, t25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t268, t269, t270, t271, t272, t273 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t267,
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
        del t267, t29, t28, t27, t26

        # pd_op.relu: (-1x16x56x56xf32) <- (-1x16x56x56xf32)
        t274 = paddle._C_ops.relu(t268)
        del t268

        # pd_op.multiply: (-1x16x56x56xf32) <- (1xf32, -1x16x56x56xf32)
        t275 = paddle._C_ops.multiply(input11, t274)
        del input11, t274

        # pd_op.add: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 1xf32)
        t276 = paddle._C_ops.add(t275, input12)
        del input12, t275

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 16x16x3x3xf32)
        t277 = paddle._C_ops.conv2d(
            t276, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t278, t279, t280, t281, t282, t283 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t277,
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
        del t277, t34, t33, t32, t31

        # pd_op.relu: (-1x16x56x56xf32) <- (-1x16x56x56xf32)
        t284 = paddle._C_ops.relu(t278)
        del t278

        # pd_op.multiply: (-1x16x56x56xf32) <- (1xf32, -1x16x56x56xf32)
        t285 = paddle._C_ops.multiply(input13, t284)
        del input13, t284

        # pd_op.add: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 1xf32)
        t286 = paddle._C_ops.add(t285, input14)
        del input14, t285

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 16x16x3x3xf32)
        t287 = paddle._C_ops.conv2d(
            t286, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t288, t289, t290, t291, t292, t293 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t287,
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
        del t287, t39, t38, t37, t36

        # pd_op.relu: (-1x16x56x56xf32) <- (-1x16x56x56xf32)
        t294 = paddle._C_ops.relu(t288)
        del t288

        # pd_op.multiply: (-1x16x56x56xf32) <- (1xf32, -1x16x56x56xf32)
        t295 = paddle._C_ops.multiply(input15, t294)
        del input15, t294

        # pd_op.add: (-1x16x56x56xf32) <- (-1x16x56x56xf32, 1xf32)
        t296 = paddle._C_ops.add(t295, input16)
        del input16, t295

        # builtin.combine: ([-1x16x56x56xf32, -1x16x56x56xf32, -1x16x56x56xf32, -1x16x56x56xf32]) <- (-1x16x56x56xf32, -1x16x56x56xf32, -1x16x56x56xf32, -1x16x56x56xf32)
        t297 = [t266, t276, t286, t296]
        del t266, t276, t286, t296

        # pd_op.concat: (-1x64x56x56xf32) <- ([-1x16x56x56xf32, -1x16x56x56xf32, -1x16x56x56xf32, -1x16x56x56xf32], 1xi32)
        t298 = paddle._C_ops.concat(t297, t244)
        del t297

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x64x56x56xf32, 32x64x1x1xf32)
        t299 = paddle._C_ops.conv2d(
            t298, t40, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t298, t40

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t300, t301, t302, t303, t304, t305 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t299,
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
        del t299, t44, t43, t42, t41

        # pd_op.relu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        t306 = paddle._C_ops.relu(t300)
        del t300

        # pd_op.multiply: (-1x32x56x56xf32) <- (1xf32, -1x32x56x56xf32)
        t307 = paddle._C_ops.multiply(input17, t306)
        del input17, t306

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1xf32)
        t308 = paddle._C_ops.add(t307, input18)
        del input18, t307

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x32x56x56xf32, 64x32x1x1xf32)
        t309 = paddle._C_ops.conv2d(
            t308, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t308, t45

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t310, t311, t312, t313, t314, t315 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t309,
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
        del t309, t49, t48, t47, t46

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t316 = paddle._C_ops.relu(t310)
        del t310

        # pd_op.multiply: (-1x64x56x56xf32) <- (1xf32, -1x64x56x56xf32)
        t317 = paddle._C_ops.multiply(input19, t316)
        del input19, t316

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1xf32)
        t318 = paddle._C_ops.add(t317, input20)
        del input20, t317

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x56x56xf32, 64x1x3x3xf32)
        t319 = paddle._C_ops.depthwise_conv2d(
            t318, t50, [2, 2], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t318, t50

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t320, t321, t322, t323, t324, t325 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t319,
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
        del t319, t54, t53, t52, t51

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x64x28x28xf32, 32x64x3x3xf32)
        t326 = paddle._C_ops.conv2d(
            t320, t55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t327, t328, t329, t330, t331, t332 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t326,
                t56,
                t57,
                t58,
                t59,
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
        del t326, t59, t58, t57, t56

        # pd_op.relu: (-1x32x28x28xf32) <- (-1x32x28x28xf32)
        t333 = paddle._C_ops.relu(t327)
        del t327

        # pd_op.multiply: (-1x32x28x28xf32) <- (1xf32, -1x32x28x28xf32)
        t334 = paddle._C_ops.multiply(input21, t333)
        del input21, t333

        # pd_op.add: (-1x32x28x28xf32) <- (-1x32x28x28xf32, 1xf32)
        t335 = paddle._C_ops.add(t334, input22)
        del input22, t334

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x32x28x28xf32, 32x32x3x3xf32)
        t336 = paddle._C_ops.conv2d(
            t335, t60, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t337, t338, t339, t340, t341, t342 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t336,
                t61,
                t62,
                t63,
                t64,
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
        del t336, t64, t63, t62, t61

        # pd_op.relu: (-1x32x28x28xf32) <- (-1x32x28x28xf32)
        t343 = paddle._C_ops.relu(t337)
        del t337

        # pd_op.multiply: (-1x32x28x28xf32) <- (1xf32, -1x32x28x28xf32)
        t344 = paddle._C_ops.multiply(input23, t343)
        del input23, t343

        # pd_op.add: (-1x32x28x28xf32) <- (-1x32x28x28xf32, 1xf32)
        t345 = paddle._C_ops.add(t344, input24)
        del input24, t344

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x32x28x28xf32, 32x32x3x3xf32)
        t346 = paddle._C_ops.conv2d(
            t345, t65, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t347, t348, t349, t350, t351, t352 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t346,
                t66,
                t67,
                t68,
                t69,
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
        del t346, t69, t68, t67, t66

        # pd_op.relu: (-1x32x28x28xf32) <- (-1x32x28x28xf32)
        t353 = paddle._C_ops.relu(t347)
        del t347

        # pd_op.multiply: (-1x32x28x28xf32) <- (1xf32, -1x32x28x28xf32)
        t354 = paddle._C_ops.multiply(input25, t353)
        del input25, t353

        # pd_op.add: (-1x32x28x28xf32) <- (-1x32x28x28xf32, 1xf32)
        t355 = paddle._C_ops.add(t354, input26)
        del input26, t354

        # builtin.combine: ([-1x64x28x28xf32, -1x32x28x28xf32, -1x32x28x28xf32, -1x32x28x28xf32]) <- (-1x64x28x28xf32, -1x32x28x28xf32, -1x32x28x28xf32, -1x32x28x28xf32)
        t356 = [t320, t335, t345, t355]
        del t335, t345, t355, t320

        # pd_op.concat: (-1x160x28x28xf32) <- ([-1x64x28x28xf32, -1x32x28x28xf32, -1x32x28x28xf32, -1x32x28x28xf32], 1xi32)
        t357 = paddle._C_ops.concat(t356, t244)
        del t356

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x160x28x28xf32, 128x160x1x1xf32)
        t358 = paddle._C_ops.conv2d(
            t357, t70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t357, t70

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t359, t360, t361, t362, t363, t364 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t358,
                t71,
                t72,
                t73,
                t74,
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
        del t358, t74, t73, t72, t71

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t365 = paddle._C_ops.relu(t359)
        del t359

        # pd_op.multiply: (-1x128x28x28xf32) <- (1xf32, -1x128x28x28xf32)
        t366 = paddle._C_ops.multiply(input27, t365)
        del input27, t365

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1xf32)
        t367 = paddle._C_ops.add(t366, input28)
        del input28, t366

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x128x28x28xf32, 256x128x1x1xf32)
        t368 = paddle._C_ops.conv2d(
            t367, t75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t367, t75

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t369, t370, t371, t372, t373, t374 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t368,
                t76,
                t77,
                t78,
                t79,
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
        del t368, t79, t78, t77, t76

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t375 = paddle._C_ops.relu(t369)
        del t369

        # pd_op.multiply: (-1x256x28x28xf32) <- (1xf32, -1x256x28x28xf32)
        t376 = paddle._C_ops.multiply(input29, t375)
        del input29, t375

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1xf32)
        t377 = paddle._C_ops.add(t376, input30)
        del input30, t376

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x28x28xf32, 256x1x3x3xf32)
        t378 = paddle._C_ops.depthwise_conv2d(
            t377, t80, [2, 2], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t377, t80

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t379, t380, t381, t382, t383, t384 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t378,
                t81,
                t82,
                t83,
                t84,
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
        del t378, t84, t83, t82, t81

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        t385 = paddle._C_ops.conv2d(
            t379, t85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t386, t387, t388, t389, t390, t391 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t385,
                t86,
                t87,
                t88,
                t89,
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
        del t385, t89, t88, t87, t86

        # pd_op.depthwise_conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x1x5x5xf32)
        t392 = paddle._C_ops.depthwise_conv2d(
            t386, t90, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t386, t90

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t393, t394, t395, t396, t397, t398 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t392,
                t91,
                t92,
                t93,
                t94,
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
        del t392, t94, t93, t92, t91

        # pd_op.relu: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        t399 = paddle._C_ops.relu(t393)
        del t393

        # pd_op.multiply: (-1x64x14x14xf32) <- (1xf32, -1x64x14x14xf32)
        t400 = paddle._C_ops.multiply(input31, t399)
        del input31, t399

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 1xf32)
        t401 = paddle._C_ops.add(t400, input32)
        del input32, t400

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x1x1xf32)
        t402 = paddle._C_ops.conv2d(
            t401, t95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t403, t404, t405, t406, t407, t408 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t402,
                t96,
                t97,
                t98,
                t99,
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
        del t402, t99, t98, t97, t96

        # pd_op.depthwise_conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x1x5x5xf32)
        t409 = paddle._C_ops.depthwise_conv2d(
            t403, t100, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t403, t100

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t410, t411, t412, t413, t414, t415 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t409,
                t101,
                t102,
                t103,
                t104,
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
        del t409, t104, t103, t102, t101

        # pd_op.relu: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        t416 = paddle._C_ops.relu(t410)
        del t410

        # pd_op.multiply: (-1x64x14x14xf32) <- (1xf32, -1x64x14x14xf32)
        t417 = paddle._C_ops.multiply(input33, t416)
        del input33, t416

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 1xf32)
        t418 = paddle._C_ops.add(t417, input34)
        del input34, t417

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x1x1xf32)
        t419 = paddle._C_ops.conv2d(
            t418, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t420, t421, t422, t423, t424, t425 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t419,
                t106,
                t107,
                t108,
                t109,
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
        del t419, t109, t108, t107, t106

        # pd_op.depthwise_conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x1x5x5xf32)
        t426 = paddle._C_ops.depthwise_conv2d(
            t420, t110, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t420, t110

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t427, t428, t429, t430, t431, t432 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t426,
                t111,
                t112,
                t113,
                t114,
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
        del t426, t112, t111, t114, t113

        # pd_op.relu: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        t433 = paddle._C_ops.relu(t427)
        del t427

        # pd_op.multiply: (-1x64x14x14xf32) <- (1xf32, -1x64x14x14xf32)
        t434 = paddle._C_ops.multiply(input35, t433)
        del input35, t433

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 1xf32)
        t435 = paddle._C_ops.add(t434, input36)
        del input36, t434

        # builtin.combine: ([-1x256x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32]) <- (-1x256x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32)
        t436 = [t379, t401, t418, t435]
        del t401, t418, t435, t379

        # pd_op.concat: (-1x448x14x14xf32) <- ([-1x256x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32], 1xi32)
        t437 = paddle._C_ops.concat(t436, t244)
        del t436

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x448x14x14xf32, 256x448x1x1xf32)
        t438 = paddle._C_ops.conv2d(
            t437, t115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t437, t115

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t439, t440, t441, t442, t443, t444 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t438,
                t116,
                t117,
                t118,
                t119,
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
        del t438, t119, t118, t117, t116

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t445 = paddle._C_ops.relu(t439)
        del t439

        # pd_op.multiply: (-1x256x14x14xf32) <- (1xf32, -1x256x14x14xf32)
        t446 = paddle._C_ops.multiply(input37, t445)
        del input37, t445

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 1xf32)
        t447 = paddle._C_ops.add(t446, input38)
        del input38, t446

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        t448 = paddle._C_ops.conv2d(
            t447, t120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t447, t120

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t449, t450, t451, t452, t453, t454 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t448,
                t121,
                t122,
                t123,
                t124,
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
        del t448, t124, t123, t122, t121

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t455 = paddle._C_ops.relu(t449)
        del t449

        # pd_op.multiply: (-1x512x14x14xf32) <- (1xf32, -1x512x14x14xf32)
        t456 = paddle._C_ops.multiply(input39, t455)
        del input39, t455

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1xf32)
        t457 = paddle._C_ops.add(t456, input40)
        del input40, t456

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x512x14x14xf32, 64x512x1x1xf32)
        t458 = paddle._C_ops.conv2d(
            t457, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t459, t460, t461, t462, t463, t464 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t458,
                t126,
                t127,
                t128,
                t129,
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
        del t458, t129, t128, t127, t126

        # pd_op.depthwise_conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x1x5x5xf32)
        t465 = paddle._C_ops.depthwise_conv2d(
            t459, t130, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t459, t130

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t466, t467, t468, t469, t470, t471 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t465,
                t131,
                t132,
                t133,
                t134,
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
        del t465, t134, t133, t132, t131

        # pd_op.relu: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        t472 = paddle._C_ops.relu(t466)
        del t466

        # pd_op.multiply: (-1x64x14x14xf32) <- (1xf32, -1x64x14x14xf32)
        t473 = paddle._C_ops.multiply(input41, t472)
        del input41, t472

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 1xf32)
        t474 = paddle._C_ops.add(t473, input42)
        del input42, t473

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x1x1xf32)
        t475 = paddle._C_ops.conv2d(
            t474, t135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t476, t477, t478, t479, t480, t481 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t475,
                t136,
                t137,
                t138,
                t139,
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
        del t475, t139, t138, t137, t136

        # pd_op.depthwise_conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x1x5x5xf32)
        t482 = paddle._C_ops.depthwise_conv2d(
            t476, t140, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t476, t140

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t483, t484, t485, t486, t487, t488 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t482,
                t141,
                t142,
                t143,
                t144,
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
        del t482, t144, t143, t142, t141

        # pd_op.relu: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        t489 = paddle._C_ops.relu(t483)
        del t483

        # pd_op.multiply: (-1x64x14x14xf32) <- (1xf32, -1x64x14x14xf32)
        t490 = paddle._C_ops.multiply(input43, t489)
        del input43, t489

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 1xf32)
        t491 = paddle._C_ops.add(t490, input44)
        del input44, t490

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x64x1x1xf32)
        t492 = paddle._C_ops.conv2d(
            t491, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t493, t494, t495, t496, t497, t498 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t492,
                t146,
                t147,
                t148,
                t149,
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
        del t492, t149, t148, t147, t146

        # pd_op.depthwise_conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x1x5x5xf32)
        t499 = paddle._C_ops.depthwise_conv2d(
            t493, t150, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del t493, t150

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t500, t501, t502, t503, t504, t505 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t499,
                t151,
                t152,
                t153,
                t154,
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
        del t499, t154, t153, t152, t151

        # pd_op.relu: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        t506 = paddle._C_ops.relu(t500)
        del t500

        # pd_op.multiply: (-1x64x14x14xf32) <- (1xf32, -1x64x14x14xf32)
        t507 = paddle._C_ops.multiply(input45, t506)
        del input45, t506

        # pd_op.add: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 1xf32)
        t508 = paddle._C_ops.add(t507, input46)
        del input46, t507

        # builtin.combine: ([-1x512x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32]) <- (-1x512x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32)
        t509 = [t457, t474, t491, t508]
        del t474, t491, t508

        # pd_op.concat: (-1x704x14x14xf32) <- ([-1x512x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32, -1x64x14x14xf32], 1xi32)
        t510 = paddle._C_ops.concat(t509, t244)
        del t509

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x704x14x14xf32, 256x704x1x1xf32)
        t511 = paddle._C_ops.conv2d(
            t510, t155, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t510, t155

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t512, t513, t514, t515, t516, t517 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t511,
                t156,
                t157,
                t158,
                t159,
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
        del t511, t159, t158, t157, t156

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t518 = paddle._C_ops.relu(t512)
        del t512

        # pd_op.multiply: (-1x256x14x14xf32) <- (1xf32, -1x256x14x14xf32)
        t519 = paddle._C_ops.multiply(input47, t518)
        del input47, t518

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 1xf32)
        t520 = paddle._C_ops.add(t519, input48)
        del input48, t519

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        t521 = paddle._C_ops.conv2d(
            t520, t160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t520, t160

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t522, t523, t524, t525, t526, t527 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t521,
                t161,
                t162,
                t163,
                t164,
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
        del t521, t164, t163, t162, t161

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t528 = paddle._C_ops.relu(t522)
        del t522

        # pd_op.multiply: (-1x512x14x14xf32) <- (1xf32, -1x512x14x14xf32)
        t529 = paddle._C_ops.multiply(input49, t528)
        del input49, t528

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1xf32)
        t530 = paddle._C_ops.add(t529, input50)
        del input50, t529

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        t531 = paddle._C_ops.add(t530, t457)
        del t457, t530

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x14x14xf32, 512x1x3x3xf32)
        t532 = paddle._C_ops.depthwise_conv2d(
            t531, t165, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t531, t165

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t533, t534, t535, t536, t537, t538 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t532,
                t166,
                t167,
                t168,
                t169,
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
        del t532, t169, t168, t167, t166

        # pd_op.conv2d: (-1x128x7x7xf32) <- (-1x512x7x7xf32, 128x512x1x1xf32)
        t539 = paddle._C_ops.conv2d(
            t533, t170, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t540, t541, t542, t543, t544, t545 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t539,
                t171,
                t172,
                t173,
                t174,
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
        del t539, t174, t173, t172, t171

        # pd_op.depthwise_conv2d: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 128x1x5x5xf32)
        t546 = paddle._C_ops.depthwise_conv2d(
            t540, t175, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t540, t175

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t547, t548, t549, t550, t551, t552 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t546,
                t176,
                t177,
                t178,
                t179,
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
        del t546, t179, t178, t177, t176

        # pd_op.relu: (-1x128x7x7xf32) <- (-1x128x7x7xf32)
        t553 = paddle._C_ops.relu(t547)
        del t547

        # pd_op.multiply: (-1x128x7x7xf32) <- (1xf32, -1x128x7x7xf32)
        t554 = paddle._C_ops.multiply(input51, t553)
        del input51, t553

        # pd_op.add: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 1xf32)
        t555 = paddle._C_ops.add(t554, input52)
        del input52, t554

        # pd_op.conv2d: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 128x128x1x1xf32)
        t556 = paddle._C_ops.conv2d(
            t555, t180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t557, t558, t559, t560, t561, t562 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t556,
                t181,
                t182,
                t183,
                t184,
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
        del t556, t184, t183, t182, t181

        # pd_op.depthwise_conv2d: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 128x1x5x5xf32)
        t563 = paddle._C_ops.depthwise_conv2d(
            t557, t185, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t557, t185

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t564, t565, t566, t567, t568, t569 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t563,
                t186,
                t187,
                t188,
                t189,
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
        del t563, t189, t188, t187, t186

        # pd_op.relu: (-1x128x7x7xf32) <- (-1x128x7x7xf32)
        t570 = paddle._C_ops.relu(t564)
        del t564

        # pd_op.multiply: (-1x128x7x7xf32) <- (1xf32, -1x128x7x7xf32)
        t571 = paddle._C_ops.multiply(input53, t570)
        del input53, t570

        # pd_op.add: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 1xf32)
        t572 = paddle._C_ops.add(t571, input54)
        del input54, t571

        # pd_op.conv2d: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 128x128x1x1xf32)
        t573 = paddle._C_ops.conv2d(
            t572, t190, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t574, t575, t576, t577, t578, t579 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t573,
                t191,
                t192,
                t193,
                t194,
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
        del t573, t194, t193, t192, t191

        # pd_op.depthwise_conv2d: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 128x1x5x5xf32)
        t580 = paddle._C_ops.depthwise_conv2d(
            t574, t195, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t574, t195

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t581, t582, t583, t584, t585, t586 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t580,
                t196,
                t197,
                t198,
                t199,
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
        del t580, t199, t198, t197, t196

        # pd_op.relu: (-1x128x7x7xf32) <- (-1x128x7x7xf32)
        t587 = paddle._C_ops.relu(t581)
        del t581

        # pd_op.multiply: (-1x128x7x7xf32) <- (1xf32, -1x128x7x7xf32)
        t588 = paddle._C_ops.multiply(input55, t587)
        del input55, t587

        # pd_op.add: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 1xf32)
        t589 = paddle._C_ops.add(t588, input56)
        del input56, t588

        # builtin.combine: ([-1x512x7x7xf32, -1x128x7x7xf32, -1x128x7x7xf32, -1x128x7x7xf32]) <- (-1x512x7x7xf32, -1x128x7x7xf32, -1x128x7x7xf32, -1x128x7x7xf32)
        t590 = [t533, t555, t572, t589]
        del t555, t572, t589, t533

        # pd_op.concat: (-1x896x7x7xf32) <- ([-1x512x7x7xf32, -1x128x7x7xf32, -1x128x7x7xf32, -1x128x7x7xf32], 1xi32)
        t591 = paddle._C_ops.concat(t590, t244)
        del t590, t244

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x896x7x7xf32, 512x896x1x1xf32)
        t592 = paddle._C_ops.conv2d(
            t591, t200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t591, t200

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t593, t594, t595, t596, t597, t598 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t592,
                t201,
                t202,
                t203,
                t204,
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
        del t592, t202, t201, t204, t203

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t599 = paddle._C_ops.relu(t593)
        del t593

        # pd_op.multiply: (-1x512x7x7xf32) <- (1xf32, -1x512x7x7xf32)
        t600 = paddle._C_ops.multiply(input57, t599)
        del input57, t599

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 1xf32)
        t601 = paddle._C_ops.add(t600, input58)
        del input58, t600

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x512x7x7xf32, 1024x512x1x1xf32)
        t602 = paddle._C_ops.conv2d(
            t601, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t601, t205

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t603, t604, t605, t606, t607, t608 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t602,
                t206,
                t207,
                t208,
                t209,
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
        del t602, t209, t208, t207, t206

        # pd_op.relu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t609 = paddle._C_ops.relu(t603)
        del t603

        # pd_op.multiply: (-1x1024x7x7xf32) <- (1xf32, -1x1024x7x7xf32)
        t610 = paddle._C_ops.multiply(input59, t609)
        del input59, t609

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1xf32)
        t611 = paddle._C_ops.add(t610, input60)
        del input60, t610

        # pd_op.full_int_array: (2xi64) <- ()
        t612 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x7x7xf32, 2xi64)
        t613 = paddle._C_ops.pool2d(
            t611,
            t612,
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
        del t611, t612

        # pd_op.conv2d: (-1x2048x1x1xf32) <- (-1x1024x1x1xf32, 2048x1024x1x1xf32)
        t614 = paddle._C_ops.conv2d(
            t613, t210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210, t613

        # pd_op.relu: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32)
        t615 = paddle._C_ops.relu(t614)
        del t614

        # pd_op.multiply: (-1x2048x1x1xf32) <- (1xf32, -1x2048x1x1xf32)
        t616 = paddle._C_ops.multiply(input61, t615)
        del input61, t615

        # pd_op.add: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32, 1xf32)
        t617 = paddle._C_ops.add(t616, input62)
        del input62, t616

        # pd_op.flatten: (-1x2048xf32) <- (-1x2048x1x1xf32)
        t618 = paddle._C_ops.flatten(t617, 1, 3)
        del t617

        # pd_op.matmul: (-1x102xf32) <- (-1x2048xf32, 2048x102xf32)
        t619 = paddle._C_ops.matmul(t618, t211, False, False)
        del t618, t211

        return t619
