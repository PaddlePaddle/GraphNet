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
    ):
        # pd_op.conv2d: (128x128x56x56xf32) <- (128x3x224x224xf32, 128x3x4x4xf32)
        t148 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t149, t150, t151, t152, t153, t154 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t148,
                t1,
                t2,
                t3,
                t4,
                False,
                float("0.1"),
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

        # pd_op.full_int_array: (2xi64) <- ()
        t155 = [32, 96]

        # pd_op.full: (1xi32) <- ()
        t156 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t157 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t158 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t159 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t160 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t161 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t162 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t163 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t164 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t165 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t166 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t167 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t168 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t169 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t170 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t171 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t172 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t173 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t174 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t175 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t176 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t177 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t178 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t179 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t180 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t181 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t182 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t183 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t184 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t185 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t186 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t187 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t188 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t189 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t190 = t156

        # pd_op.assign: (1xi32) <- (1xi32)
        t191 = t156

        # pd_op.split: ([128x32x56x56xf32, 128x96x56x56xf32]) <- (128x128x56x56xf32, 2xi64, 1xi32)
        t192 = paddle._C_ops.split(t149, t155, t156)
        del t155

        # builtin.split: (128x32x56x56xf32, 128x96x56x56xf32) <- ([128x32x56x56xf32, 128x96x56x56xf32])
        (
            t193,
            t194,
        ) = t192
        del t192

        # pd_op.conv2d: (128x32x56x56xf32) <- (128x32x56x56xf32, 32x32x3x3xf32)
        t195 = paddle._C_ops.conv2d(
            t193, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # builtin.combine: ([128x32x56x56xf32, 128x96x56x56xf32]) <- (128x32x56x56xf32, 128x96x56x56xf32)
        t196 = [t195, t194]

        # pd_op.concat: (128x128x56x56xf32) <- ([128x32x56x56xf32, 128x96x56x56xf32], 1xi32)
        t197 = paddle._C_ops.concat(t196, t156)
        del t196

        # pd_op.conv2d: (128x256x56x56xf32) <- (128x128x56x56xf32, 256x128x1x1xf32)
        t198 = paddle._C_ops.conv2d(
            t197, t6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t6

        # pd_op.batch_norm_: (128x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t199, t200, t201, t202, t203, t204 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t198,
                t7,
                t8,
                t9,
                t10,
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
        del t10, t9, t8, t7

        # pd_op.relu: (128x256x56x56xf32) <- (128x256x56x56xf32)
        t205 = paddle._C_ops.relu(t199)
        del t199

        # pd_op.conv2d: (128x128x56x56xf32) <- (128x256x56x56xf32, 128x256x1x1xf32)
        t206 = paddle._C_ops.conv2d(
            t205, t11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t11

        # pd_op.add: (128x128x56x56xf32) <- (128x128x56x56xf32, 128x128x56x56xf32)
        t207 = paddle._C_ops.add(t149, t206)

        # pd_op.conv2d: (128x256x28x28xf32) <- (128x128x56x56xf32, 256x128x2x2xf32)
        t208 = paddle._C_ops.conv2d(
            t207, t12, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t12

        # pd_op.batch_norm_: (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t209, t210, t211, t212, t213, t214 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t208,
                t13,
                t14,
                t15,
                t16,
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
        del t16, t15, t14, t13

        # pd_op.full_int_array: (2xi64) <- ()
        t215 = [64, 192]

        # pd_op.split: ([128x64x28x28xf32, 128x192x28x28xf32]) <- (128x256x28x28xf32, 2xi64, 1xi32)
        t216 = paddle._C_ops.split(t209, t215, t156)

        # builtin.split: (128x64x28x28xf32, 128x192x28x28xf32) <- ([128x64x28x28xf32, 128x192x28x28xf32])
        (
            t217,
            t218,
        ) = t216
        del t216

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x64x28x28xf32, 64x64x3x3xf32)
        t219 = paddle._C_ops.conv2d(
            t217, t17, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t17

        # builtin.combine: ([128x64x28x28xf32, 128x192x28x28xf32]) <- (128x64x28x28xf32, 128x192x28x28xf32)
        t220 = [t219, t218]

        # pd_op.concat: (128x256x28x28xf32) <- ([128x64x28x28xf32, 128x192x28x28xf32], 1xi32)
        t221 = paddle._C_ops.concat(t220, t156)
        del t220

        # pd_op.conv2d: (128x512x28x28xf32) <- (128x256x28x28xf32, 512x256x1x1xf32)
        t222 = paddle._C_ops.conv2d(
            t221, t18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t18

        # pd_op.batch_norm_: (128x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t223, t224, t225, t226, t227, t228 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t222,
                t19,
                t20,
                t21,
                t22,
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
        del t22, t21, t20, t19

        # pd_op.relu: (128x512x28x28xf32) <- (128x512x28x28xf32)
        t229 = paddle._C_ops.relu(t223)
        del t223

        # pd_op.conv2d: (128x256x28x28xf32) <- (128x512x28x28xf32, 256x512x1x1xf32)
        t230 = paddle._C_ops.conv2d(
            t229, t23, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t23

        # pd_op.full: (xf32) <- ()
        t231 = paddle._C_ops.full(
            [],
            float("0.994118"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (4xi64) <- ()
        t232 = [128, 1, 1, 1]

        # pd_op.full: (1xf32) <- ()
        t233 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t234 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t235 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t236 = paddle._C_ops.add(t231, t235)
        del t235

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t237 = paddle._C_ops.floor(t236)
        del t236

        # pd_op.divide: (128x256x28x28xf32) <- (128x256x28x28xf32, xf32)
        t238 = paddle._C_ops.divide(t230, t231)

        # pd_op.multiply: (128x256x28x28xf32) <- (128x256x28x28xf32, 128x1x1x1xf32)
        t239 = paddle._C_ops.multiply(t238, t237)

        # pd_op.add: (128x256x28x28xf32) <- (128x256x28x28xf32, 128x256x28x28xf32)
        t240 = paddle._C_ops.add(t209, t239)

        # pd_op.split: ([128x64x28x28xf32, 128x192x28x28xf32]) <- (128x256x28x28xf32, 2xi64, 1xi32)
        t241 = paddle._C_ops.split(t240, t215, t156)
        del t215

        # builtin.split: (128x64x28x28xf32, 128x192x28x28xf32) <- ([128x64x28x28xf32, 128x192x28x28xf32])
        (
            t242,
            t243,
        ) = t241
        del t241

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x64x28x28xf32, 64x64x3x3xf32)
        t244 = paddle._C_ops.conv2d(
            t242, t24, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t24

        # builtin.combine: ([128x64x28x28xf32, 128x192x28x28xf32]) <- (128x64x28x28xf32, 128x192x28x28xf32)
        t245 = [t244, t243]

        # pd_op.concat: (128x256x28x28xf32) <- ([128x64x28x28xf32, 128x192x28x28xf32], 1xi32)
        t246 = paddle._C_ops.concat(t245, t156)
        del t245

        # pd_op.conv2d: (128x512x28x28xf32) <- (128x256x28x28xf32, 512x256x1x1xf32)
        t247 = paddle._C_ops.conv2d(
            t246, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (128x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t248, t249, t250, t251, t252, t253 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t247,
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

        # pd_op.relu: (128x512x28x28xf32) <- (128x512x28x28xf32)
        t254 = paddle._C_ops.relu(t248)
        del t248

        # pd_op.conv2d: (128x256x28x28xf32) <- (128x512x28x28xf32, 256x512x1x1xf32)
        t255 = paddle._C_ops.conv2d(
            t254, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.full: (xf32) <- ()
        t256 = paddle._C_ops.full(
            [],
            float("0.988235"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t257 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t258 = paddle._C_ops.add(t256, t257)
        del t257

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t259 = paddle._C_ops.floor(t258)
        del t258

        # pd_op.divide: (128x256x28x28xf32) <- (128x256x28x28xf32, xf32)
        t260 = paddle._C_ops.divide(t255, t256)

        # pd_op.multiply: (128x256x28x28xf32) <- (128x256x28x28xf32, 128x1x1x1xf32)
        t261 = paddle._C_ops.multiply(t260, t259)

        # pd_op.add: (128x256x28x28xf32) <- (128x256x28x28xf32, 128x256x28x28xf32)
        t262 = paddle._C_ops.add(t240, t261)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x256x28x28xf32, 512x256x2x2xf32)
        t263 = paddle._C_ops.conv2d(
            t262, t31, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t31

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t264, t265, t266, t267, t268, t269 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t263,
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

        # pd_op.full_int_array: (2xi64) <- ()
        t270 = [128, 384]

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t271 = paddle._C_ops.split(t264, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t272,
            t273,
        ) = t271
        del t271

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t274 = paddle._C_ops.conv2d(
            t272, t36, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t36

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t275 = [t274, t273]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t276 = paddle._C_ops.concat(t275, t156)
        del t275

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t277 = paddle._C_ops.conv2d(
            t276, t37, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t37

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t278, t279, t280, t281, t282, t283 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t277,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t284 = paddle._C_ops.relu(t278)
        del t278

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t285 = paddle._C_ops.conv2d(
            t284, t42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t42

        # pd_op.full: (xf32) <- ()
        t286 = paddle._C_ops.full(
            [],
            float("0.982353"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t287 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t288 = paddle._C_ops.add(t286, t287)
        del t287

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t289 = paddle._C_ops.floor(t288)
        del t288

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t290 = paddle._C_ops.divide(t285, t286)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t291 = paddle._C_ops.multiply(t290, t289)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t292 = paddle._C_ops.add(t264, t291)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t293 = paddle._C_ops.split(t292, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t294,
            t295,
        ) = t293
        del t293

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t296 = paddle._C_ops.conv2d(
            t294, t43, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t43

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t297 = [t296, t295]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t298 = paddle._C_ops.concat(t297, t156)
        del t297

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t299 = paddle._C_ops.conv2d(
            t298, t44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t44

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t300, t301, t302, t303, t304, t305 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t299,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t306 = paddle._C_ops.relu(t300)
        del t300

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t307 = paddle._C_ops.conv2d(
            t306, t49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t49

        # pd_op.full: (xf32) <- ()
        t308 = paddle._C_ops.full(
            [],
            float("0.976471"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t309 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t310 = paddle._C_ops.add(t308, t309)
        del t309

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t311 = paddle._C_ops.floor(t310)
        del t310

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t312 = paddle._C_ops.divide(t307, t308)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t313 = paddle._C_ops.multiply(t312, t311)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t314 = paddle._C_ops.add(t292, t313)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t315 = paddle._C_ops.split(t314, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t316,
            t317,
        ) = t315
        del t315

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t318 = paddle._C_ops.conv2d(
            t316, t50, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t319 = [t318, t317]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t320 = paddle._C_ops.concat(t319, t156)
        del t319

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t321 = paddle._C_ops.conv2d(
            t320, t51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t51

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t322, t323, t324, t325, t326, t327 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t321,
                t52,
                t53,
                t54,
                t55,
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
        del t55, t54, t53, t52

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t328 = paddle._C_ops.relu(t322)
        del t322

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t329 = paddle._C_ops.conv2d(
            t328, t56, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t56

        # pd_op.full: (xf32) <- ()
        t330 = paddle._C_ops.full(
            [],
            float("0.970588"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t331 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t332 = paddle._C_ops.add(t330, t331)
        del t331

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t333 = paddle._C_ops.floor(t332)
        del t332

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t334 = paddle._C_ops.divide(t329, t330)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t335 = paddle._C_ops.multiply(t334, t333)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t336 = paddle._C_ops.add(t314, t335)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t337 = paddle._C_ops.split(t336, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t338,
            t339,
        ) = t337
        del t337

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t340 = paddle._C_ops.conv2d(
            t338, t57, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t57

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t341 = [t340, t339]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t342 = paddle._C_ops.concat(t341, t156)
        del t341

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t343 = paddle._C_ops.conv2d(
            t342, t58, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t58

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t344, t345, t346, t347, t348, t349 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t343,
                t59,
                t60,
                t61,
                t62,
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
        del t62, t61, t60, t59

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t350 = paddle._C_ops.relu(t344)
        del t344

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t351 = paddle._C_ops.conv2d(
            t350, t63, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t63

        # pd_op.full: (xf32) <- ()
        t352 = paddle._C_ops.full(
            [],
            float("0.964706"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t353 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t354 = paddle._C_ops.add(t352, t353)
        del t353

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t355 = paddle._C_ops.floor(t354)
        del t354

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t356 = paddle._C_ops.divide(t351, t352)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t357 = paddle._C_ops.multiply(t356, t355)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t358 = paddle._C_ops.add(t336, t357)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t359 = paddle._C_ops.split(t358, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t360,
            t361,
        ) = t359
        del t359

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t362 = paddle._C_ops.conv2d(
            t360, t64, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t64

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t363 = [t362, t361]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t364 = paddle._C_ops.concat(t363, t156)
        del t363

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t365 = paddle._C_ops.conv2d(
            t364, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t366, t367, t368, t369, t370, t371 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t365,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t372 = paddle._C_ops.relu(t366)
        del t366

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t373 = paddle._C_ops.conv2d(
            t372, t70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.full: (xf32) <- ()
        t374 = paddle._C_ops.full(
            [],
            float("0.958824"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t375 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t376 = paddle._C_ops.add(t374, t375)
        del t375

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t377 = paddle._C_ops.floor(t376)
        del t376

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t378 = paddle._C_ops.divide(t373, t374)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t379 = paddle._C_ops.multiply(t378, t377)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t380 = paddle._C_ops.add(t358, t379)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t381 = paddle._C_ops.split(t380, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t382,
            t383,
        ) = t381
        del t381

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t384 = paddle._C_ops.conv2d(
            t382, t71, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t71

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t385 = [t384, t383]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t386 = paddle._C_ops.concat(t385, t156)
        del t385

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t387 = paddle._C_ops.conv2d(
            t386, t72, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t72

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t388, t389, t390, t391, t392, t393 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t387,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t394 = paddle._C_ops.relu(t388)
        del t388

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t395 = paddle._C_ops.conv2d(
            t394, t77, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t77

        # pd_op.full: (xf32) <- ()
        t396 = paddle._C_ops.full(
            [],
            float("0.952941"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t397 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t398 = paddle._C_ops.add(t396, t397)
        del t397

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t399 = paddle._C_ops.floor(t398)
        del t398

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t400 = paddle._C_ops.divide(t395, t396)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t401 = paddle._C_ops.multiply(t400, t399)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t402 = paddle._C_ops.add(t380, t401)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t403 = paddle._C_ops.split(t402, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t404,
            t405,
        ) = t403
        del t403

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t406 = paddle._C_ops.conv2d(
            t404, t78, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t78

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t407 = [t406, t405]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t408 = paddle._C_ops.concat(t407, t156)
        del t407

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t409 = paddle._C_ops.conv2d(
            t408, t79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t79

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t410, t411, t412, t413, t414, t415 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t409,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t416 = paddle._C_ops.relu(t410)
        del t410

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t417 = paddle._C_ops.conv2d(
            t416, t84, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t84

        # pd_op.full: (xf32) <- ()
        t418 = paddle._C_ops.full(
            [],
            float("0.947059"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t419 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t420 = paddle._C_ops.add(t418, t419)
        del t419

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t421 = paddle._C_ops.floor(t420)
        del t420

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t422 = paddle._C_ops.divide(t417, t418)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t423 = paddle._C_ops.multiply(t422, t421)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t424 = paddle._C_ops.add(t402, t423)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t425 = paddle._C_ops.split(t424, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t426,
            t427,
        ) = t425
        del t425

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t428 = paddle._C_ops.conv2d(
            t426, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t429 = [t428, t427]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t430 = paddle._C_ops.concat(t429, t156)
        del t429

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t431 = paddle._C_ops.conv2d(
            t430, t86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t86

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t432, t433, t434, t435, t436, t437 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t431,
                t87,
                t88,
                t89,
                t90,
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
        del t90, t89, t88, t87

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t438 = paddle._C_ops.relu(t432)
        del t432

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t439 = paddle._C_ops.conv2d(
            t438, t91, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t91

        # pd_op.full: (xf32) <- ()
        t440 = paddle._C_ops.full(
            [],
            float("0.941176"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t441 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t442 = paddle._C_ops.add(t440, t441)
        del t441

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t443 = paddle._C_ops.floor(t442)
        del t442

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t444 = paddle._C_ops.divide(t439, t440)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t445 = paddle._C_ops.multiply(t444, t443)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t446 = paddle._C_ops.add(t424, t445)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t447 = paddle._C_ops.split(t446, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t448,
            t449,
        ) = t447
        del t447

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t450 = paddle._C_ops.conv2d(
            t448, t92, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t92

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t451 = [t450, t449]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t452 = paddle._C_ops.concat(t451, t156)
        del t451

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t453 = paddle._C_ops.conv2d(
            t452, t93, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t93

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t454, t455, t456, t457, t458, t459 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t453,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t460 = paddle._C_ops.relu(t454)
        del t454

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t461 = paddle._C_ops.conv2d(
            t460, t98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t98

        # pd_op.full: (xf32) <- ()
        t462 = paddle._C_ops.full(
            [],
            float("0.935294"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t463 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t464 = paddle._C_ops.add(t462, t463)
        del t463

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t465 = paddle._C_ops.floor(t464)
        del t464

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t466 = paddle._C_ops.divide(t461, t462)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t467 = paddle._C_ops.multiply(t466, t465)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t468 = paddle._C_ops.add(t446, t467)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t469 = paddle._C_ops.split(t468, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t470,
            t471,
        ) = t469
        del t469

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t472 = paddle._C_ops.conv2d(
            t470, t99, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t99

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t473 = [t472, t471]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t474 = paddle._C_ops.concat(t473, t156)
        del t473

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t475 = paddle._C_ops.conv2d(
            t474, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t476, t477, t478, t479, t480, t481 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t475,
                t101,
                t102,
                t103,
                t104,
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
        del t104, t103, t102, t101

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t482 = paddle._C_ops.relu(t476)
        del t476

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t483 = paddle._C_ops.conv2d(
            t482, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.full: (xf32) <- ()
        t484 = paddle._C_ops.full(
            [],
            float("0.929412"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t485 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t486 = paddle._C_ops.add(t484, t485)
        del t485

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t487 = paddle._C_ops.floor(t486)
        del t486

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t488 = paddle._C_ops.divide(t483, t484)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t489 = paddle._C_ops.multiply(t488, t487)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t490 = paddle._C_ops.add(t468, t489)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t491 = paddle._C_ops.split(t490, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t492,
            t493,
        ) = t491
        del t491

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t494 = paddle._C_ops.conv2d(
            t492, t106, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t106

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t495 = [t494, t493]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t496 = paddle._C_ops.concat(t495, t156)
        del t495

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t497 = paddle._C_ops.conv2d(
            t496, t107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t107

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t498, t499, t500, t501, t502, t503 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t497,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t504 = paddle._C_ops.relu(t498)
        del t498

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t505 = paddle._C_ops.conv2d(
            t504, t112, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t112

        # pd_op.full: (xf32) <- ()
        t506 = paddle._C_ops.full(
            [],
            float("0.923529"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t507 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t508 = paddle._C_ops.add(t506, t507)
        del t507

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t509 = paddle._C_ops.floor(t508)
        del t508

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t510 = paddle._C_ops.divide(t505, t506)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t511 = paddle._C_ops.multiply(t510, t509)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t512 = paddle._C_ops.add(t490, t511)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t513 = paddle._C_ops.split(t512, t270, t156)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t514,
            t515,
        ) = t513
        del t513

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t516 = paddle._C_ops.conv2d(
            t514, t113, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t113

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t517 = [t516, t515]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t518 = paddle._C_ops.concat(t517, t156)
        del t517

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t519 = paddle._C_ops.conv2d(
            t518, t114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t114

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t520, t521, t522, t523, t524, t525 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t519,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t526 = paddle._C_ops.relu(t520)
        del t520

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t527 = paddle._C_ops.conv2d(
            t526, t119, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t119

        # pd_op.full: (xf32) <- ()
        t528 = paddle._C_ops.full(
            [],
            float("0.917647"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t529 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t530 = paddle._C_ops.add(t528, t529)
        del t529

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t531 = paddle._C_ops.floor(t530)
        del t530

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t532 = paddle._C_ops.divide(t527, t528)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t533 = paddle._C_ops.multiply(t532, t531)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t534 = paddle._C_ops.add(t512, t533)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        t535 = paddle._C_ops.split(t534, t270, t156)
        del t270

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            t536,
            t537,
        ) = t535
        del t535

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        t538 = paddle._C_ops.conv2d(
            t536, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        t539 = [t538, t537]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        t540 = paddle._C_ops.concat(t539, t156)
        del t539

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        t541 = paddle._C_ops.conv2d(
            t540, t121, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t121

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t542, t543, t544, t545, t546, t547 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t541,
                t122,
                t123,
                t124,
                t125,
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
        del t125, t124, t123, t122

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        t548 = paddle._C_ops.relu(t542)
        del t542

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        t549 = paddle._C_ops.conv2d(
            t548, t126, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t126

        # pd_op.full: (xf32) <- ()
        t550 = paddle._C_ops.full(
            [],
            float("0.911765"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t551 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t552 = paddle._C_ops.add(t550, t551)
        del t551

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t553 = paddle._C_ops.floor(t552)
        del t552

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        t554 = paddle._C_ops.divide(t549, t550)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        t555 = paddle._C_ops.multiply(t554, t553)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        t556 = paddle._C_ops.add(t534, t555)

        # pd_op.conv2d: (128x1024x7x7xf32) <- (128x512x14x14xf32, 1024x512x2x2xf32)
        t557 = paddle._C_ops.conv2d(
            t556, t127, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t127

        # pd_op.batch_norm_: (128x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t558, t559, t560, t561, t562, t563 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t557,
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

        # pd_op.full_int_array: (2xi64) <- ()
        t564 = [256, 768]

        # pd_op.split: ([128x256x7x7xf32, 128x768x7x7xf32]) <- (128x1024x7x7xf32, 2xi64, 1xi32)
        t565 = paddle._C_ops.split(t558, t564, t156)

        # builtin.split: (128x256x7x7xf32, 128x768x7x7xf32) <- ([128x256x7x7xf32, 128x768x7x7xf32])
        (
            t566,
            t567,
        ) = t565
        del t565

        # pd_op.conv2d: (128x256x7x7xf32) <- (128x256x7x7xf32, 256x256x3x3xf32)
        t568 = paddle._C_ops.conv2d(
            t566, t132, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t132

        # builtin.combine: ([128x256x7x7xf32, 128x768x7x7xf32]) <- (128x256x7x7xf32, 128x768x7x7xf32)
        t569 = [t568, t567]

        # pd_op.concat: (128x1024x7x7xf32) <- ([128x256x7x7xf32, 128x768x7x7xf32], 1xi32)
        t570 = paddle._C_ops.concat(t569, t156)
        del t569

        # pd_op.conv2d: (128x2048x7x7xf32) <- (128x1024x7x7xf32, 2048x1024x1x1xf32)
        t571 = paddle._C_ops.conv2d(
            t570, t133, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t133

        # pd_op.batch_norm_: (128x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (128x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t572, t573, t574, t575, t576, t577 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t571,
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

        # pd_op.relu: (128x2048x7x7xf32) <- (128x2048x7x7xf32)
        t578 = paddle._C_ops.relu(t572)
        del t572

        # pd_op.conv2d: (128x1024x7x7xf32) <- (128x2048x7x7xf32, 1024x2048x1x1xf32)
        t579 = paddle._C_ops.conv2d(
            t578, t138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t138

        # pd_op.full: (xf32) <- ()
        t580 = paddle._C_ops.full(
            [],
            float("0.905882"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t581 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t582 = paddle._C_ops.add(t580, t581)
        del t581

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t583 = paddle._C_ops.floor(t582)
        del t582

        # pd_op.divide: (128x1024x7x7xf32) <- (128x1024x7x7xf32, xf32)
        t584 = paddle._C_ops.divide(t579, t580)

        # pd_op.multiply: (128x1024x7x7xf32) <- (128x1024x7x7xf32, 128x1x1x1xf32)
        t585 = paddle._C_ops.multiply(t584, t583)

        # pd_op.add: (128x1024x7x7xf32) <- (128x1024x7x7xf32, 128x1024x7x7xf32)
        t586 = paddle._C_ops.add(t558, t585)

        # pd_op.split: ([128x256x7x7xf32, 128x768x7x7xf32]) <- (128x1024x7x7xf32, 2xi64, 1xi32)
        t587 = paddle._C_ops.split(t586, t564, t156)
        del t564

        # builtin.split: (128x256x7x7xf32, 128x768x7x7xf32) <- ([128x256x7x7xf32, 128x768x7x7xf32])
        (
            t588,
            t589,
        ) = t587
        del t587

        # pd_op.conv2d: (128x256x7x7xf32) <- (128x256x7x7xf32, 256x256x3x3xf32)
        t590 = paddle._C_ops.conv2d(
            t588, t139, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t139

        # builtin.combine: ([128x256x7x7xf32, 128x768x7x7xf32]) <- (128x256x7x7xf32, 128x768x7x7xf32)
        t591 = [t590, t589]

        # pd_op.concat: (128x1024x7x7xf32) <- ([128x256x7x7xf32, 128x768x7x7xf32], 1xi32)
        t592 = paddle._C_ops.concat(t591, t156)
        del t591

        # pd_op.conv2d: (128x2048x7x7xf32) <- (128x1024x7x7xf32, 2048x1024x1x1xf32)
        t593 = paddle._C_ops.conv2d(
            t592, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (128x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (128x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t594, t595, t596, t597, t598, t599 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t593,
                t141,
                t142,
                t143,
                t144,
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
        del t144, t143, t142, t141

        # pd_op.relu: (128x2048x7x7xf32) <- (128x2048x7x7xf32)
        t600 = paddle._C_ops.relu(t594)
        del t594

        # pd_op.conv2d: (128x1024x7x7xf32) <- (128x2048x7x7xf32, 1024x2048x1x1xf32)
        t601 = paddle._C_ops.conv2d(
            t600, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145

        # pd_op.full: (xf32) <- ()
        t602 = paddle._C_ops.full(
            [], float("0.9"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t603 = paddle._C_ops.uniform(
            t232,
            paddle.float32,
            t233,
            t234,
            0,
            paddle.framework._current_expected_place(),
        )
        del t233, t234, t232

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        t604 = paddle._C_ops.add(t602, t603)
        del t603

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        t605 = paddle._C_ops.floor(t604)
        del t604

        # pd_op.divide: (128x1024x7x7xf32) <- (128x1024x7x7xf32, xf32)
        t606 = paddle._C_ops.divide(t601, t602)

        # pd_op.multiply: (128x1024x7x7xf32) <- (128x1024x7x7xf32, 128x1x1x1xf32)
        t607 = paddle._C_ops.multiply(t606, t605)

        # pd_op.add: (128x1024x7x7xf32) <- (128x1024x7x7xf32, 128x1024x7x7xf32)
        t608 = paddle._C_ops.add(t586, t607)

        # pd_op.full_int_array: (2xi64) <- ()
        t609 = [1, 1]

        # pd_op.pool2d: (128x1024x1x1xf32) <- (128x1024x7x7xf32, 2xi64)
        t610 = paddle._C_ops.pool2d(
            t608,
            t609,
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

        # pd_op.conv2d: (128x1280x1x1xf32) <- (128x1024x1x1xf32, 1280x1024x1x1xf32)
        t611 = paddle._C_ops.conv2d(
            t610, t146, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t146

        # pd_op.relu: (128x1280x1x1xf32) <- (128x1280x1x1xf32)
        t612 = paddle._C_ops.relu(t611)
        del t611

        # pd_op.flatten: (128x1280xf32) <- (128x1280x1x1xf32)
        t613 = paddle._C_ops.flatten(t612, 1, 3)

        # pd_op.matmul: (128x102xf32) <- (128x1280xf32, 1280x102xf32)
        t614 = paddle._C_ops.matmul(t613, t147, False, False)
        del t147

        return (
            t148,
            t149,
            t150,
            t151,
            t152,
            t153,
            t154,
            t156,
            t157,
            t158,
            t159,
            t160,
            t161,
            t162,
            t163,
            t164,
            t165,
            t166,
            t167,
            t168,
            t169,
            t170,
            t171,
            t172,
            t173,
            t174,
            t175,
            t176,
            t177,
            t178,
            t179,
            t180,
            t181,
            t182,
            t183,
            t184,
            t185,
            t186,
            t187,
            t188,
            t189,
            t190,
            t191,
            t193,
            t194,
            t195,
            t197,
            t198,
            t200,
            t201,
            t202,
            t203,
            t204,
            t205,
            t206,
            t207,
            t208,
            t209,
            t210,
            t211,
            t212,
            t213,
            t214,
            t217,
            t218,
            t219,
            t221,
            t222,
            t224,
            t225,
            t226,
            t227,
            t228,
            t229,
            t230,
            t231,
            t237,
            t238,
            t239,
            t240,
            t242,
            t243,
            t244,
            t246,
            t247,
            t249,
            t250,
            t251,
            t252,
            t253,
            t254,
            t255,
            t256,
            t259,
            t260,
            t261,
            t262,
            t263,
            t264,
            t265,
            t266,
            t267,
            t268,
            t269,
            t272,
            t273,
            t274,
            t276,
            t277,
            t279,
            t280,
            t281,
            t282,
            t283,
            t284,
            t285,
            t286,
            t289,
            t290,
            t291,
            t292,
            t294,
            t295,
            t296,
            t298,
            t299,
            t301,
            t302,
            t303,
            t304,
            t305,
            t306,
            t307,
            t308,
            t311,
            t312,
            t313,
            t314,
            t316,
            t317,
            t318,
            t320,
            t321,
            t323,
            t324,
            t325,
            t326,
            t327,
            t328,
            t329,
            t330,
            t333,
            t334,
            t335,
            t336,
            t338,
            t339,
            t340,
            t342,
            t343,
            t345,
            t346,
            t347,
            t348,
            t349,
            t350,
            t351,
            t352,
            t355,
            t356,
            t357,
            t358,
            t360,
            t361,
            t362,
            t364,
            t365,
            t367,
            t368,
            t369,
            t370,
            t371,
            t372,
            t373,
            t374,
            t377,
            t378,
            t379,
            t380,
            t382,
            t383,
            t384,
            t386,
            t387,
            t389,
            t390,
            t391,
            t392,
            t393,
            t394,
            t395,
            t396,
            t399,
            t400,
            t401,
            t402,
            t404,
            t405,
            t406,
            t408,
            t409,
            t411,
            t412,
            t413,
            t414,
            t415,
            t416,
            t417,
            t418,
            t421,
            t422,
            t423,
            t424,
            t426,
            t427,
            t428,
            t430,
            t431,
            t433,
            t434,
            t435,
            t436,
            t437,
            t438,
            t439,
            t440,
            t443,
            t444,
            t445,
            t446,
            t448,
            t449,
            t450,
            t452,
            t453,
            t455,
            t456,
            t457,
            t458,
            t459,
            t460,
            t461,
            t462,
            t465,
            t466,
            t467,
            t468,
            t470,
            t471,
            t472,
            t474,
            t475,
            t477,
            t478,
            t479,
            t480,
            t481,
            t482,
            t483,
            t484,
            t487,
            t488,
            t489,
            t490,
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
            t506,
            t509,
            t510,
            t511,
            t512,
            t514,
            t515,
            t516,
            t518,
            t519,
            t521,
            t522,
            t523,
            t524,
            t525,
            t526,
            t527,
            t528,
            t531,
            t532,
            t533,
            t534,
            t536,
            t537,
            t538,
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
            t566,
            t567,
            t568,
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
            t583,
            t584,
            t585,
            t586,
            t588,
            t589,
            t590,
            t592,
            t593,
            t595,
            t596,
            t597,
            t598,
            t599,
            t600,
            t601,
            t602,
            t605,
            t606,
            t607,
            t608,
            t609,
            t610,
            t612,
            t613,
            t614,
        )
