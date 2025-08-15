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
    ):
        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x3x224x224xf32, 192x3x4x4xf32)
        t177 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t178, t179, t180, t181, t182, t183 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t177,
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
        t184 = [48, 144]

        # pd_op.full: (1xi32) <- ()
        t185 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t186 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t187 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t188 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t189 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t190 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t191 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t192 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t193 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t194 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t195 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t196 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t197 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t198 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t199 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t200 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t201 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t202 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t203 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t204 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t205 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t206 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t207 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t208 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t209 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t210 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t211 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t212 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t213 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t214 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t215 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t216 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t217 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t218 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t219 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t220 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t221 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t222 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t223 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t224 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t225 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t226 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t227 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t228 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t229 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t230 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t231 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t232 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t233 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t234 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t235 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t236 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t237 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t238 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t239 = t185

        # pd_op.assign: (1xi32) <- (1xi32)
        t240 = t185

        # pd_op.split: ([-1x48x56x56xf32, -1x144x56x56xf32]) <- (-1x192x56x56xf32, 2xi64, 1xi32)
        t241 = paddle._C_ops.split(t178, t184, t185)

        # builtin.split: (-1x48x56x56xf32, -1x144x56x56xf32) <- ([-1x48x56x56xf32, -1x144x56x56xf32])
        (
            t242,
            t243,
        ) = t241
        del t241

        # pd_op.conv2d: (-1x48x56x56xf32) <- (-1x48x56x56xf32, 48x48x3x3xf32)
        t244 = paddle._C_ops.conv2d(
            t242, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # builtin.combine: ([-1x48x56x56xf32, -1x144x56x56xf32]) <- (-1x48x56x56xf32, -1x144x56x56xf32)
        t245 = [t244, t243]

        # pd_op.concat: (-1x192x56x56xf32) <- ([-1x48x56x56xf32, -1x144x56x56xf32], 1xi32)
        t246 = paddle._C_ops.concat(t245, t185)
        del t245

        # pd_op.conv2d: (-1x384x56x56xf32) <- (-1x192x56x56xf32, 384x192x1x1xf32)
        t247 = paddle._C_ops.conv2d(
            t246, t6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t6

        # pd_op.batch_norm_: (-1x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t248, t249, t250, t251, t252, t253 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t247,
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

        # pd_op.relu: (-1x384x56x56xf32) <- (-1x384x56x56xf32)
        t254 = paddle._C_ops.relu(t248)
        del t248

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x384x56x56xf32, 192x384x1x1xf32)
        t255 = paddle._C_ops.conv2d(
            t254, t11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t11

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x192x56x56xf32)
        t256 = paddle._C_ops.add(t178, t255)

        # pd_op.split: ([-1x48x56x56xf32, -1x144x56x56xf32]) <- (-1x192x56x56xf32, 2xi64, 1xi32)
        t257 = paddle._C_ops.split(t256, t184, t185)

        # builtin.split: (-1x48x56x56xf32, -1x144x56x56xf32) <- ([-1x48x56x56xf32, -1x144x56x56xf32])
        (
            t258,
            t259,
        ) = t257
        del t257

        # pd_op.conv2d: (-1x48x56x56xf32) <- (-1x48x56x56xf32, 48x48x3x3xf32)
        t260 = paddle._C_ops.conv2d(
            t258, t12, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t12

        # builtin.combine: ([-1x48x56x56xf32, -1x144x56x56xf32]) <- (-1x48x56x56xf32, -1x144x56x56xf32)
        t261 = [t260, t259]

        # pd_op.concat: (-1x192x56x56xf32) <- ([-1x48x56x56xf32, -1x144x56x56xf32], 1xi32)
        t262 = paddle._C_ops.concat(t261, t185)
        del t261

        # pd_op.conv2d: (-1x384x56x56xf32) <- (-1x192x56x56xf32, 384x192x1x1xf32)
        t263 = paddle._C_ops.conv2d(
            t262, t13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t13

        # pd_op.batch_norm_: (-1x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t264, t265, t266, t267, t268, t269 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t263,
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

        # pd_op.relu: (-1x384x56x56xf32) <- (-1x384x56x56xf32)
        t270 = paddle._C_ops.relu(t264)
        del t264

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x384x56x56xf32, 192x384x1x1xf32)
        t271 = paddle._C_ops.conv2d(
            t270, t18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t18

        # pd_op.full: (xf32) <- ()
        t272 = paddle._C_ops.full(
            [],
            float("0.988889"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x192x56x56xf32)
        t273 = paddle._C_ops.shape64(t271)

        # pd_op.full_int_array: (1xi64) <- ()
        t274 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        t275 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t276 = paddle._C_ops.slice(t273, [0], t274, t275, [1], [0])
        del t273

        # pd_op.full: (xi64) <- ()
        t277 = paddle._C_ops.full([], float("1"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t278 = [t276, t277, t277, t277]
        del t276

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t279 = paddle._C_ops.stack(t278, 0)
        del t278

        # pd_op.full: (1xf32) <- ()
        t280 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t281 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t282 = paddle._C_ops.uniform(
            t279,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t279

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t283 = paddle._C_ops.add(t272, t282)
        del t282

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t284 = paddle._C_ops.floor(t283)
        del t283

        # pd_op.divide: (-1x192x56x56xf32) <- (-1x192x56x56xf32, xf32)
        t285 = paddle._C_ops.divide(t271, t272)

        # pd_op.multiply: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x1x1x1xf32)
        t286 = paddle._C_ops.multiply(t285, t284)

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x192x56x56xf32)
        t287 = paddle._C_ops.add(t256, t286)

        # pd_op.split: ([-1x48x56x56xf32, -1x144x56x56xf32]) <- (-1x192x56x56xf32, 2xi64, 1xi32)
        t288 = paddle._C_ops.split(t287, t184, t185)
        del t184

        # builtin.split: (-1x48x56x56xf32, -1x144x56x56xf32) <- ([-1x48x56x56xf32, -1x144x56x56xf32])
        (
            t289,
            t290,
        ) = t288
        del t288

        # pd_op.conv2d: (-1x48x56x56xf32) <- (-1x48x56x56xf32, 48x48x3x3xf32)
        t291 = paddle._C_ops.conv2d(
            t289, t19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t19

        # builtin.combine: ([-1x48x56x56xf32, -1x144x56x56xf32]) <- (-1x48x56x56xf32, -1x144x56x56xf32)
        t292 = [t291, t290]

        # pd_op.concat: (-1x192x56x56xf32) <- ([-1x48x56x56xf32, -1x144x56x56xf32], 1xi32)
        t293 = paddle._C_ops.concat(t292, t185)
        del t292

        # pd_op.conv2d: (-1x384x56x56xf32) <- (-1x192x56x56xf32, 384x192x1x1xf32)
        t294 = paddle._C_ops.conv2d(
            t293, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (-1x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t295, t296, t297, t298, t299, t300 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t294,
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

        # pd_op.relu: (-1x384x56x56xf32) <- (-1x384x56x56xf32)
        t301 = paddle._C_ops.relu(t295)
        del t295

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x384x56x56xf32, 192x384x1x1xf32)
        t302 = paddle._C_ops.conv2d(
            t301, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.full: (xf32) <- ()
        t303 = paddle._C_ops.full(
            [],
            float("0.977778"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x192x56x56xf32)
        t304 = paddle._C_ops.shape64(t302)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t305 = paddle._C_ops.slice(t304, [0], t274, t275, [1], [0])
        del t304

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t306 = [t305, t277, t277, t277]
        del t305

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t307 = paddle._C_ops.stack(t306, 0)
        del t306

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t308 = paddle._C_ops.uniform(
            t307,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t307

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t309 = paddle._C_ops.add(t303, t308)
        del t308

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t310 = paddle._C_ops.floor(t309)
        del t309

        # pd_op.divide: (-1x192x56x56xf32) <- (-1x192x56x56xf32, xf32)
        t311 = paddle._C_ops.divide(t302, t303)

        # pd_op.multiply: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x1x1x1xf32)
        t312 = paddle._C_ops.multiply(t311, t310)

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x192x56x56xf32)
        t313 = paddle._C_ops.add(t287, t312)

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x192x56x56xf32, 384x192x2x2xf32)
        t314 = paddle._C_ops.conv2d(
            t313, t26, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t26

        # pd_op.batch_norm_: (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t315, t316, t317, t318, t319, t320 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t314,
                t27,
                t28,
                t29,
                t30,
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
        del t30, t29, t28, t27

        # pd_op.full_int_array: (2xi64) <- ()
        t321 = [96, 288]

        # pd_op.split: ([-1x96x28x28xf32, -1x288x28x28xf32]) <- (-1x384x28x28xf32, 2xi64, 1xi32)
        t322 = paddle._C_ops.split(t315, t321, t185)

        # builtin.split: (-1x96x28x28xf32, -1x288x28x28xf32) <- ([-1x96x28x28xf32, -1x288x28x28xf32])
        (
            t323,
            t324,
        ) = t322
        del t322

        # pd_op.conv2d: (-1x96x28x28xf32) <- (-1x96x28x28xf32, 96x96x3x3xf32)
        t325 = paddle._C_ops.conv2d(
            t323, t31, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t31

        # builtin.combine: ([-1x96x28x28xf32, -1x288x28x28xf32]) <- (-1x96x28x28xf32, -1x288x28x28xf32)
        t326 = [t325, t324]

        # pd_op.concat: (-1x384x28x28xf32) <- ([-1x96x28x28xf32, -1x288x28x28xf32], 1xi32)
        t327 = paddle._C_ops.concat(t326, t185)
        del t326

        # pd_op.conv2d: (-1x768x28x28xf32) <- (-1x384x28x28xf32, 768x384x1x1xf32)
        t328 = paddle._C_ops.conv2d(
            t327, t32, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t32

        # pd_op.batch_norm_: (-1x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t329, t330, t331, t332, t333, t334 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t328,
                t33,
                t34,
                t35,
                t36,
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
        del t36, t35, t34, t33

        # pd_op.relu: (-1x768x28x28xf32) <- (-1x768x28x28xf32)
        t335 = paddle._C_ops.relu(t329)
        del t329

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x768x28x28xf32, 384x768x1x1xf32)
        t336 = paddle._C_ops.conv2d(
            t335, t37, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t37

        # pd_op.full: (xf32) <- ()
        t337 = paddle._C_ops.full(
            [],
            float("0.966667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x28x28xf32)
        t338 = paddle._C_ops.shape64(t336)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t339 = paddle._C_ops.slice(t338, [0], t274, t275, [1], [0])
        del t338

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t340 = [t339, t277, t277, t277]
        del t339

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t341 = paddle._C_ops.stack(t340, 0)
        del t340

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t342 = paddle._C_ops.uniform(
            t341,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t341

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t343 = paddle._C_ops.add(t337, t342)
        del t342

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t344 = paddle._C_ops.floor(t343)
        del t343

        # pd_op.divide: (-1x384x28x28xf32) <- (-1x384x28x28xf32, xf32)
        t345 = paddle._C_ops.divide(t336, t337)

        # pd_op.multiply: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x1x1x1xf32)
        t346 = paddle._C_ops.multiply(t345, t344)

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x384x28x28xf32)
        t347 = paddle._C_ops.add(t315, t346)

        # pd_op.split: ([-1x96x28x28xf32, -1x288x28x28xf32]) <- (-1x384x28x28xf32, 2xi64, 1xi32)
        t348 = paddle._C_ops.split(t347, t321, t185)

        # builtin.split: (-1x96x28x28xf32, -1x288x28x28xf32) <- ([-1x96x28x28xf32, -1x288x28x28xf32])
        (
            t349,
            t350,
        ) = t348
        del t348

        # pd_op.conv2d: (-1x96x28x28xf32) <- (-1x96x28x28xf32, 96x96x3x3xf32)
        t351 = paddle._C_ops.conv2d(
            t349, t38, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t38

        # builtin.combine: ([-1x96x28x28xf32, -1x288x28x28xf32]) <- (-1x96x28x28xf32, -1x288x28x28xf32)
        t352 = [t351, t350]

        # pd_op.concat: (-1x384x28x28xf32) <- ([-1x96x28x28xf32, -1x288x28x28xf32], 1xi32)
        t353 = paddle._C_ops.concat(t352, t185)
        del t352

        # pd_op.conv2d: (-1x768x28x28xf32) <- (-1x384x28x28xf32, 768x384x1x1xf32)
        t354 = paddle._C_ops.conv2d(
            t353, t39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t39

        # pd_op.batch_norm_: (-1x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t355, t356, t357, t358, t359, t360 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t354,
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

        # pd_op.relu: (-1x768x28x28xf32) <- (-1x768x28x28xf32)
        t361 = paddle._C_ops.relu(t355)
        del t355

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x768x28x28xf32, 384x768x1x1xf32)
        t362 = paddle._C_ops.conv2d(
            t361, t44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t44

        # pd_op.full: (xf32) <- ()
        t363 = paddle._C_ops.full(
            [],
            float("0.955556"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x28x28xf32)
        t364 = paddle._C_ops.shape64(t362)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t365 = paddle._C_ops.slice(t364, [0], t274, t275, [1], [0])
        del t364

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t366 = [t365, t277, t277, t277]
        del t365

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t367 = paddle._C_ops.stack(t366, 0)
        del t366

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t368 = paddle._C_ops.uniform(
            t367,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t367

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t369 = paddle._C_ops.add(t363, t368)
        del t368

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t370 = paddle._C_ops.floor(t369)
        del t369

        # pd_op.divide: (-1x384x28x28xf32) <- (-1x384x28x28xf32, xf32)
        t371 = paddle._C_ops.divide(t362, t363)

        # pd_op.multiply: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x1x1x1xf32)
        t372 = paddle._C_ops.multiply(t371, t370)

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x384x28x28xf32)
        t373 = paddle._C_ops.add(t347, t372)

        # pd_op.split: ([-1x96x28x28xf32, -1x288x28x28xf32]) <- (-1x384x28x28xf32, 2xi64, 1xi32)
        t374 = paddle._C_ops.split(t373, t321, t185)

        # builtin.split: (-1x96x28x28xf32, -1x288x28x28xf32) <- ([-1x96x28x28xf32, -1x288x28x28xf32])
        (
            t375,
            t376,
        ) = t374
        del t374

        # pd_op.conv2d: (-1x96x28x28xf32) <- (-1x96x28x28xf32, 96x96x3x3xf32)
        t377 = paddle._C_ops.conv2d(
            t375, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # builtin.combine: ([-1x96x28x28xf32, -1x288x28x28xf32]) <- (-1x96x28x28xf32, -1x288x28x28xf32)
        t378 = [t377, t376]

        # pd_op.concat: (-1x384x28x28xf32) <- ([-1x96x28x28xf32, -1x288x28x28xf32], 1xi32)
        t379 = paddle._C_ops.concat(t378, t185)
        del t378

        # pd_op.conv2d: (-1x768x28x28xf32) <- (-1x384x28x28xf32, 768x384x1x1xf32)
        t380 = paddle._C_ops.conv2d(
            t379, t46, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t46

        # pd_op.batch_norm_: (-1x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t381, t382, t383, t384, t385, t386 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t380,
                t47,
                t48,
                t49,
                t50,
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
        del t50, t49, t48, t47

        # pd_op.relu: (-1x768x28x28xf32) <- (-1x768x28x28xf32)
        t387 = paddle._C_ops.relu(t381)
        del t381

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x768x28x28xf32, 384x768x1x1xf32)
        t388 = paddle._C_ops.conv2d(
            t387, t51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t51

        # pd_op.full: (xf32) <- ()
        t389 = paddle._C_ops.full(
            [],
            float("0.944444"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x28x28xf32)
        t390 = paddle._C_ops.shape64(t388)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t391 = paddle._C_ops.slice(t390, [0], t274, t275, [1], [0])
        del t390

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t392 = [t391, t277, t277, t277]
        del t391

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t393 = paddle._C_ops.stack(t392, 0)
        del t392

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t394 = paddle._C_ops.uniform(
            t393,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t393

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t395 = paddle._C_ops.add(t389, t394)
        del t394

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t396 = paddle._C_ops.floor(t395)
        del t395

        # pd_op.divide: (-1x384x28x28xf32) <- (-1x384x28x28xf32, xf32)
        t397 = paddle._C_ops.divide(t388, t389)

        # pd_op.multiply: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x1x1x1xf32)
        t398 = paddle._C_ops.multiply(t397, t396)

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x384x28x28xf32)
        t399 = paddle._C_ops.add(t373, t398)

        # pd_op.split: ([-1x96x28x28xf32, -1x288x28x28xf32]) <- (-1x384x28x28xf32, 2xi64, 1xi32)
        t400 = paddle._C_ops.split(t399, t321, t185)
        del t321

        # builtin.split: (-1x96x28x28xf32, -1x288x28x28xf32) <- ([-1x96x28x28xf32, -1x288x28x28xf32])
        (
            t401,
            t402,
        ) = t400
        del t400

        # pd_op.conv2d: (-1x96x28x28xf32) <- (-1x96x28x28xf32, 96x96x3x3xf32)
        t403 = paddle._C_ops.conv2d(
            t401, t52, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t52

        # builtin.combine: ([-1x96x28x28xf32, -1x288x28x28xf32]) <- (-1x96x28x28xf32, -1x288x28x28xf32)
        t404 = [t403, t402]

        # pd_op.concat: (-1x384x28x28xf32) <- ([-1x96x28x28xf32, -1x288x28x28xf32], 1xi32)
        t405 = paddle._C_ops.concat(t404, t185)
        del t404

        # pd_op.conv2d: (-1x768x28x28xf32) <- (-1x384x28x28xf32, 768x384x1x1xf32)
        t406 = paddle._C_ops.conv2d(
            t405, t53, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t53

        # pd_op.batch_norm_: (-1x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t407, t408, t409, t410, t411, t412 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t406,
                t54,
                t55,
                t56,
                t57,
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
        del t57, t56, t55, t54

        # pd_op.relu: (-1x768x28x28xf32) <- (-1x768x28x28xf32)
        t413 = paddle._C_ops.relu(t407)
        del t407

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x768x28x28xf32, 384x768x1x1xf32)
        t414 = paddle._C_ops.conv2d(
            t413, t58, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t58

        # pd_op.full: (xf32) <- ()
        t415 = paddle._C_ops.full(
            [],
            float("0.933333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x28x28xf32)
        t416 = paddle._C_ops.shape64(t414)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t417 = paddle._C_ops.slice(t416, [0], t274, t275, [1], [0])
        del t416

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t418 = [t417, t277, t277, t277]
        del t417

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t419 = paddle._C_ops.stack(t418, 0)
        del t418

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t420 = paddle._C_ops.uniform(
            t419,
            paddle.float32,
            t280,
            t281,
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

        # pd_op.divide: (-1x384x28x28xf32) <- (-1x384x28x28xf32, xf32)
        t423 = paddle._C_ops.divide(t414, t415)

        # pd_op.multiply: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x1x1x1xf32)
        t424 = paddle._C_ops.multiply(t423, t422)

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x384x28x28xf32)
        t425 = paddle._C_ops.add(t399, t424)

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x28x28xf32, 768x384x2x2xf32)
        t426 = paddle._C_ops.conv2d(
            t425, t59, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t59

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t427, t428, t429, t430, t431, t432 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t426,
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

        # pd_op.full_int_array: (2xi64) <- ()
        t433 = [192, 576]

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t434 = paddle._C_ops.split(t427, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t435,
            t436,
        ) = t434
        del t434

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t437 = paddle._C_ops.conv2d(
            t435, t64, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t64

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t438 = [t437, t436]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t439 = paddle._C_ops.concat(t438, t185)
        del t438

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t440 = paddle._C_ops.conv2d(
            t439, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t441, t442, t443, t444, t445, t446 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t440,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t447 = paddle._C_ops.relu(t441)
        del t441

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t448 = paddle._C_ops.conv2d(
            t447, t70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.full: (xf32) <- ()
        t449 = paddle._C_ops.full(
            [],
            float("0.922222"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t450 = paddle._C_ops.shape64(t448)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t451 = paddle._C_ops.slice(t450, [0], t274, t275, [1], [0])
        del t450

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t452 = [t451, t277, t277, t277]
        del t451

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t453 = paddle._C_ops.stack(t452, 0)
        del t452

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t454 = paddle._C_ops.uniform(
            t453,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t453

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t455 = paddle._C_ops.add(t449, t454)
        del t454

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t456 = paddle._C_ops.floor(t455)
        del t455

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t457 = paddle._C_ops.divide(t448, t449)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t458 = paddle._C_ops.multiply(t457, t456)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t459 = paddle._C_ops.add(t427, t458)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t460 = paddle._C_ops.split(t459, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t461,
            t462,
        ) = t460
        del t460

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t463 = paddle._C_ops.conv2d(
            t461, t71, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t71

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t464 = [t463, t462]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t465 = paddle._C_ops.concat(t464, t185)
        del t464

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t466 = paddle._C_ops.conv2d(
            t465, t72, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t72

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t467, t468, t469, t470, t471, t472 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t466,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t473 = paddle._C_ops.relu(t467)
        del t467

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t474 = paddle._C_ops.conv2d(
            t473, t77, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t77

        # pd_op.full: (xf32) <- ()
        t475 = paddle._C_ops.full(
            [],
            float("0.911111"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t476 = paddle._C_ops.shape64(t474)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t477 = paddle._C_ops.slice(t476, [0], t274, t275, [1], [0])
        del t476

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t478 = [t477, t277, t277, t277]
        del t477

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t479 = paddle._C_ops.stack(t478, 0)
        del t478

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t480 = paddle._C_ops.uniform(
            t479,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t479

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t481 = paddle._C_ops.add(t475, t480)
        del t480

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t482 = paddle._C_ops.floor(t481)
        del t481

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t483 = paddle._C_ops.divide(t474, t475)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t484 = paddle._C_ops.multiply(t483, t482)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t485 = paddle._C_ops.add(t459, t484)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t486 = paddle._C_ops.split(t485, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t487,
            t488,
        ) = t486
        del t486

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t489 = paddle._C_ops.conv2d(
            t487, t78, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t78

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t490 = [t489, t488]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t491 = paddle._C_ops.concat(t490, t185)
        del t490

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t492 = paddle._C_ops.conv2d(
            t491, t79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t79

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t493, t494, t495, t496, t497, t498 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t492,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t499 = paddle._C_ops.relu(t493)
        del t493

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t500 = paddle._C_ops.conv2d(
            t499, t84, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t84

        # pd_op.full: (xf32) <- ()
        t501 = paddle._C_ops.full(
            [], float("0.9"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t502 = paddle._C_ops.shape64(t500)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t503 = paddle._C_ops.slice(t502, [0], t274, t275, [1], [0])
        del t502

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t504 = [t503, t277, t277, t277]
        del t503

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t505 = paddle._C_ops.stack(t504, 0)
        del t504

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t506 = paddle._C_ops.uniform(
            t505,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t505

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t507 = paddle._C_ops.add(t501, t506)
        del t506

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t508 = paddle._C_ops.floor(t507)
        del t507

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t509 = paddle._C_ops.divide(t500, t501)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t510 = paddle._C_ops.multiply(t509, t508)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t511 = paddle._C_ops.add(t485, t510)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t512 = paddle._C_ops.split(t511, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t513,
            t514,
        ) = t512
        del t512

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t515 = paddle._C_ops.conv2d(
            t513, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t516 = [t515, t514]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t517 = paddle._C_ops.concat(t516, t185)
        del t516

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t518 = paddle._C_ops.conv2d(
            t517, t86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t86

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t519, t520, t521, t522, t523, t524 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t518,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t525 = paddle._C_ops.relu(t519)
        del t519

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t526 = paddle._C_ops.conv2d(
            t525, t91, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t91

        # pd_op.full: (xf32) <- ()
        t527 = paddle._C_ops.full(
            [],
            float("0.888889"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t528 = paddle._C_ops.shape64(t526)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t529 = paddle._C_ops.slice(t528, [0], t274, t275, [1], [0])
        del t528

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t530 = [t529, t277, t277, t277]
        del t529

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t531 = paddle._C_ops.stack(t530, 0)
        del t530

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t532 = paddle._C_ops.uniform(
            t531,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t531

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t533 = paddle._C_ops.add(t527, t532)
        del t532

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t534 = paddle._C_ops.floor(t533)
        del t533

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t535 = paddle._C_ops.divide(t526, t527)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t536 = paddle._C_ops.multiply(t535, t534)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t537 = paddle._C_ops.add(t511, t536)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t538 = paddle._C_ops.split(t537, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t539,
            t540,
        ) = t538
        del t538

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t541 = paddle._C_ops.conv2d(
            t539, t92, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t92

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t542 = [t541, t540]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t543 = paddle._C_ops.concat(t542, t185)
        del t542

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t544 = paddle._C_ops.conv2d(
            t543, t93, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t93

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t545, t546, t547, t548, t549, t550 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t544,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t551 = paddle._C_ops.relu(t545)
        del t545

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t552 = paddle._C_ops.conv2d(
            t551, t98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t98

        # pd_op.full: (xf32) <- ()
        t553 = paddle._C_ops.full(
            [],
            float("0.877778"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t554 = paddle._C_ops.shape64(t552)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t555 = paddle._C_ops.slice(t554, [0], t274, t275, [1], [0])
        del t554

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t556 = [t555, t277, t277, t277]
        del t555

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t557 = paddle._C_ops.stack(t556, 0)
        del t556

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t558 = paddle._C_ops.uniform(
            t557,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t557

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t559 = paddle._C_ops.add(t553, t558)
        del t558

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t560 = paddle._C_ops.floor(t559)
        del t559

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t561 = paddle._C_ops.divide(t552, t553)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t562 = paddle._C_ops.multiply(t561, t560)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t563 = paddle._C_ops.add(t537, t562)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t564 = paddle._C_ops.split(t563, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t565,
            t566,
        ) = t564
        del t564

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t567 = paddle._C_ops.conv2d(
            t565, t99, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t99

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t568 = [t567, t566]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t569 = paddle._C_ops.concat(t568, t185)
        del t568

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t570 = paddle._C_ops.conv2d(
            t569, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t571, t572, t573, t574, t575, t576 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t570,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t577 = paddle._C_ops.relu(t571)
        del t571

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t578 = paddle._C_ops.conv2d(
            t577, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.full: (xf32) <- ()
        t579 = paddle._C_ops.full(
            [],
            float("0.866667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t580 = paddle._C_ops.shape64(t578)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t581 = paddle._C_ops.slice(t580, [0], t274, t275, [1], [0])
        del t580

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t582 = [t581, t277, t277, t277]
        del t581

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t583 = paddle._C_ops.stack(t582, 0)
        del t582

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t584 = paddle._C_ops.uniform(
            t583,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t583

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t585 = paddle._C_ops.add(t579, t584)
        del t584

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t586 = paddle._C_ops.floor(t585)
        del t585

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t587 = paddle._C_ops.divide(t578, t579)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t588 = paddle._C_ops.multiply(t587, t586)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t589 = paddle._C_ops.add(t563, t588)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t590 = paddle._C_ops.split(t589, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t591,
            t592,
        ) = t590
        del t590

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t593 = paddle._C_ops.conv2d(
            t591, t106, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t106

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t594 = [t593, t592]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t595 = paddle._C_ops.concat(t594, t185)
        del t594

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t596 = paddle._C_ops.conv2d(
            t595, t107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t107

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t597, t598, t599, t600, t601, t602 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t596,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t603 = paddle._C_ops.relu(t597)
        del t597

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t604 = paddle._C_ops.conv2d(
            t603, t112, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t112

        # pd_op.full: (xf32) <- ()
        t605 = paddle._C_ops.full(
            [],
            float("0.855556"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t606 = paddle._C_ops.shape64(t604)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t607 = paddle._C_ops.slice(t606, [0], t274, t275, [1], [0])
        del t606

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t608 = [t607, t277, t277, t277]
        del t607

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t609 = paddle._C_ops.stack(t608, 0)
        del t608

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t610 = paddle._C_ops.uniform(
            t609,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t609

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t611 = paddle._C_ops.add(t605, t610)
        del t610

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t612 = paddle._C_ops.floor(t611)
        del t611

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t613 = paddle._C_ops.divide(t604, t605)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t614 = paddle._C_ops.multiply(t613, t612)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t615 = paddle._C_ops.add(t589, t614)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t616 = paddle._C_ops.split(t615, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t617,
            t618,
        ) = t616
        del t616

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t619 = paddle._C_ops.conv2d(
            t617, t113, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t113

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t620 = [t619, t618]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t621 = paddle._C_ops.concat(t620, t185)
        del t620

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t622 = paddle._C_ops.conv2d(
            t621, t114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t114

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t623, t624, t625, t626, t627, t628 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t622,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t629 = paddle._C_ops.relu(t623)
        del t623

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t630 = paddle._C_ops.conv2d(
            t629, t119, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t119

        # pd_op.full: (xf32) <- ()
        t631 = paddle._C_ops.full(
            [],
            float("0.844444"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t632 = paddle._C_ops.shape64(t630)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t633 = paddle._C_ops.slice(t632, [0], t274, t275, [1], [0])
        del t632

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t634 = [t633, t277, t277, t277]
        del t633

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t635 = paddle._C_ops.stack(t634, 0)
        del t634

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t636 = paddle._C_ops.uniform(
            t635,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t635

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t637 = paddle._C_ops.add(t631, t636)
        del t636

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t638 = paddle._C_ops.floor(t637)
        del t637

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t639 = paddle._C_ops.divide(t630, t631)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t640 = paddle._C_ops.multiply(t639, t638)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t641 = paddle._C_ops.add(t615, t640)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t642 = paddle._C_ops.split(t641, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t643,
            t644,
        ) = t642
        del t642

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t645 = paddle._C_ops.conv2d(
            t643, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t646 = [t645, t644]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t647 = paddle._C_ops.concat(t646, t185)
        del t646

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t648 = paddle._C_ops.conv2d(
            t647, t121, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t121

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t649, t650, t651, t652, t653, t654 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t648,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t655 = paddle._C_ops.relu(t649)
        del t649

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t656 = paddle._C_ops.conv2d(
            t655, t126, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t126

        # pd_op.full: (xf32) <- ()
        t657 = paddle._C_ops.full(
            [],
            float("0.833333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t658 = paddle._C_ops.shape64(t656)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t659 = paddle._C_ops.slice(t658, [0], t274, t275, [1], [0])
        del t658

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t660 = [t659, t277, t277, t277]
        del t659

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t661 = paddle._C_ops.stack(t660, 0)
        del t660

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t662 = paddle._C_ops.uniform(
            t661,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t661

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t663 = paddle._C_ops.add(t657, t662)
        del t662

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t664 = paddle._C_ops.floor(t663)
        del t663

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t665 = paddle._C_ops.divide(t656, t657)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t666 = paddle._C_ops.multiply(t665, t664)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t667 = paddle._C_ops.add(t641, t666)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t668 = paddle._C_ops.split(t667, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t669,
            t670,
        ) = t668
        del t668

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t671 = paddle._C_ops.conv2d(
            t669, t127, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t127

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t672 = [t671, t670]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t673 = paddle._C_ops.concat(t672, t185)
        del t672

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t674 = paddle._C_ops.conv2d(
            t673, t128, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t128

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t675, t676, t677, t678, t679, t680 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t674,
                t129,
                t130,
                t131,
                t132,
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
        del t132, t131, t130, t129

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t681 = paddle._C_ops.relu(t675)
        del t675

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t682 = paddle._C_ops.conv2d(
            t681, t133, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t133

        # pd_op.full: (xf32) <- ()
        t683 = paddle._C_ops.full(
            [],
            float("0.822222"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t684 = paddle._C_ops.shape64(t682)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t685 = paddle._C_ops.slice(t684, [0], t274, t275, [1], [0])
        del t684

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t686 = [t685, t277, t277, t277]
        del t685

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t687 = paddle._C_ops.stack(t686, 0)
        del t686

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t688 = paddle._C_ops.uniform(
            t687,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t687

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t689 = paddle._C_ops.add(t683, t688)
        del t688

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t690 = paddle._C_ops.floor(t689)
        del t689

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t691 = paddle._C_ops.divide(t682, t683)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t692 = paddle._C_ops.multiply(t691, t690)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t693 = paddle._C_ops.add(t667, t692)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t694 = paddle._C_ops.split(t693, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t695,
            t696,
        ) = t694
        del t694

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t697 = paddle._C_ops.conv2d(
            t695, t134, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t134

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t698 = [t697, t696]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t699 = paddle._C_ops.concat(t698, t185)
        del t698

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t700 = paddle._C_ops.conv2d(
            t699, t135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t701, t702, t703, t704, t705, t706 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t700,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t707 = paddle._C_ops.relu(t701)
        del t701

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t708 = paddle._C_ops.conv2d(
            t707, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.full: (xf32) <- ()
        t709 = paddle._C_ops.full(
            [],
            float("0.811111"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t710 = paddle._C_ops.shape64(t708)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t711 = paddle._C_ops.slice(t710, [0], t274, t275, [1], [0])
        del t710

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t712 = [t711, t277, t277, t277]
        del t711

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t713 = paddle._C_ops.stack(t712, 0)
        del t712

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t714 = paddle._C_ops.uniform(
            t713,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t713

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t715 = paddle._C_ops.add(t709, t714)
        del t714

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t716 = paddle._C_ops.floor(t715)
        del t715

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t717 = paddle._C_ops.divide(t708, t709)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t718 = paddle._C_ops.multiply(t717, t716)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t719 = paddle._C_ops.add(t693, t718)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t720 = paddle._C_ops.split(t719, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t721,
            t722,
        ) = t720
        del t720

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t723 = paddle._C_ops.conv2d(
            t721, t141, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t141

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t724 = [t723, t722]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t725 = paddle._C_ops.concat(t724, t185)
        del t724

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t726 = paddle._C_ops.conv2d(
            t725, t142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t142

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t727, t728, t729, t730, t731, t732 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t726,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t733 = paddle._C_ops.relu(t727)
        del t727

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t734 = paddle._C_ops.conv2d(
            t733, t147, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t147

        # pd_op.full: (xf32) <- ()
        t735 = paddle._C_ops.full(
            [], float("0.8"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t736 = paddle._C_ops.shape64(t734)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t737 = paddle._C_ops.slice(t736, [0], t274, t275, [1], [0])
        del t736

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t738 = [t737, t277, t277, t277]
        del t737

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t739 = paddle._C_ops.stack(t738, 0)
        del t738

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t740 = paddle._C_ops.uniform(
            t739,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t739

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t741 = paddle._C_ops.add(t735, t740)
        del t740

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t742 = paddle._C_ops.floor(t741)
        del t741

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t743 = paddle._C_ops.divide(t734, t735)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t744 = paddle._C_ops.multiply(t743, t742)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t745 = paddle._C_ops.add(t719, t744)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t746 = paddle._C_ops.split(t745, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t747,
            t748,
        ) = t746
        del t746

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t749 = paddle._C_ops.conv2d(
            t747, t148, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t148

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t750 = [t749, t748]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t751 = paddle._C_ops.concat(t750, t185)
        del t750

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t752 = paddle._C_ops.conv2d(
            t751, t149, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t149

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t753, t754, t755, t756, t757, t758 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t752,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t759 = paddle._C_ops.relu(t753)
        del t753

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t760 = paddle._C_ops.conv2d(
            t759, t154, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t154

        # pd_op.full: (xf32) <- ()
        t761 = paddle._C_ops.full(
            [],
            float("0.788889"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t762 = paddle._C_ops.shape64(t760)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t763 = paddle._C_ops.slice(t762, [0], t274, t275, [1], [0])
        del t762

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t764 = [t763, t277, t277, t277]
        del t763

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t765 = paddle._C_ops.stack(t764, 0)
        del t764

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t766 = paddle._C_ops.uniform(
            t765,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t765

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t767 = paddle._C_ops.add(t761, t766)
        del t766

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t768 = paddle._C_ops.floor(t767)
        del t767

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t769 = paddle._C_ops.divide(t760, t761)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t770 = paddle._C_ops.multiply(t769, t768)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t771 = paddle._C_ops.add(t745, t770)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t772 = paddle._C_ops.split(t771, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t773,
            t774,
        ) = t772
        del t772

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t775 = paddle._C_ops.conv2d(
            t773, t155, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t776 = [t775, t774]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t777 = paddle._C_ops.concat(t776, t185)
        del t776

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t778 = paddle._C_ops.conv2d(
            t777, t156, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t779, t780, t781, t782, t783, t784 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t778,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t785 = paddle._C_ops.relu(t779)
        del t779

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t786 = paddle._C_ops.conv2d(
            t785, t161, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t161

        # pd_op.full: (xf32) <- ()
        t787 = paddle._C_ops.full(
            [],
            float("0.777778"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t788 = paddle._C_ops.shape64(t786)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t789 = paddle._C_ops.slice(t788, [0], t274, t275, [1], [0])
        del t788

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t790 = [t789, t277, t277, t277]
        del t789

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t791 = paddle._C_ops.stack(t790, 0)
        del t790

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t792 = paddle._C_ops.uniform(
            t791,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t791

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t793 = paddle._C_ops.add(t787, t792)
        del t792

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t794 = paddle._C_ops.floor(t793)
        del t793

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t795 = paddle._C_ops.divide(t786, t787)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t796 = paddle._C_ops.multiply(t795, t794)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t797 = paddle._C_ops.add(t771, t796)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t798 = paddle._C_ops.split(t797, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t799,
            t800,
        ) = t798
        del t798

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t801 = paddle._C_ops.conv2d(
            t799, t162, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t162

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t802 = [t801, t800]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t803 = paddle._C_ops.concat(t802, t185)
        del t802

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t804 = paddle._C_ops.conv2d(
            t803, t163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t163

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t805, t806, t807, t808, t809, t810 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t804,
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

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t811 = paddle._C_ops.relu(t805)
        del t805

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t812 = paddle._C_ops.conv2d(
            t811, t168, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t168

        # pd_op.full: (xf32) <- ()
        t813 = paddle._C_ops.full(
            [],
            float("0.766667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t814 = paddle._C_ops.shape64(t812)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t815 = paddle._C_ops.slice(t814, [0], t274, t275, [1], [0])
        del t814

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t816 = [t815, t277, t277, t277]
        del t815

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t817 = paddle._C_ops.stack(t816, 0)
        del t816

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t818 = paddle._C_ops.uniform(
            t817,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t817

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t819 = paddle._C_ops.add(t813, t818)
        del t818

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t820 = paddle._C_ops.floor(t819)
        del t819

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t821 = paddle._C_ops.divide(t812, t813)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t822 = paddle._C_ops.multiply(t821, t820)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t823 = paddle._C_ops.add(t797, t822)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t824 = paddle._C_ops.split(t823, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t825,
            t826,
        ) = t824
        del t824

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t827 = paddle._C_ops.conv2d(
            t825, t169, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t169

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t828 = [t827, t826]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t829 = paddle._C_ops.concat(t828, t185)
        del t828

        # pd_op.conv2d: (-1x1536x14x14xf32) <- (-1x768x14x14xf32, 1536x768x1x1xf32)
        t830 = paddle._C_ops.conv2d(
            t829, t170, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t831, t832, t833, t834, t835, t836 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t830,
                t171,
                t172,
                t173,
                t174,
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
        del t174, t173, t172, t171

        # pd_op.relu: (-1x1536x14x14xf32) <- (-1x1536x14x14xf32)
        t837 = paddle._C_ops.relu(t831)
        del t831

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x1536x14x14xf32, 768x1536x1x1xf32)
        t838 = paddle._C_ops.conv2d(
            t837, t175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.full: (xf32) <- ()
        t839 = paddle._C_ops.full(
            [],
            float("0.755556"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        t840 = paddle._C_ops.shape64(t838)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t841 = paddle._C_ops.slice(t840, [0], t274, t275, [1], [0])
        del t840

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t842 = [t841, t277, t277, t277]
        del t841

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t843 = paddle._C_ops.stack(t842, 0)
        del t842

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t844 = paddle._C_ops.uniform(
            t843,
            paddle.float32,
            t280,
            t281,
            0,
            paddle.framework._current_expected_place(),
        )
        del t843

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t845 = paddle._C_ops.add(t839, t844)
        del t844

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t846 = paddle._C_ops.floor(t845)
        del t845

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, xf32)
        t847 = paddle._C_ops.divide(t838, t839)

        # pd_op.multiply: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x1x1xf32)
        t848 = paddle._C_ops.multiply(t847, t846)

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t849 = paddle._C_ops.add(t823, t848)

        # pd_op.split: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x768x14x14xf32, 2xi64, 1xi32)
        t850 = paddle._C_ops.split(t849, t433, t185)

        # builtin.split: (-1x192x14x14xf32, -1x576x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32])
        (
            t851,
            t852,
        ) = t850
        del t850

        # pd_op.conv2d: (-1x192x14x14xf32) <- (-1x192x14x14xf32, 192x192x3x3xf32)
        t853 = paddle._C_ops.conv2d(
            t851, t176, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t176

        # builtin.combine: ([-1x192x14x14xf32, -1x576x14x14xf32]) <- (-1x192x14x14xf32, -1x576x14x14xf32)
        t854 = [t853, t852]

        # pd_op.concat: (-1x768x14x14xf32) <- ([-1x192x14x14xf32, -1x576x14x14xf32], 1xi32)
        t855 = paddle._C_ops.concat(t854, t185)
        del t854

        return (
            t177,
            t178,
            t179,
            t180,
            t181,
            t182,
            t183,
            t185,
            t186,
            t187,
            t188,
            t189,
            t190,
            t191,
            t192,
            t193,
            t194,
            t195,
            t196,
            t197,
            t198,
            t199,
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
            t215,
            t216,
            t217,
            t218,
            t219,
            t220,
            t221,
            t222,
            t223,
            t224,
            t225,
            t226,
            t227,
            t228,
            t229,
            t230,
            t231,
            t232,
            t233,
            t234,
            t235,
            t236,
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
            t258,
            t259,
            t260,
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
            t274,
            t275,
            t277,
            t280,
            t281,
            t284,
            t285,
            t286,
            t287,
            t289,
            t290,
            t291,
            t293,
            t294,
            t296,
            t297,
            t298,
            t299,
            t300,
            t301,
            t302,
            t303,
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
            t323,
            t324,
            t325,
            t327,
            t328,
            t330,
            t331,
            t332,
            t333,
            t334,
            t335,
            t336,
            t337,
            t344,
            t345,
            t346,
            t347,
            t349,
            t350,
            t351,
            t353,
            t354,
            t356,
            t357,
            t358,
            t359,
            t360,
            t361,
            t362,
            t363,
            t370,
            t371,
            t372,
            t373,
            t375,
            t376,
            t377,
            t379,
            t380,
            t382,
            t383,
            t384,
            t385,
            t386,
            t387,
            t388,
            t389,
            t396,
            t397,
            t398,
            t399,
            t401,
            t402,
            t403,
            t405,
            t406,
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
            t435,
            t436,
            t437,
            t439,
            t440,
            t442,
            t443,
            t444,
            t445,
            t446,
            t447,
            t448,
            t449,
            t456,
            t457,
            t458,
            t459,
            t461,
            t462,
            t463,
            t465,
            t466,
            t468,
            t469,
            t470,
            t471,
            t472,
            t473,
            t474,
            t475,
            t482,
            t483,
            t484,
            t485,
            t487,
            t488,
            t489,
            t491,
            t492,
            t494,
            t495,
            t496,
            t497,
            t498,
            t499,
            t500,
            t501,
            t508,
            t509,
            t510,
            t511,
            t513,
            t514,
            t515,
            t517,
            t518,
            t520,
            t521,
            t522,
            t523,
            t524,
            t525,
            t526,
            t527,
            t534,
            t535,
            t536,
            t537,
            t539,
            t540,
            t541,
            t543,
            t544,
            t546,
            t547,
            t548,
            t549,
            t550,
            t551,
            t552,
            t553,
            t560,
            t561,
            t562,
            t563,
            t565,
            t566,
            t567,
            t569,
            t570,
            t572,
            t573,
            t574,
            t575,
            t576,
            t577,
            t578,
            t579,
            t586,
            t587,
            t588,
            t589,
            t591,
            t592,
            t593,
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
            t612,
            t613,
            t614,
            t615,
            t617,
            t618,
            t619,
            t621,
            t622,
            t624,
            t625,
            t626,
            t627,
            t628,
            t629,
            t630,
            t631,
            t638,
            t639,
            t640,
            t641,
            t643,
            t644,
            t645,
            t647,
            t648,
            t650,
            t651,
            t652,
            t653,
            t654,
            t655,
            t656,
            t657,
            t664,
            t665,
            t666,
            t667,
            t669,
            t670,
            t671,
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
            t690,
            t691,
            t692,
            t693,
            t695,
            t696,
            t697,
            t699,
            t700,
            t702,
            t703,
            t704,
            t705,
            t706,
            t707,
            t708,
            t709,
            t716,
            t717,
            t718,
            t719,
            t721,
            t722,
            t723,
            t725,
            t726,
            t728,
            t729,
            t730,
            t731,
            t732,
            t733,
            t734,
            t735,
            t742,
            t743,
            t744,
            t745,
            t747,
            t748,
            t749,
            t751,
            t752,
            t754,
            t755,
            t756,
            t757,
            t758,
            t759,
            t760,
            t761,
            t768,
            t769,
            t770,
            t771,
            t773,
            t774,
            t775,
            t777,
            t778,
            t780,
            t781,
            t782,
            t783,
            t784,
            t785,
            t786,
            t787,
            t794,
            t795,
            t796,
            t797,
            t799,
            t800,
            t801,
            t803,
            t804,
            t806,
            t807,
            t808,
            t809,
            t810,
            t811,
            t812,
            t813,
            t820,
            t821,
            t822,
            t823,
            t825,
            t826,
            t827,
            t829,
            t830,
            t832,
            t833,
            t834,
            t835,
            t836,
            t837,
            t838,
            t839,
            t846,
            t847,
            t848,
            t849,
            t851,
            t852,
            t853,
            t855,
        )
