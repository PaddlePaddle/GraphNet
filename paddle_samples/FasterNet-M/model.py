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
    ):
        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x3x224x224xf32, 144x3x4x4xf32)
        t218 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        t219, t220, t221, t222, t223, t224 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t218,
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
        t225 = [36, 108]

        # pd_op.full: (1xi32) <- ()
        t226 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t227 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t228 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t229 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t230 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t231 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t232 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t233 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t234 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t235 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t236 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t237 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t238 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t239 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t240 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t241 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t242 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t243 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t244 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t245 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t246 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t247 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t248 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t249 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t250 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t251 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t252 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t253 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t254 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t255 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t256 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t257 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t258 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t259 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t260 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t261 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t262 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t263 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t264 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t265 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t266 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t267 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t268 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t269 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t270 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t271 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t272 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t273 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t274 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t275 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t276 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t277 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t278 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t279 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t280 = t226

        # pd_op.assign: (1xi32) <- (1xi32)
        t281 = t226

        # pd_op.split: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        t282 = paddle._C_ops.split(t219, t225, t226)

        # builtin.split: (-1x36x56x56xf32, -1x108x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32])
        (
            t283,
            t284,
        ) = t282
        del t282

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x36x56x56xf32, 36x36x3x3xf32)
        t285 = paddle._C_ops.conv2d(
            t283, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # builtin.combine: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x36x56x56xf32, -1x108x56x56xf32)
        t286 = [t285, t284]

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32], 1xi32)
        t287 = paddle._C_ops.concat(t286, t226)
        del t286

        # pd_op.conv2d: (-1x288x56x56xf32) <- (-1x144x56x56xf32, 288x144x1x1xf32)
        t288 = paddle._C_ops.conv2d(
            t287, t6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t6

        # pd_op.batch_norm_: (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t289, t290, t291, t292, t293, t294 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t288,
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

        # pd_op.relu: (-1x288x56x56xf32) <- (-1x288x56x56xf32)
        t295 = paddle._C_ops.relu(t289)
        del t289

        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x288x56x56xf32, 144x288x1x1xf32)
        t296 = paddle._C_ops.conv2d(
            t295, t11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t11

        # pd_op.add: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x144x56x56xf32)
        t297 = paddle._C_ops.add(t219, t296)

        # pd_op.split: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        t298 = paddle._C_ops.split(t297, t225, t226)

        # builtin.split: (-1x36x56x56xf32, -1x108x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32])
        (
            t299,
            t300,
        ) = t298
        del t298

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x36x56x56xf32, 36x36x3x3xf32)
        t301 = paddle._C_ops.conv2d(
            t299, t12, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t12

        # builtin.combine: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x36x56x56xf32, -1x108x56x56xf32)
        t302 = [t301, t300]

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32], 1xi32)
        t303 = paddle._C_ops.concat(t302, t226)
        del t302

        # pd_op.conv2d: (-1x288x56x56xf32) <- (-1x144x56x56xf32, 288x144x1x1xf32)
        t304 = paddle._C_ops.conv2d(
            t303, t13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t13

        # pd_op.batch_norm_: (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t305, t306, t307, t308, t309, t310 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t304,
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

        # pd_op.relu: (-1x288x56x56xf32) <- (-1x288x56x56xf32)
        t311 = paddle._C_ops.relu(t305)
        del t305

        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x288x56x56xf32, 144x288x1x1xf32)
        t312 = paddle._C_ops.conv2d(
            t311, t18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t18

        # pd_op.full: (xf32) <- ()
        t313 = paddle._C_ops.full(
            [],
            float("0.992593"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x144x56x56xf32)
        t314 = paddle._C_ops.shape64(t312)

        # pd_op.full_int_array: (1xi64) <- ()
        t315 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        t316 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t317 = paddle._C_ops.slice(t314, [0], t315, t316, [1], [0])
        del t314

        # pd_op.full: (xi64) <- ()
        t318 = paddle._C_ops.full([], float("1"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t319 = [t317, t318, t318, t318]
        del t317

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t320 = paddle._C_ops.stack(t319, 0)
        del t319

        # pd_op.full: (1xf32) <- ()
        t321 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t322 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t323 = paddle._C_ops.uniform(
            t320,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t320

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t324 = paddle._C_ops.add(t313, t323)
        del t323

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t325 = paddle._C_ops.floor(t324)
        del t324

        # pd_op.divide: (-1x144x56x56xf32) <- (-1x144x56x56xf32, xf32)
        t326 = paddle._C_ops.divide(t312, t313)

        # pd_op.multiply: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x1x1x1xf32)
        t327 = paddle._C_ops.multiply(t326, t325)

        # pd_op.add: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x144x56x56xf32)
        t328 = paddle._C_ops.add(t297, t327)

        # pd_op.split: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        t329 = paddle._C_ops.split(t328, t225, t226)
        del t225

        # builtin.split: (-1x36x56x56xf32, -1x108x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32])
        (
            t330,
            t331,
        ) = t329
        del t329

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x36x56x56xf32, 36x36x3x3xf32)
        t332 = paddle._C_ops.conv2d(
            t330, t19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t19

        # builtin.combine: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x36x56x56xf32, -1x108x56x56xf32)
        t333 = [t332, t331]

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32], 1xi32)
        t334 = paddle._C_ops.concat(t333, t226)
        del t333

        # pd_op.conv2d: (-1x288x56x56xf32) <- (-1x144x56x56xf32, 288x144x1x1xf32)
        t335 = paddle._C_ops.conv2d(
            t334, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t336, t337, t338, t339, t340, t341 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t335,
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

        # pd_op.relu: (-1x288x56x56xf32) <- (-1x288x56x56xf32)
        t342 = paddle._C_ops.relu(t336)
        del t336

        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x288x56x56xf32, 144x288x1x1xf32)
        t343 = paddle._C_ops.conv2d(
            t342, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.full: (xf32) <- ()
        t344 = paddle._C_ops.full(
            [],
            float("0.985185"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x144x56x56xf32)
        t345 = paddle._C_ops.shape64(t343)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t346 = paddle._C_ops.slice(t345, [0], t315, t316, [1], [0])
        del t345

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t347 = [t346, t318, t318, t318]
        del t346

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t348 = paddle._C_ops.stack(t347, 0)
        del t347

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t349 = paddle._C_ops.uniform(
            t348,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t348

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t350 = paddle._C_ops.add(t344, t349)
        del t349

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t351 = paddle._C_ops.floor(t350)
        del t350

        # pd_op.divide: (-1x144x56x56xf32) <- (-1x144x56x56xf32, xf32)
        t352 = paddle._C_ops.divide(t343, t344)

        # pd_op.multiply: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x1x1x1xf32)
        t353 = paddle._C_ops.multiply(t352, t351)

        # pd_op.add: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x144x56x56xf32)
        t354 = paddle._C_ops.add(t328, t353)

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x144x56x56xf32, 288x144x2x2xf32)
        t355 = paddle._C_ops.conv2d(
            t354, t26, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t26

        # pd_op.batch_norm_: (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        t356, t357, t358, t359, t360, t361 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t355,
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
        t362 = [72, 216]

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        t363 = paddle._C_ops.split(t356, t362, t226)

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            t364,
            t365,
        ) = t363
        del t363

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        t366 = paddle._C_ops.conv2d(
            t364, t31, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t31

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        t367 = [t366, t365]

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        t368 = paddle._C_ops.concat(t367, t226)
        del t367

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        t369 = paddle._C_ops.conv2d(
            t368, t32, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t32

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t370, t371, t372, t373, t374, t375 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t369,
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

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        t376 = paddle._C_ops.relu(t370)
        del t370

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        t377 = paddle._C_ops.conv2d(
            t376, t37, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t37

        # pd_op.full: (xf32) <- ()
        t378 = paddle._C_ops.full(
            [],
            float("0.977778"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x288x28x28xf32)
        t379 = paddle._C_ops.shape64(t377)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t380 = paddle._C_ops.slice(t379, [0], t315, t316, [1], [0])
        del t379

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t381 = [t380, t318, t318, t318]
        del t380

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t382 = paddle._C_ops.stack(t381, 0)
        del t381

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t383 = paddle._C_ops.uniform(
            t382,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t382

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t384 = paddle._C_ops.add(t378, t383)
        del t383

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t385 = paddle._C_ops.floor(t384)
        del t384

        # pd_op.divide: (-1x288x28x28xf32) <- (-1x288x28x28xf32, xf32)
        t386 = paddle._C_ops.divide(t377, t378)

        # pd_op.multiply: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x1x1x1xf32)
        t387 = paddle._C_ops.multiply(t386, t385)

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        t388 = paddle._C_ops.add(t356, t387)

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        t389 = paddle._C_ops.split(t388, t362, t226)

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            t390,
            t391,
        ) = t389
        del t389

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        t392 = paddle._C_ops.conv2d(
            t390, t38, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t38

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        t393 = [t392, t391]

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        t394 = paddle._C_ops.concat(t393, t226)
        del t393

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        t395 = paddle._C_ops.conv2d(
            t394, t39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t39

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t396, t397, t398, t399, t400, t401 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t395,
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

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        t402 = paddle._C_ops.relu(t396)
        del t396

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        t403 = paddle._C_ops.conv2d(
            t402, t44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t44

        # pd_op.full: (xf32) <- ()
        t404 = paddle._C_ops.full(
            [],
            float("0.97037"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x288x28x28xf32)
        t405 = paddle._C_ops.shape64(t403)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t406 = paddle._C_ops.slice(t405, [0], t315, t316, [1], [0])
        del t405

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t407 = [t406, t318, t318, t318]
        del t406

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t408 = paddle._C_ops.stack(t407, 0)
        del t407

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t409 = paddle._C_ops.uniform(
            t408,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t408

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t410 = paddle._C_ops.add(t404, t409)
        del t409

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t411 = paddle._C_ops.floor(t410)
        del t410

        # pd_op.divide: (-1x288x28x28xf32) <- (-1x288x28x28xf32, xf32)
        t412 = paddle._C_ops.divide(t403, t404)

        # pd_op.multiply: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x1x1x1xf32)
        t413 = paddle._C_ops.multiply(t412, t411)

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        t414 = paddle._C_ops.add(t388, t413)

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        t415 = paddle._C_ops.split(t414, t362, t226)

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            t416,
            t417,
        ) = t415
        del t415

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        t418 = paddle._C_ops.conv2d(
            t416, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        t419 = [t418, t417]

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        t420 = paddle._C_ops.concat(t419, t226)
        del t419

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        t421 = paddle._C_ops.conv2d(
            t420, t46, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t46

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t422, t423, t424, t425, t426, t427 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t421,
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

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        t428 = paddle._C_ops.relu(t422)
        del t422

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        t429 = paddle._C_ops.conv2d(
            t428, t51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t51

        # pd_op.full: (xf32) <- ()
        t430 = paddle._C_ops.full(
            [],
            float("0.962963"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x288x28x28xf32)
        t431 = paddle._C_ops.shape64(t429)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t432 = paddle._C_ops.slice(t431, [0], t315, t316, [1], [0])
        del t431

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t433 = [t432, t318, t318, t318]
        del t432

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t434 = paddle._C_ops.stack(t433, 0)
        del t433

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t435 = paddle._C_ops.uniform(
            t434,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t434

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t436 = paddle._C_ops.add(t430, t435)
        del t435

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t437 = paddle._C_ops.floor(t436)
        del t436

        # pd_op.divide: (-1x288x28x28xf32) <- (-1x288x28x28xf32, xf32)
        t438 = paddle._C_ops.divide(t429, t430)

        # pd_op.multiply: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x1x1x1xf32)
        t439 = paddle._C_ops.multiply(t438, t437)

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        t440 = paddle._C_ops.add(t414, t439)

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        t441 = paddle._C_ops.split(t440, t362, t226)
        del t362

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            t442,
            t443,
        ) = t441
        del t441

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        t444 = paddle._C_ops.conv2d(
            t442, t52, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t52

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        t445 = [t444, t443]

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        t446 = paddle._C_ops.concat(t445, t226)
        del t445

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        t447 = paddle._C_ops.conv2d(
            t446, t53, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t53

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t448, t449, t450, t451, t452, t453 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t447,
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

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        t454 = paddle._C_ops.relu(t448)
        del t448

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        t455 = paddle._C_ops.conv2d(
            t454, t58, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t58

        # pd_op.full: (xf32) <- ()
        t456 = paddle._C_ops.full(
            [],
            float("0.955556"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x288x28x28xf32)
        t457 = paddle._C_ops.shape64(t455)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t458 = paddle._C_ops.slice(t457, [0], t315, t316, [1], [0])
        del t457

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t459 = [t458, t318, t318, t318]
        del t458

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t460 = paddle._C_ops.stack(t459, 0)
        del t459

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t461 = paddle._C_ops.uniform(
            t460,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t460

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t462 = paddle._C_ops.add(t456, t461)
        del t461

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t463 = paddle._C_ops.floor(t462)
        del t462

        # pd_op.divide: (-1x288x28x28xf32) <- (-1x288x28x28xf32, xf32)
        t464 = paddle._C_ops.divide(t455, t456)

        # pd_op.multiply: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x1x1x1xf32)
        t465 = paddle._C_ops.multiply(t464, t463)

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        t466 = paddle._C_ops.add(t440, t465)

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x288x28x28xf32, 576x288x2x2xf32)
        t467 = paddle._C_ops.conv2d(
            t466, t59, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t59

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        t468, t469, t470, t471, t472, t473 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t467,
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
        t474 = [144, 432]

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t475 = paddle._C_ops.split(t468, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t476,
            t477,
        ) = t475
        del t475

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t478 = paddle._C_ops.conv2d(
            t476, t64, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t64

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t479 = [t478, t477]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t480 = paddle._C_ops.concat(t479, t226)
        del t479

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t481 = paddle._C_ops.conv2d(
            t480, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t482, t483, t484, t485, t486, t487 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t481,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t488 = paddle._C_ops.relu(t482)
        del t482

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t489 = paddle._C_ops.conv2d(
            t488, t70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.full: (xf32) <- ()
        t490 = paddle._C_ops.full(
            [],
            float("0.948148"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t491 = paddle._C_ops.shape64(t489)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t492 = paddle._C_ops.slice(t491, [0], t315, t316, [1], [0])
        del t491

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t493 = [t492, t318, t318, t318]
        del t492

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t494 = paddle._C_ops.stack(t493, 0)
        del t493

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t495 = paddle._C_ops.uniform(
            t494,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t494

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t496 = paddle._C_ops.add(t490, t495)
        del t495

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t497 = paddle._C_ops.floor(t496)
        del t496

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t498 = paddle._C_ops.divide(t489, t490)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t499 = paddle._C_ops.multiply(t498, t497)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t500 = paddle._C_ops.add(t468, t499)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t501 = paddle._C_ops.split(t500, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t502,
            t503,
        ) = t501
        del t501

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t504 = paddle._C_ops.conv2d(
            t502, t71, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t71

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t505 = [t504, t503]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t506 = paddle._C_ops.concat(t505, t226)
        del t505

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t507 = paddle._C_ops.conv2d(
            t506, t72, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t72

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t508, t509, t510, t511, t512, t513 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t507,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t514 = paddle._C_ops.relu(t508)
        del t508

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t515 = paddle._C_ops.conv2d(
            t514, t77, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t77

        # pd_op.full: (xf32) <- ()
        t516 = paddle._C_ops.full(
            [],
            float("0.940741"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t517 = paddle._C_ops.shape64(t515)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t518 = paddle._C_ops.slice(t517, [0], t315, t316, [1], [0])
        del t517

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t519 = [t518, t318, t318, t318]
        del t518

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t520 = paddle._C_ops.stack(t519, 0)
        del t519

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t521 = paddle._C_ops.uniform(
            t520,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t520

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t522 = paddle._C_ops.add(t516, t521)
        del t521

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t523 = paddle._C_ops.floor(t522)
        del t522

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t524 = paddle._C_ops.divide(t515, t516)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t525 = paddle._C_ops.multiply(t524, t523)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t526 = paddle._C_ops.add(t500, t525)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t527 = paddle._C_ops.split(t526, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t528,
            t529,
        ) = t527
        del t527

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t530 = paddle._C_ops.conv2d(
            t528, t78, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t78

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t531 = [t530, t529]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t532 = paddle._C_ops.concat(t531, t226)
        del t531

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t533 = paddle._C_ops.conv2d(
            t532, t79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t79

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t534, t535, t536, t537, t538, t539 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t533,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t540 = paddle._C_ops.relu(t534)
        del t534

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t541 = paddle._C_ops.conv2d(
            t540, t84, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t84

        # pd_op.full: (xf32) <- ()
        t542 = paddle._C_ops.full(
            [],
            float("0.933333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t543 = paddle._C_ops.shape64(t541)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t544 = paddle._C_ops.slice(t543, [0], t315, t316, [1], [0])
        del t543

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t545 = [t544, t318, t318, t318]
        del t544

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t546 = paddle._C_ops.stack(t545, 0)
        del t545

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t547 = paddle._C_ops.uniform(
            t546,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t546

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t548 = paddle._C_ops.add(t542, t547)
        del t547

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t549 = paddle._C_ops.floor(t548)
        del t548

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t550 = paddle._C_ops.divide(t541, t542)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t551 = paddle._C_ops.multiply(t550, t549)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t552 = paddle._C_ops.add(t526, t551)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t553 = paddle._C_ops.split(t552, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t554,
            t555,
        ) = t553
        del t553

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t556 = paddle._C_ops.conv2d(
            t554, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t557 = [t556, t555]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t558 = paddle._C_ops.concat(t557, t226)
        del t557

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t559 = paddle._C_ops.conv2d(
            t558, t86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t86

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t560, t561, t562, t563, t564, t565 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t559,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t566 = paddle._C_ops.relu(t560)
        del t560

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t567 = paddle._C_ops.conv2d(
            t566, t91, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t91

        # pd_op.full: (xf32) <- ()
        t568 = paddle._C_ops.full(
            [],
            float("0.925926"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t569 = paddle._C_ops.shape64(t567)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t570 = paddle._C_ops.slice(t569, [0], t315, t316, [1], [0])
        del t569

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t571 = [t570, t318, t318, t318]
        del t570

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t572 = paddle._C_ops.stack(t571, 0)
        del t571

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t573 = paddle._C_ops.uniform(
            t572,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t572

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t574 = paddle._C_ops.add(t568, t573)
        del t573

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t575 = paddle._C_ops.floor(t574)
        del t574

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t576 = paddle._C_ops.divide(t567, t568)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t577 = paddle._C_ops.multiply(t576, t575)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t578 = paddle._C_ops.add(t552, t577)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t579 = paddle._C_ops.split(t578, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t580,
            t581,
        ) = t579
        del t579

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t582 = paddle._C_ops.conv2d(
            t580, t92, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t92

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t583 = [t582, t581]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t584 = paddle._C_ops.concat(t583, t226)
        del t583

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t585 = paddle._C_ops.conv2d(
            t584, t93, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t93

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t586, t587, t588, t589, t590, t591 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t585,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t592 = paddle._C_ops.relu(t586)
        del t586

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t593 = paddle._C_ops.conv2d(
            t592, t98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t98

        # pd_op.full: (xf32) <- ()
        t594 = paddle._C_ops.full(
            [],
            float("0.918519"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t595 = paddle._C_ops.shape64(t593)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t596 = paddle._C_ops.slice(t595, [0], t315, t316, [1], [0])
        del t595

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t597 = [t596, t318, t318, t318]
        del t596

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t598 = paddle._C_ops.stack(t597, 0)
        del t597

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t599 = paddle._C_ops.uniform(
            t598,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t598

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t600 = paddle._C_ops.add(t594, t599)
        del t599

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t601 = paddle._C_ops.floor(t600)
        del t600

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t602 = paddle._C_ops.divide(t593, t594)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t603 = paddle._C_ops.multiply(t602, t601)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t604 = paddle._C_ops.add(t578, t603)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t605 = paddle._C_ops.split(t604, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t606,
            t607,
        ) = t605
        del t605

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t608 = paddle._C_ops.conv2d(
            t606, t99, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t99

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t609 = [t608, t607]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t610 = paddle._C_ops.concat(t609, t226)
        del t609

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t611 = paddle._C_ops.conv2d(
            t610, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t612, t613, t614, t615, t616, t617 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t611,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t618 = paddle._C_ops.relu(t612)
        del t612

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t619 = paddle._C_ops.conv2d(
            t618, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.full: (xf32) <- ()
        t620 = paddle._C_ops.full(
            [],
            float("0.911111"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t621 = paddle._C_ops.shape64(t619)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t622 = paddle._C_ops.slice(t621, [0], t315, t316, [1], [0])
        del t621

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t623 = [t622, t318, t318, t318]
        del t622

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t624 = paddle._C_ops.stack(t623, 0)
        del t623

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t625 = paddle._C_ops.uniform(
            t624,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t624

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t626 = paddle._C_ops.add(t620, t625)
        del t625

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t627 = paddle._C_ops.floor(t626)
        del t626

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t628 = paddle._C_ops.divide(t619, t620)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t629 = paddle._C_ops.multiply(t628, t627)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t630 = paddle._C_ops.add(t604, t629)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t631 = paddle._C_ops.split(t630, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t632,
            t633,
        ) = t631
        del t631

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t634 = paddle._C_ops.conv2d(
            t632, t106, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t106

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t635 = [t634, t633]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t636 = paddle._C_ops.concat(t635, t226)
        del t635

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t637 = paddle._C_ops.conv2d(
            t636, t107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t107

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t638, t639, t640, t641, t642, t643 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t637,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t644 = paddle._C_ops.relu(t638)
        del t638

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t645 = paddle._C_ops.conv2d(
            t644, t112, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t112

        # pd_op.full: (xf32) <- ()
        t646 = paddle._C_ops.full(
            [],
            float("0.903704"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t647 = paddle._C_ops.shape64(t645)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t648 = paddle._C_ops.slice(t647, [0], t315, t316, [1], [0])
        del t647

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t649 = [t648, t318, t318, t318]
        del t648

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t650 = paddle._C_ops.stack(t649, 0)
        del t649

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t651 = paddle._C_ops.uniform(
            t650,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t650

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t652 = paddle._C_ops.add(t646, t651)
        del t651

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t653 = paddle._C_ops.floor(t652)
        del t652

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t654 = paddle._C_ops.divide(t645, t646)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t655 = paddle._C_ops.multiply(t654, t653)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t656 = paddle._C_ops.add(t630, t655)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t657 = paddle._C_ops.split(t656, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t658,
            t659,
        ) = t657
        del t657

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t660 = paddle._C_ops.conv2d(
            t658, t113, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t113

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t661 = [t660, t659]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t662 = paddle._C_ops.concat(t661, t226)
        del t661

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t663 = paddle._C_ops.conv2d(
            t662, t114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t114

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t664, t665, t666, t667, t668, t669 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t663,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t670 = paddle._C_ops.relu(t664)
        del t664

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t671 = paddle._C_ops.conv2d(
            t670, t119, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t119

        # pd_op.full: (xf32) <- ()
        t672 = paddle._C_ops.full(
            [],
            float("0.896296"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t673 = paddle._C_ops.shape64(t671)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t674 = paddle._C_ops.slice(t673, [0], t315, t316, [1], [0])
        del t673

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t675 = [t674, t318, t318, t318]
        del t674

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t676 = paddle._C_ops.stack(t675, 0)
        del t675

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t677 = paddle._C_ops.uniform(
            t676,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t676

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t678 = paddle._C_ops.add(t672, t677)
        del t677

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t679 = paddle._C_ops.floor(t678)
        del t678

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t680 = paddle._C_ops.divide(t671, t672)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t681 = paddle._C_ops.multiply(t680, t679)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t682 = paddle._C_ops.add(t656, t681)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t683 = paddle._C_ops.split(t682, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t684,
            t685,
        ) = t683
        del t683

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t686 = paddle._C_ops.conv2d(
            t684, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t687 = [t686, t685]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t688 = paddle._C_ops.concat(t687, t226)
        del t687

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t689 = paddle._C_ops.conv2d(
            t688, t121, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t121

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t690, t691, t692, t693, t694, t695 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t689,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t696 = paddle._C_ops.relu(t690)
        del t690

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t697 = paddle._C_ops.conv2d(
            t696, t126, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t126

        # pd_op.full: (xf32) <- ()
        t698 = paddle._C_ops.full(
            [],
            float("0.888889"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t699 = paddle._C_ops.shape64(t697)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t700 = paddle._C_ops.slice(t699, [0], t315, t316, [1], [0])
        del t699

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t701 = [t700, t318, t318, t318]
        del t700

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t702 = paddle._C_ops.stack(t701, 0)
        del t701

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t703 = paddle._C_ops.uniform(
            t702,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t702

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t704 = paddle._C_ops.add(t698, t703)
        del t703

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t705 = paddle._C_ops.floor(t704)
        del t704

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t706 = paddle._C_ops.divide(t697, t698)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t707 = paddle._C_ops.multiply(t706, t705)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t708 = paddle._C_ops.add(t682, t707)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t709 = paddle._C_ops.split(t708, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t710,
            t711,
        ) = t709
        del t709

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t712 = paddle._C_ops.conv2d(
            t710, t127, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t127

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t713 = [t712, t711]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t714 = paddle._C_ops.concat(t713, t226)
        del t713

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t715 = paddle._C_ops.conv2d(
            t714, t128, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t128

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t716, t717, t718, t719, t720, t721 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t715,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t722 = paddle._C_ops.relu(t716)
        del t716

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t723 = paddle._C_ops.conv2d(
            t722, t133, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t133

        # pd_op.full: (xf32) <- ()
        t724 = paddle._C_ops.full(
            [],
            float("0.881481"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t725 = paddle._C_ops.shape64(t723)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t726 = paddle._C_ops.slice(t725, [0], t315, t316, [1], [0])
        del t725

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t727 = [t726, t318, t318, t318]
        del t726

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t728 = paddle._C_ops.stack(t727, 0)
        del t727

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t729 = paddle._C_ops.uniform(
            t728,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t728

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t730 = paddle._C_ops.add(t724, t729)
        del t729

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t731 = paddle._C_ops.floor(t730)
        del t730

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t732 = paddle._C_ops.divide(t723, t724)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t733 = paddle._C_ops.multiply(t732, t731)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t734 = paddle._C_ops.add(t708, t733)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t735 = paddle._C_ops.split(t734, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t736,
            t737,
        ) = t735
        del t735

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t738 = paddle._C_ops.conv2d(
            t736, t134, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t134

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t739 = [t738, t737]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t740 = paddle._C_ops.concat(t739, t226)
        del t739

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t741 = paddle._C_ops.conv2d(
            t740, t135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t742, t743, t744, t745, t746, t747 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t741,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t748 = paddle._C_ops.relu(t742)
        del t742

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t749 = paddle._C_ops.conv2d(
            t748, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.full: (xf32) <- ()
        t750 = paddle._C_ops.full(
            [],
            float("0.874074"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t751 = paddle._C_ops.shape64(t749)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t752 = paddle._C_ops.slice(t751, [0], t315, t316, [1], [0])
        del t751

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t753 = [t752, t318, t318, t318]
        del t752

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t754 = paddle._C_ops.stack(t753, 0)
        del t753

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t755 = paddle._C_ops.uniform(
            t754,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t754

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t756 = paddle._C_ops.add(t750, t755)
        del t755

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t757 = paddle._C_ops.floor(t756)
        del t756

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t758 = paddle._C_ops.divide(t749, t750)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t759 = paddle._C_ops.multiply(t758, t757)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t760 = paddle._C_ops.add(t734, t759)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t761 = paddle._C_ops.split(t760, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t762,
            t763,
        ) = t761
        del t761

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t764 = paddle._C_ops.conv2d(
            t762, t141, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t141

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t765 = [t764, t763]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t766 = paddle._C_ops.concat(t765, t226)
        del t765

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t767 = paddle._C_ops.conv2d(
            t766, t142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t142

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t768, t769, t770, t771, t772, t773 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t767,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t774 = paddle._C_ops.relu(t768)
        del t768

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t775 = paddle._C_ops.conv2d(
            t774, t147, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t147

        # pd_op.full: (xf32) <- ()
        t776 = paddle._C_ops.full(
            [],
            float("0.866667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t777 = paddle._C_ops.shape64(t775)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t778 = paddle._C_ops.slice(t777, [0], t315, t316, [1], [0])
        del t777

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t779 = [t778, t318, t318, t318]
        del t778

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t780 = paddle._C_ops.stack(t779, 0)
        del t779

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t781 = paddle._C_ops.uniform(
            t780,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t780

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t782 = paddle._C_ops.add(t776, t781)
        del t781

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t783 = paddle._C_ops.floor(t782)
        del t782

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t784 = paddle._C_ops.divide(t775, t776)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t785 = paddle._C_ops.multiply(t784, t783)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t786 = paddle._C_ops.add(t760, t785)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t787 = paddle._C_ops.split(t786, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t788,
            t789,
        ) = t787
        del t787

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t790 = paddle._C_ops.conv2d(
            t788, t148, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t148

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t791 = [t790, t789]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t792 = paddle._C_ops.concat(t791, t226)
        del t791

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t793 = paddle._C_ops.conv2d(
            t792, t149, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t149

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t794, t795, t796, t797, t798, t799 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t793,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t800 = paddle._C_ops.relu(t794)
        del t794

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t801 = paddle._C_ops.conv2d(
            t800, t154, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t154

        # pd_op.full: (xf32) <- ()
        t802 = paddle._C_ops.full(
            [],
            float("0.859259"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t803 = paddle._C_ops.shape64(t801)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t804 = paddle._C_ops.slice(t803, [0], t315, t316, [1], [0])
        del t803

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t805 = [t804, t318, t318, t318]
        del t804

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t806 = paddle._C_ops.stack(t805, 0)
        del t805

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t807 = paddle._C_ops.uniform(
            t806,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t806

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t808 = paddle._C_ops.add(t802, t807)
        del t807

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t809 = paddle._C_ops.floor(t808)
        del t808

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t810 = paddle._C_ops.divide(t801, t802)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t811 = paddle._C_ops.multiply(t810, t809)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t812 = paddle._C_ops.add(t786, t811)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t813 = paddle._C_ops.split(t812, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t814,
            t815,
        ) = t813
        del t813

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t816 = paddle._C_ops.conv2d(
            t814, t155, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t817 = [t816, t815]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t818 = paddle._C_ops.concat(t817, t226)
        del t817

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t819 = paddle._C_ops.conv2d(
            t818, t156, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t820, t821, t822, t823, t824, t825 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t819,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t826 = paddle._C_ops.relu(t820)
        del t820

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t827 = paddle._C_ops.conv2d(
            t826, t161, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t161

        # pd_op.full: (xf32) <- ()
        t828 = paddle._C_ops.full(
            [],
            float("0.851852"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t829 = paddle._C_ops.shape64(t827)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t830 = paddle._C_ops.slice(t829, [0], t315, t316, [1], [0])
        del t829

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t831 = [t830, t318, t318, t318]
        del t830

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t832 = paddle._C_ops.stack(t831, 0)
        del t831

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t833 = paddle._C_ops.uniform(
            t832,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t832

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t834 = paddle._C_ops.add(t828, t833)
        del t833

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t835 = paddle._C_ops.floor(t834)
        del t834

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t836 = paddle._C_ops.divide(t827, t828)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t837 = paddle._C_ops.multiply(t836, t835)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t838 = paddle._C_ops.add(t812, t837)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t839 = paddle._C_ops.split(t838, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t840,
            t841,
        ) = t839
        del t839

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t842 = paddle._C_ops.conv2d(
            t840, t162, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t162

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t843 = [t842, t841]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t844 = paddle._C_ops.concat(t843, t226)
        del t843

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t845 = paddle._C_ops.conv2d(
            t844, t163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t163

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t846, t847, t848, t849, t850, t851 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t845,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t852 = paddle._C_ops.relu(t846)
        del t846

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t853 = paddle._C_ops.conv2d(
            t852, t168, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t168

        # pd_op.full: (xf32) <- ()
        t854 = paddle._C_ops.full(
            [],
            float("0.844444"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t855 = paddle._C_ops.shape64(t853)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t856 = paddle._C_ops.slice(t855, [0], t315, t316, [1], [0])
        del t855

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t857 = [t856, t318, t318, t318]
        del t856

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t858 = paddle._C_ops.stack(t857, 0)
        del t857

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t859 = paddle._C_ops.uniform(
            t858,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t858

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t860 = paddle._C_ops.add(t854, t859)
        del t859

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t861 = paddle._C_ops.floor(t860)
        del t860

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t862 = paddle._C_ops.divide(t853, t854)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t863 = paddle._C_ops.multiply(t862, t861)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t864 = paddle._C_ops.add(t838, t863)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t865 = paddle._C_ops.split(t864, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t866,
            t867,
        ) = t865
        del t865

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t868 = paddle._C_ops.conv2d(
            t866, t169, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t169

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t869 = [t868, t867]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t870 = paddle._C_ops.concat(t869, t226)
        del t869

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t871 = paddle._C_ops.conv2d(
            t870, t170, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t872, t873, t874, t875, t876, t877 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t871,
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

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t878 = paddle._C_ops.relu(t872)
        del t872

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t879 = paddle._C_ops.conv2d(
            t878, t175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.full: (xf32) <- ()
        t880 = paddle._C_ops.full(
            [],
            float("0.837037"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t881 = paddle._C_ops.shape64(t879)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t882 = paddle._C_ops.slice(t881, [0], t315, t316, [1], [0])
        del t881

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t883 = [t882, t318, t318, t318]
        del t882

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t884 = paddle._C_ops.stack(t883, 0)
        del t883

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t885 = paddle._C_ops.uniform(
            t884,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t884

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t886 = paddle._C_ops.add(t880, t885)
        del t885

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t887 = paddle._C_ops.floor(t886)
        del t886

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t888 = paddle._C_ops.divide(t879, t880)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t889 = paddle._C_ops.multiply(t888, t887)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t890 = paddle._C_ops.add(t864, t889)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t891 = paddle._C_ops.split(t890, t474, t226)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t892,
            t893,
        ) = t891
        del t891

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t894 = paddle._C_ops.conv2d(
            t892, t176, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t176

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t895 = [t894, t893]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t896 = paddle._C_ops.concat(t895, t226)
        del t895

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t897 = paddle._C_ops.conv2d(
            t896, t177, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t177

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t898, t899, t900, t901, t902, t903 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t897,
                t178,
                t179,
                t180,
                t181,
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
        del t181, t180, t179, t178

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t904 = paddle._C_ops.relu(t898)
        del t898

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t905 = paddle._C_ops.conv2d(
            t904, t182, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t182

        # pd_op.full: (xf32) <- ()
        t906 = paddle._C_ops.full(
            [],
            float("0.82963"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t907 = paddle._C_ops.shape64(t905)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t908 = paddle._C_ops.slice(t907, [0], t315, t316, [1], [0])
        del t907

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t909 = [t908, t318, t318, t318]
        del t908

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t910 = paddle._C_ops.stack(t909, 0)
        del t909

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t911 = paddle._C_ops.uniform(
            t910,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t910

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t912 = paddle._C_ops.add(t906, t911)
        del t911

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t913 = paddle._C_ops.floor(t912)
        del t912

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t914 = paddle._C_ops.divide(t905, t906)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t915 = paddle._C_ops.multiply(t914, t913)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t916 = paddle._C_ops.add(t890, t915)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        t917 = paddle._C_ops.split(t916, t474, t226)
        del t474

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            t918,
            t919,
        ) = t917
        del t917

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        t920 = paddle._C_ops.conv2d(
            t918, t183, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t183

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        t921 = [t920, t919]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        t922 = paddle._C_ops.concat(t921, t226)
        del t921

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        t923 = paddle._C_ops.conv2d(
            t922, t184, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t184

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t924, t925, t926, t927, t928, t929 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t923,
                t185,
                t186,
                t187,
                t188,
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
        del t188, t187, t186, t185

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        t930 = paddle._C_ops.relu(t924)
        del t924

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        t931 = paddle._C_ops.conv2d(
            t930, t189, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t189

        # pd_op.full: (xf32) <- ()
        t932 = paddle._C_ops.full(
            [],
            float("0.822222"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        t933 = paddle._C_ops.shape64(t931)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t934 = paddle._C_ops.slice(t933, [0], t315, t316, [1], [0])
        del t933

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t935 = [t934, t318, t318, t318]
        del t934

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t936 = paddle._C_ops.stack(t935, 0)
        del t935

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t937 = paddle._C_ops.uniform(
            t936,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t936

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t938 = paddle._C_ops.add(t932, t937)
        del t937

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t939 = paddle._C_ops.floor(t938)
        del t938

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        t940 = paddle._C_ops.divide(t931, t932)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        t941 = paddle._C_ops.multiply(t940, t939)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        t942 = paddle._C_ops.add(t916, t941)

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x576x14x14xf32, 1152x576x2x2xf32)
        t943 = paddle._C_ops.conv2d(
            t942, t190, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (-1x1152x7x7xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x7x7xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        t944, t945, t946, t947, t948, t949 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t943,
                t191,
                t192,
                t193,
                t194,
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
        del t194, t193, t192, t191

        # pd_op.full_int_array: (2xi64) <- ()
        t950 = [288, 864]

        # pd_op.split: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x1152x7x7xf32, 2xi64, 1xi32)
        t951 = paddle._C_ops.split(t944, t950, t226)

        # builtin.split: (-1x288x7x7xf32, -1x864x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32])
        (
            t952,
            t953,
        ) = t951
        del t951

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t954 = paddle._C_ops.conv2d(
            t952, t195, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195

        # builtin.combine: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x288x7x7xf32, -1x864x7x7xf32)
        t955 = [t954, t953]

        # pd_op.concat: (-1x1152x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32], 1xi32)
        t956 = paddle._C_ops.concat(t955, t226)
        del t955

        # pd_op.conv2d: (-1x2304x7x7xf32) <- (-1x1152x7x7xf32, 2304x1152x1x1xf32)
        t957 = paddle._C_ops.conv2d(
            t956, t196, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t196

        # pd_op.batch_norm_: (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32, -1xui8) <- (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32)
        t958, t959, t960, t961, t962, t963 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t957,
                t197,
                t198,
                t199,
                t200,
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
        del t200, t199, t198, t197

        # pd_op.relu: (-1x2304x7x7xf32) <- (-1x2304x7x7xf32)
        t964 = paddle._C_ops.relu(t958)
        del t958

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x2304x7x7xf32, 1152x2304x1x1xf32)
        t965 = paddle._C_ops.conv2d(
            t964, t201, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t201

        # pd_op.full: (xf32) <- ()
        t966 = paddle._C_ops.full(
            [],
            float("0.814815"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x1152x7x7xf32)
        t967 = paddle._C_ops.shape64(t965)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t968 = paddle._C_ops.slice(t967, [0], t315, t316, [1], [0])
        del t967

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t969 = [t968, t318, t318, t318]
        del t968

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t970 = paddle._C_ops.stack(t969, 0)
        del t969

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t971 = paddle._C_ops.uniform(
            t970,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t970

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t972 = paddle._C_ops.add(t966, t971)
        del t971

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t973 = paddle._C_ops.floor(t972)
        del t972

        # pd_op.divide: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, xf32)
        t974 = paddle._C_ops.divide(t965, t966)

        # pd_op.multiply: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1x1x1xf32)
        t975 = paddle._C_ops.multiply(t974, t973)

        # pd_op.add: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1152x7x7xf32)
        t976 = paddle._C_ops.add(t944, t975)

        # pd_op.split: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x1152x7x7xf32, 2xi64, 1xi32)
        t977 = paddle._C_ops.split(t976, t950, t226)

        # builtin.split: (-1x288x7x7xf32, -1x864x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32])
        (
            t978,
            t979,
        ) = t977
        del t977

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t980 = paddle._C_ops.conv2d(
            t978, t202, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t202

        # builtin.combine: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x288x7x7xf32, -1x864x7x7xf32)
        t981 = [t980, t979]

        # pd_op.concat: (-1x1152x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32], 1xi32)
        t982 = paddle._C_ops.concat(t981, t226)
        del t981

        # pd_op.conv2d: (-1x2304x7x7xf32) <- (-1x1152x7x7xf32, 2304x1152x1x1xf32)
        t983 = paddle._C_ops.conv2d(
            t982, t203, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t203

        # pd_op.batch_norm_: (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32, -1xui8) <- (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32)
        t984, t985, t986, t987, t988, t989 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t983,
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

        # pd_op.relu: (-1x2304x7x7xf32) <- (-1x2304x7x7xf32)
        t990 = paddle._C_ops.relu(t984)
        del t984

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x2304x7x7xf32, 1152x2304x1x1xf32)
        t991 = paddle._C_ops.conv2d(
            t990, t208, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t208

        # pd_op.full: (xf32) <- ()
        t992 = paddle._C_ops.full(
            [],
            float("0.807407"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x1152x7x7xf32)
        t993 = paddle._C_ops.shape64(t991)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t994 = paddle._C_ops.slice(t993, [0], t315, t316, [1], [0])
        del t993

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t995 = [t994, t318, t318, t318]
        del t994

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t996 = paddle._C_ops.stack(t995, 0)
        del t995

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t997 = paddle._C_ops.uniform(
            t996,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t996

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t998 = paddle._C_ops.add(t992, t997)
        del t997

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t999 = paddle._C_ops.floor(t998)
        del t998

        # pd_op.divide: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, xf32)
        t1000 = paddle._C_ops.divide(t991, t992)

        # pd_op.multiply: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1x1x1xf32)
        t1001 = paddle._C_ops.multiply(t1000, t999)

        # pd_op.add: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1152x7x7xf32)
        t1002 = paddle._C_ops.add(t976, t1001)

        # pd_op.split: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x1152x7x7xf32, 2xi64, 1xi32)
        t1003 = paddle._C_ops.split(t1002, t950, t226)
        del t950

        # builtin.split: (-1x288x7x7xf32, -1x864x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32])
        (
            t1004,
            t1005,
        ) = t1003
        del t1003

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        t1006 = paddle._C_ops.conv2d(
            t1004, t209, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t209

        # builtin.combine: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x288x7x7xf32, -1x864x7x7xf32)
        t1007 = [t1006, t1005]

        # pd_op.concat: (-1x1152x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32], 1xi32)
        t1008 = paddle._C_ops.concat(t1007, t226)
        del t1007

        # pd_op.conv2d: (-1x2304x7x7xf32) <- (-1x1152x7x7xf32, 2304x1152x1x1xf32)
        t1009 = paddle._C_ops.conv2d(
            t1008, t210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32, -1xui8) <- (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32)
        t1010, t1011, t1012, t1013, t1014, t1015 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1009,
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

        # pd_op.relu: (-1x2304x7x7xf32) <- (-1x2304x7x7xf32)
        t1016 = paddle._C_ops.relu(t1010)
        del t1010

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x2304x7x7xf32, 1152x2304x1x1xf32)
        t1017 = paddle._C_ops.conv2d(
            t1016, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.full: (xf32) <- ()
        t1018 = paddle._C_ops.full(
            [], float("0.8"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.shape64: (4xi64) <- (-1x1152x7x7xf32)
        t1019 = paddle._C_ops.shape64(t1017)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1020 = paddle._C_ops.slice(t1019, [0], t315, t316, [1], [0])
        del t315, t316, t1019

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1021 = [t1020, t318, t318, t318]
        del t318, t1020

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1022 = paddle._C_ops.stack(t1021, 0)
        del t1021

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t1023 = paddle._C_ops.uniform(
            t1022,
            paddle.float32,
            t321,
            t322,
            0,
            paddle.framework._current_expected_place(),
        )
        del t321, t322, t1022

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t1024 = paddle._C_ops.add(t1018, t1023)
        del t1023

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t1025 = paddle._C_ops.floor(t1024)
        del t1024

        # pd_op.divide: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, xf32)
        t1026 = paddle._C_ops.divide(t1017, t1018)

        # pd_op.multiply: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1x1x1xf32)
        t1027 = paddle._C_ops.multiply(t1026, t1025)

        # pd_op.add: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1152x7x7xf32)
        t1028 = paddle._C_ops.add(t1002, t1027)

        # pd_op.full_int_array: (2xi64) <- ()
        t1029 = [1, 1]

        # pd_op.pool2d: (-1x1152x1x1xf32) <- (-1x1152x7x7xf32, 2xi64)
        t1030 = paddle._C_ops.pool2d(
            t1028,
            t1029,
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

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x1152x1x1xf32, 1280x1152x1x1xf32)
        t1031 = paddle._C_ops.conv2d(
            t1030, t216, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t216

        # pd_op.relu: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        t1032 = paddle._C_ops.relu(t1031)
        del t1031

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        t1033 = paddle._C_ops.flatten(t1032, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        t1034 = paddle._C_ops.matmul(t1033, t217, False, False)
        del t217

        return (
            t218,
            t219,
            t220,
            t221,
            t222,
            t223,
            t224,
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
            t264,
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
            t283,
            t284,
            t285,
            t287,
            t288,
            t290,
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
            t303,
            t304,
            t306,
            t307,
            t308,
            t309,
            t310,
            t311,
            t312,
            t313,
            t325,
            t326,
            t327,
            t328,
            t330,
            t331,
            t332,
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
            t364,
            t365,
            t366,
            t368,
            t369,
            t371,
            t372,
            t373,
            t374,
            t375,
            t376,
            t377,
            t378,
            t385,
            t386,
            t387,
            t388,
            t390,
            t391,
            t392,
            t394,
            t395,
            t397,
            t398,
            t399,
            t400,
            t401,
            t402,
            t403,
            t404,
            t411,
            t412,
            t413,
            t414,
            t416,
            t417,
            t418,
            t420,
            t421,
            t423,
            t424,
            t425,
            t426,
            t427,
            t428,
            t429,
            t430,
            t437,
            t438,
            t439,
            t440,
            t442,
            t443,
            t444,
            t446,
            t447,
            t449,
            t450,
            t451,
            t452,
            t453,
            t454,
            t455,
            t456,
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
            t476,
            t477,
            t478,
            t480,
            t481,
            t483,
            t484,
            t485,
            t486,
            t487,
            t488,
            t489,
            t490,
            t497,
            t498,
            t499,
            t500,
            t502,
            t503,
            t504,
            t506,
            t507,
            t509,
            t510,
            t511,
            t512,
            t513,
            t514,
            t515,
            t516,
            t523,
            t524,
            t525,
            t526,
            t528,
            t529,
            t530,
            t532,
            t533,
            t535,
            t536,
            t537,
            t538,
            t539,
            t540,
            t541,
            t542,
            t549,
            t550,
            t551,
            t552,
            t554,
            t555,
            t556,
            t558,
            t559,
            t561,
            t562,
            t563,
            t564,
            t565,
            t566,
            t567,
            t568,
            t575,
            t576,
            t577,
            t578,
            t580,
            t581,
            t582,
            t584,
            t585,
            t587,
            t588,
            t589,
            t590,
            t591,
            t592,
            t593,
            t594,
            t601,
            t602,
            t603,
            t604,
            t606,
            t607,
            t608,
            t610,
            t611,
            t613,
            t614,
            t615,
            t616,
            t617,
            t618,
            t619,
            t620,
            t627,
            t628,
            t629,
            t630,
            t632,
            t633,
            t634,
            t636,
            t637,
            t639,
            t640,
            t641,
            t642,
            t643,
            t644,
            t645,
            t646,
            t653,
            t654,
            t655,
            t656,
            t658,
            t659,
            t660,
            t662,
            t663,
            t665,
            t666,
            t667,
            t668,
            t669,
            t670,
            t671,
            t672,
            t679,
            t680,
            t681,
            t682,
            t684,
            t685,
            t686,
            t688,
            t689,
            t691,
            t692,
            t693,
            t694,
            t695,
            t696,
            t697,
            t698,
            t705,
            t706,
            t707,
            t708,
            t710,
            t711,
            t712,
            t714,
            t715,
            t717,
            t718,
            t719,
            t720,
            t721,
            t722,
            t723,
            t724,
            t731,
            t732,
            t733,
            t734,
            t736,
            t737,
            t738,
            t740,
            t741,
            t743,
            t744,
            t745,
            t746,
            t747,
            t748,
            t749,
            t750,
            t757,
            t758,
            t759,
            t760,
            t762,
            t763,
            t764,
            t766,
            t767,
            t769,
            t770,
            t771,
            t772,
            t773,
            t774,
            t775,
            t776,
            t783,
            t784,
            t785,
            t786,
            t788,
            t789,
            t790,
            t792,
            t793,
            t795,
            t796,
            t797,
            t798,
            t799,
            t800,
            t801,
            t802,
            t809,
            t810,
            t811,
            t812,
            t814,
            t815,
            t816,
            t818,
            t819,
            t821,
            t822,
            t823,
            t824,
            t825,
            t826,
            t827,
            t828,
            t835,
            t836,
            t837,
            t838,
            t840,
            t841,
            t842,
            t844,
            t845,
            t847,
            t848,
            t849,
            t850,
            t851,
            t852,
            t853,
            t854,
            t861,
            t862,
            t863,
            t864,
            t866,
            t867,
            t868,
            t870,
            t871,
            t873,
            t874,
            t875,
            t876,
            t877,
            t878,
            t879,
            t880,
            t887,
            t888,
            t889,
            t890,
            t892,
            t893,
            t894,
            t896,
            t897,
            t899,
            t900,
            t901,
            t902,
            t903,
            t904,
            t905,
            t906,
            t913,
            t914,
            t915,
            t916,
            t918,
            t919,
            t920,
            t922,
            t923,
            t925,
            t926,
            t927,
            t928,
            t929,
            t930,
            t931,
            t932,
            t939,
            t940,
            t941,
            t942,
            t943,
            t944,
            t945,
            t946,
            t947,
            t948,
            t949,
            t952,
            t953,
            t954,
            t956,
            t957,
            t959,
            t960,
            t961,
            t962,
            t963,
            t964,
            t965,
            t966,
            t973,
            t974,
            t975,
            t976,
            t978,
            t979,
            t980,
            t982,
            t983,
            t985,
            t986,
            t987,
            t988,
            t989,
            t990,
            t991,
            t992,
            t999,
            t1000,
            t1001,
            t1002,
            t1004,
            t1005,
            t1006,
            t1008,
            t1009,
            t1011,
            t1012,
            t1013,
            t1014,
            t1015,
            t1016,
            t1017,
            t1018,
            t1025,
            t1026,
            t1027,
            t1028,
            t1029,
            t1030,
            t1032,
            t1033,
            t1034,
        )
