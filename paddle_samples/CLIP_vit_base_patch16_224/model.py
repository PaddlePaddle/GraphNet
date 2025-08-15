import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input0: paddle.Tensor,
        t0: paddle.Tensor,
        input1: paddle.Tensor,
        input2: paddle.Tensor,
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
    ):
        # pd_op.conv2d: (32x768x14x14xf32) <- (32x3x224x224xf32, 768x3x16x16xf32)
        t150 = paddle._C_ops.conv2d(
            input0, t0, [16, 16], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.flatten: (32x768x196xf32) <- (32x768x14x14xf32)
        t151 = paddle._C_ops.flatten(t150, 2, 3)

        # pd_op.transpose: (32x196x768xf32) <- (32x768x196xf32)
        t152 = paddle._C_ops.transpose(t151, [0, 2, 1])
        del t151

        # pd_op.full_int_array: (3xi64) <- ()
        t153 = [32, -1, -1]

        # pd_op.expand: (32x1x768xf32) <- (1x1x768xf32, 3xi64)
        t154 = paddle._C_ops.expand(input1, t153)
        del input1

        # pd_op.full: (1xi32) <- ()
        t155 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([32x1x768xf32, 32x196x768xf32]) <- (32x1x768xf32, 32x196x768xf32)
        t156 = [t154, t152]

        # pd_op.concat: (32x197x768xf32) <- ([32x1x768xf32, 32x196x768xf32], 1xi32)
        t157 = paddle._C_ops.concat(t156, t155)
        del t156

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 1x197x768xf32)
        t158 = paddle._C_ops.add(t157, input2)
        del input2

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t159, t160, t161 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t158, t1, t2, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t2, t1

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t162, t163, t164 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t159, t3, t4, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t4, t3

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t165 = paddle._C_ops.matmul(t162, t5, False, False)
        del t5

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t166 = paddle._C_ops.add(t165, t6)
        del t6

        # pd_op.full_int_array: (5xi64) <- ()
        t167 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t168 = paddle._C_ops.reshape(t166, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t169 = paddle._C_ops.transpose(t168, [2, 0, 3, 1, 4])
        del t168

        # pd_op.full_int_array: (1xi64) <- ()
        t170 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        t171 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t172 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t173 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t174 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t175 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t176 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t177 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t178 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t179 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t180 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t181 = t170

        # pd_op.assign: (1xi64) <- (1xi64)
        t182 = t170

        # pd_op.full_int_array: (1xi64) <- ()
        t183 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t184 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t185 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t186 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t187 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t188 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t189 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t190 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t191 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t192 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t193 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t194 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t195 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t196 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t197 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t198 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t199 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t200 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t201 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t202 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t203 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t204 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t205 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t206 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t207 = t183

        # pd_op.assign: (1xi64) <- (1xi64)
        t208 = t183

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t209 = paddle._C_ops.slice(t169, [0], t170, t183, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t210 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        t211 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t212 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t213 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t214 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t215 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t216 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t217 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t218 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t219 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t220 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t221 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t222 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t223 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t224 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t225 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t226 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t227 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t228 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t229 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t230 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t231 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t232 = t210

        # pd_op.assign: (1xi64) <- (1xi64)
        t233 = t210

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t234 = paddle._C_ops.slice(t169, [0], t183, t210, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t235 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        t236 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t237 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t238 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t239 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t240 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t241 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t242 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t243 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t244 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t245 = t235

        # pd_op.assign: (1xi64) <- (1xi64)
        t246 = t235

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t247 = paddle._C_ops.slice(t169, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t248 = paddle._C_ops.transpose(t234, [0, 1, 3, 2])
        del t234

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t249 = paddle._C_ops.matmul(t209, t248, False, False)

        # pd_op.full: (1xf32) <- ()
        t250 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t251 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t252 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t253 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t254 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t255 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t256 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t257 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t258 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t259 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t260 = t250

        # pd_op.assign: (1xf32) <- (1xf32)
        t261 = t250

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t262 = paddle._C_ops.scale(t249, t250, float("0"), True)
        del t249

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t263 = paddle._C_ops.softmax(t262, -1)
        del t262

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t264 = paddle._C_ops.matmul(t263, t247, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t265 = paddle._C_ops.transpose(t264, [0, 2, 1, 3])
        del t264

        # pd_op.full_int_array: (3xi64) <- ()
        t266 = [-1, 197, 768]

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t267 = paddle._C_ops.reshape(t265, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t268 = paddle._C_ops.matmul(t267, t7, False, False)
        del t7

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t269 = paddle._C_ops.add(t268, t8)
        del t8

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t270 = paddle._C_ops.add(t159, t269)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t271, t272, t273 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t270, t9, t10, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t10, t9

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t274 = paddle._C_ops.matmul(t271, t11, False, False)
        del t11

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t275 = paddle._C_ops.add(t274, t12)
        del t12

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t276 = paddle._C_ops.gelu(t275, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t277 = paddle._C_ops.matmul(t276, t13, False, False)
        del t13

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t278 = paddle._C_ops.add(t277, t14)
        del t14

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t279 = paddle._C_ops.add(t270, t278)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t280, t281, t282 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t279, t15, t16, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t16, t15

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t283 = paddle._C_ops.matmul(t280, t17, False, False)
        del t17

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t284 = paddle._C_ops.add(t283, t18)
        del t18

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t285 = paddle._C_ops.reshape(t284, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t286 = paddle._C_ops.transpose(t285, [2, 0, 3, 1, 4])
        del t285

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t287 = paddle._C_ops.slice(t286, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t288 = paddle._C_ops.slice(t286, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t289 = paddle._C_ops.slice(t286, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t290 = paddle._C_ops.transpose(t288, [0, 1, 3, 2])
        del t288

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t291 = paddle._C_ops.matmul(t287, t290, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t292 = paddle._C_ops.scale(t291, t250, float("0"), True)
        del t291

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t293 = paddle._C_ops.softmax(t292, -1)
        del t292

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t294 = paddle._C_ops.matmul(t293, t289, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t295 = paddle._C_ops.transpose(t294, [0, 2, 1, 3])
        del t294

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t296 = paddle._C_ops.reshape(t295, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t297 = paddle._C_ops.matmul(t296, t19, False, False)
        del t19

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t298 = paddle._C_ops.add(t297, t20)
        del t20

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t299 = paddle._C_ops.add(t279, t298)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t300, t301, t302 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t299, t21, t22, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t22, t21

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t303 = paddle._C_ops.matmul(t300, t23, False, False)
        del t23

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t304 = paddle._C_ops.add(t303, t24)
        del t24

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t305 = paddle._C_ops.gelu(t304, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t306 = paddle._C_ops.matmul(t305, t25, False, False)
        del t25

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t307 = paddle._C_ops.add(t306, t26)
        del t26

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t308 = paddle._C_ops.add(t299, t307)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t309, t310, t311 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t308, t27, t28, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t28, t27

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t312 = paddle._C_ops.matmul(t309, t29, False, False)
        del t29

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t313 = paddle._C_ops.add(t312, t30)
        del t30

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t314 = paddle._C_ops.reshape(t313, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t315 = paddle._C_ops.transpose(t314, [2, 0, 3, 1, 4])
        del t314

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t316 = paddle._C_ops.slice(t315, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t317 = paddle._C_ops.slice(t315, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t318 = paddle._C_ops.slice(t315, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t319 = paddle._C_ops.transpose(t317, [0, 1, 3, 2])
        del t317

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t320 = paddle._C_ops.matmul(t316, t319, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t321 = paddle._C_ops.scale(t320, t250, float("0"), True)
        del t320

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t322 = paddle._C_ops.softmax(t321, -1)
        del t321

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t323 = paddle._C_ops.matmul(t322, t318, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t324 = paddle._C_ops.transpose(t323, [0, 2, 1, 3])
        del t323

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t325 = paddle._C_ops.reshape(t324, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t326 = paddle._C_ops.matmul(t325, t31, False, False)
        del t31

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t327 = paddle._C_ops.add(t326, t32)
        del t32

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t328 = paddle._C_ops.add(t308, t327)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t329, t330, t331 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t328, t33, t34, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t34, t33

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t332 = paddle._C_ops.matmul(t329, t35, False, False)
        del t35

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t333 = paddle._C_ops.add(t332, t36)
        del t36

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t334 = paddle._C_ops.gelu(t333, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t335 = paddle._C_ops.matmul(t334, t37, False, False)
        del t37

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t336 = paddle._C_ops.add(t335, t38)
        del t38

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t337 = paddle._C_ops.add(t328, t336)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t338, t339, t340 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t337, t39, t40, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t40, t39

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t341 = paddle._C_ops.matmul(t338, t41, False, False)
        del t41

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t342 = paddle._C_ops.add(t341, t42)
        del t42

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t343 = paddle._C_ops.reshape(t342, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t344 = paddle._C_ops.transpose(t343, [2, 0, 3, 1, 4])
        del t343

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t345 = paddle._C_ops.slice(t344, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t346 = paddle._C_ops.slice(t344, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t347 = paddle._C_ops.slice(t344, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t348 = paddle._C_ops.transpose(t346, [0, 1, 3, 2])
        del t346

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t349 = paddle._C_ops.matmul(t345, t348, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t350 = paddle._C_ops.scale(t349, t250, float("0"), True)
        del t349

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t351 = paddle._C_ops.softmax(t350, -1)
        del t350

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t352 = paddle._C_ops.matmul(t351, t347, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t353 = paddle._C_ops.transpose(t352, [0, 2, 1, 3])
        del t352

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t354 = paddle._C_ops.reshape(t353, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t355 = paddle._C_ops.matmul(t354, t43, False, False)
        del t43

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t356 = paddle._C_ops.add(t355, t44)
        del t44

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t357 = paddle._C_ops.add(t337, t356)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t358, t359, t360 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t357, t45, t46, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t46, t45

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t361 = paddle._C_ops.matmul(t358, t47, False, False)
        del t47

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t362 = paddle._C_ops.add(t361, t48)
        del t48

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t363 = paddle._C_ops.gelu(t362, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t364 = paddle._C_ops.matmul(t363, t49, False, False)
        del t49

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t365 = paddle._C_ops.add(t364, t50)
        del t50

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t366 = paddle._C_ops.add(t357, t365)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t367, t368, t369 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t366, t51, t52, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t52, t51

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t370 = paddle._C_ops.matmul(t367, t53, False, False)
        del t53

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t371 = paddle._C_ops.add(t370, t54)
        del t54

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t372 = paddle._C_ops.reshape(t371, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t373 = paddle._C_ops.transpose(t372, [2, 0, 3, 1, 4])
        del t372

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t374 = paddle._C_ops.slice(t373, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t375 = paddle._C_ops.slice(t373, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t376 = paddle._C_ops.slice(t373, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t377 = paddle._C_ops.transpose(t375, [0, 1, 3, 2])
        del t375

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t378 = paddle._C_ops.matmul(t374, t377, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t379 = paddle._C_ops.scale(t378, t250, float("0"), True)
        del t378

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t380 = paddle._C_ops.softmax(t379, -1)
        del t379

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t381 = paddle._C_ops.matmul(t380, t376, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t382 = paddle._C_ops.transpose(t381, [0, 2, 1, 3])
        del t381

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t383 = paddle._C_ops.reshape(t382, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t384 = paddle._C_ops.matmul(t383, t55, False, False)
        del t55

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t385 = paddle._C_ops.add(t384, t56)
        del t56

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t386 = paddle._C_ops.add(t366, t385)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t387, t388, t389 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t386, t57, t58, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t58, t57

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t390 = paddle._C_ops.matmul(t387, t59, False, False)
        del t59

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t391 = paddle._C_ops.add(t390, t60)
        del t60

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t392 = paddle._C_ops.gelu(t391, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t393 = paddle._C_ops.matmul(t392, t61, False, False)
        del t61

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t394 = paddle._C_ops.add(t393, t62)
        del t62

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t395 = paddle._C_ops.add(t386, t394)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t396, t397, t398 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t395, t63, t64, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t64, t63

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t399 = paddle._C_ops.matmul(t396, t65, False, False)
        del t65

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t400 = paddle._C_ops.add(t399, t66)
        del t66

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t401 = paddle._C_ops.reshape(t400, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t402 = paddle._C_ops.transpose(t401, [2, 0, 3, 1, 4])
        del t401

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t403 = paddle._C_ops.slice(t402, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t404 = paddle._C_ops.slice(t402, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t405 = paddle._C_ops.slice(t402, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t406 = paddle._C_ops.transpose(t404, [0, 1, 3, 2])
        del t404

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t407 = paddle._C_ops.matmul(t403, t406, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t408 = paddle._C_ops.scale(t407, t250, float("0"), True)
        del t407

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t409 = paddle._C_ops.softmax(t408, -1)
        del t408

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t410 = paddle._C_ops.matmul(t409, t405, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t411 = paddle._C_ops.transpose(t410, [0, 2, 1, 3])
        del t410

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t412 = paddle._C_ops.reshape(t411, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t413 = paddle._C_ops.matmul(t412, t67, False, False)
        del t67

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t414 = paddle._C_ops.add(t413, t68)
        del t68

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t415 = paddle._C_ops.add(t395, t414)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t416, t417, t418 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t415, t69, t70, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t70, t69

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t419 = paddle._C_ops.matmul(t416, t71, False, False)
        del t71

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t420 = paddle._C_ops.add(t419, t72)
        del t72

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t421 = paddle._C_ops.gelu(t420, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t422 = paddle._C_ops.matmul(t421, t73, False, False)
        del t73

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t423 = paddle._C_ops.add(t422, t74)
        del t74

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t424 = paddle._C_ops.add(t415, t423)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t425, t426, t427 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t424, t75, t76, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t76, t75

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t428 = paddle._C_ops.matmul(t425, t77, False, False)
        del t77

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t429 = paddle._C_ops.add(t428, t78)
        del t78

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t430 = paddle._C_ops.reshape(t429, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t431 = paddle._C_ops.transpose(t430, [2, 0, 3, 1, 4])
        del t430

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t432 = paddle._C_ops.slice(t431, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t433 = paddle._C_ops.slice(t431, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t434 = paddle._C_ops.slice(t431, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t435 = paddle._C_ops.transpose(t433, [0, 1, 3, 2])
        del t433

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t436 = paddle._C_ops.matmul(t432, t435, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t437 = paddle._C_ops.scale(t436, t250, float("0"), True)
        del t436

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t438 = paddle._C_ops.softmax(t437, -1)
        del t437

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t439 = paddle._C_ops.matmul(t438, t434, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t440 = paddle._C_ops.transpose(t439, [0, 2, 1, 3])
        del t439

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t441 = paddle._C_ops.reshape(t440, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t442 = paddle._C_ops.matmul(t441, t79, False, False)
        del t79

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t443 = paddle._C_ops.add(t442, t80)
        del t80

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t444 = paddle._C_ops.add(t424, t443)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t445, t446, t447 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t444, t81, t82, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t82, t81

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t448 = paddle._C_ops.matmul(t445, t83, False, False)
        del t83

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t449 = paddle._C_ops.add(t448, t84)
        del t84

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t450 = paddle._C_ops.gelu(t449, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t451 = paddle._C_ops.matmul(t450, t85, False, False)
        del t85

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t452 = paddle._C_ops.add(t451, t86)
        del t86

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t453 = paddle._C_ops.add(t444, t452)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t454, t455, t456 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t453, t87, t88, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t88, t87

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t457 = paddle._C_ops.matmul(t454, t89, False, False)
        del t89

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t458 = paddle._C_ops.add(t457, t90)
        del t90

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t459 = paddle._C_ops.reshape(t458, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t460 = paddle._C_ops.transpose(t459, [2, 0, 3, 1, 4])
        del t459

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t461 = paddle._C_ops.slice(t460, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t462 = paddle._C_ops.slice(t460, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t463 = paddle._C_ops.slice(t460, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t464 = paddle._C_ops.transpose(t462, [0, 1, 3, 2])
        del t462

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t465 = paddle._C_ops.matmul(t461, t464, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t466 = paddle._C_ops.scale(t465, t250, float("0"), True)
        del t465

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t467 = paddle._C_ops.softmax(t466, -1)
        del t466

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t468 = paddle._C_ops.matmul(t467, t463, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t469 = paddle._C_ops.transpose(t468, [0, 2, 1, 3])
        del t468

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t470 = paddle._C_ops.reshape(t469, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t471 = paddle._C_ops.matmul(t470, t91, False, False)
        del t91

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t472 = paddle._C_ops.add(t471, t92)
        del t92

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t473 = paddle._C_ops.add(t453, t472)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t474, t475, t476 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t473, t93, t94, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t94, t93

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t477 = paddle._C_ops.matmul(t474, t95, False, False)
        del t95

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t478 = paddle._C_ops.add(t477, t96)
        del t96

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t479 = paddle._C_ops.gelu(t478, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t480 = paddle._C_ops.matmul(t479, t97, False, False)
        del t97

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t481 = paddle._C_ops.add(t480, t98)
        del t98

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t482 = paddle._C_ops.add(t473, t481)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t483, t484, t485 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t482, t99, t100, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t100, t99

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t486 = paddle._C_ops.matmul(t483, t101, False, False)
        del t101

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t487 = paddle._C_ops.add(t486, t102)
        del t102

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t488 = paddle._C_ops.reshape(t487, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t489 = paddle._C_ops.transpose(t488, [2, 0, 3, 1, 4])
        del t488

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t490 = paddle._C_ops.slice(t489, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t491 = paddle._C_ops.slice(t489, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t492 = paddle._C_ops.slice(t489, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t493 = paddle._C_ops.transpose(t491, [0, 1, 3, 2])
        del t491

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t494 = paddle._C_ops.matmul(t490, t493, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t495 = paddle._C_ops.scale(t494, t250, float("0"), True)
        del t494

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t496 = paddle._C_ops.softmax(t495, -1)
        del t495

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t497 = paddle._C_ops.matmul(t496, t492, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t498 = paddle._C_ops.transpose(t497, [0, 2, 1, 3])
        del t497

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t499 = paddle._C_ops.reshape(t498, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t500 = paddle._C_ops.matmul(t499, t103, False, False)
        del t103

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t501 = paddle._C_ops.add(t500, t104)
        del t104

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t502 = paddle._C_ops.add(t482, t501)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t503, t504, t505 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t502, t105, t106, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t106, t105

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t506 = paddle._C_ops.matmul(t503, t107, False, False)
        del t107

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t507 = paddle._C_ops.add(t506, t108)
        del t108

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t508 = paddle._C_ops.gelu(t507, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t509 = paddle._C_ops.matmul(t508, t109, False, False)
        del t109

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t510 = paddle._C_ops.add(t509, t110)
        del t110

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t511 = paddle._C_ops.add(t502, t510)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t512, t513, t514 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t511, t111, t112, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t112, t111

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t515 = paddle._C_ops.matmul(t512, t113, False, False)
        del t113

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t516 = paddle._C_ops.add(t515, t114)
        del t114

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t517 = paddle._C_ops.reshape(t516, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t518 = paddle._C_ops.transpose(t517, [2, 0, 3, 1, 4])
        del t517

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t519 = paddle._C_ops.slice(t518, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t520 = paddle._C_ops.slice(t518, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t521 = paddle._C_ops.slice(t518, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t522 = paddle._C_ops.transpose(t520, [0, 1, 3, 2])
        del t520

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t523 = paddle._C_ops.matmul(t519, t522, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t524 = paddle._C_ops.scale(t523, t250, float("0"), True)
        del t523

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t525 = paddle._C_ops.softmax(t524, -1)
        del t524

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t526 = paddle._C_ops.matmul(t525, t521, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t527 = paddle._C_ops.transpose(t526, [0, 2, 1, 3])
        del t526

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t528 = paddle._C_ops.reshape(t527, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t529 = paddle._C_ops.matmul(t528, t115, False, False)
        del t115

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t530 = paddle._C_ops.add(t529, t116)
        del t116

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t531 = paddle._C_ops.add(t511, t530)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t532, t533, t534 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t531, t117, t118, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t118, t117

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t535 = paddle._C_ops.matmul(t532, t119, False, False)
        del t119

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t536 = paddle._C_ops.add(t535, t120)
        del t120

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t537 = paddle._C_ops.gelu(t536, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t538 = paddle._C_ops.matmul(t537, t121, False, False)
        del t121

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t539 = paddle._C_ops.add(t538, t122)
        del t122

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t540 = paddle._C_ops.add(t531, t539)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t541, t542, t543 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t540, t123, t124, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t124, t123

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t544 = paddle._C_ops.matmul(t541, t125, False, False)
        del t125

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t545 = paddle._C_ops.add(t544, t126)
        del t126

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t546 = paddle._C_ops.reshape(t545, t167)

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t547 = paddle._C_ops.transpose(t546, [2, 0, 3, 1, 4])
        del t546

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t548 = paddle._C_ops.slice(t547, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t549 = paddle._C_ops.slice(t547, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t550 = paddle._C_ops.slice(t547, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t551 = paddle._C_ops.transpose(t549, [0, 1, 3, 2])
        del t549

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t552 = paddle._C_ops.matmul(t548, t551, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t553 = paddle._C_ops.scale(t552, t250, float("0"), True)
        del t552

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t554 = paddle._C_ops.softmax(t553, -1)
        del t553

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t555 = paddle._C_ops.matmul(t554, t550, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t556 = paddle._C_ops.transpose(t555, [0, 2, 1, 3])
        del t555

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t557 = paddle._C_ops.reshape(t556, t266)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t558 = paddle._C_ops.matmul(t557, t127, False, False)
        del t127

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t559 = paddle._C_ops.add(t558, t128)
        del t128

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t560 = paddle._C_ops.add(t540, t559)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t561, t562, t563 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t560, t129, t130, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t130, t129

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t564 = paddle._C_ops.matmul(t561, t131, False, False)
        del t131

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t565 = paddle._C_ops.add(t564, t132)
        del t132

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t566 = paddle._C_ops.gelu(t565, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t567 = paddle._C_ops.matmul(t566, t133, False, False)
        del t133

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t568 = paddle._C_ops.add(t567, t134)
        del t134

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t569 = paddle._C_ops.add(t560, t568)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t570, t571, t572 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t569, t135, t136, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t136, t135

        # pd_op.matmul: (32x197x2304xf32) <- (32x197x768xf32, 768x2304xf32)
        t573 = paddle._C_ops.matmul(t570, t137, False, False)
        del t137

        # pd_op.add: (32x197x2304xf32) <- (32x197x2304xf32, 2304xf32)
        t574 = paddle._C_ops.add(t573, t138)
        del t138

        # pd_op.reshape: (32x197x3x12x64xf32) <- (32x197x2304xf32, 5xi64)
        t575 = paddle._C_ops.reshape(t574, t167)
        del t167

        # pd_op.transpose: (3x32x12x197x64xf32) <- (32x197x3x12x64xf32)
        t576 = paddle._C_ops.transpose(t575, [2, 0, 3, 1, 4])
        del t575

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t577 = paddle._C_ops.slice(t576, [0], t170, t183, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t578 = paddle._C_ops.slice(t576, [0], t183, t210, [1], [0])

        # pd_op.slice: (32x12x197x64xf32) <- (3x32x12x197x64xf32, 1xi64, 1xi64)
        t579 = paddle._C_ops.slice(t576, [0], t210, t235, [1], [0])

        # pd_op.transpose: (32x12x64x197xf32) <- (32x12x197x64xf32)
        t580 = paddle._C_ops.transpose(t578, [0, 1, 3, 2])
        del t578

        # pd_op.matmul: (32x12x197x197xf32) <- (32x12x197x64xf32, 32x12x64x197xf32)
        t581 = paddle._C_ops.matmul(t577, t580, False, False)

        # pd_op.scale: (32x12x197x197xf32) <- (32x12x197x197xf32, 1xf32)
        t582 = paddle._C_ops.scale(t581, t250, float("0"), True)
        del t581

        # pd_op.softmax: (32x12x197x197xf32) <- (32x12x197x197xf32)
        t583 = paddle._C_ops.softmax(t582, -1)
        del t582

        # pd_op.matmul: (32x12x197x64xf32) <- (32x12x197x197xf32, 32x12x197x64xf32)
        t584 = paddle._C_ops.matmul(t583, t579, False, False)

        # pd_op.transpose: (32x197x12x64xf32) <- (32x12x197x64xf32)
        t585 = paddle._C_ops.transpose(t584, [0, 2, 1, 3])
        del t584

        # pd_op.reshape: (32x197x768xf32) <- (32x197x12x64xf32, 3xi64)
        t586 = paddle._C_ops.reshape(t585, t266)
        del t266

        # pd_op.matmul: (32x197x768xf32) <- (32x197x768xf32, 768x768xf32)
        t587 = paddle._C_ops.matmul(t586, t139, False, False)
        del t139

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t588 = paddle._C_ops.add(t587, t140)
        del t140

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t589 = paddle._C_ops.add(t569, t588)

        # pd_op.layer_norm: (32x197x768xf32, 32x197xf32, 32x197xf32) <- (32x197x768xf32, 768xf32, 768xf32)
        t590, t591, t592 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t589, t141, t142, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t142, t141

        # pd_op.matmul: (32x197x3072xf32) <- (32x197x768xf32, 768x3072xf32)
        t593 = paddle._C_ops.matmul(t590, t143, False, False)
        del t143

        # pd_op.add: (32x197x3072xf32) <- (32x197x3072xf32, 3072xf32)
        t594 = paddle._C_ops.add(t593, t144)
        del t144

        # pd_op.gelu: (32x197x3072xf32) <- (32x197x3072xf32)
        t595 = paddle._C_ops.gelu(t594, False)

        # pd_op.matmul: (32x197x768xf32) <- (32x197x3072xf32, 3072x768xf32)
        t596 = paddle._C_ops.matmul(t595, t145, False, False)
        del t145

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 768xf32)
        t597 = paddle._C_ops.add(t596, t146)
        del t146

        # pd_op.add: (32x197x768xf32) <- (32x197x768xf32, 32x197x768xf32)
        t598 = paddle._C_ops.add(t589, t597)

        # pd_op.full_int_array: (1xi64) <- ()
        t599 = [2147483647]

        # pd_op.slice: (32x196x768xf32) <- (32x197x768xf32, 1xi64, 1xi64)
        t600 = paddle._C_ops.slice(t598, [1], t183, t599, [1], [])

        # pd_op.slice: (32x768xf32) <- (32x196x768xf32, 1xi64, 1xi64)
        t601 = paddle._C_ops.slice(t600, [1], t170, t183, [1], [1])

        # pd_op.slice: (32x195x768xf32) <- (32x196x768xf32, 1xi64, 1xi64)
        t602 = paddle._C_ops.slice(t600, [1], t183, t599, [1], [])

        # pd_op.layer_norm: (32x768xf32, 32xf32, 32xf32) <- (32x768xf32, 768xf32, 768xf32)
        t603, t604, t605 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t601, t147, t148, float("1e-05"), 1),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t148, t147

        # pd_op.matmul: (32x102xf32) <- (32x768xf32, 768x102xf32)
        t606 = paddle._C_ops.matmul(t603, t149, False, False)
        del t149

        return (
            t150,
            t152,
            t153,
            t154,
            t155,
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
            t263,
            t265,
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
            t286,
            t287,
            t289,
            t290,
            t293,
            t295,
            t296,
            t297,
            t298,
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
            t315,
            t316,
            t318,
            t319,
            t322,
            t324,
            t325,
            t326,
            t327,
            t328,
            t329,
            t330,
            t331,
            t332,
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
            t344,
            t345,
            t347,
            t348,
            t351,
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
            t367,
            t368,
            t369,
            t370,
            t371,
            t373,
            t374,
            t376,
            t377,
            t380,
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
            t399,
            t400,
            t402,
            t403,
            t405,
            t406,
            t409,
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
            t431,
            t432,
            t434,
            t435,
            t438,
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
            t460,
            t461,
            t463,
            t464,
            t467,
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
            t489,
            t490,
            t492,
            t493,
            t496,
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
            t518,
            t519,
            t521,
            t522,
            t525,
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
            t547,
            t548,
            t550,
            t551,
            t554,
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
            t574,
            t576,
            t577,
            t579,
            t580,
            t583,
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
            t603,
            t604,
            t605,
            t606,
        )
