import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input0: paddle.Tensor,
        t0: paddle.Tensor,
        t1: paddle.Tensor,
        input1: paddle.Tensor,
        input2: paddle.Tensor,
        t2: paddle.Tensor,
        t3: paddle.Tensor,
        t4: paddle.Tensor,
        t5: paddle.Tensor,
        t6: paddle.Tensor,
        t7: paddle.Tensor,
        t8: paddle.Tensor,
        t9: paddle.Tensor,
        input3: paddle.Tensor,
        t10: paddle.Tensor,
        t11: paddle.Tensor,
        t12: paddle.Tensor,
        t13: paddle.Tensor,
        t14: paddle.Tensor,
        t15: paddle.Tensor,
        t16: paddle.Tensor,
        t17: paddle.Tensor,
        input4: paddle.Tensor,
        t18: paddle.Tensor,
        t19: paddle.Tensor,
        t20: paddle.Tensor,
        t21: paddle.Tensor,
        t22: paddle.Tensor,
        t23: paddle.Tensor,
        t24: paddle.Tensor,
        t25: paddle.Tensor,
        input5: paddle.Tensor,
        input6: paddle.Tensor,
        input7: paddle.Tensor,
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
        input8: paddle.Tensor,
        t36: paddle.Tensor,
        t37: paddle.Tensor,
        t38: paddle.Tensor,
        t39: paddle.Tensor,
        t40: paddle.Tensor,
        t41: paddle.Tensor,
        t42: paddle.Tensor,
        t43: paddle.Tensor,
        input9: paddle.Tensor,
        t44: paddle.Tensor,
        t45: paddle.Tensor,
        t46: paddle.Tensor,
        t47: paddle.Tensor,
        t48: paddle.Tensor,
        t49: paddle.Tensor,
        t50: paddle.Tensor,
        t51: paddle.Tensor,
        input10: paddle.Tensor,
        input11: paddle.Tensor,
        input12: paddle.Tensor,
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
        input13: paddle.Tensor,
        t62: paddle.Tensor,
        t63: paddle.Tensor,
        t64: paddle.Tensor,
        t65: paddle.Tensor,
        t66: paddle.Tensor,
        t67: paddle.Tensor,
        t68: paddle.Tensor,
        t69: paddle.Tensor,
        input14: paddle.Tensor,
        t70: paddle.Tensor,
        t71: paddle.Tensor,
        t72: paddle.Tensor,
        t73: paddle.Tensor,
        t74: paddle.Tensor,
        t75: paddle.Tensor,
        t76: paddle.Tensor,
        t77: paddle.Tensor,
        input15: paddle.Tensor,
        t78: paddle.Tensor,
        t79: paddle.Tensor,
        t80: paddle.Tensor,
        t81: paddle.Tensor,
        t82: paddle.Tensor,
        t83: paddle.Tensor,
        t84: paddle.Tensor,
        t85: paddle.Tensor,
        input16: paddle.Tensor,
        t86: paddle.Tensor,
        t87: paddle.Tensor,
        t88: paddle.Tensor,
        t89: paddle.Tensor,
        t90: paddle.Tensor,
        t91: paddle.Tensor,
        t92: paddle.Tensor,
        t93: paddle.Tensor,
        input17: paddle.Tensor,
        t94: paddle.Tensor,
        t95: paddle.Tensor,
        t96: paddle.Tensor,
        t97: paddle.Tensor,
        t98: paddle.Tensor,
        t99: paddle.Tensor,
        t100: paddle.Tensor,
        t101: paddle.Tensor,
        input18: paddle.Tensor,
        t102: paddle.Tensor,
        t103: paddle.Tensor,
        t104: paddle.Tensor,
        t105: paddle.Tensor,
        t106: paddle.Tensor,
        t107: paddle.Tensor,
        t108: paddle.Tensor,
        t109: paddle.Tensor,
        input19: paddle.Tensor,
        t110: paddle.Tensor,
        t111: paddle.Tensor,
        t112: paddle.Tensor,
        t113: paddle.Tensor,
        t114: paddle.Tensor,
        t115: paddle.Tensor,
        t116: paddle.Tensor,
        t117: paddle.Tensor,
        input20: paddle.Tensor,
        t118: paddle.Tensor,
        t119: paddle.Tensor,
        t120: paddle.Tensor,
        t121: paddle.Tensor,
        t122: paddle.Tensor,
        t123: paddle.Tensor,
        t124: paddle.Tensor,
        t125: paddle.Tensor,
        input21: paddle.Tensor,
        t126: paddle.Tensor,
        t127: paddle.Tensor,
        t128: paddle.Tensor,
        t129: paddle.Tensor,
        t130: paddle.Tensor,
        t131: paddle.Tensor,
        t132: paddle.Tensor,
        t133: paddle.Tensor,
        input22: paddle.Tensor,
        t134: paddle.Tensor,
        t135: paddle.Tensor,
        t136: paddle.Tensor,
        t137: paddle.Tensor,
        t138: paddle.Tensor,
        t139: paddle.Tensor,
        t140: paddle.Tensor,
        t141: paddle.Tensor,
        input23: paddle.Tensor,
        t142: paddle.Tensor,
        t143: paddle.Tensor,
        t144: paddle.Tensor,
        t145: paddle.Tensor,
        t146: paddle.Tensor,
        t147: paddle.Tensor,
        t148: paddle.Tensor,
        t149: paddle.Tensor,
        input24: paddle.Tensor,
        t150: paddle.Tensor,
        t151: paddle.Tensor,
        t152: paddle.Tensor,
        t153: paddle.Tensor,
        t154: paddle.Tensor,
        t155: paddle.Tensor,
        t156: paddle.Tensor,
        t157: paddle.Tensor,
        input25: paddle.Tensor,
        t158: paddle.Tensor,
        t159: paddle.Tensor,
        t160: paddle.Tensor,
        t161: paddle.Tensor,
        t162: paddle.Tensor,
        t163: paddle.Tensor,
        t164: paddle.Tensor,
        t165: paddle.Tensor,
        input26: paddle.Tensor,
        t166: paddle.Tensor,
        t167: paddle.Tensor,
        t168: paddle.Tensor,
        t169: paddle.Tensor,
        t170: paddle.Tensor,
        t171: paddle.Tensor,
        t172: paddle.Tensor,
        t173: paddle.Tensor,
        input27: paddle.Tensor,
        t174: paddle.Tensor,
        t175: paddle.Tensor,
        t176: paddle.Tensor,
        t177: paddle.Tensor,
    ):
        # pd_op.conv2d: (64x96x56x56xf32) <- (64x3x224x224xf32, 96x3x4x4xf32)
        t178 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t179 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t180 = paddle._C_ops.reshape(t1, t179)
        del t1

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 1x96x1x1xf32)
        t181 = paddle._C_ops.add(t178, t180)

        # pd_op.mean: (64x1x56x56xf32) <- (64x96x56x56xf32)
        t182 = paddle._C_ops.mean(t181, [1], True)

        # pd_op.subtract: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x1x56x56xf32)
        t183 = paddle._C_ops.subtract(t181, t182)

        # pd_op.assign: (64x96x56x56xf32) <- (64x96x56x56xf32)
        t184 = t183

        # pd_op.pow: (64x96x56x56xf32) <- (64x96x56x56xf32)
        t185 = paddle._C_ops.pow(t183, float("2"))

        # pd_op.mean: (64x1x56x56xf32) <- (64x96x56x56xf32)
        t186 = paddle._C_ops.mean(t185, [1], True)

        # pd_op.full: (1xf32) <- ()
        t187 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t188 = t187

        # pd_op.assign: (1xf32) <- (1xf32)
        t189 = t187

        # pd_op.assign: (1xf32) <- (1xf32)
        t190 = t187

        # pd_op.scale: (64x1x56x56xf32) <- (64x1x56x56xf32, 1xf32)
        t191 = paddle._C_ops.scale(t186, t187, float("1e-06"), True)
        del t186

        # pd_op.sqrt: (64x1x56x56xf32) <- (64x1x56x56xf32)
        t192 = paddle._C_ops.sqrt(t191)
        del t191

        # pd_op.divide: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x1x56x56xf32)
        t193 = paddle._C_ops.divide(t183, t192)

        # pd_op.full_int_array: (2xi64) <- ()
        t194 = [1, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        t195 = t194

        # pd_op.assign: (2xi64) <- (2xi64)
        t196 = t194

        # pd_op.assign: (2xi64) <- (2xi64)
        t197 = t194

        # pd_op.assign: (2xi64) <- (2xi64)
        t198 = t194

        # pd_op.assign: (2xi64) <- (2xi64)
        t199 = t194

        # pd_op.assign: (2xi64) <- (2xi64)
        t200 = t194

        # pd_op.assign: (2xi64) <- (2xi64)
        t201 = t194

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        t202 = paddle._C_ops.unsqueeze(input1, t194)
        del input1

        # pd_op.multiply: (64x96x56x56xf32) <- (96x1x1xf32, 64x96x56x56xf32)
        t203 = paddle._C_ops.multiply(t202, t193)

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        t204 = paddle._C_ops.unsqueeze(input2, t194)
        del input2

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 96x1x1xf32)
        t205 = paddle._C_ops.add(t203, t204)

        # pd_op.depthwise_conv2d: (64x96x56x56xf32) <- (64x96x56x56xf32, 96x1x7x7xf32)
        t206 = paddle._C_ops.depthwise_conv2d(
            t205, t2, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t2

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t207 = paddle._C_ops.reshape(t3, t179)
        del t3

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 1x96x1x1xf32)
        t208 = paddle._C_ops.add(t206, t207)

        # pd_op.transpose: (64x56x56x96xf32) <- (64x96x56x56xf32)
        t209 = paddle._C_ops.transpose(t208, [0, 2, 3, 1])
        del t208

        # pd_op.layer_norm: (64x56x56x96xf32, 64x56x56xf32, 64x56x56xf32) <- (64x56x56x96xf32, 96xf32, 96xf32)
        t210, t211, t212 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t209, t4, t5, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t5, t4

        # pd_op.matmul: (64x56x56x384xf32) <- (64x56x56x96xf32, 96x384xf32)
        t213 = paddle._C_ops.matmul(t210, t6, False, False)
        del t6

        # pd_op.add: (64x56x56x384xf32) <- (64x56x56x384xf32, 384xf32)
        t214 = paddle._C_ops.add(t213, t7)
        del t7

        # pd_op.gelu: (64x56x56x384xf32) <- (64x56x56x384xf32)
        t215 = paddle._C_ops.gelu(t214, False)

        # pd_op.matmul: (64x56x56x96xf32) <- (64x56x56x384xf32, 384x96xf32)
        t216 = paddle._C_ops.matmul(t215, t8, False, False)
        del t8

        # pd_op.add: (64x56x56x96xf32) <- (64x56x56x96xf32, 96xf32)
        t217 = paddle._C_ops.add(t216, t9)
        del t9

        # pd_op.multiply: (64x56x56x96xf32) <- (96xf32, 64x56x56x96xf32)
        t218 = paddle._C_ops.multiply(input3, t217)
        del input3

        # pd_op.transpose: (64x96x56x56xf32) <- (64x56x56x96xf32)
        t219 = paddle._C_ops.transpose(t218, [0, 3, 1, 2])
        del t218

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x96x56x56xf32)
        t220 = paddle._C_ops.add(t205, t219)

        # pd_op.depthwise_conv2d: (64x96x56x56xf32) <- (64x96x56x56xf32, 96x1x7x7xf32)
        t221 = paddle._C_ops.depthwise_conv2d(
            t220, t10, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t10

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t222 = paddle._C_ops.reshape(t11, t179)
        del t11

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 1x96x1x1xf32)
        t223 = paddle._C_ops.add(t221, t222)

        # pd_op.transpose: (64x56x56x96xf32) <- (64x96x56x56xf32)
        t224 = paddle._C_ops.transpose(t223, [0, 2, 3, 1])
        del t223

        # pd_op.layer_norm: (64x56x56x96xf32, 64x56x56xf32, 64x56x56xf32) <- (64x56x56x96xf32, 96xf32, 96xf32)
        t225, t226, t227 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t224, t12, t13, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t13, t12

        # pd_op.matmul: (64x56x56x384xf32) <- (64x56x56x96xf32, 96x384xf32)
        t228 = paddle._C_ops.matmul(t225, t14, False, False)
        del t14

        # pd_op.add: (64x56x56x384xf32) <- (64x56x56x384xf32, 384xf32)
        t229 = paddle._C_ops.add(t228, t15)
        del t15

        # pd_op.gelu: (64x56x56x384xf32) <- (64x56x56x384xf32)
        t230 = paddle._C_ops.gelu(t229, False)

        # pd_op.matmul: (64x56x56x96xf32) <- (64x56x56x384xf32, 384x96xf32)
        t231 = paddle._C_ops.matmul(t230, t16, False, False)
        del t16

        # pd_op.add: (64x56x56x96xf32) <- (64x56x56x96xf32, 96xf32)
        t232 = paddle._C_ops.add(t231, t17)
        del t17

        # pd_op.multiply: (64x56x56x96xf32) <- (96xf32, 64x56x56x96xf32)
        t233 = paddle._C_ops.multiply(input4, t232)
        del input4

        # pd_op.transpose: (64x96x56x56xf32) <- (64x56x56x96xf32)
        t234 = paddle._C_ops.transpose(t233, [0, 3, 1, 2])
        del t233

        # pd_op.full: (xf32) <- ()
        t235 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t236 = paddle._C_ops.assign_value_(
            t235,
            [],
            paddle.float32,
            [float("0.997143")],
            paddle.framework._current_expected_place(),
        )
        del t235

        # pd_op.full_int_array: (4xi64) <- ()
        t237 = [64, 1, 1, 1]

        # pd_op.full: (1xf32) <- ()
        t238 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t239 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t240 = paddle._C_ops.add(t236, t239)
        del t239

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t241 = paddle._C_ops.floor(t240)
        del t240

        # pd_op.divide: (64x96x56x56xf32) <- (64x96x56x56xf32, xf32)
        t242 = paddle._C_ops.divide(t234, t236)

        # pd_op.multiply: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x1x1x1xf32)
        t243 = paddle._C_ops.multiply(t242, t241)

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x96x56x56xf32)
        t244 = paddle._C_ops.add(t220, t243)

        # pd_op.depthwise_conv2d: (64x96x56x56xf32) <- (64x96x56x56xf32, 96x1x7x7xf32)
        t245 = paddle._C_ops.depthwise_conv2d(
            t244, t18, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t18

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t246 = paddle._C_ops.reshape(t19, t179)
        del t19

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 1x96x1x1xf32)
        t247 = paddle._C_ops.add(t245, t246)

        # pd_op.transpose: (64x56x56x96xf32) <- (64x96x56x56xf32)
        t248 = paddle._C_ops.transpose(t247, [0, 2, 3, 1])
        del t247

        # pd_op.layer_norm: (64x56x56x96xf32, 64x56x56xf32, 64x56x56xf32) <- (64x56x56x96xf32, 96xf32, 96xf32)
        t249, t250, t251 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t248, t20, t21, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t21, t20

        # pd_op.matmul: (64x56x56x384xf32) <- (64x56x56x96xf32, 96x384xf32)
        t252 = paddle._C_ops.matmul(t249, t22, False, False)
        del t22

        # pd_op.add: (64x56x56x384xf32) <- (64x56x56x384xf32, 384xf32)
        t253 = paddle._C_ops.add(t252, t23)
        del t23

        # pd_op.gelu: (64x56x56x384xf32) <- (64x56x56x384xf32)
        t254 = paddle._C_ops.gelu(t253, False)

        # pd_op.matmul: (64x56x56x96xf32) <- (64x56x56x384xf32, 384x96xf32)
        t255 = paddle._C_ops.matmul(t254, t24, False, False)
        del t24

        # pd_op.add: (64x56x56x96xf32) <- (64x56x56x96xf32, 96xf32)
        t256 = paddle._C_ops.add(t255, t25)
        del t25

        # pd_op.multiply: (64x56x56x96xf32) <- (96xf32, 64x56x56x96xf32)
        t257 = paddle._C_ops.multiply(input5, t256)
        del input5

        # pd_op.transpose: (64x96x56x56xf32) <- (64x56x56x96xf32)
        t258 = paddle._C_ops.transpose(t257, [0, 3, 1, 2])
        del t257

        # pd_op.full: (xf32) <- ()
        t259 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t260 = paddle._C_ops.assign_value_(
            t259,
            [],
            paddle.float32,
            [float("0.994286")],
            paddle.framework._current_expected_place(),
        )
        del t259

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t261 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t262 = paddle._C_ops.add(t260, t261)
        del t261

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t263 = paddle._C_ops.floor(t262)
        del t262

        # pd_op.divide: (64x96x56x56xf32) <- (64x96x56x56xf32, xf32)
        t264 = paddle._C_ops.divide(t258, t260)

        # pd_op.multiply: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x1x1x1xf32)
        t265 = paddle._C_ops.multiply(t264, t263)

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x96x56x56xf32)
        t266 = paddle._C_ops.add(t244, t265)

        # pd_op.mean: (64x1x56x56xf32) <- (64x96x56x56xf32)
        t267 = paddle._C_ops.mean(t266, [1], True)

        # pd_op.subtract: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x1x56x56xf32)
        t268 = paddle._C_ops.subtract(t266, t267)

        # pd_op.pow: (64x96x56x56xf32) <- (64x96x56x56xf32)
        t269 = paddle._C_ops.pow(t268, float("2"))

        # pd_op.mean: (64x1x56x56xf32) <- (64x96x56x56xf32)
        t270 = paddle._C_ops.mean(t269, [1], True)

        # pd_op.subtract: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x1x56x56xf32)
        t271 = paddle._C_ops.subtract(t266, t267)

        # pd_op.scale: (64x1x56x56xf32) <- (64x1x56x56xf32, 1xf32)
        t272 = paddle._C_ops.scale(t270, t187, float("1e-06"), True)
        del t270

        # pd_op.sqrt: (64x1x56x56xf32) <- (64x1x56x56xf32)
        t273 = paddle._C_ops.sqrt(t272)
        del t272

        # pd_op.divide: (64x96x56x56xf32) <- (64x96x56x56xf32, 64x1x56x56xf32)
        t274 = paddle._C_ops.divide(t271, t273)

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        t275 = paddle._C_ops.unsqueeze(input6, t194)
        del input6

        # pd_op.multiply: (64x96x56x56xf32) <- (96x1x1xf32, 64x96x56x56xf32)
        t276 = paddle._C_ops.multiply(t275, t274)

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        t277 = paddle._C_ops.unsqueeze(input7, t194)
        del input7

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 96x1x1xf32)
        t278 = paddle._C_ops.add(t276, t277)

        # pd_op.conv2d: (64x192x28x28xf32) <- (64x96x56x56xf32, 192x96x2x2xf32)
        t279 = paddle._C_ops.conv2d(
            t278, t26, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t26

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t280 = paddle._C_ops.reshape(t27, t179)
        del t27

        # pd_op.add: (64x192x28x28xf32) <- (64x192x28x28xf32, 1x192x1x1xf32)
        t281 = paddle._C_ops.add(t279, t280)

        # pd_op.depthwise_conv2d: (64x192x28x28xf32) <- (64x192x28x28xf32, 192x1x7x7xf32)
        t282 = paddle._C_ops.depthwise_conv2d(
            t281, t28, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t28

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t283 = paddle._C_ops.reshape(t29, t179)
        del t29

        # pd_op.add: (64x192x28x28xf32) <- (64x192x28x28xf32, 1x192x1x1xf32)
        t284 = paddle._C_ops.add(t282, t283)

        # pd_op.transpose: (64x28x28x192xf32) <- (64x192x28x28xf32)
        t285 = paddle._C_ops.transpose(t284, [0, 2, 3, 1])
        del t284

        # pd_op.layer_norm: (64x28x28x192xf32, 64x28x28xf32, 64x28x28xf32) <- (64x28x28x192xf32, 192xf32, 192xf32)
        t286, t287, t288 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t285, t30, t31, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t31, t30

        # pd_op.matmul: (64x28x28x768xf32) <- (64x28x28x192xf32, 192x768xf32)
        t289 = paddle._C_ops.matmul(t286, t32, False, False)
        del t32

        # pd_op.add: (64x28x28x768xf32) <- (64x28x28x768xf32, 768xf32)
        t290 = paddle._C_ops.add(t289, t33)
        del t33

        # pd_op.gelu: (64x28x28x768xf32) <- (64x28x28x768xf32)
        t291 = paddle._C_ops.gelu(t290, False)

        # pd_op.matmul: (64x28x28x192xf32) <- (64x28x28x768xf32, 768x192xf32)
        t292 = paddle._C_ops.matmul(t291, t34, False, False)
        del t34

        # pd_op.add: (64x28x28x192xf32) <- (64x28x28x192xf32, 192xf32)
        t293 = paddle._C_ops.add(t292, t35)
        del t35

        # pd_op.multiply: (64x28x28x192xf32) <- (192xf32, 64x28x28x192xf32)
        t294 = paddle._C_ops.multiply(input8, t293)
        del input8

        # pd_op.transpose: (64x192x28x28xf32) <- (64x28x28x192xf32)
        t295 = paddle._C_ops.transpose(t294, [0, 3, 1, 2])
        del t294

        # pd_op.full: (xf32) <- ()
        t296 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t297 = paddle._C_ops.assign_value_(
            t296,
            [],
            paddle.float32,
            [float("0.991429")],
            paddle.framework._current_expected_place(),
        )
        del t296

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t298 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t299 = paddle._C_ops.add(t297, t298)
        del t298

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t300 = paddle._C_ops.floor(t299)
        del t299

        # pd_op.divide: (64x192x28x28xf32) <- (64x192x28x28xf32, xf32)
        t301 = paddle._C_ops.divide(t295, t297)

        # pd_op.multiply: (64x192x28x28xf32) <- (64x192x28x28xf32, 64x1x1x1xf32)
        t302 = paddle._C_ops.multiply(t301, t300)

        # pd_op.add: (64x192x28x28xf32) <- (64x192x28x28xf32, 64x192x28x28xf32)
        t303 = paddle._C_ops.add(t281, t302)

        # pd_op.depthwise_conv2d: (64x192x28x28xf32) <- (64x192x28x28xf32, 192x1x7x7xf32)
        t304 = paddle._C_ops.depthwise_conv2d(
            t303, t36, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t36

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t305 = paddle._C_ops.reshape(t37, t179)
        del t37

        # pd_op.add: (64x192x28x28xf32) <- (64x192x28x28xf32, 1x192x1x1xf32)
        t306 = paddle._C_ops.add(t304, t305)

        # pd_op.transpose: (64x28x28x192xf32) <- (64x192x28x28xf32)
        t307 = paddle._C_ops.transpose(t306, [0, 2, 3, 1])
        del t306

        # pd_op.layer_norm: (64x28x28x192xf32, 64x28x28xf32, 64x28x28xf32) <- (64x28x28x192xf32, 192xf32, 192xf32)
        t308, t309, t310 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t307, t38, t39, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t39, t38

        # pd_op.matmul: (64x28x28x768xf32) <- (64x28x28x192xf32, 192x768xf32)
        t311 = paddle._C_ops.matmul(t308, t40, False, False)
        del t40

        # pd_op.add: (64x28x28x768xf32) <- (64x28x28x768xf32, 768xf32)
        t312 = paddle._C_ops.add(t311, t41)
        del t41

        # pd_op.gelu: (64x28x28x768xf32) <- (64x28x28x768xf32)
        t313 = paddle._C_ops.gelu(t312, False)

        # pd_op.matmul: (64x28x28x192xf32) <- (64x28x28x768xf32, 768x192xf32)
        t314 = paddle._C_ops.matmul(t313, t42, False, False)
        del t42

        # pd_op.add: (64x28x28x192xf32) <- (64x28x28x192xf32, 192xf32)
        t315 = paddle._C_ops.add(t314, t43)
        del t43

        # pd_op.multiply: (64x28x28x192xf32) <- (192xf32, 64x28x28x192xf32)
        t316 = paddle._C_ops.multiply(input9, t315)
        del input9

        # pd_op.transpose: (64x192x28x28xf32) <- (64x28x28x192xf32)
        t317 = paddle._C_ops.transpose(t316, [0, 3, 1, 2])
        del t316

        # pd_op.full: (xf32) <- ()
        t318 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t319 = paddle._C_ops.assign_value_(
            t318,
            [],
            paddle.float32,
            [float("0.988571")],
            paddle.framework._current_expected_place(),
        )
        del t318

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t320 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t321 = paddle._C_ops.add(t319, t320)
        del t320

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t322 = paddle._C_ops.floor(t321)
        del t321

        # pd_op.divide: (64x192x28x28xf32) <- (64x192x28x28xf32, xf32)
        t323 = paddle._C_ops.divide(t317, t319)

        # pd_op.multiply: (64x192x28x28xf32) <- (64x192x28x28xf32, 64x1x1x1xf32)
        t324 = paddle._C_ops.multiply(t323, t322)

        # pd_op.add: (64x192x28x28xf32) <- (64x192x28x28xf32, 64x192x28x28xf32)
        t325 = paddle._C_ops.add(t303, t324)

        # pd_op.depthwise_conv2d: (64x192x28x28xf32) <- (64x192x28x28xf32, 192x1x7x7xf32)
        t326 = paddle._C_ops.depthwise_conv2d(
            t325, t44, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t44

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t327 = paddle._C_ops.reshape(t45, t179)
        del t45

        # pd_op.add: (64x192x28x28xf32) <- (64x192x28x28xf32, 1x192x1x1xf32)
        t328 = paddle._C_ops.add(t326, t327)

        # pd_op.transpose: (64x28x28x192xf32) <- (64x192x28x28xf32)
        t329 = paddle._C_ops.transpose(t328, [0, 2, 3, 1])
        del t328

        # pd_op.layer_norm: (64x28x28x192xf32, 64x28x28xf32, 64x28x28xf32) <- (64x28x28x192xf32, 192xf32, 192xf32)
        t330, t331, t332 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t329, t46, t47, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t47, t46

        # pd_op.matmul: (64x28x28x768xf32) <- (64x28x28x192xf32, 192x768xf32)
        t333 = paddle._C_ops.matmul(t330, t48, False, False)
        del t48

        # pd_op.add: (64x28x28x768xf32) <- (64x28x28x768xf32, 768xf32)
        t334 = paddle._C_ops.add(t333, t49)
        del t49

        # pd_op.gelu: (64x28x28x768xf32) <- (64x28x28x768xf32)
        t335 = paddle._C_ops.gelu(t334, False)

        # pd_op.matmul: (64x28x28x192xf32) <- (64x28x28x768xf32, 768x192xf32)
        t336 = paddle._C_ops.matmul(t335, t50, False, False)
        del t50

        # pd_op.add: (64x28x28x192xf32) <- (64x28x28x192xf32, 192xf32)
        t337 = paddle._C_ops.add(t336, t51)
        del t51

        # pd_op.multiply: (64x28x28x192xf32) <- (192xf32, 64x28x28x192xf32)
        t338 = paddle._C_ops.multiply(input10, t337)
        del input10

        # pd_op.transpose: (64x192x28x28xf32) <- (64x28x28x192xf32)
        t339 = paddle._C_ops.transpose(t338, [0, 3, 1, 2])
        del t338

        # pd_op.full: (xf32) <- ()
        t340 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t341 = paddle._C_ops.assign_value_(
            t340,
            [],
            paddle.float32,
            [float("0.985714")],
            paddle.framework._current_expected_place(),
        )
        del t340

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t342 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t343 = paddle._C_ops.add(t341, t342)
        del t342

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t344 = paddle._C_ops.floor(t343)
        del t343

        # pd_op.divide: (64x192x28x28xf32) <- (64x192x28x28xf32, xf32)
        t345 = paddle._C_ops.divide(t339, t341)

        # pd_op.multiply: (64x192x28x28xf32) <- (64x192x28x28xf32, 64x1x1x1xf32)
        t346 = paddle._C_ops.multiply(t345, t344)

        # pd_op.add: (64x192x28x28xf32) <- (64x192x28x28xf32, 64x192x28x28xf32)
        t347 = paddle._C_ops.add(t325, t346)

        # pd_op.mean: (64x1x28x28xf32) <- (64x192x28x28xf32)
        t348 = paddle._C_ops.mean(t347, [1], True)

        # pd_op.subtract: (64x192x28x28xf32) <- (64x192x28x28xf32, 64x1x28x28xf32)
        t349 = paddle._C_ops.subtract(t347, t348)

        # pd_op.pow: (64x192x28x28xf32) <- (64x192x28x28xf32)
        t350 = paddle._C_ops.pow(t349, float("2"))

        # pd_op.mean: (64x1x28x28xf32) <- (64x192x28x28xf32)
        t351 = paddle._C_ops.mean(t350, [1], True)

        # pd_op.subtract: (64x192x28x28xf32) <- (64x192x28x28xf32, 64x1x28x28xf32)
        t352 = paddle._C_ops.subtract(t347, t348)

        # pd_op.scale: (64x1x28x28xf32) <- (64x1x28x28xf32, 1xf32)
        t353 = paddle._C_ops.scale(t351, t187, float("1e-06"), True)
        del t351

        # pd_op.sqrt: (64x1x28x28xf32) <- (64x1x28x28xf32)
        t354 = paddle._C_ops.sqrt(t353)
        del t353

        # pd_op.divide: (64x192x28x28xf32) <- (64x192x28x28xf32, 64x1x28x28xf32)
        t355 = paddle._C_ops.divide(t352, t354)

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        t356 = paddle._C_ops.unsqueeze(input11, t194)
        del input11

        # pd_op.multiply: (64x192x28x28xf32) <- (192x1x1xf32, 64x192x28x28xf32)
        t357 = paddle._C_ops.multiply(t356, t355)

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        t358 = paddle._C_ops.unsqueeze(input12, t194)
        del input12

        # pd_op.add: (64x192x28x28xf32) <- (64x192x28x28xf32, 192x1x1xf32)
        t359 = paddle._C_ops.add(t357, t358)

        # pd_op.conv2d: (64x384x14x14xf32) <- (64x192x28x28xf32, 384x192x2x2xf32)
        t360 = paddle._C_ops.conv2d(
            t359, t52, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t52

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t361 = paddle._C_ops.reshape(t53, t179)
        del t53

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t362 = paddle._C_ops.add(t360, t361)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t363 = paddle._C_ops.depthwise_conv2d(
            t362, t54, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t54

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t364 = paddle._C_ops.reshape(t55, t179)
        del t55

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t365 = paddle._C_ops.add(t363, t364)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t366 = paddle._C_ops.transpose(t365, [0, 2, 3, 1])
        del t365

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t367, t368, t369 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t366, t56, t57, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t57, t56

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t370 = paddle._C_ops.matmul(t367, t58, False, False)
        del t58

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t371 = paddle._C_ops.add(t370, t59)
        del t59

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t372 = paddle._C_ops.gelu(t371, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t373 = paddle._C_ops.matmul(t372, t60, False, False)
        del t60

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t374 = paddle._C_ops.add(t373, t61)
        del t61

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t375 = paddle._C_ops.multiply(input13, t374)
        del input13

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t376 = paddle._C_ops.transpose(t375, [0, 3, 1, 2])
        del t375

        # pd_op.full: (xf32) <- ()
        t377 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t378 = paddle._C_ops.assign_value_(
            t377,
            [],
            paddle.float32,
            [float("0.982857")],
            paddle.framework._current_expected_place(),
        )
        del t377

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t379 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t380 = paddle._C_ops.add(t378, t379)
        del t379

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t381 = paddle._C_ops.floor(t380)
        del t380

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t382 = paddle._C_ops.divide(t376, t378)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t383 = paddle._C_ops.multiply(t382, t381)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t384 = paddle._C_ops.add(t362, t383)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t385 = paddle._C_ops.depthwise_conv2d(
            t384, t62, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t62

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t386 = paddle._C_ops.reshape(t63, t179)
        del t63

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t387 = paddle._C_ops.add(t385, t386)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t388 = paddle._C_ops.transpose(t387, [0, 2, 3, 1])
        del t387

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t389, t390, t391 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t388, t64, t65, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t65, t64

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t392 = paddle._C_ops.matmul(t389, t66, False, False)
        del t66

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t393 = paddle._C_ops.add(t392, t67)
        del t67

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t394 = paddle._C_ops.gelu(t393, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t395 = paddle._C_ops.matmul(t394, t68, False, False)
        del t68

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t396 = paddle._C_ops.add(t395, t69)
        del t69

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t397 = paddle._C_ops.multiply(input14, t396)
        del input14

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t398 = paddle._C_ops.transpose(t397, [0, 3, 1, 2])
        del t397

        # pd_op.full: (xf32) <- ()
        t399 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t400 = paddle._C_ops.assign_value_(
            t399,
            [],
            paddle.float32,
            [float("0.98")],
            paddle.framework._current_expected_place(),
        )
        del t399

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t401 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t402 = paddle._C_ops.add(t400, t401)
        del t401

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t403 = paddle._C_ops.floor(t402)
        del t402

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t404 = paddle._C_ops.divide(t398, t400)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t405 = paddle._C_ops.multiply(t404, t403)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t406 = paddle._C_ops.add(t384, t405)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t407 = paddle._C_ops.depthwise_conv2d(
            t406, t70, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t70

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t408 = paddle._C_ops.reshape(t71, t179)
        del t71

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t409 = paddle._C_ops.add(t407, t408)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t410 = paddle._C_ops.transpose(t409, [0, 2, 3, 1])
        del t409

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t411, t412, t413 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t410, t72, t73, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t73, t72

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t414 = paddle._C_ops.matmul(t411, t74, False, False)
        del t74

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t415 = paddle._C_ops.add(t414, t75)
        del t75

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t416 = paddle._C_ops.gelu(t415, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t417 = paddle._C_ops.matmul(t416, t76, False, False)
        del t76

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t418 = paddle._C_ops.add(t417, t77)
        del t77

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t419 = paddle._C_ops.multiply(input15, t418)
        del input15

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t420 = paddle._C_ops.transpose(t419, [0, 3, 1, 2])
        del t419

        # pd_op.full: (xf32) <- ()
        t421 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t422 = paddle._C_ops.assign_value_(
            t421,
            [],
            paddle.float32,
            [float("0.977143")],
            paddle.framework._current_expected_place(),
        )
        del t421

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t423 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t424 = paddle._C_ops.add(t422, t423)
        del t423

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t425 = paddle._C_ops.floor(t424)
        del t424

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t426 = paddle._C_ops.divide(t420, t422)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t427 = paddle._C_ops.multiply(t426, t425)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t428 = paddle._C_ops.add(t406, t427)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t429 = paddle._C_ops.depthwise_conv2d(
            t428, t78, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t78

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t430 = paddle._C_ops.reshape(t79, t179)
        del t79

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t431 = paddle._C_ops.add(t429, t430)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t432 = paddle._C_ops.transpose(t431, [0, 2, 3, 1])
        del t431

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t433, t434, t435 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t432, t80, t81, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t81, t80

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t436 = paddle._C_ops.matmul(t433, t82, False, False)
        del t82

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t437 = paddle._C_ops.add(t436, t83)
        del t83

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t438 = paddle._C_ops.gelu(t437, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t439 = paddle._C_ops.matmul(t438, t84, False, False)
        del t84

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t440 = paddle._C_ops.add(t439, t85)
        del t85

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t441 = paddle._C_ops.multiply(input16, t440)
        del input16

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t442 = paddle._C_ops.transpose(t441, [0, 3, 1, 2])
        del t441

        # pd_op.full: (xf32) <- ()
        t443 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t444 = paddle._C_ops.assign_value_(
            t443,
            [],
            paddle.float32,
            [float("0.974286")],
            paddle.framework._current_expected_place(),
        )
        del t443

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t445 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t446 = paddle._C_ops.add(t444, t445)
        del t445

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t447 = paddle._C_ops.floor(t446)
        del t446

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t448 = paddle._C_ops.divide(t442, t444)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t449 = paddle._C_ops.multiply(t448, t447)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t450 = paddle._C_ops.add(t428, t449)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t451 = paddle._C_ops.depthwise_conv2d(
            t450, t86, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t86

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t452 = paddle._C_ops.reshape(t87, t179)
        del t87

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t453 = paddle._C_ops.add(t451, t452)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t454 = paddle._C_ops.transpose(t453, [0, 2, 3, 1])
        del t453

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t455, t456, t457 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t454, t88, t89, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t89, t88

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t458 = paddle._C_ops.matmul(t455, t90, False, False)
        del t90

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t459 = paddle._C_ops.add(t458, t91)
        del t91

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t460 = paddle._C_ops.gelu(t459, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t461 = paddle._C_ops.matmul(t460, t92, False, False)
        del t92

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t462 = paddle._C_ops.add(t461, t93)
        del t93

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t463 = paddle._C_ops.multiply(input17, t462)
        del input17

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t464 = paddle._C_ops.transpose(t463, [0, 3, 1, 2])
        del t463

        # pd_op.full: (xf32) <- ()
        t465 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t466 = paddle._C_ops.assign_value_(
            t465,
            [],
            paddle.float32,
            [float("0.971429")],
            paddle.framework._current_expected_place(),
        )
        del t465

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t467 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t468 = paddle._C_ops.add(t466, t467)
        del t467

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t469 = paddle._C_ops.floor(t468)
        del t468

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t470 = paddle._C_ops.divide(t464, t466)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t471 = paddle._C_ops.multiply(t470, t469)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t472 = paddle._C_ops.add(t450, t471)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t473 = paddle._C_ops.depthwise_conv2d(
            t472, t94, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t94

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t474 = paddle._C_ops.reshape(t95, t179)
        del t95

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t475 = paddle._C_ops.add(t473, t474)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t476 = paddle._C_ops.transpose(t475, [0, 2, 3, 1])
        del t475

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t477, t478, t479 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t476, t96, t97, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t97, t96

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t480 = paddle._C_ops.matmul(t477, t98, False, False)
        del t98

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t481 = paddle._C_ops.add(t480, t99)
        del t99

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t482 = paddle._C_ops.gelu(t481, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t483 = paddle._C_ops.matmul(t482, t100, False, False)
        del t100

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t484 = paddle._C_ops.add(t483, t101)
        del t101

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t485 = paddle._C_ops.multiply(input18, t484)
        del input18

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t486 = paddle._C_ops.transpose(t485, [0, 3, 1, 2])
        del t485

        # pd_op.full: (xf32) <- ()
        t487 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t488 = paddle._C_ops.assign_value_(
            t487,
            [],
            paddle.float32,
            [float("0.968571")],
            paddle.framework._current_expected_place(),
        )
        del t487

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t489 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t490 = paddle._C_ops.add(t488, t489)
        del t489

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t491 = paddle._C_ops.floor(t490)
        del t490

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t492 = paddle._C_ops.divide(t486, t488)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t493 = paddle._C_ops.multiply(t492, t491)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t494 = paddle._C_ops.add(t472, t493)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t495 = paddle._C_ops.depthwise_conv2d(
            t494, t102, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t102

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t496 = paddle._C_ops.reshape(t103, t179)
        del t103

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t497 = paddle._C_ops.add(t495, t496)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t498 = paddle._C_ops.transpose(t497, [0, 2, 3, 1])
        del t497

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t499, t500, t501 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t498, t104, t105, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t105, t104

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t502 = paddle._C_ops.matmul(t499, t106, False, False)
        del t106

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t503 = paddle._C_ops.add(t502, t107)
        del t107

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t504 = paddle._C_ops.gelu(t503, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t505 = paddle._C_ops.matmul(t504, t108, False, False)
        del t108

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t506 = paddle._C_ops.add(t505, t109)
        del t109

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t507 = paddle._C_ops.multiply(input19, t506)
        del input19

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t508 = paddle._C_ops.transpose(t507, [0, 3, 1, 2])
        del t507

        # pd_op.full: (xf32) <- ()
        t509 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t510 = paddle._C_ops.assign_value_(
            t509,
            [],
            paddle.float32,
            [float("0.965714")],
            paddle.framework._current_expected_place(),
        )
        del t509

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t511 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t512 = paddle._C_ops.add(t510, t511)
        del t511

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t513 = paddle._C_ops.floor(t512)
        del t512

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t514 = paddle._C_ops.divide(t508, t510)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t515 = paddle._C_ops.multiply(t514, t513)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t516 = paddle._C_ops.add(t494, t515)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t517 = paddle._C_ops.depthwise_conv2d(
            t516, t110, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t110

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t518 = paddle._C_ops.reshape(t111, t179)
        del t111

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t519 = paddle._C_ops.add(t517, t518)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t520 = paddle._C_ops.transpose(t519, [0, 2, 3, 1])
        del t519

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t521, t522, t523 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t520, t112, t113, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t113, t112

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t524 = paddle._C_ops.matmul(t521, t114, False, False)
        del t114

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t525 = paddle._C_ops.add(t524, t115)
        del t115

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t526 = paddle._C_ops.gelu(t525, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t527 = paddle._C_ops.matmul(t526, t116, False, False)
        del t116

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t528 = paddle._C_ops.add(t527, t117)
        del t117

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t529 = paddle._C_ops.multiply(input20, t528)
        del input20

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t530 = paddle._C_ops.transpose(t529, [0, 3, 1, 2])
        del t529

        # pd_op.full: (xf32) <- ()
        t531 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t532 = paddle._C_ops.assign_value_(
            t531,
            [],
            paddle.float32,
            [float("0.962857")],
            paddle.framework._current_expected_place(),
        )
        del t531

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t533 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t534 = paddle._C_ops.add(t532, t533)
        del t533

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t535 = paddle._C_ops.floor(t534)
        del t534

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t536 = paddle._C_ops.divide(t530, t532)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t537 = paddle._C_ops.multiply(t536, t535)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t538 = paddle._C_ops.add(t516, t537)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t539 = paddle._C_ops.depthwise_conv2d(
            t538, t118, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t118

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t540 = paddle._C_ops.reshape(t119, t179)
        del t119

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t541 = paddle._C_ops.add(t539, t540)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t542 = paddle._C_ops.transpose(t541, [0, 2, 3, 1])
        del t541

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t543, t544, t545 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t542, t120, t121, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t121, t120

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t546 = paddle._C_ops.matmul(t543, t122, False, False)
        del t122

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t547 = paddle._C_ops.add(t546, t123)
        del t123

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t548 = paddle._C_ops.gelu(t547, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t549 = paddle._C_ops.matmul(t548, t124, False, False)
        del t124

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t550 = paddle._C_ops.add(t549, t125)
        del t125

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t551 = paddle._C_ops.multiply(input21, t550)
        del input21

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t552 = paddle._C_ops.transpose(t551, [0, 3, 1, 2])
        del t551

        # pd_op.full: (xf32) <- ()
        t553 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t554 = paddle._C_ops.assign_value_(
            t553,
            [],
            paddle.float32,
            [float("0.96")],
            paddle.framework._current_expected_place(),
        )
        del t553

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t555 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t556 = paddle._C_ops.add(t554, t555)
        del t555

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t557 = paddle._C_ops.floor(t556)
        del t556

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t558 = paddle._C_ops.divide(t552, t554)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t559 = paddle._C_ops.multiply(t558, t557)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t560 = paddle._C_ops.add(t538, t559)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t561 = paddle._C_ops.depthwise_conv2d(
            t560, t126, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t126

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t562 = paddle._C_ops.reshape(t127, t179)
        del t127

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t563 = paddle._C_ops.add(t561, t562)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t564 = paddle._C_ops.transpose(t563, [0, 2, 3, 1])
        del t563

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t565, t566, t567 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t564, t128, t129, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t129, t128

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t568 = paddle._C_ops.matmul(t565, t130, False, False)
        del t130

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t569 = paddle._C_ops.add(t568, t131)
        del t131

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t570 = paddle._C_ops.gelu(t569, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t571 = paddle._C_ops.matmul(t570, t132, False, False)
        del t132

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t572 = paddle._C_ops.add(t571, t133)
        del t133

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t573 = paddle._C_ops.multiply(input22, t572)
        del input22

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t574 = paddle._C_ops.transpose(t573, [0, 3, 1, 2])
        del t573

        # pd_op.full: (xf32) <- ()
        t575 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t576 = paddle._C_ops.assign_value_(
            t575,
            [],
            paddle.float32,
            [float("0.957143")],
            paddle.framework._current_expected_place(),
        )
        del t575

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t577 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t578 = paddle._C_ops.add(t576, t577)
        del t577

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t579 = paddle._C_ops.floor(t578)
        del t578

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t580 = paddle._C_ops.divide(t574, t576)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t581 = paddle._C_ops.multiply(t580, t579)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t582 = paddle._C_ops.add(t560, t581)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t583 = paddle._C_ops.depthwise_conv2d(
            t582, t134, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t134

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t584 = paddle._C_ops.reshape(t135, t179)
        del t135

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t585 = paddle._C_ops.add(t583, t584)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t586 = paddle._C_ops.transpose(t585, [0, 2, 3, 1])
        del t585

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t587, t588, t589 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t586, t136, t137, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t137, t136

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t590 = paddle._C_ops.matmul(t587, t138, False, False)
        del t138

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t591 = paddle._C_ops.add(t590, t139)
        del t139

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t592 = paddle._C_ops.gelu(t591, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t593 = paddle._C_ops.matmul(t592, t140, False, False)
        del t140

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t594 = paddle._C_ops.add(t593, t141)
        del t141

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t595 = paddle._C_ops.multiply(input23, t594)
        del input23

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t596 = paddle._C_ops.transpose(t595, [0, 3, 1, 2])
        del t595

        # pd_op.full: (xf32) <- ()
        t597 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t598 = paddle._C_ops.assign_value_(
            t597,
            [],
            paddle.float32,
            [float("0.954286")],
            paddle.framework._current_expected_place(),
        )
        del t597

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t599 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t600 = paddle._C_ops.add(t598, t599)
        del t599

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t601 = paddle._C_ops.floor(t600)
        del t600

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t602 = paddle._C_ops.divide(t596, t598)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t603 = paddle._C_ops.multiply(t602, t601)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t604 = paddle._C_ops.add(t582, t603)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t605 = paddle._C_ops.depthwise_conv2d(
            t604, t142, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t142

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t606 = paddle._C_ops.reshape(t143, t179)
        del t143

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t607 = paddle._C_ops.add(t605, t606)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t608 = paddle._C_ops.transpose(t607, [0, 2, 3, 1])
        del t607

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t609, t610, t611 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t608, t144, t145, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t145, t144

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t612 = paddle._C_ops.matmul(t609, t146, False, False)
        del t146

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t613 = paddle._C_ops.add(t612, t147)
        del t147

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t614 = paddle._C_ops.gelu(t613, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t615 = paddle._C_ops.matmul(t614, t148, False, False)
        del t148

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t616 = paddle._C_ops.add(t615, t149)
        del t149

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t617 = paddle._C_ops.multiply(input24, t616)
        del input24

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t618 = paddle._C_ops.transpose(t617, [0, 3, 1, 2])
        del t617

        # pd_op.full: (xf32) <- ()
        t619 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t620 = paddle._C_ops.assign_value_(
            t619,
            [],
            paddle.float32,
            [float("0.951429")],
            paddle.framework._current_expected_place(),
        )
        del t619

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t621 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t622 = paddle._C_ops.add(t620, t621)
        del t621

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t623 = paddle._C_ops.floor(t622)
        del t622

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t624 = paddle._C_ops.divide(t618, t620)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t625 = paddle._C_ops.multiply(t624, t623)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t626 = paddle._C_ops.add(t604, t625)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t627 = paddle._C_ops.depthwise_conv2d(
            t626, t150, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t150

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t628 = paddle._C_ops.reshape(t151, t179)
        del t151

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t629 = paddle._C_ops.add(t627, t628)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t630 = paddle._C_ops.transpose(t629, [0, 2, 3, 1])
        del t629

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t631, t632, t633 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t630, t152, t153, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t153, t152

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t634 = paddle._C_ops.matmul(t631, t154, False, False)
        del t154

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t635 = paddle._C_ops.add(t634, t155)
        del t155

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t636 = paddle._C_ops.gelu(t635, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t637 = paddle._C_ops.matmul(t636, t156, False, False)
        del t156

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t638 = paddle._C_ops.add(t637, t157)
        del t157

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t639 = paddle._C_ops.multiply(input25, t638)
        del input25

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t640 = paddle._C_ops.transpose(t639, [0, 3, 1, 2])
        del t639

        # pd_op.full: (xf32) <- ()
        t641 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t642 = paddle._C_ops.assign_value_(
            t641,
            [],
            paddle.float32,
            [float("0.948571")],
            paddle.framework._current_expected_place(),
        )
        del t641

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t643 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t644 = paddle._C_ops.add(t642, t643)
        del t643

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t645 = paddle._C_ops.floor(t644)
        del t644

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t646 = paddle._C_ops.divide(t640, t642)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t647 = paddle._C_ops.multiply(t646, t645)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t648 = paddle._C_ops.add(t626, t647)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t649 = paddle._C_ops.depthwise_conv2d(
            t648, t158, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t158

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t650 = paddle._C_ops.reshape(t159, t179)
        del t159

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t651 = paddle._C_ops.add(t649, t650)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t652 = paddle._C_ops.transpose(t651, [0, 2, 3, 1])
        del t651

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t653, t654, t655 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t652, t160, t161, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t161, t160

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t656 = paddle._C_ops.matmul(t653, t162, False, False)
        del t162

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t657 = paddle._C_ops.add(t656, t163)
        del t163

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t658 = paddle._C_ops.gelu(t657, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t659 = paddle._C_ops.matmul(t658, t164, False, False)
        del t164

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t660 = paddle._C_ops.add(t659, t165)
        del t165

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t661 = paddle._C_ops.multiply(input26, t660)
        del input26

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t662 = paddle._C_ops.transpose(t661, [0, 3, 1, 2])
        del t661

        # pd_op.full: (xf32) <- ()
        t663 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t664 = paddle._C_ops.assign_value_(
            t663,
            [],
            paddle.float32,
            [float("0.945714")],
            paddle.framework._current_expected_place(),
        )
        del t663

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t665 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t666 = paddle._C_ops.add(t664, t665)
        del t665

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t667 = paddle._C_ops.floor(t666)
        del t666

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t668 = paddle._C_ops.divide(t662, t664)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t669 = paddle._C_ops.multiply(t668, t667)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t670 = paddle._C_ops.add(t648, t669)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t671 = paddle._C_ops.depthwise_conv2d(
            t670, t166, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t166

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t672 = paddle._C_ops.reshape(t167, t179)
        del t167

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t673 = paddle._C_ops.add(t671, t672)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t674 = paddle._C_ops.transpose(t673, [0, 2, 3, 1])
        del t673

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t675, t676, t677 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t674, t168, t169, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t169, t168

        # pd_op.matmul: (64x14x14x1536xf32) <- (64x14x14x384xf32, 384x1536xf32)
        t678 = paddle._C_ops.matmul(t675, t170, False, False)
        del t170

        # pd_op.add: (64x14x14x1536xf32) <- (64x14x14x1536xf32, 1536xf32)
        t679 = paddle._C_ops.add(t678, t171)
        del t171

        # pd_op.gelu: (64x14x14x1536xf32) <- (64x14x14x1536xf32)
        t680 = paddle._C_ops.gelu(t679, False)

        # pd_op.matmul: (64x14x14x384xf32) <- (64x14x14x1536xf32, 1536x384xf32)
        t681 = paddle._C_ops.matmul(t680, t172, False, False)
        del t172

        # pd_op.add: (64x14x14x384xf32) <- (64x14x14x384xf32, 384xf32)
        t682 = paddle._C_ops.add(t681, t173)
        del t173

        # pd_op.multiply: (64x14x14x384xf32) <- (384xf32, 64x14x14x384xf32)
        t683 = paddle._C_ops.multiply(input27, t682)
        del input27

        # pd_op.transpose: (64x384x14x14xf32) <- (64x14x14x384xf32)
        t684 = paddle._C_ops.transpose(t683, [0, 3, 1, 2])
        del t683

        # pd_op.full: (xf32) <- ()
        t685 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t686 = paddle._C_ops.assign_value_(
            t685,
            [],
            paddle.float32,
            [float("0.942857")],
            paddle.framework._current_expected_place(),
        )
        del t685

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t687 = paddle._C_ops.uniform(
            t237,
            paddle.float32,
            t238,
            t187,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        t688 = paddle._C_ops.add(t686, t687)
        del t687

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        t689 = paddle._C_ops.floor(t688)
        del t688

        # pd_op.divide: (64x384x14x14xf32) <- (64x384x14x14xf32, xf32)
        t690 = paddle._C_ops.divide(t684, t686)

        # pd_op.multiply: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x1x1x1xf32)
        t691 = paddle._C_ops.multiply(t690, t689)

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 64x384x14x14xf32)
        t692 = paddle._C_ops.add(t670, t691)

        # pd_op.depthwise_conv2d: (64x384x14x14xf32) <- (64x384x14x14xf32, 384x1x7x7xf32)
        t693 = paddle._C_ops.depthwise_conv2d(
            t692, t174, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t174

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t694 = paddle._C_ops.reshape(t175, t179)
        del t175

        # pd_op.add: (64x384x14x14xf32) <- (64x384x14x14xf32, 1x384x1x1xf32)
        t695 = paddle._C_ops.add(t693, t694)

        # pd_op.transpose: (64x14x14x384xf32) <- (64x384x14x14xf32)
        t696 = paddle._C_ops.transpose(t695, [0, 2, 3, 1])
        del t695

        # pd_op.layer_norm: (64x14x14x384xf32, 64x14x14xf32, 64x14x14xf32) <- (64x14x14x384xf32, 384xf32, 384xf32)
        t697, t698, t699 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t696, t176, t177, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t177, t176

        return (
            t178,
            t179,
            t180,
            t181,
            t182,
            t183,
            t184,
            t185,
            t187,
            t188,
            t189,
            t190,
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
            t209,
            t210,
            t211,
            t212,
            t213,
            t214,
            t215,
            t216,
            t217,
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
            t231,
            t232,
            t234,
            t236,
            t237,
            t238,
            t241,
            t242,
            t243,
            t244,
            t245,
            t246,
            t248,
            t249,
            t250,
            t251,
            t252,
            t253,
            t254,
            t255,
            t256,
            t258,
            t260,
            t263,
            t264,
            t265,
            t266,
            t267,
            t268,
            t269,
            t271,
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
            t285,
            t286,
            t287,
            t288,
            t289,
            t290,
            t291,
            t292,
            t293,
            t295,
            t297,
            t300,
            t301,
            t302,
            t303,
            t304,
            t305,
            t307,
            t308,
            t309,
            t310,
            t311,
            t312,
            t313,
            t314,
            t315,
            t317,
            t319,
            t322,
            t323,
            t324,
            t325,
            t326,
            t327,
            t329,
            t330,
            t331,
            t332,
            t333,
            t334,
            t335,
            t336,
            t337,
            t339,
            t341,
            t344,
            t345,
            t346,
            t347,
            t348,
            t349,
            t350,
            t352,
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
            t369,
            t370,
            t371,
            t372,
            t373,
            t374,
            t376,
            t378,
            t381,
            t382,
            t383,
            t384,
            t385,
            t386,
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
            t400,
            t403,
            t404,
            t405,
            t406,
            t407,
            t408,
            t410,
            t411,
            t412,
            t413,
            t414,
            t415,
            t416,
            t417,
            t418,
            t420,
            t422,
            t425,
            t426,
            t427,
            t428,
            t429,
            t430,
            t432,
            t433,
            t434,
            t435,
            t436,
            t437,
            t438,
            t439,
            t440,
            t442,
            t444,
            t447,
            t448,
            t449,
            t450,
            t451,
            t452,
            t454,
            t455,
            t456,
            t457,
            t458,
            t459,
            t460,
            t461,
            t462,
            t464,
            t466,
            t469,
            t470,
            t471,
            t472,
            t473,
            t474,
            t476,
            t477,
            t478,
            t479,
            t480,
            t481,
            t482,
            t483,
            t484,
            t486,
            t488,
            t491,
            t492,
            t493,
            t494,
            t495,
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
            t508,
            t510,
            t513,
            t514,
            t515,
            t516,
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
            t528,
            t530,
            t532,
            t535,
            t536,
            t537,
            t538,
            t539,
            t540,
            t542,
            t543,
            t544,
            t545,
            t546,
            t547,
            t548,
            t549,
            t550,
            t552,
            t554,
            t557,
            t558,
            t559,
            t560,
            t561,
            t562,
            t564,
            t565,
            t566,
            t567,
            t568,
            t569,
            t570,
            t571,
            t572,
            t574,
            t576,
            t579,
            t580,
            t581,
            t582,
            t583,
            t584,
            t586,
            t587,
            t588,
            t589,
            t590,
            t591,
            t592,
            t593,
            t594,
            t596,
            t598,
            t601,
            t602,
            t603,
            t604,
            t605,
            t606,
            t608,
            t609,
            t610,
            t611,
            t612,
            t613,
            t614,
            t615,
            t616,
            t618,
            t620,
            t623,
            t624,
            t625,
            t626,
            t627,
            t628,
            t630,
            t631,
            t632,
            t633,
            t634,
            t635,
            t636,
            t637,
            t638,
            t640,
            t642,
            t645,
            t646,
            t647,
            t648,
            t649,
            t650,
            t652,
            t653,
            t654,
            t655,
            t656,
            t657,
            t658,
            t659,
            t660,
            t662,
            t664,
            t667,
            t668,
            t669,
            t670,
            t671,
            t672,
            t674,
            t675,
            t676,
            t677,
            t678,
            t679,
            t680,
            t681,
            t682,
            t684,
            t686,
            t689,
            t690,
            t691,
            t692,
            t693,
            t694,
            t696,
            t697,
            t698,
            t699,
        )
