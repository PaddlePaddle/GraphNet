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
        input22: paddle.Tensor,
        input23: paddle.Tensor,
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
        input24: paddle.Tensor,
        t136: paddle.Tensor,
        t137: paddle.Tensor,
        t138: paddle.Tensor,
        t139: paddle.Tensor,
        t140: paddle.Tensor,
        t141: paddle.Tensor,
        t142: paddle.Tensor,
        t143: paddle.Tensor,
        input25: paddle.Tensor,
        t144: paddle.Tensor,
        t145: paddle.Tensor,
        t146: paddle.Tensor,
        t147: paddle.Tensor,
        t148: paddle.Tensor,
        t149: paddle.Tensor,
        t150: paddle.Tensor,
        t151: paddle.Tensor,
        input26: paddle.Tensor,
        t152: paddle.Tensor,
        t153: paddle.Tensor,
        t154: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x3x224x224xf32, 96x3x4x4xf32)
        t155 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t156 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t157 = paddle._C_ops.reshape(t1, t156)
        del t1

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        t158 = paddle._C_ops.add(t155, t157)
        del t155, t157

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x96x56x56xf32)
        t159 = paddle._C_ops.mean(t158, [1], True)

        # pd_op.subtract: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x1x56x56xf32)
        t160 = paddle._C_ops.subtract(t158, t159)
        del t158, t159

        # pd_op.pow: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t161 = paddle._C_ops.pow(t160, float("2"))

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x96x56x56xf32)
        t162 = paddle._C_ops.mean(t161, [1], True)
        del t161

        # pd_op.full: (1xf32) <- ()
        t163 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1x56x56xf32) <- (-1x1x56x56xf32, 1xf32)
        t164 = paddle._C_ops.scale(t162, t163, float("1e-06"), True)
        del t162

        # pd_op.sqrt: (-1x1x56x56xf32) <- (-1x1x56x56xf32)
        t165 = paddle._C_ops.sqrt(t164)
        del t164

        # pd_op.divide: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x1x56x56xf32)
        t166 = paddle._C_ops.divide(t160, t165)
        del t165, t160

        # pd_op.full_int_array: (2xi64) <- ()
        t167 = [1, 2]

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        t168 = paddle._C_ops.unsqueeze(input1, t167)
        del input1

        # pd_op.multiply: (-1x96x56x56xf32) <- (96x1x1xf32, -1x96x56x56xf32)
        t169 = paddle._C_ops.multiply(t168, t166)
        del t166, t168

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        t170 = paddle._C_ops.unsqueeze(input2, t167)
        del input2

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x1xf32)
        t171 = paddle._C_ops.add(t169, t170)
        del t169, t170

        # pd_op.depthwise_conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x7x7xf32)
        t172 = paddle._C_ops.depthwise_conv2d(
            t171, t2, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t2

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t173 = paddle._C_ops.reshape(t3, t156)
        del t3

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        t174 = paddle._C_ops.add(t172, t173)
        del t172, t173

        # pd_op.transpose: (-1x56x56x96xf32) <- (-1x96x56x56xf32)
        t175 = paddle._C_ops.transpose(t174, [0, 2, 3, 1])
        del t174

        # pd_op.layer_norm: (-1x56x56x96xf32, -1x56x56xf32, -1x56x56xf32) <- (-1x56x56x96xf32, 96xf32, 96xf32)
        t176, t177, t178 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t175, t4, t5, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t5, t4, t175

        # pd_op.matmul: (-1x56x56x384xf32) <- (-1x56x56x96xf32, 96x384xf32)
        t179 = paddle._C_ops.matmul(t176, t6, False, False)
        del t176, t6

        # pd_op.add: (-1x56x56x384xf32) <- (-1x56x56x384xf32, 384xf32)
        t180 = paddle._C_ops.add(t179, t7)
        del t179, t7

        # pd_op.gelu: (-1x56x56x384xf32) <- (-1x56x56x384xf32)
        t181 = paddle._C_ops.gelu(t180, False)
        del t180

        # pd_op.matmul: (-1x56x56x96xf32) <- (-1x56x56x384xf32, 384x96xf32)
        t182 = paddle._C_ops.matmul(t181, t8, False, False)
        del t181, t8

        # pd_op.add: (-1x56x56x96xf32) <- (-1x56x56x96xf32, 96xf32)
        t183 = paddle._C_ops.add(t182, t9)
        del t182, t9

        # pd_op.multiply: (-1x56x56x96xf32) <- (96xf32, -1x56x56x96xf32)
        t184 = paddle._C_ops.multiply(input3, t183)
        del t183, input3

        # pd_op.transpose: (-1x96x56x56xf32) <- (-1x56x56x96xf32)
        t185 = paddle._C_ops.transpose(t184, [0, 3, 1, 2])
        del t184

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        t186 = paddle._C_ops.add(t171, t185)
        del t171, t185

        # pd_op.depthwise_conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x7x7xf32)
        t187 = paddle._C_ops.depthwise_conv2d(
            t186, t10, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t10

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t188 = paddle._C_ops.reshape(t11, t156)
        del t11

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        t189 = paddle._C_ops.add(t187, t188)
        del t187, t188

        # pd_op.transpose: (-1x56x56x96xf32) <- (-1x96x56x56xf32)
        t190 = paddle._C_ops.transpose(t189, [0, 2, 3, 1])
        del t189

        # pd_op.layer_norm: (-1x56x56x96xf32, -1x56x56xf32, -1x56x56xf32) <- (-1x56x56x96xf32, 96xf32, 96xf32)
        t191, t192, t193 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t190, t12, t13, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t13, t12, t190

        # pd_op.matmul: (-1x56x56x384xf32) <- (-1x56x56x96xf32, 96x384xf32)
        t194 = paddle._C_ops.matmul(t191, t14, False, False)
        del t191, t14

        # pd_op.add: (-1x56x56x384xf32) <- (-1x56x56x384xf32, 384xf32)
        t195 = paddle._C_ops.add(t194, t15)
        del t194, t15

        # pd_op.gelu: (-1x56x56x384xf32) <- (-1x56x56x384xf32)
        t196 = paddle._C_ops.gelu(t195, False)
        del t195

        # pd_op.matmul: (-1x56x56x96xf32) <- (-1x56x56x384xf32, 384x96xf32)
        t197 = paddle._C_ops.matmul(t196, t16, False, False)
        del t196, t16

        # pd_op.add: (-1x56x56x96xf32) <- (-1x56x56x96xf32, 96xf32)
        t198 = paddle._C_ops.add(t197, t17)
        del t197, t17

        # pd_op.multiply: (-1x56x56x96xf32) <- (96xf32, -1x56x56x96xf32)
        t199 = paddle._C_ops.multiply(input4, t198)
        del t198, input4

        # pd_op.transpose: (-1x96x56x56xf32) <- (-1x56x56x96xf32)
        t200 = paddle._C_ops.transpose(t199, [0, 3, 1, 2])
        del t199

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        t201 = paddle._C_ops.add(t186, t200)
        del t186, t200

        # pd_op.depthwise_conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x7x7xf32)
        t202 = paddle._C_ops.depthwise_conv2d(
            t201, t18, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t18

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t203 = paddle._C_ops.reshape(t19, t156)
        del t19

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        t204 = paddle._C_ops.add(t202, t203)
        del t202, t203

        # pd_op.transpose: (-1x56x56x96xf32) <- (-1x96x56x56xf32)
        t205 = paddle._C_ops.transpose(t204, [0, 2, 3, 1])
        del t204

        # pd_op.layer_norm: (-1x56x56x96xf32, -1x56x56xf32, -1x56x56xf32) <- (-1x56x56x96xf32, 96xf32, 96xf32)
        t206, t207, t208 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t205, t20, t21, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t21, t20, t205

        # pd_op.matmul: (-1x56x56x384xf32) <- (-1x56x56x96xf32, 96x384xf32)
        t209 = paddle._C_ops.matmul(t206, t22, False, False)
        del t206, t22

        # pd_op.add: (-1x56x56x384xf32) <- (-1x56x56x384xf32, 384xf32)
        t210 = paddle._C_ops.add(t209, t23)
        del t209, t23

        # pd_op.gelu: (-1x56x56x384xf32) <- (-1x56x56x384xf32)
        t211 = paddle._C_ops.gelu(t210, False)
        del t210

        # pd_op.matmul: (-1x56x56x96xf32) <- (-1x56x56x384xf32, 384x96xf32)
        t212 = paddle._C_ops.matmul(t211, t24, False, False)
        del t211, t24

        # pd_op.add: (-1x56x56x96xf32) <- (-1x56x56x96xf32, 96xf32)
        t213 = paddle._C_ops.add(t212, t25)
        del t212, t25

        # pd_op.multiply: (-1x56x56x96xf32) <- (96xf32, -1x56x56x96xf32)
        t214 = paddle._C_ops.multiply(input5, t213)
        del t213, input5

        # pd_op.transpose: (-1x96x56x56xf32) <- (-1x56x56x96xf32)
        t215 = paddle._C_ops.transpose(t214, [0, 3, 1, 2])
        del t214

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        t216 = paddle._C_ops.add(t201, t215)
        del t201, t215

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x96x56x56xf32)
        t217 = paddle._C_ops.mean(t216, [1], True)

        # pd_op.subtract: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x1x56x56xf32)
        t218 = paddle._C_ops.subtract(t216, t217)
        del t216, t217

        # pd_op.pow: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t219 = paddle._C_ops.pow(t218, float("2"))

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x96x56x56xf32)
        t220 = paddle._C_ops.mean(t219, [1], True)
        del t219

        # pd_op.scale: (-1x1x56x56xf32) <- (-1x1x56x56xf32, 1xf32)
        t221 = paddle._C_ops.scale(t220, t163, float("1e-06"), True)
        del t220

        # pd_op.sqrt: (-1x1x56x56xf32) <- (-1x1x56x56xf32)
        t222 = paddle._C_ops.sqrt(t221)
        del t221

        # pd_op.divide: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x1x56x56xf32)
        t223 = paddle._C_ops.divide(t218, t222)
        del t222, t218

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        t224 = paddle._C_ops.unsqueeze(input6, t167)
        del input6

        # pd_op.multiply: (-1x96x56x56xf32) <- (96x1x1xf32, -1x96x56x56xf32)
        t225 = paddle._C_ops.multiply(t224, t223)
        del t223, t224

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        t226 = paddle._C_ops.unsqueeze(input7, t167)
        del input7

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x1xf32)
        t227 = paddle._C_ops.add(t225, t226)
        del t225, t226

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x96x56x56xf32, 192x96x2x2xf32)
        t228 = paddle._C_ops.conv2d(
            t227, t26, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t227, t26

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t229 = paddle._C_ops.reshape(t27, t156)
        del t27

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        t230 = paddle._C_ops.add(t228, t229)
        del t228, t229

        # pd_op.depthwise_conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x1x7x7xf32)
        t231 = paddle._C_ops.depthwise_conv2d(
            t230, t28, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t28

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t232 = paddle._C_ops.reshape(t29, t156)
        del t29

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        t233 = paddle._C_ops.add(t231, t232)
        del t231, t232

        # pd_op.transpose: (-1x28x28x192xf32) <- (-1x192x28x28xf32)
        t234 = paddle._C_ops.transpose(t233, [0, 2, 3, 1])
        del t233

        # pd_op.layer_norm: (-1x28x28x192xf32, -1x28x28xf32, -1x28x28xf32) <- (-1x28x28x192xf32, 192xf32, 192xf32)
        t235, t236, t237 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t234, t30, t31, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t31, t30, t234

        # pd_op.matmul: (-1x28x28x768xf32) <- (-1x28x28x192xf32, 192x768xf32)
        t238 = paddle._C_ops.matmul(t235, t32, False, False)
        del t235, t32

        # pd_op.add: (-1x28x28x768xf32) <- (-1x28x28x768xf32, 768xf32)
        t239 = paddle._C_ops.add(t238, t33)
        del t238, t33

        # pd_op.gelu: (-1x28x28x768xf32) <- (-1x28x28x768xf32)
        t240 = paddle._C_ops.gelu(t239, False)
        del t239

        # pd_op.matmul: (-1x28x28x192xf32) <- (-1x28x28x768xf32, 768x192xf32)
        t241 = paddle._C_ops.matmul(t240, t34, False, False)
        del t240, t34

        # pd_op.add: (-1x28x28x192xf32) <- (-1x28x28x192xf32, 192xf32)
        t242 = paddle._C_ops.add(t241, t35)
        del t241, t35

        # pd_op.multiply: (-1x28x28x192xf32) <- (192xf32, -1x28x28x192xf32)
        t243 = paddle._C_ops.multiply(input8, t242)
        del t242, input8

        # pd_op.transpose: (-1x192x28x28xf32) <- (-1x28x28x192xf32)
        t244 = paddle._C_ops.transpose(t243, [0, 3, 1, 2])
        del t243

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        t245 = paddle._C_ops.add(t230, t244)
        del t230, t244

        # pd_op.depthwise_conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x1x7x7xf32)
        t246 = paddle._C_ops.depthwise_conv2d(
            t245, t36, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t36

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t247 = paddle._C_ops.reshape(t37, t156)
        del t37

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        t248 = paddle._C_ops.add(t246, t247)
        del t246, t247

        # pd_op.transpose: (-1x28x28x192xf32) <- (-1x192x28x28xf32)
        t249 = paddle._C_ops.transpose(t248, [0, 2, 3, 1])
        del t248

        # pd_op.layer_norm: (-1x28x28x192xf32, -1x28x28xf32, -1x28x28xf32) <- (-1x28x28x192xf32, 192xf32, 192xf32)
        t250, t251, t252 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t249, t38, t39, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t39, t38, t249

        # pd_op.matmul: (-1x28x28x768xf32) <- (-1x28x28x192xf32, 192x768xf32)
        t253 = paddle._C_ops.matmul(t250, t40, False, False)
        del t250, t40

        # pd_op.add: (-1x28x28x768xf32) <- (-1x28x28x768xf32, 768xf32)
        t254 = paddle._C_ops.add(t253, t41)
        del t253, t41

        # pd_op.gelu: (-1x28x28x768xf32) <- (-1x28x28x768xf32)
        t255 = paddle._C_ops.gelu(t254, False)
        del t254

        # pd_op.matmul: (-1x28x28x192xf32) <- (-1x28x28x768xf32, 768x192xf32)
        t256 = paddle._C_ops.matmul(t255, t42, False, False)
        del t255, t42

        # pd_op.add: (-1x28x28x192xf32) <- (-1x28x28x192xf32, 192xf32)
        t257 = paddle._C_ops.add(t256, t43)
        del t256, t43

        # pd_op.multiply: (-1x28x28x192xf32) <- (192xf32, -1x28x28x192xf32)
        t258 = paddle._C_ops.multiply(input9, t257)
        del t257, input9

        # pd_op.transpose: (-1x192x28x28xf32) <- (-1x28x28x192xf32)
        t259 = paddle._C_ops.transpose(t258, [0, 3, 1, 2])
        del t258

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        t260 = paddle._C_ops.add(t245, t259)
        del t245, t259

        # pd_op.depthwise_conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x1x7x7xf32)
        t261 = paddle._C_ops.depthwise_conv2d(
            t260, t44, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t44

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t262 = paddle._C_ops.reshape(t45, t156)
        del t45

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        t263 = paddle._C_ops.add(t261, t262)
        del t261, t262

        # pd_op.transpose: (-1x28x28x192xf32) <- (-1x192x28x28xf32)
        t264 = paddle._C_ops.transpose(t263, [0, 2, 3, 1])
        del t263

        # pd_op.layer_norm: (-1x28x28x192xf32, -1x28x28xf32, -1x28x28xf32) <- (-1x28x28x192xf32, 192xf32, 192xf32)
        t265, t266, t267 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t264, t46, t47, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t47, t46, t264

        # pd_op.matmul: (-1x28x28x768xf32) <- (-1x28x28x192xf32, 192x768xf32)
        t268 = paddle._C_ops.matmul(t265, t48, False, False)
        del t265, t48

        # pd_op.add: (-1x28x28x768xf32) <- (-1x28x28x768xf32, 768xf32)
        t269 = paddle._C_ops.add(t268, t49)
        del t268, t49

        # pd_op.gelu: (-1x28x28x768xf32) <- (-1x28x28x768xf32)
        t270 = paddle._C_ops.gelu(t269, False)
        del t269

        # pd_op.matmul: (-1x28x28x192xf32) <- (-1x28x28x768xf32, 768x192xf32)
        t271 = paddle._C_ops.matmul(t270, t50, False, False)
        del t270, t50

        # pd_op.add: (-1x28x28x192xf32) <- (-1x28x28x192xf32, 192xf32)
        t272 = paddle._C_ops.add(t271, t51)
        del t271, t51

        # pd_op.multiply: (-1x28x28x192xf32) <- (192xf32, -1x28x28x192xf32)
        t273 = paddle._C_ops.multiply(input10, t272)
        del t272, input10

        # pd_op.transpose: (-1x192x28x28xf32) <- (-1x28x28x192xf32)
        t274 = paddle._C_ops.transpose(t273, [0, 3, 1, 2])
        del t273

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        t275 = paddle._C_ops.add(t260, t274)
        del t260, t274

        # pd_op.mean: (-1x1x28x28xf32) <- (-1x192x28x28xf32)
        t276 = paddle._C_ops.mean(t275, [1], True)

        # pd_op.subtract: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x1x28x28xf32)
        t277 = paddle._C_ops.subtract(t275, t276)
        del t275, t276

        # pd_op.pow: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t278 = paddle._C_ops.pow(t277, float("2"))

        # pd_op.mean: (-1x1x28x28xf32) <- (-1x192x28x28xf32)
        t279 = paddle._C_ops.mean(t278, [1], True)
        del t278

        # pd_op.scale: (-1x1x28x28xf32) <- (-1x1x28x28xf32, 1xf32)
        t280 = paddle._C_ops.scale(t279, t163, float("1e-06"), True)
        del t279

        # pd_op.sqrt: (-1x1x28x28xf32) <- (-1x1x28x28xf32)
        t281 = paddle._C_ops.sqrt(t280)
        del t280

        # pd_op.divide: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x1x28x28xf32)
        t282 = paddle._C_ops.divide(t277, t281)
        del t281, t277

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        t283 = paddle._C_ops.unsqueeze(input11, t167)
        del input11

        # pd_op.multiply: (-1x192x28x28xf32) <- (192x1x1xf32, -1x192x28x28xf32)
        t284 = paddle._C_ops.multiply(t283, t282)
        del t282, t283

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        t285 = paddle._C_ops.unsqueeze(input12, t167)
        del input12

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x1x1xf32)
        t286 = paddle._C_ops.add(t284, t285)
        del t284, t285

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x192x28x28xf32, 384x192x2x2xf32)
        t287 = paddle._C_ops.conv2d(
            t286, t52, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t286, t52

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t288 = paddle._C_ops.reshape(t53, t156)
        del t53

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t289 = paddle._C_ops.add(t287, t288)
        del t287, t288

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        t290 = paddle._C_ops.depthwise_conv2d(
            t289, t54, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t54

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t291 = paddle._C_ops.reshape(t55, t156)
        del t55

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t292 = paddle._C_ops.add(t290, t291)
        del t290, t291

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        t293 = paddle._C_ops.transpose(t292, [0, 2, 3, 1])
        del t292

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        t294, t295, t296 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t293, t56, t57, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t57, t56, t293

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        t297 = paddle._C_ops.matmul(t294, t58, False, False)
        del t294, t58

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        t298 = paddle._C_ops.add(t297, t59)
        del t297, t59

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        t299 = paddle._C_ops.gelu(t298, False)
        del t298

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        t300 = paddle._C_ops.matmul(t299, t60, False, False)
        del t299, t60

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        t301 = paddle._C_ops.add(t300, t61)
        del t300, t61

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        t302 = paddle._C_ops.multiply(input13, t301)
        del t301, input13

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        t303 = paddle._C_ops.transpose(t302, [0, 3, 1, 2])
        del t302

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t304 = paddle._C_ops.add(t289, t303)
        del t289, t303

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        t305 = paddle._C_ops.depthwise_conv2d(
            t304, t62, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t62

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t306 = paddle._C_ops.reshape(t63, t156)
        del t63

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t307 = paddle._C_ops.add(t305, t306)
        del t305, t306

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        t308 = paddle._C_ops.transpose(t307, [0, 2, 3, 1])
        del t307

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        t309, t310, t311 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t308, t64, t65, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t65, t64, t308

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        t312 = paddle._C_ops.matmul(t309, t66, False, False)
        del t309, t66

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        t313 = paddle._C_ops.add(t312, t67)
        del t312, t67

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        t314 = paddle._C_ops.gelu(t313, False)
        del t313

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        t315 = paddle._C_ops.matmul(t314, t68, False, False)
        del t314, t68

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        t316 = paddle._C_ops.add(t315, t69)
        del t315, t69

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        t317 = paddle._C_ops.multiply(input14, t316)
        del t316, input14

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        t318 = paddle._C_ops.transpose(t317, [0, 3, 1, 2])
        del t317

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t319 = paddle._C_ops.add(t304, t318)
        del t304, t318

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        t320 = paddle._C_ops.depthwise_conv2d(
            t319, t70, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t70

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t321 = paddle._C_ops.reshape(t71, t156)
        del t71

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t322 = paddle._C_ops.add(t320, t321)
        del t320, t321

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        t323 = paddle._C_ops.transpose(t322, [0, 2, 3, 1])
        del t322

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        t324, t325, t326 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t323, t72, t73, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t73, t72, t323

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        t327 = paddle._C_ops.matmul(t324, t74, False, False)
        del t324, t74

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        t328 = paddle._C_ops.add(t327, t75)
        del t327, t75

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        t329 = paddle._C_ops.gelu(t328, False)
        del t328

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        t330 = paddle._C_ops.matmul(t329, t76, False, False)
        del t329, t76

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        t331 = paddle._C_ops.add(t330, t77)
        del t330, t77

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        t332 = paddle._C_ops.multiply(input15, t331)
        del t331, input15

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        t333 = paddle._C_ops.transpose(t332, [0, 3, 1, 2])
        del t332

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t334 = paddle._C_ops.add(t319, t333)
        del t319, t333

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        t335 = paddle._C_ops.depthwise_conv2d(
            t334, t78, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t78

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t336 = paddle._C_ops.reshape(t79, t156)
        del t79

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t337 = paddle._C_ops.add(t335, t336)
        del t335, t336

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        t338 = paddle._C_ops.transpose(t337, [0, 2, 3, 1])
        del t337

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        t339, t340, t341 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t338, t80, t81, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t81, t80, t338

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        t342 = paddle._C_ops.matmul(t339, t82, False, False)
        del t339, t82

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        t343 = paddle._C_ops.add(t342, t83)
        del t342, t83

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        t344 = paddle._C_ops.gelu(t343, False)
        del t343

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        t345 = paddle._C_ops.matmul(t344, t84, False, False)
        del t344, t84

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        t346 = paddle._C_ops.add(t345, t85)
        del t345, t85

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        t347 = paddle._C_ops.multiply(input16, t346)
        del t346, input16

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        t348 = paddle._C_ops.transpose(t347, [0, 3, 1, 2])
        del t347

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t349 = paddle._C_ops.add(t334, t348)
        del t334, t348

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        t350 = paddle._C_ops.depthwise_conv2d(
            t349, t86, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t86

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t351 = paddle._C_ops.reshape(t87, t156)
        del t87

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t352 = paddle._C_ops.add(t350, t351)
        del t350, t351

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        t353 = paddle._C_ops.transpose(t352, [0, 2, 3, 1])
        del t352

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        t354, t355, t356 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t353, t88, t89, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t89, t88, t353

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        t357 = paddle._C_ops.matmul(t354, t90, False, False)
        del t354, t90

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        t358 = paddle._C_ops.add(t357, t91)
        del t357, t91

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        t359 = paddle._C_ops.gelu(t358, False)
        del t358

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        t360 = paddle._C_ops.matmul(t359, t92, False, False)
        del t359, t92

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        t361 = paddle._C_ops.add(t360, t93)
        del t360, t93

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        t362 = paddle._C_ops.multiply(input17, t361)
        del t361, input17

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        t363 = paddle._C_ops.transpose(t362, [0, 3, 1, 2])
        del t362

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t364 = paddle._C_ops.add(t349, t363)
        del t349, t363

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        t365 = paddle._C_ops.depthwise_conv2d(
            t364, t94, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t94

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t366 = paddle._C_ops.reshape(t95, t156)
        del t95

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t367 = paddle._C_ops.add(t365, t366)
        del t365, t366

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        t368 = paddle._C_ops.transpose(t367, [0, 2, 3, 1])
        del t367

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        t369, t370, t371 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t368, t96, t97, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t97, t96, t368

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        t372 = paddle._C_ops.matmul(t369, t98, False, False)
        del t369, t98

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        t373 = paddle._C_ops.add(t372, t99)
        del t372, t99

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        t374 = paddle._C_ops.gelu(t373, False)
        del t373

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        t375 = paddle._C_ops.matmul(t374, t100, False, False)
        del t374, t100

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        t376 = paddle._C_ops.add(t375, t101)
        del t375, t101

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        t377 = paddle._C_ops.multiply(input18, t376)
        del t376, input18

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        t378 = paddle._C_ops.transpose(t377, [0, 3, 1, 2])
        del t377

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t379 = paddle._C_ops.add(t364, t378)
        del t364, t378

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        t380 = paddle._C_ops.depthwise_conv2d(
            t379, t102, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t102

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t381 = paddle._C_ops.reshape(t103, t156)
        del t103

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t382 = paddle._C_ops.add(t380, t381)
        del t380, t381

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        t383 = paddle._C_ops.transpose(t382, [0, 2, 3, 1])
        del t382

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        t384, t385, t386 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t383, t104, t105, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t105, t104, t383

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        t387 = paddle._C_ops.matmul(t384, t106, False, False)
        del t384, t106

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        t388 = paddle._C_ops.add(t387, t107)
        del t387, t107

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        t389 = paddle._C_ops.gelu(t388, False)
        del t388

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        t390 = paddle._C_ops.matmul(t389, t108, False, False)
        del t389, t108

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        t391 = paddle._C_ops.add(t390, t109)
        del t390, t109

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        t392 = paddle._C_ops.multiply(input19, t391)
        del t391, input19

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        t393 = paddle._C_ops.transpose(t392, [0, 3, 1, 2])
        del t392

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t394 = paddle._C_ops.add(t379, t393)
        del t379, t393

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        t395 = paddle._C_ops.depthwise_conv2d(
            t394, t110, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t110

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t396 = paddle._C_ops.reshape(t111, t156)
        del t111

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t397 = paddle._C_ops.add(t395, t396)
        del t395, t396

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        t398 = paddle._C_ops.transpose(t397, [0, 2, 3, 1])
        del t397

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        t399, t400, t401 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t398, t112, t113, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t113, t112, t398

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        t402 = paddle._C_ops.matmul(t399, t114, False, False)
        del t399, t114

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        t403 = paddle._C_ops.add(t402, t115)
        del t402, t115

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        t404 = paddle._C_ops.gelu(t403, False)
        del t403

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        t405 = paddle._C_ops.matmul(t404, t116, False, False)
        del t404, t116

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        t406 = paddle._C_ops.add(t405, t117)
        del t405, t117

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        t407 = paddle._C_ops.multiply(input20, t406)
        del t406, input20

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        t408 = paddle._C_ops.transpose(t407, [0, 3, 1, 2])
        del t407

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t409 = paddle._C_ops.add(t394, t408)
        del t394, t408

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        t410 = paddle._C_ops.depthwise_conv2d(
            t409, t118, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t118

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t411 = paddle._C_ops.reshape(t119, t156)
        del t119

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t412 = paddle._C_ops.add(t410, t411)
        del t410, t411

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        t413 = paddle._C_ops.transpose(t412, [0, 2, 3, 1])
        del t412

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        t414, t415, t416 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t413, t120, t121, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t121, t120, t413

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        t417 = paddle._C_ops.matmul(t414, t122, False, False)
        del t414, t122

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        t418 = paddle._C_ops.add(t417, t123)
        del t417, t123

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        t419 = paddle._C_ops.gelu(t418, False)
        del t418

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        t420 = paddle._C_ops.matmul(t419, t124, False, False)
        del t419, t124

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        t421 = paddle._C_ops.add(t420, t125)
        del t420, t125

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        t422 = paddle._C_ops.multiply(input21, t421)
        del t421, input21

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        t423 = paddle._C_ops.transpose(t422, [0, 3, 1, 2])
        del t422

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t424 = paddle._C_ops.add(t409, t423)
        del t409, t423

        # pd_op.mean: (-1x1x14x14xf32) <- (-1x384x14x14xf32)
        t425 = paddle._C_ops.mean(t424, [1], True)

        # pd_op.subtract: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x14x14xf32)
        t426 = paddle._C_ops.subtract(t424, t425)
        del t424, t425

        # pd_op.pow: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t427 = paddle._C_ops.pow(t426, float("2"))

        # pd_op.mean: (-1x1x14x14xf32) <- (-1x384x14x14xf32)
        t428 = paddle._C_ops.mean(t427, [1], True)
        del t427

        # pd_op.scale: (-1x1x14x14xf32) <- (-1x1x14x14xf32, 1xf32)
        t429 = paddle._C_ops.scale(t428, t163, float("1e-06"), True)
        del t163, t428

        # pd_op.sqrt: (-1x1x14x14xf32) <- (-1x1x14x14xf32)
        t430 = paddle._C_ops.sqrt(t429)
        del t429

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x14x14xf32)
        t431 = paddle._C_ops.divide(t426, t430)
        del t430, t426

        # pd_op.unsqueeze: (384x1x1xf32) <- (384xf32, 2xi64)
        t432 = paddle._C_ops.unsqueeze(input22, t167)
        del input22

        # pd_op.multiply: (-1x384x14x14xf32) <- (384x1x1xf32, -1x384x14x14xf32)
        t433 = paddle._C_ops.multiply(t432, t431)
        del t431, t432

        # pd_op.unsqueeze: (384x1x1xf32) <- (384xf32, 2xi64)
        t434 = paddle._C_ops.unsqueeze(input23, t167)
        del input23, t167

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x1xf32)
        t435 = paddle._C_ops.add(t433, t434)
        del t433, t434

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x384x14x14xf32, 768x384x2x2xf32)
        t436 = paddle._C_ops.conv2d(
            t435, t126, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t435, t126

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t437 = paddle._C_ops.reshape(t127, t156)
        del t127

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t438 = paddle._C_ops.add(t436, t437)
        del t436, t437

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x7x7xf32)
        t439 = paddle._C_ops.depthwise_conv2d(
            t438, t128, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t128

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t440 = paddle._C_ops.reshape(t129, t156)
        del t129

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t441 = paddle._C_ops.add(t439, t440)
        del t439, t440

        # pd_op.transpose: (-1x7x7x768xf32) <- (-1x768x7x7xf32)
        t442 = paddle._C_ops.transpose(t441, [0, 2, 3, 1])
        del t441

        # pd_op.layer_norm: (-1x7x7x768xf32, -1x7x7xf32, -1x7x7xf32) <- (-1x7x7x768xf32, 768xf32, 768xf32)
        t443, t444, t445 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t442, t130, t131, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t131, t130, t442

        # pd_op.matmul: (-1x7x7x3072xf32) <- (-1x7x7x768xf32, 768x3072xf32)
        t446 = paddle._C_ops.matmul(t443, t132, False, False)
        del t443, t132

        # pd_op.add: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32, 3072xf32)
        t447 = paddle._C_ops.add(t446, t133)
        del t446, t133

        # pd_op.gelu: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32)
        t448 = paddle._C_ops.gelu(t447, False)
        del t447

        # pd_op.matmul: (-1x7x7x768xf32) <- (-1x7x7x3072xf32, 3072x768xf32)
        t449 = paddle._C_ops.matmul(t448, t134, False, False)
        del t448, t134

        # pd_op.add: (-1x7x7x768xf32) <- (-1x7x7x768xf32, 768xf32)
        t450 = paddle._C_ops.add(t449, t135)
        del t449, t135

        # pd_op.multiply: (-1x7x7x768xf32) <- (768xf32, -1x7x7x768xf32)
        t451 = paddle._C_ops.multiply(input24, t450)
        del t450, input24

        # pd_op.transpose: (-1x768x7x7xf32) <- (-1x7x7x768xf32)
        t452 = paddle._C_ops.transpose(t451, [0, 3, 1, 2])
        del t451

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        t453 = paddle._C_ops.add(t438, t452)
        del t438, t452

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x7x7xf32)
        t454 = paddle._C_ops.depthwise_conv2d(
            t453, t136, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t136

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t455 = paddle._C_ops.reshape(t137, t156)
        del t137

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t456 = paddle._C_ops.add(t454, t455)
        del t454, t455

        # pd_op.transpose: (-1x7x7x768xf32) <- (-1x768x7x7xf32)
        t457 = paddle._C_ops.transpose(t456, [0, 2, 3, 1])
        del t456

        # pd_op.layer_norm: (-1x7x7x768xf32, -1x7x7xf32, -1x7x7xf32) <- (-1x7x7x768xf32, 768xf32, 768xf32)
        t458, t459, t460 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t457, t138, t139, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t139, t138, t457

        # pd_op.matmul: (-1x7x7x3072xf32) <- (-1x7x7x768xf32, 768x3072xf32)
        t461 = paddle._C_ops.matmul(t458, t140, False, False)
        del t458, t140

        # pd_op.add: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32, 3072xf32)
        t462 = paddle._C_ops.add(t461, t141)
        del t461, t141

        # pd_op.gelu: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32)
        t463 = paddle._C_ops.gelu(t462, False)
        del t462

        # pd_op.matmul: (-1x7x7x768xf32) <- (-1x7x7x3072xf32, 3072x768xf32)
        t464 = paddle._C_ops.matmul(t463, t142, False, False)
        del t463, t142

        # pd_op.add: (-1x7x7x768xf32) <- (-1x7x7x768xf32, 768xf32)
        t465 = paddle._C_ops.add(t464, t143)
        del t464, t143

        # pd_op.multiply: (-1x7x7x768xf32) <- (768xf32, -1x7x7x768xf32)
        t466 = paddle._C_ops.multiply(input25, t465)
        del t465, input25

        # pd_op.transpose: (-1x768x7x7xf32) <- (-1x7x7x768xf32)
        t467 = paddle._C_ops.transpose(t466, [0, 3, 1, 2])
        del t466

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        t468 = paddle._C_ops.add(t453, t467)
        del t453, t467

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x7x7xf32)
        t469 = paddle._C_ops.depthwise_conv2d(
            t468, t144, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t144

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t470 = paddle._C_ops.reshape(t145, t156)
        del t156, t145

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t471 = paddle._C_ops.add(t469, t470)
        del t469, t470

        # pd_op.transpose: (-1x7x7x768xf32) <- (-1x768x7x7xf32)
        t472 = paddle._C_ops.transpose(t471, [0, 2, 3, 1])
        del t471

        # pd_op.layer_norm: (-1x7x7x768xf32, -1x7x7xf32, -1x7x7xf32) <- (-1x7x7x768xf32, 768xf32, 768xf32)
        t473, t474, t475 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t472, t146, t147, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t147, t146, t472

        # pd_op.matmul: (-1x7x7x3072xf32) <- (-1x7x7x768xf32, 768x3072xf32)
        t476 = paddle._C_ops.matmul(t473, t148, False, False)
        del t473, t148

        # pd_op.add: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32, 3072xf32)
        t477 = paddle._C_ops.add(t476, t149)
        del t476, t149

        # pd_op.gelu: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32)
        t478 = paddle._C_ops.gelu(t477, False)
        del t477

        # pd_op.matmul: (-1x7x7x768xf32) <- (-1x7x7x3072xf32, 3072x768xf32)
        t479 = paddle._C_ops.matmul(t478, t150, False, False)
        del t478, t150

        # pd_op.add: (-1x7x7x768xf32) <- (-1x7x7x768xf32, 768xf32)
        t480 = paddle._C_ops.add(t479, t151)
        del t479, t151

        # pd_op.multiply: (-1x7x7x768xf32) <- (768xf32, -1x7x7x768xf32)
        t481 = paddle._C_ops.multiply(input26, t480)
        del t480, input26

        # pd_op.transpose: (-1x768x7x7xf32) <- (-1x7x7x768xf32)
        t482 = paddle._C_ops.transpose(t481, [0, 3, 1, 2])
        del t481

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        t483 = paddle._C_ops.add(t468, t482)
        del t468, t482

        # pd_op.mean: (-1x768xf32) <- (-1x768x7x7xf32)
        t484 = paddle._C_ops.mean(t483, [-2, -1], False)
        del t483

        # pd_op.layer_norm: (-1x768xf32, -1xf32, -1xf32) <- (-1x768xf32, 768xf32, 768xf32)
        t485, t486, t487 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t484, t152, t153, float("1e-06"), 1),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t484, t153, t152

        # pd_op.matmul: (-1x102xf32) <- (-1x768xf32, 768x102xf32)
        t488 = paddle._C_ops.matmul(t485, t154, False, False)
        del t485, t154

        return t488
