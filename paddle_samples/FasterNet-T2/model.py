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
    ):
        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x3x224x224xf32, 96x3x4x4xf32)
        t113 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t114, t115, t116, t117, t118, t119 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t113,
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
        t120 = [24, 72]

        # pd_op.full: (1xi32) <- ()
        t121 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t122 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t123 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t124 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t125 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t126 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t127 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t128 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t129 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t130 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t131 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t132 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t133 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t134 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t135 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t136 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t137 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t138 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t139 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t140 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t141 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t142 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t143 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t144 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t145 = t121

        # pd_op.assign: (1xi32) <- (1xi32)
        t146 = t121

        # pd_op.split: ([-1x24x56x56xf32, -1x72x56x56xf32]) <- (-1x96x56x56xf32, 2xi64, 1xi32)
        t147 = paddle._C_ops.split(t114, t120, t121)
        del t120

        # builtin.split: (-1x24x56x56xf32, -1x72x56x56xf32) <- ([-1x24x56x56xf32, -1x72x56x56xf32])
        (
            t148,
            t149,
        ) = t147
        del t147

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x24x3x3xf32)
        t150 = paddle._C_ops.conv2d(
            t148, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # builtin.combine: ([-1x24x56x56xf32, -1x72x56x56xf32]) <- (-1x24x56x56xf32, -1x72x56x56xf32)
        t151 = [t150, t149]

        # pd_op.concat: (-1x96x56x56xf32) <- ([-1x24x56x56xf32, -1x72x56x56xf32], 1xi32)
        t152 = paddle._C_ops.concat(t151, t121)
        del t151

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x96x56x56xf32, 192x96x1x1xf32)
        t153 = paddle._C_ops.conv2d(
            t152, t6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t6

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t154, t155, t156, t157, t158, t159 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t153,
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

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t160 = paddle._C_ops.relu(t154)
        del t154

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x192x56x56xf32, 96x192x1x1xf32)
        t161 = paddle._C_ops.conv2d(
            t160, t11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t11

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        t162 = paddle._C_ops.add(t114, t161)

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x96x56x56xf32, 192x96x2x2xf32)
        t163 = paddle._C_ops.conv2d(
            t162, t12, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t12

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t164, t165, t166, t167, t168, t169 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t163,
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
        del t13, t16, t15, t14

        # pd_op.full_int_array: (2xi64) <- ()
        t170 = [48, 144]

        # pd_op.split: ([-1x48x28x28xf32, -1x144x28x28xf32]) <- (-1x192x28x28xf32, 2xi64, 1xi32)
        t171 = paddle._C_ops.split(t164, t170, t121)

        # builtin.split: (-1x48x28x28xf32, -1x144x28x28xf32) <- ([-1x48x28x28xf32, -1x144x28x28xf32])
        (
            t172,
            t173,
        ) = t171
        del t171

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x48x3x3xf32)
        t174 = paddle._C_ops.conv2d(
            t172, t17, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t17

        # builtin.combine: ([-1x48x28x28xf32, -1x144x28x28xf32]) <- (-1x48x28x28xf32, -1x144x28x28xf32)
        t175 = [t174, t173]

        # pd_op.concat: (-1x192x28x28xf32) <- ([-1x48x28x28xf32, -1x144x28x28xf32], 1xi32)
        t176 = paddle._C_ops.concat(t175, t121)
        del t175

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x192x28x28xf32, 384x192x1x1xf32)
        t177 = paddle._C_ops.conv2d(
            t176, t18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t18

        # pd_op.batch_norm_: (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t178, t179, t180, t181, t182, t183 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t177,
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

        # pd_op.relu: (-1x384x28x28xf32) <- (-1x384x28x28xf32)
        t184 = paddle._C_ops.relu(t178)
        del t178

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x384x28x28xf32, 192x384x1x1xf32)
        t185 = paddle._C_ops.conv2d(
            t184, t23, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t23

        # pd_op.full: (xf32) <- ()
        t186 = paddle._C_ops.full(
            [],
            float("0.995833"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x192x28x28xf32)
        t187 = paddle._C_ops.shape64(t185)

        # pd_op.full_int_array: (1xi64) <- ()
        t188 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        t189 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t190 = paddle._C_ops.slice(t187, [0], t188, t189, [1], [0])
        del t187

        # pd_op.full: (xi64) <- ()
        t191 = paddle._C_ops.full([], float("1"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t192 = [t190, t191, t191, t191]
        del t190

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t193 = paddle._C_ops.stack(t192, 0)
        del t192

        # pd_op.full: (1xf32) <- ()
        t194 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t195 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t196 = paddle._C_ops.uniform(
            t193,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t193

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t197 = paddle._C_ops.add(t186, t196)
        del t196

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t198 = paddle._C_ops.floor(t197)
        del t197

        # pd_op.divide: (-1x192x28x28xf32) <- (-1x192x28x28xf32, xf32)
        t199 = paddle._C_ops.divide(t185, t186)

        # pd_op.multiply: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x1x1x1xf32)
        t200 = paddle._C_ops.multiply(t199, t198)

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        t201 = paddle._C_ops.add(t164, t200)

        # pd_op.split: ([-1x48x28x28xf32, -1x144x28x28xf32]) <- (-1x192x28x28xf32, 2xi64, 1xi32)
        t202 = paddle._C_ops.split(t201, t170, t121)
        del t170

        # builtin.split: (-1x48x28x28xf32, -1x144x28x28xf32) <- ([-1x48x28x28xf32, -1x144x28x28xf32])
        (
            t203,
            t204,
        ) = t202
        del t202

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x48x3x3xf32)
        t205 = paddle._C_ops.conv2d(
            t203, t24, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t24

        # builtin.combine: ([-1x48x28x28xf32, -1x144x28x28xf32]) <- (-1x48x28x28xf32, -1x144x28x28xf32)
        t206 = [t205, t204]

        # pd_op.concat: (-1x192x28x28xf32) <- ([-1x48x28x28xf32, -1x144x28x28xf32], 1xi32)
        t207 = paddle._C_ops.concat(t206, t121)
        del t206

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x192x28x28xf32, 384x192x1x1xf32)
        t208 = paddle._C_ops.conv2d(
            t207, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t209, t210, t211, t212, t213, t214 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t208,
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

        # pd_op.relu: (-1x384x28x28xf32) <- (-1x384x28x28xf32)
        t215 = paddle._C_ops.relu(t209)
        del t209

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x384x28x28xf32, 192x384x1x1xf32)
        t216 = paddle._C_ops.conv2d(
            t215, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.full: (xf32) <- ()
        t217 = paddle._C_ops.full(
            [],
            float("0.991667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x192x28x28xf32)
        t218 = paddle._C_ops.shape64(t216)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t219 = paddle._C_ops.slice(t218, [0], t188, t189, [1], [0])
        del t218

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t220 = [t219, t191, t191, t191]
        del t219

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t221 = paddle._C_ops.stack(t220, 0)
        del t220

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t222 = paddle._C_ops.uniform(
            t221,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t221

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t223 = paddle._C_ops.add(t217, t222)
        del t222

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t224 = paddle._C_ops.floor(t223)
        del t223

        # pd_op.divide: (-1x192x28x28xf32) <- (-1x192x28x28xf32, xf32)
        t225 = paddle._C_ops.divide(t216, t217)

        # pd_op.multiply: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x1x1x1xf32)
        t226 = paddle._C_ops.multiply(t225, t224)

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        t227 = paddle._C_ops.add(t201, t226)

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x192x28x28xf32, 384x192x2x2xf32)
        t228 = paddle._C_ops.conv2d(
            t227, t31, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t31

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t229, t230, t231, t232, t233, t234 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t228,
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
        t235 = [96, 288]

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        t236 = paddle._C_ops.split(t229, t235, t121)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            t237,
            t238,
        ) = t236
        del t236

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        t239 = paddle._C_ops.conv2d(
            t237, t36, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t36

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        t240 = [t239, t238]

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        t241 = paddle._C_ops.concat(t240, t121)
        del t240

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        t242 = paddle._C_ops.conv2d(
            t241, t37, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t37

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t243, t244, t245, t246, t247, t248 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t242,
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

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        t249 = paddle._C_ops.relu(t243)
        del t243

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        t250 = paddle._C_ops.conv2d(
            t249, t42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t42

        # pd_op.full: (xf32) <- ()
        t251 = paddle._C_ops.full(
            [],
            float("0.9875"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x14x14xf32)
        t252 = paddle._C_ops.shape64(t250)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t253 = paddle._C_ops.slice(t252, [0], t188, t189, [1], [0])
        del t252

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t254 = [t253, t191, t191, t191]
        del t253

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t255 = paddle._C_ops.stack(t254, 0)
        del t254

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t256 = paddle._C_ops.uniform(
            t255,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t255

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t257 = paddle._C_ops.add(t251, t256)
        del t256

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t258 = paddle._C_ops.floor(t257)
        del t257

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, xf32)
        t259 = paddle._C_ops.divide(t250, t251)

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x1x1xf32)
        t260 = paddle._C_ops.multiply(t259, t258)

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t261 = paddle._C_ops.add(t229, t260)

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        t262 = paddle._C_ops.split(t261, t235, t121)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            t263,
            t264,
        ) = t262
        del t262

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        t265 = paddle._C_ops.conv2d(
            t263, t43, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t43

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        t266 = [t265, t264]

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        t267 = paddle._C_ops.concat(t266, t121)
        del t266

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        t268 = paddle._C_ops.conv2d(
            t267, t44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t44

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t269, t270, t271, t272, t273, t274 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t268,
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

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        t275 = paddle._C_ops.relu(t269)
        del t269

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        t276 = paddle._C_ops.conv2d(
            t275, t49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t49

        # pd_op.full: (xf32) <- ()
        t277 = paddle._C_ops.full(
            [],
            float("0.983333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x14x14xf32)
        t278 = paddle._C_ops.shape64(t276)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t279 = paddle._C_ops.slice(t278, [0], t188, t189, [1], [0])
        del t278

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t280 = [t279, t191, t191, t191]
        del t279

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t281 = paddle._C_ops.stack(t280, 0)
        del t280

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t282 = paddle._C_ops.uniform(
            t281,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t281

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t283 = paddle._C_ops.add(t277, t282)
        del t282

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t284 = paddle._C_ops.floor(t283)
        del t283

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, xf32)
        t285 = paddle._C_ops.divide(t276, t277)

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x1x1xf32)
        t286 = paddle._C_ops.multiply(t285, t284)

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t287 = paddle._C_ops.add(t261, t286)

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        t288 = paddle._C_ops.split(t287, t235, t121)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            t289,
            t290,
        ) = t288
        del t288

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        t291 = paddle._C_ops.conv2d(
            t289, t50, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        t292 = [t291, t290]

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        t293 = paddle._C_ops.concat(t292, t121)
        del t292

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        t294 = paddle._C_ops.conv2d(
            t293, t51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t51

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t295, t296, t297, t298, t299, t300 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t294,
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

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        t301 = paddle._C_ops.relu(t295)
        del t295

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        t302 = paddle._C_ops.conv2d(
            t301, t56, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t56

        # pd_op.full: (xf32) <- ()
        t303 = paddle._C_ops.full(
            [],
            float("0.979167"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x14x14xf32)
        t304 = paddle._C_ops.shape64(t302)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t305 = paddle._C_ops.slice(t304, [0], t188, t189, [1], [0])
        del t304

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t306 = [t305, t191, t191, t191]
        del t305

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t307 = paddle._C_ops.stack(t306, 0)
        del t306

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t308 = paddle._C_ops.uniform(
            t307,
            paddle.float32,
            t194,
            t195,
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

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, xf32)
        t311 = paddle._C_ops.divide(t302, t303)

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x1x1xf32)
        t312 = paddle._C_ops.multiply(t311, t310)

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t313 = paddle._C_ops.add(t287, t312)

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        t314 = paddle._C_ops.split(t313, t235, t121)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            t315,
            t316,
        ) = t314
        del t314

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        t317 = paddle._C_ops.conv2d(
            t315, t57, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t57

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        t318 = [t317, t316]

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        t319 = paddle._C_ops.concat(t318, t121)
        del t318

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        t320 = paddle._C_ops.conv2d(
            t319, t58, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t58

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t321, t322, t323, t324, t325, t326 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t320,
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

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        t327 = paddle._C_ops.relu(t321)
        del t321

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        t328 = paddle._C_ops.conv2d(
            t327, t63, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t63

        # pd_op.full: (xf32) <- ()
        t329 = paddle._C_ops.full(
            [],
            float("0.975"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x14x14xf32)
        t330 = paddle._C_ops.shape64(t328)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t331 = paddle._C_ops.slice(t330, [0], t188, t189, [1], [0])
        del t330

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t332 = [t331, t191, t191, t191]
        del t331

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t333 = paddle._C_ops.stack(t332, 0)
        del t332

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t334 = paddle._C_ops.uniform(
            t333,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t333

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t335 = paddle._C_ops.add(t329, t334)
        del t334

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t336 = paddle._C_ops.floor(t335)
        del t335

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, xf32)
        t337 = paddle._C_ops.divide(t328, t329)

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x1x1xf32)
        t338 = paddle._C_ops.multiply(t337, t336)

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t339 = paddle._C_ops.add(t313, t338)

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        t340 = paddle._C_ops.split(t339, t235, t121)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            t341,
            t342,
        ) = t340
        del t340

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        t343 = paddle._C_ops.conv2d(
            t341, t64, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t64

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        t344 = [t343, t342]

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        t345 = paddle._C_ops.concat(t344, t121)
        del t344

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        t346 = paddle._C_ops.conv2d(
            t345, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t347, t348, t349, t350, t351, t352 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t346,
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

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        t353 = paddle._C_ops.relu(t347)
        del t347

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        t354 = paddle._C_ops.conv2d(
            t353, t70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.full: (xf32) <- ()
        t355 = paddle._C_ops.full(
            [],
            float("0.970833"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x14x14xf32)
        t356 = paddle._C_ops.shape64(t354)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t357 = paddle._C_ops.slice(t356, [0], t188, t189, [1], [0])
        del t356

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t358 = [t357, t191, t191, t191]
        del t357

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t359 = paddle._C_ops.stack(t358, 0)
        del t358

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t360 = paddle._C_ops.uniform(
            t359,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t359

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t361 = paddle._C_ops.add(t355, t360)
        del t360

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t362 = paddle._C_ops.floor(t361)
        del t361

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, xf32)
        t363 = paddle._C_ops.divide(t354, t355)

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x1x1xf32)
        t364 = paddle._C_ops.multiply(t363, t362)

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t365 = paddle._C_ops.add(t339, t364)

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        t366 = paddle._C_ops.split(t365, t235, t121)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            t367,
            t368,
        ) = t366
        del t366

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        t369 = paddle._C_ops.conv2d(
            t367, t71, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t71

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        t370 = [t369, t368]

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        t371 = paddle._C_ops.concat(t370, t121)
        del t370

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        t372 = paddle._C_ops.conv2d(
            t371, t72, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t72

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t373, t374, t375, t376, t377, t378 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t372,
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

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        t379 = paddle._C_ops.relu(t373)
        del t373

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        t380 = paddle._C_ops.conv2d(
            t379, t77, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t77

        # pd_op.full: (xf32) <- ()
        t381 = paddle._C_ops.full(
            [],
            float("0.966667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x14x14xf32)
        t382 = paddle._C_ops.shape64(t380)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t383 = paddle._C_ops.slice(t382, [0], t188, t189, [1], [0])
        del t382

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t384 = [t383, t191, t191, t191]
        del t383

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t385 = paddle._C_ops.stack(t384, 0)
        del t384

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t386 = paddle._C_ops.uniform(
            t385,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t385

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t387 = paddle._C_ops.add(t381, t386)
        del t386

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t388 = paddle._C_ops.floor(t387)
        del t387

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, xf32)
        t389 = paddle._C_ops.divide(t380, t381)

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x1x1xf32)
        t390 = paddle._C_ops.multiply(t389, t388)

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t391 = paddle._C_ops.add(t365, t390)

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        t392 = paddle._C_ops.split(t391, t235, t121)

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            t393,
            t394,
        ) = t392
        del t392

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        t395 = paddle._C_ops.conv2d(
            t393, t78, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t78

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        t396 = [t395, t394]

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        t397 = paddle._C_ops.concat(t396, t121)
        del t396

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        t398 = paddle._C_ops.conv2d(
            t397, t79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t79

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t399, t400, t401, t402, t403, t404 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t398,
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

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        t405 = paddle._C_ops.relu(t399)
        del t399

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        t406 = paddle._C_ops.conv2d(
            t405, t84, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t84

        # pd_op.full: (xf32) <- ()
        t407 = paddle._C_ops.full(
            [],
            float("0.9625"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x14x14xf32)
        t408 = paddle._C_ops.shape64(t406)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t409 = paddle._C_ops.slice(t408, [0], t188, t189, [1], [0])
        del t408

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t410 = [t409, t191, t191, t191]
        del t409

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t411 = paddle._C_ops.stack(t410, 0)
        del t410

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t412 = paddle._C_ops.uniform(
            t411,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t411

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t413 = paddle._C_ops.add(t407, t412)
        del t412

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t414 = paddle._C_ops.floor(t413)
        del t413

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, xf32)
        t415 = paddle._C_ops.divide(t406, t407)

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x1x1xf32)
        t416 = paddle._C_ops.multiply(t415, t414)

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t417 = paddle._C_ops.add(t391, t416)

        # pd_op.split: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x384x14x14xf32, 2xi64, 1xi32)
        t418 = paddle._C_ops.split(t417, t235, t121)
        del t235

        # builtin.split: (-1x96x14x14xf32, -1x288x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32])
        (
            t419,
            t420,
        ) = t418
        del t418

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x96x3x3xf32)
        t421 = paddle._C_ops.conv2d(
            t419, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # builtin.combine: ([-1x96x14x14xf32, -1x288x14x14xf32]) <- (-1x96x14x14xf32, -1x288x14x14xf32)
        t422 = [t421, t420]

        # pd_op.concat: (-1x384x14x14xf32) <- ([-1x96x14x14xf32, -1x288x14x14xf32], 1xi32)
        t423 = paddle._C_ops.concat(t422, t121)
        del t422

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x14x14xf32, 768x384x1x1xf32)
        t424 = paddle._C_ops.conv2d(
            t423, t86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t86

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t425, t426, t427, t428, t429, t430 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t424,
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

        # pd_op.relu: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        t431 = paddle._C_ops.relu(t425)
        del t425

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x768x14x14xf32, 384x768x1x1xf32)
        t432 = paddle._C_ops.conv2d(
            t431, t91, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t91

        # pd_op.full: (xf32) <- ()
        t433 = paddle._C_ops.full(
            [],
            float("0.958333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x384x14x14xf32)
        t434 = paddle._C_ops.shape64(t432)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t435 = paddle._C_ops.slice(t434, [0], t188, t189, [1], [0])
        del t434

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t436 = [t435, t191, t191, t191]
        del t435

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t437 = paddle._C_ops.stack(t436, 0)
        del t436

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t438 = paddle._C_ops.uniform(
            t437,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t437

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t439 = paddle._C_ops.add(t433, t438)
        del t438

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t440 = paddle._C_ops.floor(t439)
        del t439

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, xf32)
        t441 = paddle._C_ops.divide(t432, t433)

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x1x1xf32)
        t442 = paddle._C_ops.multiply(t441, t440)

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t443 = paddle._C_ops.add(t417, t442)

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x384x14x14xf32, 768x384x2x2xf32)
        t444 = paddle._C_ops.conv2d(
            t443, t92, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t92

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t445, t446, t447, t448, t449, t450 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t444,
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

        # pd_op.full_int_array: (2xi64) <- ()
        t451 = [192, 576]

        # pd_op.split: ([-1x192x7x7xf32, -1x576x7x7xf32]) <- (-1x768x7x7xf32, 2xi64, 1xi32)
        t452 = paddle._C_ops.split(t445, t451, t121)

        # builtin.split: (-1x192x7x7xf32, -1x576x7x7xf32) <- ([-1x192x7x7xf32, -1x576x7x7xf32])
        (
            t453,
            t454,
        ) = t452
        del t452

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x192x3x3xf32)
        t455 = paddle._C_ops.conv2d(
            t453, t97, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t97

        # builtin.combine: ([-1x192x7x7xf32, -1x576x7x7xf32]) <- (-1x192x7x7xf32, -1x576x7x7xf32)
        t456 = [t455, t454]

        # pd_op.concat: (-1x768x7x7xf32) <- ([-1x192x7x7xf32, -1x576x7x7xf32], 1xi32)
        t457 = paddle._C_ops.concat(t456, t121)
        del t456

        # pd_op.conv2d: (-1x1536x7x7xf32) <- (-1x768x7x7xf32, 1536x768x1x1xf32)
        t458 = paddle._C_ops.conv2d(
            t457, t98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t98

        # pd_op.batch_norm_: (-1x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t459, t460, t461, t462, t463, t464 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t458,
                t99,
                t100,
                t101,
                t102,
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
        del t102, t101, t100, t99

        # pd_op.relu: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32)
        t465 = paddle._C_ops.relu(t459)
        del t459

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x1536x7x7xf32, 768x1536x1x1xf32)
        t466 = paddle._C_ops.conv2d(
            t465, t103, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t103

        # pd_op.full: (xf32) <- ()
        t467 = paddle._C_ops.full(
            [],
            float("0.954167"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x7x7xf32)
        t468 = paddle._C_ops.shape64(t466)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t469 = paddle._C_ops.slice(t468, [0], t188, t189, [1], [0])
        del t468

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t470 = [t469, t191, t191, t191]
        del t469

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t471 = paddle._C_ops.stack(t470, 0)
        del t470

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t472 = paddle._C_ops.uniform(
            t471,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t471

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t473 = paddle._C_ops.add(t467, t472)
        del t472

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t474 = paddle._C_ops.floor(t473)
        del t473

        # pd_op.divide: (-1x768x7x7xf32) <- (-1x768x7x7xf32, xf32)
        t475 = paddle._C_ops.divide(t466, t467)

        # pd_op.multiply: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x1x1x1xf32)
        t476 = paddle._C_ops.multiply(t475, t474)

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        t477 = paddle._C_ops.add(t445, t476)

        # pd_op.split: ([-1x192x7x7xf32, -1x576x7x7xf32]) <- (-1x768x7x7xf32, 2xi64, 1xi32)
        t478 = paddle._C_ops.split(t477, t451, t121)
        del t451

        # builtin.split: (-1x192x7x7xf32, -1x576x7x7xf32) <- ([-1x192x7x7xf32, -1x576x7x7xf32])
        (
            t479,
            t480,
        ) = t478
        del t478

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x192x3x3xf32)
        t481 = paddle._C_ops.conv2d(
            t479, t104, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t104

        # builtin.combine: ([-1x192x7x7xf32, -1x576x7x7xf32]) <- (-1x192x7x7xf32, -1x576x7x7xf32)
        t482 = [t481, t480]

        # pd_op.concat: (-1x768x7x7xf32) <- ([-1x192x7x7xf32, -1x576x7x7xf32], 1xi32)
        t483 = paddle._C_ops.concat(t482, t121)
        del t482

        # pd_op.conv2d: (-1x1536x7x7xf32) <- (-1x768x7x7xf32, 1536x768x1x1xf32)
        t484 = paddle._C_ops.conv2d(
            t483, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (-1x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (-1x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        t485, t486, t487, t488, t489, t490 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t484,
                t106,
                t107,
                t108,
                t109,
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
        del t109, t108, t107, t106

        # pd_op.relu: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32)
        t491 = paddle._C_ops.relu(t485)
        del t485

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x1536x7x7xf32, 768x1536x1x1xf32)
        t492 = paddle._C_ops.conv2d(
            t491, t110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.full: (xf32) <- ()
        t493 = paddle._C_ops.full(
            [],
            float("0.95"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x768x7x7xf32)
        t494 = paddle._C_ops.shape64(t492)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t495 = paddle._C_ops.slice(t494, [0], t188, t189, [1], [0])
        del t188, t189, t494

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t496 = [t495, t191, t191, t191]
        del t191, t495

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t497 = paddle._C_ops.stack(t496, 0)
        del t496

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        t498 = paddle._C_ops.uniform(
            t497,
            paddle.float32,
            t194,
            t195,
            0,
            paddle.framework._current_expected_place(),
        )
        del t194, t195, t497

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        t499 = paddle._C_ops.add(t493, t498)
        del t498

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        t500 = paddle._C_ops.floor(t499)
        del t499

        # pd_op.divide: (-1x768x7x7xf32) <- (-1x768x7x7xf32, xf32)
        t501 = paddle._C_ops.divide(t492, t493)

        # pd_op.multiply: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x1x1x1xf32)
        t502 = paddle._C_ops.multiply(t501, t500)

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        t503 = paddle._C_ops.add(t477, t502)

        # pd_op.full_int_array: (2xi64) <- ()
        t504 = [1, 1]

        # pd_op.pool2d: (-1x768x1x1xf32) <- (-1x768x7x7xf32, 2xi64)
        t505 = paddle._C_ops.pool2d(
            t503,
            t504,
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

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x768x1x1xf32, 1280x768x1x1xf32)
        t506 = paddle._C_ops.conv2d(
            t505, t111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t111

        # pd_op.relu: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        t507 = paddle._C_ops.relu(t506)
        del t506

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        t508 = paddle._C_ops.flatten(t507, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        t509 = paddle._C_ops.matmul(t508, t112, False, False)
        del t112

        return (
            t113,
            t114,
            t115,
            t116,
            t117,
            t118,
            t119,
            t121,
            t122,
            t123,
            t124,
            t125,
            t126,
            t127,
            t128,
            t129,
            t130,
            t131,
            t132,
            t133,
            t134,
            t135,
            t136,
            t137,
            t138,
            t139,
            t140,
            t141,
            t142,
            t143,
            t144,
            t145,
            t146,
            t148,
            t149,
            t150,
            t152,
            t153,
            t155,
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
            t172,
            t173,
            t174,
            t176,
            t177,
            t179,
            t180,
            t181,
            t182,
            t183,
            t184,
            t185,
            t186,
            t198,
            t199,
            t200,
            t201,
            t203,
            t204,
            t205,
            t207,
            t208,
            t210,
            t211,
            t212,
            t213,
            t214,
            t215,
            t216,
            t217,
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
            t237,
            t238,
            t239,
            t241,
            t242,
            t244,
            t245,
            t246,
            t247,
            t248,
            t249,
            t250,
            t251,
            t258,
            t259,
            t260,
            t261,
            t263,
            t264,
            t265,
            t267,
            t268,
            t270,
            t271,
            t272,
            t273,
            t274,
            t275,
            t276,
            t277,
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
            t315,
            t316,
            t317,
            t319,
            t320,
            t322,
            t323,
            t324,
            t325,
            t326,
            t327,
            t328,
            t329,
            t336,
            t337,
            t338,
            t339,
            t341,
            t342,
            t343,
            t345,
            t346,
            t348,
            t349,
            t350,
            t351,
            t352,
            t353,
            t354,
            t355,
            t362,
            t363,
            t364,
            t365,
            t367,
            t368,
            t369,
            t371,
            t372,
            t374,
            t375,
            t376,
            t377,
            t378,
            t379,
            t380,
            t381,
            t388,
            t389,
            t390,
            t391,
            t393,
            t394,
            t395,
            t397,
            t398,
            t400,
            t401,
            t402,
            t403,
            t404,
            t405,
            t406,
            t407,
            t414,
            t415,
            t416,
            t417,
            t419,
            t420,
            t421,
            t423,
            t424,
            t426,
            t427,
            t428,
            t429,
            t430,
            t431,
            t432,
            t433,
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
            t453,
            t454,
            t455,
            t457,
            t458,
            t460,
            t461,
            t462,
            t463,
            t464,
            t465,
            t466,
            t467,
            t474,
            t475,
            t476,
            t477,
            t479,
            t480,
            t481,
            t483,
            t484,
            t486,
            t487,
            t488,
            t489,
            t490,
            t491,
            t492,
            t493,
            t500,
            t501,
            t502,
            t503,
            t504,
            t505,
            t507,
            t508,
            t509,
        )
