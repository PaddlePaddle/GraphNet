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
    ):
        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x3x224x224xf32, 64x3x7x7xf32)
        t106 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t107, t108, t109, t110, t111, t112 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t106,
                t1,
                t2,
                t3,
                t4,
                True,
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
        del t106, t4, t3, t2, t1

        # pd_op.relu: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        t113 = paddle._C_ops.relu(t107)
        del t107

        # pd_op.full_int_array: (2xi64) <- ()
        t114 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        t115 = paddle._C_ops.pool2d(
            t113,
            t114,
            [2, 2],
            [1, 1],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )
        del t114, t113

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t116 = paddle._C_ops.conv2d(
            t115, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t117, t118, t119, t120, t121, t122 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t116,
                t6,
                t7,
                t8,
                t9,
                True,
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
        del t116, t6, t9, t8, t7

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t123 = paddle._C_ops.relu(t117)
        del t117

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t124 = paddle._C_ops.conv2d(
            t123, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10, t123

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t125, t126, t127, t128, t129, t130 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t124,
                t11,
                t12,
                t13,
                t14,
                True,
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
        del t124, t14, t13, t12, t11

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        t131 = paddle._C_ops.conv2d(
            t115, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15, t115

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t132, t133, t134, t135, t136, t137 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t131,
                t16,
                t17,
                t18,
                t19,
                True,
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
        del t131, t19, t18, t17, t16

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t138 = paddle._C_ops.add(t125, t132)
        del t125, t132

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t139 = paddle._C_ops.relu(t138)
        del t138

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t140 = paddle._C_ops.conv2d(
            t139, t20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t141, t142, t143, t144, t145, t146 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t140,
                t21,
                t22,
                t23,
                t24,
                True,
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
        del t140, t24, t23, t22, t21

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t147 = paddle._C_ops.relu(t141)
        del t141

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t148 = paddle._C_ops.conv2d(
            t147, t25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25, t147

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t149, t150, t151, t152, t153, t154 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t148,
                t26,
                t27,
                t28,
                t29,
                True,
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
        del t148, t29, t28, t27, t26

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        t155 = paddle._C_ops.add(t149, t139)
        del t149, t139

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t156 = paddle._C_ops.relu(t155)
        del t155

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x56x56xf32, 128x64x3x3xf32)
        t157 = paddle._C_ops.conv2d(
            t156, t30, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t158, t159, t160, t161, t162, t163 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t157,
                t31,
                t32,
                t33,
                t34,
                True,
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
        del t157, t34, t33, t32, t31

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t164 = paddle._C_ops.relu(t158)
        del t158

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t165 = paddle._C_ops.conv2d(
            t164, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35, t164

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t166, t167, t168, t169, t170, t171 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t165,
                t36,
                t37,
                t38,
                t39,
                True,
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
        del t165, t39, t38, t37, t36

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x56x56xf32, 128x64x1x1xf32)
        t172 = paddle._C_ops.conv2d(
            t156, t40, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40, t156

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t173, t174, t175, t176, t177, t178 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t172,
                t41,
                t42,
                t43,
                t44,
                True,
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
        del t172, t44, t43, t42, t41

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t179 = paddle._C_ops.add(t166, t173)
        del t166, t173

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t180 = paddle._C_ops.relu(t179)
        del t179

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t181 = paddle._C_ops.conv2d(
            t180, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t182, t183, t184, t185, t186, t187 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t181,
                t46,
                t47,
                t48,
                t49,
                True,
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
        del t181, t49, t48, t47, t46

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t188 = paddle._C_ops.relu(t182)
        del t182

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t189 = paddle._C_ops.conv2d(
            t188, t50, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50, t188

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t190, t191, t192, t193, t194, t195 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t189,
                t51,
                t52,
                t53,
                t54,
                True,
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
        del t189, t54, t53, t52, t51

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        t196 = paddle._C_ops.add(t190, t180)
        del t190, t180

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t197 = paddle._C_ops.relu(t196)
        del t196

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x28x28xf32, 256x128x3x3xf32)
        t198 = paddle._C_ops.conv2d(
            t197, t55, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t199, t200, t201, t202, t203, t204 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t198,
                t56,
                t57,
                t58,
                t59,
                True,
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
        del t198, t59, t58, t57, t56

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t205 = paddle._C_ops.relu(t199)
        del t199

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t206 = paddle._C_ops.conv2d(
            t205, t60, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60, t205

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t207, t208, t209, t210, t211, t212 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t206,
                t61,
                t62,
                t63,
                t64,
                True,
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
        del t206, t64, t63, t62, t61

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x28x28xf32, 256x128x1x1xf32)
        t213 = paddle._C_ops.conv2d(
            t197, t65, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65, t197

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t214, t215, t216, t217, t218, t219 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t213,
                t66,
                t67,
                t68,
                t69,
                True,
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
        del t213, t69, t68, t67, t66

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t220 = paddle._C_ops.add(t207, t214)
        del t207, t214

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t221 = paddle._C_ops.relu(t220)
        del t220

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t222 = paddle._C_ops.conv2d(
            t221, t70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t223, t224, t225, t226, t227, t228 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t222,
                t71,
                t72,
                t73,
                t74,
                True,
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
        del t222, t74, t73, t72, t71

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t229 = paddle._C_ops.relu(t223)
        del t223

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t230 = paddle._C_ops.conv2d(
            t229, t75, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75, t229

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t231, t232, t233, t234, t235, t236 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t230,
                t76,
                t77,
                t78,
                t79,
                True,
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
        del t230, t79, t78, t77, t76

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        t237 = paddle._C_ops.add(t231, t221)
        del t231, t221

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t238 = paddle._C_ops.relu(t237)
        del t237

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x14x14xf32, 512x256x3x3xf32)
        t239 = paddle._C_ops.conv2d(
            t238, t80, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t240, t241, t242, t243, t244, t245 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t239,
                t81,
                t82,
                t83,
                t84,
                True,
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
        del t239, t84, t83, t82, t81

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t246 = paddle._C_ops.relu(t240)
        del t240

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t247 = paddle._C_ops.conv2d(
            t246, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85, t246

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t248, t249, t250, t251, t252, t253 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t247,
                t86,
                t87,
                t88,
                t89,
                True,
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
        del t247, t89, t88, t87, t86

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x14x14xf32, 512x256x1x1xf32)
        t254 = paddle._C_ops.conv2d(
            t238, t90, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90, t238

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t255, t256, t257, t258, t259, t260 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t254,
                t91,
                t92,
                t93,
                t94,
                True,
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
        del t254, t94, t93, t92, t91

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t261 = paddle._C_ops.add(t248, t255)
        del t248, t255

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t262 = paddle._C_ops.relu(t261)
        del t261

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t263 = paddle._C_ops.conv2d(
            t262, t95, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t264, t265, t266, t267, t268, t269 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t263,
                t96,
                t97,
                t98,
                t99,
                True,
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
        del t263, t96, t99, t98, t97

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t270 = paddle._C_ops.relu(t264)
        del t264

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t271 = paddle._C_ops.conv2d(
            t270, t100, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100, t270

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t272, t273, t274, t275, t276, t277 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t271,
                t101,
                t102,
                t103,
                t104,
                True,
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
        del t271, t104, t103, t102, t101

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        t278 = paddle._C_ops.add(t272, t262)
        del t272, t262

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t279 = paddle._C_ops.relu(t278)
        del t278

        # pd_op.full_int_array: (2xi64) <- ()
        t280 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x7x7xf32, 2xi64)
        t281 = paddle._C_ops.pool2d(
            t279,
            t280,
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
        del t280, t279

        # pd_op.flatten: (-1x512xf32) <- (-1x512x1x1xf32)
        t282 = paddle._C_ops.flatten(t281, 1, 3)
        del t281

        # pd_op.matmul: (-1x102xf32) <- (-1x512xf32, 512x102xf32)
        t283 = paddle._C_ops.matmul(t282, t105, False, False)
        del t282, t105

        return t283
