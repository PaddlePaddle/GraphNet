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
        input21: paddle.Tensor,
        input22: paddle.Tensor,
        t55: paddle.Tensor,
        t56: paddle.Tensor,
        t57: paddle.Tensor,
        t58: paddle.Tensor,
        t59: paddle.Tensor,
        input23: paddle.Tensor,
        input24: paddle.Tensor,
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
        input31: paddle.Tensor,
        input32: paddle.Tensor,
        t85: paddle.Tensor,
        t86: paddle.Tensor,
        t87: paddle.Tensor,
        t88: paddle.Tensor,
        t89: paddle.Tensor,
        input33: paddle.Tensor,
        input34: paddle.Tensor,
        t90: paddle.Tensor,
        t91: paddle.Tensor,
        t92: paddle.Tensor,
        t93: paddle.Tensor,
        t94: paddle.Tensor,
        input35: paddle.Tensor,
        input36: paddle.Tensor,
        t95: paddle.Tensor,
        t96: paddle.Tensor,
        t97: paddle.Tensor,
        t98: paddle.Tensor,
        t99: paddle.Tensor,
        input37: paddle.Tensor,
        input38: paddle.Tensor,
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
        input39: paddle.Tensor,
        input40: paddle.Tensor,
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
        input41: paddle.Tensor,
        input42: paddle.Tensor,
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
        input43: paddle.Tensor,
        input44: paddle.Tensor,
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
        input45: paddle.Tensor,
        input46: paddle.Tensor,
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
        input47: paddle.Tensor,
        input48: paddle.Tensor,
        t155: paddle.Tensor,
        t156: paddle.Tensor,
        t157: paddle.Tensor,
        t158: paddle.Tensor,
        t159: paddle.Tensor,
        input49: paddle.Tensor,
        input50: paddle.Tensor,
        t160: paddle.Tensor,
        t161: paddle.Tensor,
        t162: paddle.Tensor,
        t163: paddle.Tensor,
        t164: paddle.Tensor,
        input51: paddle.Tensor,
        input52: paddle.Tensor,
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
        input53: paddle.Tensor,
        input54: paddle.Tensor,
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
        input55: paddle.Tensor,
        input56: paddle.Tensor,
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
        input57: paddle.Tensor,
        input58: paddle.Tensor,
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
        input59: paddle.Tensor,
        input60: paddle.Tensor,
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
        input61: paddle.Tensor,
        input62: paddle.Tensor,
        t215: paddle.Tensor,
        t216: paddle.Tensor,
        t217: paddle.Tensor,
        t218: paddle.Tensor,
        t219: paddle.Tensor,
        input63: paddle.Tensor,
        input64: paddle.Tensor,
        t220: paddle.Tensor,
        t221: paddle.Tensor,
        t222: paddle.Tensor,
        t223: paddle.Tensor,
        t224: paddle.Tensor,
        input65: paddle.Tensor,
        input66: paddle.Tensor,
        t225: paddle.Tensor,
        t226: paddle.Tensor,
        t227: paddle.Tensor,
        t228: paddle.Tensor,
        t229: paddle.Tensor,
        t230: paddle.Tensor,
        t231: paddle.Tensor,
        t232: paddle.Tensor,
        t233: paddle.Tensor,
        t234: paddle.Tensor,
        input67: paddle.Tensor,
        input68: paddle.Tensor,
        t235: paddle.Tensor,
        t236: paddle.Tensor,
        t237: paddle.Tensor,
        t238: paddle.Tensor,
        t239: paddle.Tensor,
        t240: paddle.Tensor,
        t241: paddle.Tensor,
        t242: paddle.Tensor,
        t243: paddle.Tensor,
        t244: paddle.Tensor,
        input69: paddle.Tensor,
        input70: paddle.Tensor,
        t245: paddle.Tensor,
        t246: paddle.Tensor,
        t247: paddle.Tensor,
        t248: paddle.Tensor,
        t249: paddle.Tensor,
        t250: paddle.Tensor,
        t251: paddle.Tensor,
        t252: paddle.Tensor,
        t253: paddle.Tensor,
        t254: paddle.Tensor,
        input71: paddle.Tensor,
        input72: paddle.Tensor,
        t255: paddle.Tensor,
        t256: paddle.Tensor,
        t257: paddle.Tensor,
        t258: paddle.Tensor,
        t259: paddle.Tensor,
        t260: paddle.Tensor,
        t261: paddle.Tensor,
        t262: paddle.Tensor,
        t263: paddle.Tensor,
        t264: paddle.Tensor,
        input73: paddle.Tensor,
        input74: paddle.Tensor,
        t265: paddle.Tensor,
        t266: paddle.Tensor,
        t267: paddle.Tensor,
        t268: paddle.Tensor,
        t269: paddle.Tensor,
        t270: paddle.Tensor,
        t271: paddle.Tensor,
        t272: paddle.Tensor,
        t273: paddle.Tensor,
        t274: paddle.Tensor,
        input75: paddle.Tensor,
        input76: paddle.Tensor,
        t275: paddle.Tensor,
        t276: paddle.Tensor,
        t277: paddle.Tensor,
        t278: paddle.Tensor,
        t279: paddle.Tensor,
        input77: paddle.Tensor,
        input78: paddle.Tensor,
        t280: paddle.Tensor,
        t281: paddle.Tensor,
        t282: paddle.Tensor,
        t283: paddle.Tensor,
        t284: paddle.Tensor,
        input79: paddle.Tensor,
        input80: paddle.Tensor,
        t285: paddle.Tensor,
        t286: paddle.Tensor,
        t287: paddle.Tensor,
        t288: paddle.Tensor,
        t289: paddle.Tensor,
        t290: paddle.Tensor,
        t291: paddle.Tensor,
        t292: paddle.Tensor,
        t293: paddle.Tensor,
        t294: paddle.Tensor,
        t295: paddle.Tensor,
        t296: paddle.Tensor,
        t297: paddle.Tensor,
        t298: paddle.Tensor,
        t299: paddle.Tensor,
        input81: paddle.Tensor,
        input82: paddle.Tensor,
        t300: paddle.Tensor,
        t301: paddle.Tensor,
        t302: paddle.Tensor,
        t303: paddle.Tensor,
        t304: paddle.Tensor,
        t305: paddle.Tensor,
        t306: paddle.Tensor,
        t307: paddle.Tensor,
        t308: paddle.Tensor,
        t309: paddle.Tensor,
        input83: paddle.Tensor,
        input84: paddle.Tensor,
        t310: paddle.Tensor,
        t311: paddle.Tensor,
        t312: paddle.Tensor,
        t313: paddle.Tensor,
        t314: paddle.Tensor,
        t315: paddle.Tensor,
        t316: paddle.Tensor,
        t317: paddle.Tensor,
        t318: paddle.Tensor,
        t319: paddle.Tensor,
        input85: paddle.Tensor,
        input86: paddle.Tensor,
        t320: paddle.Tensor,
        t321: paddle.Tensor,
        t322: paddle.Tensor,
        t323: paddle.Tensor,
        t324: paddle.Tensor,
        t325: paddle.Tensor,
        t326: paddle.Tensor,
        t327: paddle.Tensor,
        t328: paddle.Tensor,
        t329: paddle.Tensor,
        input87: paddle.Tensor,
        input88: paddle.Tensor,
        t330: paddle.Tensor,
        t331: paddle.Tensor,
        t332: paddle.Tensor,
        t333: paddle.Tensor,
        t334: paddle.Tensor,
        t335: paddle.Tensor,
        t336: paddle.Tensor,
        t337: paddle.Tensor,
        t338: paddle.Tensor,
        t339: paddle.Tensor,
        input89: paddle.Tensor,
        input90: paddle.Tensor,
        t340: paddle.Tensor,
        t341: paddle.Tensor,
        t342: paddle.Tensor,
        t343: paddle.Tensor,
        t344: paddle.Tensor,
        input91: paddle.Tensor,
        input92: paddle.Tensor,
        t345: paddle.Tensor,
        t346: paddle.Tensor,
        t347: paddle.Tensor,
        t348: paddle.Tensor,
        t349: paddle.Tensor,
        input93: paddle.Tensor,
        input94: paddle.Tensor,
        t350: paddle.Tensor,
        input95: paddle.Tensor,
        input96: paddle.Tensor,
        t351: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x3x224x224xf32, 24x3x3x3xf32)
        t352 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t353, t354, t355, t356, t357, t358 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t352,
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

        # pd_op.relu: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        t359 = paddle._C_ops.relu(t353)
        del t353

        # pd_op.multiply: (-1x24x112x112xf32) <- (1xf32, -1x24x112x112xf32)
        t360 = paddle._C_ops.multiply(input1, t359)
        del input1

        # pd_op.add: (-1x24x112x112xf32) <- (-1x24x112x112xf32, 1xf32)
        t361 = paddle._C_ops.add(t360, input2)
        del input2

        # pd_op.conv2d: (-1x12x112x112xf32) <- (-1x24x112x112xf32, 12x24x2x2xf32)
        t362 = paddle._C_ops.conv2d(t361, t5, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW")
        del t5

        # pd_op.batch_norm_: (-1x12x112x112xf32, 12xf32, 12xf32, 12xf32, 12xf32, -1xui8) <- (-1x12x112x112xf32, 12xf32, 12xf32, 12xf32, 12xf32)
        t363, t364, t365, t366, t367, t368 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t362,
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

        # pd_op.relu: (-1x12x112x112xf32) <- (-1x12x112x112xf32)
        t369 = paddle._C_ops.relu(t363)
        del t363

        # pd_op.multiply: (-1x12x112x112xf32) <- (1xf32, -1x12x112x112xf32)
        t370 = paddle._C_ops.multiply(input3, t369)
        del input3

        # pd_op.add: (-1x12x112x112xf32) <- (-1x12x112x112xf32, 1xf32)
        t371 = paddle._C_ops.add(t370, input4)
        del input4

        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x12x112x112xf32, 24x12x2x2xf32)
        t372 = paddle._C_ops.conv2d(
            t371, t10, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t373, t374, t375, t376, t377, t378 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t372,
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

        # pd_op.relu: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        t379 = paddle._C_ops.relu(t373)
        del t373

        # pd_op.multiply: (-1x24x112x112xf32) <- (1xf32, -1x24x112x112xf32)
        t380 = paddle._C_ops.multiply(input5, t379)
        del input5

        # pd_op.add: (-1x24x112x112xf32) <- (-1x24x112x112xf32, 1xf32)
        t381 = paddle._C_ops.add(t380, input6)
        del input6

        # pd_op.full_int_array: (2xi64) <- ()
        t382 = [2, 2]

        # pd_op.pool2d: (-1x24x112x112xf32) <- (-1x24x112x112xf32, 2xi64)
        t383 = paddle._C_ops.pool2d(
            t361, t382, [1, 1], [0, 0], True, True, "NCHW", "max", False, False, "SAME"
        )

        # pd_op.full: (1xi32) <- ()
        t384 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t385 = t384

        # pd_op.assign: (1xi32) <- (1xi32)
        t386 = t384

        # pd_op.assign: (1xi32) <- (1xi32)
        t387 = t384

        # pd_op.assign: (1xi32) <- (1xi32)
        t388 = t384

        # pd_op.assign: (1xi32) <- (1xi32)
        t389 = t384

        # pd_op.assign: (1xi32) <- (1xi32)
        t390 = t384

        # builtin.combine: ([-1x24x112x112xf32, -1x24x112x112xf32]) <- (-1x24x112x112xf32, -1x24x112x112xf32)
        t391 = [t383, t381]

        # pd_op.concat: (-1x48x112x112xf32) <- ([-1x24x112x112xf32, -1x24x112x112xf32], 1xi32)
        t392 = paddle._C_ops.concat(t391, t384)
        del t391

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x48x112x112xf32, 24x48x3x3xf32)
        t393 = paddle._C_ops.conv2d(
            t392, t15, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t394, t395, t396, t397, t398, t399 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t393,
                t16,
                t17,
                t18,
                t19,
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
        del t19, t18, t17, t16

        # pd_op.relu: (-1x24x56x56xf32) <- (-1x24x56x56xf32)
        t400 = paddle._C_ops.relu(t394)
        del t394

        # pd_op.multiply: (-1x24x56x56xf32) <- (1xf32, -1x24x56x56xf32)
        t401 = paddle._C_ops.multiply(input7, t400)
        del input7

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1xf32)
        t402 = paddle._C_ops.add(t401, input8)
        del input8

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x24x56x56xf32, 32x24x1x1xf32)
        t403 = paddle._C_ops.conv2d(
            t402, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t404, t405, t406, t407, t408, t409 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t403,
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

        # pd_op.relu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        t410 = paddle._C_ops.relu(t404)
        del t404

        # pd_op.multiply: (-1x32x56x56xf32) <- (1xf32, -1x32x56x56xf32)
        t411 = paddle._C_ops.multiply(input9, t410)
        del input9

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1xf32)
        t412 = paddle._C_ops.add(t411, input10)
        del input10

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x3x3xf32)
        t413 = paddle._C_ops.conv2d(
            t412, t25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t414, t415, t416, t417, t418, t419 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t413,
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

        # pd_op.relu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        t420 = paddle._C_ops.relu(t414)
        del t414

        # pd_op.multiply: (-1x32x56x56xf32) <- (1xf32, -1x32x56x56xf32)
        t421 = paddle._C_ops.multiply(input11, t420)
        del input11

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1xf32)
        t422 = paddle._C_ops.add(t421, input12)
        del input12

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x3x3xf32)
        t423 = paddle._C_ops.conv2d(
            t422, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t424, t425, t426, t427, t428, t429 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t423,
                t31,
                t32,
                t33,
                t34,
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
        del t34, t33, t32, t31

        # pd_op.relu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        t430 = paddle._C_ops.relu(t424)
        del t424

        # pd_op.multiply: (-1x32x56x56xf32) <- (1xf32, -1x32x56x56xf32)
        t431 = paddle._C_ops.multiply(input13, t430)
        del input13

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1xf32)
        t432 = paddle._C_ops.add(t431, input14)
        del input14

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x3x3xf32)
        t433 = paddle._C_ops.conv2d(
            t432, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t434, t435, t436, t437, t438, t439 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t433,
                t36,
                t37,
                t38,
                t39,
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
        del t39, t38, t37, t36

        # pd_op.relu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        t440 = paddle._C_ops.relu(t434)
        del t434

        # pd_op.multiply: (-1x32x56x56xf32) <- (1xf32, -1x32x56x56xf32)
        t441 = paddle._C_ops.multiply(input15, t440)
        del input15

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1xf32)
        t442 = paddle._C_ops.add(t441, input16)
        del input16

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x3x3xf32)
        t443 = paddle._C_ops.conv2d(
            t442, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t444, t445, t446, t447, t448, t449 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t443,
                t41,
                t42,
                t43,
                t44,
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
        del t44, t43, t42, t41

        # pd_op.relu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        t450 = paddle._C_ops.relu(t444)
        del t444

        # pd_op.multiply: (-1x32x56x56xf32) <- (1xf32, -1x32x56x56xf32)
        t451 = paddle._C_ops.multiply(input17, t450)
        del input17

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1xf32)
        t452 = paddle._C_ops.add(t451, input18)
        del input18

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x3x3xf32)
        t453 = paddle._C_ops.conv2d(
            t452, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t454, t455, t456, t457, t458, t459 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t453,
                t46,
                t47,
                t48,
                t49,
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
        del t49, t48, t47, t46

        # pd_op.relu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        t460 = paddle._C_ops.relu(t454)
        del t454

        # pd_op.multiply: (-1x32x56x56xf32) <- (1xf32, -1x32x56x56xf32)
        t461 = paddle._C_ops.multiply(input19, t460)
        del input19

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1xf32)
        t462 = paddle._C_ops.add(t461, input20)
        del input20

        # builtin.combine: ([-1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32]) <- (-1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32)
        t463 = [t412, t422, t432, t442, t452, t462]

        # pd_op.concat: (-1x192x56x56xf32) <- ([-1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32, -1x32x56x56xf32], 1xi32)
        t464 = paddle._C_ops.concat(t463, t384)
        del t463

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x192x56x56xf32, 64x192x1x1xf32)
        t465 = paddle._C_ops.conv2d(
            t464, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t466, t467, t468, t469, t470, t471 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t465,
                t51,
                t52,
                t53,
                t54,
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
        del t54, t53, t52, t51

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t472 = paddle._C_ops.relu(t466)
        del t466

        # pd_op.multiply: (-1x64x56x56xf32) <- (1xf32, -1x64x56x56xf32)
        t473 = paddle._C_ops.multiply(input21, t472)
        del input21

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1xf32)
        t474 = paddle._C_ops.add(t473, input22)
        del input22

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x64x56x56xf32, 128x64x1x1xf32)
        t475 = paddle._C_ops.conv2d(
            t474, t55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t476, t477, t478, t479, t480, t481 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t475,
                t56,
                t57,
                t58,
                t59,
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
        del t59, t58, t57, t56

        # pd_op.relu: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        t482 = paddle._C_ops.relu(t476)
        del t476

        # pd_op.multiply: (-1x128x56x56xf32) <- (1xf32, -1x128x56x56xf32)
        t483 = paddle._C_ops.multiply(input23, t482)
        del input23

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1xf32)
        t484 = paddle._C_ops.add(t483, input24)
        del input24

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x56x56xf32, 128x1x3x3xf32)
        t485 = paddle._C_ops.depthwise_conv2d(
            t484, t60, [2, 2], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t486, t487, t488, t489, t490, t491 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t485,
                t61,
                t62,
                t63,
                t64,
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
        del t64, t63, t62, t61

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x128x28x28xf32, 64x128x3x3xf32)
        t492 = paddle._C_ops.conv2d(
            t486, t65, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t493, t494, t495, t496, t497, t498 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t492,
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

        # pd_op.relu: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        t499 = paddle._C_ops.relu(t493)
        del t493

        # pd_op.multiply: (-1x64x28x28xf32) <- (1xf32, -1x64x28x28xf32)
        t500 = paddle._C_ops.multiply(input25, t499)
        del input25

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1xf32)
        t501 = paddle._C_ops.add(t500, input26)
        del input26

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x3x3xf32)
        t502 = paddle._C_ops.conv2d(
            t501, t70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t503, t504, t505, t506, t507, t508 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t502,
                t71,
                t72,
                t73,
                t74,
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
        del t74, t73, t72, t71

        # pd_op.relu: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        t509 = paddle._C_ops.relu(t503)
        del t503

        # pd_op.multiply: (-1x64x28x28xf32) <- (1xf32, -1x64x28x28xf32)
        t510 = paddle._C_ops.multiply(input27, t509)
        del input27

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1xf32)
        t511 = paddle._C_ops.add(t510, input28)
        del input28

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x3x3xf32)
        t512 = paddle._C_ops.conv2d(
            t511, t75, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t513, t514, t515, t516, t517, t518 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t512,
                t76,
                t77,
                t78,
                t79,
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
        del t79, t78, t77, t76

        # pd_op.relu: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        t519 = paddle._C_ops.relu(t513)
        del t513

        # pd_op.multiply: (-1x64x28x28xf32) <- (1xf32, -1x64x28x28xf32)
        t520 = paddle._C_ops.multiply(input29, t519)
        del input29

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1xf32)
        t521 = paddle._C_ops.add(t520, input30)
        del input30

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x3x3xf32)
        t522 = paddle._C_ops.conv2d(
            t521, t80, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t523, t524, t525, t526, t527, t528 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t522,
                t81,
                t82,
                t83,
                t84,
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
        del t84, t83, t82, t81

        # pd_op.relu: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        t529 = paddle._C_ops.relu(t523)
        del t523

        # pd_op.multiply: (-1x64x28x28xf32) <- (1xf32, -1x64x28x28xf32)
        t530 = paddle._C_ops.multiply(input31, t529)
        del input31

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1xf32)
        t531 = paddle._C_ops.add(t530, input32)
        del input32

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x3x3xf32)
        t532 = paddle._C_ops.conv2d(
            t531, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t533, t534, t535, t536, t537, t538 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t532,
                t86,
                t87,
                t88,
                t89,
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
        del t89, t88, t87, t86

        # pd_op.relu: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        t539 = paddle._C_ops.relu(t533)
        del t533

        # pd_op.multiply: (-1x64x28x28xf32) <- (1xf32, -1x64x28x28xf32)
        t540 = paddle._C_ops.multiply(input33, t539)
        del input33

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1xf32)
        t541 = paddle._C_ops.add(t540, input34)
        del input34

        # builtin.combine: ([-1x128x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32]) <- (-1x128x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32)
        t542 = [t486, t501, t511, t521, t531, t541]

        # pd_op.concat: (-1x448x28x28xf32) <- ([-1x128x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32, -1x64x28x28xf32], 1xi32)
        t543 = paddle._C_ops.concat(t542, t384)
        del t542

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x448x28x28xf32, 256x448x1x1xf32)
        t544 = paddle._C_ops.conv2d(
            t543, t90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t545, t546, t547, t548, t549, t550 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t544,
                t91,
                t92,
                t93,
                t94,
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
        del t94, t93, t92, t91

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t551 = paddle._C_ops.relu(t545)
        del t545

        # pd_op.multiply: (-1x256x28x28xf32) <- (1xf32, -1x256x28x28xf32)
        t552 = paddle._C_ops.multiply(input35, t551)
        del input35

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1xf32)
        t553 = paddle._C_ops.add(t552, input36)
        del input36

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        t554 = paddle._C_ops.conv2d(
            t553, t95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t555, t556, t557, t558, t559, t560 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t554,
                t96,
                t97,
                t98,
                t99,
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
        del t99, t98, t97, t96

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t561 = paddle._C_ops.relu(t555)
        del t555

        # pd_op.multiply: (-1x512x28x28xf32) <- (1xf32, -1x512x28x28xf32)
        t562 = paddle._C_ops.multiply(input37, t561)
        del input37

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1xf32)
        t563 = paddle._C_ops.add(t562, input38)
        del input38

        # pd_op.depthwise_conv2d: (-1x512x14x14xf32) <- (-1x512x28x28xf32, 512x1x3x3xf32)
        t564 = paddle._C_ops.depthwise_conv2d(
            t563, t100, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t565, t566, t567, t568, t569, t570 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t564,
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

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        t571 = paddle._C_ops.conv2d(
            t565, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t572, t573, t574, t575, t576, t577 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t571,
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

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t578 = paddle._C_ops.depthwise_conv2d(
            t572, t110, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t579, t580, t581, t582, t583, t584 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t578,
                t111,
                t112,
                t113,
                t114,
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
        del t114, t113, t112, t111

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t585 = paddle._C_ops.relu(t579)
        del t579

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t586 = paddle._C_ops.multiply(input39, t585)
        del input39

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t587 = paddle._C_ops.add(t586, input40)
        del input40

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t588 = paddle._C_ops.conv2d(
            t587, t115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t589, t590, t591, t592, t593, t594 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t588,
                t116,
                t117,
                t118,
                t119,
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
        del t119, t118, t117, t116

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t595 = paddle._C_ops.depthwise_conv2d(
            t589, t120, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t596, t597, t598, t599, t600, t601 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t595,
                t121,
                t122,
                t123,
                t124,
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
        del t124, t123, t122, t121

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t602 = paddle._C_ops.relu(t596)
        del t596

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t603 = paddle._C_ops.multiply(input41, t602)
        del input41

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t604 = paddle._C_ops.add(t603, input42)
        del input42

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t605 = paddle._C_ops.conv2d(
            t604, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t606, t607, t608, t609, t610, t611 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t605,
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

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t612 = paddle._C_ops.depthwise_conv2d(
            t606, t130, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t613, t614, t615, t616, t617, t618 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t612,
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

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t619 = paddle._C_ops.relu(t613)
        del t613

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t620 = paddle._C_ops.multiply(input43, t619)
        del input43

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t621 = paddle._C_ops.add(t620, input44)
        del input44

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t622 = paddle._C_ops.conv2d(
            t621, t135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t623, t624, t625, t626, t627, t628 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t622,
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

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t629 = paddle._C_ops.depthwise_conv2d(
            t623, t140, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t630, t631, t632, t633, t634, t635 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t629,
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

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t636 = paddle._C_ops.relu(t630)
        del t630

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t637 = paddle._C_ops.multiply(input45, t636)
        del input45

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t638 = paddle._C_ops.add(t637, input46)
        del input46

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t639 = paddle._C_ops.conv2d(
            t638, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t640, t641, t642, t643, t644, t645 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t639,
                t146,
                t147,
                t148,
                t149,
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
        del t149, t148, t147, t146

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t646 = paddle._C_ops.depthwise_conv2d(
            t640, t150, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t647, t648, t649, t650, t651, t652 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t646,
                t151,
                t152,
                t153,
                t154,
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
        del t154, t153, t152, t151

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t653 = paddle._C_ops.relu(t647)
        del t647

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t654 = paddle._C_ops.multiply(input47, t653)
        del input47

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t655 = paddle._C_ops.add(t654, input48)
        del input48

        # builtin.combine: ([-1x512x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32]) <- (-1x512x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32)
        t656 = [t565, t587, t604, t621, t638, t655]

        # pd_op.concat: (-1x1152x14x14xf32) <- ([-1x512x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32], 1xi32)
        t657 = paddle._C_ops.concat(t656, t384)
        del t656

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1152x14x14xf32, 512x1152x1x1xf32)
        t658 = paddle._C_ops.conv2d(
            t657, t155, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t659, t660, t661, t662, t663, t664 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t658,
                t156,
                t157,
                t158,
                t159,
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
        del t159, t158, t157, t156

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t665 = paddle._C_ops.relu(t659)
        del t659

        # pd_op.multiply: (-1x512x14x14xf32) <- (1xf32, -1x512x14x14xf32)
        t666 = paddle._C_ops.multiply(input49, t665)
        del input49

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1xf32)
        t667 = paddle._C_ops.add(t666, input50)
        del input50

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t668 = paddle._C_ops.conv2d(
            t667, t160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t669, t670, t671, t672, t673, t674 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t668,
                t161,
                t162,
                t163,
                t164,
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
        del t164, t163, t162, t161

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t675 = paddle._C_ops.relu(t669)
        del t669

        # pd_op.multiply: (-1x1024x14x14xf32) <- (1xf32, -1x1024x14x14xf32)
        t676 = paddle._C_ops.multiply(input51, t675)
        del input51

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, 1xf32)
        t677 = paddle._C_ops.add(t676, input52)
        del input52

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x1024x14x14xf32, 128x1024x1x1xf32)
        t678 = paddle._C_ops.conv2d(
            t677, t165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t679, t680, t681, t682, t683, t684 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t678,
                t166,
                t167,
                t168,
                t169,
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
        del t169, t168, t167, t166

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t685 = paddle._C_ops.depthwise_conv2d(
            t679, t170, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t686, t687, t688, t689, t690, t691 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t685,
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

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t692 = paddle._C_ops.relu(t686)
        del t686

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t693 = paddle._C_ops.multiply(input53, t692)
        del input53

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t694 = paddle._C_ops.add(t693, input54)
        del input54

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t695 = paddle._C_ops.conv2d(
            t694, t175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t696, t697, t698, t699, t700, t701 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t695,
                t176,
                t177,
                t178,
                t179,
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
        del t179, t178, t177, t176

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t702 = paddle._C_ops.depthwise_conv2d(
            t696, t180, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t180

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t703, t704, t705, t706, t707, t708 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t702,
                t181,
                t182,
                t183,
                t184,
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
        del t184, t183, t182, t181

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t709 = paddle._C_ops.relu(t703)
        del t703

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t710 = paddle._C_ops.multiply(input55, t709)
        del input55

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t711 = paddle._C_ops.add(t710, input56)
        del input56

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t712 = paddle._C_ops.conv2d(
            t711, t185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t713, t714, t715, t716, t717, t718 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t712,
                t186,
                t187,
                t188,
                t189,
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
        del t189, t188, t187, t186

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t719 = paddle._C_ops.depthwise_conv2d(
            t713, t190, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t720, t721, t722, t723, t724, t725 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t719,
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

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t726 = paddle._C_ops.relu(t720)
        del t720

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t727 = paddle._C_ops.multiply(input57, t726)
        del input57

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t728 = paddle._C_ops.add(t727, input58)
        del input58

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t729 = paddle._C_ops.conv2d(
            t728, t195, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t730, t731, t732, t733, t734, t735 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t729,
                t196,
                t197,
                t198,
                t199,
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
        del t199, t198, t197, t196

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t736 = paddle._C_ops.depthwise_conv2d(
            t730, t200, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t737, t738, t739, t740, t741, t742 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t736,
                t201,
                t202,
                t203,
                t204,
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
        del t204, t203, t202, t201

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t743 = paddle._C_ops.relu(t737)
        del t737

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t744 = paddle._C_ops.multiply(input59, t743)
        del input59

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t745 = paddle._C_ops.add(t744, input60)
        del input60

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t746 = paddle._C_ops.conv2d(
            t745, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t747, t748, t749, t750, t751, t752 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t746,
                t206,
                t207,
                t208,
                t209,
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
        del t209, t208, t207, t206

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t753 = paddle._C_ops.depthwise_conv2d(
            t747, t210, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t754, t755, t756, t757, t758, t759 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t753,
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

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t760 = paddle._C_ops.relu(t754)
        del t754

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t761 = paddle._C_ops.multiply(input61, t760)
        del input61

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t762 = paddle._C_ops.add(t761, input62)
        del input62

        # builtin.combine: ([-1x1024x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32]) <- (-1x1024x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32)
        t763 = [t677, t694, t711, t728, t745, t762]

        # pd_op.concat: (-1x1664x14x14xf32) <- ([-1x1024x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32], 1xi32)
        t764 = paddle._C_ops.concat(t763, t384)
        del t763

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1664x14x14xf32, 512x1664x1x1xf32)
        t765 = paddle._C_ops.conv2d(
            t764, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t766, t767, t768, t769, t770, t771 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t765,
                t216,
                t217,
                t218,
                t219,
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
        del t219, t218, t217, t216

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t772 = paddle._C_ops.relu(t766)
        del t766

        # pd_op.multiply: (-1x512x14x14xf32) <- (1xf32, -1x512x14x14xf32)
        t773 = paddle._C_ops.multiply(input63, t772)
        del input63

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1xf32)
        t774 = paddle._C_ops.add(t773, input64)
        del input64

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t775 = paddle._C_ops.conv2d(
            t774, t220, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t220

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t776, t777, t778, t779, t780, t781 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t775,
                t221,
                t222,
                t223,
                t224,
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
        del t224, t223, t222, t221

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t782 = paddle._C_ops.relu(t776)
        del t776

        # pd_op.multiply: (-1x1024x14x14xf32) <- (1xf32, -1x1024x14x14xf32)
        t783 = paddle._C_ops.multiply(input65, t782)
        del input65

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, 1xf32)
        t784 = paddle._C_ops.add(t783, input66)
        del input66

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t785 = paddle._C_ops.add(t784, t677)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x1024x14x14xf32, 128x1024x1x1xf32)
        t786 = paddle._C_ops.conv2d(
            t785, t225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t787, t788, t789, t790, t791, t792 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t786,
                t226,
                t227,
                t228,
                t229,
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
        del t229, t228, t227, t226

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t793 = paddle._C_ops.depthwise_conv2d(
            t787, t230, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t230

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t794, t795, t796, t797, t798, t799 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t793,
                t231,
                t232,
                t233,
                t234,
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
        del t234, t233, t232, t231

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t800 = paddle._C_ops.relu(t794)
        del t794

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t801 = paddle._C_ops.multiply(input67, t800)
        del input67

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t802 = paddle._C_ops.add(t801, input68)
        del input68

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t803 = paddle._C_ops.conv2d(
            t802, t235, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t804, t805, t806, t807, t808, t809 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t803,
                t236,
                t237,
                t238,
                t239,
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
        del t239, t238, t237, t236

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t810 = paddle._C_ops.depthwise_conv2d(
            t804, t240, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t240

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t811, t812, t813, t814, t815, t816 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t810,
                t241,
                t242,
                t243,
                t244,
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
        del t244, t243, t242, t241

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t817 = paddle._C_ops.relu(t811)
        del t811

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t818 = paddle._C_ops.multiply(input69, t817)
        del input69

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t819 = paddle._C_ops.add(t818, input70)
        del input70

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t820 = paddle._C_ops.conv2d(
            t819, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t821, t822, t823, t824, t825, t826 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t820,
                t246,
                t247,
                t248,
                t249,
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
        del t249, t248, t247, t246

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t827 = paddle._C_ops.depthwise_conv2d(
            t821, t250, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t250

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t828, t829, t830, t831, t832, t833 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t827,
                t251,
                t252,
                t253,
                t254,
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
        del t252, t251, t254, t253

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t834 = paddle._C_ops.relu(t828)
        del t828

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t835 = paddle._C_ops.multiply(input71, t834)
        del input71

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t836 = paddle._C_ops.add(t835, input72)
        del input72

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t837 = paddle._C_ops.conv2d(
            t836, t255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t838, t839, t840, t841, t842, t843 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t837,
                t256,
                t257,
                t258,
                t259,
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
        del t259, t258, t257, t256

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t844 = paddle._C_ops.depthwise_conv2d(
            t838, t260, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t260

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t845, t846, t847, t848, t849, t850 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t844,
                t261,
                t262,
                t263,
                t264,
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
        del t264, t263, t262, t261

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t851 = paddle._C_ops.relu(t845)
        del t845

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t852 = paddle._C_ops.multiply(input73, t851)
        del input73

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t853 = paddle._C_ops.add(t852, input74)
        del input74

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x1x1xf32)
        t854 = paddle._C_ops.conv2d(
            t853, t265, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t265

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t855, t856, t857, t858, t859, t860 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t854,
                t266,
                t267,
                t268,
                t269,
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
        del t269, t268, t267, t266

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x5x5xf32)
        t861 = paddle._C_ops.depthwise_conv2d(
            t855, t270, [1, 1], [2, 2], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t270

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t862, t863, t864, t865, t866, t867 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t861,
                t271,
                t272,
                t273,
                t274,
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
        del t274, t273, t272, t271

        # pd_op.relu: (-1x128x14x14xf32) <- (-1x128x14x14xf32)
        t868 = paddle._C_ops.relu(t862)
        del t862

        # pd_op.multiply: (-1x128x14x14xf32) <- (1xf32, -1x128x14x14xf32)
        t869 = paddle._C_ops.multiply(input75, t868)
        del input75

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1xf32)
        t870 = paddle._C_ops.add(t869, input76)
        del input76

        # builtin.combine: ([-1x1024x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32]) <- (-1x1024x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32)
        t871 = [t785, t802, t819, t836, t853, t870]

        # pd_op.concat: (-1x1664x14x14xf32) <- ([-1x1024x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32, -1x128x14x14xf32], 1xi32)
        t872 = paddle._C_ops.concat(t871, t384)
        del t871

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1664x14x14xf32, 512x1664x1x1xf32)
        t873 = paddle._C_ops.conv2d(
            t872, t275, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t275

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t874, t875, t876, t877, t878, t879 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t873,
                t276,
                t277,
                t278,
                t279,
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
        del t279, t278, t277, t276

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t880 = paddle._C_ops.relu(t874)
        del t874

        # pd_op.multiply: (-1x512x14x14xf32) <- (1xf32, -1x512x14x14xf32)
        t881 = paddle._C_ops.multiply(input77, t880)
        del input77

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1xf32)
        t882 = paddle._C_ops.add(t881, input78)
        del input78

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t883 = paddle._C_ops.conv2d(
            t882, t280, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t280

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t884, t885, t886, t887, t888, t889 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t883,
                t281,
                t282,
                t283,
                t284,
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
        del t284, t283, t282, t281

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t890 = paddle._C_ops.relu(t884)
        del t884

        # pd_op.multiply: (-1x1024x14x14xf32) <- (1xf32, -1x1024x14x14xf32)
        t891 = paddle._C_ops.multiply(input79, t890)
        del input79

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, 1xf32)
        t892 = paddle._C_ops.add(t891, input80)
        del input80

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t893 = paddle._C_ops.add(t892, t785)

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf32) <- (-1x1024x14x14xf32, 1024x1x3x3xf32)
        t894 = paddle._C_ops.depthwise_conv2d(
            t893, t285, [2, 2], [1, 1], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del t285

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t895, t896, t897, t898, t899, t900 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t894,
                t286,
                t287,
                t288,
                t289,
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
        del t289, t288, t287, t286

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        t901 = paddle._C_ops.conv2d(
            t895, t290, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t290

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t902, t903, t904, t905, t906, t907 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t901,
                t291,
                t292,
                t293,
                t294,
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
        del t294, t293, t292, t291

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x5x5xf32)
        t908 = paddle._C_ops.depthwise_conv2d(
            t902, t295, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t295

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t909, t910, t911, t912, t913, t914 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t908,
                t296,
                t297,
                t298,
                t299,
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
        del t299, t298, t297, t296

        # pd_op.relu: (-1x256x7x7xf32) <- (-1x256x7x7xf32)
        t915 = paddle._C_ops.relu(t909)
        del t909

        # pd_op.multiply: (-1x256x7x7xf32) <- (1xf32, -1x256x7x7xf32)
        t916 = paddle._C_ops.multiply(input81, t915)
        del input81

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1xf32)
        t917 = paddle._C_ops.add(t916, input82)
        del input82

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        t918 = paddle._C_ops.conv2d(
            t917, t300, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t300

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t919, t920, t921, t922, t923, t924 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t918,
                t301,
                t302,
                t303,
                t304,
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
        del t304, t303, t302, t301

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x5x5xf32)
        t925 = paddle._C_ops.depthwise_conv2d(
            t919, t305, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t305

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t926, t927, t928, t929, t930, t931 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t925,
                t306,
                t307,
                t308,
                t309,
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
        del t309, t308, t307, t306

        # pd_op.relu: (-1x256x7x7xf32) <- (-1x256x7x7xf32)
        t932 = paddle._C_ops.relu(t926)
        del t926

        # pd_op.multiply: (-1x256x7x7xf32) <- (1xf32, -1x256x7x7xf32)
        t933 = paddle._C_ops.multiply(input83, t932)
        del input83

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1xf32)
        t934 = paddle._C_ops.add(t933, input84)
        del input84

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        t935 = paddle._C_ops.conv2d(
            t934, t310, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t310

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t936, t937, t938, t939, t940, t941 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t935,
                t311,
                t312,
                t313,
                t314,
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
        del t314, t313, t312, t311

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x5x5xf32)
        t942 = paddle._C_ops.depthwise_conv2d(
            t936, t315, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t315

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t943, t944, t945, t946, t947, t948 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t942,
                t316,
                t317,
                t318,
                t319,
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
        del t319, t318, t317, t316

        # pd_op.relu: (-1x256x7x7xf32) <- (-1x256x7x7xf32)
        t949 = paddle._C_ops.relu(t943)
        del t943

        # pd_op.multiply: (-1x256x7x7xf32) <- (1xf32, -1x256x7x7xf32)
        t950 = paddle._C_ops.multiply(input85, t949)
        del input85

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1xf32)
        t951 = paddle._C_ops.add(t950, input86)
        del input86

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        t952 = paddle._C_ops.conv2d(
            t951, t320, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t320

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t953, t954, t955, t956, t957, t958 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t952,
                t321,
                t322,
                t323,
                t324,
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
        del t324, t323, t322, t321

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x5x5xf32)
        t959 = paddle._C_ops.depthwise_conv2d(
            t953, t325, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t325

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t960, t961, t962, t963, t964, t965 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t959,
                t326,
                t327,
                t328,
                t329,
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
        del t329, t328, t327, t326

        # pd_op.relu: (-1x256x7x7xf32) <- (-1x256x7x7xf32)
        t966 = paddle._C_ops.relu(t960)
        del t960

        # pd_op.multiply: (-1x256x7x7xf32) <- (1xf32, -1x256x7x7xf32)
        t967 = paddle._C_ops.multiply(input87, t966)
        del input87

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1xf32)
        t968 = paddle._C_ops.add(t967, input88)
        del input88

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        t969 = paddle._C_ops.conv2d(
            t968, t330, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t330

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t970, t971, t972, t973, t974, t975 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t969,
                t331,
                t332,
                t333,
                t334,
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
        del t334, t333, t332, t331

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x5x5xf32)
        t976 = paddle._C_ops.depthwise_conv2d(
            t970, t335, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t335

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t977, t978, t979, t980, t981, t982 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t976,
                t336,
                t337,
                t338,
                t339,
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
        del t339, t338, t337, t336

        # pd_op.relu: (-1x256x7x7xf32) <- (-1x256x7x7xf32)
        t983 = paddle._C_ops.relu(t977)
        del t977

        # pd_op.multiply: (-1x256x7x7xf32) <- (1xf32, -1x256x7x7xf32)
        t984 = paddle._C_ops.multiply(input89, t983)
        del input89

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1xf32)
        t985 = paddle._C_ops.add(t984, input90)
        del input90

        # builtin.combine: ([-1x1024x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32]) <- (-1x1024x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32)
        t986 = [t895, t917, t934, t951, t968, t985]

        # pd_op.concat: (-1x2304x7x7xf32) <- ([-1x1024x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32], 1xi32)
        t987 = paddle._C_ops.concat(t986, t384)
        del t986

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x2304x7x7xf32, 1024x2304x1x1xf32)
        t988 = paddle._C_ops.conv2d(
            t987, t340, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t340

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t989, t990, t991, t992, t993, t994 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t988,
                t341,
                t342,
                t343,
                t344,
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
        del t342, t341, t344, t343

        # pd_op.relu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t995 = paddle._C_ops.relu(t989)
        del t989

        # pd_op.multiply: (-1x1024x7x7xf32) <- (1xf32, -1x1024x7x7xf32)
        t996 = paddle._C_ops.multiply(input91, t995)
        del input91

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1xf32)
        t997 = paddle._C_ops.add(t996, input92)
        del input92

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        t998 = paddle._C_ops.conv2d(
            t997, t345, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t345

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t999, t1000, t1001, t1002, t1003, t1004 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t998,
                t346,
                t347,
                t348,
                t349,
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
        del t349, t348, t347, t346

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t1005 = paddle._C_ops.relu(t999)
        del t999

        # pd_op.multiply: (-1x2048x7x7xf32) <- (1xf32, -1x2048x7x7xf32)
        t1006 = paddle._C_ops.multiply(input93, t1005)
        del input93

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, 1xf32)
        t1007 = paddle._C_ops.add(t1006, input94)
        del input94

        # pd_op.full_int_array: (2xi64) <- ()
        t1008 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        t1009 = paddle._C_ops.pool2d(
            t1007,
            t1008,
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

        # pd_op.conv2d: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32, 2048x2048x1x1xf32)
        t1010 = paddle._C_ops.conv2d(
            t1009, t350, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t350

        # pd_op.relu: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32)
        t1011 = paddle._C_ops.relu(t1010)
        del t1010

        # pd_op.multiply: (-1x2048x1x1xf32) <- (1xf32, -1x2048x1x1xf32)
        t1012 = paddle._C_ops.multiply(input95, t1011)
        del input95

        # pd_op.add: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32, 1xf32)
        t1013 = paddle._C_ops.add(t1012, input96)
        del input96

        # pd_op.flatten: (-1x2048xf32) <- (-1x2048x1x1xf32)
        t1014 = paddle._C_ops.flatten(t1013, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x2048xf32, 2048x102xf32)
        t1015 = paddle._C_ops.matmul(t1014, t351, False, False)
        del t351

        return (
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
            t364,
            t365,
            t366,
            t367,
            t368,
            t369,
            t370,
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
            t382,
            t383,
            t384,
            t385,
            t386,
            t387,
            t388,
            t389,
            t390,
            t392,
            t393,
            t395,
            t396,
            t397,
            t398,
            t399,
            t400,
            t401,
            t402,
            t403,
            t405,
            t406,
            t407,
            t408,
            t409,
            t410,
            t411,
            t412,
            t413,
            t415,
            t416,
            t417,
            t418,
            t419,
            t420,
            t421,
            t422,
            t423,
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
            t438,
            t439,
            t440,
            t441,
            t442,
            t443,
            t445,
            t446,
            t447,
            t448,
            t449,
            t450,
            t451,
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
            t464,
            t465,
            t467,
            t468,
            t469,
            t470,
            t471,
            t472,
            t473,
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
            t485,
            t486,
            t487,
            t488,
            t489,
            t490,
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
            t502,
            t504,
            t505,
            t506,
            t507,
            t508,
            t509,
            t510,
            t511,
            t512,
            t514,
            t515,
            t516,
            t517,
            t518,
            t519,
            t520,
            t521,
            t522,
            t524,
            t525,
            t526,
            t527,
            t528,
            t529,
            t530,
            t531,
            t532,
            t534,
            t535,
            t536,
            t537,
            t538,
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
            t575,
            t576,
            t577,
            t578,
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
            t593,
            t594,
            t595,
            t597,
            t598,
            t599,
            t600,
            t601,
            t602,
            t603,
            t604,
            t605,
            t606,
            t607,
            t608,
            t609,
            t610,
            t611,
            t612,
            t614,
            t615,
            t616,
            t617,
            t618,
            t619,
            t620,
            t621,
            t622,
            t623,
            t624,
            t625,
            t626,
            t627,
            t628,
            t629,
            t631,
            t632,
            t633,
            t634,
            t635,
            t636,
            t637,
            t638,
            t639,
            t640,
            t641,
            t642,
            t643,
            t644,
            t645,
            t646,
            t648,
            t649,
            t650,
            t651,
            t652,
            t653,
            t654,
            t655,
            t657,
            t658,
            t660,
            t661,
            t662,
            t663,
            t664,
            t665,
            t666,
            t667,
            t668,
            t670,
            t671,
            t672,
            t673,
            t674,
            t675,
            t676,
            t677,
            t678,
            t679,
            t680,
            t681,
            t682,
            t683,
            t684,
            t685,
            t687,
            t688,
            t689,
            t690,
            t691,
            t692,
            t693,
            t694,
            t695,
            t696,
            t697,
            t698,
            t699,
            t700,
            t701,
            t702,
            t704,
            t705,
            t706,
            t707,
            t708,
            t709,
            t710,
            t711,
            t712,
            t713,
            t714,
            t715,
            t716,
            t717,
            t718,
            t719,
            t721,
            t722,
            t723,
            t724,
            t725,
            t726,
            t727,
            t728,
            t729,
            t730,
            t731,
            t732,
            t733,
            t734,
            t735,
            t736,
            t738,
            t739,
            t740,
            t741,
            t742,
            t743,
            t744,
            t745,
            t746,
            t747,
            t748,
            t749,
            t750,
            t751,
            t752,
            t753,
            t755,
            t756,
            t757,
            t758,
            t759,
            t760,
            t761,
            t762,
            t764,
            t765,
            t767,
            t768,
            t769,
            t770,
            t771,
            t772,
            t773,
            t774,
            t775,
            t777,
            t778,
            t779,
            t780,
            t781,
            t782,
            t783,
            t784,
            t785,
            t786,
            t787,
            t788,
            t789,
            t790,
            t791,
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
            t803,
            t804,
            t805,
            t806,
            t807,
            t808,
            t809,
            t810,
            t812,
            t813,
            t814,
            t815,
            t816,
            t817,
            t818,
            t819,
            t820,
            t821,
            t822,
            t823,
            t824,
            t825,
            t826,
            t827,
            t829,
            t830,
            t831,
            t832,
            t833,
            t834,
            t835,
            t836,
            t837,
            t838,
            t839,
            t840,
            t841,
            t842,
            t843,
            t844,
            t846,
            t847,
            t848,
            t849,
            t850,
            t851,
            t852,
            t853,
            t854,
            t855,
            t856,
            t857,
            t858,
            t859,
            t860,
            t861,
            t863,
            t864,
            t865,
            t866,
            t867,
            t868,
            t869,
            t870,
            t872,
            t873,
            t875,
            t876,
            t877,
            t878,
            t879,
            t880,
            t881,
            t882,
            t883,
            t885,
            t886,
            t887,
            t888,
            t889,
            t890,
            t891,
            t892,
            t893,
            t894,
            t895,
            t896,
            t897,
            t898,
            t899,
            t900,
            t901,
            t902,
            t903,
            t904,
            t905,
            t906,
            t907,
            t908,
            t910,
            t911,
            t912,
            t913,
            t914,
            t915,
            t916,
            t917,
            t918,
            t919,
            t920,
            t921,
            t922,
            t923,
            t924,
            t925,
            t927,
            t928,
            t929,
            t930,
            t931,
            t932,
            t933,
            t934,
            t935,
            t936,
            t937,
            t938,
            t939,
            t940,
            t941,
            t942,
            t944,
            t945,
            t946,
            t947,
            t948,
            t949,
            t950,
            t951,
            t952,
            t953,
            t954,
            t955,
            t956,
            t957,
            t958,
            t959,
            t961,
            t962,
            t963,
            t964,
            t965,
            t966,
            t967,
            t968,
            t969,
            t970,
            t971,
            t972,
            t973,
            t974,
            t975,
            t976,
            t978,
            t979,
            t980,
            t981,
            t982,
            t983,
            t984,
            t985,
            t987,
            t988,
            t990,
            t991,
            t992,
            t993,
            t994,
            t995,
            t996,
            t997,
            t998,
            t1000,
            t1001,
            t1002,
            t1003,
            t1004,
            t1005,
            t1006,
            t1007,
            t1008,
            t1009,
            t1011,
            t1012,
            t1013,
            t1014,
            t1015,
        )
