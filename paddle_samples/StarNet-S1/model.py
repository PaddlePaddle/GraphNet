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
        t218: paddle.Tensor,
        t219: paddle.Tensor,
        t220: paddle.Tensor,
        t221: paddle.Tensor,
        t222: paddle.Tensor,
        t223: paddle.Tensor,
        t224: paddle.Tensor,
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
        t275: paddle.Tensor,
        t276: paddle.Tensor,
        t277: paddle.Tensor,
        t278: paddle.Tensor,
        t279: paddle.Tensor,
        t280: paddle.Tensor,
        t281: paddle.Tensor,
        t282: paddle.Tensor,
        t283: paddle.Tensor,
        t284: paddle.Tensor,
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
        t300: paddle.Tensor,
        t301: paddle.Tensor,
        t302: paddle.Tensor,
        t303: paddle.Tensor,
        t304: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        t305 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t306 = [1, -1, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        t307 = paddle._C_ops.reshape(t1, t306)
        del t1

        # pd_op.add: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 1x32x1x1xf32)
        t308 = paddle._C_ops.add(t305, t307)
        del t305, t307

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t309, t310, t311, t312, t313, t314 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t308,
                t2,
                t3,
                t4,
                t5,
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
        del t308, t5, t4, t3, t2

        # pd_op.relu6: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t315 = paddle._C_ops.relu6(t309)
        del t309

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x32x112x112xf32, 24x32x3x3xf32)
        t316 = paddle._C_ops.conv2d(
            t315, t6, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t6, t315

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t317 = paddle._C_ops.reshape(t7, t306)
        del t7

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        t318 = paddle._C_ops.add(t316, t317)
        del t316, t317

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t319, t320, t321, t322, t323, t324 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t318,
                t8,
                t9,
                t10,
                t11,
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
        del t318, t11, t10, t9, t8

        # pd_op.depthwise_conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x1x7x7xf32)
        t325 = paddle._C_ops.depthwise_conv2d(
            t319, t12, [1, 1], [3, 3], "EXPLICIT", 24, [1, 1], "NCHW"
        )
        del t12

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t326 = paddle._C_ops.reshape(t13, t306)
        del t13

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        t327 = paddle._C_ops.add(t325, t326)
        del t325, t326

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t328, t329, t330, t331, t332, t333 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t327,
                t14,
                t15,
                t16,
                t17,
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
        del t327, t17, t16, t15, t14

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x24x56x56xf32, 96x24x1x1xf32)
        t334 = paddle._C_ops.conv2d(
            t328, t18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t18

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t335 = paddle._C_ops.reshape(t19, t306)
        del t19

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        t336 = paddle._C_ops.add(t334, t335)
        del t334, t335

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x24x56x56xf32, 96x24x1x1xf32)
        t337 = paddle._C_ops.conv2d(
            t328, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t328, t20

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t338 = paddle._C_ops.reshape(t21, t306)
        del t21

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        t339 = paddle._C_ops.add(t337, t338)
        del t337, t338

        # pd_op.relu6: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t340 = paddle._C_ops.relu6(t336)
        del t336

        # pd_op.multiply: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        t341 = paddle._C_ops.multiply(t340, t339)
        del t339, t340

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x96x56x56xf32, 24x96x1x1xf32)
        t342 = paddle._C_ops.conv2d(
            t341, t22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t341, t22

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t343 = paddle._C_ops.reshape(t23, t306)
        del t23

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        t344 = paddle._C_ops.add(t342, t343)
        del t342, t343

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t345, t346, t347, t348, t349, t350 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t344,
                t24,
                t25,
                t26,
                t27,
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
        del t344, t27, t26, t25, t24

        # pd_op.depthwise_conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x1x7x7xf32)
        t351 = paddle._C_ops.depthwise_conv2d(
            t345, t28, [1, 1], [3, 3], "EXPLICIT", 24, [1, 1], "NCHW"
        )
        del t345, t28

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t352 = paddle._C_ops.reshape(t29, t306)
        del t29

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        t353 = paddle._C_ops.add(t351, t352)
        del t351, t352

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, -1x24x56x56xf32)
        t354 = paddle._C_ops.add(t319, t353)
        del t353, t319

        # pd_op.depthwise_conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x1x7x7xf32)
        t355 = paddle._C_ops.depthwise_conv2d(
            t354, t30, [1, 1], [3, 3], "EXPLICIT", 24, [1, 1], "NCHW"
        )
        del t30

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t356 = paddle._C_ops.reshape(t31, t306)
        del t31

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        t357 = paddle._C_ops.add(t355, t356)
        del t355, t356

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t358, t359, t360, t361, t362, t363 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t357,
                t32,
                t33,
                t34,
                t35,
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
        del t357, t35, t34, t33, t32

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x24x56x56xf32, 96x24x1x1xf32)
        t364 = paddle._C_ops.conv2d(
            t358, t36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t36

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t365 = paddle._C_ops.reshape(t37, t306)
        del t37

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        t366 = paddle._C_ops.add(t364, t365)
        del t364, t365

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x24x56x56xf32, 96x24x1x1xf32)
        t367 = paddle._C_ops.conv2d(
            t358, t38, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t358, t38

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t368 = paddle._C_ops.reshape(t39, t306)
        del t39

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        t369 = paddle._C_ops.add(t367, t368)
        del t367, t368

        # pd_op.relu6: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t370 = paddle._C_ops.relu6(t366)
        del t366

        # pd_op.multiply: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        t371 = paddle._C_ops.multiply(t370, t369)
        del t369, t370

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x96x56x56xf32, 24x96x1x1xf32)
        t372 = paddle._C_ops.conv2d(
            t371, t40, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t371, t40

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t373 = paddle._C_ops.reshape(t41, t306)
        del t41

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        t374 = paddle._C_ops.add(t372, t373)
        del t372, t373

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t375, t376, t377, t378, t379, t380 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t374,
                t42,
                t43,
                t44,
                t45,
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
        del t374, t45, t44, t43, t42

        # pd_op.depthwise_conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x1x7x7xf32)
        t381 = paddle._C_ops.depthwise_conv2d(
            t375, t46, [1, 1], [3, 3], "EXPLICIT", 24, [1, 1], "NCHW"
        )
        del t375, t46

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t382 = paddle._C_ops.reshape(t47, t306)
        del t47

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        t383 = paddle._C_ops.add(t381, t382)
        del t381, t382

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, -1x24x56x56xf32)
        t384 = paddle._C_ops.add(t354, t383)
        del t383, t354

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x24x56x56xf32, 48x24x3x3xf32)
        t385 = paddle._C_ops.conv2d(
            t384, t48, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t384, t48

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t386 = paddle._C_ops.reshape(t49, t306)
        del t49

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        t387 = paddle._C_ops.add(t385, t386)
        del t385, t386

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t388, t389, t390, t391, t392, t393 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t387,
                t50,
                t51,
                t52,
                t53,
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
        del t387, t53, t52, t51, t50

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x1x7x7xf32)
        t394 = paddle._C_ops.depthwise_conv2d(
            t388, t54, [1, 1], [3, 3], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t54

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t395 = paddle._C_ops.reshape(t55, t306)
        del t55

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        t396 = paddle._C_ops.add(t394, t395)
        del t394, t395

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t397, t398, t399, t400, t401, t402 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t396,
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
        del t396, t59, t58, t57, t56

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x48x28x28xf32, 192x48x1x1xf32)
        t403 = paddle._C_ops.conv2d(
            t397, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t404 = paddle._C_ops.reshape(t61, t306)
        del t61

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        t405 = paddle._C_ops.add(t403, t404)
        del t403, t404

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x48x28x28xf32, 192x48x1x1xf32)
        t406 = paddle._C_ops.conv2d(
            t397, t62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t397, t62

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t407 = paddle._C_ops.reshape(t63, t306)
        del t63

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        t408 = paddle._C_ops.add(t406, t407)
        del t406, t407

        # pd_op.relu6: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t409 = paddle._C_ops.relu6(t405)
        del t405

        # pd_op.multiply: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        t410 = paddle._C_ops.multiply(t409, t408)
        del t408, t409

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x192x28x28xf32, 48x192x1x1xf32)
        t411 = paddle._C_ops.conv2d(
            t410, t64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t410, t64

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t412 = paddle._C_ops.reshape(t65, t306)
        del t65

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        t413 = paddle._C_ops.add(t411, t412)
        del t411, t412

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t414, t415, t416, t417, t418, t419 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t413,
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
        del t413, t69, t68, t67, t66

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x1x7x7xf32)
        t420 = paddle._C_ops.depthwise_conv2d(
            t414, t70, [1, 1], [3, 3], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t414, t70

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t421 = paddle._C_ops.reshape(t71, t306)
        del t71

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        t422 = paddle._C_ops.add(t420, t421)
        del t420, t421

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, -1x48x28x28xf32)
        t423 = paddle._C_ops.add(t388, t422)
        del t422, t388

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x1x7x7xf32)
        t424 = paddle._C_ops.depthwise_conv2d(
            t423, t72, [1, 1], [3, 3], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t72

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t425 = paddle._C_ops.reshape(t73, t306)
        del t73

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        t426 = paddle._C_ops.add(t424, t425)
        del t424, t425

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t427, t428, t429, t430, t431, t432 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t426,
                t74,
                t75,
                t76,
                t77,
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
        del t426, t77, t76, t75, t74

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x48x28x28xf32, 192x48x1x1xf32)
        t433 = paddle._C_ops.conv2d(
            t427, t78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t78

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t434 = paddle._C_ops.reshape(t79, t306)
        del t79

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        t435 = paddle._C_ops.add(t433, t434)
        del t433, t434

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x48x28x28xf32, 192x48x1x1xf32)
        t436 = paddle._C_ops.conv2d(
            t427, t80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t427, t80

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t437 = paddle._C_ops.reshape(t81, t306)
        del t81

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        t438 = paddle._C_ops.add(t436, t437)
        del t436, t437

        # pd_op.relu6: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t439 = paddle._C_ops.relu6(t435)
        del t435

        # pd_op.multiply: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        t440 = paddle._C_ops.multiply(t439, t438)
        del t438, t439

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x192x28x28xf32, 48x192x1x1xf32)
        t441 = paddle._C_ops.conv2d(
            t440, t82, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t440, t82

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t442 = paddle._C_ops.reshape(t83, t306)
        del t83

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        t443 = paddle._C_ops.add(t441, t442)
        del t441, t442

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t444, t445, t446, t447, t448, t449 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t443,
                t84,
                t85,
                t86,
                t87,
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
        del t443, t87, t86, t85, t84

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x1x7x7xf32)
        t450 = paddle._C_ops.depthwise_conv2d(
            t444, t88, [1, 1], [3, 3], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t444, t88

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t451 = paddle._C_ops.reshape(t89, t306)
        del t89

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        t452 = paddle._C_ops.add(t450, t451)
        del t450, t451

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, -1x48x28x28xf32)
        t453 = paddle._C_ops.add(t423, t452)
        del t423, t452

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x48x28x28xf32, 96x48x3x3xf32)
        t454 = paddle._C_ops.conv2d(
            t453, t90, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t453, t90

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t455 = paddle._C_ops.reshape(t91, t306)
        del t91

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t456 = paddle._C_ops.add(t454, t455)
        del t454, t455

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t457, t458, t459, t460, t461, t462 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t456,
                t92,
                t93,
                t94,
                t95,
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
        del t456, t95, t94, t93, t92

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t463 = paddle._C_ops.depthwise_conv2d(
            t457, t96, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t96

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t464 = paddle._C_ops.reshape(t97, t306)
        del t97

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t465 = paddle._C_ops.add(t463, t464)
        del t463, t464

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t466, t467, t468, t469, t470, t471 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t465,
                t98,
                t99,
                t100,
                t101,
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
        del t465, t101, t100, t99, t98

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t472 = paddle._C_ops.conv2d(
            t466, t102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t102

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t473 = paddle._C_ops.reshape(t103, t306)
        del t103

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t474 = paddle._C_ops.add(t472, t473)
        del t472, t473

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t475 = paddle._C_ops.conv2d(
            t466, t104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t466, t104

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t476 = paddle._C_ops.reshape(t105, t306)
        del t105

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t477 = paddle._C_ops.add(t475, t476)
        del t475, t476

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t478 = paddle._C_ops.relu6(t474)
        del t474

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t479 = paddle._C_ops.multiply(t478, t477)
        del t477, t478

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        t480 = paddle._C_ops.conv2d(
            t479, t106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t479, t106

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t481 = paddle._C_ops.reshape(t107, t306)
        del t107

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t482 = paddle._C_ops.add(t480, t481)
        del t480, t481

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t483, t484, t485, t486, t487, t488 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t482,
                t108,
                t109,
                t110,
                t111,
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
        del t482, t111, t110, t109, t108

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t489 = paddle._C_ops.depthwise_conv2d(
            t483, t112, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t483, t112

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t490 = paddle._C_ops.reshape(t113, t306)
        del t113

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t491 = paddle._C_ops.add(t489, t490)
        del t489, t490

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        t492 = paddle._C_ops.add(t457, t491)
        del t491, t457

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t493 = paddle._C_ops.depthwise_conv2d(
            t492, t114, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t114

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t494 = paddle._C_ops.reshape(t115, t306)
        del t115

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t495 = paddle._C_ops.add(t493, t494)
        del t493, t494

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t496, t497, t498, t499, t500, t501 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t495,
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
        del t495, t119, t118, t117, t116

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t502 = paddle._C_ops.conv2d(
            t496, t120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t503 = paddle._C_ops.reshape(t121, t306)
        del t121

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t504 = paddle._C_ops.add(t502, t503)
        del t502, t503

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t505 = paddle._C_ops.conv2d(
            t496, t122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t496, t122

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t506 = paddle._C_ops.reshape(t123, t306)
        del t123

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t507 = paddle._C_ops.add(t505, t506)
        del t505, t506

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t508 = paddle._C_ops.relu6(t504)
        del t504

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t509 = paddle._C_ops.multiply(t508, t507)
        del t507, t508

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        t510 = paddle._C_ops.conv2d(
            t509, t124, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t509, t124

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t511 = paddle._C_ops.reshape(t125, t306)
        del t125

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t512 = paddle._C_ops.add(t510, t511)
        del t510, t511

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t513, t514, t515, t516, t517, t518 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t512,
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
        del t512, t129, t128, t127, t126

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t519 = paddle._C_ops.depthwise_conv2d(
            t513, t130, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t513, t130

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t520 = paddle._C_ops.reshape(t131, t306)
        del t131

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t521 = paddle._C_ops.add(t519, t520)
        del t519, t520

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        t522 = paddle._C_ops.add(t492, t521)
        del t492, t521

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t523 = paddle._C_ops.depthwise_conv2d(
            t522, t132, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t132

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t524 = paddle._C_ops.reshape(t133, t306)
        del t133

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t525 = paddle._C_ops.add(t523, t524)
        del t523, t524

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t526, t527, t528, t529, t530, t531 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t525,
                t134,
                t135,
                t136,
                t137,
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
        del t525, t137, t136, t135, t134

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t532 = paddle._C_ops.conv2d(
            t526, t138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t138

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t533 = paddle._C_ops.reshape(t139, t306)
        del t139

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t534 = paddle._C_ops.add(t532, t533)
        del t532, t533

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t535 = paddle._C_ops.conv2d(
            t526, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t526, t140

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t536 = paddle._C_ops.reshape(t141, t306)
        del t141

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t537 = paddle._C_ops.add(t535, t536)
        del t535, t536

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t538 = paddle._C_ops.relu6(t534)
        del t534

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t539 = paddle._C_ops.multiply(t538, t537)
        del t537, t538

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        t540 = paddle._C_ops.conv2d(
            t539, t142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t539, t142

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t541 = paddle._C_ops.reshape(t143, t306)
        del t143

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t542 = paddle._C_ops.add(t540, t541)
        del t540, t541

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t543, t544, t545, t546, t547, t548 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t542,
                t144,
                t145,
                t146,
                t147,
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
        del t542, t147, t146, t145, t144

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t549 = paddle._C_ops.depthwise_conv2d(
            t543, t148, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t543, t148

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t550 = paddle._C_ops.reshape(t149, t306)
        del t149

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t551 = paddle._C_ops.add(t549, t550)
        del t549, t550

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        t552 = paddle._C_ops.add(t522, t551)
        del t522, t551

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t553 = paddle._C_ops.depthwise_conv2d(
            t552, t150, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t150

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t554 = paddle._C_ops.reshape(t151, t306)
        del t151

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t555 = paddle._C_ops.add(t553, t554)
        del t553, t554

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t556, t557, t558, t559, t560, t561 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t555,
                t152,
                t153,
                t154,
                t155,
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
        del t555, t155, t154, t153, t152

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t562 = paddle._C_ops.conv2d(
            t556, t156, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t156

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t563 = paddle._C_ops.reshape(t157, t306)
        del t157

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t564 = paddle._C_ops.add(t562, t563)
        del t562, t563

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t565 = paddle._C_ops.conv2d(
            t556, t158, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t556, t158

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t566 = paddle._C_ops.reshape(t159, t306)
        del t159

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t567 = paddle._C_ops.add(t565, t566)
        del t565, t566

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t568 = paddle._C_ops.relu6(t564)
        del t564

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t569 = paddle._C_ops.multiply(t568, t567)
        del t567, t568

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        t570 = paddle._C_ops.conv2d(
            t569, t160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t569, t160

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t571 = paddle._C_ops.reshape(t161, t306)
        del t161

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t572 = paddle._C_ops.add(t570, t571)
        del t570, t571

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t573, t574, t575, t576, t577, t578 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t572,
                t162,
                t163,
                t164,
                t165,
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
        del t572, t165, t164, t163, t162

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t579 = paddle._C_ops.depthwise_conv2d(
            t573, t166, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t573, t166

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t580 = paddle._C_ops.reshape(t167, t306)
        del t167

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t581 = paddle._C_ops.add(t579, t580)
        del t579, t580

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        t582 = paddle._C_ops.add(t552, t581)
        del t552, t581

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t583 = paddle._C_ops.depthwise_conv2d(
            t582, t168, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t168

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t584 = paddle._C_ops.reshape(t169, t306)
        del t169

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t585 = paddle._C_ops.add(t583, t584)
        del t583, t584

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t586, t587, t588, t589, t590, t591 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t585,
                t170,
                t171,
                t172,
                t173,
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
        del t585, t173, t172, t171, t170

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t592 = paddle._C_ops.conv2d(
            t586, t174, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t174

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t593 = paddle._C_ops.reshape(t175, t306)
        del t175

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t594 = paddle._C_ops.add(t592, t593)
        del t592, t593

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t595 = paddle._C_ops.conv2d(
            t586, t176, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t586, t176

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t596 = paddle._C_ops.reshape(t177, t306)
        del t177

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t597 = paddle._C_ops.add(t595, t596)
        del t595, t596

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t598 = paddle._C_ops.relu6(t594)
        del t594

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t599 = paddle._C_ops.multiply(t598, t597)
        del t597, t598

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        t600 = paddle._C_ops.conv2d(
            t599, t178, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t599, t178

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t601 = paddle._C_ops.reshape(t179, t306)
        del t179

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t602 = paddle._C_ops.add(t600, t601)
        del t600, t601

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t603, t604, t605, t606, t607, t608 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t602,
                t180,
                t181,
                t182,
                t183,
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
        del t602, t183, t182, t181, t180

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t609 = paddle._C_ops.depthwise_conv2d(
            t603, t184, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t603, t184

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t610 = paddle._C_ops.reshape(t185, t306)
        del t185

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t611 = paddle._C_ops.add(t609, t610)
        del t609, t610

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        t612 = paddle._C_ops.add(t582, t611)
        del t582, t611

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t613 = paddle._C_ops.depthwise_conv2d(
            t612, t186, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t186

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t614 = paddle._C_ops.reshape(t187, t306)
        del t187

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t615 = paddle._C_ops.add(t613, t614)
        del t613, t614

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t616, t617, t618, t619, t620, t621 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t615,
                t188,
                t189,
                t190,
                t191,
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
        del t615, t191, t190, t189, t188

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t622 = paddle._C_ops.conv2d(
            t616, t192, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t192

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t623 = paddle._C_ops.reshape(t193, t306)
        del t193

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t624 = paddle._C_ops.add(t622, t623)
        del t622, t623

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t625 = paddle._C_ops.conv2d(
            t616, t194, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t616, t194

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t626 = paddle._C_ops.reshape(t195, t306)
        del t195

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t627 = paddle._C_ops.add(t625, t626)
        del t625, t626

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t628 = paddle._C_ops.relu6(t624)
        del t624

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t629 = paddle._C_ops.multiply(t628, t627)
        del t627, t628

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        t630 = paddle._C_ops.conv2d(
            t629, t196, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t629, t196

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t631 = paddle._C_ops.reshape(t197, t306)
        del t197

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t632 = paddle._C_ops.add(t630, t631)
        del t630, t631

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t633, t634, t635, t636, t637, t638 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t632,
                t198,
                t199,
                t200,
                t201,
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
        del t632, t201, t200, t199, t198

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t639 = paddle._C_ops.depthwise_conv2d(
            t633, t202, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t633, t202

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t640 = paddle._C_ops.reshape(t203, t306)
        del t203

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t641 = paddle._C_ops.add(t639, t640)
        del t639, t640

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        t642 = paddle._C_ops.add(t612, t641)
        del t612, t641

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t643 = paddle._C_ops.depthwise_conv2d(
            t642, t204, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t204

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t644 = paddle._C_ops.reshape(t205, t306)
        del t205

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t645 = paddle._C_ops.add(t643, t644)
        del t643, t644

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t646, t647, t648, t649, t650, t651 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t645,
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
        del t645, t209, t208, t207, t206

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t652 = paddle._C_ops.conv2d(
            t646, t210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t653 = paddle._C_ops.reshape(t211, t306)
        del t211

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t654 = paddle._C_ops.add(t652, t653)
        del t652, t653

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t655 = paddle._C_ops.conv2d(
            t646, t212, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t646, t212

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t656 = paddle._C_ops.reshape(t213, t306)
        del t213

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t657 = paddle._C_ops.add(t655, t656)
        del t655, t656

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t658 = paddle._C_ops.relu6(t654)
        del t654

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t659 = paddle._C_ops.multiply(t658, t657)
        del t657, t658

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        t660 = paddle._C_ops.conv2d(
            t659, t214, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t659, t214

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t661 = paddle._C_ops.reshape(t215, t306)
        del t215

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t662 = paddle._C_ops.add(t660, t661)
        del t660, t661

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t663, t664, t665, t666, t667, t668 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t662,
                t216,
                t217,
                t218,
                t219,
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
        del t662, t219, t218, t217, t216

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t669 = paddle._C_ops.depthwise_conv2d(
            t663, t220, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t663, t220

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t670 = paddle._C_ops.reshape(t221, t306)
        del t221

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t671 = paddle._C_ops.add(t669, t670)
        del t669, t670

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        t672 = paddle._C_ops.add(t642, t671)
        del t642, t671

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t673 = paddle._C_ops.depthwise_conv2d(
            t672, t222, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t222

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t674 = paddle._C_ops.reshape(t223, t306)
        del t223

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t675 = paddle._C_ops.add(t673, t674)
        del t673, t674

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t676, t677, t678, t679, t680, t681 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t675,
                t224,
                t225,
                t226,
                t227,
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
        del t675, t227, t226, t225, t224

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t682 = paddle._C_ops.conv2d(
            t676, t228, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t228

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t683 = paddle._C_ops.reshape(t229, t306)
        del t229

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t684 = paddle._C_ops.add(t682, t683)
        del t682, t683

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        t685 = paddle._C_ops.conv2d(
            t676, t230, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t676, t230

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t686 = paddle._C_ops.reshape(t231, t306)
        del t231

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        t687 = paddle._C_ops.add(t685, t686)
        del t685, t686

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t688 = paddle._C_ops.relu6(t684)
        del t684

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        t689 = paddle._C_ops.multiply(t688, t687)
        del t687, t688

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        t690 = paddle._C_ops.conv2d(
            t689, t232, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t689, t232

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t691 = paddle._C_ops.reshape(t233, t306)
        del t233

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t692 = paddle._C_ops.add(t690, t691)
        del t690, t691

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t693, t694, t695, t696, t697, t698 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t692,
                t234,
                t235,
                t236,
                t237,
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
        del t692, t237, t236, t235, t234

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        t699 = paddle._C_ops.depthwise_conv2d(
            t693, t238, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t693, t238

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t700 = paddle._C_ops.reshape(t239, t306)
        del t239

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        t701 = paddle._C_ops.add(t699, t700)
        del t699, t700

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        t702 = paddle._C_ops.add(t672, t701)
        del t672, t701

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x96x14x14xf32, 192x96x3x3xf32)
        t703 = paddle._C_ops.conv2d(
            t702, t240, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t702, t240

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t704 = paddle._C_ops.reshape(t241, t306)
        del t241

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t705 = paddle._C_ops.add(t703, t704)
        del t703, t704

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t706, t707, t708, t709, t710, t711 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t705,
                t242,
                t243,
                t244,
                t245,
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
        del t705, t245, t244, t243, t242

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        t712 = paddle._C_ops.depthwise_conv2d(
            t706, t246, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t246

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t713 = paddle._C_ops.reshape(t247, t306)
        del t247

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t714 = paddle._C_ops.add(t712, t713)
        del t712, t713

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t715, t716, t717, t718, t719, t720 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t714,
                t248,
                t249,
                t250,
                t251,
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
        del t714, t251, t250, t249, t248

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        t721 = paddle._C_ops.conv2d(
            t715, t252, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t252

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t722 = paddle._C_ops.reshape(t253, t306)
        del t253

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t723 = paddle._C_ops.add(t721, t722)
        del t721, t722

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        t724 = paddle._C_ops.conv2d(
            t715, t254, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t715, t254

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t725 = paddle._C_ops.reshape(t255, t306)
        del t255

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t726 = paddle._C_ops.add(t724, t725)
        del t724, t725

        # pd_op.relu6: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t727 = paddle._C_ops.relu6(t723)
        del t723

        # pd_op.multiply: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        t728 = paddle._C_ops.multiply(t727, t726)
        del t726, t727

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x768x7x7xf32, 192x768x1x1xf32)
        t729 = paddle._C_ops.conv2d(
            t728, t256, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t728, t256

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t730 = paddle._C_ops.reshape(t257, t306)
        del t257

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t731 = paddle._C_ops.add(t729, t730)
        del t729, t730

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t732, t733, t734, t735, t736, t737 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t731,
                t258,
                t259,
                t260,
                t261,
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
        del t731, t261, t260, t259, t258

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        t738 = paddle._C_ops.depthwise_conv2d(
            t732, t262, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t732, t262

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t739 = paddle._C_ops.reshape(t263, t306)
        del t263

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t740 = paddle._C_ops.add(t738, t739)
        del t738, t739

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, -1x192x7x7xf32)
        t741 = paddle._C_ops.add(t706, t740)
        del t740, t706

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        t742 = paddle._C_ops.depthwise_conv2d(
            t741, t264, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t264

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t743 = paddle._C_ops.reshape(t265, t306)
        del t265

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t744 = paddle._C_ops.add(t742, t743)
        del t742, t743

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t745, t746, t747, t748, t749, t750 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t744,
                t266,
                t267,
                t268,
                t269,
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
        del t744, t269, t268, t267, t266

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        t751 = paddle._C_ops.conv2d(
            t745, t270, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t270

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t752 = paddle._C_ops.reshape(t271, t306)
        del t271

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t753 = paddle._C_ops.add(t751, t752)
        del t751, t752

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        t754 = paddle._C_ops.conv2d(
            t745, t272, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t745, t272

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t755 = paddle._C_ops.reshape(t273, t306)
        del t273

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t756 = paddle._C_ops.add(t754, t755)
        del t754, t755

        # pd_op.relu6: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t757 = paddle._C_ops.relu6(t753)
        del t753

        # pd_op.multiply: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        t758 = paddle._C_ops.multiply(t757, t756)
        del t756, t757

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x768x7x7xf32, 192x768x1x1xf32)
        t759 = paddle._C_ops.conv2d(
            t758, t274, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t758, t274

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t760 = paddle._C_ops.reshape(t275, t306)
        del t275

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t761 = paddle._C_ops.add(t759, t760)
        del t759, t760

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t762, t763, t764, t765, t766, t767 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t761,
                t276,
                t277,
                t278,
                t279,
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
        del t761, t279, t278, t277, t276

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        t768 = paddle._C_ops.depthwise_conv2d(
            t762, t280, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t762, t280

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t769 = paddle._C_ops.reshape(t281, t306)
        del t281

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t770 = paddle._C_ops.add(t768, t769)
        del t768, t769

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, -1x192x7x7xf32)
        t771 = paddle._C_ops.add(t741, t770)
        del t741, t770

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        t772 = paddle._C_ops.depthwise_conv2d(
            t771, t282, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t282

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t773 = paddle._C_ops.reshape(t283, t306)
        del t283

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t774 = paddle._C_ops.add(t772, t773)
        del t772, t773

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t775, t776, t777, t778, t779, t780 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t774,
                t284,
                t285,
                t286,
                t287,
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
        del t774, t287, t286, t285, t284

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        t781 = paddle._C_ops.conv2d(
            t775, t288, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t288

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t782 = paddle._C_ops.reshape(t289, t306)
        del t289

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t783 = paddle._C_ops.add(t781, t782)
        del t781, t782

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        t784 = paddle._C_ops.conv2d(
            t775, t290, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t775, t290

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t785 = paddle._C_ops.reshape(t291, t306)
        del t291

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        t786 = paddle._C_ops.add(t784, t785)
        del t784, t785

        # pd_op.relu6: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t787 = paddle._C_ops.relu6(t783)
        del t783

        # pd_op.multiply: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        t788 = paddle._C_ops.multiply(t787, t786)
        del t786, t787

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x768x7x7xf32, 192x768x1x1xf32)
        t789 = paddle._C_ops.conv2d(
            t788, t292, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t788, t292

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t790 = paddle._C_ops.reshape(t293, t306)
        del t293

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t791 = paddle._C_ops.add(t789, t790)
        del t789, t790

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t792, t793, t794, t795, t796, t797 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t791,
                t294,
                t295,
                t296,
                t297,
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
        del t791, t295, t294, t297, t296

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        t798 = paddle._C_ops.depthwise_conv2d(
            t792, t298, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t792, t298

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t799 = paddle._C_ops.reshape(t299, t306)
        del t306, t299

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        t800 = paddle._C_ops.add(t798, t799)
        del t798, t799

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, -1x192x7x7xf32)
        t801 = paddle._C_ops.add(t771, t800)
        del t771, t800

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t802, t803, t804, t805, t806, t807 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t801,
                t300,
                t301,
                t302,
                t303,
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
        del t801, t303, t302, t301, t300

        # pd_op.full_int_array: (2xi64) <- ()
        t808 = [1, 1]

        # pd_op.pool2d: (-1x192x1x1xf32) <- (-1x192x7x7xf32, 2xi64)
        t809 = paddle._C_ops.pool2d(
            t802,
            t808,
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
        del t802, t808

        # pd_op.flatten: (-1x192xf32) <- (-1x192x1x1xf32)
        t810 = paddle._C_ops.flatten(t809, 1, 3)
        del t809

        # pd_op.matmul: (-1x102xf32) <- (-1x192xf32, 192x102xf32)
        t811 = paddle._C_ops.matmul(t810, t304, False, False)
        del t810, t304

        return t811
