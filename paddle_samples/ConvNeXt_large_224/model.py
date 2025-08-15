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
        t178: paddle.Tensor,
        t179: paddle.Tensor,
        t180: paddle.Tensor,
        t181: paddle.Tensor,
        input28: paddle.Tensor,
        t182: paddle.Tensor,
        t183: paddle.Tensor,
        t184: paddle.Tensor,
        t185: paddle.Tensor,
        t186: paddle.Tensor,
        t187: paddle.Tensor,
        t188: paddle.Tensor,
        t189: paddle.Tensor,
        input29: paddle.Tensor,
        t190: paddle.Tensor,
        t191: paddle.Tensor,
        t192: paddle.Tensor,
        t193: paddle.Tensor,
        t194: paddle.Tensor,
        t195: paddle.Tensor,
        t196: paddle.Tensor,
        t197: paddle.Tensor,
        input30: paddle.Tensor,
        t198: paddle.Tensor,
        t199: paddle.Tensor,
        t200: paddle.Tensor,
        t201: paddle.Tensor,
        t202: paddle.Tensor,
        t203: paddle.Tensor,
        t204: paddle.Tensor,
        t205: paddle.Tensor,
        input31: paddle.Tensor,
        t206: paddle.Tensor,
        t207: paddle.Tensor,
        t208: paddle.Tensor,
        t209: paddle.Tensor,
        t210: paddle.Tensor,
        t211: paddle.Tensor,
        t212: paddle.Tensor,
        t213: paddle.Tensor,
        input32: paddle.Tensor,
        t214: paddle.Tensor,
        t215: paddle.Tensor,
        t216: paddle.Tensor,
        t217: paddle.Tensor,
        t218: paddle.Tensor,
        t219: paddle.Tensor,
        t220: paddle.Tensor,
        t221: paddle.Tensor,
        input33: paddle.Tensor,
        t222: paddle.Tensor,
        t223: paddle.Tensor,
        t224: paddle.Tensor,
        t225: paddle.Tensor,
        t226: paddle.Tensor,
        t227: paddle.Tensor,
        t228: paddle.Tensor,
        t229: paddle.Tensor,
        input34: paddle.Tensor,
        t230: paddle.Tensor,
        t231: paddle.Tensor,
        t232: paddle.Tensor,
        t233: paddle.Tensor,
        t234: paddle.Tensor,
        t235: paddle.Tensor,
        t236: paddle.Tensor,
        t237: paddle.Tensor,
        input35: paddle.Tensor,
        t238: paddle.Tensor,
        t239: paddle.Tensor,
        t240: paddle.Tensor,
        t241: paddle.Tensor,
        t242: paddle.Tensor,
        t243: paddle.Tensor,
        t244: paddle.Tensor,
        t245: paddle.Tensor,
        input36: paddle.Tensor,
        t246: paddle.Tensor,
        t247: paddle.Tensor,
        t248: paddle.Tensor,
        t249: paddle.Tensor,
        t250: paddle.Tensor,
        t251: paddle.Tensor,
        t252: paddle.Tensor,
        t253: paddle.Tensor,
        input37: paddle.Tensor,
        t254: paddle.Tensor,
        t255: paddle.Tensor,
        t256: paddle.Tensor,
        t257: paddle.Tensor,
        t258: paddle.Tensor,
        t259: paddle.Tensor,
        t260: paddle.Tensor,
        t261: paddle.Tensor,
        input38: paddle.Tensor,
        t262: paddle.Tensor,
        t263: paddle.Tensor,
        t264: paddle.Tensor,
        t265: paddle.Tensor,
        t266: paddle.Tensor,
        t267: paddle.Tensor,
        t268: paddle.Tensor,
        t269: paddle.Tensor,
        input39: paddle.Tensor,
        input40: paddle.Tensor,
        input41: paddle.Tensor,
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
        input42: paddle.Tensor,
        t280: paddle.Tensor,
        t281: paddle.Tensor,
        t282: paddle.Tensor,
        t283: paddle.Tensor,
        t284: paddle.Tensor,
        t285: paddle.Tensor,
        t286: paddle.Tensor,
        t287: paddle.Tensor,
        input43: paddle.Tensor,
        t288: paddle.Tensor,
        t289: paddle.Tensor,
        t290: paddle.Tensor,
        t291: paddle.Tensor,
        t292: paddle.Tensor,
        t293: paddle.Tensor,
        t294: paddle.Tensor,
        t295: paddle.Tensor,
        input44: paddle.Tensor,
        t296: paddle.Tensor,
        t297: paddle.Tensor,
        t298: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x3x224x224xf32, 192x3x4x4xf32)
        t299 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t300 = [1, -1, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t301 = paddle._C_ops.reshape(t1, t300)
        del t1

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 1x192x1x1xf32)
        t302 = paddle._C_ops.add(t299, t301)
        del t299, t301

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x192x56x56xf32)
        t303 = paddle._C_ops.mean(t302, [1], True)

        # pd_op.subtract: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x1x56x56xf32)
        t304 = paddle._C_ops.subtract(t302, t303)
        del t302, t303

        # pd_op.pow: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t305 = paddle._C_ops.pow(t304, float("2"))

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x192x56x56xf32)
        t306 = paddle._C_ops.mean(t305, [1], True)
        del t305

        # pd_op.full: (1xf32) <- ()
        t307 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1x56x56xf32) <- (-1x1x56x56xf32, 1xf32)
        t308 = paddle._C_ops.scale(t306, t307, float("1e-06"), True)
        del t306

        # pd_op.sqrt: (-1x1x56x56xf32) <- (-1x1x56x56xf32)
        t309 = paddle._C_ops.sqrt(t308)
        del t308

        # pd_op.divide: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x1x56x56xf32)
        t310 = paddle._C_ops.divide(t304, t309)
        del t309, t304

        # pd_op.full_int_array: (2xi64) <- ()
        t311 = [1, 2]

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        t312 = paddle._C_ops.unsqueeze(input1, t311)
        del input1

        # pd_op.multiply: (-1x192x56x56xf32) <- (192x1x1xf32, -1x192x56x56xf32)
        t313 = paddle._C_ops.multiply(t312, t310)
        del t310, t312

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        t314 = paddle._C_ops.unsqueeze(input2, t311)
        del input2

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x1x1xf32)
        t315 = paddle._C_ops.add(t313, t314)
        del t313, t314

        # pd_op.depthwise_conv2d: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x1x7x7xf32)
        t316 = paddle._C_ops.depthwise_conv2d(
            t315, t2, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t2

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t317 = paddle._C_ops.reshape(t3, t300)
        del t3

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 1x192x1x1xf32)
        t318 = paddle._C_ops.add(t316, t317)
        del t316, t317

        # pd_op.transpose: (-1x56x56x192xf32) <- (-1x192x56x56xf32)
        t319 = paddle._C_ops.transpose(t318, [0, 2, 3, 1])
        del t318

        # pd_op.layer_norm: (-1x56x56x192xf32, -1x56x56xf32, -1x56x56xf32) <- (-1x56x56x192xf32, 192xf32, 192xf32)
        t320, t321, t322 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t319, t4, t5, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t5, t4, t319

        # pd_op.matmul: (-1x56x56x768xf32) <- (-1x56x56x192xf32, 192x768xf32)
        t323 = paddle._C_ops.matmul(t320, t6, False, False)
        del t320, t6

        # pd_op.add: (-1x56x56x768xf32) <- (-1x56x56x768xf32, 768xf32)
        t324 = paddle._C_ops.add(t323, t7)
        del t323, t7

        # pd_op.gelu: (-1x56x56x768xf32) <- (-1x56x56x768xf32)
        t325 = paddle._C_ops.gelu(t324, False)
        del t324

        # pd_op.matmul: (-1x56x56x192xf32) <- (-1x56x56x768xf32, 768x192xf32)
        t326 = paddle._C_ops.matmul(t325, t8, False, False)
        del t325, t8

        # pd_op.add: (-1x56x56x192xf32) <- (-1x56x56x192xf32, 192xf32)
        t327 = paddle._C_ops.add(t326, t9)
        del t326, t9

        # pd_op.multiply: (-1x56x56x192xf32) <- (192xf32, -1x56x56x192xf32)
        t328 = paddle._C_ops.multiply(input3, t327)
        del t327, input3

        # pd_op.transpose: (-1x192x56x56xf32) <- (-1x56x56x192xf32)
        t329 = paddle._C_ops.transpose(t328, [0, 3, 1, 2])
        del t328

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x192x56x56xf32)
        t330 = paddle._C_ops.add(t315, t329)
        del t315, t329

        # pd_op.depthwise_conv2d: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x1x7x7xf32)
        t331 = paddle._C_ops.depthwise_conv2d(
            t330, t10, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t10

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t332 = paddle._C_ops.reshape(t11, t300)
        del t11

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 1x192x1x1xf32)
        t333 = paddle._C_ops.add(t331, t332)
        del t331, t332

        # pd_op.transpose: (-1x56x56x192xf32) <- (-1x192x56x56xf32)
        t334 = paddle._C_ops.transpose(t333, [0, 2, 3, 1])
        del t333

        # pd_op.layer_norm: (-1x56x56x192xf32, -1x56x56xf32, -1x56x56xf32) <- (-1x56x56x192xf32, 192xf32, 192xf32)
        t335, t336, t337 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t334, t12, t13, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t13, t12, t334

        # pd_op.matmul: (-1x56x56x768xf32) <- (-1x56x56x192xf32, 192x768xf32)
        t338 = paddle._C_ops.matmul(t335, t14, False, False)
        del t335, t14

        # pd_op.add: (-1x56x56x768xf32) <- (-1x56x56x768xf32, 768xf32)
        t339 = paddle._C_ops.add(t338, t15)
        del t338, t15

        # pd_op.gelu: (-1x56x56x768xf32) <- (-1x56x56x768xf32)
        t340 = paddle._C_ops.gelu(t339, False)
        del t339

        # pd_op.matmul: (-1x56x56x192xf32) <- (-1x56x56x768xf32, 768x192xf32)
        t341 = paddle._C_ops.matmul(t340, t16, False, False)
        del t340, t16

        # pd_op.add: (-1x56x56x192xf32) <- (-1x56x56x192xf32, 192xf32)
        t342 = paddle._C_ops.add(t341, t17)
        del t341, t17

        # pd_op.multiply: (-1x56x56x192xf32) <- (192xf32, -1x56x56x192xf32)
        t343 = paddle._C_ops.multiply(input4, t342)
        del t342, input4

        # pd_op.transpose: (-1x192x56x56xf32) <- (-1x56x56x192xf32)
        t344 = paddle._C_ops.transpose(t343, [0, 3, 1, 2])
        del t343

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x192x56x56xf32)
        t345 = paddle._C_ops.add(t330, t344)
        del t330, t344

        # pd_op.depthwise_conv2d: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x1x7x7xf32)
        t346 = paddle._C_ops.depthwise_conv2d(
            t345, t18, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t18

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t347 = paddle._C_ops.reshape(t19, t300)
        del t19

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 1x192x1x1xf32)
        t348 = paddle._C_ops.add(t346, t347)
        del t346, t347

        # pd_op.transpose: (-1x56x56x192xf32) <- (-1x192x56x56xf32)
        t349 = paddle._C_ops.transpose(t348, [0, 2, 3, 1])
        del t348

        # pd_op.layer_norm: (-1x56x56x192xf32, -1x56x56xf32, -1x56x56xf32) <- (-1x56x56x192xf32, 192xf32, 192xf32)
        t350, t351, t352 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t349, t20, t21, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t21, t20, t349

        # pd_op.matmul: (-1x56x56x768xf32) <- (-1x56x56x192xf32, 192x768xf32)
        t353 = paddle._C_ops.matmul(t350, t22, False, False)
        del t350, t22

        # pd_op.add: (-1x56x56x768xf32) <- (-1x56x56x768xf32, 768xf32)
        t354 = paddle._C_ops.add(t353, t23)
        del t353, t23

        # pd_op.gelu: (-1x56x56x768xf32) <- (-1x56x56x768xf32)
        t355 = paddle._C_ops.gelu(t354, False)
        del t354

        # pd_op.matmul: (-1x56x56x192xf32) <- (-1x56x56x768xf32, 768x192xf32)
        t356 = paddle._C_ops.matmul(t355, t24, False, False)
        del t355, t24

        # pd_op.add: (-1x56x56x192xf32) <- (-1x56x56x192xf32, 192xf32)
        t357 = paddle._C_ops.add(t356, t25)
        del t356, t25

        # pd_op.multiply: (-1x56x56x192xf32) <- (192xf32, -1x56x56x192xf32)
        t358 = paddle._C_ops.multiply(input5, t357)
        del t357, input5

        # pd_op.transpose: (-1x192x56x56xf32) <- (-1x56x56x192xf32)
        t359 = paddle._C_ops.transpose(t358, [0, 3, 1, 2])
        del t358

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x192x56x56xf32)
        t360 = paddle._C_ops.add(t345, t359)
        del t345, t359

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x192x56x56xf32)
        t361 = paddle._C_ops.mean(t360, [1], True)

        # pd_op.subtract: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x1x56x56xf32)
        t362 = paddle._C_ops.subtract(t360, t361)
        del t360, t361

        # pd_op.pow: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t363 = paddle._C_ops.pow(t362, float("2"))

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x192x56x56xf32)
        t364 = paddle._C_ops.mean(t363, [1], True)
        del t363

        # pd_op.scale: (-1x1x56x56xf32) <- (-1x1x56x56xf32, 1xf32)
        t365 = paddle._C_ops.scale(t364, t307, float("1e-06"), True)
        del t364

        # pd_op.sqrt: (-1x1x56x56xf32) <- (-1x1x56x56xf32)
        t366 = paddle._C_ops.sqrt(t365)
        del t365

        # pd_op.divide: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x1x56x56xf32)
        t367 = paddle._C_ops.divide(t362, t366)
        del t366, t362

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        t368 = paddle._C_ops.unsqueeze(input6, t311)
        del input6

        # pd_op.multiply: (-1x192x56x56xf32) <- (192x1x1xf32, -1x192x56x56xf32)
        t369 = paddle._C_ops.multiply(t368, t367)
        del t367, t368

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        t370 = paddle._C_ops.unsqueeze(input7, t311)
        del input7

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 192x1x1xf32)
        t371 = paddle._C_ops.add(t369, t370)
        del t369, t370

        # pd_op.conv2d: (-1x384x28x28xf32) <- (-1x192x56x56xf32, 384x192x2x2xf32)
        t372 = paddle._C_ops.conv2d(
            t371, t26, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t371, t26

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t373 = paddle._C_ops.reshape(t27, t300)
        del t27

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, 1x384x1x1xf32)
        t374 = paddle._C_ops.add(t372, t373)
        del t372, t373

        # pd_op.depthwise_conv2d: (-1x384x28x28xf32) <- (-1x384x28x28xf32, 384x1x7x7xf32)
        t375 = paddle._C_ops.depthwise_conv2d(
            t374, t28, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t28

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t376 = paddle._C_ops.reshape(t29, t300)
        del t29

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, 1x384x1x1xf32)
        t377 = paddle._C_ops.add(t375, t376)
        del t375, t376

        # pd_op.transpose: (-1x28x28x384xf32) <- (-1x384x28x28xf32)
        t378 = paddle._C_ops.transpose(t377, [0, 2, 3, 1])
        del t377

        # pd_op.layer_norm: (-1x28x28x384xf32, -1x28x28xf32, -1x28x28xf32) <- (-1x28x28x384xf32, 384xf32, 384xf32)
        t379, t380, t381 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t378, t30, t31, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t31, t30, t378

        # pd_op.matmul: (-1x28x28x1536xf32) <- (-1x28x28x384xf32, 384x1536xf32)
        t382 = paddle._C_ops.matmul(t379, t32, False, False)
        del t379, t32

        # pd_op.add: (-1x28x28x1536xf32) <- (-1x28x28x1536xf32, 1536xf32)
        t383 = paddle._C_ops.add(t382, t33)
        del t382, t33

        # pd_op.gelu: (-1x28x28x1536xf32) <- (-1x28x28x1536xf32)
        t384 = paddle._C_ops.gelu(t383, False)
        del t383

        # pd_op.matmul: (-1x28x28x384xf32) <- (-1x28x28x1536xf32, 1536x384xf32)
        t385 = paddle._C_ops.matmul(t384, t34, False, False)
        del t384, t34

        # pd_op.add: (-1x28x28x384xf32) <- (-1x28x28x384xf32, 384xf32)
        t386 = paddle._C_ops.add(t385, t35)
        del t385, t35

        # pd_op.multiply: (-1x28x28x384xf32) <- (384xf32, -1x28x28x384xf32)
        t387 = paddle._C_ops.multiply(input8, t386)
        del t386, input8

        # pd_op.transpose: (-1x384x28x28xf32) <- (-1x28x28x384xf32)
        t388 = paddle._C_ops.transpose(t387, [0, 3, 1, 2])
        del t387

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x384x28x28xf32)
        t389 = paddle._C_ops.add(t374, t388)
        del t374, t388

        # pd_op.depthwise_conv2d: (-1x384x28x28xf32) <- (-1x384x28x28xf32, 384x1x7x7xf32)
        t390 = paddle._C_ops.depthwise_conv2d(
            t389, t36, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t36

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t391 = paddle._C_ops.reshape(t37, t300)
        del t37

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, 1x384x1x1xf32)
        t392 = paddle._C_ops.add(t390, t391)
        del t390, t391

        # pd_op.transpose: (-1x28x28x384xf32) <- (-1x384x28x28xf32)
        t393 = paddle._C_ops.transpose(t392, [0, 2, 3, 1])
        del t392

        # pd_op.layer_norm: (-1x28x28x384xf32, -1x28x28xf32, -1x28x28xf32) <- (-1x28x28x384xf32, 384xf32, 384xf32)
        t394, t395, t396 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t393, t38, t39, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t39, t38, t393

        # pd_op.matmul: (-1x28x28x1536xf32) <- (-1x28x28x384xf32, 384x1536xf32)
        t397 = paddle._C_ops.matmul(t394, t40, False, False)
        del t394, t40

        # pd_op.add: (-1x28x28x1536xf32) <- (-1x28x28x1536xf32, 1536xf32)
        t398 = paddle._C_ops.add(t397, t41)
        del t397, t41

        # pd_op.gelu: (-1x28x28x1536xf32) <- (-1x28x28x1536xf32)
        t399 = paddle._C_ops.gelu(t398, False)
        del t398

        # pd_op.matmul: (-1x28x28x384xf32) <- (-1x28x28x1536xf32, 1536x384xf32)
        t400 = paddle._C_ops.matmul(t399, t42, False, False)
        del t399, t42

        # pd_op.add: (-1x28x28x384xf32) <- (-1x28x28x384xf32, 384xf32)
        t401 = paddle._C_ops.add(t400, t43)
        del t400, t43

        # pd_op.multiply: (-1x28x28x384xf32) <- (384xf32, -1x28x28x384xf32)
        t402 = paddle._C_ops.multiply(input9, t401)
        del t401, input9

        # pd_op.transpose: (-1x384x28x28xf32) <- (-1x28x28x384xf32)
        t403 = paddle._C_ops.transpose(t402, [0, 3, 1, 2])
        del t402

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x384x28x28xf32)
        t404 = paddle._C_ops.add(t389, t403)
        del t389, t403

        # pd_op.depthwise_conv2d: (-1x384x28x28xf32) <- (-1x384x28x28xf32, 384x1x7x7xf32)
        t405 = paddle._C_ops.depthwise_conv2d(
            t404, t44, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t44

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t406 = paddle._C_ops.reshape(t45, t300)
        del t45

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, 1x384x1x1xf32)
        t407 = paddle._C_ops.add(t405, t406)
        del t405, t406

        # pd_op.transpose: (-1x28x28x384xf32) <- (-1x384x28x28xf32)
        t408 = paddle._C_ops.transpose(t407, [0, 2, 3, 1])
        del t407

        # pd_op.layer_norm: (-1x28x28x384xf32, -1x28x28xf32, -1x28x28xf32) <- (-1x28x28x384xf32, 384xf32, 384xf32)
        t409, t410, t411 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t408, t46, t47, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t47, t46, t408

        # pd_op.matmul: (-1x28x28x1536xf32) <- (-1x28x28x384xf32, 384x1536xf32)
        t412 = paddle._C_ops.matmul(t409, t48, False, False)
        del t409, t48

        # pd_op.add: (-1x28x28x1536xf32) <- (-1x28x28x1536xf32, 1536xf32)
        t413 = paddle._C_ops.add(t412, t49)
        del t412, t49

        # pd_op.gelu: (-1x28x28x1536xf32) <- (-1x28x28x1536xf32)
        t414 = paddle._C_ops.gelu(t413, False)
        del t413

        # pd_op.matmul: (-1x28x28x384xf32) <- (-1x28x28x1536xf32, 1536x384xf32)
        t415 = paddle._C_ops.matmul(t414, t50, False, False)
        del t414, t50

        # pd_op.add: (-1x28x28x384xf32) <- (-1x28x28x384xf32, 384xf32)
        t416 = paddle._C_ops.add(t415, t51)
        del t415, t51

        # pd_op.multiply: (-1x28x28x384xf32) <- (384xf32, -1x28x28x384xf32)
        t417 = paddle._C_ops.multiply(input10, t416)
        del t416, input10

        # pd_op.transpose: (-1x384x28x28xf32) <- (-1x28x28x384xf32)
        t418 = paddle._C_ops.transpose(t417, [0, 3, 1, 2])
        del t417

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x384x28x28xf32)
        t419 = paddle._C_ops.add(t404, t418)
        del t404, t418

        # pd_op.mean: (-1x1x28x28xf32) <- (-1x384x28x28xf32)
        t420 = paddle._C_ops.mean(t419, [1], True)

        # pd_op.subtract: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x1x28x28xf32)
        t421 = paddle._C_ops.subtract(t419, t420)
        del t419, t420

        # pd_op.pow: (-1x384x28x28xf32) <- (-1x384x28x28xf32)
        t422 = paddle._C_ops.pow(t421, float("2"))

        # pd_op.mean: (-1x1x28x28xf32) <- (-1x384x28x28xf32)
        t423 = paddle._C_ops.mean(t422, [1], True)
        del t422

        # pd_op.scale: (-1x1x28x28xf32) <- (-1x1x28x28xf32, 1xf32)
        t424 = paddle._C_ops.scale(t423, t307, float("1e-06"), True)
        del t423

        # pd_op.sqrt: (-1x1x28x28xf32) <- (-1x1x28x28xf32)
        t425 = paddle._C_ops.sqrt(t424)
        del t424

        # pd_op.divide: (-1x384x28x28xf32) <- (-1x384x28x28xf32, -1x1x28x28xf32)
        t426 = paddle._C_ops.divide(t421, t425)
        del t425, t421

        # pd_op.unsqueeze: (384x1x1xf32) <- (384xf32, 2xi64)
        t427 = paddle._C_ops.unsqueeze(input11, t311)
        del input11

        # pd_op.multiply: (-1x384x28x28xf32) <- (384x1x1xf32, -1x384x28x28xf32)
        t428 = paddle._C_ops.multiply(t427, t426)
        del t426, t427

        # pd_op.unsqueeze: (384x1x1xf32) <- (384xf32, 2xi64)
        t429 = paddle._C_ops.unsqueeze(input12, t311)
        del input12

        # pd_op.add: (-1x384x28x28xf32) <- (-1x384x28x28xf32, 384x1x1xf32)
        t430 = paddle._C_ops.add(t428, t429)
        del t428, t429

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x384x28x28xf32, 768x384x2x2xf32)
        t431 = paddle._C_ops.conv2d(
            t430, t52, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t430, t52

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t432 = paddle._C_ops.reshape(t53, t300)
        del t53

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t433 = paddle._C_ops.add(t431, t432)
        del t431, t432

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t434 = paddle._C_ops.depthwise_conv2d(
            t433, t54, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t54

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t435 = paddle._C_ops.reshape(t55, t300)
        del t55

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t436 = paddle._C_ops.add(t434, t435)
        del t434, t435

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t437 = paddle._C_ops.transpose(t436, [0, 2, 3, 1])
        del t436

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t438, t439, t440 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t437, t56, t57, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t57, t56, t437

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t441 = paddle._C_ops.matmul(t438, t58, False, False)
        del t438, t58

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t442 = paddle._C_ops.add(t441, t59)
        del t441, t59

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t443 = paddle._C_ops.gelu(t442, False)
        del t442

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t444 = paddle._C_ops.matmul(t443, t60, False, False)
        del t443, t60

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t445 = paddle._C_ops.add(t444, t61)
        del t444, t61

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t446 = paddle._C_ops.multiply(input13, t445)
        del t445, input13

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t447 = paddle._C_ops.transpose(t446, [0, 3, 1, 2])
        del t446

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t448 = paddle._C_ops.add(t433, t447)
        del t433, t447

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t449 = paddle._C_ops.depthwise_conv2d(
            t448, t62, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t62

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t450 = paddle._C_ops.reshape(t63, t300)
        del t63

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t451 = paddle._C_ops.add(t449, t450)
        del t449, t450

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t452 = paddle._C_ops.transpose(t451, [0, 2, 3, 1])
        del t451

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t453, t454, t455 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t452, t64, t65, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t65, t64, t452

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t456 = paddle._C_ops.matmul(t453, t66, False, False)
        del t453, t66

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t457 = paddle._C_ops.add(t456, t67)
        del t456, t67

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t458 = paddle._C_ops.gelu(t457, False)
        del t457

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t459 = paddle._C_ops.matmul(t458, t68, False, False)
        del t458, t68

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t460 = paddle._C_ops.add(t459, t69)
        del t459, t69

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t461 = paddle._C_ops.multiply(input14, t460)
        del t460, input14

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t462 = paddle._C_ops.transpose(t461, [0, 3, 1, 2])
        del t461

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t463 = paddle._C_ops.add(t448, t462)
        del t448, t462

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t464 = paddle._C_ops.depthwise_conv2d(
            t463, t70, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t70

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t465 = paddle._C_ops.reshape(t71, t300)
        del t71

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t466 = paddle._C_ops.add(t464, t465)
        del t464, t465

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t467 = paddle._C_ops.transpose(t466, [0, 2, 3, 1])
        del t466

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t468, t469, t470 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t467, t72, t73, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t73, t72, t467

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t471 = paddle._C_ops.matmul(t468, t74, False, False)
        del t468, t74

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t472 = paddle._C_ops.add(t471, t75)
        del t471, t75

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t473 = paddle._C_ops.gelu(t472, False)
        del t472

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t474 = paddle._C_ops.matmul(t473, t76, False, False)
        del t473, t76

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t475 = paddle._C_ops.add(t474, t77)
        del t474, t77

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t476 = paddle._C_ops.multiply(input15, t475)
        del t475, input15

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t477 = paddle._C_ops.transpose(t476, [0, 3, 1, 2])
        del t476

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t478 = paddle._C_ops.add(t463, t477)
        del t463, t477

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t479 = paddle._C_ops.depthwise_conv2d(
            t478, t78, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t78

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t480 = paddle._C_ops.reshape(t79, t300)
        del t79

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t481 = paddle._C_ops.add(t479, t480)
        del t479, t480

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t482 = paddle._C_ops.transpose(t481, [0, 2, 3, 1])
        del t481

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t483, t484, t485 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t482, t80, t81, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t81, t80, t482

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t486 = paddle._C_ops.matmul(t483, t82, False, False)
        del t483, t82

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t487 = paddle._C_ops.add(t486, t83)
        del t486, t83

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t488 = paddle._C_ops.gelu(t487, False)
        del t487

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t489 = paddle._C_ops.matmul(t488, t84, False, False)
        del t488, t84

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t490 = paddle._C_ops.add(t489, t85)
        del t489, t85

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t491 = paddle._C_ops.multiply(input16, t490)
        del t490, input16

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t492 = paddle._C_ops.transpose(t491, [0, 3, 1, 2])
        del t491

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t493 = paddle._C_ops.add(t478, t492)
        del t478, t492

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t494 = paddle._C_ops.depthwise_conv2d(
            t493, t86, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t86

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t495 = paddle._C_ops.reshape(t87, t300)
        del t87

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t496 = paddle._C_ops.add(t494, t495)
        del t494, t495

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t497 = paddle._C_ops.transpose(t496, [0, 2, 3, 1])
        del t496

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t498, t499, t500 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t497, t88, t89, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t89, t88, t497

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t501 = paddle._C_ops.matmul(t498, t90, False, False)
        del t498, t90

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t502 = paddle._C_ops.add(t501, t91)
        del t501, t91

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t503 = paddle._C_ops.gelu(t502, False)
        del t502

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t504 = paddle._C_ops.matmul(t503, t92, False, False)
        del t503, t92

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t505 = paddle._C_ops.add(t504, t93)
        del t504, t93

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t506 = paddle._C_ops.multiply(input17, t505)
        del t505, input17

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t507 = paddle._C_ops.transpose(t506, [0, 3, 1, 2])
        del t506

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t508 = paddle._C_ops.add(t493, t507)
        del t493, t507

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t509 = paddle._C_ops.depthwise_conv2d(
            t508, t94, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t94

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t510 = paddle._C_ops.reshape(t95, t300)
        del t95

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t511 = paddle._C_ops.add(t509, t510)
        del t509, t510

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t512 = paddle._C_ops.transpose(t511, [0, 2, 3, 1])
        del t511

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t513, t514, t515 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t512, t96, t97, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t97, t96, t512

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t516 = paddle._C_ops.matmul(t513, t98, False, False)
        del t513, t98

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t517 = paddle._C_ops.add(t516, t99)
        del t516, t99

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t518 = paddle._C_ops.gelu(t517, False)
        del t517

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t519 = paddle._C_ops.matmul(t518, t100, False, False)
        del t518, t100

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t520 = paddle._C_ops.add(t519, t101)
        del t519, t101

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t521 = paddle._C_ops.multiply(input18, t520)
        del t520, input18

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t522 = paddle._C_ops.transpose(t521, [0, 3, 1, 2])
        del t521

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t523 = paddle._C_ops.add(t508, t522)
        del t508, t522

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t524 = paddle._C_ops.depthwise_conv2d(
            t523, t102, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t102

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t525 = paddle._C_ops.reshape(t103, t300)
        del t103

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t526 = paddle._C_ops.add(t524, t525)
        del t524, t525

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t527 = paddle._C_ops.transpose(t526, [0, 2, 3, 1])
        del t526

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t528, t529, t530 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t527, t104, t105, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t105, t104, t527

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t531 = paddle._C_ops.matmul(t528, t106, False, False)
        del t528, t106

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t532 = paddle._C_ops.add(t531, t107)
        del t531, t107

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t533 = paddle._C_ops.gelu(t532, False)
        del t532

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t534 = paddle._C_ops.matmul(t533, t108, False, False)
        del t533, t108

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t535 = paddle._C_ops.add(t534, t109)
        del t534, t109

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t536 = paddle._C_ops.multiply(input19, t535)
        del t535, input19

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t537 = paddle._C_ops.transpose(t536, [0, 3, 1, 2])
        del t536

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t538 = paddle._C_ops.add(t523, t537)
        del t523, t537

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t539 = paddle._C_ops.depthwise_conv2d(
            t538, t110, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t110

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t540 = paddle._C_ops.reshape(t111, t300)
        del t111

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t541 = paddle._C_ops.add(t539, t540)
        del t539, t540

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t542 = paddle._C_ops.transpose(t541, [0, 2, 3, 1])
        del t541

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t543, t544, t545 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t542, t112, t113, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t113, t112, t542

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t546 = paddle._C_ops.matmul(t543, t114, False, False)
        del t543, t114

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t547 = paddle._C_ops.add(t546, t115)
        del t546, t115

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t548 = paddle._C_ops.gelu(t547, False)
        del t547

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t549 = paddle._C_ops.matmul(t548, t116, False, False)
        del t548, t116

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t550 = paddle._C_ops.add(t549, t117)
        del t549, t117

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t551 = paddle._C_ops.multiply(input20, t550)
        del t550, input20

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t552 = paddle._C_ops.transpose(t551, [0, 3, 1, 2])
        del t551

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t553 = paddle._C_ops.add(t538, t552)
        del t538, t552

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t554 = paddle._C_ops.depthwise_conv2d(
            t553, t118, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t118

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t555 = paddle._C_ops.reshape(t119, t300)
        del t119

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t556 = paddle._C_ops.add(t554, t555)
        del t554, t555

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t557 = paddle._C_ops.transpose(t556, [0, 2, 3, 1])
        del t556

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t558, t559, t560 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t557, t120, t121, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t121, t120, t557

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t561 = paddle._C_ops.matmul(t558, t122, False, False)
        del t558, t122

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t562 = paddle._C_ops.add(t561, t123)
        del t561, t123

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t563 = paddle._C_ops.gelu(t562, False)
        del t562

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t564 = paddle._C_ops.matmul(t563, t124, False, False)
        del t563, t124

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t565 = paddle._C_ops.add(t564, t125)
        del t564, t125

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t566 = paddle._C_ops.multiply(input21, t565)
        del t565, input21

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t567 = paddle._C_ops.transpose(t566, [0, 3, 1, 2])
        del t566

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t568 = paddle._C_ops.add(t553, t567)
        del t553, t567

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t569 = paddle._C_ops.depthwise_conv2d(
            t568, t126, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t126

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t570 = paddle._C_ops.reshape(t127, t300)
        del t127

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t571 = paddle._C_ops.add(t569, t570)
        del t569, t570

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t572 = paddle._C_ops.transpose(t571, [0, 2, 3, 1])
        del t571

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t573, t574, t575 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t572, t128, t129, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t129, t128, t572

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t576 = paddle._C_ops.matmul(t573, t130, False, False)
        del t573, t130

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t577 = paddle._C_ops.add(t576, t131)
        del t576, t131

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t578 = paddle._C_ops.gelu(t577, False)
        del t577

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t579 = paddle._C_ops.matmul(t578, t132, False, False)
        del t578, t132

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t580 = paddle._C_ops.add(t579, t133)
        del t579, t133

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t581 = paddle._C_ops.multiply(input22, t580)
        del t580, input22

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t582 = paddle._C_ops.transpose(t581, [0, 3, 1, 2])
        del t581

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t583 = paddle._C_ops.add(t568, t582)
        del t568, t582

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t584 = paddle._C_ops.depthwise_conv2d(
            t583, t134, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t134

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t585 = paddle._C_ops.reshape(t135, t300)
        del t135

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t586 = paddle._C_ops.add(t584, t585)
        del t584, t585

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t587 = paddle._C_ops.transpose(t586, [0, 2, 3, 1])
        del t586

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t588, t589, t590 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t587, t136, t137, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t137, t136, t587

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t591 = paddle._C_ops.matmul(t588, t138, False, False)
        del t588, t138

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t592 = paddle._C_ops.add(t591, t139)
        del t591, t139

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t593 = paddle._C_ops.gelu(t592, False)
        del t592

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t594 = paddle._C_ops.matmul(t593, t140, False, False)
        del t593, t140

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t595 = paddle._C_ops.add(t594, t141)
        del t594, t141

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t596 = paddle._C_ops.multiply(input23, t595)
        del t595, input23

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t597 = paddle._C_ops.transpose(t596, [0, 3, 1, 2])
        del t596

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t598 = paddle._C_ops.add(t583, t597)
        del t583, t597

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t599 = paddle._C_ops.depthwise_conv2d(
            t598, t142, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t142

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t600 = paddle._C_ops.reshape(t143, t300)
        del t143

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t601 = paddle._C_ops.add(t599, t600)
        del t599, t600

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t602 = paddle._C_ops.transpose(t601, [0, 2, 3, 1])
        del t601

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t603, t604, t605 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t602, t144, t145, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t145, t144, t602

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t606 = paddle._C_ops.matmul(t603, t146, False, False)
        del t603, t146

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t607 = paddle._C_ops.add(t606, t147)
        del t606, t147

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t608 = paddle._C_ops.gelu(t607, False)
        del t607

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t609 = paddle._C_ops.matmul(t608, t148, False, False)
        del t608, t148

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t610 = paddle._C_ops.add(t609, t149)
        del t609, t149

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t611 = paddle._C_ops.multiply(input24, t610)
        del t610, input24

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t612 = paddle._C_ops.transpose(t611, [0, 3, 1, 2])
        del t611

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t613 = paddle._C_ops.add(t598, t612)
        del t598, t612

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t614 = paddle._C_ops.depthwise_conv2d(
            t613, t150, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t150

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t615 = paddle._C_ops.reshape(t151, t300)
        del t151

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t616 = paddle._C_ops.add(t614, t615)
        del t614, t615

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t617 = paddle._C_ops.transpose(t616, [0, 2, 3, 1])
        del t616

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t618, t619, t620 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t617, t152, t153, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t153, t152, t617

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t621 = paddle._C_ops.matmul(t618, t154, False, False)
        del t618, t154

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t622 = paddle._C_ops.add(t621, t155)
        del t621, t155

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t623 = paddle._C_ops.gelu(t622, False)
        del t622

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t624 = paddle._C_ops.matmul(t623, t156, False, False)
        del t623, t156

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t625 = paddle._C_ops.add(t624, t157)
        del t624, t157

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t626 = paddle._C_ops.multiply(input25, t625)
        del t625, input25

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t627 = paddle._C_ops.transpose(t626, [0, 3, 1, 2])
        del t626

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t628 = paddle._C_ops.add(t613, t627)
        del t613, t627

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t629 = paddle._C_ops.depthwise_conv2d(
            t628, t158, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t158

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t630 = paddle._C_ops.reshape(t159, t300)
        del t159

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t631 = paddle._C_ops.add(t629, t630)
        del t629, t630

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t632 = paddle._C_ops.transpose(t631, [0, 2, 3, 1])
        del t631

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t633, t634, t635 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t632, t160, t161, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t161, t160, t632

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t636 = paddle._C_ops.matmul(t633, t162, False, False)
        del t633, t162

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t637 = paddle._C_ops.add(t636, t163)
        del t636, t163

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t638 = paddle._C_ops.gelu(t637, False)
        del t637

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t639 = paddle._C_ops.matmul(t638, t164, False, False)
        del t638, t164

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t640 = paddle._C_ops.add(t639, t165)
        del t639, t165

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t641 = paddle._C_ops.multiply(input26, t640)
        del t640, input26

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t642 = paddle._C_ops.transpose(t641, [0, 3, 1, 2])
        del t641

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t643 = paddle._C_ops.add(t628, t642)
        del t628, t642

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t644 = paddle._C_ops.depthwise_conv2d(
            t643, t166, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t166

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t645 = paddle._C_ops.reshape(t167, t300)
        del t167

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t646 = paddle._C_ops.add(t644, t645)
        del t644, t645

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t647 = paddle._C_ops.transpose(t646, [0, 2, 3, 1])
        del t646

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t648, t649, t650 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t647, t168, t169, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t169, t168, t647

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t651 = paddle._C_ops.matmul(t648, t170, False, False)
        del t648, t170

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t652 = paddle._C_ops.add(t651, t171)
        del t651, t171

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t653 = paddle._C_ops.gelu(t652, False)
        del t652

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t654 = paddle._C_ops.matmul(t653, t172, False, False)
        del t653, t172

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t655 = paddle._C_ops.add(t654, t173)
        del t654, t173

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t656 = paddle._C_ops.multiply(input27, t655)
        del t655, input27

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t657 = paddle._C_ops.transpose(t656, [0, 3, 1, 2])
        del t656

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t658 = paddle._C_ops.add(t643, t657)
        del t643, t657

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t659 = paddle._C_ops.depthwise_conv2d(
            t658, t174, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t174

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t660 = paddle._C_ops.reshape(t175, t300)
        del t175

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t661 = paddle._C_ops.add(t659, t660)
        del t659, t660

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t662 = paddle._C_ops.transpose(t661, [0, 2, 3, 1])
        del t661

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t663, t664, t665 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t662, t176, t177, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t177, t176, t662

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t666 = paddle._C_ops.matmul(t663, t178, False, False)
        del t663, t178

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t667 = paddle._C_ops.add(t666, t179)
        del t666, t179

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t668 = paddle._C_ops.gelu(t667, False)
        del t667

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t669 = paddle._C_ops.matmul(t668, t180, False, False)
        del t668, t180

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t670 = paddle._C_ops.add(t669, t181)
        del t669, t181

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t671 = paddle._C_ops.multiply(input28, t670)
        del t670, input28

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t672 = paddle._C_ops.transpose(t671, [0, 3, 1, 2])
        del t671

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t673 = paddle._C_ops.add(t658, t672)
        del t658, t672

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t674 = paddle._C_ops.depthwise_conv2d(
            t673, t182, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t182

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t675 = paddle._C_ops.reshape(t183, t300)
        del t183

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t676 = paddle._C_ops.add(t674, t675)
        del t674, t675

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t677 = paddle._C_ops.transpose(t676, [0, 2, 3, 1])
        del t676

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t678, t679, t680 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t677, t184, t185, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t185, t184, t677

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t681 = paddle._C_ops.matmul(t678, t186, False, False)
        del t678, t186

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t682 = paddle._C_ops.add(t681, t187)
        del t681, t187

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t683 = paddle._C_ops.gelu(t682, False)
        del t682

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t684 = paddle._C_ops.matmul(t683, t188, False, False)
        del t683, t188

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t685 = paddle._C_ops.add(t684, t189)
        del t684, t189

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t686 = paddle._C_ops.multiply(input29, t685)
        del t685, input29

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t687 = paddle._C_ops.transpose(t686, [0, 3, 1, 2])
        del t686

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t688 = paddle._C_ops.add(t673, t687)
        del t673, t687

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t689 = paddle._C_ops.depthwise_conv2d(
            t688, t190, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t190

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t690 = paddle._C_ops.reshape(t191, t300)
        del t191

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t691 = paddle._C_ops.add(t689, t690)
        del t689, t690

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t692 = paddle._C_ops.transpose(t691, [0, 2, 3, 1])
        del t691

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t693, t694, t695 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t692, t192, t193, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t193, t192, t692

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t696 = paddle._C_ops.matmul(t693, t194, False, False)
        del t693, t194

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t697 = paddle._C_ops.add(t696, t195)
        del t696, t195

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t698 = paddle._C_ops.gelu(t697, False)
        del t697

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t699 = paddle._C_ops.matmul(t698, t196, False, False)
        del t698, t196

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t700 = paddle._C_ops.add(t699, t197)
        del t699, t197

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t701 = paddle._C_ops.multiply(input30, t700)
        del t700, input30

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t702 = paddle._C_ops.transpose(t701, [0, 3, 1, 2])
        del t701

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t703 = paddle._C_ops.add(t688, t702)
        del t688, t702

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t704 = paddle._C_ops.depthwise_conv2d(
            t703, t198, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t198

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t705 = paddle._C_ops.reshape(t199, t300)
        del t199

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t706 = paddle._C_ops.add(t704, t705)
        del t704, t705

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t707 = paddle._C_ops.transpose(t706, [0, 2, 3, 1])
        del t706

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t708, t709, t710 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t707, t200, t201, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t201, t200, t707

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t711 = paddle._C_ops.matmul(t708, t202, False, False)
        del t708, t202

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t712 = paddle._C_ops.add(t711, t203)
        del t711, t203

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t713 = paddle._C_ops.gelu(t712, False)
        del t712

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t714 = paddle._C_ops.matmul(t713, t204, False, False)
        del t713, t204

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t715 = paddle._C_ops.add(t714, t205)
        del t714, t205

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t716 = paddle._C_ops.multiply(input31, t715)
        del t715, input31

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t717 = paddle._C_ops.transpose(t716, [0, 3, 1, 2])
        del t716

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t718 = paddle._C_ops.add(t703, t717)
        del t703, t717

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t719 = paddle._C_ops.depthwise_conv2d(
            t718, t206, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t206

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t720 = paddle._C_ops.reshape(t207, t300)
        del t207

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t721 = paddle._C_ops.add(t719, t720)
        del t719, t720

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t722 = paddle._C_ops.transpose(t721, [0, 2, 3, 1])
        del t721

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t723, t724, t725 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t722, t208, t209, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t209, t208, t722

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t726 = paddle._C_ops.matmul(t723, t210, False, False)
        del t723, t210

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t727 = paddle._C_ops.add(t726, t211)
        del t726, t211

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t728 = paddle._C_ops.gelu(t727, False)
        del t727

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t729 = paddle._C_ops.matmul(t728, t212, False, False)
        del t728, t212

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t730 = paddle._C_ops.add(t729, t213)
        del t729, t213

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t731 = paddle._C_ops.multiply(input32, t730)
        del t730, input32

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t732 = paddle._C_ops.transpose(t731, [0, 3, 1, 2])
        del t731

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t733 = paddle._C_ops.add(t718, t732)
        del t718, t732

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t734 = paddle._C_ops.depthwise_conv2d(
            t733, t214, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t214

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t735 = paddle._C_ops.reshape(t215, t300)
        del t215

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t736 = paddle._C_ops.add(t734, t735)
        del t734, t735

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t737 = paddle._C_ops.transpose(t736, [0, 2, 3, 1])
        del t736

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t738, t739, t740 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t737, t216, t217, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t217, t216, t737

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t741 = paddle._C_ops.matmul(t738, t218, False, False)
        del t738, t218

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t742 = paddle._C_ops.add(t741, t219)
        del t741, t219

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t743 = paddle._C_ops.gelu(t742, False)
        del t742

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t744 = paddle._C_ops.matmul(t743, t220, False, False)
        del t743, t220

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t745 = paddle._C_ops.add(t744, t221)
        del t744, t221

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t746 = paddle._C_ops.multiply(input33, t745)
        del t745, input33

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t747 = paddle._C_ops.transpose(t746, [0, 3, 1, 2])
        del t746

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t748 = paddle._C_ops.add(t733, t747)
        del t733, t747

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t749 = paddle._C_ops.depthwise_conv2d(
            t748, t222, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t222

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t750 = paddle._C_ops.reshape(t223, t300)
        del t223

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t751 = paddle._C_ops.add(t749, t750)
        del t749, t750

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t752 = paddle._C_ops.transpose(t751, [0, 2, 3, 1])
        del t751

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t753, t754, t755 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t752, t224, t225, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t225, t224, t752

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t756 = paddle._C_ops.matmul(t753, t226, False, False)
        del t753, t226

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t757 = paddle._C_ops.add(t756, t227)
        del t756, t227

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t758 = paddle._C_ops.gelu(t757, False)
        del t757

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t759 = paddle._C_ops.matmul(t758, t228, False, False)
        del t758, t228

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t760 = paddle._C_ops.add(t759, t229)
        del t759, t229

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t761 = paddle._C_ops.multiply(input34, t760)
        del t760, input34

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t762 = paddle._C_ops.transpose(t761, [0, 3, 1, 2])
        del t761

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t763 = paddle._C_ops.add(t748, t762)
        del t748, t762

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t764 = paddle._C_ops.depthwise_conv2d(
            t763, t230, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t230

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t765 = paddle._C_ops.reshape(t231, t300)
        del t231

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t766 = paddle._C_ops.add(t764, t765)
        del t764, t765

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t767 = paddle._C_ops.transpose(t766, [0, 2, 3, 1])
        del t766

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t768, t769, t770 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t767, t232, t233, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t233, t232, t767

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t771 = paddle._C_ops.matmul(t768, t234, False, False)
        del t768, t234

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t772 = paddle._C_ops.add(t771, t235)
        del t771, t235

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t773 = paddle._C_ops.gelu(t772, False)
        del t772

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t774 = paddle._C_ops.matmul(t773, t236, False, False)
        del t773, t236

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t775 = paddle._C_ops.add(t774, t237)
        del t774, t237

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t776 = paddle._C_ops.multiply(input35, t775)
        del t775, input35

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t777 = paddle._C_ops.transpose(t776, [0, 3, 1, 2])
        del t776

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t778 = paddle._C_ops.add(t763, t777)
        del t763, t777

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t779 = paddle._C_ops.depthwise_conv2d(
            t778, t238, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t238

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t780 = paddle._C_ops.reshape(t239, t300)
        del t239

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t781 = paddle._C_ops.add(t779, t780)
        del t779, t780

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t782 = paddle._C_ops.transpose(t781, [0, 2, 3, 1])
        del t781

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t783, t784, t785 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t782, t240, t241, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t241, t240, t782

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t786 = paddle._C_ops.matmul(t783, t242, False, False)
        del t783, t242

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t787 = paddle._C_ops.add(t786, t243)
        del t786, t243

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t788 = paddle._C_ops.gelu(t787, False)
        del t787

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t789 = paddle._C_ops.matmul(t788, t244, False, False)
        del t788, t244

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t790 = paddle._C_ops.add(t789, t245)
        del t789, t245

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t791 = paddle._C_ops.multiply(input36, t790)
        del t790, input36

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t792 = paddle._C_ops.transpose(t791, [0, 3, 1, 2])
        del t791

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t793 = paddle._C_ops.add(t778, t792)
        del t778, t792

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t794 = paddle._C_ops.depthwise_conv2d(
            t793, t246, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t246

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t795 = paddle._C_ops.reshape(t247, t300)
        del t247

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t796 = paddle._C_ops.add(t794, t795)
        del t794, t795

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t797 = paddle._C_ops.transpose(t796, [0, 2, 3, 1])
        del t796

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t798, t799, t800 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t797, t248, t249, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t249, t248, t797

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t801 = paddle._C_ops.matmul(t798, t250, False, False)
        del t798, t250

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t802 = paddle._C_ops.add(t801, t251)
        del t801, t251

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t803 = paddle._C_ops.gelu(t802, False)
        del t802

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t804 = paddle._C_ops.matmul(t803, t252, False, False)
        del t803, t252

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t805 = paddle._C_ops.add(t804, t253)
        del t804, t253

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t806 = paddle._C_ops.multiply(input37, t805)
        del t805, input37

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t807 = paddle._C_ops.transpose(t806, [0, 3, 1, 2])
        del t806

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t808 = paddle._C_ops.add(t793, t807)
        del t793, t807

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t809 = paddle._C_ops.depthwise_conv2d(
            t808, t254, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t254

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t810 = paddle._C_ops.reshape(t255, t300)
        del t255

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t811 = paddle._C_ops.add(t809, t810)
        del t809, t810

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t812 = paddle._C_ops.transpose(t811, [0, 2, 3, 1])
        del t811

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t813, t814, t815 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t812, t256, t257, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t257, t256, t812

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t816 = paddle._C_ops.matmul(t813, t258, False, False)
        del t813, t258

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t817 = paddle._C_ops.add(t816, t259)
        del t816, t259

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t818 = paddle._C_ops.gelu(t817, False)
        del t817

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t819 = paddle._C_ops.matmul(t818, t260, False, False)
        del t818, t260

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t820 = paddle._C_ops.add(t819, t261)
        del t819, t261

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t821 = paddle._C_ops.multiply(input38, t820)
        del t820, input38

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t822 = paddle._C_ops.transpose(t821, [0, 3, 1, 2])
        del t821

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t823 = paddle._C_ops.add(t808, t822)
        del t808, t822

        # pd_op.depthwise_conv2d: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x7x7xf32)
        t824 = paddle._C_ops.depthwise_conv2d(
            t823, t262, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t262

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        t825 = paddle._C_ops.reshape(t263, t300)
        del t263

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 1x768x1x1xf32)
        t826 = paddle._C_ops.add(t824, t825)
        del t824, t825

        # pd_op.transpose: (-1x14x14x768xf32) <- (-1x768x14x14xf32)
        t827 = paddle._C_ops.transpose(t826, [0, 2, 3, 1])
        del t826

        # pd_op.layer_norm: (-1x14x14x768xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x768xf32, 768xf32, 768xf32)
        t828, t829, t830 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t827, t264, t265, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t265, t264, t827

        # pd_op.matmul: (-1x14x14x3072xf32) <- (-1x14x14x768xf32, 768x3072xf32)
        t831 = paddle._C_ops.matmul(t828, t266, False, False)
        del t828, t266

        # pd_op.add: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32, 3072xf32)
        t832 = paddle._C_ops.add(t831, t267)
        del t831, t267

        # pd_op.gelu: (-1x14x14x3072xf32) <- (-1x14x14x3072xf32)
        t833 = paddle._C_ops.gelu(t832, False)
        del t832

        # pd_op.matmul: (-1x14x14x768xf32) <- (-1x14x14x3072xf32, 3072x768xf32)
        t834 = paddle._C_ops.matmul(t833, t268, False, False)
        del t833, t268

        # pd_op.add: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 768xf32)
        t835 = paddle._C_ops.add(t834, t269)
        del t834, t269

        # pd_op.multiply: (-1x14x14x768xf32) <- (768xf32, -1x14x14x768xf32)
        t836 = paddle._C_ops.multiply(input39, t835)
        del t835, input39

        # pd_op.transpose: (-1x768x14x14xf32) <- (-1x14x14x768xf32)
        t837 = paddle._C_ops.transpose(t836, [0, 3, 1, 2])
        del t836

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x768x14x14xf32)
        t838 = paddle._C_ops.add(t823, t837)
        del t823, t837

        # pd_op.mean: (-1x1x14x14xf32) <- (-1x768x14x14xf32)
        t839 = paddle._C_ops.mean(t838, [1], True)

        # pd_op.subtract: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x14x14xf32)
        t840 = paddle._C_ops.subtract(t838, t839)
        del t838, t839

        # pd_op.pow: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        t841 = paddle._C_ops.pow(t840, float("2"))

        # pd_op.mean: (-1x1x14x14xf32) <- (-1x768x14x14xf32)
        t842 = paddle._C_ops.mean(t841, [1], True)
        del t841

        # pd_op.scale: (-1x1x14x14xf32) <- (-1x1x14x14xf32, 1xf32)
        t843 = paddle._C_ops.scale(t842, t307, float("1e-06"), True)
        del t307, t842

        # pd_op.sqrt: (-1x1x14x14xf32) <- (-1x1x14x14xf32)
        t844 = paddle._C_ops.sqrt(t843)
        del t843

        # pd_op.divide: (-1x768x14x14xf32) <- (-1x768x14x14xf32, -1x1x14x14xf32)
        t845 = paddle._C_ops.divide(t840, t844)
        del t844, t840

        # pd_op.unsqueeze: (768x1x1xf32) <- (768xf32, 2xi64)
        t846 = paddle._C_ops.unsqueeze(input40, t311)
        del input40

        # pd_op.multiply: (-1x768x14x14xf32) <- (768x1x1xf32, -1x768x14x14xf32)
        t847 = paddle._C_ops.multiply(t846, t845)
        del t845, t846

        # pd_op.unsqueeze: (768x1x1xf32) <- (768xf32, 2xi64)
        t848 = paddle._C_ops.unsqueeze(input41, t311)
        del input41, t311

        # pd_op.add: (-1x768x14x14xf32) <- (-1x768x14x14xf32, 768x1x1xf32)
        t849 = paddle._C_ops.add(t847, t848)
        del t847, t848

        # pd_op.conv2d: (-1x1536x7x7xf32) <- (-1x768x14x14xf32, 1536x768x2x2xf32)
        t850 = paddle._C_ops.conv2d(
            t849, t270, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t849, t270

        # pd_op.reshape: (1x1536x1x1xf32) <- (1536xf32, 4xi64)
        t851 = paddle._C_ops.reshape(t271, t300)
        del t271

        # pd_op.add: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, 1x1536x1x1xf32)
        t852 = paddle._C_ops.add(t850, t851)
        del t850, t851

        # pd_op.depthwise_conv2d: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, 1536x1x7x7xf32)
        t853 = paddle._C_ops.depthwise_conv2d(
            t852, t272, [1, 1], [3, 3], "EXPLICIT", 1536, [1, 1], "NCHW"
        )
        del t272

        # pd_op.reshape: (1x1536x1x1xf32) <- (1536xf32, 4xi64)
        t854 = paddle._C_ops.reshape(t273, t300)
        del t273

        # pd_op.add: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, 1x1536x1x1xf32)
        t855 = paddle._C_ops.add(t853, t854)
        del t853, t854

        # pd_op.transpose: (-1x7x7x1536xf32) <- (-1x1536x7x7xf32)
        t856 = paddle._C_ops.transpose(t855, [0, 2, 3, 1])
        del t855

        # pd_op.layer_norm: (-1x7x7x1536xf32, -1x7x7xf32, -1x7x7xf32) <- (-1x7x7x1536xf32, 1536xf32, 1536xf32)
        t857, t858, t859 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t856, t274, t275, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t275, t274, t856

        # pd_op.matmul: (-1x7x7x6144xf32) <- (-1x7x7x1536xf32, 1536x6144xf32)
        t860 = paddle._C_ops.matmul(t857, t276, False, False)
        del t857, t276

        # pd_op.add: (-1x7x7x6144xf32) <- (-1x7x7x6144xf32, 6144xf32)
        t861 = paddle._C_ops.add(t860, t277)
        del t860, t277

        # pd_op.gelu: (-1x7x7x6144xf32) <- (-1x7x7x6144xf32)
        t862 = paddle._C_ops.gelu(t861, False)
        del t861

        # pd_op.matmul: (-1x7x7x1536xf32) <- (-1x7x7x6144xf32, 6144x1536xf32)
        t863 = paddle._C_ops.matmul(t862, t278, False, False)
        del t862, t278

        # pd_op.add: (-1x7x7x1536xf32) <- (-1x7x7x1536xf32, 1536xf32)
        t864 = paddle._C_ops.add(t863, t279)
        del t863, t279

        # pd_op.multiply: (-1x7x7x1536xf32) <- (1536xf32, -1x7x7x1536xf32)
        t865 = paddle._C_ops.multiply(input42, t864)
        del t864, input42

        # pd_op.transpose: (-1x1536x7x7xf32) <- (-1x7x7x1536xf32)
        t866 = paddle._C_ops.transpose(t865, [0, 3, 1, 2])
        del t865

        # pd_op.add: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, -1x1536x7x7xf32)
        t867 = paddle._C_ops.add(t852, t866)
        del t852, t866

        # pd_op.depthwise_conv2d: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, 1536x1x7x7xf32)
        t868 = paddle._C_ops.depthwise_conv2d(
            t867, t280, [1, 1], [3, 3], "EXPLICIT", 1536, [1, 1], "NCHW"
        )
        del t280

        # pd_op.reshape: (1x1536x1x1xf32) <- (1536xf32, 4xi64)
        t869 = paddle._C_ops.reshape(t281, t300)
        del t281

        # pd_op.add: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, 1x1536x1x1xf32)
        t870 = paddle._C_ops.add(t868, t869)
        del t868, t869

        # pd_op.transpose: (-1x7x7x1536xf32) <- (-1x1536x7x7xf32)
        t871 = paddle._C_ops.transpose(t870, [0, 2, 3, 1])
        del t870

        # pd_op.layer_norm: (-1x7x7x1536xf32, -1x7x7xf32, -1x7x7xf32) <- (-1x7x7x1536xf32, 1536xf32, 1536xf32)
        t872, t873, t874 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t871, t282, t283, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t283, t282, t871

        # pd_op.matmul: (-1x7x7x6144xf32) <- (-1x7x7x1536xf32, 1536x6144xf32)
        t875 = paddle._C_ops.matmul(t872, t284, False, False)
        del t872, t284

        # pd_op.add: (-1x7x7x6144xf32) <- (-1x7x7x6144xf32, 6144xf32)
        t876 = paddle._C_ops.add(t875, t285)
        del t875, t285

        # pd_op.gelu: (-1x7x7x6144xf32) <- (-1x7x7x6144xf32)
        t877 = paddle._C_ops.gelu(t876, False)
        del t876

        # pd_op.matmul: (-1x7x7x1536xf32) <- (-1x7x7x6144xf32, 6144x1536xf32)
        t878 = paddle._C_ops.matmul(t877, t286, False, False)
        del t877, t286

        # pd_op.add: (-1x7x7x1536xf32) <- (-1x7x7x1536xf32, 1536xf32)
        t879 = paddle._C_ops.add(t878, t287)
        del t878, t287

        # pd_op.multiply: (-1x7x7x1536xf32) <- (1536xf32, -1x7x7x1536xf32)
        t880 = paddle._C_ops.multiply(input43, t879)
        del t879, input43

        # pd_op.transpose: (-1x1536x7x7xf32) <- (-1x7x7x1536xf32)
        t881 = paddle._C_ops.transpose(t880, [0, 3, 1, 2])
        del t880

        # pd_op.add: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, -1x1536x7x7xf32)
        t882 = paddle._C_ops.add(t867, t881)
        del t867, t881

        # pd_op.depthwise_conv2d: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, 1536x1x7x7xf32)
        t883 = paddle._C_ops.depthwise_conv2d(
            t882, t288, [1, 1], [3, 3], "EXPLICIT", 1536, [1, 1], "NCHW"
        )
        del t288

        # pd_op.reshape: (1x1536x1x1xf32) <- (1536xf32, 4xi64)
        t884 = paddle._C_ops.reshape(t289, t300)
        del t300, t289

        # pd_op.add: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, 1x1536x1x1xf32)
        t885 = paddle._C_ops.add(t883, t884)
        del t883, t884

        # pd_op.transpose: (-1x7x7x1536xf32) <- (-1x1536x7x7xf32)
        t886 = paddle._C_ops.transpose(t885, [0, 2, 3, 1])
        del t885

        # pd_op.layer_norm: (-1x7x7x1536xf32, -1x7x7xf32, -1x7x7xf32) <- (-1x7x7x1536xf32, 1536xf32, 1536xf32)
        t887, t888, t889 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t886, t290, t291, float("1e-06"), 3),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t291, t290, t886

        # pd_op.matmul: (-1x7x7x6144xf32) <- (-1x7x7x1536xf32, 1536x6144xf32)
        t890 = paddle._C_ops.matmul(t887, t292, False, False)
        del t887, t292

        # pd_op.add: (-1x7x7x6144xf32) <- (-1x7x7x6144xf32, 6144xf32)
        t891 = paddle._C_ops.add(t890, t293)
        del t890, t293

        # pd_op.gelu: (-1x7x7x6144xf32) <- (-1x7x7x6144xf32)
        t892 = paddle._C_ops.gelu(t891, False)
        del t891

        # pd_op.matmul: (-1x7x7x1536xf32) <- (-1x7x7x6144xf32, 6144x1536xf32)
        t893 = paddle._C_ops.matmul(t892, t294, False, False)
        del t892, t294

        # pd_op.add: (-1x7x7x1536xf32) <- (-1x7x7x1536xf32, 1536xf32)
        t894 = paddle._C_ops.add(t893, t295)
        del t893, t295

        # pd_op.multiply: (-1x7x7x1536xf32) <- (1536xf32, -1x7x7x1536xf32)
        t895 = paddle._C_ops.multiply(input44, t894)
        del t894, input44

        # pd_op.transpose: (-1x1536x7x7xf32) <- (-1x7x7x1536xf32)
        t896 = paddle._C_ops.transpose(t895, [0, 3, 1, 2])
        del t895

        # pd_op.add: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32, -1x1536x7x7xf32)
        t897 = paddle._C_ops.add(t882, t896)
        del t882, t896

        # pd_op.mean: (-1x1536xf32) <- (-1x1536x7x7xf32)
        t898 = paddle._C_ops.mean(t897, [-2, -1], False)
        del t897

        # pd_op.layer_norm: (-1x1536xf32, -1xf32, -1xf32) <- (-1x1536xf32, 1536xf32, 1536xf32)
        t899, t900, t901 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t898, t296, t297, float("1e-06"), 1),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t898, t297, t296

        # pd_op.matmul: (-1x102xf32) <- (-1x1536xf32, 1536x102xf32)
        t902 = paddle._C_ops.matmul(t899, t298, False, False)
        del t899, t298

        return t902
