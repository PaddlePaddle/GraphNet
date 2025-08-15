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
        input1: paddle.Tensor,
        input2: paddle.Tensor,
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
        input3: paddle.Tensor,
        input4: paddle.Tensor,
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
        input5: paddle.Tensor,
        input6: paddle.Tensor,
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
        input7: paddle.Tensor,
        input8: paddle.Tensor,
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
        input9: paddle.Tensor,
        input10: paddle.Tensor,
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
        input11: paddle.Tensor,
        input12: paddle.Tensor,
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
        input13: paddle.Tensor,
        input14: paddle.Tensor,
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
        input15: paddle.Tensor,
        input16: paddle.Tensor,
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
        input17: paddle.Tensor,
        input18: paddle.Tensor,
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
        input19: paddle.Tensor,
        input20: paddle.Tensor,
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
        input21: paddle.Tensor,
        input22: paddle.Tensor,
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
        input23: paddle.Tensor,
        input24: paddle.Tensor,
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
        input25: paddle.Tensor,
        input26: paddle.Tensor,
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
        input27: paddle.Tensor,
        input28: paddle.Tensor,
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
        input29: paddle.Tensor,
        input30: paddle.Tensor,
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
        input31: paddle.Tensor,
        input32: paddle.Tensor,
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
        input33: paddle.Tensor,
        input34: paddle.Tensor,
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
        input35: paddle.Tensor,
        input36: paddle.Tensor,
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
        input37: paddle.Tensor,
        input38: paddle.Tensor,
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
        input39: paddle.Tensor,
        input40: paddle.Tensor,
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
        input41: paddle.Tensor,
        input42: paddle.Tensor,
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
        input43: paddle.Tensor,
        input44: paddle.Tensor,
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
        input45: paddle.Tensor,
        input46: paddle.Tensor,
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
        input47: paddle.Tensor,
        input48: paddle.Tensor,
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
    ):
        # pd_op.shape64: (4xi64) <- (-1x3x384x384xf32)
        t304 = paddle._C_ops.shape64(input0)

        # pd_op.full_int_array: (1xi64) <- ()
        t305 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        t306 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t307 = paddle._C_ops.slice(t304, [0], t305, t306, [1], [0])
        del t304

        # pd_op.conv2d: (-1x192x96x96xf32) <- (-1x3x384x384xf32, 192x3x4x4xf32)
        t308 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t309 = [1, -1, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t310 = paddle._C_ops.reshape(t1, t309)
        del t309, t1

        # pd_op.add: (-1x192x96x96xf32) <- (-1x192x96x96xf32, 1x192x1x1xf32)
        t311 = paddle._C_ops.add(t308, t310)
        del t308, t310

        # pd_op.shape64: (4xi64) <- (-1x192x96x96xf32)
        t312 = paddle._C_ops.shape64(t311)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t313 = paddle._C_ops.slice(t312, [0], t305, t306, [1], [0])
        del t312

        # pd_op.flatten: (-1x192x9216xf32) <- (-1x192x96x96xf32)
        t314 = paddle._C_ops.flatten(t311, 2, 3)
        del t311

        # pd_op.transpose: (-1x9216x192xf32) <- (-1x192x9216xf32)
        t315 = paddle._C_ops.transpose(t314, [0, 2, 1])
        del t314

        # pd_op.layer_norm: (-1x9216x192xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x192xf32, 192xf32, 192xf32)
        t316, t317, t318 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t315, t2, t3, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t3, t2, t315

        # pd_op.shape64: (3xi64) <- (-1x9216x192xf32)
        t319 = paddle._C_ops.shape64(t316)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t320 = paddle._C_ops.slice(t319, [0], t305, t306, [1], [0])
        del t319

        # pd_op.layer_norm: (-1x9216x192xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x192xf32, 192xf32, 192xf32)
        t321, t322, t323 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t316, t4, t5, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t5, t4

        # pd_op.full: (xi64) <- ()
        t324 = paddle._C_ops.full([], float("96"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t325 = paddle._C_ops.full(
            [], float("192"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t326 = [t320, t324, t324, t325]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t327 = paddle._C_ops.stack(t326, 0)
        del t326

        # pd_op.reshape: (-1x96x96x192xf32) <- (-1x9216x192xf32, 4xi64)
        t328 = paddle._C_ops.reshape(t321, t327)
        del t321, t327

        # pd_op.shape64: (4xi64) <- (-1x96x96x192xf32)
        t329 = paddle._C_ops.shape64(t328)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t330 = paddle._C_ops.slice(t329, [0], t305, t306, [1], [0])
        del t329

        # pd_op.full: (xi64) <- ()
        t331 = paddle._C_ops.full([], float("8"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t332 = paddle._C_ops.full([], float("12"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t333 = [t330, t331, t332, t331, t332, t325]
        del t330

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t334 = paddle._C_ops.stack(t333, 0)
        del t333

        # pd_op.reshape: (-1x8x12x8x12x192xf32) <- (-1x96x96x192xf32, 6xi64)
        t335 = paddle._C_ops.reshape(t328, t334)
        del t328, t334

        # pd_op.transpose: (-1x8x8x12x12x192xf32) <- (-1x8x12x8x12x192xf32)
        t336 = paddle._C_ops.transpose(t335, [0, 1, 3, 2, 4, 5])
        del t335

        # pd_op.full_int_array: (4xi64) <- ()
        t337 = [-1, 12, 12, 192]

        # pd_op.reshape: (-1x12x12x192xf32) <- (-1x8x8x12x12x192xf32, 4xi64)
        t338 = paddle._C_ops.reshape(t336, t337)
        del t336

        # pd_op.full_int_array: (3xi64) <- ()
        t339 = [-1, 144, 192]

        # pd_op.reshape: (-1x144x192xf32) <- (-1x12x12x192xf32, 3xi64)
        t340 = paddle._C_ops.reshape(t338, t339)
        del t338

        # pd_op.shape64: (3xi64) <- (-1x144x192xf32)
        t341 = paddle._C_ops.shape64(t340)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t342 = paddle._C_ops.slice(t341, [0], t305, t306, [1], [0])
        del t341

        # pd_op.matmul: (-1x144x576xf32) <- (-1x144x192xf32, 192x576xf32)
        t343 = paddle._C_ops.matmul(t340, t6, False, False)
        del t6, t340

        # pd_op.add: (-1x144x576xf32) <- (-1x144x576xf32, 576xf32)
        t344 = paddle._C_ops.add(t343, t7)
        del t343, t7

        # pd_op.full: (xi64) <- ()
        t345 = paddle._C_ops.full(
            [], float("144"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        t346 = paddle._C_ops.full([], float("3"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t347 = paddle._C_ops.full([], float("6"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t348 = paddle._C_ops.full([], float("32"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t349 = [t342, t345, t346, t347, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t350 = paddle._C_ops.stack(t349, 0)
        del t349

        # pd_op.reshape: (-1x144x3x6x32xf32) <- (-1x144x576xf32, 5xi64)
        t351 = paddle._C_ops.reshape(t344, t350)
        del t344, t350

        # pd_op.transpose: (3x-1x6x144x32xf32) <- (-1x144x3x6x32xf32)
        t352 = paddle._C_ops.transpose(t351, [2, 0, 3, 1, 4])
        del t351

        # pd_op.slice: (-1x6x144x32xf32) <- (3x-1x6x144x32xf32, 1xi64, 1xi64)
        t353 = paddle._C_ops.slice(t352, [0], t305, t306, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t354 = [2]

        # pd_op.slice: (-1x6x144x32xf32) <- (3x-1x6x144x32xf32, 1xi64, 1xi64)
        t355 = paddle._C_ops.slice(t352, [0], t306, t354, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t356 = [3]

        # pd_op.slice: (-1x6x144x32xf32) <- (3x-1x6x144x32xf32, 1xi64, 1xi64)
        t357 = paddle._C_ops.slice(t352, [0], t354, t356, [1], [0])
        del t352

        # pd_op.full: (1xf32) <- ()
        t358 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x6x144x32xf32) <- (-1x6x144x32xf32, 1xf32)
        t359 = paddle._C_ops.scale(t353, t358, float("0"), True)
        del t353

        # pd_op.transpose: (-1x6x32x144xf32) <- (-1x6x144x32xf32)
        t360 = paddle._C_ops.transpose(t355, [0, 1, 3, 2])
        del t355

        # pd_op.matmul: (-1x6x144x144xf32) <- (-1x6x144x32xf32, -1x6x32x144xf32)
        t361 = paddle._C_ops.matmul(t359, t360, False, False)
        del t359, t360

        # pd_op.full_int_array: (1xi64) <- ()
        t362 = [-1]

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t363 = paddle._C_ops.reshape(input1, t362)
        del input1

        # pd_op.index_select: (20736x6xf32) <- (529x6xf32, 20736xi64)
        t364 = paddle._C_ops.index_select(input2, t363, 0)
        del input2, t363

        # pd_op.full_int_array: (3xi64) <- ()
        t365 = [144, 144, -1]

        # pd_op.reshape: (144x144x6xf32) <- (20736x6xf32, 3xi64)
        t366 = paddle._C_ops.reshape(t364, t365)
        del t364

        # pd_op.transpose: (6x144x144xf32) <- (144x144x6xf32)
        t367 = paddle._C_ops.transpose(t366, [2, 0, 1])
        del t366

        # pd_op.unsqueeze: (1x6x144x144xf32) <- (6x144x144xf32, 1xi64)
        t368 = paddle._C_ops.unsqueeze(t367, t305)
        del t367

        # pd_op.add: (-1x6x144x144xf32) <- (-1x6x144x144xf32, 1x6x144x144xf32)
        t369 = paddle._C_ops.add(t361, t368)
        del t361, t368

        # pd_op.softmax: (-1x6x144x144xf32) <- (-1x6x144x144xf32)
        t370 = paddle._C_ops.softmax(t369, -1)
        del t369

        # pd_op.matmul: (-1x6x144x32xf32) <- (-1x6x144x144xf32, -1x6x144x32xf32)
        t371 = paddle._C_ops.matmul(t370, t357, False, False)
        del t357, t370

        # pd_op.transpose: (-1x144x6x32xf32) <- (-1x6x144x32xf32)
        t372 = paddle._C_ops.transpose(t371, [0, 2, 1, 3])
        del t371

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t373 = [t342, t345, t325]
        del t342

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t374 = paddle._C_ops.stack(t373, 0)
        del t373

        # pd_op.reshape: (-1x144x192xf32) <- (-1x144x6x32xf32, 3xi64)
        t375 = paddle._C_ops.reshape(t372, t374)
        del t374, t372

        # pd_op.matmul: (-1x144x192xf32) <- (-1x144x192xf32, 192x192xf32)
        t376 = paddle._C_ops.matmul(t375, t8, False, False)
        del t8, t375

        # pd_op.add: (-1x144x192xf32) <- (-1x144x192xf32, 192xf32)
        t377 = paddle._C_ops.add(t376, t9)
        del t376, t9

        # pd_op.reshape: (-1x12x12x192xf32) <- (-1x144x192xf32, 4xi64)
        t378 = paddle._C_ops.reshape(t377, t337)
        del t377

        # pd_op.full_int_array: (6xi64) <- ()
        t379 = [-1, 8, 8, 12, 12, 192]

        # pd_op.reshape: (-1x8x8x12x12x192xf32) <- (-1x12x12x192xf32, 6xi64)
        t380 = paddle._C_ops.reshape(t378, t379)
        del t378

        # pd_op.transpose: (-1x8x12x8x12x192xf32) <- (-1x8x8x12x12x192xf32)
        t381 = paddle._C_ops.transpose(t380, [0, 1, 3, 2, 4, 5])
        del t380

        # pd_op.full_int_array: (4xi64) <- ()
        t382 = [-1, 96, 96, 192]

        # pd_op.reshape: (-1x96x96x192xf32) <- (-1x8x12x8x12x192xf32, 4xi64)
        t383 = paddle._C_ops.reshape(t381, t382)
        del t381

        # pd_op.full: (xi64) <- ()
        t384 = paddle._C_ops.full(
            [], float("9216"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t385 = [t320, t384, t325]
        del t320

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t386 = paddle._C_ops.stack(t385, 0)
        del t385

        # pd_op.reshape: (-1x9216x192xf32) <- (-1x96x96x192xf32, 3xi64)
        t387 = paddle._C_ops.reshape(t383, t386)
        del t383, t386

        # pd_op.add: (-1x9216x192xf32) <- (-1x9216x192xf32, -1x9216x192xf32)
        t388 = paddle._C_ops.add(t316, t387)
        del t316, t387

        # pd_op.layer_norm: (-1x9216x192xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x192xf32, 192xf32, 192xf32)
        t389, t390, t391 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t388, t10, t11, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t11, t10

        # pd_op.matmul: (-1x9216x768xf32) <- (-1x9216x192xf32, 192x768xf32)
        t392 = paddle._C_ops.matmul(t389, t12, False, False)
        del t389, t12

        # pd_op.add: (-1x9216x768xf32) <- (-1x9216x768xf32, 768xf32)
        t393 = paddle._C_ops.add(t392, t13)
        del t392, t13

        # pd_op.gelu: (-1x9216x768xf32) <- (-1x9216x768xf32)
        t394 = paddle._C_ops.gelu(t393, False)
        del t393

        # pd_op.matmul: (-1x9216x192xf32) <- (-1x9216x768xf32, 768x192xf32)
        t395 = paddle._C_ops.matmul(t394, t14, False, False)
        del t394, t14

        # pd_op.add: (-1x9216x192xf32) <- (-1x9216x192xf32, 192xf32)
        t396 = paddle._C_ops.add(t395, t15)
        del t395, t15

        # pd_op.add: (-1x9216x192xf32) <- (-1x9216x192xf32, -1x9216x192xf32)
        t397 = paddle._C_ops.add(t388, t396)
        del t388, t396

        # pd_op.shape64: (3xi64) <- (-1x9216x192xf32)
        t398 = paddle._C_ops.shape64(t397)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t399 = paddle._C_ops.slice(t398, [0], t305, t306, [1], [0])
        del t398

        # pd_op.layer_norm: (-1x9216x192xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x192xf32, 192xf32, 192xf32)
        t400, t401, t402 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t397, t16, t17, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t17, t16

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t403 = [t399, t324, t324, t325]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t404 = paddle._C_ops.stack(t403, 0)
        del t403

        # pd_op.reshape: (-1x96x96x192xf32) <- (-1x9216x192xf32, 4xi64)
        t405 = paddle._C_ops.reshape(t400, t404)
        del t400, t404

        # pd_op.shape64: (4xi64) <- (-1x96x96x192xf32)
        t406 = paddle._C_ops.shape64(t405)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t407 = paddle._C_ops.slice(t406, [0], t305, t306, [1], [0])
        del t406

        # pd_op.full_int_array: (2xi64) <- ()
        t408 = [-6, -6]

        # pd_op.roll: (-1x96x96x192xf32) <- (-1x96x96x192xf32, 2xi64)
        t409 = paddle._C_ops.roll(t405, t408, [1, 2])
        del t405

        # pd_op.shape64: (4xi64) <- (-1x96x96x192xf32)
        t410 = paddle._C_ops.shape64(t409)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t411 = paddle._C_ops.slice(t410, [0], t305, t306, [1], [0])
        del t410

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t412 = [t411, t331, t332, t331, t332, t325]
        del t331, t411

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t413 = paddle._C_ops.stack(t412, 0)
        del t412

        # pd_op.reshape: (-1x8x12x8x12x192xf32) <- (-1x96x96x192xf32, 6xi64)
        t414 = paddle._C_ops.reshape(t409, t413)
        del t409, t413

        # pd_op.transpose: (-1x8x8x12x12x192xf32) <- (-1x8x12x8x12x192xf32)
        t415 = paddle._C_ops.transpose(t414, [0, 1, 3, 2, 4, 5])
        del t414

        # pd_op.reshape: (-1x12x12x192xf32) <- (-1x8x8x12x12x192xf32, 4xi64)
        t416 = paddle._C_ops.reshape(t415, t337)
        del t415

        # pd_op.reshape: (-1x144x192xf32) <- (-1x12x12x192xf32, 3xi64)
        t417 = paddle._C_ops.reshape(t416, t339)
        del t339, t416

        # pd_op.full: (1x96x96x1xf32) <- ()
        t418 = paddle._C_ops.full(
            [1, 96, 96, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (2xi64) <- ()
        t419 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        t420 = [-12, -12]

        # pd_op.full_int_array: (2xi64) <- ()
        t421 = [1, 1]

        # pd_op.set_value_: (1x96x96x1xf32) <- (1x96x96x1xf32, 2xi64, 2xi64, 2xi64)
        t422 = paddle._C_ops.set_value_(
            t418, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t418

        # pd_op.full_int_array: (2xi64) <- ()
        t423 = [0, -12]

        # pd_op.full_int_array: (2xi64) <- ()
        t424 = [-12, -6]

        # pd_op.set_value_: (1x96x96x1xf32) <- (1x96x96x1xf32, 2xi64, 2xi64, 2xi64)
        t425 = paddle._C_ops.set_value_(
            t422, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t422

        # pd_op.full_int_array: (2xi64) <- ()
        t426 = [0, -6]

        # pd_op.full_int_array: (2xi64) <- ()
        t427 = [-12, 2147483647]

        # pd_op.set_value_: (1x96x96x1xf32) <- (1x96x96x1xf32, 2xi64, 2xi64, 2xi64)
        t428 = paddle._C_ops.set_value_(
            t425, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t425

        # pd_op.full_int_array: (2xi64) <- ()
        t429 = [-12, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        t430 = [-6, -12]

        # pd_op.set_value_: (1x96x96x1xf32) <- (1x96x96x1xf32, 2xi64, 2xi64, 2xi64)
        t431 = paddle._C_ops.set_value_(
            t428, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t428

        # pd_op.set_value_: (1x96x96x1xf32) <- (1x96x96x1xf32, 2xi64, 2xi64, 2xi64)
        t432 = paddle._C_ops.set_value_(
            t431, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t431

        # pd_op.full_int_array: (2xi64) <- ()
        t433 = [-6, 2147483647]

        # pd_op.set_value_: (1x96x96x1xf32) <- (1x96x96x1xf32, 2xi64, 2xi64, 2xi64)
        t434 = paddle._C_ops.set_value_(
            t432, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t432

        # pd_op.full_int_array: (2xi64) <- ()
        t435 = [-6, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        t436 = [2147483647, -12]

        # pd_op.set_value_: (1x96x96x1xf32) <- (1x96x96x1xf32, 2xi64, 2xi64, 2xi64)
        t437 = paddle._C_ops.set_value_(
            t434, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t434

        # pd_op.full_int_array: (2xi64) <- ()
        t438 = [2147483647, -6]

        # pd_op.set_value_: (1x96x96x1xf32) <- (1x96x96x1xf32, 2xi64, 2xi64, 2xi64)
        t439 = paddle._C_ops.set_value_(
            t437, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t437

        # pd_op.full_int_array: (2xi64) <- ()
        t440 = [2147483647, 2147483647]

        # pd_op.set_value_: (1x96x96x1xf32) <- (1x96x96x1xf32, 2xi64, 2xi64, 2xi64)
        t441 = paddle._C_ops.set_value_(
            t439, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t439

        # pd_op.full_int_array: (6xi64) <- ()
        t442 = [1, 8, 12, 8, 12, 1]

        # pd_op.reshape: (1x8x12x8x12x1xf32) <- (1x96x96x1xf32, 6xi64)
        t443 = paddle._C_ops.reshape(t441, t442)
        del t442

        # pd_op.transpose: (1x8x8x12x12x1xf32) <- (1x8x12x8x12x1xf32)
        t444 = paddle._C_ops.transpose(t443, [0, 1, 3, 2, 4, 5])
        del t443

        # pd_op.full_int_array: (4xi64) <- ()
        t445 = [-1, 12, 12, 1]

        # pd_op.reshape: (64x12x12x1xf32) <- (1x8x8x12x12x1xf32, 4xi64)
        t446 = paddle._C_ops.reshape(t444, t445)
        del t444

        # pd_op.full_int_array: (2xi64) <- ()
        t447 = [-1, 144]

        # pd_op.reshape: (64x144xf32) <- (64x12x12x1xf32, 2xi64)
        t448 = paddle._C_ops.reshape(t446, t447)
        del t446

        # pd_op.unsqueeze: (64x1x144xf32) <- (64x144xf32, 1xi64)
        t449 = paddle._C_ops.unsqueeze(t448, t306)

        # pd_op.unsqueeze: (64x144x1xf32) <- (64x144xf32, 1xi64)
        t450 = paddle._C_ops.unsqueeze(t448, t354)
        del t448

        # pd_op.subtract: (64x144x144xf32) <- (64x1x144xf32, 64x144x1xf32)
        t451 = paddle._C_ops.subtract(t449, t450)
        del t449, t450

        # pd_op.full: (xf32) <- ()
        t452 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (64x144x144xb) <- (64x144x144xf32, xf32)
        t453 = paddle._C_ops.not_equal(t451, t452)

        # pd_op.full: (64x144x144xf32) <- ()
        t454 = paddle._C_ops.full(
            [64, 144, 144],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (64x144x144xf32) <- (64x144x144xb, 64x144x144xf32, 64x144x144xf32)
        t455 = paddle._C_ops.where(t453, t454, t451)
        del t454, t453, t451

        # pd_op.equal: (64x144x144xb) <- (64x144x144xf32, xf32)
        t456 = paddle._C_ops.equal(t455, t452)

        # pd_op.full: (64x144x144xf32) <- ()
        t457 = paddle._C_ops.full(
            [64, 144, 144],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (64x144x144xf32) <- (64x144x144xb, 64x144x144xf32, 64x144x144xf32)
        t458 = paddle._C_ops.where(t456, t457, t455)
        del t456, t457, t455

        # pd_op.shape64: (3xi64) <- (-1x144x192xf32)
        t459 = paddle._C_ops.shape64(t417)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t460 = paddle._C_ops.slice(t459, [0], t305, t306, [1], [0])
        del t459

        # pd_op.matmul: (-1x144x576xf32) <- (-1x144x192xf32, 192x576xf32)
        t461 = paddle._C_ops.matmul(t417, t18, False, False)
        del t18, t417

        # pd_op.add: (-1x144x576xf32) <- (-1x144x576xf32, 576xf32)
        t462 = paddle._C_ops.add(t461, t19)
        del t461, t19

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t463 = [t460, t345, t346, t347, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t464 = paddle._C_ops.stack(t463, 0)
        del t463

        # pd_op.reshape: (-1x144x3x6x32xf32) <- (-1x144x576xf32, 5xi64)
        t465 = paddle._C_ops.reshape(t462, t464)
        del t462, t464

        # pd_op.transpose: (3x-1x6x144x32xf32) <- (-1x144x3x6x32xf32)
        t466 = paddle._C_ops.transpose(t465, [2, 0, 3, 1, 4])
        del t465

        # pd_op.slice: (-1x6x144x32xf32) <- (3x-1x6x144x32xf32, 1xi64, 1xi64)
        t467 = paddle._C_ops.slice(t466, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x6x144x32xf32) <- (3x-1x6x144x32xf32, 1xi64, 1xi64)
        t468 = paddle._C_ops.slice(t466, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x6x144x32xf32) <- (3x-1x6x144x32xf32, 1xi64, 1xi64)
        t469 = paddle._C_ops.slice(t466, [0], t354, t356, [1], [0])
        del t466

        # pd_op.scale: (-1x6x144x32xf32) <- (-1x6x144x32xf32, 1xf32)
        t470 = paddle._C_ops.scale(t467, t358, float("0"), True)
        del t467

        # pd_op.transpose: (-1x6x32x144xf32) <- (-1x6x144x32xf32)
        t471 = paddle._C_ops.transpose(t468, [0, 1, 3, 2])
        del t468

        # pd_op.matmul: (-1x6x144x144xf32) <- (-1x6x144x32xf32, -1x6x32x144xf32)
        t472 = paddle._C_ops.matmul(t470, t471, False, False)
        del t470, t471

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t473 = paddle._C_ops.reshape(input3, t362)
        del input3

        # pd_op.index_select: (20736x6xf32) <- (529x6xf32, 20736xi64)
        t474 = paddle._C_ops.index_select(input4, t473, 0)
        del input4, t473

        # pd_op.reshape: (144x144x6xf32) <- (20736x6xf32, 3xi64)
        t475 = paddle._C_ops.reshape(t474, t365)
        del t474

        # pd_op.transpose: (6x144x144xf32) <- (144x144x6xf32)
        t476 = paddle._C_ops.transpose(t475, [2, 0, 1])
        del t475

        # pd_op.unsqueeze: (1x6x144x144xf32) <- (6x144x144xf32, 1xi64)
        t477 = paddle._C_ops.unsqueeze(t476, t305)
        del t476

        # pd_op.add: (-1x6x144x144xf32) <- (-1x6x144x144xf32, 1x6x144x144xf32)
        t478 = paddle._C_ops.add(t472, t477)
        del t472, t477

        # pd_op.full: (xi64) <- ()
        t479 = paddle._C_ops.full(
            [], float("64"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t480 = paddle._C_ops.floor_divide(t460, t479)
        del t479

        # pd_op.full: (xi64) <- ()
        t481 = paddle._C_ops.full([], float("64"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t482 = [t480, t481, t347, t345, t345]
        del t480, t481

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t483 = paddle._C_ops.stack(t482, 0)
        del t482

        # pd_op.reshape: (-1x64x6x144x144xf32) <- (-1x6x144x144xf32, 5xi64)
        t484 = paddle._C_ops.reshape(t478, t483)
        del t478, t483

        # pd_op.unsqueeze: (64x1x144x144xf32) <- (64x144x144xf32, 1xi64)
        t485 = paddle._C_ops.unsqueeze(t458, t306)
        del t458

        # pd_op.unsqueeze: (1x64x1x144x144xf32) <- (64x1x144x144xf32, 1xi64)
        t486 = paddle._C_ops.unsqueeze(t485, t305)
        del t485

        # pd_op.add: (-1x64x6x144x144xf32) <- (-1x64x6x144x144xf32, 1x64x1x144x144xf32)
        t487 = paddle._C_ops.add(t484, t486)
        del t484, t486

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t488 = [t460, t347, t345, t345]
        del t347

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t489 = paddle._C_ops.stack(t488, 0)
        del t488

        # pd_op.reshape: (-1x6x144x144xf32) <- (-1x64x6x144x144xf32, 4xi64)
        t490 = paddle._C_ops.reshape(t487, t489)
        del t487, t489

        # pd_op.softmax: (-1x6x144x144xf32) <- (-1x6x144x144xf32)
        t491 = paddle._C_ops.softmax(t490, -1)
        del t490

        # pd_op.matmul: (-1x6x144x32xf32) <- (-1x6x144x144xf32, -1x6x144x32xf32)
        t492 = paddle._C_ops.matmul(t491, t469, False, False)
        del t469, t491

        # pd_op.transpose: (-1x144x6x32xf32) <- (-1x6x144x32xf32)
        t493 = paddle._C_ops.transpose(t492, [0, 2, 1, 3])
        del t492

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t494 = [t460, t345, t325]
        del t460

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t495 = paddle._C_ops.stack(t494, 0)
        del t494

        # pd_op.reshape: (-1x144x192xf32) <- (-1x144x6x32xf32, 3xi64)
        t496 = paddle._C_ops.reshape(t493, t495)
        del t495, t493

        # pd_op.matmul: (-1x144x192xf32) <- (-1x144x192xf32, 192x192xf32)
        t497 = paddle._C_ops.matmul(t496, t20, False, False)
        del t20, t496

        # pd_op.add: (-1x144x192xf32) <- (-1x144x192xf32, 192xf32)
        t498 = paddle._C_ops.add(t497, t21)
        del t497, t21

        # pd_op.reshape: (-1x12x12x192xf32) <- (-1x144x192xf32, 4xi64)
        t499 = paddle._C_ops.reshape(t498, t337)
        del t498, t337

        # pd_op.reshape: (-1x8x8x12x12x192xf32) <- (-1x12x12x192xf32, 6xi64)
        t500 = paddle._C_ops.reshape(t499, t379)
        del t379, t499

        # pd_op.transpose: (-1x8x12x8x12x192xf32) <- (-1x8x8x12x12x192xf32)
        t501 = paddle._C_ops.transpose(t500, [0, 1, 3, 2, 4, 5])
        del t500

        # pd_op.reshape: (-1x96x96x192xf32) <- (-1x8x12x8x12x192xf32, 4xi64)
        t502 = paddle._C_ops.reshape(t501, t382)
        del t382, t501

        # pd_op.full_int_array: (2xi64) <- ()
        t503 = [6, 6]

        # pd_op.roll: (-1x96x96x192xf32) <- (-1x96x96x192xf32, 2xi64)
        t504 = paddle._C_ops.roll(t502, t503, [1, 2])
        del t502

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t505 = [t399, t384, t325]
        del t384, t399

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t506 = paddle._C_ops.stack(t505, 0)
        del t505

        # pd_op.reshape: (-1x9216x192xf32) <- (-1x96x96x192xf32, 3xi64)
        t507 = paddle._C_ops.reshape(t504, t506)
        del t504, t506

        # pd_op.add: (-1x9216x192xf32) <- (-1x9216x192xf32, -1x9216x192xf32)
        t508 = paddle._C_ops.add(t397, t507)
        del t397, t507

        # pd_op.layer_norm: (-1x9216x192xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x192xf32, 192xf32, 192xf32)
        t509, t510, t511 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t508, t22, t23, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t23, t22

        # pd_op.matmul: (-1x9216x768xf32) <- (-1x9216x192xf32, 192x768xf32)
        t512 = paddle._C_ops.matmul(t509, t24, False, False)
        del t509, t24

        # pd_op.add: (-1x9216x768xf32) <- (-1x9216x768xf32, 768xf32)
        t513 = paddle._C_ops.add(t512, t25)
        del t512, t25

        # pd_op.gelu: (-1x9216x768xf32) <- (-1x9216x768xf32)
        t514 = paddle._C_ops.gelu(t513, False)
        del t513

        # pd_op.matmul: (-1x9216x192xf32) <- (-1x9216x768xf32, 768x192xf32)
        t515 = paddle._C_ops.matmul(t514, t26, False, False)
        del t514, t26

        # pd_op.add: (-1x9216x192xf32) <- (-1x9216x192xf32, 192xf32)
        t516 = paddle._C_ops.add(t515, t27)
        del t515, t27

        # pd_op.add: (-1x9216x192xf32) <- (-1x9216x192xf32, -1x9216x192xf32)
        t517 = paddle._C_ops.add(t508, t516)
        del t508, t516

        # pd_op.shape64: (3xi64) <- (-1x9216x192xf32)
        t518 = paddle._C_ops.shape64(t517)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t519 = paddle._C_ops.slice(t518, [0], t305, t306, [1], [0])
        del t518

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t520 = [t519, t324, t324, t325]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t521 = paddle._C_ops.stack(t520, 0)
        del t520

        # pd_op.reshape: (-1x96x96x192xf32) <- (-1x9216x192xf32, 4xi64)
        t522 = paddle._C_ops.reshape(t517, t521)
        del t517, t521

        # pd_op.full_int_array: (2xi64) <- ()
        t523 = [2, 2]

        # pd_op.strided_slice: (-1x48x48x192xf32) <- (-1x96x96x192xf32, 2xi64, 2xi64, 2xi64)
        t524 = paddle._C_ops.strided_slice(t522, [1, 2], t419, t440, t523)

        # pd_op.full_int_array: (2xi64) <- ()
        t525 = [1, 0]

        # pd_op.strided_slice: (-1x48x48x192xf32) <- (-1x96x96x192xf32, 2xi64, 2xi64, 2xi64)
        t526 = paddle._C_ops.strided_slice(t522, [1, 2], t525, t440, t523)

        # pd_op.full_int_array: (2xi64) <- ()
        t527 = [0, 1]

        # pd_op.strided_slice: (-1x48x48x192xf32) <- (-1x96x96x192xf32, 2xi64, 2xi64, 2xi64)
        t528 = paddle._C_ops.strided_slice(t522, [1, 2], t527, t440, t523)

        # pd_op.strided_slice: (-1x48x48x192xf32) <- (-1x96x96x192xf32, 2xi64, 2xi64, 2xi64)
        t529 = paddle._C_ops.strided_slice(t522, [1, 2], t421, t440, t523)

        # pd_op.shape64: (4xi64) <- (-1x96x96x192xf32)
        t530 = paddle._C_ops.shape64(t522)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t531 = paddle._C_ops.slice(t530, [0], t305, t306, [1], [0])
        del t530

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t532 = [t531, t324, t324, t325]
        del t324, t325, t531

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t533 = paddle._C_ops.stack(t532, 0)
        del t532

        # pd_op.reshape: (-1x96x96x192xf32) <- (-1x96x96x192xf32, 4xi64)
        t534 = paddle._C_ops.reshape(t522, t533)
        del t522, t533

        # pd_op.full: (1xi32) <- ()
        t535 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x48x48x192xf32, -1x48x48x192xf32, -1x48x48x192xf32, -1x48x48x192xf32]) <- (-1x48x48x192xf32, -1x48x48x192xf32, -1x48x48x192xf32, -1x48x48x192xf32)
        t536 = [t524, t526, t528, t529]
        del t524, t526, t528, t529

        # pd_op.concat: (-1x48x48x768xf32) <- ([-1x48x48x192xf32, -1x48x48x192xf32, -1x48x48x192xf32, -1x48x48x192xf32], 1xi32)
        t537 = paddle._C_ops.concat(t536, t535)
        del t536

        # pd_op.full: (xi64) <- ()
        t538 = paddle._C_ops.full([], float("-1"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t539 = paddle._C_ops.full(
            [], float("768"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t540 = [t519, t538, t539]
        del t519

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t541 = paddle._C_ops.stack(t540, 0)
        del t540

        # pd_op.reshape: (-1x-1x768xf32) <- (-1x48x48x768xf32, 3xi64)
        t542 = paddle._C_ops.reshape(t537, t541)
        del t537, t541

        # pd_op.layer_norm: (-1x-1x768xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x768xf32, 768xf32, 768xf32)
        t543, t544, t545 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t542, t28, t29, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t29, t28, t542

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        t546 = paddle._C_ops.matmul(t543, t30, False, False)
        del t543, t30

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        t547 = paddle._C_ops.shape64(t546)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t548 = paddle._C_ops.slice(t547, [0], t305, t306, [1], [0])
        del t547

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        t549 = paddle._C_ops.shape64(t546)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t550 = paddle._C_ops.slice(t549, [0], t306, t354, [1], [0])
        del t549

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        t551, t552, t553 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t546, t31, t32, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t32, t31

        # pd_op.full: (xi64) <- ()
        t554 = paddle._C_ops.full([], float("48"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t555 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t556 = [t548, t554, t554, t555]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t557 = paddle._C_ops.stack(t556, 0)
        del t556

        # pd_op.reshape: (-1x48x48x384xf32) <- (-1x-1x384xf32, 4xi64)
        t558 = paddle._C_ops.reshape(t551, t557)
        del t551, t557

        # pd_op.shape64: (4xi64) <- (-1x48x48x384xf32)
        t559 = paddle._C_ops.shape64(t558)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t560 = paddle._C_ops.slice(t559, [0], t305, t306, [1], [0])
        del t559

        # pd_op.full: (xi64) <- ()
        t561 = paddle._C_ops.full([], float("4"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t562 = [t560, t561, t332, t561, t332, t555]
        del t560

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t563 = paddle._C_ops.stack(t562, 0)
        del t562

        # pd_op.reshape: (-1x4x12x4x12x384xf32) <- (-1x48x48x384xf32, 6xi64)
        t564 = paddle._C_ops.reshape(t558, t563)
        del t558, t563

        # pd_op.transpose: (-1x4x4x12x12x384xf32) <- (-1x4x12x4x12x384xf32)
        t565 = paddle._C_ops.transpose(t564, [0, 1, 3, 2, 4, 5])
        del t564

        # pd_op.full_int_array: (4xi64) <- ()
        t566 = [-1, 12, 12, 384]

        # pd_op.reshape: (-1x12x12x384xf32) <- (-1x4x4x12x12x384xf32, 4xi64)
        t567 = paddle._C_ops.reshape(t565, t566)
        del t565

        # pd_op.full_int_array: (3xi64) <- ()
        t568 = [-1, 144, 384]

        # pd_op.reshape: (-1x144x384xf32) <- (-1x12x12x384xf32, 3xi64)
        t569 = paddle._C_ops.reshape(t567, t568)
        del t567

        # pd_op.shape64: (3xi64) <- (-1x144x384xf32)
        t570 = paddle._C_ops.shape64(t569)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t571 = paddle._C_ops.slice(t570, [0], t305, t306, [1], [0])
        del t570

        # pd_op.matmul: (-1x144x1152xf32) <- (-1x144x384xf32, 384x1152xf32)
        t572 = paddle._C_ops.matmul(t569, t33, False, False)
        del t33, t569

        # pd_op.add: (-1x144x1152xf32) <- (-1x144x1152xf32, 1152xf32)
        t573 = paddle._C_ops.add(t572, t34)
        del t572, t34

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t574 = [t571, t345, t346, t332, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t575 = paddle._C_ops.stack(t574, 0)
        del t574

        # pd_op.reshape: (-1x144x3x12x32xf32) <- (-1x144x1152xf32, 5xi64)
        t576 = paddle._C_ops.reshape(t573, t575)
        del t573, t575

        # pd_op.transpose: (3x-1x12x144x32xf32) <- (-1x144x3x12x32xf32)
        t577 = paddle._C_ops.transpose(t576, [2, 0, 3, 1, 4])
        del t576

        # pd_op.slice: (-1x12x144x32xf32) <- (3x-1x12x144x32xf32, 1xi64, 1xi64)
        t578 = paddle._C_ops.slice(t577, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x12x144x32xf32) <- (3x-1x12x144x32xf32, 1xi64, 1xi64)
        t579 = paddle._C_ops.slice(t577, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x12x144x32xf32) <- (3x-1x12x144x32xf32, 1xi64, 1xi64)
        t580 = paddle._C_ops.slice(t577, [0], t354, t356, [1], [0])
        del t577

        # pd_op.scale: (-1x12x144x32xf32) <- (-1x12x144x32xf32, 1xf32)
        t581 = paddle._C_ops.scale(t578, t358, float("0"), True)
        del t578

        # pd_op.transpose: (-1x12x32x144xf32) <- (-1x12x144x32xf32)
        t582 = paddle._C_ops.transpose(t579, [0, 1, 3, 2])
        del t579

        # pd_op.matmul: (-1x12x144x144xf32) <- (-1x12x144x32xf32, -1x12x32x144xf32)
        t583 = paddle._C_ops.matmul(t581, t582, False, False)
        del t581, t582

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t584 = paddle._C_ops.reshape(input5, t362)
        del input5

        # pd_op.index_select: (20736x12xf32) <- (529x12xf32, 20736xi64)
        t585 = paddle._C_ops.index_select(input6, t584, 0)
        del input6, t584

        # pd_op.reshape: (144x144x12xf32) <- (20736x12xf32, 3xi64)
        t586 = paddle._C_ops.reshape(t585, t365)
        del t585

        # pd_op.transpose: (12x144x144xf32) <- (144x144x12xf32)
        t587 = paddle._C_ops.transpose(t586, [2, 0, 1])
        del t586

        # pd_op.unsqueeze: (1x12x144x144xf32) <- (12x144x144xf32, 1xi64)
        t588 = paddle._C_ops.unsqueeze(t587, t305)
        del t587

        # pd_op.add: (-1x12x144x144xf32) <- (-1x12x144x144xf32, 1x12x144x144xf32)
        t589 = paddle._C_ops.add(t583, t588)
        del t583, t588

        # pd_op.softmax: (-1x12x144x144xf32) <- (-1x12x144x144xf32)
        t590 = paddle._C_ops.softmax(t589, -1)
        del t589

        # pd_op.matmul: (-1x12x144x32xf32) <- (-1x12x144x144xf32, -1x12x144x32xf32)
        t591 = paddle._C_ops.matmul(t590, t580, False, False)
        del t580, t590

        # pd_op.transpose: (-1x144x12x32xf32) <- (-1x12x144x32xf32)
        t592 = paddle._C_ops.transpose(t591, [0, 2, 1, 3])
        del t591

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t593 = [t571, t345, t555]
        del t571

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t594 = paddle._C_ops.stack(t593, 0)
        del t593

        # pd_op.reshape: (-1x144x384xf32) <- (-1x144x12x32xf32, 3xi64)
        t595 = paddle._C_ops.reshape(t592, t594)
        del t594, t592

        # pd_op.matmul: (-1x144x384xf32) <- (-1x144x384xf32, 384x384xf32)
        t596 = paddle._C_ops.matmul(t595, t35, False, False)
        del t35, t595

        # pd_op.add: (-1x144x384xf32) <- (-1x144x384xf32, 384xf32)
        t597 = paddle._C_ops.add(t596, t36)
        del t596, t36

        # pd_op.reshape: (-1x12x12x384xf32) <- (-1x144x384xf32, 4xi64)
        t598 = paddle._C_ops.reshape(t597, t566)
        del t597

        # pd_op.full_int_array: (6xi64) <- ()
        t599 = [-1, 4, 4, 12, 12, 384]

        # pd_op.reshape: (-1x4x4x12x12x384xf32) <- (-1x12x12x384xf32, 6xi64)
        t600 = paddle._C_ops.reshape(t598, t599)
        del t598

        # pd_op.transpose: (-1x4x12x4x12x384xf32) <- (-1x4x4x12x12x384xf32)
        t601 = paddle._C_ops.transpose(t600, [0, 1, 3, 2, 4, 5])
        del t600

        # pd_op.full_int_array: (4xi64) <- ()
        t602 = [-1, 48, 48, 384]

        # pd_op.reshape: (-1x48x48x384xf32) <- (-1x4x12x4x12x384xf32, 4xi64)
        t603 = paddle._C_ops.reshape(t601, t602)
        del t601

        # pd_op.full: (xi64) <- ()
        t604 = paddle._C_ops.full(
            [], float("2304"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t605 = [t548, t604, t555]
        del t548

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t606 = paddle._C_ops.stack(t605, 0)
        del t605

        # pd_op.reshape: (-1x2304x384xf32) <- (-1x48x48x384xf32, 3xi64)
        t607 = paddle._C_ops.reshape(t603, t606)
        del t603, t606

        # pd_op.add: (-1x2304x384xf32) <- (-1x-1x384xf32, -1x2304x384xf32)
        t608 = paddle._C_ops.add(t546, t607)
        del t546, t607

        # pd_op.layer_norm: (-1x2304x384xf32, -1x2304xf32, -1x2304xf32) <- (-1x2304x384xf32, 384xf32, 384xf32)
        t609, t610, t611 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t608, t37, t38, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t38, t37

        # pd_op.matmul: (-1x2304x1536xf32) <- (-1x2304x384xf32, 384x1536xf32)
        t612 = paddle._C_ops.matmul(t609, t39, False, False)
        del t609, t39

        # pd_op.add: (-1x2304x1536xf32) <- (-1x2304x1536xf32, 1536xf32)
        t613 = paddle._C_ops.add(t612, t40)
        del t612, t40

        # pd_op.gelu: (-1x2304x1536xf32) <- (-1x2304x1536xf32)
        t614 = paddle._C_ops.gelu(t613, False)
        del t613

        # pd_op.matmul: (-1x2304x384xf32) <- (-1x2304x1536xf32, 1536x384xf32)
        t615 = paddle._C_ops.matmul(t614, t41, False, False)
        del t614, t41

        # pd_op.add: (-1x2304x384xf32) <- (-1x2304x384xf32, 384xf32)
        t616 = paddle._C_ops.add(t615, t42)
        del t615, t42

        # pd_op.add: (-1x2304x384xf32) <- (-1x2304x384xf32, -1x2304x384xf32)
        t617 = paddle._C_ops.add(t608, t616)
        del t608, t616

        # pd_op.shape64: (3xi64) <- (-1x2304x384xf32)
        t618 = paddle._C_ops.shape64(t617)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t619 = paddle._C_ops.slice(t618, [0], t305, t306, [1], [0])
        del t618

        # pd_op.layer_norm: (-1x2304x384xf32, -1x2304xf32, -1x2304xf32) <- (-1x2304x384xf32, 384xf32, 384xf32)
        t620, t621, t622 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t617, t43, t44, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t44, t43

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t623 = [t619, t554, t554, t555]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t624 = paddle._C_ops.stack(t623, 0)
        del t623

        # pd_op.reshape: (-1x48x48x384xf32) <- (-1x2304x384xf32, 4xi64)
        t625 = paddle._C_ops.reshape(t620, t624)
        del t620, t624

        # pd_op.shape64: (4xi64) <- (-1x48x48x384xf32)
        t626 = paddle._C_ops.shape64(t625)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t627 = paddle._C_ops.slice(t626, [0], t305, t306, [1], [0])
        del t626

        # pd_op.roll: (-1x48x48x384xf32) <- (-1x48x48x384xf32, 2xi64)
        t628 = paddle._C_ops.roll(t625, t408, [1, 2])
        del t625

        # pd_op.shape64: (4xi64) <- (-1x48x48x384xf32)
        t629 = paddle._C_ops.shape64(t628)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t630 = paddle._C_ops.slice(t629, [0], t305, t306, [1], [0])
        del t629

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t631 = [t630, t561, t332, t561, t332, t555]
        del t630

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t632 = paddle._C_ops.stack(t631, 0)
        del t631

        # pd_op.reshape: (-1x4x12x4x12x384xf32) <- (-1x48x48x384xf32, 6xi64)
        t633 = paddle._C_ops.reshape(t628, t632)
        del t628, t632

        # pd_op.transpose: (-1x4x4x12x12x384xf32) <- (-1x4x12x4x12x384xf32)
        t634 = paddle._C_ops.transpose(t633, [0, 1, 3, 2, 4, 5])
        del t633

        # pd_op.reshape: (-1x12x12x384xf32) <- (-1x4x4x12x12x384xf32, 4xi64)
        t635 = paddle._C_ops.reshape(t634, t566)
        del t634

        # pd_op.reshape: (-1x144x384xf32) <- (-1x12x12x384xf32, 3xi64)
        t636 = paddle._C_ops.reshape(t635, t568)
        del t568, t635

        # pd_op.full: (1x48x48x1xf32) <- ()
        t637 = paddle._C_ops.full(
            [1, 48, 48, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t638 = paddle._C_ops.set_value_(
            t637, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t637

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t639 = paddle._C_ops.set_value_(
            t638, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t638

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t640 = paddle._C_ops.set_value_(
            t639, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t639

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t641 = paddle._C_ops.set_value_(
            t640, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t640

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t642 = paddle._C_ops.set_value_(
            t641, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t641

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t643 = paddle._C_ops.set_value_(
            t642, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t642

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t644 = paddle._C_ops.set_value_(
            t643, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t643

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t645 = paddle._C_ops.set_value_(
            t644, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t644

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t646 = paddle._C_ops.set_value_(
            t645, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t645

        # pd_op.full_int_array: (6xi64) <- ()
        t647 = [1, 4, 12, 4, 12, 1]

        # pd_op.reshape: (1x4x12x4x12x1xf32) <- (1x48x48x1xf32, 6xi64)
        t648 = paddle._C_ops.reshape(t646, t647)
        del t647

        # pd_op.transpose: (1x4x4x12x12x1xf32) <- (1x4x12x4x12x1xf32)
        t649 = paddle._C_ops.transpose(t648, [0, 1, 3, 2, 4, 5])
        del t648

        # pd_op.reshape: (16x12x12x1xf32) <- (1x4x4x12x12x1xf32, 4xi64)
        t650 = paddle._C_ops.reshape(t649, t445)
        del t649

        # pd_op.reshape: (16x144xf32) <- (16x12x12x1xf32, 2xi64)
        t651 = paddle._C_ops.reshape(t650, t447)
        del t650

        # pd_op.unsqueeze: (16x1x144xf32) <- (16x144xf32, 1xi64)
        t652 = paddle._C_ops.unsqueeze(t651, t306)

        # pd_op.unsqueeze: (16x144x1xf32) <- (16x144xf32, 1xi64)
        t653 = paddle._C_ops.unsqueeze(t651, t354)
        del t651

        # pd_op.subtract: (16x144x144xf32) <- (16x1x144xf32, 16x144x1xf32)
        t654 = paddle._C_ops.subtract(t652, t653)
        del t652, t653

        # pd_op.not_equal: (16x144x144xb) <- (16x144x144xf32, xf32)
        t655 = paddle._C_ops.not_equal(t654, t452)

        # pd_op.full: (16x144x144xf32) <- ()
        t656 = paddle._C_ops.full(
            [16, 144, 144],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x144x144xf32) <- (16x144x144xb, 16x144x144xf32, 16x144x144xf32)
        t657 = paddle._C_ops.where(t655, t656, t654)
        del t656, t655, t654

        # pd_op.equal: (16x144x144xb) <- (16x144x144xf32, xf32)
        t658 = paddle._C_ops.equal(t657, t452)

        # pd_op.full: (16x144x144xf32) <- ()
        t659 = paddle._C_ops.full(
            [16, 144, 144],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x144x144xf32) <- (16x144x144xb, 16x144x144xf32, 16x144x144xf32)
        t660 = paddle._C_ops.where(t658, t659, t657)
        del t658, t659, t657

        # pd_op.shape64: (3xi64) <- (-1x144x384xf32)
        t661 = paddle._C_ops.shape64(t636)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t662 = paddle._C_ops.slice(t661, [0], t305, t306, [1], [0])
        del t661

        # pd_op.matmul: (-1x144x1152xf32) <- (-1x144x384xf32, 384x1152xf32)
        t663 = paddle._C_ops.matmul(t636, t45, False, False)
        del t45, t636

        # pd_op.add: (-1x144x1152xf32) <- (-1x144x1152xf32, 1152xf32)
        t664 = paddle._C_ops.add(t663, t46)
        del t663, t46

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t665 = [t662, t345, t346, t332, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t666 = paddle._C_ops.stack(t665, 0)
        del t665

        # pd_op.reshape: (-1x144x3x12x32xf32) <- (-1x144x1152xf32, 5xi64)
        t667 = paddle._C_ops.reshape(t664, t666)
        del t664, t666

        # pd_op.transpose: (3x-1x12x144x32xf32) <- (-1x144x3x12x32xf32)
        t668 = paddle._C_ops.transpose(t667, [2, 0, 3, 1, 4])
        del t667

        # pd_op.slice: (-1x12x144x32xf32) <- (3x-1x12x144x32xf32, 1xi64, 1xi64)
        t669 = paddle._C_ops.slice(t668, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x12x144x32xf32) <- (3x-1x12x144x32xf32, 1xi64, 1xi64)
        t670 = paddle._C_ops.slice(t668, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x12x144x32xf32) <- (3x-1x12x144x32xf32, 1xi64, 1xi64)
        t671 = paddle._C_ops.slice(t668, [0], t354, t356, [1], [0])
        del t668

        # pd_op.scale: (-1x12x144x32xf32) <- (-1x12x144x32xf32, 1xf32)
        t672 = paddle._C_ops.scale(t669, t358, float("0"), True)
        del t669

        # pd_op.transpose: (-1x12x32x144xf32) <- (-1x12x144x32xf32)
        t673 = paddle._C_ops.transpose(t670, [0, 1, 3, 2])
        del t670

        # pd_op.matmul: (-1x12x144x144xf32) <- (-1x12x144x32xf32, -1x12x32x144xf32)
        t674 = paddle._C_ops.matmul(t672, t673, False, False)
        del t672, t673

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t675 = paddle._C_ops.reshape(input7, t362)
        del input7

        # pd_op.index_select: (20736x12xf32) <- (529x12xf32, 20736xi64)
        t676 = paddle._C_ops.index_select(input8, t675, 0)
        del input8, t675

        # pd_op.reshape: (144x144x12xf32) <- (20736x12xf32, 3xi64)
        t677 = paddle._C_ops.reshape(t676, t365)
        del t676

        # pd_op.transpose: (12x144x144xf32) <- (144x144x12xf32)
        t678 = paddle._C_ops.transpose(t677, [2, 0, 1])
        del t677

        # pd_op.unsqueeze: (1x12x144x144xf32) <- (12x144x144xf32, 1xi64)
        t679 = paddle._C_ops.unsqueeze(t678, t305)
        del t678

        # pd_op.add: (-1x12x144x144xf32) <- (-1x12x144x144xf32, 1x12x144x144xf32)
        t680 = paddle._C_ops.add(t674, t679)
        del t674, t679

        # pd_op.full: (xi64) <- ()
        t681 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t682 = paddle._C_ops.floor_divide(t662, t681)
        del t681

        # pd_op.full: (xi64) <- ()
        t683 = paddle._C_ops.full([], float("16"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t684 = [t682, t683, t332, t345, t345]
        del t682, t683

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t685 = paddle._C_ops.stack(t684, 0)
        del t684

        # pd_op.reshape: (-1x16x12x144x144xf32) <- (-1x12x144x144xf32, 5xi64)
        t686 = paddle._C_ops.reshape(t680, t685)
        del t680, t685

        # pd_op.unsqueeze: (16x1x144x144xf32) <- (16x144x144xf32, 1xi64)
        t687 = paddle._C_ops.unsqueeze(t660, t306)
        del t660

        # pd_op.unsqueeze: (1x16x1x144x144xf32) <- (16x1x144x144xf32, 1xi64)
        t688 = paddle._C_ops.unsqueeze(t687, t305)
        del t687

        # pd_op.add: (-1x16x12x144x144xf32) <- (-1x16x12x144x144xf32, 1x16x1x144x144xf32)
        t689 = paddle._C_ops.add(t686, t688)
        del t686, t688

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t690 = [t662, t332, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t691 = paddle._C_ops.stack(t690, 0)
        del t690

        # pd_op.reshape: (-1x12x144x144xf32) <- (-1x16x12x144x144xf32, 4xi64)
        t692 = paddle._C_ops.reshape(t689, t691)
        del t689, t691

        # pd_op.softmax: (-1x12x144x144xf32) <- (-1x12x144x144xf32)
        t693 = paddle._C_ops.softmax(t692, -1)
        del t692

        # pd_op.matmul: (-1x12x144x32xf32) <- (-1x12x144x144xf32, -1x12x144x32xf32)
        t694 = paddle._C_ops.matmul(t693, t671, False, False)
        del t671, t693

        # pd_op.transpose: (-1x144x12x32xf32) <- (-1x12x144x32xf32)
        t695 = paddle._C_ops.transpose(t694, [0, 2, 1, 3])
        del t694

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t696 = [t662, t345, t555]
        del t662

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t697 = paddle._C_ops.stack(t696, 0)
        del t696

        # pd_op.reshape: (-1x144x384xf32) <- (-1x144x12x32xf32, 3xi64)
        t698 = paddle._C_ops.reshape(t695, t697)
        del t697, t695

        # pd_op.matmul: (-1x144x384xf32) <- (-1x144x384xf32, 384x384xf32)
        t699 = paddle._C_ops.matmul(t698, t47, False, False)
        del t47, t698

        # pd_op.add: (-1x144x384xf32) <- (-1x144x384xf32, 384xf32)
        t700 = paddle._C_ops.add(t699, t48)
        del t699, t48

        # pd_op.reshape: (-1x12x12x384xf32) <- (-1x144x384xf32, 4xi64)
        t701 = paddle._C_ops.reshape(t700, t566)
        del t700, t566

        # pd_op.reshape: (-1x4x4x12x12x384xf32) <- (-1x12x12x384xf32, 6xi64)
        t702 = paddle._C_ops.reshape(t701, t599)
        del t599, t701

        # pd_op.transpose: (-1x4x12x4x12x384xf32) <- (-1x4x4x12x12x384xf32)
        t703 = paddle._C_ops.transpose(t702, [0, 1, 3, 2, 4, 5])
        del t702

        # pd_op.reshape: (-1x48x48x384xf32) <- (-1x4x12x4x12x384xf32, 4xi64)
        t704 = paddle._C_ops.reshape(t703, t602)
        del t602, t703

        # pd_op.roll: (-1x48x48x384xf32) <- (-1x48x48x384xf32, 2xi64)
        t705 = paddle._C_ops.roll(t704, t503, [1, 2])
        del t704

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t706 = [t619, t604, t555]
        del t604, t619

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t707 = paddle._C_ops.stack(t706, 0)
        del t706

        # pd_op.reshape: (-1x2304x384xf32) <- (-1x48x48x384xf32, 3xi64)
        t708 = paddle._C_ops.reshape(t705, t707)
        del t705, t707

        # pd_op.add: (-1x2304x384xf32) <- (-1x2304x384xf32, -1x2304x384xf32)
        t709 = paddle._C_ops.add(t617, t708)
        del t617, t708

        # pd_op.layer_norm: (-1x2304x384xf32, -1x2304xf32, -1x2304xf32) <- (-1x2304x384xf32, 384xf32, 384xf32)
        t710, t711, t712 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t709, t49, t50, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t50, t49

        # pd_op.matmul: (-1x2304x1536xf32) <- (-1x2304x384xf32, 384x1536xf32)
        t713 = paddle._C_ops.matmul(t710, t51, False, False)
        del t710, t51

        # pd_op.add: (-1x2304x1536xf32) <- (-1x2304x1536xf32, 1536xf32)
        t714 = paddle._C_ops.add(t713, t52)
        del t713, t52

        # pd_op.gelu: (-1x2304x1536xf32) <- (-1x2304x1536xf32)
        t715 = paddle._C_ops.gelu(t714, False)
        del t714

        # pd_op.matmul: (-1x2304x384xf32) <- (-1x2304x1536xf32, 1536x384xf32)
        t716 = paddle._C_ops.matmul(t715, t53, False, False)
        del t715, t53

        # pd_op.add: (-1x2304x384xf32) <- (-1x2304x384xf32, 384xf32)
        t717 = paddle._C_ops.add(t716, t54)
        del t716, t54

        # pd_op.add: (-1x2304x384xf32) <- (-1x2304x384xf32, -1x2304x384xf32)
        t718 = paddle._C_ops.add(t709, t717)
        del t709, t717

        # pd_op.shape64: (3xi64) <- (-1x2304x384xf32)
        t719 = paddle._C_ops.shape64(t718)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t720 = paddle._C_ops.slice(t719, [0], t305, t306, [1], [0])
        del t719

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t721 = [t720, t554, t554, t555]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t722 = paddle._C_ops.stack(t721, 0)
        del t721

        # pd_op.reshape: (-1x48x48x384xf32) <- (-1x2304x384xf32, 4xi64)
        t723 = paddle._C_ops.reshape(t718, t722)
        del t718, t722

        # pd_op.strided_slice: (-1x24x24x384xf32) <- (-1x48x48x384xf32, 2xi64, 2xi64, 2xi64)
        t724 = paddle._C_ops.strided_slice(t723, [1, 2], t419, t440, t523)

        # pd_op.strided_slice: (-1x24x24x384xf32) <- (-1x48x48x384xf32, 2xi64, 2xi64, 2xi64)
        t725 = paddle._C_ops.strided_slice(t723, [1, 2], t525, t440, t523)

        # pd_op.strided_slice: (-1x24x24x384xf32) <- (-1x48x48x384xf32, 2xi64, 2xi64, 2xi64)
        t726 = paddle._C_ops.strided_slice(t723, [1, 2], t527, t440, t523)

        # pd_op.strided_slice: (-1x24x24x384xf32) <- (-1x48x48x384xf32, 2xi64, 2xi64, 2xi64)
        t727 = paddle._C_ops.strided_slice(t723, [1, 2], t421, t440, t523)

        # pd_op.shape64: (4xi64) <- (-1x48x48x384xf32)
        t728 = paddle._C_ops.shape64(t723)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t729 = paddle._C_ops.slice(t728, [0], t305, t306, [1], [0])
        del t728

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t730 = [t729, t554, t554, t555]
        del t555, t729

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t731 = paddle._C_ops.stack(t730, 0)
        del t730

        # pd_op.reshape: (-1x48x48x384xf32) <- (-1x48x48x384xf32, 4xi64)
        t732 = paddle._C_ops.reshape(t723, t731)
        del t723, t731

        # builtin.combine: ([-1x24x24x384xf32, -1x24x24x384xf32, -1x24x24x384xf32, -1x24x24x384xf32]) <- (-1x24x24x384xf32, -1x24x24x384xf32, -1x24x24x384xf32, -1x24x24x384xf32)
        t733 = [t724, t725, t726, t727]
        del t724, t725, t726, t727

        # pd_op.concat: (-1x24x24x1536xf32) <- ([-1x24x24x384xf32, -1x24x24x384xf32, -1x24x24x384xf32, -1x24x24x384xf32], 1xi32)
        t734 = paddle._C_ops.concat(t733, t535)
        del t733

        # pd_op.full: (xi64) <- ()
        t735 = paddle._C_ops.full(
            [], float("1536"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t736 = [t720, t538, t735]
        del t720

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t737 = paddle._C_ops.stack(t736, 0)
        del t736

        # pd_op.reshape: (-1x-1x1536xf32) <- (-1x24x24x1536xf32, 3xi64)
        t738 = paddle._C_ops.reshape(t734, t737)
        del t734, t737

        # pd_op.layer_norm: (-1x-1x1536xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x1536xf32, 1536xf32, 1536xf32)
        t739, t740, t741 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t738, t55, t56, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t56, t55, t738

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x1536xf32, 1536x768xf32)
        t742 = paddle._C_ops.matmul(t739, t57, False, False)
        del t739, t57

        # pd_op.shape64: (3xi64) <- (-1x-1x768xf32)
        t743 = paddle._C_ops.shape64(t742)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t744 = paddle._C_ops.slice(t743, [0], t305, t306, [1], [0])
        del t743

        # pd_op.shape64: (3xi64) <- (-1x-1x768xf32)
        t745 = paddle._C_ops.shape64(t742)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t746 = paddle._C_ops.slice(t745, [0], t306, t354, [1], [0])
        del t745

        # pd_op.layer_norm: (-1x-1x768xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x768xf32, 768xf32, 768xf32)
        t747, t748, t749 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t742, t58, t59, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t59, t58

        # pd_op.full: (xi64) <- ()
        t750 = paddle._C_ops.full([], float("24"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t751 = [t744, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t752 = paddle._C_ops.stack(t751, 0)
        del t751

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x-1x768xf32, 4xi64)
        t753 = paddle._C_ops.reshape(t747, t752)
        del t747, t752

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t754 = paddle._C_ops.shape64(t753)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t755 = paddle._C_ops.slice(t754, [0], t305, t306, [1], [0])
        del t754

        # pd_op.full: (xi64) <- ()
        t756 = paddle._C_ops.full([], float("2"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t757 = [t755, t756, t332, t756, t332, t539]
        del t755

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t758 = paddle._C_ops.stack(t757, 0)
        del t757

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t759 = paddle._C_ops.reshape(t753, t758)
        del t753, t758

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t760 = paddle._C_ops.transpose(t759, [0, 1, 3, 2, 4, 5])
        del t759

        # pd_op.full_int_array: (4xi64) <- ()
        t761 = [-1, 12, 12, 768]

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t762 = paddle._C_ops.reshape(t760, t761)
        del t760

        # pd_op.full_int_array: (3xi64) <- ()
        t763 = [-1, 144, 768]

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t764 = paddle._C_ops.reshape(t762, t763)
        del t762

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t765 = paddle._C_ops.shape64(t764)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t766 = paddle._C_ops.slice(t765, [0], t305, t306, [1], [0])
        del t765

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t767 = paddle._C_ops.matmul(t764, t60, False, False)
        del t60, t764

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t768 = paddle._C_ops.add(t767, t61)
        del t767, t61

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t769 = [t766, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t770 = paddle._C_ops.stack(t769, 0)
        del t769

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t771 = paddle._C_ops.reshape(t768, t770)
        del t768, t770

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t772 = paddle._C_ops.transpose(t771, [2, 0, 3, 1, 4])
        del t771

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t773 = paddle._C_ops.slice(t772, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t774 = paddle._C_ops.slice(t772, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t775 = paddle._C_ops.slice(t772, [0], t354, t356, [1], [0])
        del t772

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t776 = paddle._C_ops.scale(t773, t358, float("0"), True)
        del t773

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t777 = paddle._C_ops.transpose(t774, [0, 1, 3, 2])
        del t774

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t778 = paddle._C_ops.matmul(t776, t777, False, False)
        del t776, t777

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t779 = paddle._C_ops.reshape(input9, t362)
        del input9

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t780 = paddle._C_ops.index_select(input10, t779, 0)
        del input10, t779

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t781 = paddle._C_ops.reshape(t780, t365)
        del t780

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t782 = paddle._C_ops.transpose(t781, [2, 0, 1])
        del t781

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t783 = paddle._C_ops.unsqueeze(t782, t305)
        del t782

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t784 = paddle._C_ops.add(t778, t783)
        del t778, t783

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t785 = paddle._C_ops.softmax(t784, -1)
        del t784

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t786 = paddle._C_ops.matmul(t785, t775, False, False)
        del t775, t785

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t787 = paddle._C_ops.transpose(t786, [0, 2, 1, 3])
        del t786

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t788 = [t766, t345, t539]
        del t766

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t789 = paddle._C_ops.stack(t788, 0)
        del t788

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t790 = paddle._C_ops.reshape(t787, t789)
        del t789, t787

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t791 = paddle._C_ops.matmul(t790, t62, False, False)
        del t62, t790

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t792 = paddle._C_ops.add(t791, t63)
        del t791, t63

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t793 = paddle._C_ops.reshape(t792, t761)
        del t792

        # pd_op.full_int_array: (6xi64) <- ()
        t794 = [-1, 2, 2, 12, 12, 768]

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t795 = paddle._C_ops.reshape(t793, t794)
        del t793

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t796 = paddle._C_ops.transpose(t795, [0, 1, 3, 2, 4, 5])
        del t795

        # pd_op.full_int_array: (4xi64) <- ()
        t797 = [-1, 24, 24, 768]

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t798 = paddle._C_ops.reshape(t796, t797)
        del t796

        # pd_op.full: (xi64) <- ()
        t799 = paddle._C_ops.full(
            [], float("576"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t800 = [t744, t799, t539]
        del t744

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t801 = paddle._C_ops.stack(t800, 0)
        del t800

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t802 = paddle._C_ops.reshape(t798, t801)
        del t798, t801

        # pd_op.add: (-1x576x768xf32) <- (-1x-1x768xf32, -1x576x768xf32)
        t803 = paddle._C_ops.add(t742, t802)
        del t742, t802

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t804, t805, t806 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t803, t64, t65, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t65, t64

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t807 = paddle._C_ops.matmul(t804, t66, False, False)
        del t804, t66

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t808 = paddle._C_ops.add(t807, t67)
        del t807, t67

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t809 = paddle._C_ops.gelu(t808, False)
        del t808

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t810 = paddle._C_ops.matmul(t809, t68, False, False)
        del t809, t68

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t811 = paddle._C_ops.add(t810, t69)
        del t810, t69

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t812 = paddle._C_ops.add(t803, t811)
        del t803, t811

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t813 = paddle._C_ops.shape64(t812)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t814 = paddle._C_ops.slice(t813, [0], t305, t306, [1], [0])
        del t813

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t815, t816, t817 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t812, t70, t71, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t71, t70

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t818 = [t814, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t819 = paddle._C_ops.stack(t818, 0)
        del t818

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t820 = paddle._C_ops.reshape(t815, t819)
        del t815, t819

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t821 = paddle._C_ops.shape64(t820)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t822 = paddle._C_ops.slice(t821, [0], t305, t306, [1], [0])
        del t821

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t823 = paddle._C_ops.roll(t820, t408, [1, 2])
        del t820

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t824 = paddle._C_ops.shape64(t823)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t825 = paddle._C_ops.slice(t824, [0], t305, t306, [1], [0])
        del t824

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t826 = [t825, t756, t332, t756, t332, t539]
        del t825

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t827 = paddle._C_ops.stack(t826, 0)
        del t826

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t828 = paddle._C_ops.reshape(t823, t827)
        del t823, t827

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t829 = paddle._C_ops.transpose(t828, [0, 1, 3, 2, 4, 5])
        del t828

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t830 = paddle._C_ops.reshape(t829, t761)
        del t829

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t831 = paddle._C_ops.reshape(t830, t763)
        del t830

        # pd_op.full: (1x24x24x1xf32) <- ()
        t832 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t833 = paddle._C_ops.set_value_(
            t832, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t832

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t834 = paddle._C_ops.set_value_(
            t833, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t833

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t835 = paddle._C_ops.set_value_(
            t834, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t834

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t836 = paddle._C_ops.set_value_(
            t835, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t835

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t837 = paddle._C_ops.set_value_(
            t836, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t836

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t838 = paddle._C_ops.set_value_(
            t837, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t837

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t839 = paddle._C_ops.set_value_(
            t838, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t838

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t840 = paddle._C_ops.set_value_(
            t839, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t839

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t841 = paddle._C_ops.set_value_(
            t840, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t840

        # pd_op.full_int_array: (6xi64) <- ()
        t842 = [1, 2, 12, 2, 12, 1]

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t843 = paddle._C_ops.reshape(t841, t842)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t844 = paddle._C_ops.transpose(t843, [0, 1, 3, 2, 4, 5])
        del t843

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t845 = paddle._C_ops.reshape(t844, t445)
        del t844

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t846 = paddle._C_ops.reshape(t845, t447)
        del t845

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t847 = paddle._C_ops.unsqueeze(t846, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t848 = paddle._C_ops.unsqueeze(t846, t354)
        del t846

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t849 = paddle._C_ops.subtract(t847, t848)
        del t847, t848

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t850 = paddle._C_ops.not_equal(t849, t452)

        # pd_op.full: (4x144x144xf32) <- ()
        t851 = paddle._C_ops.full(
            [4, 144, 144],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t852 = paddle._C_ops.where(t850, t851, t849)
        del t850, t849

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t853 = paddle._C_ops.equal(t852, t452)

        # pd_op.full: (4x144x144xf32) <- ()
        t854 = paddle._C_ops.full(
            [4, 144, 144],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t855 = paddle._C_ops.where(t853, t854, t852)
        del t853, t852

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t856 = paddle._C_ops.shape64(t831)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t857 = paddle._C_ops.slice(t856, [0], t305, t306, [1], [0])
        del t856

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t858 = paddle._C_ops.matmul(t831, t72, False, False)
        del t72, t831

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t859 = paddle._C_ops.add(t858, t73)
        del t858, t73

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t860 = [t857, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t861 = paddle._C_ops.stack(t860, 0)
        del t860

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t862 = paddle._C_ops.reshape(t859, t861)
        del t859, t861

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t863 = paddle._C_ops.transpose(t862, [2, 0, 3, 1, 4])
        del t862

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t864 = paddle._C_ops.slice(t863, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t865 = paddle._C_ops.slice(t863, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t866 = paddle._C_ops.slice(t863, [0], t354, t356, [1], [0])
        del t863

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t867 = paddle._C_ops.scale(t864, t358, float("0"), True)
        del t864

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t868 = paddle._C_ops.transpose(t865, [0, 1, 3, 2])
        del t865

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t869 = paddle._C_ops.matmul(t867, t868, False, False)
        del t867, t868

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t870 = paddle._C_ops.reshape(input11, t362)
        del input11

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t871 = paddle._C_ops.index_select(input12, t870, 0)
        del input12, t870

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t872 = paddle._C_ops.reshape(t871, t365)
        del t871

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t873 = paddle._C_ops.transpose(t872, [2, 0, 1])
        del t872

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t874 = paddle._C_ops.unsqueeze(t873, t305)
        del t873

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t875 = paddle._C_ops.add(t869, t874)
        del t869, t874

        # pd_op.full: (xi64) <- ()
        t876 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t877 = paddle._C_ops.floor_divide(t857, t876)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t878 = [t877, t561, t750, t345, t345]
        del t877

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t879 = paddle._C_ops.stack(t878, 0)
        del t878

        # pd_op.reshape: (-1x4x24x144x144xf32) <- (-1x24x144x144xf32, 5xi64)
        t880 = paddle._C_ops.reshape(t875, t879)
        del t875, t879

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t881 = paddle._C_ops.unsqueeze(t855, t306)
        del t855

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t882 = paddle._C_ops.unsqueeze(t881, t305)
        del t881

        # pd_op.add: (-1x4x24x144x144xf32) <- (-1x4x24x144x144xf32, 1x4x1x144x144xf32)
        t883 = paddle._C_ops.add(t880, t882)
        del t880, t882

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t884 = [t857, t750, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t885 = paddle._C_ops.stack(t884, 0)
        del t884

        # pd_op.reshape: (-1x24x144x144xf32) <- (-1x4x24x144x144xf32, 4xi64)
        t886 = paddle._C_ops.reshape(t883, t885)
        del t883, t885

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t887 = paddle._C_ops.softmax(t886, -1)
        del t886

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t888 = paddle._C_ops.matmul(t887, t866, False, False)
        del t866, t887

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t889 = paddle._C_ops.transpose(t888, [0, 2, 1, 3])
        del t888

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t890 = [t857, t345, t539]
        del t857

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t891 = paddle._C_ops.stack(t890, 0)
        del t890

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t892 = paddle._C_ops.reshape(t889, t891)
        del t891, t889

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t893 = paddle._C_ops.matmul(t892, t74, False, False)
        del t74, t892

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t894 = paddle._C_ops.add(t893, t75)
        del t893, t75

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t895 = paddle._C_ops.reshape(t894, t761)
        del t894

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t896 = paddle._C_ops.reshape(t895, t794)
        del t895

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t897 = paddle._C_ops.transpose(t896, [0, 1, 3, 2, 4, 5])
        del t896

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t898 = paddle._C_ops.reshape(t897, t797)
        del t897

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t899 = paddle._C_ops.roll(t898, t503, [1, 2])
        del t898

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t900 = [t814, t799, t539]
        del t814

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t901 = paddle._C_ops.stack(t900, 0)
        del t900

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t902 = paddle._C_ops.reshape(t899, t901)
        del t899, t901

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t903 = paddle._C_ops.add(t812, t902)
        del t812, t902

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t904, t905, t906 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t903, t76, t77, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t77, t76

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t907 = paddle._C_ops.matmul(t904, t78, False, False)
        del t904, t78

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t908 = paddle._C_ops.add(t907, t79)
        del t907, t79

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t909 = paddle._C_ops.gelu(t908, False)
        del t908

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t910 = paddle._C_ops.matmul(t909, t80, False, False)
        del t909, t80

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t911 = paddle._C_ops.add(t910, t81)
        del t910, t81

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t912 = paddle._C_ops.add(t903, t911)
        del t903, t911

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t913 = paddle._C_ops.shape64(t912)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t914 = paddle._C_ops.slice(t913, [0], t305, t306, [1], [0])
        del t913

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t915, t916, t917 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t912, t82, t83, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t83, t82

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t918 = [t914, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t919 = paddle._C_ops.stack(t918, 0)
        del t918

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t920 = paddle._C_ops.reshape(t915, t919)
        del t915, t919

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t921 = paddle._C_ops.shape64(t920)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t922 = paddle._C_ops.slice(t921, [0], t305, t306, [1], [0])
        del t921

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t923 = [t922, t756, t332, t756, t332, t539]
        del t922

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t924 = paddle._C_ops.stack(t923, 0)
        del t923

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t925 = paddle._C_ops.reshape(t920, t924)
        del t920, t924

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t926 = paddle._C_ops.transpose(t925, [0, 1, 3, 2, 4, 5])
        del t925

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t927 = paddle._C_ops.reshape(t926, t761)
        del t926

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t928 = paddle._C_ops.reshape(t927, t763)
        del t927

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t929 = paddle._C_ops.shape64(t928)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t930 = paddle._C_ops.slice(t929, [0], t305, t306, [1], [0])
        del t929

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t931 = paddle._C_ops.matmul(t928, t84, False, False)
        del t84, t928

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t932 = paddle._C_ops.add(t931, t85)
        del t931, t85

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t933 = [t930, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t934 = paddle._C_ops.stack(t933, 0)
        del t933

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t935 = paddle._C_ops.reshape(t932, t934)
        del t932, t934

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t936 = paddle._C_ops.transpose(t935, [2, 0, 3, 1, 4])
        del t935

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t937 = paddle._C_ops.slice(t936, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t938 = paddle._C_ops.slice(t936, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t939 = paddle._C_ops.slice(t936, [0], t354, t356, [1], [0])
        del t936

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t940 = paddle._C_ops.scale(t937, t358, float("0"), True)
        del t937

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t941 = paddle._C_ops.transpose(t938, [0, 1, 3, 2])
        del t938

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t942 = paddle._C_ops.matmul(t940, t941, False, False)
        del t940, t941

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t943 = paddle._C_ops.reshape(input13, t362)
        del input13

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t944 = paddle._C_ops.index_select(input14, t943, 0)
        del input14, t943

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t945 = paddle._C_ops.reshape(t944, t365)
        del t944

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t946 = paddle._C_ops.transpose(t945, [2, 0, 1])
        del t945

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t947 = paddle._C_ops.unsqueeze(t946, t305)
        del t946

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t948 = paddle._C_ops.add(t942, t947)
        del t942, t947

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t949 = paddle._C_ops.softmax(t948, -1)
        del t948

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t950 = paddle._C_ops.matmul(t949, t939, False, False)
        del t939, t949

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t951 = paddle._C_ops.transpose(t950, [0, 2, 1, 3])
        del t950

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t952 = [t930, t345, t539]
        del t930

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t953 = paddle._C_ops.stack(t952, 0)
        del t952

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t954 = paddle._C_ops.reshape(t951, t953)
        del t953, t951

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t955 = paddle._C_ops.matmul(t954, t86, False, False)
        del t86, t954

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t956 = paddle._C_ops.add(t955, t87)
        del t955, t87

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t957 = paddle._C_ops.reshape(t956, t761)
        del t956

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t958 = paddle._C_ops.reshape(t957, t794)
        del t957

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t959 = paddle._C_ops.transpose(t958, [0, 1, 3, 2, 4, 5])
        del t958

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t960 = paddle._C_ops.reshape(t959, t797)
        del t959

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t961 = [t914, t799, t539]
        del t914

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t962 = paddle._C_ops.stack(t961, 0)
        del t961

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t963 = paddle._C_ops.reshape(t960, t962)
        del t960, t962

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t964 = paddle._C_ops.add(t912, t963)
        del t912, t963

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t965, t966, t967 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t964, t88, t89, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t89, t88

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t968 = paddle._C_ops.matmul(t965, t90, False, False)
        del t965, t90

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t969 = paddle._C_ops.add(t968, t91)
        del t968, t91

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t970 = paddle._C_ops.gelu(t969, False)
        del t969

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t971 = paddle._C_ops.matmul(t970, t92, False, False)
        del t970, t92

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t972 = paddle._C_ops.add(t971, t93)
        del t971, t93

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t973 = paddle._C_ops.add(t964, t972)
        del t964, t972

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t974 = paddle._C_ops.shape64(t973)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t975 = paddle._C_ops.slice(t974, [0], t305, t306, [1], [0])
        del t974

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t976, t977, t978 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t973, t94, t95, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t95, t94

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t979 = [t975, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t980 = paddle._C_ops.stack(t979, 0)
        del t979

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t981 = paddle._C_ops.reshape(t976, t980)
        del t976, t980

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t982 = paddle._C_ops.shape64(t981)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t983 = paddle._C_ops.slice(t982, [0], t305, t306, [1], [0])
        del t982

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t984 = paddle._C_ops.roll(t981, t408, [1, 2])
        del t981

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t985 = paddle._C_ops.shape64(t984)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t986 = paddle._C_ops.slice(t985, [0], t305, t306, [1], [0])
        del t985

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t987 = [t986, t756, t332, t756, t332, t539]
        del t986

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t988 = paddle._C_ops.stack(t987, 0)
        del t987

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t989 = paddle._C_ops.reshape(t984, t988)
        del t984, t988

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t990 = paddle._C_ops.transpose(t989, [0, 1, 3, 2, 4, 5])
        del t989

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t991 = paddle._C_ops.reshape(t990, t761)
        del t990

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t992 = paddle._C_ops.reshape(t991, t763)
        del t991

        # pd_op.full: (1x24x24x1xf32) <- ()
        t993 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t994 = paddle._C_ops.set_value_(
            t993, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t993

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t995 = paddle._C_ops.set_value_(
            t994, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t994

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t996 = paddle._C_ops.set_value_(
            t995, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t995

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t997 = paddle._C_ops.set_value_(
            t996, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t996

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t998 = paddle._C_ops.set_value_(
            t997, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t997

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t999 = paddle._C_ops.set_value_(
            t998, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t998

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1000 = paddle._C_ops.set_value_(
            t999, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t999

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1001 = paddle._C_ops.set_value_(
            t1000, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1000

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1002 = paddle._C_ops.set_value_(
            t1001, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1001

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1003 = paddle._C_ops.reshape(t1002, t842)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1004 = paddle._C_ops.transpose(t1003, [0, 1, 3, 2, 4, 5])
        del t1003

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1005 = paddle._C_ops.reshape(t1004, t445)
        del t1004

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1006 = paddle._C_ops.reshape(t1005, t447)
        del t1005

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1007 = paddle._C_ops.unsqueeze(t1006, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1008 = paddle._C_ops.unsqueeze(t1006, t354)
        del t1006

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1009 = paddle._C_ops.subtract(t1007, t1008)
        del t1007, t1008

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1010 = paddle._C_ops.not_equal(t1009, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1011 = paddle._C_ops.where(t1010, t851, t1009)
        del t1010, t1009

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1012 = paddle._C_ops.equal(t1011, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1013 = paddle._C_ops.where(t1012, t854, t1011)
        del t1012, t1011

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1014 = paddle._C_ops.shape64(t992)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1015 = paddle._C_ops.slice(t1014, [0], t305, t306, [1], [0])
        del t1014

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1016 = paddle._C_ops.matmul(t992, t96, False, False)
        del t96, t992

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1017 = paddle._C_ops.add(t1016, t97)
        del t1016, t97

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1018 = [t1015, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1019 = paddle._C_ops.stack(t1018, 0)
        del t1018

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1020 = paddle._C_ops.reshape(t1017, t1019)
        del t1017, t1019

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1021 = paddle._C_ops.transpose(t1020, [2, 0, 3, 1, 4])
        del t1020

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1022 = paddle._C_ops.slice(t1021, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1023 = paddle._C_ops.slice(t1021, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1024 = paddle._C_ops.slice(t1021, [0], t354, t356, [1], [0])
        del t1021

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1025 = paddle._C_ops.scale(t1022, t358, float("0"), True)
        del t1022

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1026 = paddle._C_ops.transpose(t1023, [0, 1, 3, 2])
        del t1023

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1027 = paddle._C_ops.matmul(t1025, t1026, False, False)
        del t1025, t1026

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1028 = paddle._C_ops.reshape(input15, t362)
        del input15

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1029 = paddle._C_ops.index_select(input16, t1028, 0)
        del input16, t1028

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1030 = paddle._C_ops.reshape(t1029, t365)
        del t1029

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1031 = paddle._C_ops.transpose(t1030, [2, 0, 1])
        del t1030

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1032 = paddle._C_ops.unsqueeze(t1031, t305)
        del t1031

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1033 = paddle._C_ops.add(t1027, t1032)
        del t1027, t1032

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1034 = paddle._C_ops.floor_divide(t1015, t876)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1035 = [t1034, t561, t750, t345, t345]
        del t1034

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1036 = paddle._C_ops.stack(t1035, 0)
        del t1035

        # pd_op.reshape: (-1x4x24x144x144xf32) <- (-1x24x144x144xf32, 5xi64)
        t1037 = paddle._C_ops.reshape(t1033, t1036)
        del t1033, t1036

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1038 = paddle._C_ops.unsqueeze(t1013, t306)
        del t1013

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1039 = paddle._C_ops.unsqueeze(t1038, t305)
        del t1038

        # pd_op.add: (-1x4x24x144x144xf32) <- (-1x4x24x144x144xf32, 1x4x1x144x144xf32)
        t1040 = paddle._C_ops.add(t1037, t1039)
        del t1037, t1039

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1041 = [t1015, t750, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1042 = paddle._C_ops.stack(t1041, 0)
        del t1041

        # pd_op.reshape: (-1x24x144x144xf32) <- (-1x4x24x144x144xf32, 4xi64)
        t1043 = paddle._C_ops.reshape(t1040, t1042)
        del t1040, t1042

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1044 = paddle._C_ops.softmax(t1043, -1)
        del t1043

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1045 = paddle._C_ops.matmul(t1044, t1024, False, False)
        del t1024, t1044

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1046 = paddle._C_ops.transpose(t1045, [0, 2, 1, 3])
        del t1045

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1047 = [t1015, t345, t539]
        del t1015

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1048 = paddle._C_ops.stack(t1047, 0)
        del t1047

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1049 = paddle._C_ops.reshape(t1046, t1048)
        del t1048, t1046

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1050 = paddle._C_ops.matmul(t1049, t98, False, False)
        del t98, t1049

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1051 = paddle._C_ops.add(t1050, t99)
        del t1050, t99

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1052 = paddle._C_ops.reshape(t1051, t761)
        del t1051

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1053 = paddle._C_ops.reshape(t1052, t794)
        del t1052

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1054 = paddle._C_ops.transpose(t1053, [0, 1, 3, 2, 4, 5])
        del t1053

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1055 = paddle._C_ops.reshape(t1054, t797)
        del t1054

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1056 = paddle._C_ops.roll(t1055, t503, [1, 2])
        del t1055

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1057 = [t975, t799, t539]
        del t975

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1058 = paddle._C_ops.stack(t1057, 0)
        del t1057

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1059 = paddle._C_ops.reshape(t1056, t1058)
        del t1056, t1058

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1060 = paddle._C_ops.add(t973, t1059)
        del t973, t1059

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1061, t1062, t1063 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1060, t100, t101, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t101, t100

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1064 = paddle._C_ops.matmul(t1061, t102, False, False)
        del t1061, t102

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1065 = paddle._C_ops.add(t1064, t103)
        del t1064, t103

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1066 = paddle._C_ops.gelu(t1065, False)
        del t1065

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1067 = paddle._C_ops.matmul(t1066, t104, False, False)
        del t1066, t104

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1068 = paddle._C_ops.add(t1067, t105)
        del t1067, t105

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1069 = paddle._C_ops.add(t1060, t1068)
        del t1060, t1068

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1070 = paddle._C_ops.shape64(t1069)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1071 = paddle._C_ops.slice(t1070, [0], t305, t306, [1], [0])
        del t1070

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1072, t1073, t1074 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1069, t106, t107, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t107, t106

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1075 = [t1071, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1076 = paddle._C_ops.stack(t1075, 0)
        del t1075

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1077 = paddle._C_ops.reshape(t1072, t1076)
        del t1072, t1076

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1078 = paddle._C_ops.shape64(t1077)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1079 = paddle._C_ops.slice(t1078, [0], t305, t306, [1], [0])
        del t1078

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1080 = [t1079, t756, t332, t756, t332, t539]
        del t1079

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1081 = paddle._C_ops.stack(t1080, 0)
        del t1080

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1082 = paddle._C_ops.reshape(t1077, t1081)
        del t1077, t1081

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1083 = paddle._C_ops.transpose(t1082, [0, 1, 3, 2, 4, 5])
        del t1082

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1084 = paddle._C_ops.reshape(t1083, t761)
        del t1083

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1085 = paddle._C_ops.reshape(t1084, t763)
        del t1084

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1086 = paddle._C_ops.shape64(t1085)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1087 = paddle._C_ops.slice(t1086, [0], t305, t306, [1], [0])
        del t1086

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1088 = paddle._C_ops.matmul(t1085, t108, False, False)
        del t108, t1085

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1089 = paddle._C_ops.add(t1088, t109)
        del t1088, t109

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1090 = [t1087, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1091 = paddle._C_ops.stack(t1090, 0)
        del t1090

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1092 = paddle._C_ops.reshape(t1089, t1091)
        del t1089, t1091

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1093 = paddle._C_ops.transpose(t1092, [2, 0, 3, 1, 4])
        del t1092

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1094 = paddle._C_ops.slice(t1093, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1095 = paddle._C_ops.slice(t1093, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1096 = paddle._C_ops.slice(t1093, [0], t354, t356, [1], [0])
        del t1093

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1097 = paddle._C_ops.scale(t1094, t358, float("0"), True)
        del t1094

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1098 = paddle._C_ops.transpose(t1095, [0, 1, 3, 2])
        del t1095

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1099 = paddle._C_ops.matmul(t1097, t1098, False, False)
        del t1097, t1098

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1100 = paddle._C_ops.reshape(input17, t362)
        del input17

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1101 = paddle._C_ops.index_select(input18, t1100, 0)
        del input18, t1100

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1102 = paddle._C_ops.reshape(t1101, t365)
        del t1101

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1103 = paddle._C_ops.transpose(t1102, [2, 0, 1])
        del t1102

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1104 = paddle._C_ops.unsqueeze(t1103, t305)
        del t1103

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1105 = paddle._C_ops.add(t1099, t1104)
        del t1099, t1104

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1106 = paddle._C_ops.softmax(t1105, -1)
        del t1105

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1107 = paddle._C_ops.matmul(t1106, t1096, False, False)
        del t1096, t1106

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1108 = paddle._C_ops.transpose(t1107, [0, 2, 1, 3])
        del t1107

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1109 = [t1087, t345, t539]
        del t1087

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1110 = paddle._C_ops.stack(t1109, 0)
        del t1109

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1111 = paddle._C_ops.reshape(t1108, t1110)
        del t1110, t1108

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1112 = paddle._C_ops.matmul(t1111, t110, False, False)
        del t110, t1111

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1113 = paddle._C_ops.add(t1112, t111)
        del t1112, t111

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1114 = paddle._C_ops.reshape(t1113, t761)
        del t1113

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1115 = paddle._C_ops.reshape(t1114, t794)
        del t1114

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1116 = paddle._C_ops.transpose(t1115, [0, 1, 3, 2, 4, 5])
        del t1115

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1117 = paddle._C_ops.reshape(t1116, t797)
        del t1116

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1118 = [t1071, t799, t539]
        del t1071

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1119 = paddle._C_ops.stack(t1118, 0)
        del t1118

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1120 = paddle._C_ops.reshape(t1117, t1119)
        del t1117, t1119

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1121 = paddle._C_ops.add(t1069, t1120)
        del t1069, t1120

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1122, t1123, t1124 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1121, t112, t113, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t113, t112

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1125 = paddle._C_ops.matmul(t1122, t114, False, False)
        del t1122, t114

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1126 = paddle._C_ops.add(t1125, t115)
        del t1125, t115

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1127 = paddle._C_ops.gelu(t1126, False)
        del t1126

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1128 = paddle._C_ops.matmul(t1127, t116, False, False)
        del t1127, t116

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1129 = paddle._C_ops.add(t1128, t117)
        del t1128, t117

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1130 = paddle._C_ops.add(t1121, t1129)
        del t1121, t1129

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1131 = paddle._C_ops.shape64(t1130)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1132 = paddle._C_ops.slice(t1131, [0], t305, t306, [1], [0])
        del t1131

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1133, t1134, t1135 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1130, t118, t119, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t119, t118

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1136 = [t1132, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1137 = paddle._C_ops.stack(t1136, 0)
        del t1136

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1138 = paddle._C_ops.reshape(t1133, t1137)
        del t1133, t1137

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1139 = paddle._C_ops.shape64(t1138)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1140 = paddle._C_ops.slice(t1139, [0], t305, t306, [1], [0])
        del t1139

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1141 = paddle._C_ops.roll(t1138, t408, [1, 2])
        del t1138

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1142 = paddle._C_ops.shape64(t1141)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1143 = paddle._C_ops.slice(t1142, [0], t305, t306, [1], [0])
        del t1142

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1144 = [t1143, t756, t332, t756, t332, t539]
        del t1143

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1145 = paddle._C_ops.stack(t1144, 0)
        del t1144

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1146 = paddle._C_ops.reshape(t1141, t1145)
        del t1141, t1145

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1147 = paddle._C_ops.transpose(t1146, [0, 1, 3, 2, 4, 5])
        del t1146

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1148 = paddle._C_ops.reshape(t1147, t761)
        del t1147

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1149 = paddle._C_ops.reshape(t1148, t763)
        del t1148

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1150 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1151 = paddle._C_ops.set_value_(
            t1150, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1150

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1152 = paddle._C_ops.set_value_(
            t1151, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1151

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1153 = paddle._C_ops.set_value_(
            t1152, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1152

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1154 = paddle._C_ops.set_value_(
            t1153, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1153

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1155 = paddle._C_ops.set_value_(
            t1154, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1154

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1156 = paddle._C_ops.set_value_(
            t1155, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1155

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1157 = paddle._C_ops.set_value_(
            t1156, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1156

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1158 = paddle._C_ops.set_value_(
            t1157, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1157

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1159 = paddle._C_ops.set_value_(
            t1158, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1158

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1160 = paddle._C_ops.reshape(t1159, t842)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1161 = paddle._C_ops.transpose(t1160, [0, 1, 3, 2, 4, 5])
        del t1160

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1162 = paddle._C_ops.reshape(t1161, t445)
        del t1161

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1163 = paddle._C_ops.reshape(t1162, t447)
        del t1162

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1164 = paddle._C_ops.unsqueeze(t1163, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1165 = paddle._C_ops.unsqueeze(t1163, t354)
        del t1163

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1166 = paddle._C_ops.subtract(t1164, t1165)
        del t1164, t1165

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1167 = paddle._C_ops.not_equal(t1166, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1168 = paddle._C_ops.where(t1167, t851, t1166)
        del t1167, t1166

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1169 = paddle._C_ops.equal(t1168, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1170 = paddle._C_ops.where(t1169, t854, t1168)
        del t1169, t1168

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1171 = paddle._C_ops.shape64(t1149)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1172 = paddle._C_ops.slice(t1171, [0], t305, t306, [1], [0])
        del t1171

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1173 = paddle._C_ops.matmul(t1149, t120, False, False)
        del t120, t1149

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1174 = paddle._C_ops.add(t1173, t121)
        del t1173, t121

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1175 = [t1172, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1176 = paddle._C_ops.stack(t1175, 0)
        del t1175

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1177 = paddle._C_ops.reshape(t1174, t1176)
        del t1174, t1176

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1178 = paddle._C_ops.transpose(t1177, [2, 0, 3, 1, 4])
        del t1177

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1179 = paddle._C_ops.slice(t1178, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1180 = paddle._C_ops.slice(t1178, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1181 = paddle._C_ops.slice(t1178, [0], t354, t356, [1], [0])
        del t1178

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1182 = paddle._C_ops.scale(t1179, t358, float("0"), True)
        del t1179

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1183 = paddle._C_ops.transpose(t1180, [0, 1, 3, 2])
        del t1180

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1184 = paddle._C_ops.matmul(t1182, t1183, False, False)
        del t1182, t1183

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1185 = paddle._C_ops.reshape(input19, t362)
        del input19

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1186 = paddle._C_ops.index_select(input20, t1185, 0)
        del input20, t1185

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1187 = paddle._C_ops.reshape(t1186, t365)
        del t1186

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1188 = paddle._C_ops.transpose(t1187, [2, 0, 1])
        del t1187

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1189 = paddle._C_ops.unsqueeze(t1188, t305)
        del t1188

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1190 = paddle._C_ops.add(t1184, t1189)
        del t1184, t1189

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1191 = paddle._C_ops.floor_divide(t1172, t876)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1192 = [t1191, t561, t750, t345, t345]
        del t1191

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1193 = paddle._C_ops.stack(t1192, 0)
        del t1192

        # pd_op.reshape: (-1x4x24x144x144xf32) <- (-1x24x144x144xf32, 5xi64)
        t1194 = paddle._C_ops.reshape(t1190, t1193)
        del t1190, t1193

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1195 = paddle._C_ops.unsqueeze(t1170, t306)
        del t1170

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1196 = paddle._C_ops.unsqueeze(t1195, t305)
        del t1195

        # pd_op.add: (-1x4x24x144x144xf32) <- (-1x4x24x144x144xf32, 1x4x1x144x144xf32)
        t1197 = paddle._C_ops.add(t1194, t1196)
        del t1194, t1196

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1198 = [t1172, t750, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1199 = paddle._C_ops.stack(t1198, 0)
        del t1198

        # pd_op.reshape: (-1x24x144x144xf32) <- (-1x4x24x144x144xf32, 4xi64)
        t1200 = paddle._C_ops.reshape(t1197, t1199)
        del t1197, t1199

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1201 = paddle._C_ops.softmax(t1200, -1)
        del t1200

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1202 = paddle._C_ops.matmul(t1201, t1181, False, False)
        del t1181, t1201

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1203 = paddle._C_ops.transpose(t1202, [0, 2, 1, 3])
        del t1202

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1204 = [t1172, t345, t539]
        del t1172

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1205 = paddle._C_ops.stack(t1204, 0)
        del t1204

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1206 = paddle._C_ops.reshape(t1203, t1205)
        del t1205, t1203

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1207 = paddle._C_ops.matmul(t1206, t122, False, False)
        del t122, t1206

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1208 = paddle._C_ops.add(t1207, t123)
        del t1207, t123

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1209 = paddle._C_ops.reshape(t1208, t761)
        del t1208

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1210 = paddle._C_ops.reshape(t1209, t794)
        del t1209

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1211 = paddle._C_ops.transpose(t1210, [0, 1, 3, 2, 4, 5])
        del t1210

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1212 = paddle._C_ops.reshape(t1211, t797)
        del t1211

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1213 = paddle._C_ops.roll(t1212, t503, [1, 2])
        del t1212

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1214 = [t1132, t799, t539]
        del t1132

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1215 = paddle._C_ops.stack(t1214, 0)
        del t1214

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1216 = paddle._C_ops.reshape(t1213, t1215)
        del t1213, t1215

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1217 = paddle._C_ops.add(t1130, t1216)
        del t1130, t1216

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1218, t1219, t1220 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1217, t124, t125, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t125, t124

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1221 = paddle._C_ops.matmul(t1218, t126, False, False)
        del t1218, t126

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1222 = paddle._C_ops.add(t1221, t127)
        del t1221, t127

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1223 = paddle._C_ops.gelu(t1222, False)
        del t1222

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1224 = paddle._C_ops.matmul(t1223, t128, False, False)
        del t1223, t128

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1225 = paddle._C_ops.add(t1224, t129)
        del t1224, t129

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1226 = paddle._C_ops.add(t1217, t1225)
        del t1217, t1225

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1227 = paddle._C_ops.shape64(t1226)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1228 = paddle._C_ops.slice(t1227, [0], t305, t306, [1], [0])
        del t1227

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1229, t1230, t1231 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1226, t130, t131, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t131, t130

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1232 = [t1228, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1233 = paddle._C_ops.stack(t1232, 0)
        del t1232

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1234 = paddle._C_ops.reshape(t1229, t1233)
        del t1229, t1233

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1235 = paddle._C_ops.shape64(t1234)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1236 = paddle._C_ops.slice(t1235, [0], t305, t306, [1], [0])
        del t1235

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1237 = [t1236, t756, t332, t756, t332, t539]
        del t1236

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1238 = paddle._C_ops.stack(t1237, 0)
        del t1237

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1239 = paddle._C_ops.reshape(t1234, t1238)
        del t1234, t1238

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1240 = paddle._C_ops.transpose(t1239, [0, 1, 3, 2, 4, 5])
        del t1239

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1241 = paddle._C_ops.reshape(t1240, t761)
        del t1240

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1242 = paddle._C_ops.reshape(t1241, t763)
        del t1241

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1243 = paddle._C_ops.shape64(t1242)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1244 = paddle._C_ops.slice(t1243, [0], t305, t306, [1], [0])
        del t1243

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1245 = paddle._C_ops.matmul(t1242, t132, False, False)
        del t132, t1242

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1246 = paddle._C_ops.add(t1245, t133)
        del t1245, t133

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1247 = [t1244, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1248 = paddle._C_ops.stack(t1247, 0)
        del t1247

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1249 = paddle._C_ops.reshape(t1246, t1248)
        del t1246, t1248

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1250 = paddle._C_ops.transpose(t1249, [2, 0, 3, 1, 4])
        del t1249

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1251 = paddle._C_ops.slice(t1250, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1252 = paddle._C_ops.slice(t1250, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1253 = paddle._C_ops.slice(t1250, [0], t354, t356, [1], [0])
        del t1250

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1254 = paddle._C_ops.scale(t1251, t358, float("0"), True)
        del t1251

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1255 = paddle._C_ops.transpose(t1252, [0, 1, 3, 2])
        del t1252

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1256 = paddle._C_ops.matmul(t1254, t1255, False, False)
        del t1254, t1255

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1257 = paddle._C_ops.reshape(input21, t362)
        del input21

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1258 = paddle._C_ops.index_select(input22, t1257, 0)
        del input22, t1257

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1259 = paddle._C_ops.reshape(t1258, t365)
        del t1258

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1260 = paddle._C_ops.transpose(t1259, [2, 0, 1])
        del t1259

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1261 = paddle._C_ops.unsqueeze(t1260, t305)
        del t1260

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1262 = paddle._C_ops.add(t1256, t1261)
        del t1256, t1261

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1263 = paddle._C_ops.softmax(t1262, -1)
        del t1262

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1264 = paddle._C_ops.matmul(t1263, t1253, False, False)
        del t1253, t1263

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1265 = paddle._C_ops.transpose(t1264, [0, 2, 1, 3])
        del t1264

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1266 = [t1244, t345, t539]
        del t1244

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1267 = paddle._C_ops.stack(t1266, 0)
        del t1266

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1268 = paddle._C_ops.reshape(t1265, t1267)
        del t1267, t1265

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1269 = paddle._C_ops.matmul(t1268, t134, False, False)
        del t134, t1268

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1270 = paddle._C_ops.add(t1269, t135)
        del t1269, t135

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1271 = paddle._C_ops.reshape(t1270, t761)
        del t1270

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1272 = paddle._C_ops.reshape(t1271, t794)
        del t1271

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1273 = paddle._C_ops.transpose(t1272, [0, 1, 3, 2, 4, 5])
        del t1272

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1274 = paddle._C_ops.reshape(t1273, t797)
        del t1273

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1275 = [t1228, t799, t539]
        del t1228

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1276 = paddle._C_ops.stack(t1275, 0)
        del t1275

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1277 = paddle._C_ops.reshape(t1274, t1276)
        del t1274, t1276

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1278 = paddle._C_ops.add(t1226, t1277)
        del t1226, t1277

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1279, t1280, t1281 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1278, t136, t137, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t137, t136

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1282 = paddle._C_ops.matmul(t1279, t138, False, False)
        del t1279, t138

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1283 = paddle._C_ops.add(t1282, t139)
        del t1282, t139

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1284 = paddle._C_ops.gelu(t1283, False)
        del t1283

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1285 = paddle._C_ops.matmul(t1284, t140, False, False)
        del t1284, t140

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1286 = paddle._C_ops.add(t1285, t141)
        del t1285, t141

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1287 = paddle._C_ops.add(t1278, t1286)
        del t1278, t1286

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1288 = paddle._C_ops.shape64(t1287)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1289 = paddle._C_ops.slice(t1288, [0], t305, t306, [1], [0])
        del t1288

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1290, t1291, t1292 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1287, t142, t143, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t143, t142

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1293 = [t1289, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1294 = paddle._C_ops.stack(t1293, 0)
        del t1293

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1295 = paddle._C_ops.reshape(t1290, t1294)
        del t1290, t1294

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1296 = paddle._C_ops.shape64(t1295)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1297 = paddle._C_ops.slice(t1296, [0], t305, t306, [1], [0])
        del t1296

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1298 = paddle._C_ops.roll(t1295, t408, [1, 2])
        del t1295

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1299 = paddle._C_ops.shape64(t1298)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1300 = paddle._C_ops.slice(t1299, [0], t305, t306, [1], [0])
        del t1299

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1301 = [t1300, t756, t332, t756, t332, t539]
        del t1300

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1302 = paddle._C_ops.stack(t1301, 0)
        del t1301

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1303 = paddle._C_ops.reshape(t1298, t1302)
        del t1298, t1302

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1304 = paddle._C_ops.transpose(t1303, [0, 1, 3, 2, 4, 5])
        del t1303

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1305 = paddle._C_ops.reshape(t1304, t761)
        del t1304

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1306 = paddle._C_ops.reshape(t1305, t763)
        del t1305

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1307 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1308 = paddle._C_ops.set_value_(
            t1307, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1307

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1309 = paddle._C_ops.set_value_(
            t1308, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1308

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1310 = paddle._C_ops.set_value_(
            t1309, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1309

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1311 = paddle._C_ops.set_value_(
            t1310, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1310

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1312 = paddle._C_ops.set_value_(
            t1311, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1311

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1313 = paddle._C_ops.set_value_(
            t1312, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1312

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1314 = paddle._C_ops.set_value_(
            t1313, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1313

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1315 = paddle._C_ops.set_value_(
            t1314, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1314

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1316 = paddle._C_ops.set_value_(
            t1315, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1315

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1317 = paddle._C_ops.reshape(t1316, t842)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1318 = paddle._C_ops.transpose(t1317, [0, 1, 3, 2, 4, 5])
        del t1317

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1319 = paddle._C_ops.reshape(t1318, t445)
        del t1318

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1320 = paddle._C_ops.reshape(t1319, t447)
        del t1319

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1321 = paddle._C_ops.unsqueeze(t1320, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1322 = paddle._C_ops.unsqueeze(t1320, t354)
        del t1320

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1323 = paddle._C_ops.subtract(t1321, t1322)
        del t1321, t1322

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1324 = paddle._C_ops.not_equal(t1323, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1325 = paddle._C_ops.where(t1324, t851, t1323)
        del t1324, t1323

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1326 = paddle._C_ops.equal(t1325, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1327 = paddle._C_ops.where(t1326, t854, t1325)
        del t1326, t1325

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1328 = paddle._C_ops.shape64(t1306)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1329 = paddle._C_ops.slice(t1328, [0], t305, t306, [1], [0])
        del t1328

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1330 = paddle._C_ops.matmul(t1306, t144, False, False)
        del t144, t1306

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1331 = paddle._C_ops.add(t1330, t145)
        del t1330, t145

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1332 = [t1329, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1333 = paddle._C_ops.stack(t1332, 0)
        del t1332

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1334 = paddle._C_ops.reshape(t1331, t1333)
        del t1331, t1333

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1335 = paddle._C_ops.transpose(t1334, [2, 0, 3, 1, 4])
        del t1334

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1336 = paddle._C_ops.slice(t1335, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1337 = paddle._C_ops.slice(t1335, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1338 = paddle._C_ops.slice(t1335, [0], t354, t356, [1], [0])
        del t1335

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1339 = paddle._C_ops.scale(t1336, t358, float("0"), True)
        del t1336

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1340 = paddle._C_ops.transpose(t1337, [0, 1, 3, 2])
        del t1337

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1341 = paddle._C_ops.matmul(t1339, t1340, False, False)
        del t1339, t1340

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1342 = paddle._C_ops.reshape(input23, t362)
        del input23

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1343 = paddle._C_ops.index_select(input24, t1342, 0)
        del input24, t1342

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1344 = paddle._C_ops.reshape(t1343, t365)
        del t1343

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1345 = paddle._C_ops.transpose(t1344, [2, 0, 1])
        del t1344

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1346 = paddle._C_ops.unsqueeze(t1345, t305)
        del t1345

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1347 = paddle._C_ops.add(t1341, t1346)
        del t1341, t1346

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1348 = paddle._C_ops.floor_divide(t1329, t876)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1349 = [t1348, t561, t750, t345, t345]
        del t1348

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1350 = paddle._C_ops.stack(t1349, 0)
        del t1349

        # pd_op.reshape: (-1x4x24x144x144xf32) <- (-1x24x144x144xf32, 5xi64)
        t1351 = paddle._C_ops.reshape(t1347, t1350)
        del t1347, t1350

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1352 = paddle._C_ops.unsqueeze(t1327, t306)
        del t1327

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1353 = paddle._C_ops.unsqueeze(t1352, t305)
        del t1352

        # pd_op.add: (-1x4x24x144x144xf32) <- (-1x4x24x144x144xf32, 1x4x1x144x144xf32)
        t1354 = paddle._C_ops.add(t1351, t1353)
        del t1351, t1353

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1355 = [t1329, t750, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1356 = paddle._C_ops.stack(t1355, 0)
        del t1355

        # pd_op.reshape: (-1x24x144x144xf32) <- (-1x4x24x144x144xf32, 4xi64)
        t1357 = paddle._C_ops.reshape(t1354, t1356)
        del t1354, t1356

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1358 = paddle._C_ops.softmax(t1357, -1)
        del t1357

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1359 = paddle._C_ops.matmul(t1358, t1338, False, False)
        del t1338, t1358

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1360 = paddle._C_ops.transpose(t1359, [0, 2, 1, 3])
        del t1359

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1361 = [t1329, t345, t539]
        del t1329

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1362 = paddle._C_ops.stack(t1361, 0)
        del t1361

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1363 = paddle._C_ops.reshape(t1360, t1362)
        del t1362, t1360

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1364 = paddle._C_ops.matmul(t1363, t146, False, False)
        del t146, t1363

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1365 = paddle._C_ops.add(t1364, t147)
        del t1364, t147

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1366 = paddle._C_ops.reshape(t1365, t761)
        del t1365

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1367 = paddle._C_ops.reshape(t1366, t794)
        del t1366

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1368 = paddle._C_ops.transpose(t1367, [0, 1, 3, 2, 4, 5])
        del t1367

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1369 = paddle._C_ops.reshape(t1368, t797)
        del t1368

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1370 = paddle._C_ops.roll(t1369, t503, [1, 2])
        del t1369

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1371 = [t1289, t799, t539]
        del t1289

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1372 = paddle._C_ops.stack(t1371, 0)
        del t1371

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1373 = paddle._C_ops.reshape(t1370, t1372)
        del t1370, t1372

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1374 = paddle._C_ops.add(t1287, t1373)
        del t1287, t1373

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1375, t1376, t1377 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1374, t148, t149, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t149, t148

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1378 = paddle._C_ops.matmul(t1375, t150, False, False)
        del t1375, t150

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1379 = paddle._C_ops.add(t1378, t151)
        del t1378, t151

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1380 = paddle._C_ops.gelu(t1379, False)
        del t1379

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1381 = paddle._C_ops.matmul(t1380, t152, False, False)
        del t1380, t152

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1382 = paddle._C_ops.add(t1381, t153)
        del t1381, t153

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1383 = paddle._C_ops.add(t1374, t1382)
        del t1374, t1382

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1384 = paddle._C_ops.shape64(t1383)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1385 = paddle._C_ops.slice(t1384, [0], t305, t306, [1], [0])
        del t1384

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1386, t1387, t1388 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1383, t154, t155, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t155, t154

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1389 = [t1385, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1390 = paddle._C_ops.stack(t1389, 0)
        del t1389

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1391 = paddle._C_ops.reshape(t1386, t1390)
        del t1386, t1390

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1392 = paddle._C_ops.shape64(t1391)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1393 = paddle._C_ops.slice(t1392, [0], t305, t306, [1], [0])
        del t1392

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1394 = [t1393, t756, t332, t756, t332, t539]
        del t1393

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1395 = paddle._C_ops.stack(t1394, 0)
        del t1394

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1396 = paddle._C_ops.reshape(t1391, t1395)
        del t1391, t1395

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1397 = paddle._C_ops.transpose(t1396, [0, 1, 3, 2, 4, 5])
        del t1396

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1398 = paddle._C_ops.reshape(t1397, t761)
        del t1397

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1399 = paddle._C_ops.reshape(t1398, t763)
        del t1398

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1400 = paddle._C_ops.shape64(t1399)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1401 = paddle._C_ops.slice(t1400, [0], t305, t306, [1], [0])
        del t1400

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1402 = paddle._C_ops.matmul(t1399, t156, False, False)
        del t156, t1399

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1403 = paddle._C_ops.add(t1402, t157)
        del t1402, t157

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1404 = [t1401, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1405 = paddle._C_ops.stack(t1404, 0)
        del t1404

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1406 = paddle._C_ops.reshape(t1403, t1405)
        del t1403, t1405

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1407 = paddle._C_ops.transpose(t1406, [2, 0, 3, 1, 4])
        del t1406

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1408 = paddle._C_ops.slice(t1407, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1409 = paddle._C_ops.slice(t1407, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1410 = paddle._C_ops.slice(t1407, [0], t354, t356, [1], [0])
        del t1407

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1411 = paddle._C_ops.scale(t1408, t358, float("0"), True)
        del t1408

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1412 = paddle._C_ops.transpose(t1409, [0, 1, 3, 2])
        del t1409

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1413 = paddle._C_ops.matmul(t1411, t1412, False, False)
        del t1411, t1412

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1414 = paddle._C_ops.reshape(input25, t362)
        del input25

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1415 = paddle._C_ops.index_select(input26, t1414, 0)
        del input26, t1414

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1416 = paddle._C_ops.reshape(t1415, t365)
        del t1415

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1417 = paddle._C_ops.transpose(t1416, [2, 0, 1])
        del t1416

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1418 = paddle._C_ops.unsqueeze(t1417, t305)
        del t1417

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1419 = paddle._C_ops.add(t1413, t1418)
        del t1413, t1418

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1420 = paddle._C_ops.softmax(t1419, -1)
        del t1419

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1421 = paddle._C_ops.matmul(t1420, t1410, False, False)
        del t1410, t1420

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1422 = paddle._C_ops.transpose(t1421, [0, 2, 1, 3])
        del t1421

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1423 = [t1401, t345, t539]
        del t1401

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1424 = paddle._C_ops.stack(t1423, 0)
        del t1423

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1425 = paddle._C_ops.reshape(t1422, t1424)
        del t1424, t1422

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1426 = paddle._C_ops.matmul(t1425, t158, False, False)
        del t158, t1425

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1427 = paddle._C_ops.add(t1426, t159)
        del t1426, t159

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1428 = paddle._C_ops.reshape(t1427, t761)
        del t1427

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1429 = paddle._C_ops.reshape(t1428, t794)
        del t1428

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1430 = paddle._C_ops.transpose(t1429, [0, 1, 3, 2, 4, 5])
        del t1429

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1431 = paddle._C_ops.reshape(t1430, t797)
        del t1430

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1432 = [t1385, t799, t539]
        del t1385

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1433 = paddle._C_ops.stack(t1432, 0)
        del t1432

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1434 = paddle._C_ops.reshape(t1431, t1433)
        del t1431, t1433

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1435 = paddle._C_ops.add(t1383, t1434)
        del t1383, t1434

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1436, t1437, t1438 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1435, t160, t161, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t161, t160

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1439 = paddle._C_ops.matmul(t1436, t162, False, False)
        del t1436, t162

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1440 = paddle._C_ops.add(t1439, t163)
        del t1439, t163

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1441 = paddle._C_ops.gelu(t1440, False)
        del t1440

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1442 = paddle._C_ops.matmul(t1441, t164, False, False)
        del t1441, t164

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1443 = paddle._C_ops.add(t1442, t165)
        del t1442, t165

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1444 = paddle._C_ops.add(t1435, t1443)
        del t1435, t1443

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1445 = paddle._C_ops.shape64(t1444)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1446 = paddle._C_ops.slice(t1445, [0], t305, t306, [1], [0])
        del t1445

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1447, t1448, t1449 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1444, t166, t167, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t167, t166

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1450 = [t1446, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1451 = paddle._C_ops.stack(t1450, 0)
        del t1450

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1452 = paddle._C_ops.reshape(t1447, t1451)
        del t1447, t1451

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1453 = paddle._C_ops.shape64(t1452)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1454 = paddle._C_ops.slice(t1453, [0], t305, t306, [1], [0])
        del t1453

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1455 = paddle._C_ops.roll(t1452, t408, [1, 2])
        del t1452

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1456 = paddle._C_ops.shape64(t1455)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1457 = paddle._C_ops.slice(t1456, [0], t305, t306, [1], [0])
        del t1456

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1458 = [t1457, t756, t332, t756, t332, t539]
        del t1457

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1459 = paddle._C_ops.stack(t1458, 0)
        del t1458

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1460 = paddle._C_ops.reshape(t1455, t1459)
        del t1455, t1459

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1461 = paddle._C_ops.transpose(t1460, [0, 1, 3, 2, 4, 5])
        del t1460

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1462 = paddle._C_ops.reshape(t1461, t761)
        del t1461

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1463 = paddle._C_ops.reshape(t1462, t763)
        del t1462

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1464 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1465 = paddle._C_ops.set_value_(
            t1464, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1464

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1466 = paddle._C_ops.set_value_(
            t1465, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1465

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1467 = paddle._C_ops.set_value_(
            t1466, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1466

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1468 = paddle._C_ops.set_value_(
            t1467, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1467

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1469 = paddle._C_ops.set_value_(
            t1468, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1468

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1470 = paddle._C_ops.set_value_(
            t1469, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1469

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1471 = paddle._C_ops.set_value_(
            t1470, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1470

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1472 = paddle._C_ops.set_value_(
            t1471, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1471

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1473 = paddle._C_ops.set_value_(
            t1472, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1472

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1474 = paddle._C_ops.reshape(t1473, t842)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1475 = paddle._C_ops.transpose(t1474, [0, 1, 3, 2, 4, 5])
        del t1474

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1476 = paddle._C_ops.reshape(t1475, t445)
        del t1475

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1477 = paddle._C_ops.reshape(t1476, t447)
        del t1476

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1478 = paddle._C_ops.unsqueeze(t1477, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1479 = paddle._C_ops.unsqueeze(t1477, t354)
        del t1477

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1480 = paddle._C_ops.subtract(t1478, t1479)
        del t1478, t1479

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1481 = paddle._C_ops.not_equal(t1480, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1482 = paddle._C_ops.where(t1481, t851, t1480)
        del t1481, t1480

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1483 = paddle._C_ops.equal(t1482, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1484 = paddle._C_ops.where(t1483, t854, t1482)
        del t1483, t1482

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1485 = paddle._C_ops.shape64(t1463)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1486 = paddle._C_ops.slice(t1485, [0], t305, t306, [1], [0])
        del t1485

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1487 = paddle._C_ops.matmul(t1463, t168, False, False)
        del t168, t1463

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1488 = paddle._C_ops.add(t1487, t169)
        del t1487, t169

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1489 = [t1486, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1490 = paddle._C_ops.stack(t1489, 0)
        del t1489

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1491 = paddle._C_ops.reshape(t1488, t1490)
        del t1488, t1490

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1492 = paddle._C_ops.transpose(t1491, [2, 0, 3, 1, 4])
        del t1491

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1493 = paddle._C_ops.slice(t1492, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1494 = paddle._C_ops.slice(t1492, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1495 = paddle._C_ops.slice(t1492, [0], t354, t356, [1], [0])
        del t1492

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1496 = paddle._C_ops.scale(t1493, t358, float("0"), True)
        del t1493

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1497 = paddle._C_ops.transpose(t1494, [0, 1, 3, 2])
        del t1494

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1498 = paddle._C_ops.matmul(t1496, t1497, False, False)
        del t1496, t1497

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1499 = paddle._C_ops.reshape(input27, t362)
        del input27

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1500 = paddle._C_ops.index_select(input28, t1499, 0)
        del input28, t1499

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1501 = paddle._C_ops.reshape(t1500, t365)
        del t1500

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1502 = paddle._C_ops.transpose(t1501, [2, 0, 1])
        del t1501

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1503 = paddle._C_ops.unsqueeze(t1502, t305)
        del t1502

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1504 = paddle._C_ops.add(t1498, t1503)
        del t1498, t1503

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1505 = paddle._C_ops.floor_divide(t1486, t876)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1506 = [t1505, t561, t750, t345, t345]
        del t1505

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1507 = paddle._C_ops.stack(t1506, 0)
        del t1506

        # pd_op.reshape: (-1x4x24x144x144xf32) <- (-1x24x144x144xf32, 5xi64)
        t1508 = paddle._C_ops.reshape(t1504, t1507)
        del t1504, t1507

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1509 = paddle._C_ops.unsqueeze(t1484, t306)
        del t1484

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1510 = paddle._C_ops.unsqueeze(t1509, t305)
        del t1509

        # pd_op.add: (-1x4x24x144x144xf32) <- (-1x4x24x144x144xf32, 1x4x1x144x144xf32)
        t1511 = paddle._C_ops.add(t1508, t1510)
        del t1508, t1510

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1512 = [t1486, t750, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1513 = paddle._C_ops.stack(t1512, 0)
        del t1512

        # pd_op.reshape: (-1x24x144x144xf32) <- (-1x4x24x144x144xf32, 4xi64)
        t1514 = paddle._C_ops.reshape(t1511, t1513)
        del t1511, t1513

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1515 = paddle._C_ops.softmax(t1514, -1)
        del t1514

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1516 = paddle._C_ops.matmul(t1515, t1495, False, False)
        del t1495, t1515

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1517 = paddle._C_ops.transpose(t1516, [0, 2, 1, 3])
        del t1516

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1518 = [t1486, t345, t539]
        del t1486

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1519 = paddle._C_ops.stack(t1518, 0)
        del t1518

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1520 = paddle._C_ops.reshape(t1517, t1519)
        del t1519, t1517

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1521 = paddle._C_ops.matmul(t1520, t170, False, False)
        del t170, t1520

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1522 = paddle._C_ops.add(t1521, t171)
        del t1521, t171

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1523 = paddle._C_ops.reshape(t1522, t761)
        del t1522

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1524 = paddle._C_ops.reshape(t1523, t794)
        del t1523

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1525 = paddle._C_ops.transpose(t1524, [0, 1, 3, 2, 4, 5])
        del t1524

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1526 = paddle._C_ops.reshape(t1525, t797)
        del t1525

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1527 = paddle._C_ops.roll(t1526, t503, [1, 2])
        del t1526

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1528 = [t1446, t799, t539]
        del t1446

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1529 = paddle._C_ops.stack(t1528, 0)
        del t1528

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1530 = paddle._C_ops.reshape(t1527, t1529)
        del t1527, t1529

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1531 = paddle._C_ops.add(t1444, t1530)
        del t1444, t1530

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1532, t1533, t1534 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1531, t172, t173, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t173, t172

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1535 = paddle._C_ops.matmul(t1532, t174, False, False)
        del t1532, t174

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1536 = paddle._C_ops.add(t1535, t175)
        del t1535, t175

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1537 = paddle._C_ops.gelu(t1536, False)
        del t1536

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1538 = paddle._C_ops.matmul(t1537, t176, False, False)
        del t1537, t176

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1539 = paddle._C_ops.add(t1538, t177)
        del t1538, t177

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1540 = paddle._C_ops.add(t1531, t1539)
        del t1531, t1539

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1541 = paddle._C_ops.shape64(t1540)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1542 = paddle._C_ops.slice(t1541, [0], t305, t306, [1], [0])
        del t1541

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1543, t1544, t1545 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1540, t178, t179, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t179, t178

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1546 = [t1542, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1547 = paddle._C_ops.stack(t1546, 0)
        del t1546

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1548 = paddle._C_ops.reshape(t1543, t1547)
        del t1543, t1547

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1549 = paddle._C_ops.shape64(t1548)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1550 = paddle._C_ops.slice(t1549, [0], t305, t306, [1], [0])
        del t1549

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1551 = [t1550, t756, t332, t756, t332, t539]
        del t1550

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1552 = paddle._C_ops.stack(t1551, 0)
        del t1551

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1553 = paddle._C_ops.reshape(t1548, t1552)
        del t1548, t1552

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1554 = paddle._C_ops.transpose(t1553, [0, 1, 3, 2, 4, 5])
        del t1553

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1555 = paddle._C_ops.reshape(t1554, t761)
        del t1554

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1556 = paddle._C_ops.reshape(t1555, t763)
        del t1555

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1557 = paddle._C_ops.shape64(t1556)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1558 = paddle._C_ops.slice(t1557, [0], t305, t306, [1], [0])
        del t1557

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1559 = paddle._C_ops.matmul(t1556, t180, False, False)
        del t180, t1556

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1560 = paddle._C_ops.add(t1559, t181)
        del t1559, t181

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1561 = [t1558, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1562 = paddle._C_ops.stack(t1561, 0)
        del t1561

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1563 = paddle._C_ops.reshape(t1560, t1562)
        del t1560, t1562

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1564 = paddle._C_ops.transpose(t1563, [2, 0, 3, 1, 4])
        del t1563

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1565 = paddle._C_ops.slice(t1564, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1566 = paddle._C_ops.slice(t1564, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1567 = paddle._C_ops.slice(t1564, [0], t354, t356, [1], [0])
        del t1564

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1568 = paddle._C_ops.scale(t1565, t358, float("0"), True)
        del t1565

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1569 = paddle._C_ops.transpose(t1566, [0, 1, 3, 2])
        del t1566

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1570 = paddle._C_ops.matmul(t1568, t1569, False, False)
        del t1568, t1569

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1571 = paddle._C_ops.reshape(input29, t362)
        del input29

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1572 = paddle._C_ops.index_select(input30, t1571, 0)
        del input30, t1571

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1573 = paddle._C_ops.reshape(t1572, t365)
        del t1572

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1574 = paddle._C_ops.transpose(t1573, [2, 0, 1])
        del t1573

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1575 = paddle._C_ops.unsqueeze(t1574, t305)
        del t1574

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1576 = paddle._C_ops.add(t1570, t1575)
        del t1570, t1575

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1577 = paddle._C_ops.softmax(t1576, -1)
        del t1576

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1578 = paddle._C_ops.matmul(t1577, t1567, False, False)
        del t1567, t1577

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1579 = paddle._C_ops.transpose(t1578, [0, 2, 1, 3])
        del t1578

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1580 = [t1558, t345, t539]
        del t1558

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1581 = paddle._C_ops.stack(t1580, 0)
        del t1580

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1582 = paddle._C_ops.reshape(t1579, t1581)
        del t1581, t1579

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1583 = paddle._C_ops.matmul(t1582, t182, False, False)
        del t182, t1582

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1584 = paddle._C_ops.add(t1583, t183)
        del t1583, t183

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1585 = paddle._C_ops.reshape(t1584, t761)
        del t1584

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1586 = paddle._C_ops.reshape(t1585, t794)
        del t1585

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1587 = paddle._C_ops.transpose(t1586, [0, 1, 3, 2, 4, 5])
        del t1586

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1588 = paddle._C_ops.reshape(t1587, t797)
        del t1587

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1589 = [t1542, t799, t539]
        del t1542

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1590 = paddle._C_ops.stack(t1589, 0)
        del t1589

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1591 = paddle._C_ops.reshape(t1588, t1590)
        del t1588, t1590

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1592 = paddle._C_ops.add(t1540, t1591)
        del t1540, t1591

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1593, t1594, t1595 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1592, t184, t185, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t185, t184

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1596 = paddle._C_ops.matmul(t1593, t186, False, False)
        del t1593, t186

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1597 = paddle._C_ops.add(t1596, t187)
        del t1596, t187

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1598 = paddle._C_ops.gelu(t1597, False)
        del t1597

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1599 = paddle._C_ops.matmul(t1598, t188, False, False)
        del t1598, t188

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1600 = paddle._C_ops.add(t1599, t189)
        del t1599, t189

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1601 = paddle._C_ops.add(t1592, t1600)
        del t1592, t1600

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1602 = paddle._C_ops.shape64(t1601)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1603 = paddle._C_ops.slice(t1602, [0], t305, t306, [1], [0])
        del t1602

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1604, t1605, t1606 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1601, t190, t191, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t191, t190

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1607 = [t1603, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1608 = paddle._C_ops.stack(t1607, 0)
        del t1607

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1609 = paddle._C_ops.reshape(t1604, t1608)
        del t1604, t1608

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1610 = paddle._C_ops.shape64(t1609)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1611 = paddle._C_ops.slice(t1610, [0], t305, t306, [1], [0])
        del t1610

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1612 = paddle._C_ops.roll(t1609, t408, [1, 2])
        del t1609

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1613 = paddle._C_ops.shape64(t1612)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1614 = paddle._C_ops.slice(t1613, [0], t305, t306, [1], [0])
        del t1613

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1615 = [t1614, t756, t332, t756, t332, t539]
        del t1614

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1616 = paddle._C_ops.stack(t1615, 0)
        del t1615

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1617 = paddle._C_ops.reshape(t1612, t1616)
        del t1612, t1616

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1618 = paddle._C_ops.transpose(t1617, [0, 1, 3, 2, 4, 5])
        del t1617

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1619 = paddle._C_ops.reshape(t1618, t761)
        del t1618

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1620 = paddle._C_ops.reshape(t1619, t763)
        del t1619

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1621 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1622 = paddle._C_ops.set_value_(
            t1621, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1621

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1623 = paddle._C_ops.set_value_(
            t1622, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1622

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1624 = paddle._C_ops.set_value_(
            t1623, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1623

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1625 = paddle._C_ops.set_value_(
            t1624, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1624

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1626 = paddle._C_ops.set_value_(
            t1625, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1625

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1627 = paddle._C_ops.set_value_(
            t1626, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1626

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1628 = paddle._C_ops.set_value_(
            t1627, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1627

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1629 = paddle._C_ops.set_value_(
            t1628, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1628

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1630 = paddle._C_ops.set_value_(
            t1629, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1629

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1631 = paddle._C_ops.reshape(t1630, t842)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1632 = paddle._C_ops.transpose(t1631, [0, 1, 3, 2, 4, 5])
        del t1631

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1633 = paddle._C_ops.reshape(t1632, t445)
        del t1632

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1634 = paddle._C_ops.reshape(t1633, t447)
        del t1633

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1635 = paddle._C_ops.unsqueeze(t1634, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1636 = paddle._C_ops.unsqueeze(t1634, t354)
        del t1634

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1637 = paddle._C_ops.subtract(t1635, t1636)
        del t1635, t1636

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1638 = paddle._C_ops.not_equal(t1637, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1639 = paddle._C_ops.where(t1638, t851, t1637)
        del t1638, t1637

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1640 = paddle._C_ops.equal(t1639, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1641 = paddle._C_ops.where(t1640, t854, t1639)
        del t1640, t1639

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1642 = paddle._C_ops.shape64(t1620)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1643 = paddle._C_ops.slice(t1642, [0], t305, t306, [1], [0])
        del t1642

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1644 = paddle._C_ops.matmul(t1620, t192, False, False)
        del t192, t1620

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1645 = paddle._C_ops.add(t1644, t193)
        del t1644, t193

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1646 = [t1643, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1647 = paddle._C_ops.stack(t1646, 0)
        del t1646

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1648 = paddle._C_ops.reshape(t1645, t1647)
        del t1645, t1647

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1649 = paddle._C_ops.transpose(t1648, [2, 0, 3, 1, 4])
        del t1648

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1650 = paddle._C_ops.slice(t1649, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1651 = paddle._C_ops.slice(t1649, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1652 = paddle._C_ops.slice(t1649, [0], t354, t356, [1], [0])
        del t1649

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1653 = paddle._C_ops.scale(t1650, t358, float("0"), True)
        del t1650

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1654 = paddle._C_ops.transpose(t1651, [0, 1, 3, 2])
        del t1651

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1655 = paddle._C_ops.matmul(t1653, t1654, False, False)
        del t1653, t1654

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1656 = paddle._C_ops.reshape(input31, t362)
        del input31

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1657 = paddle._C_ops.index_select(input32, t1656, 0)
        del input32, t1656

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1658 = paddle._C_ops.reshape(t1657, t365)
        del t1657

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1659 = paddle._C_ops.transpose(t1658, [2, 0, 1])
        del t1658

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1660 = paddle._C_ops.unsqueeze(t1659, t305)
        del t1659

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1661 = paddle._C_ops.add(t1655, t1660)
        del t1655, t1660

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1662 = paddle._C_ops.floor_divide(t1643, t876)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1663 = [t1662, t561, t750, t345, t345]
        del t1662

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1664 = paddle._C_ops.stack(t1663, 0)
        del t1663

        # pd_op.reshape: (-1x4x24x144x144xf32) <- (-1x24x144x144xf32, 5xi64)
        t1665 = paddle._C_ops.reshape(t1661, t1664)
        del t1661, t1664

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1666 = paddle._C_ops.unsqueeze(t1641, t306)
        del t1641

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1667 = paddle._C_ops.unsqueeze(t1666, t305)
        del t1666

        # pd_op.add: (-1x4x24x144x144xf32) <- (-1x4x24x144x144xf32, 1x4x1x144x144xf32)
        t1668 = paddle._C_ops.add(t1665, t1667)
        del t1665, t1667

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1669 = [t1643, t750, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1670 = paddle._C_ops.stack(t1669, 0)
        del t1669

        # pd_op.reshape: (-1x24x144x144xf32) <- (-1x4x24x144x144xf32, 4xi64)
        t1671 = paddle._C_ops.reshape(t1668, t1670)
        del t1668, t1670

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1672 = paddle._C_ops.softmax(t1671, -1)
        del t1671

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1673 = paddle._C_ops.matmul(t1672, t1652, False, False)
        del t1652, t1672

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1674 = paddle._C_ops.transpose(t1673, [0, 2, 1, 3])
        del t1673

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1675 = [t1643, t345, t539]
        del t1643

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1676 = paddle._C_ops.stack(t1675, 0)
        del t1675

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1677 = paddle._C_ops.reshape(t1674, t1676)
        del t1676, t1674

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1678 = paddle._C_ops.matmul(t1677, t194, False, False)
        del t194, t1677

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1679 = paddle._C_ops.add(t1678, t195)
        del t1678, t195

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1680 = paddle._C_ops.reshape(t1679, t761)
        del t1679

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1681 = paddle._C_ops.reshape(t1680, t794)
        del t1680

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1682 = paddle._C_ops.transpose(t1681, [0, 1, 3, 2, 4, 5])
        del t1681

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1683 = paddle._C_ops.reshape(t1682, t797)
        del t1682

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1684 = paddle._C_ops.roll(t1683, t503, [1, 2])
        del t1683

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1685 = [t1603, t799, t539]
        del t1603

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1686 = paddle._C_ops.stack(t1685, 0)
        del t1685

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1687 = paddle._C_ops.reshape(t1684, t1686)
        del t1684, t1686

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1688 = paddle._C_ops.add(t1601, t1687)
        del t1601, t1687

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1689, t1690, t1691 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1688, t196, t197, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t197, t196

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1692 = paddle._C_ops.matmul(t1689, t198, False, False)
        del t1689, t198

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1693 = paddle._C_ops.add(t1692, t199)
        del t1692, t199

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1694 = paddle._C_ops.gelu(t1693, False)
        del t1693

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1695 = paddle._C_ops.matmul(t1694, t200, False, False)
        del t1694, t200

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1696 = paddle._C_ops.add(t1695, t201)
        del t1695, t201

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1697 = paddle._C_ops.add(t1688, t1696)
        del t1688, t1696

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1698 = paddle._C_ops.shape64(t1697)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1699 = paddle._C_ops.slice(t1698, [0], t305, t306, [1], [0])
        del t1698

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1700, t1701, t1702 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1697, t202, t203, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t203, t202

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1703 = [t1699, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1704 = paddle._C_ops.stack(t1703, 0)
        del t1703

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1705 = paddle._C_ops.reshape(t1700, t1704)
        del t1700, t1704

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1706 = paddle._C_ops.shape64(t1705)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1707 = paddle._C_ops.slice(t1706, [0], t305, t306, [1], [0])
        del t1706

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1708 = [t1707, t756, t332, t756, t332, t539]
        del t1707

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1709 = paddle._C_ops.stack(t1708, 0)
        del t1708

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1710 = paddle._C_ops.reshape(t1705, t1709)
        del t1705, t1709

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1711 = paddle._C_ops.transpose(t1710, [0, 1, 3, 2, 4, 5])
        del t1710

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1712 = paddle._C_ops.reshape(t1711, t761)
        del t1711

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1713 = paddle._C_ops.reshape(t1712, t763)
        del t1712

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1714 = paddle._C_ops.shape64(t1713)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1715 = paddle._C_ops.slice(t1714, [0], t305, t306, [1], [0])
        del t1714

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1716 = paddle._C_ops.matmul(t1713, t204, False, False)
        del t204, t1713

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1717 = paddle._C_ops.add(t1716, t205)
        del t1716, t205

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1718 = [t1715, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1719 = paddle._C_ops.stack(t1718, 0)
        del t1718

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1720 = paddle._C_ops.reshape(t1717, t1719)
        del t1717, t1719

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1721 = paddle._C_ops.transpose(t1720, [2, 0, 3, 1, 4])
        del t1720

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1722 = paddle._C_ops.slice(t1721, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1723 = paddle._C_ops.slice(t1721, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1724 = paddle._C_ops.slice(t1721, [0], t354, t356, [1], [0])
        del t1721

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1725 = paddle._C_ops.scale(t1722, t358, float("0"), True)
        del t1722

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1726 = paddle._C_ops.transpose(t1723, [0, 1, 3, 2])
        del t1723

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1727 = paddle._C_ops.matmul(t1725, t1726, False, False)
        del t1725, t1726

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1728 = paddle._C_ops.reshape(input33, t362)
        del input33

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1729 = paddle._C_ops.index_select(input34, t1728, 0)
        del input34, t1728

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1730 = paddle._C_ops.reshape(t1729, t365)
        del t1729

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1731 = paddle._C_ops.transpose(t1730, [2, 0, 1])
        del t1730

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1732 = paddle._C_ops.unsqueeze(t1731, t305)
        del t1731

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1733 = paddle._C_ops.add(t1727, t1732)
        del t1727, t1732

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1734 = paddle._C_ops.softmax(t1733, -1)
        del t1733

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1735 = paddle._C_ops.matmul(t1734, t1724, False, False)
        del t1724, t1734

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1736 = paddle._C_ops.transpose(t1735, [0, 2, 1, 3])
        del t1735

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1737 = [t1715, t345, t539]
        del t1715

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1738 = paddle._C_ops.stack(t1737, 0)
        del t1737

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1739 = paddle._C_ops.reshape(t1736, t1738)
        del t1738, t1736

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1740 = paddle._C_ops.matmul(t1739, t206, False, False)
        del t206, t1739

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1741 = paddle._C_ops.add(t1740, t207)
        del t1740, t207

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1742 = paddle._C_ops.reshape(t1741, t761)
        del t1741

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1743 = paddle._C_ops.reshape(t1742, t794)
        del t1742

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1744 = paddle._C_ops.transpose(t1743, [0, 1, 3, 2, 4, 5])
        del t1743

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1745 = paddle._C_ops.reshape(t1744, t797)
        del t1744

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1746 = [t1699, t799, t539]
        del t1699

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1747 = paddle._C_ops.stack(t1746, 0)
        del t1746

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1748 = paddle._C_ops.reshape(t1745, t1747)
        del t1745, t1747

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1749 = paddle._C_ops.add(t1697, t1748)
        del t1697, t1748

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1750, t1751, t1752 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1749, t208, t209, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t209, t208

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1753 = paddle._C_ops.matmul(t1750, t210, False, False)
        del t1750, t210

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1754 = paddle._C_ops.add(t1753, t211)
        del t1753, t211

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1755 = paddle._C_ops.gelu(t1754, False)
        del t1754

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1756 = paddle._C_ops.matmul(t1755, t212, False, False)
        del t1755, t212

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1757 = paddle._C_ops.add(t1756, t213)
        del t1756, t213

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1758 = paddle._C_ops.add(t1749, t1757)
        del t1749, t1757

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1759 = paddle._C_ops.shape64(t1758)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1760 = paddle._C_ops.slice(t1759, [0], t305, t306, [1], [0])
        del t1759

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1761, t1762, t1763 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1758, t214, t215, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t215, t214

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1764 = [t1760, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1765 = paddle._C_ops.stack(t1764, 0)
        del t1764

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1766 = paddle._C_ops.reshape(t1761, t1765)
        del t1761, t1765

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1767 = paddle._C_ops.shape64(t1766)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1768 = paddle._C_ops.slice(t1767, [0], t305, t306, [1], [0])
        del t1767

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1769 = paddle._C_ops.roll(t1766, t408, [1, 2])
        del t1766

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1770 = paddle._C_ops.shape64(t1769)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1771 = paddle._C_ops.slice(t1770, [0], t305, t306, [1], [0])
        del t1770

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1772 = [t1771, t756, t332, t756, t332, t539]
        del t1771

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1773 = paddle._C_ops.stack(t1772, 0)
        del t1772

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1774 = paddle._C_ops.reshape(t1769, t1773)
        del t1769, t1773

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1775 = paddle._C_ops.transpose(t1774, [0, 1, 3, 2, 4, 5])
        del t1774

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1776 = paddle._C_ops.reshape(t1775, t761)
        del t1775

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1777 = paddle._C_ops.reshape(t1776, t763)
        del t1776

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1778 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1779 = paddle._C_ops.set_value_(
            t1778, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1778

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1780 = paddle._C_ops.set_value_(
            t1779, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1779

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1781 = paddle._C_ops.set_value_(
            t1780, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1780

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1782 = paddle._C_ops.set_value_(
            t1781, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1781

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1783 = paddle._C_ops.set_value_(
            t1782, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1782

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1784 = paddle._C_ops.set_value_(
            t1783, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1783

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1785 = paddle._C_ops.set_value_(
            t1784, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1784

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1786 = paddle._C_ops.set_value_(
            t1785, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1785

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1787 = paddle._C_ops.set_value_(
            t1786, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1786

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1788 = paddle._C_ops.reshape(t1787, t842)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1789 = paddle._C_ops.transpose(t1788, [0, 1, 3, 2, 4, 5])
        del t1788

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1790 = paddle._C_ops.reshape(t1789, t445)
        del t1789

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1791 = paddle._C_ops.reshape(t1790, t447)
        del t1790

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1792 = paddle._C_ops.unsqueeze(t1791, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1793 = paddle._C_ops.unsqueeze(t1791, t354)
        del t1791

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1794 = paddle._C_ops.subtract(t1792, t1793)
        del t1792, t1793

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1795 = paddle._C_ops.not_equal(t1794, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1796 = paddle._C_ops.where(t1795, t851, t1794)
        del t1795, t1794

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1797 = paddle._C_ops.equal(t1796, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1798 = paddle._C_ops.where(t1797, t854, t1796)
        del t1797, t1796

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1799 = paddle._C_ops.shape64(t1777)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1800 = paddle._C_ops.slice(t1799, [0], t305, t306, [1], [0])
        del t1799

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1801 = paddle._C_ops.matmul(t1777, t216, False, False)
        del t216, t1777

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1802 = paddle._C_ops.add(t1801, t217)
        del t1801, t217

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1803 = [t1800, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1804 = paddle._C_ops.stack(t1803, 0)
        del t1803

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1805 = paddle._C_ops.reshape(t1802, t1804)
        del t1802, t1804

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1806 = paddle._C_ops.transpose(t1805, [2, 0, 3, 1, 4])
        del t1805

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1807 = paddle._C_ops.slice(t1806, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1808 = paddle._C_ops.slice(t1806, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1809 = paddle._C_ops.slice(t1806, [0], t354, t356, [1], [0])
        del t1806

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1810 = paddle._C_ops.scale(t1807, t358, float("0"), True)
        del t1807

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1811 = paddle._C_ops.transpose(t1808, [0, 1, 3, 2])
        del t1808

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1812 = paddle._C_ops.matmul(t1810, t1811, False, False)
        del t1810, t1811

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1813 = paddle._C_ops.reshape(input35, t362)
        del input35

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1814 = paddle._C_ops.index_select(input36, t1813, 0)
        del input36, t1813

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1815 = paddle._C_ops.reshape(t1814, t365)
        del t1814

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1816 = paddle._C_ops.transpose(t1815, [2, 0, 1])
        del t1815

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1817 = paddle._C_ops.unsqueeze(t1816, t305)
        del t1816

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1818 = paddle._C_ops.add(t1812, t1817)
        del t1812, t1817

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1819 = paddle._C_ops.floor_divide(t1800, t876)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1820 = [t1819, t561, t750, t345, t345]
        del t1819

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1821 = paddle._C_ops.stack(t1820, 0)
        del t1820

        # pd_op.reshape: (-1x4x24x144x144xf32) <- (-1x24x144x144xf32, 5xi64)
        t1822 = paddle._C_ops.reshape(t1818, t1821)
        del t1818, t1821

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1823 = paddle._C_ops.unsqueeze(t1798, t306)
        del t1798

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1824 = paddle._C_ops.unsqueeze(t1823, t305)
        del t1823

        # pd_op.add: (-1x4x24x144x144xf32) <- (-1x4x24x144x144xf32, 1x4x1x144x144xf32)
        t1825 = paddle._C_ops.add(t1822, t1824)
        del t1822, t1824

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1826 = [t1800, t750, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1827 = paddle._C_ops.stack(t1826, 0)
        del t1826

        # pd_op.reshape: (-1x24x144x144xf32) <- (-1x4x24x144x144xf32, 4xi64)
        t1828 = paddle._C_ops.reshape(t1825, t1827)
        del t1825, t1827

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1829 = paddle._C_ops.softmax(t1828, -1)
        del t1828

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1830 = paddle._C_ops.matmul(t1829, t1809, False, False)
        del t1809, t1829

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1831 = paddle._C_ops.transpose(t1830, [0, 2, 1, 3])
        del t1830

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1832 = [t1800, t345, t539]
        del t1800

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1833 = paddle._C_ops.stack(t1832, 0)
        del t1832

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1834 = paddle._C_ops.reshape(t1831, t1833)
        del t1833, t1831

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1835 = paddle._C_ops.matmul(t1834, t218, False, False)
        del t218, t1834

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1836 = paddle._C_ops.add(t1835, t219)
        del t1835, t219

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1837 = paddle._C_ops.reshape(t1836, t761)
        del t1836

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1838 = paddle._C_ops.reshape(t1837, t794)
        del t1837

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1839 = paddle._C_ops.transpose(t1838, [0, 1, 3, 2, 4, 5])
        del t1838

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1840 = paddle._C_ops.reshape(t1839, t797)
        del t1839

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1841 = paddle._C_ops.roll(t1840, t503, [1, 2])
        del t1840

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1842 = [t1760, t799, t539]
        del t1760

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1843 = paddle._C_ops.stack(t1842, 0)
        del t1842

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1844 = paddle._C_ops.reshape(t1841, t1843)
        del t1841, t1843

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1845 = paddle._C_ops.add(t1758, t1844)
        del t1758, t1844

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1846, t1847, t1848 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1845, t220, t221, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t221, t220

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1849 = paddle._C_ops.matmul(t1846, t222, False, False)
        del t1846, t222

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1850 = paddle._C_ops.add(t1849, t223)
        del t1849, t223

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1851 = paddle._C_ops.gelu(t1850, False)
        del t1850

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1852 = paddle._C_ops.matmul(t1851, t224, False, False)
        del t1851, t224

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1853 = paddle._C_ops.add(t1852, t225)
        del t1852, t225

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1854 = paddle._C_ops.add(t1845, t1853)
        del t1845, t1853

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1855 = paddle._C_ops.shape64(t1854)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1856 = paddle._C_ops.slice(t1855, [0], t305, t306, [1], [0])
        del t1855

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1857, t1858, t1859 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1854, t226, t227, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t227, t226

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1860 = [t1856, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1861 = paddle._C_ops.stack(t1860, 0)
        del t1860

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1862 = paddle._C_ops.reshape(t1857, t1861)
        del t1857, t1861

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1863 = paddle._C_ops.shape64(t1862)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1864 = paddle._C_ops.slice(t1863, [0], t305, t306, [1], [0])
        del t1863

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1865 = [t1864, t756, t332, t756, t332, t539]
        del t1864

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1866 = paddle._C_ops.stack(t1865, 0)
        del t1865

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1867 = paddle._C_ops.reshape(t1862, t1866)
        del t1862, t1866

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1868 = paddle._C_ops.transpose(t1867, [0, 1, 3, 2, 4, 5])
        del t1867

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1869 = paddle._C_ops.reshape(t1868, t761)
        del t1868

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1870 = paddle._C_ops.reshape(t1869, t763)
        del t1869

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1871 = paddle._C_ops.shape64(t1870)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1872 = paddle._C_ops.slice(t1871, [0], t305, t306, [1], [0])
        del t1871

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1873 = paddle._C_ops.matmul(t1870, t228, False, False)
        del t228, t1870

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1874 = paddle._C_ops.add(t1873, t229)
        del t1873, t229

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1875 = [t1872, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1876 = paddle._C_ops.stack(t1875, 0)
        del t1875

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1877 = paddle._C_ops.reshape(t1874, t1876)
        del t1874, t1876

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1878 = paddle._C_ops.transpose(t1877, [2, 0, 3, 1, 4])
        del t1877

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1879 = paddle._C_ops.slice(t1878, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1880 = paddle._C_ops.slice(t1878, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1881 = paddle._C_ops.slice(t1878, [0], t354, t356, [1], [0])
        del t1878

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1882 = paddle._C_ops.scale(t1879, t358, float("0"), True)
        del t1879

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1883 = paddle._C_ops.transpose(t1880, [0, 1, 3, 2])
        del t1880

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1884 = paddle._C_ops.matmul(t1882, t1883, False, False)
        del t1882, t1883

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1885 = paddle._C_ops.reshape(input37, t362)
        del input37

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1886 = paddle._C_ops.index_select(input38, t1885, 0)
        del input38, t1885

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1887 = paddle._C_ops.reshape(t1886, t365)
        del t1886

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1888 = paddle._C_ops.transpose(t1887, [2, 0, 1])
        del t1887

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1889 = paddle._C_ops.unsqueeze(t1888, t305)
        del t1888

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1890 = paddle._C_ops.add(t1884, t1889)
        del t1884, t1889

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1891 = paddle._C_ops.softmax(t1890, -1)
        del t1890

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1892 = paddle._C_ops.matmul(t1891, t1881, False, False)
        del t1881, t1891

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1893 = paddle._C_ops.transpose(t1892, [0, 2, 1, 3])
        del t1892

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1894 = [t1872, t345, t539]
        del t1872

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1895 = paddle._C_ops.stack(t1894, 0)
        del t1894

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1896 = paddle._C_ops.reshape(t1893, t1895)
        del t1895, t1893

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1897 = paddle._C_ops.matmul(t1896, t230, False, False)
        del t230, t1896

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1898 = paddle._C_ops.add(t1897, t231)
        del t1897, t231

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1899 = paddle._C_ops.reshape(t1898, t761)
        del t1898

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1900 = paddle._C_ops.reshape(t1899, t794)
        del t1899

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1901 = paddle._C_ops.transpose(t1900, [0, 1, 3, 2, 4, 5])
        del t1900

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1902 = paddle._C_ops.reshape(t1901, t797)
        del t1901

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1903 = [t1856, t799, t539]
        del t1856

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1904 = paddle._C_ops.stack(t1903, 0)
        del t1903

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t1905 = paddle._C_ops.reshape(t1902, t1904)
        del t1902, t1904

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1906 = paddle._C_ops.add(t1854, t1905)
        del t1854, t1905

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1907, t1908, t1909 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1906, t232, t233, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t233, t232

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t1910 = paddle._C_ops.matmul(t1907, t234, False, False)
        del t1907, t234

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t1911 = paddle._C_ops.add(t1910, t235)
        del t1910, t235

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t1912 = paddle._C_ops.gelu(t1911, False)
        del t1911

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t1913 = paddle._C_ops.matmul(t1912, t236, False, False)
        del t1912, t236

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t1914 = paddle._C_ops.add(t1913, t237)
        del t1913, t237

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t1915 = paddle._C_ops.add(t1906, t1914)
        del t1906, t1914

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t1916 = paddle._C_ops.shape64(t1915)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1917 = paddle._C_ops.slice(t1916, [0], t305, t306, [1], [0])
        del t1916

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t1918, t1919, t1920 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1915, t238, t239, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t239, t238

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1921 = [t1917, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1922 = paddle._C_ops.stack(t1921, 0)
        del t1921

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t1923 = paddle._C_ops.reshape(t1918, t1922)
        del t1918, t1922

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1924 = paddle._C_ops.shape64(t1923)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1925 = paddle._C_ops.slice(t1924, [0], t305, t306, [1], [0])
        del t1924

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1926 = paddle._C_ops.roll(t1923, t408, [1, 2])
        del t1923

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t1927 = paddle._C_ops.shape64(t1926)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1928 = paddle._C_ops.slice(t1927, [0], t305, t306, [1], [0])
        del t1927

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1929 = [t1928, t756, t332, t756, t332, t539]
        del t1928

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1930 = paddle._C_ops.stack(t1929, 0)
        del t1929

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t1931 = paddle._C_ops.reshape(t1926, t1930)
        del t1926, t1930

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t1932 = paddle._C_ops.transpose(t1931, [0, 1, 3, 2, 4, 5])
        del t1931

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t1933 = paddle._C_ops.reshape(t1932, t761)
        del t1932

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t1934 = paddle._C_ops.reshape(t1933, t763)
        del t1933

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1935 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1936 = paddle._C_ops.set_value_(
            t1935, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1935

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1937 = paddle._C_ops.set_value_(
            t1936, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1936

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1938 = paddle._C_ops.set_value_(
            t1937, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1937

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1939 = paddle._C_ops.set_value_(
            t1938, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1938

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1940 = paddle._C_ops.set_value_(
            t1939, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1939

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1941 = paddle._C_ops.set_value_(
            t1940, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1940

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1942 = paddle._C_ops.set_value_(
            t1941, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1941

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1943 = paddle._C_ops.set_value_(
            t1942, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1942

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1944 = paddle._C_ops.set_value_(
            t1943, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1943

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1945 = paddle._C_ops.reshape(t1944, t842)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1946 = paddle._C_ops.transpose(t1945, [0, 1, 3, 2, 4, 5])
        del t1945

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1947 = paddle._C_ops.reshape(t1946, t445)
        del t1946

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1948 = paddle._C_ops.reshape(t1947, t447)
        del t1947

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1949 = paddle._C_ops.unsqueeze(t1948, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1950 = paddle._C_ops.unsqueeze(t1948, t354)
        del t1948

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1951 = paddle._C_ops.subtract(t1949, t1950)
        del t1949, t1950

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1952 = paddle._C_ops.not_equal(t1951, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1953 = paddle._C_ops.where(t1952, t851, t1951)
        del t1952, t1951

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1954 = paddle._C_ops.equal(t1953, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1955 = paddle._C_ops.where(t1954, t854, t1953)
        del t1954, t1953

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t1956 = paddle._C_ops.shape64(t1934)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1957 = paddle._C_ops.slice(t1956, [0], t305, t306, [1], [0])
        del t1956

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t1958 = paddle._C_ops.matmul(t1934, t240, False, False)
        del t240, t1934

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t1959 = paddle._C_ops.add(t1958, t241)
        del t1958, t241

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1960 = [t1957, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1961 = paddle._C_ops.stack(t1960, 0)
        del t1960

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t1962 = paddle._C_ops.reshape(t1959, t1961)
        del t1959, t1961

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t1963 = paddle._C_ops.transpose(t1962, [2, 0, 3, 1, 4])
        del t1962

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1964 = paddle._C_ops.slice(t1963, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1965 = paddle._C_ops.slice(t1963, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t1966 = paddle._C_ops.slice(t1963, [0], t354, t356, [1], [0])
        del t1963

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t1967 = paddle._C_ops.scale(t1964, t358, float("0"), True)
        del t1964

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t1968 = paddle._C_ops.transpose(t1965, [0, 1, 3, 2])
        del t1965

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t1969 = paddle._C_ops.matmul(t1967, t1968, False, False)
        del t1967, t1968

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1970 = paddle._C_ops.reshape(input39, t362)
        del input39

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t1971 = paddle._C_ops.index_select(input40, t1970, 0)
        del input40, t1970

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t1972 = paddle._C_ops.reshape(t1971, t365)
        del t1971

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t1973 = paddle._C_ops.transpose(t1972, [2, 0, 1])
        del t1972

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t1974 = paddle._C_ops.unsqueeze(t1973, t305)
        del t1973

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t1975 = paddle._C_ops.add(t1969, t1974)
        del t1969, t1974

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1976 = paddle._C_ops.floor_divide(t1957, t876)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1977 = [t1976, t561, t750, t345, t345]
        del t1976

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1978 = paddle._C_ops.stack(t1977, 0)
        del t1977

        # pd_op.reshape: (-1x4x24x144x144xf32) <- (-1x24x144x144xf32, 5xi64)
        t1979 = paddle._C_ops.reshape(t1975, t1978)
        del t1975, t1978

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1980 = paddle._C_ops.unsqueeze(t1955, t306)
        del t1955

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1981 = paddle._C_ops.unsqueeze(t1980, t305)
        del t1980

        # pd_op.add: (-1x4x24x144x144xf32) <- (-1x4x24x144x144xf32, 1x4x1x144x144xf32)
        t1982 = paddle._C_ops.add(t1979, t1981)
        del t1979, t1981

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1983 = [t1957, t750, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1984 = paddle._C_ops.stack(t1983, 0)
        del t1983

        # pd_op.reshape: (-1x24x144x144xf32) <- (-1x4x24x144x144xf32, 4xi64)
        t1985 = paddle._C_ops.reshape(t1982, t1984)
        del t1982, t1984

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t1986 = paddle._C_ops.softmax(t1985, -1)
        del t1985

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t1987 = paddle._C_ops.matmul(t1986, t1966, False, False)
        del t1966, t1986

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t1988 = paddle._C_ops.transpose(t1987, [0, 2, 1, 3])
        del t1987

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1989 = [t1957, t345, t539]
        del t1957

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1990 = paddle._C_ops.stack(t1989, 0)
        del t1989

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t1991 = paddle._C_ops.reshape(t1988, t1990)
        del t1990, t1988

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t1992 = paddle._C_ops.matmul(t1991, t242, False, False)
        del t242, t1991

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t1993 = paddle._C_ops.add(t1992, t243)
        del t1992, t243

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t1994 = paddle._C_ops.reshape(t1993, t761)
        del t1993

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t1995 = paddle._C_ops.reshape(t1994, t794)
        del t1994

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t1996 = paddle._C_ops.transpose(t1995, [0, 1, 3, 2, 4, 5])
        del t1995

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t1997 = paddle._C_ops.reshape(t1996, t797)
        del t1996

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t1998 = paddle._C_ops.roll(t1997, t503, [1, 2])
        del t1997

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1999 = [t1917, t799, t539]
        del t1917

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2000 = paddle._C_ops.stack(t1999, 0)
        del t1999

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t2001 = paddle._C_ops.reshape(t1998, t2000)
        del t1998, t2000

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t2002 = paddle._C_ops.add(t1915, t2001)
        del t1915, t2001

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t2003, t2004, t2005 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2002, t244, t245, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t245, t244

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t2006 = paddle._C_ops.matmul(t2003, t246, False, False)
        del t2003, t246

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t2007 = paddle._C_ops.add(t2006, t247)
        del t2006, t247

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t2008 = paddle._C_ops.gelu(t2007, False)
        del t2007

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t2009 = paddle._C_ops.matmul(t2008, t248, False, False)
        del t2008, t248

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t2010 = paddle._C_ops.add(t2009, t249)
        del t2009, t249

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t2011 = paddle._C_ops.add(t2002, t2010)
        del t2002, t2010

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t2012 = paddle._C_ops.shape64(t2011)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2013 = paddle._C_ops.slice(t2012, [0], t305, t306, [1], [0])
        del t2012

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t2014, t2015, t2016 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2011, t250, t251, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t251, t250

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2017 = [t2013, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2018 = paddle._C_ops.stack(t2017, 0)
        del t2017

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t2019 = paddle._C_ops.reshape(t2014, t2018)
        del t2014, t2018

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t2020 = paddle._C_ops.shape64(t2019)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2021 = paddle._C_ops.slice(t2020, [0], t305, t306, [1], [0])
        del t2020

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2022 = [t2021, t756, t332, t756, t332, t539]
        del t2021

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2023 = paddle._C_ops.stack(t2022, 0)
        del t2022

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t2024 = paddle._C_ops.reshape(t2019, t2023)
        del t2019, t2023

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t2025 = paddle._C_ops.transpose(t2024, [0, 1, 3, 2, 4, 5])
        del t2024

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t2026 = paddle._C_ops.reshape(t2025, t761)
        del t2025

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t2027 = paddle._C_ops.reshape(t2026, t763)
        del t2026

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t2028 = paddle._C_ops.shape64(t2027)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2029 = paddle._C_ops.slice(t2028, [0], t305, t306, [1], [0])
        del t2028

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t2030 = paddle._C_ops.matmul(t2027, t252, False, False)
        del t252, t2027

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t2031 = paddle._C_ops.add(t2030, t253)
        del t2030, t253

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2032 = [t2029, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2033 = paddle._C_ops.stack(t2032, 0)
        del t2032

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t2034 = paddle._C_ops.reshape(t2031, t2033)
        del t2031, t2033

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t2035 = paddle._C_ops.transpose(t2034, [2, 0, 3, 1, 4])
        del t2034

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t2036 = paddle._C_ops.slice(t2035, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t2037 = paddle._C_ops.slice(t2035, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t2038 = paddle._C_ops.slice(t2035, [0], t354, t356, [1], [0])
        del t2035

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t2039 = paddle._C_ops.scale(t2036, t358, float("0"), True)
        del t2036

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t2040 = paddle._C_ops.transpose(t2037, [0, 1, 3, 2])
        del t2037

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t2041 = paddle._C_ops.matmul(t2039, t2040, False, False)
        del t2039, t2040

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t2042 = paddle._C_ops.reshape(input41, t362)
        del input41

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t2043 = paddle._C_ops.index_select(input42, t2042, 0)
        del input42, t2042

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t2044 = paddle._C_ops.reshape(t2043, t365)
        del t2043

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t2045 = paddle._C_ops.transpose(t2044, [2, 0, 1])
        del t2044

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t2046 = paddle._C_ops.unsqueeze(t2045, t305)
        del t2045

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t2047 = paddle._C_ops.add(t2041, t2046)
        del t2041, t2046

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t2048 = paddle._C_ops.softmax(t2047, -1)
        del t2047

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t2049 = paddle._C_ops.matmul(t2048, t2038, False, False)
        del t2038, t2048

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t2050 = paddle._C_ops.transpose(t2049, [0, 2, 1, 3])
        del t2049

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2051 = [t2029, t345, t539]
        del t2029

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2052 = paddle._C_ops.stack(t2051, 0)
        del t2051

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t2053 = paddle._C_ops.reshape(t2050, t2052)
        del t2052, t2050

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t2054 = paddle._C_ops.matmul(t2053, t254, False, False)
        del t254, t2053

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t2055 = paddle._C_ops.add(t2054, t255)
        del t2054, t255

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t2056 = paddle._C_ops.reshape(t2055, t761)
        del t2055

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t2057 = paddle._C_ops.reshape(t2056, t794)
        del t2056

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t2058 = paddle._C_ops.transpose(t2057, [0, 1, 3, 2, 4, 5])
        del t2057

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t2059 = paddle._C_ops.reshape(t2058, t797)
        del t2058

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2060 = [t2013, t799, t539]
        del t2013

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2061 = paddle._C_ops.stack(t2060, 0)
        del t2060

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t2062 = paddle._C_ops.reshape(t2059, t2061)
        del t2059, t2061

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t2063 = paddle._C_ops.add(t2011, t2062)
        del t2011, t2062

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t2064, t2065, t2066 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2063, t256, t257, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t257, t256

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t2067 = paddle._C_ops.matmul(t2064, t258, False, False)
        del t2064, t258

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t2068 = paddle._C_ops.add(t2067, t259)
        del t2067, t259

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t2069 = paddle._C_ops.gelu(t2068, False)
        del t2068

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t2070 = paddle._C_ops.matmul(t2069, t260, False, False)
        del t2069, t260

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t2071 = paddle._C_ops.add(t2070, t261)
        del t2070, t261

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t2072 = paddle._C_ops.add(t2063, t2071)
        del t2063, t2071

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t2073 = paddle._C_ops.shape64(t2072)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2074 = paddle._C_ops.slice(t2073, [0], t305, t306, [1], [0])
        del t2073

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t2075, t2076, t2077 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2072, t262, t263, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t263, t262

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2078 = [t2074, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2079 = paddle._C_ops.stack(t2078, 0)
        del t2078

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t2080 = paddle._C_ops.reshape(t2075, t2079)
        del t2075, t2079

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t2081 = paddle._C_ops.shape64(t2080)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2082 = paddle._C_ops.slice(t2081, [0], t305, t306, [1], [0])
        del t2081

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t2083 = paddle._C_ops.roll(t2080, t408, [1, 2])
        del t2080

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t2084 = paddle._C_ops.shape64(t2083)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2085 = paddle._C_ops.slice(t2084, [0], t305, t306, [1], [0])
        del t2084

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2086 = [t2085, t756, t332, t756, t332, t539]
        del t756, t2085

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2087 = paddle._C_ops.stack(t2086, 0)
        del t2086

        # pd_op.reshape: (-1x2x12x2x12x768xf32) <- (-1x24x24x768xf32, 6xi64)
        t2088 = paddle._C_ops.reshape(t2083, t2087)
        del t2083, t2087

        # pd_op.transpose: (-1x2x2x12x12x768xf32) <- (-1x2x12x2x12x768xf32)
        t2089 = paddle._C_ops.transpose(t2088, [0, 1, 3, 2, 4, 5])
        del t2088

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x2x2x12x12x768xf32, 4xi64)
        t2090 = paddle._C_ops.reshape(t2089, t761)
        del t2089

        # pd_op.reshape: (-1x144x768xf32) <- (-1x12x12x768xf32, 3xi64)
        t2091 = paddle._C_ops.reshape(t2090, t763)
        del t763, t2090

        # pd_op.full: (1x24x24x1xf32) <- ()
        t2092 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2093 = paddle._C_ops.set_value_(
            t2092, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t2092

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2094 = paddle._C_ops.set_value_(
            t2093, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t2093

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2095 = paddle._C_ops.set_value_(
            t2094, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t2094

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2096 = paddle._C_ops.set_value_(
            t2095, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t2095

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2097 = paddle._C_ops.set_value_(
            t2096, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t2096

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2098 = paddle._C_ops.set_value_(
            t2097, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t2097

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2099 = paddle._C_ops.set_value_(
            t2098, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t2098

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2100 = paddle._C_ops.set_value_(
            t2099, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t2099

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2101 = paddle._C_ops.set_value_(
            t2100, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t2100

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t2102 = paddle._C_ops.reshape(t2101, t842)
        del t842

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t2103 = paddle._C_ops.transpose(t2102, [0, 1, 3, 2, 4, 5])
        del t2102

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t2104 = paddle._C_ops.reshape(t2103, t445)
        del t2103

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t2105 = paddle._C_ops.reshape(t2104, t447)
        del t2104

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t2106 = paddle._C_ops.unsqueeze(t2105, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t2107 = paddle._C_ops.unsqueeze(t2105, t354)
        del t2105

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t2108 = paddle._C_ops.subtract(t2106, t2107)
        del t2106, t2107

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t2109 = paddle._C_ops.not_equal(t2108, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t2110 = paddle._C_ops.where(t2109, t851, t2108)
        del t851, t2109, t2108

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t2111 = paddle._C_ops.equal(t2110, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t2112 = paddle._C_ops.where(t2111, t854, t2110)
        del t2111, t854, t2110

        # pd_op.shape64: (3xi64) <- (-1x144x768xf32)
        t2113 = paddle._C_ops.shape64(t2091)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2114 = paddle._C_ops.slice(t2113, [0], t305, t306, [1], [0])
        del t2113

        # pd_op.matmul: (-1x144x2304xf32) <- (-1x144x768xf32, 768x2304xf32)
        t2115 = paddle._C_ops.matmul(t2091, t264, False, False)
        del t264, t2091

        # pd_op.add: (-1x144x2304xf32) <- (-1x144x2304xf32, 2304xf32)
        t2116 = paddle._C_ops.add(t2115, t265)
        del t2115, t265

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2117 = [t2114, t345, t346, t750, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2118 = paddle._C_ops.stack(t2117, 0)
        del t2117

        # pd_op.reshape: (-1x144x3x24x32xf32) <- (-1x144x2304xf32, 5xi64)
        t2119 = paddle._C_ops.reshape(t2116, t2118)
        del t2116, t2118

        # pd_op.transpose: (3x-1x24x144x32xf32) <- (-1x144x3x24x32xf32)
        t2120 = paddle._C_ops.transpose(t2119, [2, 0, 3, 1, 4])
        del t2119

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t2121 = paddle._C_ops.slice(t2120, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t2122 = paddle._C_ops.slice(t2120, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x24x144x32xf32) <- (3x-1x24x144x32xf32, 1xi64, 1xi64)
        t2123 = paddle._C_ops.slice(t2120, [0], t354, t356, [1], [0])
        del t2120

        # pd_op.scale: (-1x24x144x32xf32) <- (-1x24x144x32xf32, 1xf32)
        t2124 = paddle._C_ops.scale(t2121, t358, float("0"), True)
        del t2121

        # pd_op.transpose: (-1x24x32x144xf32) <- (-1x24x144x32xf32)
        t2125 = paddle._C_ops.transpose(t2122, [0, 1, 3, 2])
        del t2122

        # pd_op.matmul: (-1x24x144x144xf32) <- (-1x24x144x32xf32, -1x24x32x144xf32)
        t2126 = paddle._C_ops.matmul(t2124, t2125, False, False)
        del t2124, t2125

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t2127 = paddle._C_ops.reshape(input43, t362)
        del input43

        # pd_op.index_select: (20736x24xf32) <- (529x24xf32, 20736xi64)
        t2128 = paddle._C_ops.index_select(input44, t2127, 0)
        del input44, t2127

        # pd_op.reshape: (144x144x24xf32) <- (20736x24xf32, 3xi64)
        t2129 = paddle._C_ops.reshape(t2128, t365)
        del t2128

        # pd_op.transpose: (24x144x144xf32) <- (144x144x24xf32)
        t2130 = paddle._C_ops.transpose(t2129, [2, 0, 1])
        del t2129

        # pd_op.unsqueeze: (1x24x144x144xf32) <- (24x144x144xf32, 1xi64)
        t2131 = paddle._C_ops.unsqueeze(t2130, t305)
        del t2130

        # pd_op.add: (-1x24x144x144xf32) <- (-1x24x144x144xf32, 1x24x144x144xf32)
        t2132 = paddle._C_ops.add(t2126, t2131)
        del t2126, t2131

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t2133 = paddle._C_ops.floor_divide(t2114, t876)
        del t876

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2134 = [t2133, t561, t750, t345, t345]
        del t2133, t561

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2135 = paddle._C_ops.stack(t2134, 0)
        del t2134

        # pd_op.reshape: (-1x4x24x144x144xf32) <- (-1x24x144x144xf32, 5xi64)
        t2136 = paddle._C_ops.reshape(t2132, t2135)
        del t2132, t2135

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t2137 = paddle._C_ops.unsqueeze(t2112, t306)
        del t2112

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t2138 = paddle._C_ops.unsqueeze(t2137, t305)
        del t2137

        # pd_op.add: (-1x4x24x144x144xf32) <- (-1x4x24x144x144xf32, 1x4x1x144x144xf32)
        t2139 = paddle._C_ops.add(t2136, t2138)
        del t2136, t2138

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2140 = [t2114, t750, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2141 = paddle._C_ops.stack(t2140, 0)
        del t2140

        # pd_op.reshape: (-1x24x144x144xf32) <- (-1x4x24x144x144xf32, 4xi64)
        t2142 = paddle._C_ops.reshape(t2139, t2141)
        del t2139, t2141

        # pd_op.softmax: (-1x24x144x144xf32) <- (-1x24x144x144xf32)
        t2143 = paddle._C_ops.softmax(t2142, -1)
        del t2142

        # pd_op.matmul: (-1x24x144x32xf32) <- (-1x24x144x144xf32, -1x24x144x32xf32)
        t2144 = paddle._C_ops.matmul(t2143, t2123, False, False)
        del t2123, t2143

        # pd_op.transpose: (-1x144x24x32xf32) <- (-1x24x144x32xf32)
        t2145 = paddle._C_ops.transpose(t2144, [0, 2, 1, 3])
        del t2144

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2146 = [t2114, t345, t539]
        del t2114

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2147 = paddle._C_ops.stack(t2146, 0)
        del t2146

        # pd_op.reshape: (-1x144x768xf32) <- (-1x144x24x32xf32, 3xi64)
        t2148 = paddle._C_ops.reshape(t2145, t2147)
        del t2147, t2145

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x768xf32, 768x768xf32)
        t2149 = paddle._C_ops.matmul(t2148, t266, False, False)
        del t266, t2148

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t2150 = paddle._C_ops.add(t2149, t267)
        del t2149, t267

        # pd_op.reshape: (-1x12x12x768xf32) <- (-1x144x768xf32, 4xi64)
        t2151 = paddle._C_ops.reshape(t2150, t761)
        del t2150, t761

        # pd_op.reshape: (-1x2x2x12x12x768xf32) <- (-1x12x12x768xf32, 6xi64)
        t2152 = paddle._C_ops.reshape(t2151, t794)
        del t794, t2151

        # pd_op.transpose: (-1x2x12x2x12x768xf32) <- (-1x2x2x12x12x768xf32)
        t2153 = paddle._C_ops.transpose(t2152, [0, 1, 3, 2, 4, 5])
        del t2152

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x2x12x2x12x768xf32, 4xi64)
        t2154 = paddle._C_ops.reshape(t2153, t797)
        del t797, t2153

        # pd_op.roll: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 2xi64)
        t2155 = paddle._C_ops.roll(t2154, t503, [1, 2])
        del t2154

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2156 = [t2074, t799, t539]
        del t799, t2074

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2157 = paddle._C_ops.stack(t2156, 0)
        del t2156

        # pd_op.reshape: (-1x576x768xf32) <- (-1x24x24x768xf32, 3xi64)
        t2158 = paddle._C_ops.reshape(t2155, t2157)
        del t2155, t2157

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t2159 = paddle._C_ops.add(t2072, t2158)
        del t2072, t2158

        # pd_op.layer_norm: (-1x576x768xf32, -1x576xf32, -1x576xf32) <- (-1x576x768xf32, 768xf32, 768xf32)
        t2160, t2161, t2162 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2159, t268, t269, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t269, t268

        # pd_op.matmul: (-1x576x3072xf32) <- (-1x576x768xf32, 768x3072xf32)
        t2163 = paddle._C_ops.matmul(t2160, t270, False, False)
        del t2160, t270

        # pd_op.add: (-1x576x3072xf32) <- (-1x576x3072xf32, 3072xf32)
        t2164 = paddle._C_ops.add(t2163, t271)
        del t2163, t271

        # pd_op.gelu: (-1x576x3072xf32) <- (-1x576x3072xf32)
        t2165 = paddle._C_ops.gelu(t2164, False)
        del t2164

        # pd_op.matmul: (-1x576x768xf32) <- (-1x576x3072xf32, 3072x768xf32)
        t2166 = paddle._C_ops.matmul(t2165, t272, False, False)
        del t2165, t272

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, 768xf32)
        t2167 = paddle._C_ops.add(t2166, t273)
        del t2166, t273

        # pd_op.add: (-1x576x768xf32) <- (-1x576x768xf32, -1x576x768xf32)
        t2168 = paddle._C_ops.add(t2159, t2167)
        del t2159, t2167

        # pd_op.shape64: (3xi64) <- (-1x576x768xf32)
        t2169 = paddle._C_ops.shape64(t2168)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2170 = paddle._C_ops.slice(t2169, [0], t305, t306, [1], [0])
        del t2169

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2171 = [t2170, t750, t750, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2172 = paddle._C_ops.stack(t2171, 0)
        del t2171

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x576x768xf32, 4xi64)
        t2173 = paddle._C_ops.reshape(t2168, t2172)
        del t2168, t2172

        # pd_op.strided_slice: (-1x12x12x768xf32) <- (-1x24x24x768xf32, 2xi64, 2xi64, 2xi64)
        t2174 = paddle._C_ops.strided_slice(t2173, [1, 2], t419, t440, t523)

        # pd_op.strided_slice: (-1x12x12x768xf32) <- (-1x24x24x768xf32, 2xi64, 2xi64, 2xi64)
        t2175 = paddle._C_ops.strided_slice(t2173, [1, 2], t525, t440, t523)
        del t525

        # pd_op.strided_slice: (-1x12x12x768xf32) <- (-1x24x24x768xf32, 2xi64, 2xi64, 2xi64)
        t2176 = paddle._C_ops.strided_slice(t2173, [1, 2], t527, t440, t523)
        del t527

        # pd_op.strided_slice: (-1x12x12x768xf32) <- (-1x24x24x768xf32, 2xi64, 2xi64, 2xi64)
        t2177 = paddle._C_ops.strided_slice(t2173, [1, 2], t421, t440, t523)
        del t523

        # pd_op.shape64: (4xi64) <- (-1x24x24x768xf32)
        t2178 = paddle._C_ops.shape64(t2173)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2179 = paddle._C_ops.slice(t2178, [0], t305, t306, [1], [0])
        del t2178

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2180 = [t2179, t750, t750, t539]
        del t539, t750, t2179

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2181 = paddle._C_ops.stack(t2180, 0)
        del t2180

        # pd_op.reshape: (-1x24x24x768xf32) <- (-1x24x24x768xf32, 4xi64)
        t2182 = paddle._C_ops.reshape(t2173, t2181)
        del t2173, t2181

        # builtin.combine: ([-1x12x12x768xf32, -1x12x12x768xf32, -1x12x12x768xf32, -1x12x12x768xf32]) <- (-1x12x12x768xf32, -1x12x12x768xf32, -1x12x12x768xf32, -1x12x12x768xf32)
        t2183 = [t2174, t2175, t2176, t2177]
        del t2176, t2177, t2174, t2175

        # pd_op.concat: (-1x12x12x3072xf32) <- ([-1x12x12x768xf32, -1x12x12x768xf32, -1x12x12x768xf32, -1x12x12x768xf32], 1xi32)
        t2184 = paddle._C_ops.concat(t2183, t535)
        del t2183, t535

        # pd_op.full: (xi64) <- ()
        t2185 = paddle._C_ops.full(
            [], float("3072"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2186 = [t2170, t538, t2185]
        del t538, t2185, t2170

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2187 = paddle._C_ops.stack(t2186, 0)
        del t2186

        # pd_op.reshape: (-1x-1x3072xf32) <- (-1x12x12x3072xf32, 3xi64)
        t2188 = paddle._C_ops.reshape(t2184, t2187)
        del t2184, t2187

        # pd_op.layer_norm: (-1x-1x3072xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x3072xf32, 3072xf32, 3072xf32)
        t2189, t2190, t2191 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2188, t274, t275, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t275, t274, t2188

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x3072xf32, 3072x1536xf32)
        t2192 = paddle._C_ops.matmul(t2189, t276, False, False)
        del t2189, t276

        # pd_op.shape64: (3xi64) <- (-1x-1x1536xf32)
        t2193 = paddle._C_ops.shape64(t2192)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2194 = paddle._C_ops.slice(t2193, [0], t305, t306, [1], [0])
        del t2193

        # pd_op.shape64: (3xi64) <- (-1x-1x1536xf32)
        t2195 = paddle._C_ops.shape64(t2192)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2196 = paddle._C_ops.slice(t2195, [0], t306, t354, [1], [0])
        del t2195

        # pd_op.layer_norm: (-1x-1x1536xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x1536xf32, 1536xf32, 1536xf32)
        t2197, t2198, t2199 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2192, t277, t278, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t278, t277

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2200 = [t2194, t332, t332, t735]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2201 = paddle._C_ops.stack(t2200, 0)
        del t2200

        # pd_op.reshape: (-1x12x12x1536xf32) <- (-1x-1x1536xf32, 4xi64)
        t2202 = paddle._C_ops.reshape(t2197, t2201)
        del t2197, t2201

        # pd_op.shape64: (4xi64) <- (-1x12x12x1536xf32)
        t2203 = paddle._C_ops.shape64(t2202)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2204 = paddle._C_ops.slice(t2203, [0], t305, t306, [1], [0])
        del t2203

        # pd_op.full: (xi64) <- ()
        t2205 = paddle._C_ops.full([], float("1"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2206 = [t2204, t2205, t332, t2205, t332, t735]
        del t2204

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2207 = paddle._C_ops.stack(t2206, 0)
        del t2206

        # pd_op.reshape: (-1x1x12x1x12x1536xf32) <- (-1x12x12x1536xf32, 6xi64)
        t2208 = paddle._C_ops.reshape(t2202, t2207)
        del t2202, t2207

        # pd_op.transpose: (-1x1x1x12x12x1536xf32) <- (-1x1x12x1x12x1536xf32)
        t2209 = paddle._C_ops.transpose(t2208, [0, 1, 3, 2, 4, 5])
        del t2208

        # pd_op.full_int_array: (4xi64) <- ()
        t2210 = [-1, 12, 12, 1536]

        # pd_op.reshape: (-1x12x12x1536xf32) <- (-1x1x1x12x12x1536xf32, 4xi64)
        t2211 = paddle._C_ops.reshape(t2209, t2210)
        del t2209

        # pd_op.full_int_array: (3xi64) <- ()
        t2212 = [-1, 144, 1536]

        # pd_op.reshape: (-1x144x1536xf32) <- (-1x12x12x1536xf32, 3xi64)
        t2213 = paddle._C_ops.reshape(t2211, t2212)
        del t2211

        # pd_op.shape64: (3xi64) <- (-1x144x1536xf32)
        t2214 = paddle._C_ops.shape64(t2213)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2215 = paddle._C_ops.slice(t2214, [0], t305, t306, [1], [0])
        del t2214

        # pd_op.matmul: (-1x144x4608xf32) <- (-1x144x1536xf32, 1536x4608xf32)
        t2216 = paddle._C_ops.matmul(t2213, t279, False, False)
        del t279, t2213

        # pd_op.add: (-1x144x4608xf32) <- (-1x144x4608xf32, 4608xf32)
        t2217 = paddle._C_ops.add(t2216, t280)
        del t2216, t280

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2218 = [t2215, t345, t346, t554, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2219 = paddle._C_ops.stack(t2218, 0)
        del t2218

        # pd_op.reshape: (-1x144x3x48x32xf32) <- (-1x144x4608xf32, 5xi64)
        t2220 = paddle._C_ops.reshape(t2217, t2219)
        del t2217, t2219

        # pd_op.transpose: (3x-1x48x144x32xf32) <- (-1x144x3x48x32xf32)
        t2221 = paddle._C_ops.transpose(t2220, [2, 0, 3, 1, 4])
        del t2220

        # pd_op.slice: (-1x48x144x32xf32) <- (3x-1x48x144x32xf32, 1xi64, 1xi64)
        t2222 = paddle._C_ops.slice(t2221, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x48x144x32xf32) <- (3x-1x48x144x32xf32, 1xi64, 1xi64)
        t2223 = paddle._C_ops.slice(t2221, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x48x144x32xf32) <- (3x-1x48x144x32xf32, 1xi64, 1xi64)
        t2224 = paddle._C_ops.slice(t2221, [0], t354, t356, [1], [0])
        del t2221

        # pd_op.scale: (-1x48x144x32xf32) <- (-1x48x144x32xf32, 1xf32)
        t2225 = paddle._C_ops.scale(t2222, t358, float("0"), True)
        del t2222

        # pd_op.transpose: (-1x48x32x144xf32) <- (-1x48x144x32xf32)
        t2226 = paddle._C_ops.transpose(t2223, [0, 1, 3, 2])
        del t2223

        # pd_op.matmul: (-1x48x144x144xf32) <- (-1x48x144x32xf32, -1x48x32x144xf32)
        t2227 = paddle._C_ops.matmul(t2225, t2226, False, False)
        del t2225, t2226

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t2228 = paddle._C_ops.reshape(input45, t362)
        del input45

        # pd_op.index_select: (20736x48xf32) <- (529x48xf32, 20736xi64)
        t2229 = paddle._C_ops.index_select(input46, t2228, 0)
        del input46, t2228

        # pd_op.reshape: (144x144x48xf32) <- (20736x48xf32, 3xi64)
        t2230 = paddle._C_ops.reshape(t2229, t365)
        del t2229

        # pd_op.transpose: (48x144x144xf32) <- (144x144x48xf32)
        t2231 = paddle._C_ops.transpose(t2230, [2, 0, 1])
        del t2230

        # pd_op.unsqueeze: (1x48x144x144xf32) <- (48x144x144xf32, 1xi64)
        t2232 = paddle._C_ops.unsqueeze(t2231, t305)
        del t2231

        # pd_op.add: (-1x48x144x144xf32) <- (-1x48x144x144xf32, 1x48x144x144xf32)
        t2233 = paddle._C_ops.add(t2227, t2232)
        del t2227, t2232

        # pd_op.softmax: (-1x48x144x144xf32) <- (-1x48x144x144xf32)
        t2234 = paddle._C_ops.softmax(t2233, -1)
        del t2233

        # pd_op.matmul: (-1x48x144x32xf32) <- (-1x48x144x144xf32, -1x48x144x32xf32)
        t2235 = paddle._C_ops.matmul(t2234, t2224, False, False)
        del t2224, t2234

        # pd_op.transpose: (-1x144x48x32xf32) <- (-1x48x144x32xf32)
        t2236 = paddle._C_ops.transpose(t2235, [0, 2, 1, 3])
        del t2235

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2237 = [t2215, t345, t735]
        del t2215

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2238 = paddle._C_ops.stack(t2237, 0)
        del t2237

        # pd_op.reshape: (-1x144x1536xf32) <- (-1x144x48x32xf32, 3xi64)
        t2239 = paddle._C_ops.reshape(t2236, t2238)
        del t2238, t2236

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536x1536xf32)
        t2240 = paddle._C_ops.matmul(t2239, t281, False, False)
        del t281, t2239

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t2241 = paddle._C_ops.add(t2240, t282)
        del t2240, t282

        # pd_op.reshape: (-1x12x12x1536xf32) <- (-1x144x1536xf32, 4xi64)
        t2242 = paddle._C_ops.reshape(t2241, t2210)
        del t2241

        # pd_op.full_int_array: (6xi64) <- ()
        t2243 = [-1, 1, 1, 12, 12, 1536]

        # pd_op.reshape: (-1x1x1x12x12x1536xf32) <- (-1x12x12x1536xf32, 6xi64)
        t2244 = paddle._C_ops.reshape(t2242, t2243)
        del t2242

        # pd_op.transpose: (-1x1x12x1x12x1536xf32) <- (-1x1x1x12x12x1536xf32)
        t2245 = paddle._C_ops.transpose(t2244, [0, 1, 3, 2, 4, 5])
        del t2244

        # pd_op.reshape: (-1x12x12x1536xf32) <- (-1x1x12x1x12x1536xf32, 4xi64)
        t2246 = paddle._C_ops.reshape(t2245, t2210)
        del t2245

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2247 = [t2194, t345, t735]
        del t2194

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2248 = paddle._C_ops.stack(t2247, 0)
        del t2247

        # pd_op.reshape: (-1x144x1536xf32) <- (-1x12x12x1536xf32, 3xi64)
        t2249 = paddle._C_ops.reshape(t2246, t2248)
        del t2246, t2248

        # pd_op.add: (-1x144x1536xf32) <- (-1x-1x1536xf32, -1x144x1536xf32)
        t2250 = paddle._C_ops.add(t2192, t2249)
        del t2192, t2249

        # pd_op.layer_norm: (-1x144x1536xf32, -1x144xf32, -1x144xf32) <- (-1x144x1536xf32, 1536xf32, 1536xf32)
        t2251, t2252, t2253 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2250, t283, t284, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t284, t283

        # pd_op.matmul: (-1x144x6144xf32) <- (-1x144x1536xf32, 1536x6144xf32)
        t2254 = paddle._C_ops.matmul(t2251, t285, False, False)
        del t2251, t285

        # pd_op.add: (-1x144x6144xf32) <- (-1x144x6144xf32, 6144xf32)
        t2255 = paddle._C_ops.add(t2254, t286)
        del t2254, t286

        # pd_op.gelu: (-1x144x6144xf32) <- (-1x144x6144xf32)
        t2256 = paddle._C_ops.gelu(t2255, False)
        del t2255

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x6144xf32, 6144x1536xf32)
        t2257 = paddle._C_ops.matmul(t2256, t287, False, False)
        del t2256, t287

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t2258 = paddle._C_ops.add(t2257, t288)
        del t2257, t288

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, -1x144x1536xf32)
        t2259 = paddle._C_ops.add(t2250, t2258)
        del t2250, t2258

        # pd_op.shape64: (3xi64) <- (-1x144x1536xf32)
        t2260 = paddle._C_ops.shape64(t2259)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2261 = paddle._C_ops.slice(t2260, [0], t305, t306, [1], [0])
        del t2260

        # pd_op.layer_norm: (-1x144x1536xf32, -1x144xf32, -1x144xf32) <- (-1x144x1536xf32, 1536xf32, 1536xf32)
        t2262, t2263, t2264 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2259, t289, t290, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t290, t289

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2265 = [t2261, t332, t332, t735]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2266 = paddle._C_ops.stack(t2265, 0)
        del t2265

        # pd_op.reshape: (-1x12x12x1536xf32) <- (-1x144x1536xf32, 4xi64)
        t2267 = paddle._C_ops.reshape(t2262, t2266)
        del t2262, t2266

        # pd_op.shape64: (4xi64) <- (-1x12x12x1536xf32)
        t2268 = paddle._C_ops.shape64(t2267)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2269 = paddle._C_ops.slice(t2268, [0], t305, t306, [1], [0])
        del t2268

        # pd_op.roll: (-1x12x12x1536xf32) <- (-1x12x12x1536xf32, 2xi64)
        t2270 = paddle._C_ops.roll(t2267, t408, [1, 2])
        del t2267

        # pd_op.shape64: (4xi64) <- (-1x12x12x1536xf32)
        t2271 = paddle._C_ops.shape64(t2270)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2272 = paddle._C_ops.slice(t2271, [0], t305, t306, [1], [0])
        del t2271

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2273 = [t2272, t2205, t332, t2205, t332, t735]
        del t332, t2272

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2274 = paddle._C_ops.stack(t2273, 0)
        del t2273

        # pd_op.reshape: (-1x1x12x1x12x1536xf32) <- (-1x12x12x1536xf32, 6xi64)
        t2275 = paddle._C_ops.reshape(t2270, t2274)
        del t2270, t2274

        # pd_op.transpose: (-1x1x1x12x12x1536xf32) <- (-1x1x12x1x12x1536xf32)
        t2276 = paddle._C_ops.transpose(t2275, [0, 1, 3, 2, 4, 5])
        del t2275

        # pd_op.reshape: (-1x12x12x1536xf32) <- (-1x1x1x12x12x1536xf32, 4xi64)
        t2277 = paddle._C_ops.reshape(t2276, t2210)
        del t2276

        # pd_op.reshape: (-1x144x1536xf32) <- (-1x12x12x1536xf32, 3xi64)
        t2278 = paddle._C_ops.reshape(t2277, t2212)
        del t2212, t2277

        # pd_op.full: (1x12x12x1xf32) <- ()
        t2279 = paddle._C_ops.full(
            [1, 12, 12, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2280 = paddle._C_ops.set_value_(
            t2279, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t2279, t419

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2281 = paddle._C_ops.set_value_(
            t2280, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t423, t2280

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2282 = paddle._C_ops.set_value_(
            t2281, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t426, t427, t2281

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2283 = paddle._C_ops.set_value_(
            t2282, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t429, t2282

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2284 = paddle._C_ops.set_value_(
            t2283, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t420, t2283

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2285 = paddle._C_ops.set_value_(
            t2284, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t424, t433, t2284

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2286 = paddle._C_ops.set_value_(
            t2285, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t435, t436, t2285

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2287 = paddle._C_ops.set_value_(
            t2286, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t430, t438, t2286

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2288 = paddle._C_ops.set_value_(
            t2287, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t408, t440, t2287

        # pd_op.full_int_array: (6xi64) <- ()
        t2289 = [1, 1, 12, 1, 12, 1]

        # pd_op.reshape: (1x1x12x1x12x1xf32) <- (1x12x12x1xf32, 6xi64)
        t2290 = paddle._C_ops.reshape(t2288, t2289)
        del t2289

        # pd_op.transpose: (1x1x1x12x12x1xf32) <- (1x1x12x1x12x1xf32)
        t2291 = paddle._C_ops.transpose(t2290, [0, 1, 3, 2, 4, 5])
        del t2290

        # pd_op.reshape: (1x12x12x1xf32) <- (1x1x1x12x12x1xf32, 4xi64)
        t2292 = paddle._C_ops.reshape(t2291, t445)
        del t445, t2291

        # pd_op.reshape: (1x144xf32) <- (1x12x12x1xf32, 2xi64)
        t2293 = paddle._C_ops.reshape(t2292, t447)
        del t447, t2292

        # pd_op.unsqueeze: (1x1x144xf32) <- (1x144xf32, 1xi64)
        t2294 = paddle._C_ops.unsqueeze(t2293, t306)

        # pd_op.unsqueeze: (1x144x1xf32) <- (1x144xf32, 1xi64)
        t2295 = paddle._C_ops.unsqueeze(t2293, t354)
        del t2293

        # pd_op.subtract: (1x144x144xf32) <- (1x1x144xf32, 1x144x1xf32)
        t2296 = paddle._C_ops.subtract(t2294, t2295)
        del t2294, t2295

        # pd_op.not_equal: (1x144x144xb) <- (1x144x144xf32, xf32)
        t2297 = paddle._C_ops.not_equal(t2296, t452)

        # pd_op.full: (1x144x144xf32) <- ()
        t2298 = paddle._C_ops.full(
            [1, 144, 144],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x144x144xf32) <- (1x144x144xb, 1x144x144xf32, 1x144x144xf32)
        t2299 = paddle._C_ops.where(t2297, t2298, t2296)
        del t2298, t2297, t2296

        # pd_op.equal: (1x144x144xb) <- (1x144x144xf32, xf32)
        t2300 = paddle._C_ops.equal(t2299, t452)
        del t452

        # pd_op.full: (1x144x144xf32) <- ()
        t2301 = paddle._C_ops.full(
            [1, 144, 144],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x144x144xf32) <- (1x144x144xb, 1x144x144xf32, 1x144x144xf32)
        t2302 = paddle._C_ops.where(t2300, t2301, t2299)
        del t2300, t2301, t2299

        # pd_op.shape64: (3xi64) <- (-1x144x1536xf32)
        t2303 = paddle._C_ops.shape64(t2278)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2304 = paddle._C_ops.slice(t2303, [0], t305, t306, [1], [0])
        del t2303

        # pd_op.matmul: (-1x144x4608xf32) <- (-1x144x1536xf32, 1536x4608xf32)
        t2305 = paddle._C_ops.matmul(t2278, t291, False, False)
        del t291, t2278

        # pd_op.add: (-1x144x4608xf32) <- (-1x144x4608xf32, 4608xf32)
        t2306 = paddle._C_ops.add(t2305, t292)
        del t2305, t292

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2307 = [t2304, t345, t346, t554, t348]
        del t346, t348

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2308 = paddle._C_ops.stack(t2307, 0)
        del t2307

        # pd_op.reshape: (-1x144x3x48x32xf32) <- (-1x144x4608xf32, 5xi64)
        t2309 = paddle._C_ops.reshape(t2306, t2308)
        del t2306, t2308

        # pd_op.transpose: (3x-1x48x144x32xf32) <- (-1x144x3x48x32xf32)
        t2310 = paddle._C_ops.transpose(t2309, [2, 0, 3, 1, 4])
        del t2309

        # pd_op.slice: (-1x48x144x32xf32) <- (3x-1x48x144x32xf32, 1xi64, 1xi64)
        t2311 = paddle._C_ops.slice(t2310, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x48x144x32xf32) <- (3x-1x48x144x32xf32, 1xi64, 1xi64)
        t2312 = paddle._C_ops.slice(t2310, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x48x144x32xf32) <- (3x-1x48x144x32xf32, 1xi64, 1xi64)
        t2313 = paddle._C_ops.slice(t2310, [0], t354, t356, [1], [0])
        del t356, t2310

        # pd_op.scale: (-1x48x144x32xf32) <- (-1x48x144x32xf32, 1xf32)
        t2314 = paddle._C_ops.scale(t2311, t358, float("0"), True)
        del t358, t2311

        # pd_op.transpose: (-1x48x32x144xf32) <- (-1x48x144x32xf32)
        t2315 = paddle._C_ops.transpose(t2312, [0, 1, 3, 2])
        del t2312

        # pd_op.matmul: (-1x48x144x144xf32) <- (-1x48x144x32xf32, -1x48x32x144xf32)
        t2316 = paddle._C_ops.matmul(t2314, t2315, False, False)
        del t2314, t2315

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t2317 = paddle._C_ops.reshape(input47, t362)
        del input47, t362

        # pd_op.index_select: (20736x48xf32) <- (529x48xf32, 20736xi64)
        t2318 = paddle._C_ops.index_select(input48, t2317, 0)
        del input48, t2317

        # pd_op.reshape: (144x144x48xf32) <- (20736x48xf32, 3xi64)
        t2319 = paddle._C_ops.reshape(t2318, t365)
        del t365, t2318

        # pd_op.transpose: (48x144x144xf32) <- (144x144x48xf32)
        t2320 = paddle._C_ops.transpose(t2319, [2, 0, 1])
        del t2319

        # pd_op.unsqueeze: (1x48x144x144xf32) <- (48x144x144xf32, 1xi64)
        t2321 = paddle._C_ops.unsqueeze(t2320, t305)
        del t2320

        # pd_op.add: (-1x48x144x144xf32) <- (-1x48x144x144xf32, 1x48x144x144xf32)
        t2322 = paddle._C_ops.add(t2316, t2321)
        del t2316, t2321

        # pd_op.full: (xi64) <- ()
        t2323 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t2324 = paddle._C_ops.floor_divide(t2304, t2323)
        del t2323

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2325 = [t2324, t2205, t554, t345, t345]
        del t2324, t2205

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2326 = paddle._C_ops.stack(t2325, 0)
        del t2325

        # pd_op.reshape: (-1x1x48x144x144xf32) <- (-1x48x144x144xf32, 5xi64)
        t2327 = paddle._C_ops.reshape(t2322, t2326)
        del t2322, t2326

        # pd_op.unsqueeze: (1x1x144x144xf32) <- (1x144x144xf32, 1xi64)
        t2328 = paddle._C_ops.unsqueeze(t2302, t306)
        del t306, t2302

        # pd_op.unsqueeze: (1x1x1x144x144xf32) <- (1x1x144x144xf32, 1xi64)
        t2329 = paddle._C_ops.unsqueeze(t2328, t305)
        del t305, t2328

        # pd_op.add: (-1x1x48x144x144xf32) <- (-1x1x48x144x144xf32, 1x1x1x144x144xf32)
        t2330 = paddle._C_ops.add(t2327, t2329)
        del t2327, t2329

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2331 = [t2304, t554, t345, t345]
        del t554

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2332 = paddle._C_ops.stack(t2331, 0)
        del t2331

        # pd_op.reshape: (-1x48x144x144xf32) <- (-1x1x48x144x144xf32, 4xi64)
        t2333 = paddle._C_ops.reshape(t2330, t2332)
        del t2330, t2332

        # pd_op.softmax: (-1x48x144x144xf32) <- (-1x48x144x144xf32)
        t2334 = paddle._C_ops.softmax(t2333, -1)
        del t2333

        # pd_op.matmul: (-1x48x144x32xf32) <- (-1x48x144x144xf32, -1x48x144x32xf32)
        t2335 = paddle._C_ops.matmul(t2334, t2313, False, False)
        del t2313, t2334

        # pd_op.transpose: (-1x144x48x32xf32) <- (-1x48x144x32xf32)
        t2336 = paddle._C_ops.transpose(t2335, [0, 2, 1, 3])
        del t2335

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2337 = [t2304, t345, t735]
        del t2304

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2338 = paddle._C_ops.stack(t2337, 0)
        del t2337

        # pd_op.reshape: (-1x144x1536xf32) <- (-1x144x48x32xf32, 3xi64)
        t2339 = paddle._C_ops.reshape(t2336, t2338)
        del t2338, t2336

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536x1536xf32)
        t2340 = paddle._C_ops.matmul(t2339, t293, False, False)
        del t293, t2339

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t2341 = paddle._C_ops.add(t2340, t294)
        del t2340, t294

        # pd_op.reshape: (-1x12x12x1536xf32) <- (-1x144x1536xf32, 4xi64)
        t2342 = paddle._C_ops.reshape(t2341, t2210)
        del t2341

        # pd_op.reshape: (-1x1x1x12x12x1536xf32) <- (-1x12x12x1536xf32, 6xi64)
        t2343 = paddle._C_ops.reshape(t2342, t2243)
        del t2243, t2342

        # pd_op.transpose: (-1x1x12x1x12x1536xf32) <- (-1x1x1x12x12x1536xf32)
        t2344 = paddle._C_ops.transpose(t2343, [0, 1, 3, 2, 4, 5])
        del t2343

        # pd_op.reshape: (-1x12x12x1536xf32) <- (-1x1x12x1x12x1536xf32, 4xi64)
        t2345 = paddle._C_ops.reshape(t2344, t2210)
        del t2210, t2344

        # pd_op.roll: (-1x12x12x1536xf32) <- (-1x12x12x1536xf32, 2xi64)
        t2346 = paddle._C_ops.roll(t2345, t503, [1, 2])
        del t503, t2345

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2347 = [t2261, t345, t735]
        del t735, t345, t2261

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2348 = paddle._C_ops.stack(t2347, 0)
        del t2347

        # pd_op.reshape: (-1x144x1536xf32) <- (-1x12x12x1536xf32, 3xi64)
        t2349 = paddle._C_ops.reshape(t2346, t2348)
        del t2346, t2348

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, -1x144x1536xf32)
        t2350 = paddle._C_ops.add(t2259, t2349)
        del t2259, t2349

        # pd_op.layer_norm: (-1x144x1536xf32, -1x144xf32, -1x144xf32) <- (-1x144x1536xf32, 1536xf32, 1536xf32)
        t2351, t2352, t2353 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2350, t295, t296, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t296, t295

        # pd_op.matmul: (-1x144x6144xf32) <- (-1x144x1536xf32, 1536x6144xf32)
        t2354 = paddle._C_ops.matmul(t2351, t297, False, False)
        del t2351, t297

        # pd_op.add: (-1x144x6144xf32) <- (-1x144x6144xf32, 6144xf32)
        t2355 = paddle._C_ops.add(t2354, t298)
        del t2354, t298

        # pd_op.gelu: (-1x144x6144xf32) <- (-1x144x6144xf32)
        t2356 = paddle._C_ops.gelu(t2355, False)
        del t2355

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x6144xf32, 6144x1536xf32)
        t2357 = paddle._C_ops.matmul(t2356, t299, False, False)
        del t2356, t299

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t2358 = paddle._C_ops.add(t2357, t300)
        del t2357, t300

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, -1x144x1536xf32)
        t2359 = paddle._C_ops.add(t2350, t2358)
        del t2350, t2358

        # pd_op.layer_norm: (-1x144x1536xf32, -1x144xf32, -1x144xf32) <- (-1x144x1536xf32, 1536xf32, 1536xf32)
        t2360, t2361, t2362 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2359, t301, t302, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t2359, t302, t301

        # pd_op.transpose: (-1x1536x144xf32) <- (-1x144x1536xf32)
        t2363 = paddle._C_ops.transpose(t2360, [0, 2, 1])
        del t2360

        # pd_op.unsqueeze: (-1x1536x1x144xf32) <- (-1x1536x144xf32, 1xi64)
        t2364 = paddle._C_ops.unsqueeze(t2363, t354)
        del t2363

        # pd_op.pool2d: (-1x1536x1x1xf32) <- (-1x1536x1x144xf32, 2xi64)
        t2365 = paddle._C_ops.pool2d(
            t2364,
            t421,
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
        del t421, t2364

        # pd_op.squeeze: (-1x1536x1xf32) <- (-1x1536x1x1xf32, 1xi64)
        t2366 = paddle._C_ops.squeeze(t2365, t354)
        del t354, t2365

        # pd_op.flatten: (-1x1536xf32) <- (-1x1536x1xf32)
        t2367 = paddle._C_ops.flatten(t2366, 1, 2)
        del t2366

        # pd_op.matmul: (-1x102xf32) <- (-1x1536xf32, 1536x102xf32)
        t2368 = paddle._C_ops.matmul(t2367, t303, False, False)
        del t2367, t303

        return (
            t441,
            t646,
            t841,
            t1002,
            t1159,
            t1316,
            t1473,
            t1630,
            t1787,
            t1944,
            t2101,
            t2288,
            t2368,
        )
