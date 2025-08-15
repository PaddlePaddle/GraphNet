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

        # pd_op.conv2d: (-1x128x96x96xf32) <- (-1x3x384x384xf32, 128x3x4x4xf32)
        t308 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t309 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        t310 = paddle._C_ops.reshape(t1, t309)
        del t309, t1

        # pd_op.add: (-1x128x96x96xf32) <- (-1x128x96x96xf32, 1x128x1x1xf32)
        t311 = paddle._C_ops.add(t308, t310)
        del t308, t310

        # pd_op.shape64: (4xi64) <- (-1x128x96x96xf32)
        t312 = paddle._C_ops.shape64(t311)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t313 = paddle._C_ops.slice(t312, [0], t305, t306, [1], [0])
        del t312

        # pd_op.flatten: (-1x128x9216xf32) <- (-1x128x96x96xf32)
        t314 = paddle._C_ops.flatten(t311, 2, 3)
        del t311

        # pd_op.transpose: (-1x9216x128xf32) <- (-1x128x9216xf32)
        t315 = paddle._C_ops.transpose(t314, [0, 2, 1])
        del t314

        # pd_op.layer_norm: (-1x9216x128xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x128xf32, 128xf32, 128xf32)
        t316, t317, t318 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t315, t2, t3, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t3, t2, t315

        # pd_op.shape64: (3xi64) <- (-1x9216x128xf32)
        t319 = paddle._C_ops.shape64(t316)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t320 = paddle._C_ops.slice(t319, [0], t305, t306, [1], [0])
        del t319

        # pd_op.layer_norm: (-1x9216x128xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x128xf32, 128xf32, 128xf32)
        t321, t322, t323 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t316, t4, t5, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t5, t4

        # pd_op.full: (xi64) <- ()
        t324 = paddle._C_ops.full([], float("96"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t325 = paddle._C_ops.full(
            [], float("128"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t326 = [t320, t324, t324, t325]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t327 = paddle._C_ops.stack(t326, 0)
        del t326

        # pd_op.reshape: (-1x96x96x128xf32) <- (-1x9216x128xf32, 4xi64)
        t328 = paddle._C_ops.reshape(t321, t327)
        del t321, t327

        # pd_op.shape64: (4xi64) <- (-1x96x96x128xf32)
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

        # pd_op.reshape: (-1x8x12x8x12x128xf32) <- (-1x96x96x128xf32, 6xi64)
        t335 = paddle._C_ops.reshape(t328, t334)
        del t328, t334

        # pd_op.transpose: (-1x8x8x12x12x128xf32) <- (-1x8x12x8x12x128xf32)
        t336 = paddle._C_ops.transpose(t335, [0, 1, 3, 2, 4, 5])
        del t335

        # pd_op.full_int_array: (4xi64) <- ()
        t337 = [-1, 12, 12, 128]

        # pd_op.reshape: (-1x12x12x128xf32) <- (-1x8x8x12x12x128xf32, 4xi64)
        t338 = paddle._C_ops.reshape(t336, t337)
        del t336

        # pd_op.full_int_array: (3xi64) <- ()
        t339 = [-1, 144, 128]

        # pd_op.reshape: (-1x144x128xf32) <- (-1x12x12x128xf32, 3xi64)
        t340 = paddle._C_ops.reshape(t338, t339)
        del t338

        # pd_op.shape64: (3xi64) <- (-1x144x128xf32)
        t341 = paddle._C_ops.shape64(t340)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t342 = paddle._C_ops.slice(t341, [0], t305, t306, [1], [0])
        del t341

        # pd_op.matmul: (-1x144x384xf32) <- (-1x144x128xf32, 128x384xf32)
        t343 = paddle._C_ops.matmul(t340, t6, False, False)
        del t6, t340

        # pd_op.add: (-1x144x384xf32) <- (-1x144x384xf32, 384xf32)
        t344 = paddle._C_ops.add(t343, t7)
        del t343, t7

        # pd_op.full: (xi64) <- ()
        t345 = paddle._C_ops.full(
            [], float("144"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        t346 = paddle._C_ops.full([], float("3"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t347 = paddle._C_ops.full([], float("4"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t348 = paddle._C_ops.full([], float("32"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t349 = [t342, t345, t346, t347, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t350 = paddle._C_ops.stack(t349, 0)
        del t349

        # pd_op.reshape: (-1x144x3x4x32xf32) <- (-1x144x384xf32, 5xi64)
        t351 = paddle._C_ops.reshape(t344, t350)
        del t344, t350

        # pd_op.transpose: (3x-1x4x144x32xf32) <- (-1x144x3x4x32xf32)
        t352 = paddle._C_ops.transpose(t351, [2, 0, 3, 1, 4])
        del t351

        # pd_op.slice: (-1x4x144x32xf32) <- (3x-1x4x144x32xf32, 1xi64, 1xi64)
        t353 = paddle._C_ops.slice(t352, [0], t305, t306, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t354 = [2]

        # pd_op.slice: (-1x4x144x32xf32) <- (3x-1x4x144x32xf32, 1xi64, 1xi64)
        t355 = paddle._C_ops.slice(t352, [0], t306, t354, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t356 = [3]

        # pd_op.slice: (-1x4x144x32xf32) <- (3x-1x4x144x32xf32, 1xi64, 1xi64)
        t357 = paddle._C_ops.slice(t352, [0], t354, t356, [1], [0])
        del t352

        # pd_op.full: (1xf32) <- ()
        t358 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x4x144x32xf32) <- (-1x4x144x32xf32, 1xf32)
        t359 = paddle._C_ops.scale(t353, t358, float("0"), True)
        del t353

        # pd_op.transpose: (-1x4x32x144xf32) <- (-1x4x144x32xf32)
        t360 = paddle._C_ops.transpose(t355, [0, 1, 3, 2])
        del t355

        # pd_op.matmul: (-1x4x144x144xf32) <- (-1x4x144x32xf32, -1x4x32x144xf32)
        t361 = paddle._C_ops.matmul(t359, t360, False, False)
        del t359, t360

        # pd_op.full_int_array: (1xi64) <- ()
        t362 = [-1]

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t363 = paddle._C_ops.reshape(input1, t362)
        del input1

        # pd_op.index_select: (20736x4xf32) <- (529x4xf32, 20736xi64)
        t364 = paddle._C_ops.index_select(input2, t363, 0)
        del input2, t363

        # pd_op.full_int_array: (3xi64) <- ()
        t365 = [144, 144, -1]

        # pd_op.reshape: (144x144x4xf32) <- (20736x4xf32, 3xi64)
        t366 = paddle._C_ops.reshape(t364, t365)
        del t364

        # pd_op.transpose: (4x144x144xf32) <- (144x144x4xf32)
        t367 = paddle._C_ops.transpose(t366, [2, 0, 1])
        del t366

        # pd_op.unsqueeze: (1x4x144x144xf32) <- (4x144x144xf32, 1xi64)
        t368 = paddle._C_ops.unsqueeze(t367, t305)
        del t367

        # pd_op.add: (-1x4x144x144xf32) <- (-1x4x144x144xf32, 1x4x144x144xf32)
        t369 = paddle._C_ops.add(t361, t368)
        del t361, t368

        # pd_op.softmax: (-1x4x144x144xf32) <- (-1x4x144x144xf32)
        t370 = paddle._C_ops.softmax(t369, -1)
        del t369

        # pd_op.matmul: (-1x4x144x32xf32) <- (-1x4x144x144xf32, -1x4x144x32xf32)
        t371 = paddle._C_ops.matmul(t370, t357, False, False)
        del t357, t370

        # pd_op.transpose: (-1x144x4x32xf32) <- (-1x4x144x32xf32)
        t372 = paddle._C_ops.transpose(t371, [0, 2, 1, 3])
        del t371

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t373 = [t342, t345, t325]
        del t342

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t374 = paddle._C_ops.stack(t373, 0)
        del t373

        # pd_op.reshape: (-1x144x128xf32) <- (-1x144x4x32xf32, 3xi64)
        t375 = paddle._C_ops.reshape(t372, t374)
        del t374, t372

        # pd_op.matmul: (-1x144x128xf32) <- (-1x144x128xf32, 128x128xf32)
        t376 = paddle._C_ops.matmul(t375, t8, False, False)
        del t8, t375

        # pd_op.add: (-1x144x128xf32) <- (-1x144x128xf32, 128xf32)
        t377 = paddle._C_ops.add(t376, t9)
        del t376, t9

        # pd_op.reshape: (-1x12x12x128xf32) <- (-1x144x128xf32, 4xi64)
        t378 = paddle._C_ops.reshape(t377, t337)
        del t377

        # pd_op.full_int_array: (6xi64) <- ()
        t379 = [-1, 8, 8, 12, 12, 128]

        # pd_op.reshape: (-1x8x8x12x12x128xf32) <- (-1x12x12x128xf32, 6xi64)
        t380 = paddle._C_ops.reshape(t378, t379)
        del t378

        # pd_op.transpose: (-1x8x12x8x12x128xf32) <- (-1x8x8x12x12x128xf32)
        t381 = paddle._C_ops.transpose(t380, [0, 1, 3, 2, 4, 5])
        del t380

        # pd_op.full_int_array: (4xi64) <- ()
        t382 = [-1, 96, 96, 128]

        # pd_op.reshape: (-1x96x96x128xf32) <- (-1x8x12x8x12x128xf32, 4xi64)
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

        # pd_op.reshape: (-1x9216x128xf32) <- (-1x96x96x128xf32, 3xi64)
        t387 = paddle._C_ops.reshape(t383, t386)
        del t383, t386

        # pd_op.add: (-1x9216x128xf32) <- (-1x9216x128xf32, -1x9216x128xf32)
        t388 = paddle._C_ops.add(t316, t387)
        del t316, t387

        # pd_op.layer_norm: (-1x9216x128xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x128xf32, 128xf32, 128xf32)
        t389, t390, t391 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t388, t10, t11, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t11, t10

        # pd_op.matmul: (-1x9216x512xf32) <- (-1x9216x128xf32, 128x512xf32)
        t392 = paddle._C_ops.matmul(t389, t12, False, False)
        del t389, t12

        # pd_op.add: (-1x9216x512xf32) <- (-1x9216x512xf32, 512xf32)
        t393 = paddle._C_ops.add(t392, t13)
        del t392, t13

        # pd_op.gelu: (-1x9216x512xf32) <- (-1x9216x512xf32)
        t394 = paddle._C_ops.gelu(t393, False)
        del t393

        # pd_op.matmul: (-1x9216x128xf32) <- (-1x9216x512xf32, 512x128xf32)
        t395 = paddle._C_ops.matmul(t394, t14, False, False)
        del t394, t14

        # pd_op.add: (-1x9216x128xf32) <- (-1x9216x128xf32, 128xf32)
        t396 = paddle._C_ops.add(t395, t15)
        del t395, t15

        # pd_op.add: (-1x9216x128xf32) <- (-1x9216x128xf32, -1x9216x128xf32)
        t397 = paddle._C_ops.add(t388, t396)
        del t388, t396

        # pd_op.shape64: (3xi64) <- (-1x9216x128xf32)
        t398 = paddle._C_ops.shape64(t397)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t399 = paddle._C_ops.slice(t398, [0], t305, t306, [1], [0])
        del t398

        # pd_op.layer_norm: (-1x9216x128xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x128xf32, 128xf32, 128xf32)
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

        # pd_op.reshape: (-1x96x96x128xf32) <- (-1x9216x128xf32, 4xi64)
        t405 = paddle._C_ops.reshape(t400, t404)
        del t400, t404

        # pd_op.shape64: (4xi64) <- (-1x96x96x128xf32)
        t406 = paddle._C_ops.shape64(t405)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t407 = paddle._C_ops.slice(t406, [0], t305, t306, [1], [0])
        del t406

        # pd_op.full_int_array: (2xi64) <- ()
        t408 = [-6, -6]

        # pd_op.roll: (-1x96x96x128xf32) <- (-1x96x96x128xf32, 2xi64)
        t409 = paddle._C_ops.roll(t405, t408, [1, 2])
        del t405

        # pd_op.shape64: (4xi64) <- (-1x96x96x128xf32)
        t410 = paddle._C_ops.shape64(t409)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t411 = paddle._C_ops.slice(t410, [0], t305, t306, [1], [0])
        del t410

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t412 = [t411, t331, t332, t331, t332, t325]
        del t411

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t413 = paddle._C_ops.stack(t412, 0)
        del t412

        # pd_op.reshape: (-1x8x12x8x12x128xf32) <- (-1x96x96x128xf32, 6xi64)
        t414 = paddle._C_ops.reshape(t409, t413)
        del t409, t413

        # pd_op.transpose: (-1x8x8x12x12x128xf32) <- (-1x8x12x8x12x128xf32)
        t415 = paddle._C_ops.transpose(t414, [0, 1, 3, 2, 4, 5])
        del t414

        # pd_op.reshape: (-1x12x12x128xf32) <- (-1x8x8x12x12x128xf32, 4xi64)
        t416 = paddle._C_ops.reshape(t415, t337)
        del t415

        # pd_op.reshape: (-1x144x128xf32) <- (-1x12x12x128xf32, 3xi64)
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

        # pd_op.shape64: (3xi64) <- (-1x144x128xf32)
        t459 = paddle._C_ops.shape64(t417)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t460 = paddle._C_ops.slice(t459, [0], t305, t306, [1], [0])
        del t459

        # pd_op.matmul: (-1x144x384xf32) <- (-1x144x128xf32, 128x384xf32)
        t461 = paddle._C_ops.matmul(t417, t18, False, False)
        del t18, t417

        # pd_op.add: (-1x144x384xf32) <- (-1x144x384xf32, 384xf32)
        t462 = paddle._C_ops.add(t461, t19)
        del t461, t19

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t463 = [t460, t345, t346, t347, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t464 = paddle._C_ops.stack(t463, 0)
        del t463

        # pd_op.reshape: (-1x144x3x4x32xf32) <- (-1x144x384xf32, 5xi64)
        t465 = paddle._C_ops.reshape(t462, t464)
        del t462, t464

        # pd_op.transpose: (3x-1x4x144x32xf32) <- (-1x144x3x4x32xf32)
        t466 = paddle._C_ops.transpose(t465, [2, 0, 3, 1, 4])
        del t465

        # pd_op.slice: (-1x4x144x32xf32) <- (3x-1x4x144x32xf32, 1xi64, 1xi64)
        t467 = paddle._C_ops.slice(t466, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x4x144x32xf32) <- (3x-1x4x144x32xf32, 1xi64, 1xi64)
        t468 = paddle._C_ops.slice(t466, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x4x144x32xf32) <- (3x-1x4x144x32xf32, 1xi64, 1xi64)
        t469 = paddle._C_ops.slice(t466, [0], t354, t356, [1], [0])
        del t466

        # pd_op.scale: (-1x4x144x32xf32) <- (-1x4x144x32xf32, 1xf32)
        t470 = paddle._C_ops.scale(t467, t358, float("0"), True)
        del t467

        # pd_op.transpose: (-1x4x32x144xf32) <- (-1x4x144x32xf32)
        t471 = paddle._C_ops.transpose(t468, [0, 1, 3, 2])
        del t468

        # pd_op.matmul: (-1x4x144x144xf32) <- (-1x4x144x32xf32, -1x4x32x144xf32)
        t472 = paddle._C_ops.matmul(t470, t471, False, False)
        del t470, t471

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t473 = paddle._C_ops.reshape(input3, t362)
        del input3

        # pd_op.index_select: (20736x4xf32) <- (529x4xf32, 20736xi64)
        t474 = paddle._C_ops.index_select(input4, t473, 0)
        del input4, t473

        # pd_op.reshape: (144x144x4xf32) <- (20736x4xf32, 3xi64)
        t475 = paddle._C_ops.reshape(t474, t365)
        del t474

        # pd_op.transpose: (4x144x144xf32) <- (144x144x4xf32)
        t476 = paddle._C_ops.transpose(t475, [2, 0, 1])
        del t475

        # pd_op.unsqueeze: (1x4x144x144xf32) <- (4x144x144xf32, 1xi64)
        t477 = paddle._C_ops.unsqueeze(t476, t305)
        del t476

        # pd_op.add: (-1x4x144x144xf32) <- (-1x4x144x144xf32, 1x4x144x144xf32)
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

        # pd_op.reshape: (-1x64x4x144x144xf32) <- (-1x4x144x144xf32, 5xi64)
        t484 = paddle._C_ops.reshape(t478, t483)
        del t478, t483

        # pd_op.unsqueeze: (64x1x144x144xf32) <- (64x144x144xf32, 1xi64)
        t485 = paddle._C_ops.unsqueeze(t458, t306)
        del t458

        # pd_op.unsqueeze: (1x64x1x144x144xf32) <- (64x1x144x144xf32, 1xi64)
        t486 = paddle._C_ops.unsqueeze(t485, t305)
        del t485

        # pd_op.add: (-1x64x4x144x144xf32) <- (-1x64x4x144x144xf32, 1x64x1x144x144xf32)
        t487 = paddle._C_ops.add(t484, t486)
        del t484, t486

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t488 = [t460, t347, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t489 = paddle._C_ops.stack(t488, 0)
        del t488

        # pd_op.reshape: (-1x4x144x144xf32) <- (-1x64x4x144x144xf32, 4xi64)
        t490 = paddle._C_ops.reshape(t487, t489)
        del t487, t489

        # pd_op.softmax: (-1x4x144x144xf32) <- (-1x4x144x144xf32)
        t491 = paddle._C_ops.softmax(t490, -1)
        del t490

        # pd_op.matmul: (-1x4x144x32xf32) <- (-1x4x144x144xf32, -1x4x144x32xf32)
        t492 = paddle._C_ops.matmul(t491, t469, False, False)
        del t469, t491

        # pd_op.transpose: (-1x144x4x32xf32) <- (-1x4x144x32xf32)
        t493 = paddle._C_ops.transpose(t492, [0, 2, 1, 3])
        del t492

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t494 = [t460, t345, t325]
        del t460

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t495 = paddle._C_ops.stack(t494, 0)
        del t494

        # pd_op.reshape: (-1x144x128xf32) <- (-1x144x4x32xf32, 3xi64)
        t496 = paddle._C_ops.reshape(t493, t495)
        del t495, t493

        # pd_op.matmul: (-1x144x128xf32) <- (-1x144x128xf32, 128x128xf32)
        t497 = paddle._C_ops.matmul(t496, t20, False, False)
        del t20, t496

        # pd_op.add: (-1x144x128xf32) <- (-1x144x128xf32, 128xf32)
        t498 = paddle._C_ops.add(t497, t21)
        del t497, t21

        # pd_op.reshape: (-1x12x12x128xf32) <- (-1x144x128xf32, 4xi64)
        t499 = paddle._C_ops.reshape(t498, t337)
        del t498, t337

        # pd_op.reshape: (-1x8x8x12x12x128xf32) <- (-1x12x12x128xf32, 6xi64)
        t500 = paddle._C_ops.reshape(t499, t379)
        del t379, t499

        # pd_op.transpose: (-1x8x12x8x12x128xf32) <- (-1x8x8x12x12x128xf32)
        t501 = paddle._C_ops.transpose(t500, [0, 1, 3, 2, 4, 5])
        del t500

        # pd_op.reshape: (-1x96x96x128xf32) <- (-1x8x12x8x12x128xf32, 4xi64)
        t502 = paddle._C_ops.reshape(t501, t382)
        del t382, t501

        # pd_op.full_int_array: (2xi64) <- ()
        t503 = [6, 6]

        # pd_op.roll: (-1x96x96x128xf32) <- (-1x96x96x128xf32, 2xi64)
        t504 = paddle._C_ops.roll(t502, t503, [1, 2])
        del t502

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t505 = [t399, t384, t325]
        del t384, t399

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t506 = paddle._C_ops.stack(t505, 0)
        del t505

        # pd_op.reshape: (-1x9216x128xf32) <- (-1x96x96x128xf32, 3xi64)
        t507 = paddle._C_ops.reshape(t504, t506)
        del t504, t506

        # pd_op.add: (-1x9216x128xf32) <- (-1x9216x128xf32, -1x9216x128xf32)
        t508 = paddle._C_ops.add(t397, t507)
        del t397, t507

        # pd_op.layer_norm: (-1x9216x128xf32, -1x9216xf32, -1x9216xf32) <- (-1x9216x128xf32, 128xf32, 128xf32)
        t509, t510, t511 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t508, t22, t23, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t23, t22

        # pd_op.matmul: (-1x9216x512xf32) <- (-1x9216x128xf32, 128x512xf32)
        t512 = paddle._C_ops.matmul(t509, t24, False, False)
        del t509, t24

        # pd_op.add: (-1x9216x512xf32) <- (-1x9216x512xf32, 512xf32)
        t513 = paddle._C_ops.add(t512, t25)
        del t512, t25

        # pd_op.gelu: (-1x9216x512xf32) <- (-1x9216x512xf32)
        t514 = paddle._C_ops.gelu(t513, False)
        del t513

        # pd_op.matmul: (-1x9216x128xf32) <- (-1x9216x512xf32, 512x128xf32)
        t515 = paddle._C_ops.matmul(t514, t26, False, False)
        del t514, t26

        # pd_op.add: (-1x9216x128xf32) <- (-1x9216x128xf32, 128xf32)
        t516 = paddle._C_ops.add(t515, t27)
        del t515, t27

        # pd_op.add: (-1x9216x128xf32) <- (-1x9216x128xf32, -1x9216x128xf32)
        t517 = paddle._C_ops.add(t508, t516)
        del t508, t516

        # pd_op.shape64: (3xi64) <- (-1x9216x128xf32)
        t518 = paddle._C_ops.shape64(t517)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t519 = paddle._C_ops.slice(t518, [0], t305, t306, [1], [0])
        del t518

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t520 = [t519, t324, t324, t325]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t521 = paddle._C_ops.stack(t520, 0)
        del t520

        # pd_op.reshape: (-1x96x96x128xf32) <- (-1x9216x128xf32, 4xi64)
        t522 = paddle._C_ops.reshape(t517, t521)
        del t517, t521

        # pd_op.full_int_array: (2xi64) <- ()
        t523 = [2, 2]

        # pd_op.strided_slice: (-1x48x48x128xf32) <- (-1x96x96x128xf32, 2xi64, 2xi64, 2xi64)
        t524 = paddle._C_ops.strided_slice(t522, [1, 2], t419, t440, t523)

        # pd_op.full_int_array: (2xi64) <- ()
        t525 = [1, 0]

        # pd_op.strided_slice: (-1x48x48x128xf32) <- (-1x96x96x128xf32, 2xi64, 2xi64, 2xi64)
        t526 = paddle._C_ops.strided_slice(t522, [1, 2], t525, t440, t523)

        # pd_op.full_int_array: (2xi64) <- ()
        t527 = [0, 1]

        # pd_op.strided_slice: (-1x48x48x128xf32) <- (-1x96x96x128xf32, 2xi64, 2xi64, 2xi64)
        t528 = paddle._C_ops.strided_slice(t522, [1, 2], t527, t440, t523)

        # pd_op.strided_slice: (-1x48x48x128xf32) <- (-1x96x96x128xf32, 2xi64, 2xi64, 2xi64)
        t529 = paddle._C_ops.strided_slice(t522, [1, 2], t421, t440, t523)

        # pd_op.shape64: (4xi64) <- (-1x96x96x128xf32)
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

        # pd_op.reshape: (-1x96x96x128xf32) <- (-1x96x96x128xf32, 4xi64)
        t534 = paddle._C_ops.reshape(t522, t533)
        del t522, t533

        # pd_op.full: (1xi32) <- ()
        t535 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x48x48x128xf32, -1x48x48x128xf32, -1x48x48x128xf32, -1x48x48x128xf32]) <- (-1x48x48x128xf32, -1x48x48x128xf32, -1x48x48x128xf32, -1x48x48x128xf32)
        t536 = [t524, t526, t528, t529]
        del t524, t526, t528, t529

        # pd_op.concat: (-1x48x48x512xf32) <- ([-1x48x48x128xf32, -1x48x48x128xf32, -1x48x48x128xf32, -1x48x48x128xf32], 1xi32)
        t537 = paddle._C_ops.concat(t536, t535)
        del t536

        # pd_op.full: (xi64) <- ()
        t538 = paddle._C_ops.full([], float("-1"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t539 = paddle._C_ops.full(
            [], float("512"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t540 = [t519, t538, t539]
        del t519

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t541 = paddle._C_ops.stack(t540, 0)
        del t540

        # pd_op.reshape: (-1x-1x512xf32) <- (-1x48x48x512xf32, 3xi64)
        t542 = paddle._C_ops.reshape(t537, t541)
        del t537, t541

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        t543, t544, t545 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t542, t28, t29, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t29, t28, t542

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        t546 = paddle._C_ops.matmul(t543, t30, False, False)
        del t543, t30

        # pd_op.shape64: (3xi64) <- (-1x-1x256xf32)
        t547 = paddle._C_ops.shape64(t546)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t548 = paddle._C_ops.slice(t547, [0], t305, t306, [1], [0])
        del t547

        # pd_op.shape64: (3xi64) <- (-1x-1x256xf32)
        t549 = paddle._C_ops.shape64(t546)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t550 = paddle._C_ops.slice(t549, [0], t306, t354, [1], [0])
        del t549

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        t551, t552, t553 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t546, t31, t32, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t32, t31

        # pd_op.full: (xi64) <- ()
        t554 = paddle._C_ops.full([], float("48"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t555 = paddle._C_ops.full(
            [], float("256"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t556 = [t548, t554, t554, t555]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t557 = paddle._C_ops.stack(t556, 0)
        del t556

        # pd_op.reshape: (-1x48x48x256xf32) <- (-1x-1x256xf32, 4xi64)
        t558 = paddle._C_ops.reshape(t551, t557)
        del t551, t557

        # pd_op.shape64: (4xi64) <- (-1x48x48x256xf32)
        t559 = paddle._C_ops.shape64(t558)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t560 = paddle._C_ops.slice(t559, [0], t305, t306, [1], [0])
        del t559

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t561 = [t560, t347, t332, t347, t332, t555]
        del t560

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t562 = paddle._C_ops.stack(t561, 0)
        del t561

        # pd_op.reshape: (-1x4x12x4x12x256xf32) <- (-1x48x48x256xf32, 6xi64)
        t563 = paddle._C_ops.reshape(t558, t562)
        del t558, t562

        # pd_op.transpose: (-1x4x4x12x12x256xf32) <- (-1x4x12x4x12x256xf32)
        t564 = paddle._C_ops.transpose(t563, [0, 1, 3, 2, 4, 5])
        del t563

        # pd_op.full_int_array: (4xi64) <- ()
        t565 = [-1, 12, 12, 256]

        # pd_op.reshape: (-1x12x12x256xf32) <- (-1x4x4x12x12x256xf32, 4xi64)
        t566 = paddle._C_ops.reshape(t564, t565)
        del t564

        # pd_op.full_int_array: (3xi64) <- ()
        t567 = [-1, 144, 256]

        # pd_op.reshape: (-1x144x256xf32) <- (-1x12x12x256xf32, 3xi64)
        t568 = paddle._C_ops.reshape(t566, t567)
        del t566

        # pd_op.shape64: (3xi64) <- (-1x144x256xf32)
        t569 = paddle._C_ops.shape64(t568)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t570 = paddle._C_ops.slice(t569, [0], t305, t306, [1], [0])
        del t569

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x256xf32, 256x768xf32)
        t571 = paddle._C_ops.matmul(t568, t33, False, False)
        del t33, t568

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t572 = paddle._C_ops.add(t571, t34)
        del t571, t34

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t573 = [t570, t345, t346, t331, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t574 = paddle._C_ops.stack(t573, 0)
        del t573

        # pd_op.reshape: (-1x144x3x8x32xf32) <- (-1x144x768xf32, 5xi64)
        t575 = paddle._C_ops.reshape(t572, t574)
        del t572, t574

        # pd_op.transpose: (3x-1x8x144x32xf32) <- (-1x144x3x8x32xf32)
        t576 = paddle._C_ops.transpose(t575, [2, 0, 3, 1, 4])
        del t575

        # pd_op.slice: (-1x8x144x32xf32) <- (3x-1x8x144x32xf32, 1xi64, 1xi64)
        t577 = paddle._C_ops.slice(t576, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x8x144x32xf32) <- (3x-1x8x144x32xf32, 1xi64, 1xi64)
        t578 = paddle._C_ops.slice(t576, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x8x144x32xf32) <- (3x-1x8x144x32xf32, 1xi64, 1xi64)
        t579 = paddle._C_ops.slice(t576, [0], t354, t356, [1], [0])
        del t576

        # pd_op.scale: (-1x8x144x32xf32) <- (-1x8x144x32xf32, 1xf32)
        t580 = paddle._C_ops.scale(t577, t358, float("0"), True)
        del t577

        # pd_op.transpose: (-1x8x32x144xf32) <- (-1x8x144x32xf32)
        t581 = paddle._C_ops.transpose(t578, [0, 1, 3, 2])
        del t578

        # pd_op.matmul: (-1x8x144x144xf32) <- (-1x8x144x32xf32, -1x8x32x144xf32)
        t582 = paddle._C_ops.matmul(t580, t581, False, False)
        del t580, t581

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t583 = paddle._C_ops.reshape(input5, t362)
        del input5

        # pd_op.index_select: (20736x8xf32) <- (529x8xf32, 20736xi64)
        t584 = paddle._C_ops.index_select(input6, t583, 0)
        del input6, t583

        # pd_op.reshape: (144x144x8xf32) <- (20736x8xf32, 3xi64)
        t585 = paddle._C_ops.reshape(t584, t365)
        del t584

        # pd_op.transpose: (8x144x144xf32) <- (144x144x8xf32)
        t586 = paddle._C_ops.transpose(t585, [2, 0, 1])
        del t585

        # pd_op.unsqueeze: (1x8x144x144xf32) <- (8x144x144xf32, 1xi64)
        t587 = paddle._C_ops.unsqueeze(t586, t305)
        del t586

        # pd_op.add: (-1x8x144x144xf32) <- (-1x8x144x144xf32, 1x8x144x144xf32)
        t588 = paddle._C_ops.add(t582, t587)
        del t582, t587

        # pd_op.softmax: (-1x8x144x144xf32) <- (-1x8x144x144xf32)
        t589 = paddle._C_ops.softmax(t588, -1)
        del t588

        # pd_op.matmul: (-1x8x144x32xf32) <- (-1x8x144x144xf32, -1x8x144x32xf32)
        t590 = paddle._C_ops.matmul(t589, t579, False, False)
        del t579, t589

        # pd_op.transpose: (-1x144x8x32xf32) <- (-1x8x144x32xf32)
        t591 = paddle._C_ops.transpose(t590, [0, 2, 1, 3])
        del t590

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t592 = [t570, t345, t555]
        del t570

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t593 = paddle._C_ops.stack(t592, 0)
        del t592

        # pd_op.reshape: (-1x144x256xf32) <- (-1x144x8x32xf32, 3xi64)
        t594 = paddle._C_ops.reshape(t591, t593)
        del t593, t591

        # pd_op.matmul: (-1x144x256xf32) <- (-1x144x256xf32, 256x256xf32)
        t595 = paddle._C_ops.matmul(t594, t35, False, False)
        del t35, t594

        # pd_op.add: (-1x144x256xf32) <- (-1x144x256xf32, 256xf32)
        t596 = paddle._C_ops.add(t595, t36)
        del t595, t36

        # pd_op.reshape: (-1x12x12x256xf32) <- (-1x144x256xf32, 4xi64)
        t597 = paddle._C_ops.reshape(t596, t565)
        del t596

        # pd_op.full_int_array: (6xi64) <- ()
        t598 = [-1, 4, 4, 12, 12, 256]

        # pd_op.reshape: (-1x4x4x12x12x256xf32) <- (-1x12x12x256xf32, 6xi64)
        t599 = paddle._C_ops.reshape(t597, t598)
        del t597

        # pd_op.transpose: (-1x4x12x4x12x256xf32) <- (-1x4x4x12x12x256xf32)
        t600 = paddle._C_ops.transpose(t599, [0, 1, 3, 2, 4, 5])
        del t599

        # pd_op.full_int_array: (4xi64) <- ()
        t601 = [-1, 48, 48, 256]

        # pd_op.reshape: (-1x48x48x256xf32) <- (-1x4x12x4x12x256xf32, 4xi64)
        t602 = paddle._C_ops.reshape(t600, t601)
        del t600

        # pd_op.full: (xi64) <- ()
        t603 = paddle._C_ops.full(
            [], float("2304"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t604 = [t548, t603, t555]
        del t548

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t605 = paddle._C_ops.stack(t604, 0)
        del t604

        # pd_op.reshape: (-1x2304x256xf32) <- (-1x48x48x256xf32, 3xi64)
        t606 = paddle._C_ops.reshape(t602, t605)
        del t602, t605

        # pd_op.add: (-1x2304x256xf32) <- (-1x-1x256xf32, -1x2304x256xf32)
        t607 = paddle._C_ops.add(t546, t606)
        del t546, t606

        # pd_op.layer_norm: (-1x2304x256xf32, -1x2304xf32, -1x2304xf32) <- (-1x2304x256xf32, 256xf32, 256xf32)
        t608, t609, t610 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t607, t37, t38, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t38, t37

        # pd_op.matmul: (-1x2304x1024xf32) <- (-1x2304x256xf32, 256x1024xf32)
        t611 = paddle._C_ops.matmul(t608, t39, False, False)
        del t608, t39

        # pd_op.add: (-1x2304x1024xf32) <- (-1x2304x1024xf32, 1024xf32)
        t612 = paddle._C_ops.add(t611, t40)
        del t611, t40

        # pd_op.gelu: (-1x2304x1024xf32) <- (-1x2304x1024xf32)
        t613 = paddle._C_ops.gelu(t612, False)
        del t612

        # pd_op.matmul: (-1x2304x256xf32) <- (-1x2304x1024xf32, 1024x256xf32)
        t614 = paddle._C_ops.matmul(t613, t41, False, False)
        del t613, t41

        # pd_op.add: (-1x2304x256xf32) <- (-1x2304x256xf32, 256xf32)
        t615 = paddle._C_ops.add(t614, t42)
        del t614, t42

        # pd_op.add: (-1x2304x256xf32) <- (-1x2304x256xf32, -1x2304x256xf32)
        t616 = paddle._C_ops.add(t607, t615)
        del t607, t615

        # pd_op.shape64: (3xi64) <- (-1x2304x256xf32)
        t617 = paddle._C_ops.shape64(t616)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t618 = paddle._C_ops.slice(t617, [0], t305, t306, [1], [0])
        del t617

        # pd_op.layer_norm: (-1x2304x256xf32, -1x2304xf32, -1x2304xf32) <- (-1x2304x256xf32, 256xf32, 256xf32)
        t619, t620, t621 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t616, t43, t44, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t44, t43

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t622 = [t618, t554, t554, t555]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t623 = paddle._C_ops.stack(t622, 0)
        del t622

        # pd_op.reshape: (-1x48x48x256xf32) <- (-1x2304x256xf32, 4xi64)
        t624 = paddle._C_ops.reshape(t619, t623)
        del t619, t623

        # pd_op.shape64: (4xi64) <- (-1x48x48x256xf32)
        t625 = paddle._C_ops.shape64(t624)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t626 = paddle._C_ops.slice(t625, [0], t305, t306, [1], [0])
        del t625

        # pd_op.roll: (-1x48x48x256xf32) <- (-1x48x48x256xf32, 2xi64)
        t627 = paddle._C_ops.roll(t624, t408, [1, 2])
        del t624

        # pd_op.shape64: (4xi64) <- (-1x48x48x256xf32)
        t628 = paddle._C_ops.shape64(t627)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t629 = paddle._C_ops.slice(t628, [0], t305, t306, [1], [0])
        del t628

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t630 = [t629, t347, t332, t347, t332, t555]
        del t629

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t631 = paddle._C_ops.stack(t630, 0)
        del t630

        # pd_op.reshape: (-1x4x12x4x12x256xf32) <- (-1x48x48x256xf32, 6xi64)
        t632 = paddle._C_ops.reshape(t627, t631)
        del t627, t631

        # pd_op.transpose: (-1x4x4x12x12x256xf32) <- (-1x4x12x4x12x256xf32)
        t633 = paddle._C_ops.transpose(t632, [0, 1, 3, 2, 4, 5])
        del t632

        # pd_op.reshape: (-1x12x12x256xf32) <- (-1x4x4x12x12x256xf32, 4xi64)
        t634 = paddle._C_ops.reshape(t633, t565)
        del t633

        # pd_op.reshape: (-1x144x256xf32) <- (-1x12x12x256xf32, 3xi64)
        t635 = paddle._C_ops.reshape(t634, t567)
        del t567, t634

        # pd_op.full: (1x48x48x1xf32) <- ()
        t636 = paddle._C_ops.full(
            [1, 48, 48, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t637 = paddle._C_ops.set_value_(
            t636, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t636

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t638 = paddle._C_ops.set_value_(
            t637, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t637

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t639 = paddle._C_ops.set_value_(
            t638, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t638

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t640 = paddle._C_ops.set_value_(
            t639, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t639

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t641 = paddle._C_ops.set_value_(
            t640, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t640

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t642 = paddle._C_ops.set_value_(
            t641, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t641

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t643 = paddle._C_ops.set_value_(
            t642, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t642

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t644 = paddle._C_ops.set_value_(
            t643, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t643

        # pd_op.set_value_: (1x48x48x1xf32) <- (1x48x48x1xf32, 2xi64, 2xi64, 2xi64)
        t645 = paddle._C_ops.set_value_(
            t644, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t644

        # pd_op.full_int_array: (6xi64) <- ()
        t646 = [1, 4, 12, 4, 12, 1]

        # pd_op.reshape: (1x4x12x4x12x1xf32) <- (1x48x48x1xf32, 6xi64)
        t647 = paddle._C_ops.reshape(t645, t646)
        del t646

        # pd_op.transpose: (1x4x4x12x12x1xf32) <- (1x4x12x4x12x1xf32)
        t648 = paddle._C_ops.transpose(t647, [0, 1, 3, 2, 4, 5])
        del t647

        # pd_op.reshape: (16x12x12x1xf32) <- (1x4x4x12x12x1xf32, 4xi64)
        t649 = paddle._C_ops.reshape(t648, t445)
        del t648

        # pd_op.reshape: (16x144xf32) <- (16x12x12x1xf32, 2xi64)
        t650 = paddle._C_ops.reshape(t649, t447)
        del t649

        # pd_op.unsqueeze: (16x1x144xf32) <- (16x144xf32, 1xi64)
        t651 = paddle._C_ops.unsqueeze(t650, t306)

        # pd_op.unsqueeze: (16x144x1xf32) <- (16x144xf32, 1xi64)
        t652 = paddle._C_ops.unsqueeze(t650, t354)
        del t650

        # pd_op.subtract: (16x144x144xf32) <- (16x1x144xf32, 16x144x1xf32)
        t653 = paddle._C_ops.subtract(t651, t652)
        del t651, t652

        # pd_op.not_equal: (16x144x144xb) <- (16x144x144xf32, xf32)
        t654 = paddle._C_ops.not_equal(t653, t452)

        # pd_op.full: (16x144x144xf32) <- ()
        t655 = paddle._C_ops.full(
            [16, 144, 144],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x144x144xf32) <- (16x144x144xb, 16x144x144xf32, 16x144x144xf32)
        t656 = paddle._C_ops.where(t654, t655, t653)
        del t655, t654, t653

        # pd_op.equal: (16x144x144xb) <- (16x144x144xf32, xf32)
        t657 = paddle._C_ops.equal(t656, t452)

        # pd_op.full: (16x144x144xf32) <- ()
        t658 = paddle._C_ops.full(
            [16, 144, 144],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x144x144xf32) <- (16x144x144xb, 16x144x144xf32, 16x144x144xf32)
        t659 = paddle._C_ops.where(t657, t658, t656)
        del t657, t658, t656

        # pd_op.shape64: (3xi64) <- (-1x144x256xf32)
        t660 = paddle._C_ops.shape64(t635)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t661 = paddle._C_ops.slice(t660, [0], t305, t306, [1], [0])
        del t660

        # pd_op.matmul: (-1x144x768xf32) <- (-1x144x256xf32, 256x768xf32)
        t662 = paddle._C_ops.matmul(t635, t45, False, False)
        del t45, t635

        # pd_op.add: (-1x144x768xf32) <- (-1x144x768xf32, 768xf32)
        t663 = paddle._C_ops.add(t662, t46)
        del t662, t46

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t664 = [t661, t345, t346, t331, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t665 = paddle._C_ops.stack(t664, 0)
        del t664

        # pd_op.reshape: (-1x144x3x8x32xf32) <- (-1x144x768xf32, 5xi64)
        t666 = paddle._C_ops.reshape(t663, t665)
        del t663, t665

        # pd_op.transpose: (3x-1x8x144x32xf32) <- (-1x144x3x8x32xf32)
        t667 = paddle._C_ops.transpose(t666, [2, 0, 3, 1, 4])
        del t666

        # pd_op.slice: (-1x8x144x32xf32) <- (3x-1x8x144x32xf32, 1xi64, 1xi64)
        t668 = paddle._C_ops.slice(t667, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x8x144x32xf32) <- (3x-1x8x144x32xf32, 1xi64, 1xi64)
        t669 = paddle._C_ops.slice(t667, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x8x144x32xf32) <- (3x-1x8x144x32xf32, 1xi64, 1xi64)
        t670 = paddle._C_ops.slice(t667, [0], t354, t356, [1], [0])
        del t667

        # pd_op.scale: (-1x8x144x32xf32) <- (-1x8x144x32xf32, 1xf32)
        t671 = paddle._C_ops.scale(t668, t358, float("0"), True)
        del t668

        # pd_op.transpose: (-1x8x32x144xf32) <- (-1x8x144x32xf32)
        t672 = paddle._C_ops.transpose(t669, [0, 1, 3, 2])
        del t669

        # pd_op.matmul: (-1x8x144x144xf32) <- (-1x8x144x32xf32, -1x8x32x144xf32)
        t673 = paddle._C_ops.matmul(t671, t672, False, False)
        del t671, t672

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t674 = paddle._C_ops.reshape(input7, t362)
        del input7

        # pd_op.index_select: (20736x8xf32) <- (529x8xf32, 20736xi64)
        t675 = paddle._C_ops.index_select(input8, t674, 0)
        del input8, t674

        # pd_op.reshape: (144x144x8xf32) <- (20736x8xf32, 3xi64)
        t676 = paddle._C_ops.reshape(t675, t365)
        del t675

        # pd_op.transpose: (8x144x144xf32) <- (144x144x8xf32)
        t677 = paddle._C_ops.transpose(t676, [2, 0, 1])
        del t676

        # pd_op.unsqueeze: (1x8x144x144xf32) <- (8x144x144xf32, 1xi64)
        t678 = paddle._C_ops.unsqueeze(t677, t305)
        del t677

        # pd_op.add: (-1x8x144x144xf32) <- (-1x8x144x144xf32, 1x8x144x144xf32)
        t679 = paddle._C_ops.add(t673, t678)
        del t673, t678

        # pd_op.full: (xi64) <- ()
        t680 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t681 = paddle._C_ops.floor_divide(t661, t680)
        del t680

        # pd_op.full: (xi64) <- ()
        t682 = paddle._C_ops.full([], float("16"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t683 = [t681, t682, t331, t345, t345]
        del t681

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t684 = paddle._C_ops.stack(t683, 0)
        del t683

        # pd_op.reshape: (-1x16x8x144x144xf32) <- (-1x8x144x144xf32, 5xi64)
        t685 = paddle._C_ops.reshape(t679, t684)
        del t679, t684

        # pd_op.unsqueeze: (16x1x144x144xf32) <- (16x144x144xf32, 1xi64)
        t686 = paddle._C_ops.unsqueeze(t659, t306)
        del t659

        # pd_op.unsqueeze: (1x16x1x144x144xf32) <- (16x1x144x144xf32, 1xi64)
        t687 = paddle._C_ops.unsqueeze(t686, t305)
        del t686

        # pd_op.add: (-1x16x8x144x144xf32) <- (-1x16x8x144x144xf32, 1x16x1x144x144xf32)
        t688 = paddle._C_ops.add(t685, t687)
        del t685, t687

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t689 = [t661, t331, t345, t345]
        del t331

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t690 = paddle._C_ops.stack(t689, 0)
        del t689

        # pd_op.reshape: (-1x8x144x144xf32) <- (-1x16x8x144x144xf32, 4xi64)
        t691 = paddle._C_ops.reshape(t688, t690)
        del t688, t690

        # pd_op.softmax: (-1x8x144x144xf32) <- (-1x8x144x144xf32)
        t692 = paddle._C_ops.softmax(t691, -1)
        del t691

        # pd_op.matmul: (-1x8x144x32xf32) <- (-1x8x144x144xf32, -1x8x144x32xf32)
        t693 = paddle._C_ops.matmul(t692, t670, False, False)
        del t670, t692

        # pd_op.transpose: (-1x144x8x32xf32) <- (-1x8x144x32xf32)
        t694 = paddle._C_ops.transpose(t693, [0, 2, 1, 3])
        del t693

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t695 = [t661, t345, t555]
        del t661

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t696 = paddle._C_ops.stack(t695, 0)
        del t695

        # pd_op.reshape: (-1x144x256xf32) <- (-1x144x8x32xf32, 3xi64)
        t697 = paddle._C_ops.reshape(t694, t696)
        del t696, t694

        # pd_op.matmul: (-1x144x256xf32) <- (-1x144x256xf32, 256x256xf32)
        t698 = paddle._C_ops.matmul(t697, t47, False, False)
        del t47, t697

        # pd_op.add: (-1x144x256xf32) <- (-1x144x256xf32, 256xf32)
        t699 = paddle._C_ops.add(t698, t48)
        del t698, t48

        # pd_op.reshape: (-1x12x12x256xf32) <- (-1x144x256xf32, 4xi64)
        t700 = paddle._C_ops.reshape(t699, t565)
        del t699, t565

        # pd_op.reshape: (-1x4x4x12x12x256xf32) <- (-1x12x12x256xf32, 6xi64)
        t701 = paddle._C_ops.reshape(t700, t598)
        del t598, t700

        # pd_op.transpose: (-1x4x12x4x12x256xf32) <- (-1x4x4x12x12x256xf32)
        t702 = paddle._C_ops.transpose(t701, [0, 1, 3, 2, 4, 5])
        del t701

        # pd_op.reshape: (-1x48x48x256xf32) <- (-1x4x12x4x12x256xf32, 4xi64)
        t703 = paddle._C_ops.reshape(t702, t601)
        del t601, t702

        # pd_op.roll: (-1x48x48x256xf32) <- (-1x48x48x256xf32, 2xi64)
        t704 = paddle._C_ops.roll(t703, t503, [1, 2])
        del t703

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t705 = [t618, t603, t555]
        del t603, t618

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t706 = paddle._C_ops.stack(t705, 0)
        del t705

        # pd_op.reshape: (-1x2304x256xf32) <- (-1x48x48x256xf32, 3xi64)
        t707 = paddle._C_ops.reshape(t704, t706)
        del t704, t706

        # pd_op.add: (-1x2304x256xf32) <- (-1x2304x256xf32, -1x2304x256xf32)
        t708 = paddle._C_ops.add(t616, t707)
        del t616, t707

        # pd_op.layer_norm: (-1x2304x256xf32, -1x2304xf32, -1x2304xf32) <- (-1x2304x256xf32, 256xf32, 256xf32)
        t709, t710, t711 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t708, t49, t50, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t50, t49

        # pd_op.matmul: (-1x2304x1024xf32) <- (-1x2304x256xf32, 256x1024xf32)
        t712 = paddle._C_ops.matmul(t709, t51, False, False)
        del t709, t51

        # pd_op.add: (-1x2304x1024xf32) <- (-1x2304x1024xf32, 1024xf32)
        t713 = paddle._C_ops.add(t712, t52)
        del t712, t52

        # pd_op.gelu: (-1x2304x1024xf32) <- (-1x2304x1024xf32)
        t714 = paddle._C_ops.gelu(t713, False)
        del t713

        # pd_op.matmul: (-1x2304x256xf32) <- (-1x2304x1024xf32, 1024x256xf32)
        t715 = paddle._C_ops.matmul(t714, t53, False, False)
        del t714, t53

        # pd_op.add: (-1x2304x256xf32) <- (-1x2304x256xf32, 256xf32)
        t716 = paddle._C_ops.add(t715, t54)
        del t715, t54

        # pd_op.add: (-1x2304x256xf32) <- (-1x2304x256xf32, -1x2304x256xf32)
        t717 = paddle._C_ops.add(t708, t716)
        del t708, t716

        # pd_op.shape64: (3xi64) <- (-1x2304x256xf32)
        t718 = paddle._C_ops.shape64(t717)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t719 = paddle._C_ops.slice(t718, [0], t305, t306, [1], [0])
        del t718

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t720 = [t719, t554, t554, t555]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t721 = paddle._C_ops.stack(t720, 0)
        del t720

        # pd_op.reshape: (-1x48x48x256xf32) <- (-1x2304x256xf32, 4xi64)
        t722 = paddle._C_ops.reshape(t717, t721)
        del t717, t721

        # pd_op.strided_slice: (-1x24x24x256xf32) <- (-1x48x48x256xf32, 2xi64, 2xi64, 2xi64)
        t723 = paddle._C_ops.strided_slice(t722, [1, 2], t419, t440, t523)

        # pd_op.strided_slice: (-1x24x24x256xf32) <- (-1x48x48x256xf32, 2xi64, 2xi64, 2xi64)
        t724 = paddle._C_ops.strided_slice(t722, [1, 2], t525, t440, t523)

        # pd_op.strided_slice: (-1x24x24x256xf32) <- (-1x48x48x256xf32, 2xi64, 2xi64, 2xi64)
        t725 = paddle._C_ops.strided_slice(t722, [1, 2], t527, t440, t523)

        # pd_op.strided_slice: (-1x24x24x256xf32) <- (-1x48x48x256xf32, 2xi64, 2xi64, 2xi64)
        t726 = paddle._C_ops.strided_slice(t722, [1, 2], t421, t440, t523)

        # pd_op.shape64: (4xi64) <- (-1x48x48x256xf32)
        t727 = paddle._C_ops.shape64(t722)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t728 = paddle._C_ops.slice(t727, [0], t305, t306, [1], [0])
        del t727

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t729 = [t728, t554, t554, t555]
        del t554, t555, t728

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t730 = paddle._C_ops.stack(t729, 0)
        del t729

        # pd_op.reshape: (-1x48x48x256xf32) <- (-1x48x48x256xf32, 4xi64)
        t731 = paddle._C_ops.reshape(t722, t730)
        del t722, t730

        # builtin.combine: ([-1x24x24x256xf32, -1x24x24x256xf32, -1x24x24x256xf32, -1x24x24x256xf32]) <- (-1x24x24x256xf32, -1x24x24x256xf32, -1x24x24x256xf32, -1x24x24x256xf32)
        t732 = [t723, t724, t725, t726]
        del t723, t724, t725, t726

        # pd_op.concat: (-1x24x24x1024xf32) <- ([-1x24x24x256xf32, -1x24x24x256xf32, -1x24x24x256xf32, -1x24x24x256xf32], 1xi32)
        t733 = paddle._C_ops.concat(t732, t535)
        del t732

        # pd_op.full: (xi64) <- ()
        t734 = paddle._C_ops.full(
            [], float("1024"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t735 = [t719, t538, t734]
        del t719

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t736 = paddle._C_ops.stack(t735, 0)
        del t735

        # pd_op.reshape: (-1x-1x1024xf32) <- (-1x24x24x1024xf32, 3xi64)
        t737 = paddle._C_ops.reshape(t733, t736)
        del t733, t736

        # pd_op.layer_norm: (-1x-1x1024xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x1024xf32, 1024xf32, 1024xf32)
        t738, t739, t740 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t737, t55, t56, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t56, t55, t737

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        t741 = paddle._C_ops.matmul(t738, t57, False, False)
        del t738, t57

        # pd_op.shape64: (3xi64) <- (-1x-1x512xf32)
        t742 = paddle._C_ops.shape64(t741)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t743 = paddle._C_ops.slice(t742, [0], t305, t306, [1], [0])
        del t742

        # pd_op.shape64: (3xi64) <- (-1x-1x512xf32)
        t744 = paddle._C_ops.shape64(t741)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t745 = paddle._C_ops.slice(t744, [0], t306, t354, [1], [0])
        del t744

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        t746, t747, t748 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t741, t58, t59, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t59, t58

        # pd_op.full: (xi64) <- ()
        t749 = paddle._C_ops.full([], float("24"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t750 = [t743, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t751 = paddle._C_ops.stack(t750, 0)
        del t750

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x-1x512xf32, 4xi64)
        t752 = paddle._C_ops.reshape(t746, t751)
        del t746, t751

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t753 = paddle._C_ops.shape64(t752)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t754 = paddle._C_ops.slice(t753, [0], t305, t306, [1], [0])
        del t753

        # pd_op.full: (xi64) <- ()
        t755 = paddle._C_ops.full([], float("2"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t756 = [t754, t755, t332, t755, t332, t539]
        del t754

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t757 = paddle._C_ops.stack(t756, 0)
        del t756

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t758 = paddle._C_ops.reshape(t752, t757)
        del t752, t757

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t759 = paddle._C_ops.transpose(t758, [0, 1, 3, 2, 4, 5])
        del t758

        # pd_op.full_int_array: (4xi64) <- ()
        t760 = [-1, 12, 12, 512]

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t761 = paddle._C_ops.reshape(t759, t760)
        del t759

        # pd_op.full_int_array: (3xi64) <- ()
        t762 = [-1, 144, 512]

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t763 = paddle._C_ops.reshape(t761, t762)
        del t761

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t764 = paddle._C_ops.shape64(t763)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t765 = paddle._C_ops.slice(t764, [0], t305, t306, [1], [0])
        del t764

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t766 = paddle._C_ops.matmul(t763, t60, False, False)
        del t60, t763

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t767 = paddle._C_ops.add(t766, t61)
        del t766, t61

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t768 = [t765, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t769 = paddle._C_ops.stack(t768, 0)
        del t768

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t770 = paddle._C_ops.reshape(t767, t769)
        del t767, t769

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t771 = paddle._C_ops.transpose(t770, [2, 0, 3, 1, 4])
        del t770

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t772 = paddle._C_ops.slice(t771, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t773 = paddle._C_ops.slice(t771, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t774 = paddle._C_ops.slice(t771, [0], t354, t356, [1], [0])
        del t771

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t775 = paddle._C_ops.scale(t772, t358, float("0"), True)
        del t772

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t776 = paddle._C_ops.transpose(t773, [0, 1, 3, 2])
        del t773

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t777 = paddle._C_ops.matmul(t775, t776, False, False)
        del t775, t776

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t778 = paddle._C_ops.reshape(input9, t362)
        del input9

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t779 = paddle._C_ops.index_select(input10, t778, 0)
        del input10, t778

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t780 = paddle._C_ops.reshape(t779, t365)
        del t779

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t781 = paddle._C_ops.transpose(t780, [2, 0, 1])
        del t780

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t782 = paddle._C_ops.unsqueeze(t781, t305)
        del t781

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t783 = paddle._C_ops.add(t777, t782)
        del t777, t782

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t784 = paddle._C_ops.softmax(t783, -1)
        del t783

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t785 = paddle._C_ops.matmul(t784, t774, False, False)
        del t774, t784

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t786 = paddle._C_ops.transpose(t785, [0, 2, 1, 3])
        del t785

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t787 = [t765, t345, t539]
        del t765

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t788 = paddle._C_ops.stack(t787, 0)
        del t787

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t789 = paddle._C_ops.reshape(t786, t788)
        del t788, t786

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t790 = paddle._C_ops.matmul(t789, t62, False, False)
        del t62, t789

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t791 = paddle._C_ops.add(t790, t63)
        del t790, t63

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t792 = paddle._C_ops.reshape(t791, t760)
        del t791

        # pd_op.full_int_array: (6xi64) <- ()
        t793 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t794 = paddle._C_ops.reshape(t792, t793)
        del t792

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t795 = paddle._C_ops.transpose(t794, [0, 1, 3, 2, 4, 5])
        del t794

        # pd_op.full_int_array: (4xi64) <- ()
        t796 = [-1, 24, 24, 512]

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t797 = paddle._C_ops.reshape(t795, t796)
        del t795

        # pd_op.full: (xi64) <- ()
        t798 = paddle._C_ops.full(
            [], float("576"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t799 = [t743, t798, t539]
        del t743

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t800 = paddle._C_ops.stack(t799, 0)
        del t799

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t801 = paddle._C_ops.reshape(t797, t800)
        del t797, t800

        # pd_op.add: (-1x576x512xf32) <- (-1x-1x512xf32, -1x576x512xf32)
        t802 = paddle._C_ops.add(t741, t801)
        del t741, t801

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t803, t804, t805 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t802, t64, t65, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t65, t64

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t806 = paddle._C_ops.matmul(t803, t66, False, False)
        del t803, t66

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t807 = paddle._C_ops.add(t806, t67)
        del t806, t67

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t808 = paddle._C_ops.gelu(t807, False)
        del t807

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t809 = paddle._C_ops.matmul(t808, t68, False, False)
        del t808, t68

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t810 = paddle._C_ops.add(t809, t69)
        del t809, t69

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t811 = paddle._C_ops.add(t802, t810)
        del t802, t810

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t812 = paddle._C_ops.shape64(t811)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t813 = paddle._C_ops.slice(t812, [0], t305, t306, [1], [0])
        del t812

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t814, t815, t816 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t811, t70, t71, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t71, t70

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t817 = [t813, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t818 = paddle._C_ops.stack(t817, 0)
        del t817

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t819 = paddle._C_ops.reshape(t814, t818)
        del t814, t818

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t820 = paddle._C_ops.shape64(t819)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t821 = paddle._C_ops.slice(t820, [0], t305, t306, [1], [0])
        del t820

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t822 = paddle._C_ops.roll(t819, t408, [1, 2])
        del t819

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t823 = paddle._C_ops.shape64(t822)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t824 = paddle._C_ops.slice(t823, [0], t305, t306, [1], [0])
        del t823

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t825 = [t824, t755, t332, t755, t332, t539]
        del t824

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t826 = paddle._C_ops.stack(t825, 0)
        del t825

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t827 = paddle._C_ops.reshape(t822, t826)
        del t822, t826

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t828 = paddle._C_ops.transpose(t827, [0, 1, 3, 2, 4, 5])
        del t827

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t829 = paddle._C_ops.reshape(t828, t760)
        del t828

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t830 = paddle._C_ops.reshape(t829, t762)
        del t829

        # pd_op.full: (1x24x24x1xf32) <- ()
        t831 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t832 = paddle._C_ops.set_value_(
            t831, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t831

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t833 = paddle._C_ops.set_value_(
            t832, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t832

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t834 = paddle._C_ops.set_value_(
            t833, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t833

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t835 = paddle._C_ops.set_value_(
            t834, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t834

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t836 = paddle._C_ops.set_value_(
            t835, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t835

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t837 = paddle._C_ops.set_value_(
            t836, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t836

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t838 = paddle._C_ops.set_value_(
            t837, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t837

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t839 = paddle._C_ops.set_value_(
            t838, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t838

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t840 = paddle._C_ops.set_value_(
            t839, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t839

        # pd_op.full_int_array: (6xi64) <- ()
        t841 = [1, 2, 12, 2, 12, 1]

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t842 = paddle._C_ops.reshape(t840, t841)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t843 = paddle._C_ops.transpose(t842, [0, 1, 3, 2, 4, 5])
        del t842

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t844 = paddle._C_ops.reshape(t843, t445)
        del t843

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t845 = paddle._C_ops.reshape(t844, t447)
        del t844

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t846 = paddle._C_ops.unsqueeze(t845, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t847 = paddle._C_ops.unsqueeze(t845, t354)
        del t845

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t848 = paddle._C_ops.subtract(t846, t847)
        del t846, t847

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t849 = paddle._C_ops.not_equal(t848, t452)

        # pd_op.full: (4x144x144xf32) <- ()
        t850 = paddle._C_ops.full(
            [4, 144, 144],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t851 = paddle._C_ops.where(t849, t850, t848)
        del t849, t848

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t852 = paddle._C_ops.equal(t851, t452)

        # pd_op.full: (4x144x144xf32) <- ()
        t853 = paddle._C_ops.full(
            [4, 144, 144],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t854 = paddle._C_ops.where(t852, t853, t851)
        del t852, t851

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t855 = paddle._C_ops.shape64(t830)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t856 = paddle._C_ops.slice(t855, [0], t305, t306, [1], [0])
        del t855

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t857 = paddle._C_ops.matmul(t830, t72, False, False)
        del t72, t830

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t858 = paddle._C_ops.add(t857, t73)
        del t857, t73

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t859 = [t856, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t860 = paddle._C_ops.stack(t859, 0)
        del t859

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t861 = paddle._C_ops.reshape(t858, t860)
        del t858, t860

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t862 = paddle._C_ops.transpose(t861, [2, 0, 3, 1, 4])
        del t861

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t863 = paddle._C_ops.slice(t862, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t864 = paddle._C_ops.slice(t862, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t865 = paddle._C_ops.slice(t862, [0], t354, t356, [1], [0])
        del t862

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t866 = paddle._C_ops.scale(t863, t358, float("0"), True)
        del t863

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t867 = paddle._C_ops.transpose(t864, [0, 1, 3, 2])
        del t864

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t868 = paddle._C_ops.matmul(t866, t867, False, False)
        del t866, t867

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t869 = paddle._C_ops.reshape(input11, t362)
        del input11

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t870 = paddle._C_ops.index_select(input12, t869, 0)
        del input12, t869

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t871 = paddle._C_ops.reshape(t870, t365)
        del t870

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t872 = paddle._C_ops.transpose(t871, [2, 0, 1])
        del t871

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t873 = paddle._C_ops.unsqueeze(t872, t305)
        del t872

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t874 = paddle._C_ops.add(t868, t873)
        del t868, t873

        # pd_op.full: (xi64) <- ()
        t875 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t876 = paddle._C_ops.floor_divide(t856, t875)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t877 = [t876, t347, t682, t345, t345]
        del t876

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t878 = paddle._C_ops.stack(t877, 0)
        del t877

        # pd_op.reshape: (-1x4x16x144x144xf32) <- (-1x16x144x144xf32, 5xi64)
        t879 = paddle._C_ops.reshape(t874, t878)
        del t874, t878

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t880 = paddle._C_ops.unsqueeze(t854, t306)
        del t854

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t881 = paddle._C_ops.unsqueeze(t880, t305)
        del t880

        # pd_op.add: (-1x4x16x144x144xf32) <- (-1x4x16x144x144xf32, 1x4x1x144x144xf32)
        t882 = paddle._C_ops.add(t879, t881)
        del t879, t881

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t883 = [t856, t682, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t884 = paddle._C_ops.stack(t883, 0)
        del t883

        # pd_op.reshape: (-1x16x144x144xf32) <- (-1x4x16x144x144xf32, 4xi64)
        t885 = paddle._C_ops.reshape(t882, t884)
        del t882, t884

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t886 = paddle._C_ops.softmax(t885, -1)
        del t885

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t887 = paddle._C_ops.matmul(t886, t865, False, False)
        del t865, t886

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t888 = paddle._C_ops.transpose(t887, [0, 2, 1, 3])
        del t887

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t889 = [t856, t345, t539]
        del t856

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t890 = paddle._C_ops.stack(t889, 0)
        del t889

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t891 = paddle._C_ops.reshape(t888, t890)
        del t890, t888

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t892 = paddle._C_ops.matmul(t891, t74, False, False)
        del t74, t891

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t893 = paddle._C_ops.add(t892, t75)
        del t892, t75

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t894 = paddle._C_ops.reshape(t893, t760)
        del t893

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t895 = paddle._C_ops.reshape(t894, t793)
        del t894

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t896 = paddle._C_ops.transpose(t895, [0, 1, 3, 2, 4, 5])
        del t895

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t897 = paddle._C_ops.reshape(t896, t796)
        del t896

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t898 = paddle._C_ops.roll(t897, t503, [1, 2])
        del t897

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t899 = [t813, t798, t539]
        del t813

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t900 = paddle._C_ops.stack(t899, 0)
        del t899

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t901 = paddle._C_ops.reshape(t898, t900)
        del t898, t900

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t902 = paddle._C_ops.add(t811, t901)
        del t811, t901

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t903, t904, t905 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t902, t76, t77, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t77, t76

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t906 = paddle._C_ops.matmul(t903, t78, False, False)
        del t903, t78

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t907 = paddle._C_ops.add(t906, t79)
        del t906, t79

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t908 = paddle._C_ops.gelu(t907, False)
        del t907

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t909 = paddle._C_ops.matmul(t908, t80, False, False)
        del t908, t80

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t910 = paddle._C_ops.add(t909, t81)
        del t909, t81

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t911 = paddle._C_ops.add(t902, t910)
        del t902, t910

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t912 = paddle._C_ops.shape64(t911)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t913 = paddle._C_ops.slice(t912, [0], t305, t306, [1], [0])
        del t912

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t914, t915, t916 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t911, t82, t83, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t83, t82

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t917 = [t913, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t918 = paddle._C_ops.stack(t917, 0)
        del t917

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t919 = paddle._C_ops.reshape(t914, t918)
        del t914, t918

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t920 = paddle._C_ops.shape64(t919)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t921 = paddle._C_ops.slice(t920, [0], t305, t306, [1], [0])
        del t920

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t922 = [t921, t755, t332, t755, t332, t539]
        del t921

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t923 = paddle._C_ops.stack(t922, 0)
        del t922

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t924 = paddle._C_ops.reshape(t919, t923)
        del t919, t923

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t925 = paddle._C_ops.transpose(t924, [0, 1, 3, 2, 4, 5])
        del t924

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t926 = paddle._C_ops.reshape(t925, t760)
        del t925

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t927 = paddle._C_ops.reshape(t926, t762)
        del t926

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t928 = paddle._C_ops.shape64(t927)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t929 = paddle._C_ops.slice(t928, [0], t305, t306, [1], [0])
        del t928

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t930 = paddle._C_ops.matmul(t927, t84, False, False)
        del t84, t927

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t931 = paddle._C_ops.add(t930, t85)
        del t930, t85

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t932 = [t929, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t933 = paddle._C_ops.stack(t932, 0)
        del t932

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t934 = paddle._C_ops.reshape(t931, t933)
        del t931, t933

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t935 = paddle._C_ops.transpose(t934, [2, 0, 3, 1, 4])
        del t934

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t936 = paddle._C_ops.slice(t935, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t937 = paddle._C_ops.slice(t935, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t938 = paddle._C_ops.slice(t935, [0], t354, t356, [1], [0])
        del t935

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t939 = paddle._C_ops.scale(t936, t358, float("0"), True)
        del t936

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t940 = paddle._C_ops.transpose(t937, [0, 1, 3, 2])
        del t937

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t941 = paddle._C_ops.matmul(t939, t940, False, False)
        del t939, t940

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t942 = paddle._C_ops.reshape(input13, t362)
        del input13

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t943 = paddle._C_ops.index_select(input14, t942, 0)
        del input14, t942

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t944 = paddle._C_ops.reshape(t943, t365)
        del t943

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t945 = paddle._C_ops.transpose(t944, [2, 0, 1])
        del t944

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t946 = paddle._C_ops.unsqueeze(t945, t305)
        del t945

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t947 = paddle._C_ops.add(t941, t946)
        del t941, t946

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t948 = paddle._C_ops.softmax(t947, -1)
        del t947

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t949 = paddle._C_ops.matmul(t948, t938, False, False)
        del t938, t948

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t950 = paddle._C_ops.transpose(t949, [0, 2, 1, 3])
        del t949

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t951 = [t929, t345, t539]
        del t929

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t952 = paddle._C_ops.stack(t951, 0)
        del t951

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t953 = paddle._C_ops.reshape(t950, t952)
        del t952, t950

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t954 = paddle._C_ops.matmul(t953, t86, False, False)
        del t86, t953

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t955 = paddle._C_ops.add(t954, t87)
        del t954, t87

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t956 = paddle._C_ops.reshape(t955, t760)
        del t955

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t957 = paddle._C_ops.reshape(t956, t793)
        del t956

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t958 = paddle._C_ops.transpose(t957, [0, 1, 3, 2, 4, 5])
        del t957

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t959 = paddle._C_ops.reshape(t958, t796)
        del t958

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t960 = [t913, t798, t539]
        del t913

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t961 = paddle._C_ops.stack(t960, 0)
        del t960

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t962 = paddle._C_ops.reshape(t959, t961)
        del t959, t961

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t963 = paddle._C_ops.add(t911, t962)
        del t911, t962

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t964, t965, t966 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t963, t88, t89, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t89, t88

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t967 = paddle._C_ops.matmul(t964, t90, False, False)
        del t964, t90

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t968 = paddle._C_ops.add(t967, t91)
        del t967, t91

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t969 = paddle._C_ops.gelu(t968, False)
        del t968

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t970 = paddle._C_ops.matmul(t969, t92, False, False)
        del t969, t92

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t971 = paddle._C_ops.add(t970, t93)
        del t970, t93

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t972 = paddle._C_ops.add(t963, t971)
        del t963, t971

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t973 = paddle._C_ops.shape64(t972)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t974 = paddle._C_ops.slice(t973, [0], t305, t306, [1], [0])
        del t973

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t975, t976, t977 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t972, t94, t95, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t95, t94

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t978 = [t974, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t979 = paddle._C_ops.stack(t978, 0)
        del t978

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t980 = paddle._C_ops.reshape(t975, t979)
        del t975, t979

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t981 = paddle._C_ops.shape64(t980)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t982 = paddle._C_ops.slice(t981, [0], t305, t306, [1], [0])
        del t981

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t983 = paddle._C_ops.roll(t980, t408, [1, 2])
        del t980

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t984 = paddle._C_ops.shape64(t983)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t985 = paddle._C_ops.slice(t984, [0], t305, t306, [1], [0])
        del t984

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t986 = [t985, t755, t332, t755, t332, t539]
        del t985

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t987 = paddle._C_ops.stack(t986, 0)
        del t986

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t988 = paddle._C_ops.reshape(t983, t987)
        del t983, t987

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t989 = paddle._C_ops.transpose(t988, [0, 1, 3, 2, 4, 5])
        del t988

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t990 = paddle._C_ops.reshape(t989, t760)
        del t989

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t991 = paddle._C_ops.reshape(t990, t762)
        del t990

        # pd_op.full: (1x24x24x1xf32) <- ()
        t992 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t993 = paddle._C_ops.set_value_(
            t992, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t992

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t994 = paddle._C_ops.set_value_(
            t993, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t993

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t995 = paddle._C_ops.set_value_(
            t994, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t994

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t996 = paddle._C_ops.set_value_(
            t995, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t995

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t997 = paddle._C_ops.set_value_(
            t996, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t996

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t998 = paddle._C_ops.set_value_(
            t997, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t997

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t999 = paddle._C_ops.set_value_(
            t998, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t998

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1000 = paddle._C_ops.set_value_(
            t999, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t999

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1001 = paddle._C_ops.set_value_(
            t1000, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1000

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1002 = paddle._C_ops.reshape(t1001, t841)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1003 = paddle._C_ops.transpose(t1002, [0, 1, 3, 2, 4, 5])
        del t1002

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1004 = paddle._C_ops.reshape(t1003, t445)
        del t1003

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1005 = paddle._C_ops.reshape(t1004, t447)
        del t1004

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1006 = paddle._C_ops.unsqueeze(t1005, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1007 = paddle._C_ops.unsqueeze(t1005, t354)
        del t1005

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1008 = paddle._C_ops.subtract(t1006, t1007)
        del t1006, t1007

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1009 = paddle._C_ops.not_equal(t1008, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1010 = paddle._C_ops.where(t1009, t850, t1008)
        del t1009, t1008

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1011 = paddle._C_ops.equal(t1010, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1012 = paddle._C_ops.where(t1011, t853, t1010)
        del t1011, t1010

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1013 = paddle._C_ops.shape64(t991)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1014 = paddle._C_ops.slice(t1013, [0], t305, t306, [1], [0])
        del t1013

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1015 = paddle._C_ops.matmul(t991, t96, False, False)
        del t96, t991

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1016 = paddle._C_ops.add(t1015, t97)
        del t1015, t97

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1017 = [t1014, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1018 = paddle._C_ops.stack(t1017, 0)
        del t1017

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1019 = paddle._C_ops.reshape(t1016, t1018)
        del t1016, t1018

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1020 = paddle._C_ops.transpose(t1019, [2, 0, 3, 1, 4])
        del t1019

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1021 = paddle._C_ops.slice(t1020, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1022 = paddle._C_ops.slice(t1020, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1023 = paddle._C_ops.slice(t1020, [0], t354, t356, [1], [0])
        del t1020

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1024 = paddle._C_ops.scale(t1021, t358, float("0"), True)
        del t1021

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1025 = paddle._C_ops.transpose(t1022, [0, 1, 3, 2])
        del t1022

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1026 = paddle._C_ops.matmul(t1024, t1025, False, False)
        del t1024, t1025

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1027 = paddle._C_ops.reshape(input15, t362)
        del input15

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1028 = paddle._C_ops.index_select(input16, t1027, 0)
        del input16, t1027

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1029 = paddle._C_ops.reshape(t1028, t365)
        del t1028

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1030 = paddle._C_ops.transpose(t1029, [2, 0, 1])
        del t1029

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1031 = paddle._C_ops.unsqueeze(t1030, t305)
        del t1030

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1032 = paddle._C_ops.add(t1026, t1031)
        del t1026, t1031

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1033 = paddle._C_ops.floor_divide(t1014, t875)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1034 = [t1033, t347, t682, t345, t345]
        del t1033

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1035 = paddle._C_ops.stack(t1034, 0)
        del t1034

        # pd_op.reshape: (-1x4x16x144x144xf32) <- (-1x16x144x144xf32, 5xi64)
        t1036 = paddle._C_ops.reshape(t1032, t1035)
        del t1032, t1035

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1037 = paddle._C_ops.unsqueeze(t1012, t306)
        del t1012

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1038 = paddle._C_ops.unsqueeze(t1037, t305)
        del t1037

        # pd_op.add: (-1x4x16x144x144xf32) <- (-1x4x16x144x144xf32, 1x4x1x144x144xf32)
        t1039 = paddle._C_ops.add(t1036, t1038)
        del t1036, t1038

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1040 = [t1014, t682, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1041 = paddle._C_ops.stack(t1040, 0)
        del t1040

        # pd_op.reshape: (-1x16x144x144xf32) <- (-1x4x16x144x144xf32, 4xi64)
        t1042 = paddle._C_ops.reshape(t1039, t1041)
        del t1039, t1041

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1043 = paddle._C_ops.softmax(t1042, -1)
        del t1042

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1044 = paddle._C_ops.matmul(t1043, t1023, False, False)
        del t1023, t1043

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1045 = paddle._C_ops.transpose(t1044, [0, 2, 1, 3])
        del t1044

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1046 = [t1014, t345, t539]
        del t1014

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1047 = paddle._C_ops.stack(t1046, 0)
        del t1046

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1048 = paddle._C_ops.reshape(t1045, t1047)
        del t1047, t1045

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1049 = paddle._C_ops.matmul(t1048, t98, False, False)
        del t98, t1048

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1050 = paddle._C_ops.add(t1049, t99)
        del t1049, t99

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1051 = paddle._C_ops.reshape(t1050, t760)
        del t1050

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1052 = paddle._C_ops.reshape(t1051, t793)
        del t1051

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1053 = paddle._C_ops.transpose(t1052, [0, 1, 3, 2, 4, 5])
        del t1052

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1054 = paddle._C_ops.reshape(t1053, t796)
        del t1053

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1055 = paddle._C_ops.roll(t1054, t503, [1, 2])
        del t1054

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1056 = [t974, t798, t539]
        del t974

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1057 = paddle._C_ops.stack(t1056, 0)
        del t1056

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1058 = paddle._C_ops.reshape(t1055, t1057)
        del t1055, t1057

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1059 = paddle._C_ops.add(t972, t1058)
        del t972, t1058

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1060, t1061, t1062 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1059, t100, t101, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t101, t100

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1063 = paddle._C_ops.matmul(t1060, t102, False, False)
        del t1060, t102

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1064 = paddle._C_ops.add(t1063, t103)
        del t1063, t103

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1065 = paddle._C_ops.gelu(t1064, False)
        del t1064

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1066 = paddle._C_ops.matmul(t1065, t104, False, False)
        del t1065, t104

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1067 = paddle._C_ops.add(t1066, t105)
        del t1066, t105

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1068 = paddle._C_ops.add(t1059, t1067)
        del t1059, t1067

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1069 = paddle._C_ops.shape64(t1068)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1070 = paddle._C_ops.slice(t1069, [0], t305, t306, [1], [0])
        del t1069

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1071, t1072, t1073 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1068, t106, t107, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t107, t106

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1074 = [t1070, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1075 = paddle._C_ops.stack(t1074, 0)
        del t1074

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1076 = paddle._C_ops.reshape(t1071, t1075)
        del t1071, t1075

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1077 = paddle._C_ops.shape64(t1076)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1078 = paddle._C_ops.slice(t1077, [0], t305, t306, [1], [0])
        del t1077

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1079 = [t1078, t755, t332, t755, t332, t539]
        del t1078

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1080 = paddle._C_ops.stack(t1079, 0)
        del t1079

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1081 = paddle._C_ops.reshape(t1076, t1080)
        del t1076, t1080

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1082 = paddle._C_ops.transpose(t1081, [0, 1, 3, 2, 4, 5])
        del t1081

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1083 = paddle._C_ops.reshape(t1082, t760)
        del t1082

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1084 = paddle._C_ops.reshape(t1083, t762)
        del t1083

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1085 = paddle._C_ops.shape64(t1084)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1086 = paddle._C_ops.slice(t1085, [0], t305, t306, [1], [0])
        del t1085

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1087 = paddle._C_ops.matmul(t1084, t108, False, False)
        del t108, t1084

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1088 = paddle._C_ops.add(t1087, t109)
        del t1087, t109

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1089 = [t1086, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1090 = paddle._C_ops.stack(t1089, 0)
        del t1089

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1091 = paddle._C_ops.reshape(t1088, t1090)
        del t1088, t1090

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1092 = paddle._C_ops.transpose(t1091, [2, 0, 3, 1, 4])
        del t1091

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1093 = paddle._C_ops.slice(t1092, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1094 = paddle._C_ops.slice(t1092, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1095 = paddle._C_ops.slice(t1092, [0], t354, t356, [1], [0])
        del t1092

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1096 = paddle._C_ops.scale(t1093, t358, float("0"), True)
        del t1093

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1097 = paddle._C_ops.transpose(t1094, [0, 1, 3, 2])
        del t1094

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1098 = paddle._C_ops.matmul(t1096, t1097, False, False)
        del t1096, t1097

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1099 = paddle._C_ops.reshape(input17, t362)
        del input17

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1100 = paddle._C_ops.index_select(input18, t1099, 0)
        del input18, t1099

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1101 = paddle._C_ops.reshape(t1100, t365)
        del t1100

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1102 = paddle._C_ops.transpose(t1101, [2, 0, 1])
        del t1101

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1103 = paddle._C_ops.unsqueeze(t1102, t305)
        del t1102

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1104 = paddle._C_ops.add(t1098, t1103)
        del t1098, t1103

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1105 = paddle._C_ops.softmax(t1104, -1)
        del t1104

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1106 = paddle._C_ops.matmul(t1105, t1095, False, False)
        del t1095, t1105

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1107 = paddle._C_ops.transpose(t1106, [0, 2, 1, 3])
        del t1106

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1108 = [t1086, t345, t539]
        del t1086

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1109 = paddle._C_ops.stack(t1108, 0)
        del t1108

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1110 = paddle._C_ops.reshape(t1107, t1109)
        del t1109, t1107

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1111 = paddle._C_ops.matmul(t1110, t110, False, False)
        del t110, t1110

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1112 = paddle._C_ops.add(t1111, t111)
        del t1111, t111

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1113 = paddle._C_ops.reshape(t1112, t760)
        del t1112

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1114 = paddle._C_ops.reshape(t1113, t793)
        del t1113

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1115 = paddle._C_ops.transpose(t1114, [0, 1, 3, 2, 4, 5])
        del t1114

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1116 = paddle._C_ops.reshape(t1115, t796)
        del t1115

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1117 = [t1070, t798, t539]
        del t1070

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1118 = paddle._C_ops.stack(t1117, 0)
        del t1117

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1119 = paddle._C_ops.reshape(t1116, t1118)
        del t1116, t1118

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1120 = paddle._C_ops.add(t1068, t1119)
        del t1068, t1119

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1121, t1122, t1123 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1120, t112, t113, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t113, t112

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1124 = paddle._C_ops.matmul(t1121, t114, False, False)
        del t1121, t114

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1125 = paddle._C_ops.add(t1124, t115)
        del t1124, t115

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1126 = paddle._C_ops.gelu(t1125, False)
        del t1125

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1127 = paddle._C_ops.matmul(t1126, t116, False, False)
        del t1126, t116

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1128 = paddle._C_ops.add(t1127, t117)
        del t1127, t117

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1129 = paddle._C_ops.add(t1120, t1128)
        del t1120, t1128

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1130 = paddle._C_ops.shape64(t1129)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1131 = paddle._C_ops.slice(t1130, [0], t305, t306, [1], [0])
        del t1130

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1132, t1133, t1134 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1129, t118, t119, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t119, t118

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1135 = [t1131, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1136 = paddle._C_ops.stack(t1135, 0)
        del t1135

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1137 = paddle._C_ops.reshape(t1132, t1136)
        del t1132, t1136

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1138 = paddle._C_ops.shape64(t1137)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1139 = paddle._C_ops.slice(t1138, [0], t305, t306, [1], [0])
        del t1138

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1140 = paddle._C_ops.roll(t1137, t408, [1, 2])
        del t1137

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1141 = paddle._C_ops.shape64(t1140)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1142 = paddle._C_ops.slice(t1141, [0], t305, t306, [1], [0])
        del t1141

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1143 = [t1142, t755, t332, t755, t332, t539]
        del t1142

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1144 = paddle._C_ops.stack(t1143, 0)
        del t1143

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1145 = paddle._C_ops.reshape(t1140, t1144)
        del t1140, t1144

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1146 = paddle._C_ops.transpose(t1145, [0, 1, 3, 2, 4, 5])
        del t1145

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1147 = paddle._C_ops.reshape(t1146, t760)
        del t1146

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1148 = paddle._C_ops.reshape(t1147, t762)
        del t1147

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1149 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1150 = paddle._C_ops.set_value_(
            t1149, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1149

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1151 = paddle._C_ops.set_value_(
            t1150, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1150

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1152 = paddle._C_ops.set_value_(
            t1151, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1151

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1153 = paddle._C_ops.set_value_(
            t1152, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1152

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1154 = paddle._C_ops.set_value_(
            t1153, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1153

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1155 = paddle._C_ops.set_value_(
            t1154, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1154

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1156 = paddle._C_ops.set_value_(
            t1155, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1155

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1157 = paddle._C_ops.set_value_(
            t1156, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1156

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1158 = paddle._C_ops.set_value_(
            t1157, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1157

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1159 = paddle._C_ops.reshape(t1158, t841)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1160 = paddle._C_ops.transpose(t1159, [0, 1, 3, 2, 4, 5])
        del t1159

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1161 = paddle._C_ops.reshape(t1160, t445)
        del t1160

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1162 = paddle._C_ops.reshape(t1161, t447)
        del t1161

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1163 = paddle._C_ops.unsqueeze(t1162, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1164 = paddle._C_ops.unsqueeze(t1162, t354)
        del t1162

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1165 = paddle._C_ops.subtract(t1163, t1164)
        del t1163, t1164

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1166 = paddle._C_ops.not_equal(t1165, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1167 = paddle._C_ops.where(t1166, t850, t1165)
        del t1166, t1165

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1168 = paddle._C_ops.equal(t1167, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1169 = paddle._C_ops.where(t1168, t853, t1167)
        del t1168, t1167

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1170 = paddle._C_ops.shape64(t1148)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1171 = paddle._C_ops.slice(t1170, [0], t305, t306, [1], [0])
        del t1170

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1172 = paddle._C_ops.matmul(t1148, t120, False, False)
        del t120, t1148

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1173 = paddle._C_ops.add(t1172, t121)
        del t1172, t121

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1174 = [t1171, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1175 = paddle._C_ops.stack(t1174, 0)
        del t1174

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1176 = paddle._C_ops.reshape(t1173, t1175)
        del t1173, t1175

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1177 = paddle._C_ops.transpose(t1176, [2, 0, 3, 1, 4])
        del t1176

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1178 = paddle._C_ops.slice(t1177, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1179 = paddle._C_ops.slice(t1177, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1180 = paddle._C_ops.slice(t1177, [0], t354, t356, [1], [0])
        del t1177

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1181 = paddle._C_ops.scale(t1178, t358, float("0"), True)
        del t1178

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1182 = paddle._C_ops.transpose(t1179, [0, 1, 3, 2])
        del t1179

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1183 = paddle._C_ops.matmul(t1181, t1182, False, False)
        del t1181, t1182

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1184 = paddle._C_ops.reshape(input19, t362)
        del input19

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1185 = paddle._C_ops.index_select(input20, t1184, 0)
        del input20, t1184

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1186 = paddle._C_ops.reshape(t1185, t365)
        del t1185

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1187 = paddle._C_ops.transpose(t1186, [2, 0, 1])
        del t1186

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1188 = paddle._C_ops.unsqueeze(t1187, t305)
        del t1187

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1189 = paddle._C_ops.add(t1183, t1188)
        del t1183, t1188

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1190 = paddle._C_ops.floor_divide(t1171, t875)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1191 = [t1190, t347, t682, t345, t345]
        del t1190

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1192 = paddle._C_ops.stack(t1191, 0)
        del t1191

        # pd_op.reshape: (-1x4x16x144x144xf32) <- (-1x16x144x144xf32, 5xi64)
        t1193 = paddle._C_ops.reshape(t1189, t1192)
        del t1189, t1192

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1194 = paddle._C_ops.unsqueeze(t1169, t306)
        del t1169

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1195 = paddle._C_ops.unsqueeze(t1194, t305)
        del t1194

        # pd_op.add: (-1x4x16x144x144xf32) <- (-1x4x16x144x144xf32, 1x4x1x144x144xf32)
        t1196 = paddle._C_ops.add(t1193, t1195)
        del t1193, t1195

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1197 = [t1171, t682, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1198 = paddle._C_ops.stack(t1197, 0)
        del t1197

        # pd_op.reshape: (-1x16x144x144xf32) <- (-1x4x16x144x144xf32, 4xi64)
        t1199 = paddle._C_ops.reshape(t1196, t1198)
        del t1196, t1198

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1200 = paddle._C_ops.softmax(t1199, -1)
        del t1199

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1201 = paddle._C_ops.matmul(t1200, t1180, False, False)
        del t1180, t1200

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1202 = paddle._C_ops.transpose(t1201, [0, 2, 1, 3])
        del t1201

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1203 = [t1171, t345, t539]
        del t1171

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1204 = paddle._C_ops.stack(t1203, 0)
        del t1203

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1205 = paddle._C_ops.reshape(t1202, t1204)
        del t1204, t1202

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1206 = paddle._C_ops.matmul(t1205, t122, False, False)
        del t122, t1205

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1207 = paddle._C_ops.add(t1206, t123)
        del t1206, t123

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1208 = paddle._C_ops.reshape(t1207, t760)
        del t1207

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1209 = paddle._C_ops.reshape(t1208, t793)
        del t1208

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1210 = paddle._C_ops.transpose(t1209, [0, 1, 3, 2, 4, 5])
        del t1209

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1211 = paddle._C_ops.reshape(t1210, t796)
        del t1210

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1212 = paddle._C_ops.roll(t1211, t503, [1, 2])
        del t1211

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1213 = [t1131, t798, t539]
        del t1131

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1214 = paddle._C_ops.stack(t1213, 0)
        del t1213

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1215 = paddle._C_ops.reshape(t1212, t1214)
        del t1212, t1214

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1216 = paddle._C_ops.add(t1129, t1215)
        del t1129, t1215

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1217, t1218, t1219 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1216, t124, t125, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t125, t124

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1220 = paddle._C_ops.matmul(t1217, t126, False, False)
        del t1217, t126

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1221 = paddle._C_ops.add(t1220, t127)
        del t1220, t127

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1222 = paddle._C_ops.gelu(t1221, False)
        del t1221

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1223 = paddle._C_ops.matmul(t1222, t128, False, False)
        del t1222, t128

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1224 = paddle._C_ops.add(t1223, t129)
        del t1223, t129

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1225 = paddle._C_ops.add(t1216, t1224)
        del t1216, t1224

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1226 = paddle._C_ops.shape64(t1225)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1227 = paddle._C_ops.slice(t1226, [0], t305, t306, [1], [0])
        del t1226

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1228, t1229, t1230 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1225, t130, t131, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t131, t130

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1231 = [t1227, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1232 = paddle._C_ops.stack(t1231, 0)
        del t1231

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1233 = paddle._C_ops.reshape(t1228, t1232)
        del t1228, t1232

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1234 = paddle._C_ops.shape64(t1233)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1235 = paddle._C_ops.slice(t1234, [0], t305, t306, [1], [0])
        del t1234

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1236 = [t1235, t755, t332, t755, t332, t539]
        del t1235

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1237 = paddle._C_ops.stack(t1236, 0)
        del t1236

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1238 = paddle._C_ops.reshape(t1233, t1237)
        del t1233, t1237

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1239 = paddle._C_ops.transpose(t1238, [0, 1, 3, 2, 4, 5])
        del t1238

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1240 = paddle._C_ops.reshape(t1239, t760)
        del t1239

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1241 = paddle._C_ops.reshape(t1240, t762)
        del t1240

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1242 = paddle._C_ops.shape64(t1241)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1243 = paddle._C_ops.slice(t1242, [0], t305, t306, [1], [0])
        del t1242

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1244 = paddle._C_ops.matmul(t1241, t132, False, False)
        del t132, t1241

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1245 = paddle._C_ops.add(t1244, t133)
        del t1244, t133

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1246 = [t1243, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1247 = paddle._C_ops.stack(t1246, 0)
        del t1246

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1248 = paddle._C_ops.reshape(t1245, t1247)
        del t1245, t1247

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1249 = paddle._C_ops.transpose(t1248, [2, 0, 3, 1, 4])
        del t1248

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1250 = paddle._C_ops.slice(t1249, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1251 = paddle._C_ops.slice(t1249, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1252 = paddle._C_ops.slice(t1249, [0], t354, t356, [1], [0])
        del t1249

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1253 = paddle._C_ops.scale(t1250, t358, float("0"), True)
        del t1250

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1254 = paddle._C_ops.transpose(t1251, [0, 1, 3, 2])
        del t1251

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1255 = paddle._C_ops.matmul(t1253, t1254, False, False)
        del t1253, t1254

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1256 = paddle._C_ops.reshape(input21, t362)
        del input21

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1257 = paddle._C_ops.index_select(input22, t1256, 0)
        del input22, t1256

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1258 = paddle._C_ops.reshape(t1257, t365)
        del t1257

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1259 = paddle._C_ops.transpose(t1258, [2, 0, 1])
        del t1258

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1260 = paddle._C_ops.unsqueeze(t1259, t305)
        del t1259

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1261 = paddle._C_ops.add(t1255, t1260)
        del t1255, t1260

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1262 = paddle._C_ops.softmax(t1261, -1)
        del t1261

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1263 = paddle._C_ops.matmul(t1262, t1252, False, False)
        del t1252, t1262

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1264 = paddle._C_ops.transpose(t1263, [0, 2, 1, 3])
        del t1263

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1265 = [t1243, t345, t539]
        del t1243

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1266 = paddle._C_ops.stack(t1265, 0)
        del t1265

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1267 = paddle._C_ops.reshape(t1264, t1266)
        del t1266, t1264

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1268 = paddle._C_ops.matmul(t1267, t134, False, False)
        del t134, t1267

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1269 = paddle._C_ops.add(t1268, t135)
        del t1268, t135

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1270 = paddle._C_ops.reshape(t1269, t760)
        del t1269

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1271 = paddle._C_ops.reshape(t1270, t793)
        del t1270

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1272 = paddle._C_ops.transpose(t1271, [0, 1, 3, 2, 4, 5])
        del t1271

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1273 = paddle._C_ops.reshape(t1272, t796)
        del t1272

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1274 = [t1227, t798, t539]
        del t1227

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1275 = paddle._C_ops.stack(t1274, 0)
        del t1274

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1276 = paddle._C_ops.reshape(t1273, t1275)
        del t1273, t1275

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1277 = paddle._C_ops.add(t1225, t1276)
        del t1225, t1276

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1278, t1279, t1280 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1277, t136, t137, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t137, t136

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1281 = paddle._C_ops.matmul(t1278, t138, False, False)
        del t1278, t138

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1282 = paddle._C_ops.add(t1281, t139)
        del t1281, t139

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1283 = paddle._C_ops.gelu(t1282, False)
        del t1282

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1284 = paddle._C_ops.matmul(t1283, t140, False, False)
        del t1283, t140

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1285 = paddle._C_ops.add(t1284, t141)
        del t1284, t141

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1286 = paddle._C_ops.add(t1277, t1285)
        del t1277, t1285

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1287 = paddle._C_ops.shape64(t1286)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1288 = paddle._C_ops.slice(t1287, [0], t305, t306, [1], [0])
        del t1287

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1289, t1290, t1291 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1286, t142, t143, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t143, t142

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1292 = [t1288, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1293 = paddle._C_ops.stack(t1292, 0)
        del t1292

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1294 = paddle._C_ops.reshape(t1289, t1293)
        del t1289, t1293

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1295 = paddle._C_ops.shape64(t1294)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1296 = paddle._C_ops.slice(t1295, [0], t305, t306, [1], [0])
        del t1295

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1297 = paddle._C_ops.roll(t1294, t408, [1, 2])
        del t1294

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1298 = paddle._C_ops.shape64(t1297)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1299 = paddle._C_ops.slice(t1298, [0], t305, t306, [1], [0])
        del t1298

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1300 = [t1299, t755, t332, t755, t332, t539]
        del t1299

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1301 = paddle._C_ops.stack(t1300, 0)
        del t1300

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1302 = paddle._C_ops.reshape(t1297, t1301)
        del t1297, t1301

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1303 = paddle._C_ops.transpose(t1302, [0, 1, 3, 2, 4, 5])
        del t1302

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1304 = paddle._C_ops.reshape(t1303, t760)
        del t1303

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1305 = paddle._C_ops.reshape(t1304, t762)
        del t1304

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1306 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1307 = paddle._C_ops.set_value_(
            t1306, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1306

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1308 = paddle._C_ops.set_value_(
            t1307, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1307

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1309 = paddle._C_ops.set_value_(
            t1308, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1308

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1310 = paddle._C_ops.set_value_(
            t1309, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1309

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1311 = paddle._C_ops.set_value_(
            t1310, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1310

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1312 = paddle._C_ops.set_value_(
            t1311, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1311

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1313 = paddle._C_ops.set_value_(
            t1312, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1312

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1314 = paddle._C_ops.set_value_(
            t1313, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1313

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1315 = paddle._C_ops.set_value_(
            t1314, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1314

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1316 = paddle._C_ops.reshape(t1315, t841)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1317 = paddle._C_ops.transpose(t1316, [0, 1, 3, 2, 4, 5])
        del t1316

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1318 = paddle._C_ops.reshape(t1317, t445)
        del t1317

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1319 = paddle._C_ops.reshape(t1318, t447)
        del t1318

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1320 = paddle._C_ops.unsqueeze(t1319, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1321 = paddle._C_ops.unsqueeze(t1319, t354)
        del t1319

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1322 = paddle._C_ops.subtract(t1320, t1321)
        del t1320, t1321

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1323 = paddle._C_ops.not_equal(t1322, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1324 = paddle._C_ops.where(t1323, t850, t1322)
        del t1323, t1322

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1325 = paddle._C_ops.equal(t1324, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1326 = paddle._C_ops.where(t1325, t853, t1324)
        del t1325, t1324

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1327 = paddle._C_ops.shape64(t1305)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1328 = paddle._C_ops.slice(t1327, [0], t305, t306, [1], [0])
        del t1327

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1329 = paddle._C_ops.matmul(t1305, t144, False, False)
        del t144, t1305

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1330 = paddle._C_ops.add(t1329, t145)
        del t1329, t145

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1331 = [t1328, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1332 = paddle._C_ops.stack(t1331, 0)
        del t1331

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1333 = paddle._C_ops.reshape(t1330, t1332)
        del t1330, t1332

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1334 = paddle._C_ops.transpose(t1333, [2, 0, 3, 1, 4])
        del t1333

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1335 = paddle._C_ops.slice(t1334, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1336 = paddle._C_ops.slice(t1334, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1337 = paddle._C_ops.slice(t1334, [0], t354, t356, [1], [0])
        del t1334

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1338 = paddle._C_ops.scale(t1335, t358, float("0"), True)
        del t1335

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1339 = paddle._C_ops.transpose(t1336, [0, 1, 3, 2])
        del t1336

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1340 = paddle._C_ops.matmul(t1338, t1339, False, False)
        del t1338, t1339

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1341 = paddle._C_ops.reshape(input23, t362)
        del input23

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1342 = paddle._C_ops.index_select(input24, t1341, 0)
        del input24, t1341

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1343 = paddle._C_ops.reshape(t1342, t365)
        del t1342

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1344 = paddle._C_ops.transpose(t1343, [2, 0, 1])
        del t1343

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1345 = paddle._C_ops.unsqueeze(t1344, t305)
        del t1344

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1346 = paddle._C_ops.add(t1340, t1345)
        del t1340, t1345

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1347 = paddle._C_ops.floor_divide(t1328, t875)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1348 = [t1347, t347, t682, t345, t345]
        del t1347

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1349 = paddle._C_ops.stack(t1348, 0)
        del t1348

        # pd_op.reshape: (-1x4x16x144x144xf32) <- (-1x16x144x144xf32, 5xi64)
        t1350 = paddle._C_ops.reshape(t1346, t1349)
        del t1346, t1349

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1351 = paddle._C_ops.unsqueeze(t1326, t306)
        del t1326

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1352 = paddle._C_ops.unsqueeze(t1351, t305)
        del t1351

        # pd_op.add: (-1x4x16x144x144xf32) <- (-1x4x16x144x144xf32, 1x4x1x144x144xf32)
        t1353 = paddle._C_ops.add(t1350, t1352)
        del t1350, t1352

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1354 = [t1328, t682, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1355 = paddle._C_ops.stack(t1354, 0)
        del t1354

        # pd_op.reshape: (-1x16x144x144xf32) <- (-1x4x16x144x144xf32, 4xi64)
        t1356 = paddle._C_ops.reshape(t1353, t1355)
        del t1353, t1355

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1357 = paddle._C_ops.softmax(t1356, -1)
        del t1356

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1358 = paddle._C_ops.matmul(t1357, t1337, False, False)
        del t1337, t1357

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1359 = paddle._C_ops.transpose(t1358, [0, 2, 1, 3])
        del t1358

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1360 = [t1328, t345, t539]
        del t1328

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1361 = paddle._C_ops.stack(t1360, 0)
        del t1360

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1362 = paddle._C_ops.reshape(t1359, t1361)
        del t1361, t1359

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1363 = paddle._C_ops.matmul(t1362, t146, False, False)
        del t146, t1362

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1364 = paddle._C_ops.add(t1363, t147)
        del t1363, t147

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1365 = paddle._C_ops.reshape(t1364, t760)
        del t1364

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1366 = paddle._C_ops.reshape(t1365, t793)
        del t1365

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1367 = paddle._C_ops.transpose(t1366, [0, 1, 3, 2, 4, 5])
        del t1366

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1368 = paddle._C_ops.reshape(t1367, t796)
        del t1367

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1369 = paddle._C_ops.roll(t1368, t503, [1, 2])
        del t1368

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1370 = [t1288, t798, t539]
        del t1288

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1371 = paddle._C_ops.stack(t1370, 0)
        del t1370

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1372 = paddle._C_ops.reshape(t1369, t1371)
        del t1369, t1371

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1373 = paddle._C_ops.add(t1286, t1372)
        del t1286, t1372

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1374, t1375, t1376 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1373, t148, t149, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t149, t148

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1377 = paddle._C_ops.matmul(t1374, t150, False, False)
        del t1374, t150

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1378 = paddle._C_ops.add(t1377, t151)
        del t1377, t151

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1379 = paddle._C_ops.gelu(t1378, False)
        del t1378

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1380 = paddle._C_ops.matmul(t1379, t152, False, False)
        del t1379, t152

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1381 = paddle._C_ops.add(t1380, t153)
        del t1380, t153

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1382 = paddle._C_ops.add(t1373, t1381)
        del t1373, t1381

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1383 = paddle._C_ops.shape64(t1382)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1384 = paddle._C_ops.slice(t1383, [0], t305, t306, [1], [0])
        del t1383

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1385, t1386, t1387 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1382, t154, t155, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t155, t154

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1388 = [t1384, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1389 = paddle._C_ops.stack(t1388, 0)
        del t1388

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1390 = paddle._C_ops.reshape(t1385, t1389)
        del t1385, t1389

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1391 = paddle._C_ops.shape64(t1390)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1392 = paddle._C_ops.slice(t1391, [0], t305, t306, [1], [0])
        del t1391

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1393 = [t1392, t755, t332, t755, t332, t539]
        del t1392

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1394 = paddle._C_ops.stack(t1393, 0)
        del t1393

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1395 = paddle._C_ops.reshape(t1390, t1394)
        del t1390, t1394

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1396 = paddle._C_ops.transpose(t1395, [0, 1, 3, 2, 4, 5])
        del t1395

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1397 = paddle._C_ops.reshape(t1396, t760)
        del t1396

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1398 = paddle._C_ops.reshape(t1397, t762)
        del t1397

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1399 = paddle._C_ops.shape64(t1398)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1400 = paddle._C_ops.slice(t1399, [0], t305, t306, [1], [0])
        del t1399

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1401 = paddle._C_ops.matmul(t1398, t156, False, False)
        del t156, t1398

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1402 = paddle._C_ops.add(t1401, t157)
        del t1401, t157

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1403 = [t1400, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1404 = paddle._C_ops.stack(t1403, 0)
        del t1403

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1405 = paddle._C_ops.reshape(t1402, t1404)
        del t1402, t1404

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1406 = paddle._C_ops.transpose(t1405, [2, 0, 3, 1, 4])
        del t1405

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1407 = paddle._C_ops.slice(t1406, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1408 = paddle._C_ops.slice(t1406, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1409 = paddle._C_ops.slice(t1406, [0], t354, t356, [1], [0])
        del t1406

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1410 = paddle._C_ops.scale(t1407, t358, float("0"), True)
        del t1407

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1411 = paddle._C_ops.transpose(t1408, [0, 1, 3, 2])
        del t1408

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1412 = paddle._C_ops.matmul(t1410, t1411, False, False)
        del t1410, t1411

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1413 = paddle._C_ops.reshape(input25, t362)
        del input25

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1414 = paddle._C_ops.index_select(input26, t1413, 0)
        del input26, t1413

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1415 = paddle._C_ops.reshape(t1414, t365)
        del t1414

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1416 = paddle._C_ops.transpose(t1415, [2, 0, 1])
        del t1415

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1417 = paddle._C_ops.unsqueeze(t1416, t305)
        del t1416

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1418 = paddle._C_ops.add(t1412, t1417)
        del t1412, t1417

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1419 = paddle._C_ops.softmax(t1418, -1)
        del t1418

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1420 = paddle._C_ops.matmul(t1419, t1409, False, False)
        del t1409, t1419

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1421 = paddle._C_ops.transpose(t1420, [0, 2, 1, 3])
        del t1420

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1422 = [t1400, t345, t539]
        del t1400

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1423 = paddle._C_ops.stack(t1422, 0)
        del t1422

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1424 = paddle._C_ops.reshape(t1421, t1423)
        del t1423, t1421

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1425 = paddle._C_ops.matmul(t1424, t158, False, False)
        del t158, t1424

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1426 = paddle._C_ops.add(t1425, t159)
        del t1425, t159

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1427 = paddle._C_ops.reshape(t1426, t760)
        del t1426

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1428 = paddle._C_ops.reshape(t1427, t793)
        del t1427

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1429 = paddle._C_ops.transpose(t1428, [0, 1, 3, 2, 4, 5])
        del t1428

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1430 = paddle._C_ops.reshape(t1429, t796)
        del t1429

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1431 = [t1384, t798, t539]
        del t1384

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1432 = paddle._C_ops.stack(t1431, 0)
        del t1431

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1433 = paddle._C_ops.reshape(t1430, t1432)
        del t1430, t1432

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1434 = paddle._C_ops.add(t1382, t1433)
        del t1382, t1433

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1435, t1436, t1437 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1434, t160, t161, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t161, t160

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1438 = paddle._C_ops.matmul(t1435, t162, False, False)
        del t1435, t162

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1439 = paddle._C_ops.add(t1438, t163)
        del t1438, t163

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1440 = paddle._C_ops.gelu(t1439, False)
        del t1439

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1441 = paddle._C_ops.matmul(t1440, t164, False, False)
        del t1440, t164

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1442 = paddle._C_ops.add(t1441, t165)
        del t1441, t165

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1443 = paddle._C_ops.add(t1434, t1442)
        del t1434, t1442

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1444 = paddle._C_ops.shape64(t1443)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1445 = paddle._C_ops.slice(t1444, [0], t305, t306, [1], [0])
        del t1444

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1446, t1447, t1448 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1443, t166, t167, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t167, t166

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1449 = [t1445, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1450 = paddle._C_ops.stack(t1449, 0)
        del t1449

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1451 = paddle._C_ops.reshape(t1446, t1450)
        del t1446, t1450

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1452 = paddle._C_ops.shape64(t1451)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1453 = paddle._C_ops.slice(t1452, [0], t305, t306, [1], [0])
        del t1452

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1454 = paddle._C_ops.roll(t1451, t408, [1, 2])
        del t1451

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1455 = paddle._C_ops.shape64(t1454)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1456 = paddle._C_ops.slice(t1455, [0], t305, t306, [1], [0])
        del t1455

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1457 = [t1456, t755, t332, t755, t332, t539]
        del t1456

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1458 = paddle._C_ops.stack(t1457, 0)
        del t1457

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1459 = paddle._C_ops.reshape(t1454, t1458)
        del t1454, t1458

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1460 = paddle._C_ops.transpose(t1459, [0, 1, 3, 2, 4, 5])
        del t1459

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1461 = paddle._C_ops.reshape(t1460, t760)
        del t1460

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1462 = paddle._C_ops.reshape(t1461, t762)
        del t1461

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1463 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1464 = paddle._C_ops.set_value_(
            t1463, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1463

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1465 = paddle._C_ops.set_value_(
            t1464, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1464

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1466 = paddle._C_ops.set_value_(
            t1465, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1465

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1467 = paddle._C_ops.set_value_(
            t1466, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1466

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1468 = paddle._C_ops.set_value_(
            t1467, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1467

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1469 = paddle._C_ops.set_value_(
            t1468, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1468

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1470 = paddle._C_ops.set_value_(
            t1469, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1469

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1471 = paddle._C_ops.set_value_(
            t1470, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1470

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1472 = paddle._C_ops.set_value_(
            t1471, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1471

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1473 = paddle._C_ops.reshape(t1472, t841)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1474 = paddle._C_ops.transpose(t1473, [0, 1, 3, 2, 4, 5])
        del t1473

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1475 = paddle._C_ops.reshape(t1474, t445)
        del t1474

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1476 = paddle._C_ops.reshape(t1475, t447)
        del t1475

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1477 = paddle._C_ops.unsqueeze(t1476, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1478 = paddle._C_ops.unsqueeze(t1476, t354)
        del t1476

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1479 = paddle._C_ops.subtract(t1477, t1478)
        del t1477, t1478

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1480 = paddle._C_ops.not_equal(t1479, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1481 = paddle._C_ops.where(t1480, t850, t1479)
        del t1480, t1479

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1482 = paddle._C_ops.equal(t1481, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1483 = paddle._C_ops.where(t1482, t853, t1481)
        del t1482, t1481

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1484 = paddle._C_ops.shape64(t1462)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1485 = paddle._C_ops.slice(t1484, [0], t305, t306, [1], [0])
        del t1484

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1486 = paddle._C_ops.matmul(t1462, t168, False, False)
        del t168, t1462

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1487 = paddle._C_ops.add(t1486, t169)
        del t1486, t169

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1488 = [t1485, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1489 = paddle._C_ops.stack(t1488, 0)
        del t1488

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1490 = paddle._C_ops.reshape(t1487, t1489)
        del t1487, t1489

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1491 = paddle._C_ops.transpose(t1490, [2, 0, 3, 1, 4])
        del t1490

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1492 = paddle._C_ops.slice(t1491, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1493 = paddle._C_ops.slice(t1491, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1494 = paddle._C_ops.slice(t1491, [0], t354, t356, [1], [0])
        del t1491

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1495 = paddle._C_ops.scale(t1492, t358, float("0"), True)
        del t1492

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1496 = paddle._C_ops.transpose(t1493, [0, 1, 3, 2])
        del t1493

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1497 = paddle._C_ops.matmul(t1495, t1496, False, False)
        del t1495, t1496

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1498 = paddle._C_ops.reshape(input27, t362)
        del input27

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1499 = paddle._C_ops.index_select(input28, t1498, 0)
        del input28, t1498

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1500 = paddle._C_ops.reshape(t1499, t365)
        del t1499

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1501 = paddle._C_ops.transpose(t1500, [2, 0, 1])
        del t1500

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1502 = paddle._C_ops.unsqueeze(t1501, t305)
        del t1501

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1503 = paddle._C_ops.add(t1497, t1502)
        del t1497, t1502

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1504 = paddle._C_ops.floor_divide(t1485, t875)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1505 = [t1504, t347, t682, t345, t345]
        del t1504

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1506 = paddle._C_ops.stack(t1505, 0)
        del t1505

        # pd_op.reshape: (-1x4x16x144x144xf32) <- (-1x16x144x144xf32, 5xi64)
        t1507 = paddle._C_ops.reshape(t1503, t1506)
        del t1503, t1506

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1508 = paddle._C_ops.unsqueeze(t1483, t306)
        del t1483

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1509 = paddle._C_ops.unsqueeze(t1508, t305)
        del t1508

        # pd_op.add: (-1x4x16x144x144xf32) <- (-1x4x16x144x144xf32, 1x4x1x144x144xf32)
        t1510 = paddle._C_ops.add(t1507, t1509)
        del t1507, t1509

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1511 = [t1485, t682, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1512 = paddle._C_ops.stack(t1511, 0)
        del t1511

        # pd_op.reshape: (-1x16x144x144xf32) <- (-1x4x16x144x144xf32, 4xi64)
        t1513 = paddle._C_ops.reshape(t1510, t1512)
        del t1510, t1512

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1514 = paddle._C_ops.softmax(t1513, -1)
        del t1513

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1515 = paddle._C_ops.matmul(t1514, t1494, False, False)
        del t1494, t1514

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1516 = paddle._C_ops.transpose(t1515, [0, 2, 1, 3])
        del t1515

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1517 = [t1485, t345, t539]
        del t1485

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1518 = paddle._C_ops.stack(t1517, 0)
        del t1517

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1519 = paddle._C_ops.reshape(t1516, t1518)
        del t1518, t1516

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1520 = paddle._C_ops.matmul(t1519, t170, False, False)
        del t170, t1519

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1521 = paddle._C_ops.add(t1520, t171)
        del t1520, t171

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1522 = paddle._C_ops.reshape(t1521, t760)
        del t1521

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1523 = paddle._C_ops.reshape(t1522, t793)
        del t1522

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1524 = paddle._C_ops.transpose(t1523, [0, 1, 3, 2, 4, 5])
        del t1523

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1525 = paddle._C_ops.reshape(t1524, t796)
        del t1524

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1526 = paddle._C_ops.roll(t1525, t503, [1, 2])
        del t1525

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1527 = [t1445, t798, t539]
        del t1445

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1528 = paddle._C_ops.stack(t1527, 0)
        del t1527

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1529 = paddle._C_ops.reshape(t1526, t1528)
        del t1526, t1528

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1530 = paddle._C_ops.add(t1443, t1529)
        del t1443, t1529

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1531, t1532, t1533 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1530, t172, t173, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t173, t172

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1534 = paddle._C_ops.matmul(t1531, t174, False, False)
        del t1531, t174

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1535 = paddle._C_ops.add(t1534, t175)
        del t1534, t175

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1536 = paddle._C_ops.gelu(t1535, False)
        del t1535

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1537 = paddle._C_ops.matmul(t1536, t176, False, False)
        del t1536, t176

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1538 = paddle._C_ops.add(t1537, t177)
        del t1537, t177

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1539 = paddle._C_ops.add(t1530, t1538)
        del t1530, t1538

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1540 = paddle._C_ops.shape64(t1539)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1541 = paddle._C_ops.slice(t1540, [0], t305, t306, [1], [0])
        del t1540

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1542, t1543, t1544 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1539, t178, t179, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t179, t178

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1545 = [t1541, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1546 = paddle._C_ops.stack(t1545, 0)
        del t1545

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1547 = paddle._C_ops.reshape(t1542, t1546)
        del t1542, t1546

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1548 = paddle._C_ops.shape64(t1547)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1549 = paddle._C_ops.slice(t1548, [0], t305, t306, [1], [0])
        del t1548

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1550 = [t1549, t755, t332, t755, t332, t539]
        del t1549

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1551 = paddle._C_ops.stack(t1550, 0)
        del t1550

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1552 = paddle._C_ops.reshape(t1547, t1551)
        del t1547, t1551

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1553 = paddle._C_ops.transpose(t1552, [0, 1, 3, 2, 4, 5])
        del t1552

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1554 = paddle._C_ops.reshape(t1553, t760)
        del t1553

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1555 = paddle._C_ops.reshape(t1554, t762)
        del t1554

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1556 = paddle._C_ops.shape64(t1555)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1557 = paddle._C_ops.slice(t1556, [0], t305, t306, [1], [0])
        del t1556

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1558 = paddle._C_ops.matmul(t1555, t180, False, False)
        del t180, t1555

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1559 = paddle._C_ops.add(t1558, t181)
        del t1558, t181

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1560 = [t1557, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1561 = paddle._C_ops.stack(t1560, 0)
        del t1560

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1562 = paddle._C_ops.reshape(t1559, t1561)
        del t1559, t1561

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1563 = paddle._C_ops.transpose(t1562, [2, 0, 3, 1, 4])
        del t1562

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1564 = paddle._C_ops.slice(t1563, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1565 = paddle._C_ops.slice(t1563, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1566 = paddle._C_ops.slice(t1563, [0], t354, t356, [1], [0])
        del t1563

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1567 = paddle._C_ops.scale(t1564, t358, float("0"), True)
        del t1564

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1568 = paddle._C_ops.transpose(t1565, [0, 1, 3, 2])
        del t1565

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1569 = paddle._C_ops.matmul(t1567, t1568, False, False)
        del t1567, t1568

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1570 = paddle._C_ops.reshape(input29, t362)
        del input29

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1571 = paddle._C_ops.index_select(input30, t1570, 0)
        del input30, t1570

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1572 = paddle._C_ops.reshape(t1571, t365)
        del t1571

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1573 = paddle._C_ops.transpose(t1572, [2, 0, 1])
        del t1572

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1574 = paddle._C_ops.unsqueeze(t1573, t305)
        del t1573

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1575 = paddle._C_ops.add(t1569, t1574)
        del t1569, t1574

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1576 = paddle._C_ops.softmax(t1575, -1)
        del t1575

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1577 = paddle._C_ops.matmul(t1576, t1566, False, False)
        del t1566, t1576

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1578 = paddle._C_ops.transpose(t1577, [0, 2, 1, 3])
        del t1577

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1579 = [t1557, t345, t539]
        del t1557

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1580 = paddle._C_ops.stack(t1579, 0)
        del t1579

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1581 = paddle._C_ops.reshape(t1578, t1580)
        del t1580, t1578

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1582 = paddle._C_ops.matmul(t1581, t182, False, False)
        del t182, t1581

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1583 = paddle._C_ops.add(t1582, t183)
        del t1582, t183

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1584 = paddle._C_ops.reshape(t1583, t760)
        del t1583

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1585 = paddle._C_ops.reshape(t1584, t793)
        del t1584

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1586 = paddle._C_ops.transpose(t1585, [0, 1, 3, 2, 4, 5])
        del t1585

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1587 = paddle._C_ops.reshape(t1586, t796)
        del t1586

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1588 = [t1541, t798, t539]
        del t1541

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1589 = paddle._C_ops.stack(t1588, 0)
        del t1588

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1590 = paddle._C_ops.reshape(t1587, t1589)
        del t1587, t1589

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1591 = paddle._C_ops.add(t1539, t1590)
        del t1539, t1590

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1592, t1593, t1594 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1591, t184, t185, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t185, t184

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1595 = paddle._C_ops.matmul(t1592, t186, False, False)
        del t1592, t186

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1596 = paddle._C_ops.add(t1595, t187)
        del t1595, t187

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1597 = paddle._C_ops.gelu(t1596, False)
        del t1596

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1598 = paddle._C_ops.matmul(t1597, t188, False, False)
        del t1597, t188

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1599 = paddle._C_ops.add(t1598, t189)
        del t1598, t189

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1600 = paddle._C_ops.add(t1591, t1599)
        del t1591, t1599

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1601 = paddle._C_ops.shape64(t1600)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1602 = paddle._C_ops.slice(t1601, [0], t305, t306, [1], [0])
        del t1601

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1603, t1604, t1605 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1600, t190, t191, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t191, t190

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1606 = [t1602, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1607 = paddle._C_ops.stack(t1606, 0)
        del t1606

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1608 = paddle._C_ops.reshape(t1603, t1607)
        del t1603, t1607

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1609 = paddle._C_ops.shape64(t1608)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1610 = paddle._C_ops.slice(t1609, [0], t305, t306, [1], [0])
        del t1609

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1611 = paddle._C_ops.roll(t1608, t408, [1, 2])
        del t1608

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1612 = paddle._C_ops.shape64(t1611)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1613 = paddle._C_ops.slice(t1612, [0], t305, t306, [1], [0])
        del t1612

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1614 = [t1613, t755, t332, t755, t332, t539]
        del t1613

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1615 = paddle._C_ops.stack(t1614, 0)
        del t1614

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1616 = paddle._C_ops.reshape(t1611, t1615)
        del t1611, t1615

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1617 = paddle._C_ops.transpose(t1616, [0, 1, 3, 2, 4, 5])
        del t1616

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1618 = paddle._C_ops.reshape(t1617, t760)
        del t1617

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1619 = paddle._C_ops.reshape(t1618, t762)
        del t1618

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1620 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1621 = paddle._C_ops.set_value_(
            t1620, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1620

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1622 = paddle._C_ops.set_value_(
            t1621, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1621

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1623 = paddle._C_ops.set_value_(
            t1622, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1622

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1624 = paddle._C_ops.set_value_(
            t1623, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1623

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1625 = paddle._C_ops.set_value_(
            t1624, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1624

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1626 = paddle._C_ops.set_value_(
            t1625, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1625

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1627 = paddle._C_ops.set_value_(
            t1626, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1626

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1628 = paddle._C_ops.set_value_(
            t1627, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1627

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1629 = paddle._C_ops.set_value_(
            t1628, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1628

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1630 = paddle._C_ops.reshape(t1629, t841)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1631 = paddle._C_ops.transpose(t1630, [0, 1, 3, 2, 4, 5])
        del t1630

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1632 = paddle._C_ops.reshape(t1631, t445)
        del t1631

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1633 = paddle._C_ops.reshape(t1632, t447)
        del t1632

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1634 = paddle._C_ops.unsqueeze(t1633, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1635 = paddle._C_ops.unsqueeze(t1633, t354)
        del t1633

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1636 = paddle._C_ops.subtract(t1634, t1635)
        del t1634, t1635

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1637 = paddle._C_ops.not_equal(t1636, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1638 = paddle._C_ops.where(t1637, t850, t1636)
        del t1637, t1636

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1639 = paddle._C_ops.equal(t1638, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1640 = paddle._C_ops.where(t1639, t853, t1638)
        del t1639, t1638

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1641 = paddle._C_ops.shape64(t1619)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1642 = paddle._C_ops.slice(t1641, [0], t305, t306, [1], [0])
        del t1641

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1643 = paddle._C_ops.matmul(t1619, t192, False, False)
        del t192, t1619

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1644 = paddle._C_ops.add(t1643, t193)
        del t1643, t193

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1645 = [t1642, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1646 = paddle._C_ops.stack(t1645, 0)
        del t1645

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1647 = paddle._C_ops.reshape(t1644, t1646)
        del t1644, t1646

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1648 = paddle._C_ops.transpose(t1647, [2, 0, 3, 1, 4])
        del t1647

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1649 = paddle._C_ops.slice(t1648, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1650 = paddle._C_ops.slice(t1648, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1651 = paddle._C_ops.slice(t1648, [0], t354, t356, [1], [0])
        del t1648

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1652 = paddle._C_ops.scale(t1649, t358, float("0"), True)
        del t1649

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1653 = paddle._C_ops.transpose(t1650, [0, 1, 3, 2])
        del t1650

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1654 = paddle._C_ops.matmul(t1652, t1653, False, False)
        del t1652, t1653

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1655 = paddle._C_ops.reshape(input31, t362)
        del input31

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1656 = paddle._C_ops.index_select(input32, t1655, 0)
        del input32, t1655

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1657 = paddle._C_ops.reshape(t1656, t365)
        del t1656

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1658 = paddle._C_ops.transpose(t1657, [2, 0, 1])
        del t1657

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1659 = paddle._C_ops.unsqueeze(t1658, t305)
        del t1658

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1660 = paddle._C_ops.add(t1654, t1659)
        del t1654, t1659

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1661 = paddle._C_ops.floor_divide(t1642, t875)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1662 = [t1661, t347, t682, t345, t345]
        del t1661

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1663 = paddle._C_ops.stack(t1662, 0)
        del t1662

        # pd_op.reshape: (-1x4x16x144x144xf32) <- (-1x16x144x144xf32, 5xi64)
        t1664 = paddle._C_ops.reshape(t1660, t1663)
        del t1660, t1663

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1665 = paddle._C_ops.unsqueeze(t1640, t306)
        del t1640

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1666 = paddle._C_ops.unsqueeze(t1665, t305)
        del t1665

        # pd_op.add: (-1x4x16x144x144xf32) <- (-1x4x16x144x144xf32, 1x4x1x144x144xf32)
        t1667 = paddle._C_ops.add(t1664, t1666)
        del t1664, t1666

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1668 = [t1642, t682, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1669 = paddle._C_ops.stack(t1668, 0)
        del t1668

        # pd_op.reshape: (-1x16x144x144xf32) <- (-1x4x16x144x144xf32, 4xi64)
        t1670 = paddle._C_ops.reshape(t1667, t1669)
        del t1667, t1669

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1671 = paddle._C_ops.softmax(t1670, -1)
        del t1670

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1672 = paddle._C_ops.matmul(t1671, t1651, False, False)
        del t1651, t1671

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1673 = paddle._C_ops.transpose(t1672, [0, 2, 1, 3])
        del t1672

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1674 = [t1642, t345, t539]
        del t1642

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1675 = paddle._C_ops.stack(t1674, 0)
        del t1674

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1676 = paddle._C_ops.reshape(t1673, t1675)
        del t1675, t1673

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1677 = paddle._C_ops.matmul(t1676, t194, False, False)
        del t194, t1676

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1678 = paddle._C_ops.add(t1677, t195)
        del t1677, t195

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1679 = paddle._C_ops.reshape(t1678, t760)
        del t1678

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1680 = paddle._C_ops.reshape(t1679, t793)
        del t1679

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1681 = paddle._C_ops.transpose(t1680, [0, 1, 3, 2, 4, 5])
        del t1680

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1682 = paddle._C_ops.reshape(t1681, t796)
        del t1681

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1683 = paddle._C_ops.roll(t1682, t503, [1, 2])
        del t1682

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1684 = [t1602, t798, t539]
        del t1602

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1685 = paddle._C_ops.stack(t1684, 0)
        del t1684

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1686 = paddle._C_ops.reshape(t1683, t1685)
        del t1683, t1685

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1687 = paddle._C_ops.add(t1600, t1686)
        del t1600, t1686

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1688, t1689, t1690 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1687, t196, t197, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t197, t196

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1691 = paddle._C_ops.matmul(t1688, t198, False, False)
        del t1688, t198

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1692 = paddle._C_ops.add(t1691, t199)
        del t1691, t199

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1693 = paddle._C_ops.gelu(t1692, False)
        del t1692

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1694 = paddle._C_ops.matmul(t1693, t200, False, False)
        del t1693, t200

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1695 = paddle._C_ops.add(t1694, t201)
        del t1694, t201

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1696 = paddle._C_ops.add(t1687, t1695)
        del t1687, t1695

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1697 = paddle._C_ops.shape64(t1696)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1698 = paddle._C_ops.slice(t1697, [0], t305, t306, [1], [0])
        del t1697

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1699, t1700, t1701 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1696, t202, t203, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t203, t202

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1702 = [t1698, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1703 = paddle._C_ops.stack(t1702, 0)
        del t1702

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1704 = paddle._C_ops.reshape(t1699, t1703)
        del t1699, t1703

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1705 = paddle._C_ops.shape64(t1704)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1706 = paddle._C_ops.slice(t1705, [0], t305, t306, [1], [0])
        del t1705

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1707 = [t1706, t755, t332, t755, t332, t539]
        del t1706

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1708 = paddle._C_ops.stack(t1707, 0)
        del t1707

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1709 = paddle._C_ops.reshape(t1704, t1708)
        del t1704, t1708

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1710 = paddle._C_ops.transpose(t1709, [0, 1, 3, 2, 4, 5])
        del t1709

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1711 = paddle._C_ops.reshape(t1710, t760)
        del t1710

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1712 = paddle._C_ops.reshape(t1711, t762)
        del t1711

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1713 = paddle._C_ops.shape64(t1712)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1714 = paddle._C_ops.slice(t1713, [0], t305, t306, [1], [0])
        del t1713

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1715 = paddle._C_ops.matmul(t1712, t204, False, False)
        del t204, t1712

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1716 = paddle._C_ops.add(t1715, t205)
        del t1715, t205

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1717 = [t1714, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1718 = paddle._C_ops.stack(t1717, 0)
        del t1717

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1719 = paddle._C_ops.reshape(t1716, t1718)
        del t1716, t1718

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1720 = paddle._C_ops.transpose(t1719, [2, 0, 3, 1, 4])
        del t1719

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1721 = paddle._C_ops.slice(t1720, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1722 = paddle._C_ops.slice(t1720, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1723 = paddle._C_ops.slice(t1720, [0], t354, t356, [1], [0])
        del t1720

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1724 = paddle._C_ops.scale(t1721, t358, float("0"), True)
        del t1721

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1725 = paddle._C_ops.transpose(t1722, [0, 1, 3, 2])
        del t1722

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1726 = paddle._C_ops.matmul(t1724, t1725, False, False)
        del t1724, t1725

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1727 = paddle._C_ops.reshape(input33, t362)
        del input33

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1728 = paddle._C_ops.index_select(input34, t1727, 0)
        del input34, t1727

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1729 = paddle._C_ops.reshape(t1728, t365)
        del t1728

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1730 = paddle._C_ops.transpose(t1729, [2, 0, 1])
        del t1729

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1731 = paddle._C_ops.unsqueeze(t1730, t305)
        del t1730

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1732 = paddle._C_ops.add(t1726, t1731)
        del t1726, t1731

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1733 = paddle._C_ops.softmax(t1732, -1)
        del t1732

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1734 = paddle._C_ops.matmul(t1733, t1723, False, False)
        del t1723, t1733

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1735 = paddle._C_ops.transpose(t1734, [0, 2, 1, 3])
        del t1734

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1736 = [t1714, t345, t539]
        del t1714

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1737 = paddle._C_ops.stack(t1736, 0)
        del t1736

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1738 = paddle._C_ops.reshape(t1735, t1737)
        del t1737, t1735

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1739 = paddle._C_ops.matmul(t1738, t206, False, False)
        del t206, t1738

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1740 = paddle._C_ops.add(t1739, t207)
        del t1739, t207

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1741 = paddle._C_ops.reshape(t1740, t760)
        del t1740

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1742 = paddle._C_ops.reshape(t1741, t793)
        del t1741

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1743 = paddle._C_ops.transpose(t1742, [0, 1, 3, 2, 4, 5])
        del t1742

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1744 = paddle._C_ops.reshape(t1743, t796)
        del t1743

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1745 = [t1698, t798, t539]
        del t1698

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1746 = paddle._C_ops.stack(t1745, 0)
        del t1745

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1747 = paddle._C_ops.reshape(t1744, t1746)
        del t1744, t1746

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1748 = paddle._C_ops.add(t1696, t1747)
        del t1696, t1747

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1749, t1750, t1751 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1748, t208, t209, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t209, t208

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1752 = paddle._C_ops.matmul(t1749, t210, False, False)
        del t1749, t210

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1753 = paddle._C_ops.add(t1752, t211)
        del t1752, t211

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1754 = paddle._C_ops.gelu(t1753, False)
        del t1753

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1755 = paddle._C_ops.matmul(t1754, t212, False, False)
        del t1754, t212

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1756 = paddle._C_ops.add(t1755, t213)
        del t1755, t213

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1757 = paddle._C_ops.add(t1748, t1756)
        del t1748, t1756

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1758 = paddle._C_ops.shape64(t1757)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1759 = paddle._C_ops.slice(t1758, [0], t305, t306, [1], [0])
        del t1758

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1760, t1761, t1762 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1757, t214, t215, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t215, t214

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1763 = [t1759, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1764 = paddle._C_ops.stack(t1763, 0)
        del t1763

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1765 = paddle._C_ops.reshape(t1760, t1764)
        del t1760, t1764

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1766 = paddle._C_ops.shape64(t1765)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1767 = paddle._C_ops.slice(t1766, [0], t305, t306, [1], [0])
        del t1766

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1768 = paddle._C_ops.roll(t1765, t408, [1, 2])
        del t1765

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1769 = paddle._C_ops.shape64(t1768)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1770 = paddle._C_ops.slice(t1769, [0], t305, t306, [1], [0])
        del t1769

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1771 = [t1770, t755, t332, t755, t332, t539]
        del t1770

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1772 = paddle._C_ops.stack(t1771, 0)
        del t1771

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1773 = paddle._C_ops.reshape(t1768, t1772)
        del t1768, t1772

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1774 = paddle._C_ops.transpose(t1773, [0, 1, 3, 2, 4, 5])
        del t1773

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1775 = paddle._C_ops.reshape(t1774, t760)
        del t1774

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1776 = paddle._C_ops.reshape(t1775, t762)
        del t1775

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1777 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1778 = paddle._C_ops.set_value_(
            t1777, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1777

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1779 = paddle._C_ops.set_value_(
            t1778, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1778

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1780 = paddle._C_ops.set_value_(
            t1779, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1779

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1781 = paddle._C_ops.set_value_(
            t1780, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1780

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1782 = paddle._C_ops.set_value_(
            t1781, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1781

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1783 = paddle._C_ops.set_value_(
            t1782, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1782

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1784 = paddle._C_ops.set_value_(
            t1783, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1783

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1785 = paddle._C_ops.set_value_(
            t1784, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1784

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1786 = paddle._C_ops.set_value_(
            t1785, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1785

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1787 = paddle._C_ops.reshape(t1786, t841)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1788 = paddle._C_ops.transpose(t1787, [0, 1, 3, 2, 4, 5])
        del t1787

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1789 = paddle._C_ops.reshape(t1788, t445)
        del t1788

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1790 = paddle._C_ops.reshape(t1789, t447)
        del t1789

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1791 = paddle._C_ops.unsqueeze(t1790, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1792 = paddle._C_ops.unsqueeze(t1790, t354)
        del t1790

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1793 = paddle._C_ops.subtract(t1791, t1792)
        del t1791, t1792

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1794 = paddle._C_ops.not_equal(t1793, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1795 = paddle._C_ops.where(t1794, t850, t1793)
        del t1794, t1793

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1796 = paddle._C_ops.equal(t1795, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1797 = paddle._C_ops.where(t1796, t853, t1795)
        del t1796, t1795

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1798 = paddle._C_ops.shape64(t1776)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1799 = paddle._C_ops.slice(t1798, [0], t305, t306, [1], [0])
        del t1798

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1800 = paddle._C_ops.matmul(t1776, t216, False, False)
        del t216, t1776

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1801 = paddle._C_ops.add(t1800, t217)
        del t1800, t217

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1802 = [t1799, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1803 = paddle._C_ops.stack(t1802, 0)
        del t1802

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1804 = paddle._C_ops.reshape(t1801, t1803)
        del t1801, t1803

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1805 = paddle._C_ops.transpose(t1804, [2, 0, 3, 1, 4])
        del t1804

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1806 = paddle._C_ops.slice(t1805, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1807 = paddle._C_ops.slice(t1805, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1808 = paddle._C_ops.slice(t1805, [0], t354, t356, [1], [0])
        del t1805

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1809 = paddle._C_ops.scale(t1806, t358, float("0"), True)
        del t1806

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1810 = paddle._C_ops.transpose(t1807, [0, 1, 3, 2])
        del t1807

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1811 = paddle._C_ops.matmul(t1809, t1810, False, False)
        del t1809, t1810

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1812 = paddle._C_ops.reshape(input35, t362)
        del input35

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1813 = paddle._C_ops.index_select(input36, t1812, 0)
        del input36, t1812

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1814 = paddle._C_ops.reshape(t1813, t365)
        del t1813

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1815 = paddle._C_ops.transpose(t1814, [2, 0, 1])
        del t1814

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1816 = paddle._C_ops.unsqueeze(t1815, t305)
        del t1815

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1817 = paddle._C_ops.add(t1811, t1816)
        del t1811, t1816

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1818 = paddle._C_ops.floor_divide(t1799, t875)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1819 = [t1818, t347, t682, t345, t345]
        del t1818

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1820 = paddle._C_ops.stack(t1819, 0)
        del t1819

        # pd_op.reshape: (-1x4x16x144x144xf32) <- (-1x16x144x144xf32, 5xi64)
        t1821 = paddle._C_ops.reshape(t1817, t1820)
        del t1817, t1820

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1822 = paddle._C_ops.unsqueeze(t1797, t306)
        del t1797

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1823 = paddle._C_ops.unsqueeze(t1822, t305)
        del t1822

        # pd_op.add: (-1x4x16x144x144xf32) <- (-1x4x16x144x144xf32, 1x4x1x144x144xf32)
        t1824 = paddle._C_ops.add(t1821, t1823)
        del t1821, t1823

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1825 = [t1799, t682, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1826 = paddle._C_ops.stack(t1825, 0)
        del t1825

        # pd_op.reshape: (-1x16x144x144xf32) <- (-1x4x16x144x144xf32, 4xi64)
        t1827 = paddle._C_ops.reshape(t1824, t1826)
        del t1824, t1826

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1828 = paddle._C_ops.softmax(t1827, -1)
        del t1827

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1829 = paddle._C_ops.matmul(t1828, t1808, False, False)
        del t1808, t1828

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1830 = paddle._C_ops.transpose(t1829, [0, 2, 1, 3])
        del t1829

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1831 = [t1799, t345, t539]
        del t1799

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1832 = paddle._C_ops.stack(t1831, 0)
        del t1831

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1833 = paddle._C_ops.reshape(t1830, t1832)
        del t1832, t1830

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1834 = paddle._C_ops.matmul(t1833, t218, False, False)
        del t218, t1833

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1835 = paddle._C_ops.add(t1834, t219)
        del t1834, t219

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1836 = paddle._C_ops.reshape(t1835, t760)
        del t1835

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1837 = paddle._C_ops.reshape(t1836, t793)
        del t1836

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1838 = paddle._C_ops.transpose(t1837, [0, 1, 3, 2, 4, 5])
        del t1837

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1839 = paddle._C_ops.reshape(t1838, t796)
        del t1838

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1840 = paddle._C_ops.roll(t1839, t503, [1, 2])
        del t1839

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1841 = [t1759, t798, t539]
        del t1759

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1842 = paddle._C_ops.stack(t1841, 0)
        del t1841

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1843 = paddle._C_ops.reshape(t1840, t1842)
        del t1840, t1842

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1844 = paddle._C_ops.add(t1757, t1843)
        del t1757, t1843

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1845, t1846, t1847 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1844, t220, t221, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t221, t220

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1848 = paddle._C_ops.matmul(t1845, t222, False, False)
        del t1845, t222

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1849 = paddle._C_ops.add(t1848, t223)
        del t1848, t223

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1850 = paddle._C_ops.gelu(t1849, False)
        del t1849

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1851 = paddle._C_ops.matmul(t1850, t224, False, False)
        del t1850, t224

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1852 = paddle._C_ops.add(t1851, t225)
        del t1851, t225

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1853 = paddle._C_ops.add(t1844, t1852)
        del t1844, t1852

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1854 = paddle._C_ops.shape64(t1853)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1855 = paddle._C_ops.slice(t1854, [0], t305, t306, [1], [0])
        del t1854

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1856, t1857, t1858 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1853, t226, t227, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t227, t226

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1859 = [t1855, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1860 = paddle._C_ops.stack(t1859, 0)
        del t1859

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1861 = paddle._C_ops.reshape(t1856, t1860)
        del t1856, t1860

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1862 = paddle._C_ops.shape64(t1861)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1863 = paddle._C_ops.slice(t1862, [0], t305, t306, [1], [0])
        del t1862

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1864 = [t1863, t755, t332, t755, t332, t539]
        del t1863

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1865 = paddle._C_ops.stack(t1864, 0)
        del t1864

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1866 = paddle._C_ops.reshape(t1861, t1865)
        del t1861, t1865

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1867 = paddle._C_ops.transpose(t1866, [0, 1, 3, 2, 4, 5])
        del t1866

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1868 = paddle._C_ops.reshape(t1867, t760)
        del t1867

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1869 = paddle._C_ops.reshape(t1868, t762)
        del t1868

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1870 = paddle._C_ops.shape64(t1869)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1871 = paddle._C_ops.slice(t1870, [0], t305, t306, [1], [0])
        del t1870

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1872 = paddle._C_ops.matmul(t1869, t228, False, False)
        del t228, t1869

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1873 = paddle._C_ops.add(t1872, t229)
        del t1872, t229

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1874 = [t1871, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1875 = paddle._C_ops.stack(t1874, 0)
        del t1874

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1876 = paddle._C_ops.reshape(t1873, t1875)
        del t1873, t1875

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1877 = paddle._C_ops.transpose(t1876, [2, 0, 3, 1, 4])
        del t1876

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1878 = paddle._C_ops.slice(t1877, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1879 = paddle._C_ops.slice(t1877, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1880 = paddle._C_ops.slice(t1877, [0], t354, t356, [1], [0])
        del t1877

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1881 = paddle._C_ops.scale(t1878, t358, float("0"), True)
        del t1878

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1882 = paddle._C_ops.transpose(t1879, [0, 1, 3, 2])
        del t1879

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1883 = paddle._C_ops.matmul(t1881, t1882, False, False)
        del t1881, t1882

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1884 = paddle._C_ops.reshape(input37, t362)
        del input37

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1885 = paddle._C_ops.index_select(input38, t1884, 0)
        del input38, t1884

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1886 = paddle._C_ops.reshape(t1885, t365)
        del t1885

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1887 = paddle._C_ops.transpose(t1886, [2, 0, 1])
        del t1886

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1888 = paddle._C_ops.unsqueeze(t1887, t305)
        del t1887

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1889 = paddle._C_ops.add(t1883, t1888)
        del t1883, t1888

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1890 = paddle._C_ops.softmax(t1889, -1)
        del t1889

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1891 = paddle._C_ops.matmul(t1890, t1880, False, False)
        del t1880, t1890

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1892 = paddle._C_ops.transpose(t1891, [0, 2, 1, 3])
        del t1891

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1893 = [t1871, t345, t539]
        del t1871

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1894 = paddle._C_ops.stack(t1893, 0)
        del t1893

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1895 = paddle._C_ops.reshape(t1892, t1894)
        del t1894, t1892

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1896 = paddle._C_ops.matmul(t1895, t230, False, False)
        del t230, t1895

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1897 = paddle._C_ops.add(t1896, t231)
        del t1896, t231

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1898 = paddle._C_ops.reshape(t1897, t760)
        del t1897

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1899 = paddle._C_ops.reshape(t1898, t793)
        del t1898

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1900 = paddle._C_ops.transpose(t1899, [0, 1, 3, 2, 4, 5])
        del t1899

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1901 = paddle._C_ops.reshape(t1900, t796)
        del t1900

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1902 = [t1855, t798, t539]
        del t1855

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1903 = paddle._C_ops.stack(t1902, 0)
        del t1902

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t1904 = paddle._C_ops.reshape(t1901, t1903)
        del t1901, t1903

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1905 = paddle._C_ops.add(t1853, t1904)
        del t1853, t1904

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1906, t1907, t1908 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1905, t232, t233, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t233, t232

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t1909 = paddle._C_ops.matmul(t1906, t234, False, False)
        del t1906, t234

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t1910 = paddle._C_ops.add(t1909, t235)
        del t1909, t235

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t1911 = paddle._C_ops.gelu(t1910, False)
        del t1910

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t1912 = paddle._C_ops.matmul(t1911, t236, False, False)
        del t1911, t236

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t1913 = paddle._C_ops.add(t1912, t237)
        del t1912, t237

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t1914 = paddle._C_ops.add(t1905, t1913)
        del t1905, t1913

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t1915 = paddle._C_ops.shape64(t1914)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1916 = paddle._C_ops.slice(t1915, [0], t305, t306, [1], [0])
        del t1915

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t1917, t1918, t1919 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1914, t238, t239, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t239, t238

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1920 = [t1916, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1921 = paddle._C_ops.stack(t1920, 0)
        del t1920

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t1922 = paddle._C_ops.reshape(t1917, t1921)
        del t1917, t1921

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1923 = paddle._C_ops.shape64(t1922)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1924 = paddle._C_ops.slice(t1923, [0], t305, t306, [1], [0])
        del t1923

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1925 = paddle._C_ops.roll(t1922, t408, [1, 2])
        del t1922

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t1926 = paddle._C_ops.shape64(t1925)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1927 = paddle._C_ops.slice(t1926, [0], t305, t306, [1], [0])
        del t1926

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1928 = [t1927, t755, t332, t755, t332, t539]
        del t1927

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1929 = paddle._C_ops.stack(t1928, 0)
        del t1928

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t1930 = paddle._C_ops.reshape(t1925, t1929)
        del t1925, t1929

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t1931 = paddle._C_ops.transpose(t1930, [0, 1, 3, 2, 4, 5])
        del t1930

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t1932 = paddle._C_ops.reshape(t1931, t760)
        del t1931

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t1933 = paddle._C_ops.reshape(t1932, t762)
        del t1932

        # pd_op.full: (1x24x24x1xf32) <- ()
        t1934 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1935 = paddle._C_ops.set_value_(
            t1934, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t1934

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1936 = paddle._C_ops.set_value_(
            t1935, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t1935

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1937 = paddle._C_ops.set_value_(
            t1936, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t1936

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1938 = paddle._C_ops.set_value_(
            t1937, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t1937

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1939 = paddle._C_ops.set_value_(
            t1938, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t1938

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1940 = paddle._C_ops.set_value_(
            t1939, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t1939

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1941 = paddle._C_ops.set_value_(
            t1940, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t1940

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1942 = paddle._C_ops.set_value_(
            t1941, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t1941

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t1943 = paddle._C_ops.set_value_(
            t1942, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t1942

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t1944 = paddle._C_ops.reshape(t1943, t841)

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t1945 = paddle._C_ops.transpose(t1944, [0, 1, 3, 2, 4, 5])
        del t1944

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t1946 = paddle._C_ops.reshape(t1945, t445)
        del t1945

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t1947 = paddle._C_ops.reshape(t1946, t447)
        del t1946

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t1948 = paddle._C_ops.unsqueeze(t1947, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t1949 = paddle._C_ops.unsqueeze(t1947, t354)
        del t1947

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t1950 = paddle._C_ops.subtract(t1948, t1949)
        del t1948, t1949

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1951 = paddle._C_ops.not_equal(t1950, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1952 = paddle._C_ops.where(t1951, t850, t1950)
        del t1951, t1950

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t1953 = paddle._C_ops.equal(t1952, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t1954 = paddle._C_ops.where(t1953, t853, t1952)
        del t1953, t1952

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t1955 = paddle._C_ops.shape64(t1933)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1956 = paddle._C_ops.slice(t1955, [0], t305, t306, [1], [0])
        del t1955

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t1957 = paddle._C_ops.matmul(t1933, t240, False, False)
        del t240, t1933

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t1958 = paddle._C_ops.add(t1957, t241)
        del t1957, t241

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1959 = [t1956, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1960 = paddle._C_ops.stack(t1959, 0)
        del t1959

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t1961 = paddle._C_ops.reshape(t1958, t1960)
        del t1958, t1960

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t1962 = paddle._C_ops.transpose(t1961, [2, 0, 3, 1, 4])
        del t1961

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1963 = paddle._C_ops.slice(t1962, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1964 = paddle._C_ops.slice(t1962, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t1965 = paddle._C_ops.slice(t1962, [0], t354, t356, [1], [0])
        del t1962

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t1966 = paddle._C_ops.scale(t1963, t358, float("0"), True)
        del t1963

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t1967 = paddle._C_ops.transpose(t1964, [0, 1, 3, 2])
        del t1964

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t1968 = paddle._C_ops.matmul(t1966, t1967, False, False)
        del t1966, t1967

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t1969 = paddle._C_ops.reshape(input39, t362)
        del input39

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t1970 = paddle._C_ops.index_select(input40, t1969, 0)
        del input40, t1969

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t1971 = paddle._C_ops.reshape(t1970, t365)
        del t1970

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t1972 = paddle._C_ops.transpose(t1971, [2, 0, 1])
        del t1971

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t1973 = paddle._C_ops.unsqueeze(t1972, t305)
        del t1972

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t1974 = paddle._C_ops.add(t1968, t1973)
        del t1968, t1973

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1975 = paddle._C_ops.floor_divide(t1956, t875)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1976 = [t1975, t347, t682, t345, t345]
        del t1975

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1977 = paddle._C_ops.stack(t1976, 0)
        del t1976

        # pd_op.reshape: (-1x4x16x144x144xf32) <- (-1x16x144x144xf32, 5xi64)
        t1978 = paddle._C_ops.reshape(t1974, t1977)
        del t1974, t1977

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t1979 = paddle._C_ops.unsqueeze(t1954, t306)
        del t1954

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t1980 = paddle._C_ops.unsqueeze(t1979, t305)
        del t1979

        # pd_op.add: (-1x4x16x144x144xf32) <- (-1x4x16x144x144xf32, 1x4x1x144x144xf32)
        t1981 = paddle._C_ops.add(t1978, t1980)
        del t1978, t1980

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1982 = [t1956, t682, t345, t345]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1983 = paddle._C_ops.stack(t1982, 0)
        del t1982

        # pd_op.reshape: (-1x16x144x144xf32) <- (-1x4x16x144x144xf32, 4xi64)
        t1984 = paddle._C_ops.reshape(t1981, t1983)
        del t1981, t1983

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t1985 = paddle._C_ops.softmax(t1984, -1)
        del t1984

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t1986 = paddle._C_ops.matmul(t1985, t1965, False, False)
        del t1965, t1985

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t1987 = paddle._C_ops.transpose(t1986, [0, 2, 1, 3])
        del t1986

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1988 = [t1956, t345, t539]
        del t1956

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1989 = paddle._C_ops.stack(t1988, 0)
        del t1988

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t1990 = paddle._C_ops.reshape(t1987, t1989)
        del t1989, t1987

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t1991 = paddle._C_ops.matmul(t1990, t242, False, False)
        del t242, t1990

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t1992 = paddle._C_ops.add(t1991, t243)
        del t1991, t243

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t1993 = paddle._C_ops.reshape(t1992, t760)
        del t1992

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t1994 = paddle._C_ops.reshape(t1993, t793)
        del t1993

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t1995 = paddle._C_ops.transpose(t1994, [0, 1, 3, 2, 4, 5])
        del t1994

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t1996 = paddle._C_ops.reshape(t1995, t796)
        del t1995

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t1997 = paddle._C_ops.roll(t1996, t503, [1, 2])
        del t1996

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1998 = [t1916, t798, t539]
        del t1916

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1999 = paddle._C_ops.stack(t1998, 0)
        del t1998

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t2000 = paddle._C_ops.reshape(t1997, t1999)
        del t1997, t1999

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t2001 = paddle._C_ops.add(t1914, t2000)
        del t1914, t2000

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t2002, t2003, t2004 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2001, t244, t245, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t245, t244

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t2005 = paddle._C_ops.matmul(t2002, t246, False, False)
        del t2002, t246

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t2006 = paddle._C_ops.add(t2005, t247)
        del t2005, t247

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t2007 = paddle._C_ops.gelu(t2006, False)
        del t2006

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t2008 = paddle._C_ops.matmul(t2007, t248, False, False)
        del t2007, t248

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t2009 = paddle._C_ops.add(t2008, t249)
        del t2008, t249

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t2010 = paddle._C_ops.add(t2001, t2009)
        del t2001, t2009

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t2011 = paddle._C_ops.shape64(t2010)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2012 = paddle._C_ops.slice(t2011, [0], t305, t306, [1], [0])
        del t2011

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t2013, t2014, t2015 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2010, t250, t251, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t251, t250

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2016 = [t2012, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2017 = paddle._C_ops.stack(t2016, 0)
        del t2016

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t2018 = paddle._C_ops.reshape(t2013, t2017)
        del t2013, t2017

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t2019 = paddle._C_ops.shape64(t2018)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2020 = paddle._C_ops.slice(t2019, [0], t305, t306, [1], [0])
        del t2019

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2021 = [t2020, t755, t332, t755, t332, t539]
        del t2020

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2022 = paddle._C_ops.stack(t2021, 0)
        del t2021

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t2023 = paddle._C_ops.reshape(t2018, t2022)
        del t2018, t2022

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t2024 = paddle._C_ops.transpose(t2023, [0, 1, 3, 2, 4, 5])
        del t2023

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t2025 = paddle._C_ops.reshape(t2024, t760)
        del t2024

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t2026 = paddle._C_ops.reshape(t2025, t762)
        del t2025

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t2027 = paddle._C_ops.shape64(t2026)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2028 = paddle._C_ops.slice(t2027, [0], t305, t306, [1], [0])
        del t2027

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t2029 = paddle._C_ops.matmul(t2026, t252, False, False)
        del t252, t2026

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t2030 = paddle._C_ops.add(t2029, t253)
        del t2029, t253

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2031 = [t2028, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2032 = paddle._C_ops.stack(t2031, 0)
        del t2031

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t2033 = paddle._C_ops.reshape(t2030, t2032)
        del t2030, t2032

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t2034 = paddle._C_ops.transpose(t2033, [2, 0, 3, 1, 4])
        del t2033

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t2035 = paddle._C_ops.slice(t2034, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t2036 = paddle._C_ops.slice(t2034, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t2037 = paddle._C_ops.slice(t2034, [0], t354, t356, [1], [0])
        del t2034

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t2038 = paddle._C_ops.scale(t2035, t358, float("0"), True)
        del t2035

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t2039 = paddle._C_ops.transpose(t2036, [0, 1, 3, 2])
        del t2036

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t2040 = paddle._C_ops.matmul(t2038, t2039, False, False)
        del t2038, t2039

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t2041 = paddle._C_ops.reshape(input41, t362)
        del input41

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t2042 = paddle._C_ops.index_select(input42, t2041, 0)
        del input42, t2041

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t2043 = paddle._C_ops.reshape(t2042, t365)
        del t2042

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t2044 = paddle._C_ops.transpose(t2043, [2, 0, 1])
        del t2043

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t2045 = paddle._C_ops.unsqueeze(t2044, t305)
        del t2044

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t2046 = paddle._C_ops.add(t2040, t2045)
        del t2040, t2045

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t2047 = paddle._C_ops.softmax(t2046, -1)
        del t2046

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t2048 = paddle._C_ops.matmul(t2047, t2037, False, False)
        del t2037, t2047

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t2049 = paddle._C_ops.transpose(t2048, [0, 2, 1, 3])
        del t2048

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2050 = [t2028, t345, t539]
        del t2028

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2051 = paddle._C_ops.stack(t2050, 0)
        del t2050

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t2052 = paddle._C_ops.reshape(t2049, t2051)
        del t2051, t2049

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t2053 = paddle._C_ops.matmul(t2052, t254, False, False)
        del t254, t2052

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t2054 = paddle._C_ops.add(t2053, t255)
        del t2053, t255

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t2055 = paddle._C_ops.reshape(t2054, t760)
        del t2054

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t2056 = paddle._C_ops.reshape(t2055, t793)
        del t2055

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t2057 = paddle._C_ops.transpose(t2056, [0, 1, 3, 2, 4, 5])
        del t2056

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t2058 = paddle._C_ops.reshape(t2057, t796)
        del t2057

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2059 = [t2012, t798, t539]
        del t2012

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2060 = paddle._C_ops.stack(t2059, 0)
        del t2059

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t2061 = paddle._C_ops.reshape(t2058, t2060)
        del t2058, t2060

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t2062 = paddle._C_ops.add(t2010, t2061)
        del t2010, t2061

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t2063, t2064, t2065 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2062, t256, t257, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t257, t256

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t2066 = paddle._C_ops.matmul(t2063, t258, False, False)
        del t2063, t258

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t2067 = paddle._C_ops.add(t2066, t259)
        del t2066, t259

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t2068 = paddle._C_ops.gelu(t2067, False)
        del t2067

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t2069 = paddle._C_ops.matmul(t2068, t260, False, False)
        del t2068, t260

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t2070 = paddle._C_ops.add(t2069, t261)
        del t2069, t261

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t2071 = paddle._C_ops.add(t2062, t2070)
        del t2062, t2070

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t2072 = paddle._C_ops.shape64(t2071)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2073 = paddle._C_ops.slice(t2072, [0], t305, t306, [1], [0])
        del t2072

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t2074, t2075, t2076 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2071, t262, t263, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t263, t262

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2077 = [t2073, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2078 = paddle._C_ops.stack(t2077, 0)
        del t2077

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t2079 = paddle._C_ops.reshape(t2074, t2078)
        del t2074, t2078

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t2080 = paddle._C_ops.shape64(t2079)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2081 = paddle._C_ops.slice(t2080, [0], t305, t306, [1], [0])
        del t2080

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t2082 = paddle._C_ops.roll(t2079, t408, [1, 2])
        del t2079

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t2083 = paddle._C_ops.shape64(t2082)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2084 = paddle._C_ops.slice(t2083, [0], t305, t306, [1], [0])
        del t2083

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2085 = [t2084, t755, t332, t755, t332, t539]
        del t755, t2084

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2086 = paddle._C_ops.stack(t2085, 0)
        del t2085

        # pd_op.reshape: (-1x2x12x2x12x512xf32) <- (-1x24x24x512xf32, 6xi64)
        t2087 = paddle._C_ops.reshape(t2082, t2086)
        del t2082, t2086

        # pd_op.transpose: (-1x2x2x12x12x512xf32) <- (-1x2x12x2x12x512xf32)
        t2088 = paddle._C_ops.transpose(t2087, [0, 1, 3, 2, 4, 5])
        del t2087

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x2x2x12x12x512xf32, 4xi64)
        t2089 = paddle._C_ops.reshape(t2088, t760)
        del t2088

        # pd_op.reshape: (-1x144x512xf32) <- (-1x12x12x512xf32, 3xi64)
        t2090 = paddle._C_ops.reshape(t2089, t762)
        del t762, t2089

        # pd_op.full: (1x24x24x1xf32) <- ()
        t2091 = paddle._C_ops.full(
            [1, 24, 24, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2092 = paddle._C_ops.set_value_(
            t2091, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t2091

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2093 = paddle._C_ops.set_value_(
            t2092, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t2092

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2094 = paddle._C_ops.set_value_(
            t2093, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t2093

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2095 = paddle._C_ops.set_value_(
            t2094, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t2094

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2096 = paddle._C_ops.set_value_(
            t2095, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t2095

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2097 = paddle._C_ops.set_value_(
            t2096, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t2096

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2098 = paddle._C_ops.set_value_(
            t2097, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t2097

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2099 = paddle._C_ops.set_value_(
            t2098, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t2098

        # pd_op.set_value_: (1x24x24x1xf32) <- (1x24x24x1xf32, 2xi64, 2xi64, 2xi64)
        t2100 = paddle._C_ops.set_value_(
            t2099, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t2099

        # pd_op.reshape: (1x2x12x2x12x1xf32) <- (1x24x24x1xf32, 6xi64)
        t2101 = paddle._C_ops.reshape(t2100, t841)
        del t841

        # pd_op.transpose: (1x2x2x12x12x1xf32) <- (1x2x12x2x12x1xf32)
        t2102 = paddle._C_ops.transpose(t2101, [0, 1, 3, 2, 4, 5])
        del t2101

        # pd_op.reshape: (4x12x12x1xf32) <- (1x2x2x12x12x1xf32, 4xi64)
        t2103 = paddle._C_ops.reshape(t2102, t445)
        del t2102

        # pd_op.reshape: (4x144xf32) <- (4x12x12x1xf32, 2xi64)
        t2104 = paddle._C_ops.reshape(t2103, t447)
        del t2103

        # pd_op.unsqueeze: (4x1x144xf32) <- (4x144xf32, 1xi64)
        t2105 = paddle._C_ops.unsqueeze(t2104, t306)

        # pd_op.unsqueeze: (4x144x1xf32) <- (4x144xf32, 1xi64)
        t2106 = paddle._C_ops.unsqueeze(t2104, t354)
        del t2104

        # pd_op.subtract: (4x144x144xf32) <- (4x1x144xf32, 4x144x1xf32)
        t2107 = paddle._C_ops.subtract(t2105, t2106)
        del t2105, t2106

        # pd_op.not_equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t2108 = paddle._C_ops.not_equal(t2107, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t2109 = paddle._C_ops.where(t2108, t850, t2107)
        del t850, t2108, t2107

        # pd_op.equal: (4x144x144xb) <- (4x144x144xf32, xf32)
        t2110 = paddle._C_ops.equal(t2109, t452)

        # pd_op.where: (4x144x144xf32) <- (4x144x144xb, 4x144x144xf32, 4x144x144xf32)
        t2111 = paddle._C_ops.where(t2110, t853, t2109)
        del t2110, t853, t2109

        # pd_op.shape64: (3xi64) <- (-1x144x512xf32)
        t2112 = paddle._C_ops.shape64(t2090)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2113 = paddle._C_ops.slice(t2112, [0], t305, t306, [1], [0])
        del t2112

        # pd_op.matmul: (-1x144x1536xf32) <- (-1x144x512xf32, 512x1536xf32)
        t2114 = paddle._C_ops.matmul(t2090, t264, False, False)
        del t264, t2090

        # pd_op.add: (-1x144x1536xf32) <- (-1x144x1536xf32, 1536xf32)
        t2115 = paddle._C_ops.add(t2114, t265)
        del t2114, t265

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2116 = [t2113, t345, t346, t682, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2117 = paddle._C_ops.stack(t2116, 0)
        del t2116

        # pd_op.reshape: (-1x144x3x16x32xf32) <- (-1x144x1536xf32, 5xi64)
        t2118 = paddle._C_ops.reshape(t2115, t2117)
        del t2115, t2117

        # pd_op.transpose: (3x-1x16x144x32xf32) <- (-1x144x3x16x32xf32)
        t2119 = paddle._C_ops.transpose(t2118, [2, 0, 3, 1, 4])
        del t2118

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t2120 = paddle._C_ops.slice(t2119, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t2121 = paddle._C_ops.slice(t2119, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x16x144x32xf32) <- (3x-1x16x144x32xf32, 1xi64, 1xi64)
        t2122 = paddle._C_ops.slice(t2119, [0], t354, t356, [1], [0])
        del t2119

        # pd_op.scale: (-1x16x144x32xf32) <- (-1x16x144x32xf32, 1xf32)
        t2123 = paddle._C_ops.scale(t2120, t358, float("0"), True)
        del t2120

        # pd_op.transpose: (-1x16x32x144xf32) <- (-1x16x144x32xf32)
        t2124 = paddle._C_ops.transpose(t2121, [0, 1, 3, 2])
        del t2121

        # pd_op.matmul: (-1x16x144x144xf32) <- (-1x16x144x32xf32, -1x16x32x144xf32)
        t2125 = paddle._C_ops.matmul(t2123, t2124, False, False)
        del t2123, t2124

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t2126 = paddle._C_ops.reshape(input43, t362)
        del input43

        # pd_op.index_select: (20736x16xf32) <- (529x16xf32, 20736xi64)
        t2127 = paddle._C_ops.index_select(input44, t2126, 0)
        del input44, t2126

        # pd_op.reshape: (144x144x16xf32) <- (20736x16xf32, 3xi64)
        t2128 = paddle._C_ops.reshape(t2127, t365)
        del t2127

        # pd_op.transpose: (16x144x144xf32) <- (144x144x16xf32)
        t2129 = paddle._C_ops.transpose(t2128, [2, 0, 1])
        del t2128

        # pd_op.unsqueeze: (1x16x144x144xf32) <- (16x144x144xf32, 1xi64)
        t2130 = paddle._C_ops.unsqueeze(t2129, t305)
        del t2129

        # pd_op.add: (-1x16x144x144xf32) <- (-1x16x144x144xf32, 1x16x144x144xf32)
        t2131 = paddle._C_ops.add(t2125, t2130)
        del t2125, t2130

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t2132 = paddle._C_ops.floor_divide(t2113, t875)
        del t875

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2133 = [t2132, t347, t682, t345, t345]
        del t2132, t347

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2134 = paddle._C_ops.stack(t2133, 0)
        del t2133

        # pd_op.reshape: (-1x4x16x144x144xf32) <- (-1x16x144x144xf32, 5xi64)
        t2135 = paddle._C_ops.reshape(t2131, t2134)
        del t2131, t2134

        # pd_op.unsqueeze: (4x1x144x144xf32) <- (4x144x144xf32, 1xi64)
        t2136 = paddle._C_ops.unsqueeze(t2111, t306)
        del t2111

        # pd_op.unsqueeze: (1x4x1x144x144xf32) <- (4x1x144x144xf32, 1xi64)
        t2137 = paddle._C_ops.unsqueeze(t2136, t305)
        del t2136

        # pd_op.add: (-1x4x16x144x144xf32) <- (-1x4x16x144x144xf32, 1x4x1x144x144xf32)
        t2138 = paddle._C_ops.add(t2135, t2137)
        del t2135, t2137

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2139 = [t2113, t682, t345, t345]
        del t682

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2140 = paddle._C_ops.stack(t2139, 0)
        del t2139

        # pd_op.reshape: (-1x16x144x144xf32) <- (-1x4x16x144x144xf32, 4xi64)
        t2141 = paddle._C_ops.reshape(t2138, t2140)
        del t2138, t2140

        # pd_op.softmax: (-1x16x144x144xf32) <- (-1x16x144x144xf32)
        t2142 = paddle._C_ops.softmax(t2141, -1)
        del t2141

        # pd_op.matmul: (-1x16x144x32xf32) <- (-1x16x144x144xf32, -1x16x144x32xf32)
        t2143 = paddle._C_ops.matmul(t2142, t2122, False, False)
        del t2122, t2142

        # pd_op.transpose: (-1x144x16x32xf32) <- (-1x16x144x32xf32)
        t2144 = paddle._C_ops.transpose(t2143, [0, 2, 1, 3])
        del t2143

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2145 = [t2113, t345, t539]
        del t2113

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2146 = paddle._C_ops.stack(t2145, 0)
        del t2145

        # pd_op.reshape: (-1x144x512xf32) <- (-1x144x16x32xf32, 3xi64)
        t2147 = paddle._C_ops.reshape(t2144, t2146)
        del t2146, t2144

        # pd_op.matmul: (-1x144x512xf32) <- (-1x144x512xf32, 512x512xf32)
        t2148 = paddle._C_ops.matmul(t2147, t266, False, False)
        del t266, t2147

        # pd_op.add: (-1x144x512xf32) <- (-1x144x512xf32, 512xf32)
        t2149 = paddle._C_ops.add(t2148, t267)
        del t2148, t267

        # pd_op.reshape: (-1x12x12x512xf32) <- (-1x144x512xf32, 4xi64)
        t2150 = paddle._C_ops.reshape(t2149, t760)
        del t2149, t760

        # pd_op.reshape: (-1x2x2x12x12x512xf32) <- (-1x12x12x512xf32, 6xi64)
        t2151 = paddle._C_ops.reshape(t2150, t793)
        del t793, t2150

        # pd_op.transpose: (-1x2x12x2x12x512xf32) <- (-1x2x2x12x12x512xf32)
        t2152 = paddle._C_ops.transpose(t2151, [0, 1, 3, 2, 4, 5])
        del t2151

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x2x12x2x12x512xf32, 4xi64)
        t2153 = paddle._C_ops.reshape(t2152, t796)
        del t796, t2152

        # pd_op.roll: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 2xi64)
        t2154 = paddle._C_ops.roll(t2153, t503, [1, 2])
        del t2153

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2155 = [t2073, t798, t539]
        del t798, t2073

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2156 = paddle._C_ops.stack(t2155, 0)
        del t2155

        # pd_op.reshape: (-1x576x512xf32) <- (-1x24x24x512xf32, 3xi64)
        t2157 = paddle._C_ops.reshape(t2154, t2156)
        del t2154, t2156

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t2158 = paddle._C_ops.add(t2071, t2157)
        del t2071, t2157

        # pd_op.layer_norm: (-1x576x512xf32, -1x576xf32, -1x576xf32) <- (-1x576x512xf32, 512xf32, 512xf32)
        t2159, t2160, t2161 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2158, t268, t269, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t269, t268

        # pd_op.matmul: (-1x576x2048xf32) <- (-1x576x512xf32, 512x2048xf32)
        t2162 = paddle._C_ops.matmul(t2159, t270, False, False)
        del t2159, t270

        # pd_op.add: (-1x576x2048xf32) <- (-1x576x2048xf32, 2048xf32)
        t2163 = paddle._C_ops.add(t2162, t271)
        del t2162, t271

        # pd_op.gelu: (-1x576x2048xf32) <- (-1x576x2048xf32)
        t2164 = paddle._C_ops.gelu(t2163, False)
        del t2163

        # pd_op.matmul: (-1x576x512xf32) <- (-1x576x2048xf32, 2048x512xf32)
        t2165 = paddle._C_ops.matmul(t2164, t272, False, False)
        del t2164, t272

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, 512xf32)
        t2166 = paddle._C_ops.add(t2165, t273)
        del t2165, t273

        # pd_op.add: (-1x576x512xf32) <- (-1x576x512xf32, -1x576x512xf32)
        t2167 = paddle._C_ops.add(t2158, t2166)
        del t2158, t2166

        # pd_op.shape64: (3xi64) <- (-1x576x512xf32)
        t2168 = paddle._C_ops.shape64(t2167)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2169 = paddle._C_ops.slice(t2168, [0], t305, t306, [1], [0])
        del t2168

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2170 = [t2169, t749, t749, t539]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2171 = paddle._C_ops.stack(t2170, 0)
        del t2170

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x576x512xf32, 4xi64)
        t2172 = paddle._C_ops.reshape(t2167, t2171)
        del t2167, t2171

        # pd_op.strided_slice: (-1x12x12x512xf32) <- (-1x24x24x512xf32, 2xi64, 2xi64, 2xi64)
        t2173 = paddle._C_ops.strided_slice(t2172, [1, 2], t419, t440, t523)

        # pd_op.strided_slice: (-1x12x12x512xf32) <- (-1x24x24x512xf32, 2xi64, 2xi64, 2xi64)
        t2174 = paddle._C_ops.strided_slice(t2172, [1, 2], t525, t440, t523)
        del t525

        # pd_op.strided_slice: (-1x12x12x512xf32) <- (-1x24x24x512xf32, 2xi64, 2xi64, 2xi64)
        t2175 = paddle._C_ops.strided_slice(t2172, [1, 2], t527, t440, t523)
        del t527

        # pd_op.strided_slice: (-1x12x12x512xf32) <- (-1x24x24x512xf32, 2xi64, 2xi64, 2xi64)
        t2176 = paddle._C_ops.strided_slice(t2172, [1, 2], t421, t440, t523)
        del t523

        # pd_op.shape64: (4xi64) <- (-1x24x24x512xf32)
        t2177 = paddle._C_ops.shape64(t2172)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2178 = paddle._C_ops.slice(t2177, [0], t305, t306, [1], [0])
        del t2177

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2179 = [t2178, t749, t749, t539]
        del t539, t749, t2178

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2180 = paddle._C_ops.stack(t2179, 0)
        del t2179

        # pd_op.reshape: (-1x24x24x512xf32) <- (-1x24x24x512xf32, 4xi64)
        t2181 = paddle._C_ops.reshape(t2172, t2180)
        del t2172, t2180

        # builtin.combine: ([-1x12x12x512xf32, -1x12x12x512xf32, -1x12x12x512xf32, -1x12x12x512xf32]) <- (-1x12x12x512xf32, -1x12x12x512xf32, -1x12x12x512xf32, -1x12x12x512xf32)
        t2182 = [t2173, t2174, t2175, t2176]
        del t2175, t2176, t2173, t2174

        # pd_op.concat: (-1x12x12x2048xf32) <- ([-1x12x12x512xf32, -1x12x12x512xf32, -1x12x12x512xf32, -1x12x12x512xf32], 1xi32)
        t2183 = paddle._C_ops.concat(t2182, t535)
        del t2182, t535

        # pd_op.full: (xi64) <- ()
        t2184 = paddle._C_ops.full(
            [], float("2048"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2185 = [t2169, t538, t2184]
        del t538, t2184, t2169

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2186 = paddle._C_ops.stack(t2185, 0)
        del t2185

        # pd_op.reshape: (-1x-1x2048xf32) <- (-1x12x12x2048xf32, 3xi64)
        t2187 = paddle._C_ops.reshape(t2183, t2186)
        del t2183, t2186

        # pd_op.layer_norm: (-1x-1x2048xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x2048xf32, 2048xf32, 2048xf32)
        t2188, t2189, t2190 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2187, t274, t275, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t275, t274, t2187

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x2048xf32, 2048x1024xf32)
        t2191 = paddle._C_ops.matmul(t2188, t276, False, False)
        del t2188, t276

        # pd_op.shape64: (3xi64) <- (-1x-1x1024xf32)
        t2192 = paddle._C_ops.shape64(t2191)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2193 = paddle._C_ops.slice(t2192, [0], t305, t306, [1], [0])
        del t2192

        # pd_op.shape64: (3xi64) <- (-1x-1x1024xf32)
        t2194 = paddle._C_ops.shape64(t2191)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2195 = paddle._C_ops.slice(t2194, [0], t306, t354, [1], [0])
        del t2194

        # pd_op.layer_norm: (-1x-1x1024xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x1024xf32, 1024xf32, 1024xf32)
        t2196, t2197, t2198 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2191, t277, t278, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t278, t277

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2199 = [t2193, t332, t332, t734]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2200 = paddle._C_ops.stack(t2199, 0)
        del t2199

        # pd_op.reshape: (-1x12x12x1024xf32) <- (-1x-1x1024xf32, 4xi64)
        t2201 = paddle._C_ops.reshape(t2196, t2200)
        del t2196, t2200

        # pd_op.shape64: (4xi64) <- (-1x12x12x1024xf32)
        t2202 = paddle._C_ops.shape64(t2201)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2203 = paddle._C_ops.slice(t2202, [0], t305, t306, [1], [0])
        del t2202

        # pd_op.full: (xi64) <- ()
        t2204 = paddle._C_ops.full([], float("1"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2205 = [t2203, t2204, t332, t2204, t332, t734]
        del t2203

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2206 = paddle._C_ops.stack(t2205, 0)
        del t2205

        # pd_op.reshape: (-1x1x12x1x12x1024xf32) <- (-1x12x12x1024xf32, 6xi64)
        t2207 = paddle._C_ops.reshape(t2201, t2206)
        del t2201, t2206

        # pd_op.transpose: (-1x1x1x12x12x1024xf32) <- (-1x1x12x1x12x1024xf32)
        t2208 = paddle._C_ops.transpose(t2207, [0, 1, 3, 2, 4, 5])
        del t2207

        # pd_op.full_int_array: (4xi64) <- ()
        t2209 = [-1, 12, 12, 1024]

        # pd_op.reshape: (-1x12x12x1024xf32) <- (-1x1x1x12x12x1024xf32, 4xi64)
        t2210 = paddle._C_ops.reshape(t2208, t2209)
        del t2208

        # pd_op.full_int_array: (3xi64) <- ()
        t2211 = [-1, 144, 1024]

        # pd_op.reshape: (-1x144x1024xf32) <- (-1x12x12x1024xf32, 3xi64)
        t2212 = paddle._C_ops.reshape(t2210, t2211)
        del t2210

        # pd_op.shape64: (3xi64) <- (-1x144x1024xf32)
        t2213 = paddle._C_ops.shape64(t2212)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2214 = paddle._C_ops.slice(t2213, [0], t305, t306, [1], [0])
        del t2213

        # pd_op.matmul: (-1x144x3072xf32) <- (-1x144x1024xf32, 1024x3072xf32)
        t2215 = paddle._C_ops.matmul(t2212, t279, False, False)
        del t279, t2212

        # pd_op.add: (-1x144x3072xf32) <- (-1x144x3072xf32, 3072xf32)
        t2216 = paddle._C_ops.add(t2215, t280)
        del t2215, t280

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2217 = [t2214, t345, t346, t348, t348]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2218 = paddle._C_ops.stack(t2217, 0)
        del t2217

        # pd_op.reshape: (-1x144x3x32x32xf32) <- (-1x144x3072xf32, 5xi64)
        t2219 = paddle._C_ops.reshape(t2216, t2218)
        del t2216, t2218

        # pd_op.transpose: (3x-1x32x144x32xf32) <- (-1x144x3x32x32xf32)
        t2220 = paddle._C_ops.transpose(t2219, [2, 0, 3, 1, 4])
        del t2219

        # pd_op.slice: (-1x32x144x32xf32) <- (3x-1x32x144x32xf32, 1xi64, 1xi64)
        t2221 = paddle._C_ops.slice(t2220, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x32x144x32xf32) <- (3x-1x32x144x32xf32, 1xi64, 1xi64)
        t2222 = paddle._C_ops.slice(t2220, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x32x144x32xf32) <- (3x-1x32x144x32xf32, 1xi64, 1xi64)
        t2223 = paddle._C_ops.slice(t2220, [0], t354, t356, [1], [0])
        del t2220

        # pd_op.scale: (-1x32x144x32xf32) <- (-1x32x144x32xf32, 1xf32)
        t2224 = paddle._C_ops.scale(t2221, t358, float("0"), True)
        del t2221

        # pd_op.transpose: (-1x32x32x144xf32) <- (-1x32x144x32xf32)
        t2225 = paddle._C_ops.transpose(t2222, [0, 1, 3, 2])
        del t2222

        # pd_op.matmul: (-1x32x144x144xf32) <- (-1x32x144x32xf32, -1x32x32x144xf32)
        t2226 = paddle._C_ops.matmul(t2224, t2225, False, False)
        del t2224, t2225

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t2227 = paddle._C_ops.reshape(input45, t362)
        del input45

        # pd_op.index_select: (20736x32xf32) <- (529x32xf32, 20736xi64)
        t2228 = paddle._C_ops.index_select(input46, t2227, 0)
        del input46, t2227

        # pd_op.reshape: (144x144x32xf32) <- (20736x32xf32, 3xi64)
        t2229 = paddle._C_ops.reshape(t2228, t365)
        del t2228

        # pd_op.transpose: (32x144x144xf32) <- (144x144x32xf32)
        t2230 = paddle._C_ops.transpose(t2229, [2, 0, 1])
        del t2229

        # pd_op.unsqueeze: (1x32x144x144xf32) <- (32x144x144xf32, 1xi64)
        t2231 = paddle._C_ops.unsqueeze(t2230, t305)
        del t2230

        # pd_op.add: (-1x32x144x144xf32) <- (-1x32x144x144xf32, 1x32x144x144xf32)
        t2232 = paddle._C_ops.add(t2226, t2231)
        del t2226, t2231

        # pd_op.softmax: (-1x32x144x144xf32) <- (-1x32x144x144xf32)
        t2233 = paddle._C_ops.softmax(t2232, -1)
        del t2232

        # pd_op.matmul: (-1x32x144x32xf32) <- (-1x32x144x144xf32, -1x32x144x32xf32)
        t2234 = paddle._C_ops.matmul(t2233, t2223, False, False)
        del t2223, t2233

        # pd_op.transpose: (-1x144x32x32xf32) <- (-1x32x144x32xf32)
        t2235 = paddle._C_ops.transpose(t2234, [0, 2, 1, 3])
        del t2234

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2236 = [t2214, t345, t734]
        del t2214

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2237 = paddle._C_ops.stack(t2236, 0)
        del t2236

        # pd_op.reshape: (-1x144x1024xf32) <- (-1x144x32x32xf32, 3xi64)
        t2238 = paddle._C_ops.reshape(t2235, t2237)
        del t2237, t2235

        # pd_op.matmul: (-1x144x1024xf32) <- (-1x144x1024xf32, 1024x1024xf32)
        t2239 = paddle._C_ops.matmul(t2238, t281, False, False)
        del t281, t2238

        # pd_op.add: (-1x144x1024xf32) <- (-1x144x1024xf32, 1024xf32)
        t2240 = paddle._C_ops.add(t2239, t282)
        del t2239, t282

        # pd_op.reshape: (-1x12x12x1024xf32) <- (-1x144x1024xf32, 4xi64)
        t2241 = paddle._C_ops.reshape(t2240, t2209)
        del t2240

        # pd_op.full_int_array: (6xi64) <- ()
        t2242 = [-1, 1, 1, 12, 12, 1024]

        # pd_op.reshape: (-1x1x1x12x12x1024xf32) <- (-1x12x12x1024xf32, 6xi64)
        t2243 = paddle._C_ops.reshape(t2241, t2242)
        del t2241

        # pd_op.transpose: (-1x1x12x1x12x1024xf32) <- (-1x1x1x12x12x1024xf32)
        t2244 = paddle._C_ops.transpose(t2243, [0, 1, 3, 2, 4, 5])
        del t2243

        # pd_op.reshape: (-1x12x12x1024xf32) <- (-1x1x12x1x12x1024xf32, 4xi64)
        t2245 = paddle._C_ops.reshape(t2244, t2209)
        del t2244

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2246 = [t2193, t345, t734]
        del t2193

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2247 = paddle._C_ops.stack(t2246, 0)
        del t2246

        # pd_op.reshape: (-1x144x1024xf32) <- (-1x12x12x1024xf32, 3xi64)
        t2248 = paddle._C_ops.reshape(t2245, t2247)
        del t2245, t2247

        # pd_op.add: (-1x144x1024xf32) <- (-1x-1x1024xf32, -1x144x1024xf32)
        t2249 = paddle._C_ops.add(t2191, t2248)
        del t2191, t2248

        # pd_op.layer_norm: (-1x144x1024xf32, -1x144xf32, -1x144xf32) <- (-1x144x1024xf32, 1024xf32, 1024xf32)
        t2250, t2251, t2252 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2249, t283, t284, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t284, t283

        # pd_op.matmul: (-1x144x4096xf32) <- (-1x144x1024xf32, 1024x4096xf32)
        t2253 = paddle._C_ops.matmul(t2250, t285, False, False)
        del t2250, t285

        # pd_op.add: (-1x144x4096xf32) <- (-1x144x4096xf32, 4096xf32)
        t2254 = paddle._C_ops.add(t2253, t286)
        del t2253, t286

        # pd_op.gelu: (-1x144x4096xf32) <- (-1x144x4096xf32)
        t2255 = paddle._C_ops.gelu(t2254, False)
        del t2254

        # pd_op.matmul: (-1x144x1024xf32) <- (-1x144x4096xf32, 4096x1024xf32)
        t2256 = paddle._C_ops.matmul(t2255, t287, False, False)
        del t2255, t287

        # pd_op.add: (-1x144x1024xf32) <- (-1x144x1024xf32, 1024xf32)
        t2257 = paddle._C_ops.add(t2256, t288)
        del t2256, t288

        # pd_op.add: (-1x144x1024xf32) <- (-1x144x1024xf32, -1x144x1024xf32)
        t2258 = paddle._C_ops.add(t2249, t2257)
        del t2249, t2257

        # pd_op.shape64: (3xi64) <- (-1x144x1024xf32)
        t2259 = paddle._C_ops.shape64(t2258)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2260 = paddle._C_ops.slice(t2259, [0], t305, t306, [1], [0])
        del t2259

        # pd_op.layer_norm: (-1x144x1024xf32, -1x144xf32, -1x144xf32) <- (-1x144x1024xf32, 1024xf32, 1024xf32)
        t2261, t2262, t2263 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2258, t289, t290, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t290, t289

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2264 = [t2260, t332, t332, t734]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2265 = paddle._C_ops.stack(t2264, 0)
        del t2264

        # pd_op.reshape: (-1x12x12x1024xf32) <- (-1x144x1024xf32, 4xi64)
        t2266 = paddle._C_ops.reshape(t2261, t2265)
        del t2261, t2265

        # pd_op.shape64: (4xi64) <- (-1x12x12x1024xf32)
        t2267 = paddle._C_ops.shape64(t2266)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2268 = paddle._C_ops.slice(t2267, [0], t305, t306, [1], [0])
        del t2267

        # pd_op.roll: (-1x12x12x1024xf32) <- (-1x12x12x1024xf32, 2xi64)
        t2269 = paddle._C_ops.roll(t2266, t408, [1, 2])
        del t2266

        # pd_op.shape64: (4xi64) <- (-1x12x12x1024xf32)
        t2270 = paddle._C_ops.shape64(t2269)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2271 = paddle._C_ops.slice(t2270, [0], t305, t306, [1], [0])
        del t2270

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2272 = [t2271, t2204, t332, t2204, t332, t734]
        del t332, t2271

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2273 = paddle._C_ops.stack(t2272, 0)
        del t2272

        # pd_op.reshape: (-1x1x12x1x12x1024xf32) <- (-1x12x12x1024xf32, 6xi64)
        t2274 = paddle._C_ops.reshape(t2269, t2273)
        del t2269, t2273

        # pd_op.transpose: (-1x1x1x12x12x1024xf32) <- (-1x1x12x1x12x1024xf32)
        t2275 = paddle._C_ops.transpose(t2274, [0, 1, 3, 2, 4, 5])
        del t2274

        # pd_op.reshape: (-1x12x12x1024xf32) <- (-1x1x1x12x12x1024xf32, 4xi64)
        t2276 = paddle._C_ops.reshape(t2275, t2209)
        del t2275

        # pd_op.reshape: (-1x144x1024xf32) <- (-1x12x12x1024xf32, 3xi64)
        t2277 = paddle._C_ops.reshape(t2276, t2211)
        del t2211, t2276

        # pd_op.full: (1x12x12x1xf32) <- ()
        t2278 = paddle._C_ops.full(
            [1, 12, 12, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2279 = paddle._C_ops.set_value_(
            t2278, t419, t420, t421, [1, 2], [], [], [1], [float("0")]
        )
        del t2278, t419

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2280 = paddle._C_ops.set_value_(
            t2279, t423, t424, t421, [1, 2], [], [], [1], [float("1")]
        )
        del t423, t2279

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2281 = paddle._C_ops.set_value_(
            t2280, t426, t427, t421, [1, 2], [], [], [1], [float("2")]
        )
        del t426, t427, t2280

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2282 = paddle._C_ops.set_value_(
            t2281, t429, t430, t421, [1, 2], [], [], [1], [float("3")]
        )
        del t429, t2281

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2283 = paddle._C_ops.set_value_(
            t2282, t420, t408, t421, [1, 2], [], [], [1], [float("4")]
        )
        del t420, t2282

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2284 = paddle._C_ops.set_value_(
            t2283, t424, t433, t421, [1, 2], [], [], [1], [float("5")]
        )
        del t424, t433, t2283

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2285 = paddle._C_ops.set_value_(
            t2284, t435, t436, t421, [1, 2], [], [], [1], [float("6")]
        )
        del t435, t436, t2284

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2286 = paddle._C_ops.set_value_(
            t2285, t430, t438, t421, [1, 2], [], [], [1], [float("7")]
        )
        del t430, t438, t2285

        # pd_op.set_value_: (1x12x12x1xf32) <- (1x12x12x1xf32, 2xi64, 2xi64, 2xi64)
        t2287 = paddle._C_ops.set_value_(
            t2286, t408, t440, t421, [1, 2], [], [], [1], [float("8")]
        )
        del t408, t440, t2286

        # pd_op.full_int_array: (6xi64) <- ()
        t2288 = [1, 1, 12, 1, 12, 1]

        # pd_op.reshape: (1x1x12x1x12x1xf32) <- (1x12x12x1xf32, 6xi64)
        t2289 = paddle._C_ops.reshape(t2287, t2288)
        del t2288

        # pd_op.transpose: (1x1x1x12x12x1xf32) <- (1x1x12x1x12x1xf32)
        t2290 = paddle._C_ops.transpose(t2289, [0, 1, 3, 2, 4, 5])
        del t2289

        # pd_op.reshape: (1x12x12x1xf32) <- (1x1x1x12x12x1xf32, 4xi64)
        t2291 = paddle._C_ops.reshape(t2290, t445)
        del t445, t2290

        # pd_op.reshape: (1x144xf32) <- (1x12x12x1xf32, 2xi64)
        t2292 = paddle._C_ops.reshape(t2291, t447)
        del t447, t2291

        # pd_op.unsqueeze: (1x1x144xf32) <- (1x144xf32, 1xi64)
        t2293 = paddle._C_ops.unsqueeze(t2292, t306)

        # pd_op.unsqueeze: (1x144x1xf32) <- (1x144xf32, 1xi64)
        t2294 = paddle._C_ops.unsqueeze(t2292, t354)
        del t2292

        # pd_op.subtract: (1x144x144xf32) <- (1x1x144xf32, 1x144x1xf32)
        t2295 = paddle._C_ops.subtract(t2293, t2294)
        del t2293, t2294

        # pd_op.not_equal: (1x144x144xb) <- (1x144x144xf32, xf32)
        t2296 = paddle._C_ops.not_equal(t2295, t452)

        # pd_op.full: (1x144x144xf32) <- ()
        t2297 = paddle._C_ops.full(
            [1, 144, 144],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x144x144xf32) <- (1x144x144xb, 1x144x144xf32, 1x144x144xf32)
        t2298 = paddle._C_ops.where(t2296, t2297, t2295)
        del t2297, t2296, t2295

        # pd_op.equal: (1x144x144xb) <- (1x144x144xf32, xf32)
        t2299 = paddle._C_ops.equal(t2298, t452)
        del t452

        # pd_op.full: (1x144x144xf32) <- ()
        t2300 = paddle._C_ops.full(
            [1, 144, 144],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x144x144xf32) <- (1x144x144xb, 1x144x144xf32, 1x144x144xf32)
        t2301 = paddle._C_ops.where(t2299, t2300, t2298)
        del t2299, t2300, t2298

        # pd_op.shape64: (3xi64) <- (-1x144x1024xf32)
        t2302 = paddle._C_ops.shape64(t2277)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2303 = paddle._C_ops.slice(t2302, [0], t305, t306, [1], [0])
        del t2302

        # pd_op.matmul: (-1x144x3072xf32) <- (-1x144x1024xf32, 1024x3072xf32)
        t2304 = paddle._C_ops.matmul(t2277, t291, False, False)
        del t291, t2277

        # pd_op.add: (-1x144x3072xf32) <- (-1x144x3072xf32, 3072xf32)
        t2305 = paddle._C_ops.add(t2304, t292)
        del t2304, t292

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2306 = [t2303, t345, t346, t348, t348]
        del t346

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2307 = paddle._C_ops.stack(t2306, 0)
        del t2306

        # pd_op.reshape: (-1x144x3x32x32xf32) <- (-1x144x3072xf32, 5xi64)
        t2308 = paddle._C_ops.reshape(t2305, t2307)
        del t2305, t2307

        # pd_op.transpose: (3x-1x32x144x32xf32) <- (-1x144x3x32x32xf32)
        t2309 = paddle._C_ops.transpose(t2308, [2, 0, 3, 1, 4])
        del t2308

        # pd_op.slice: (-1x32x144x32xf32) <- (3x-1x32x144x32xf32, 1xi64, 1xi64)
        t2310 = paddle._C_ops.slice(t2309, [0], t305, t306, [1], [0])

        # pd_op.slice: (-1x32x144x32xf32) <- (3x-1x32x144x32xf32, 1xi64, 1xi64)
        t2311 = paddle._C_ops.slice(t2309, [0], t306, t354, [1], [0])

        # pd_op.slice: (-1x32x144x32xf32) <- (3x-1x32x144x32xf32, 1xi64, 1xi64)
        t2312 = paddle._C_ops.slice(t2309, [0], t354, t356, [1], [0])
        del t356, t2309

        # pd_op.scale: (-1x32x144x32xf32) <- (-1x32x144x32xf32, 1xf32)
        t2313 = paddle._C_ops.scale(t2310, t358, float("0"), True)
        del t358, t2310

        # pd_op.transpose: (-1x32x32x144xf32) <- (-1x32x144x32xf32)
        t2314 = paddle._C_ops.transpose(t2311, [0, 1, 3, 2])
        del t2311

        # pd_op.matmul: (-1x32x144x144xf32) <- (-1x32x144x32xf32, -1x32x32x144xf32)
        t2315 = paddle._C_ops.matmul(t2313, t2314, False, False)
        del t2313, t2314

        # pd_op.reshape: (20736xi64) <- (144x144xi64, 1xi64)
        t2316 = paddle._C_ops.reshape(input47, t362)
        del input47, t362

        # pd_op.index_select: (20736x32xf32) <- (529x32xf32, 20736xi64)
        t2317 = paddle._C_ops.index_select(input48, t2316, 0)
        del input48, t2316

        # pd_op.reshape: (144x144x32xf32) <- (20736x32xf32, 3xi64)
        t2318 = paddle._C_ops.reshape(t2317, t365)
        del t365, t2317

        # pd_op.transpose: (32x144x144xf32) <- (144x144x32xf32)
        t2319 = paddle._C_ops.transpose(t2318, [2, 0, 1])
        del t2318

        # pd_op.unsqueeze: (1x32x144x144xf32) <- (32x144x144xf32, 1xi64)
        t2320 = paddle._C_ops.unsqueeze(t2319, t305)
        del t2319

        # pd_op.add: (-1x32x144x144xf32) <- (-1x32x144x144xf32, 1x32x144x144xf32)
        t2321 = paddle._C_ops.add(t2315, t2320)
        del t2315, t2320

        # pd_op.full: (xi64) <- ()
        t2322 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t2323 = paddle._C_ops.floor_divide(t2303, t2322)
        del t2322

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2324 = [t2323, t2204, t348, t345, t345]
        del t2323, t2204

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2325 = paddle._C_ops.stack(t2324, 0)
        del t2324

        # pd_op.reshape: (-1x1x32x144x144xf32) <- (-1x32x144x144xf32, 5xi64)
        t2326 = paddle._C_ops.reshape(t2321, t2325)
        del t2321, t2325

        # pd_op.unsqueeze: (1x1x144x144xf32) <- (1x144x144xf32, 1xi64)
        t2327 = paddle._C_ops.unsqueeze(t2301, t306)
        del t306, t2301

        # pd_op.unsqueeze: (1x1x1x144x144xf32) <- (1x1x144x144xf32, 1xi64)
        t2328 = paddle._C_ops.unsqueeze(t2327, t305)
        del t305, t2327

        # pd_op.add: (-1x1x32x144x144xf32) <- (-1x1x32x144x144xf32, 1x1x1x144x144xf32)
        t2329 = paddle._C_ops.add(t2326, t2328)
        del t2326, t2328

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2330 = [t2303, t348, t345, t345]
        del t348

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2331 = paddle._C_ops.stack(t2330, 0)
        del t2330

        # pd_op.reshape: (-1x32x144x144xf32) <- (-1x1x32x144x144xf32, 4xi64)
        t2332 = paddle._C_ops.reshape(t2329, t2331)
        del t2329, t2331

        # pd_op.softmax: (-1x32x144x144xf32) <- (-1x32x144x144xf32)
        t2333 = paddle._C_ops.softmax(t2332, -1)
        del t2332

        # pd_op.matmul: (-1x32x144x32xf32) <- (-1x32x144x144xf32, -1x32x144x32xf32)
        t2334 = paddle._C_ops.matmul(t2333, t2312, False, False)
        del t2312, t2333

        # pd_op.transpose: (-1x144x32x32xf32) <- (-1x32x144x32xf32)
        t2335 = paddle._C_ops.transpose(t2334, [0, 2, 1, 3])
        del t2334

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2336 = [t2303, t345, t734]
        del t2303

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2337 = paddle._C_ops.stack(t2336, 0)
        del t2336

        # pd_op.reshape: (-1x144x1024xf32) <- (-1x144x32x32xf32, 3xi64)
        t2338 = paddle._C_ops.reshape(t2335, t2337)
        del t2337, t2335

        # pd_op.matmul: (-1x144x1024xf32) <- (-1x144x1024xf32, 1024x1024xf32)
        t2339 = paddle._C_ops.matmul(t2338, t293, False, False)
        del t293, t2338

        # pd_op.add: (-1x144x1024xf32) <- (-1x144x1024xf32, 1024xf32)
        t2340 = paddle._C_ops.add(t2339, t294)
        del t2339, t294

        # pd_op.reshape: (-1x12x12x1024xf32) <- (-1x144x1024xf32, 4xi64)
        t2341 = paddle._C_ops.reshape(t2340, t2209)
        del t2340

        # pd_op.reshape: (-1x1x1x12x12x1024xf32) <- (-1x12x12x1024xf32, 6xi64)
        t2342 = paddle._C_ops.reshape(t2341, t2242)
        del t2242, t2341

        # pd_op.transpose: (-1x1x12x1x12x1024xf32) <- (-1x1x1x12x12x1024xf32)
        t2343 = paddle._C_ops.transpose(t2342, [0, 1, 3, 2, 4, 5])
        del t2342

        # pd_op.reshape: (-1x12x12x1024xf32) <- (-1x1x12x1x12x1024xf32, 4xi64)
        t2344 = paddle._C_ops.reshape(t2343, t2209)
        del t2209, t2343

        # pd_op.roll: (-1x12x12x1024xf32) <- (-1x12x12x1024xf32, 2xi64)
        t2345 = paddle._C_ops.roll(t2344, t503, [1, 2])
        del t503, t2344

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2346 = [t2260, t345, t734]
        del t734, t345, t2260

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2347 = paddle._C_ops.stack(t2346, 0)
        del t2346

        # pd_op.reshape: (-1x144x1024xf32) <- (-1x12x12x1024xf32, 3xi64)
        t2348 = paddle._C_ops.reshape(t2345, t2347)
        del t2345, t2347

        # pd_op.add: (-1x144x1024xf32) <- (-1x144x1024xf32, -1x144x1024xf32)
        t2349 = paddle._C_ops.add(t2258, t2348)
        del t2258, t2348

        # pd_op.layer_norm: (-1x144x1024xf32, -1x144xf32, -1x144xf32) <- (-1x144x1024xf32, 1024xf32, 1024xf32)
        t2350, t2351, t2352 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2349, t295, t296, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t296, t295

        # pd_op.matmul: (-1x144x4096xf32) <- (-1x144x1024xf32, 1024x4096xf32)
        t2353 = paddle._C_ops.matmul(t2350, t297, False, False)
        del t2350, t297

        # pd_op.add: (-1x144x4096xf32) <- (-1x144x4096xf32, 4096xf32)
        t2354 = paddle._C_ops.add(t2353, t298)
        del t2353, t298

        # pd_op.gelu: (-1x144x4096xf32) <- (-1x144x4096xf32)
        t2355 = paddle._C_ops.gelu(t2354, False)
        del t2354

        # pd_op.matmul: (-1x144x1024xf32) <- (-1x144x4096xf32, 4096x1024xf32)
        t2356 = paddle._C_ops.matmul(t2355, t299, False, False)
        del t2355, t299

        # pd_op.add: (-1x144x1024xf32) <- (-1x144x1024xf32, 1024xf32)
        t2357 = paddle._C_ops.add(t2356, t300)
        del t2356, t300

        # pd_op.add: (-1x144x1024xf32) <- (-1x144x1024xf32, -1x144x1024xf32)
        t2358 = paddle._C_ops.add(t2349, t2357)
        del t2349, t2357

        # pd_op.layer_norm: (-1x144x1024xf32, -1x144xf32, -1x144xf32) <- (-1x144x1024xf32, 1024xf32, 1024xf32)
        t2359, t2360, t2361 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2358, t301, t302, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t2358, t302, t301

        # pd_op.transpose: (-1x1024x144xf32) <- (-1x144x1024xf32)
        t2362 = paddle._C_ops.transpose(t2359, [0, 2, 1])
        del t2359

        # pd_op.unsqueeze: (-1x1024x1x144xf32) <- (-1x1024x144xf32, 1xi64)
        t2363 = paddle._C_ops.unsqueeze(t2362, t354)
        del t2362

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x1x144xf32, 2xi64)
        t2364 = paddle._C_ops.pool2d(
            t2363,
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
        del t421, t2363

        # pd_op.squeeze: (-1x1024x1xf32) <- (-1x1024x1x1xf32, 1xi64)
        t2365 = paddle._C_ops.squeeze(t2364, t354)
        del t354, t2364

        # pd_op.flatten: (-1x1024xf32) <- (-1x1024x1xf32)
        t2366 = paddle._C_ops.flatten(t2365, 1, 2)
        del t2365

        # pd_op.matmul: (-1x102xf32) <- (-1x1024xf32, 1024x102xf32)
        t2367 = paddle._C_ops.matmul(t2366, t303, False, False)
        del t2366, t303

        return (
            t441,
            t645,
            t840,
            t1001,
            t1158,
            t1315,
            t1472,
            t1629,
            t1786,
            t1943,
            t2100,
            t2287,
            t2367,
        )
