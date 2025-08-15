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
        # pd_op.shape64: (4xi64) <- (-1x3x224x224xf32)
        t304 = paddle._C_ops.shape64(input0)

        # pd_op.full_int_array: (1xi64) <- ()
        t305 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        t306 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t307 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t308 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t309 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t310 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t311 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t312 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t313 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t314 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t315 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t316 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t317 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t318 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t319 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t320 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t321 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t322 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t323 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t324 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t325 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t326 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t327 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t328 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t329 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t330 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t331 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t332 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t333 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t334 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t335 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t336 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t337 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t338 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t339 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t340 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t341 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t342 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t343 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t344 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t345 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t346 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t347 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t348 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t349 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t350 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t351 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t352 = t305

        # pd_op.assign: (1xi64) <- (1xi64)
        t353 = t305

        # pd_op.full_int_array: (1xi64) <- ()
        t354 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t355 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t356 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t357 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t358 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t359 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t360 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t361 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t362 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t363 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t364 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t365 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t366 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t367 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t368 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t369 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t370 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t371 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t372 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t373 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t374 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t375 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t376 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t377 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t378 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t379 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t380 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t381 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t382 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t383 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t384 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t385 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t386 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t387 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t388 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t389 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t390 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t391 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t392 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t393 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t394 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t395 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t396 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t397 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t398 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t399 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t400 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t401 = t354

        # pd_op.assign: (1xi64) <- (1xi64)
        t402 = t354

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t403 = paddle._C_ops.slice(t304, [0], t305, t354, [1], [0])
        del t304

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x3x224x224xf32, 192x3x4x4xf32)
        t404 = paddle._C_ops.conv2d(
            input0, t0, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t405 = [1, -1, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t406 = paddle._C_ops.reshape(t1, t405)
        del t405, t1

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, 1x192x1x1xf32)
        t407 = paddle._C_ops.add(t404, t406)

        # pd_op.shape64: (4xi64) <- (-1x192x56x56xf32)
        t408 = paddle._C_ops.shape64(t407)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t409 = paddle._C_ops.slice(t408, [0], t305, t354, [1], [0])
        del t408

        # pd_op.flatten: (-1x192x3136xf32) <- (-1x192x56x56xf32)
        t410 = paddle._C_ops.flatten(t407, 2, 3)

        # pd_op.transpose: (-1x3136x192xf32) <- (-1x192x3136xf32)
        t411 = paddle._C_ops.transpose(t410, [0, 2, 1])
        del t410

        # pd_op.layer_norm: (-1x3136x192xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x192xf32, 192xf32, 192xf32)
        t412, t413, t414 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t411, t2, t3, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t3, t2

        # pd_op.shape64: (3xi64) <- (-1x3136x192xf32)
        t415 = paddle._C_ops.shape64(t412)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t416 = paddle._C_ops.slice(t415, [0], t305, t354, [1], [0])
        del t415

        # pd_op.layer_norm: (-1x3136x192xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x192xf32, 192xf32, 192xf32)
        t417, t418, t419 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t412, t4, t5, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t5, t4

        # pd_op.full: (xi64) <- ()
        t420 = paddle._C_ops.full([], float("56"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t421 = paddle._C_ops.full(
            [], float("192"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t422 = [t416, t420, t420, t421]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t423 = paddle._C_ops.stack(t422, 0)
        del t422

        # pd_op.reshape: (-1x56x56x192xf32) <- (-1x3136x192xf32, 4xi64)
        t424 = paddle._C_ops.reshape(t417, t423)
        del t423

        # pd_op.shape64: (4xi64) <- (-1x56x56x192xf32)
        t425 = paddle._C_ops.shape64(t424)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t426 = paddle._C_ops.slice(t425, [0], t305, t354, [1], [0])
        del t425

        # pd_op.full: (xi64) <- ()
        t427 = paddle._C_ops.full([], float("8"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t428 = paddle._C_ops.full([], float("7"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t429 = [t426, t427, t428, t427, t428, t421]
        del t426

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t430 = paddle._C_ops.stack(t429, 0)
        del t429

        # pd_op.reshape: (-1x8x7x8x7x192xf32) <- (-1x56x56x192xf32, 6xi64)
        t431 = paddle._C_ops.reshape(t424, t430)
        del t430

        # pd_op.transpose: (-1x8x8x7x7x192xf32) <- (-1x8x7x8x7x192xf32)
        t432 = paddle._C_ops.transpose(t431, [0, 1, 3, 2, 4, 5])
        del t431

        # pd_op.full_int_array: (4xi64) <- ()
        t433 = [-1, 7, 7, 192]

        # pd_op.reshape: (-1x7x7x192xf32) <- (-1x8x8x7x7x192xf32, 4xi64)
        t434 = paddle._C_ops.reshape(t432, t433)

        # pd_op.full_int_array: (3xi64) <- ()
        t435 = [-1, 49, 192]

        # pd_op.reshape: (-1x49x192xf32) <- (-1x7x7x192xf32, 3xi64)
        t436 = paddle._C_ops.reshape(t434, t435)

        # pd_op.shape64: (3xi64) <- (-1x49x192xf32)
        t437 = paddle._C_ops.shape64(t436)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t438 = paddle._C_ops.slice(t437, [0], t305, t354, [1], [0])
        del t437

        # pd_op.matmul: (-1x49x576xf32) <- (-1x49x192xf32, 192x576xf32)
        t439 = paddle._C_ops.matmul(t436, t6, False, False)
        del t6

        # pd_op.add: (-1x49x576xf32) <- (-1x49x576xf32, 576xf32)
        t440 = paddle._C_ops.add(t439, t7)
        del t7

        # pd_op.full: (xi64) <- ()
        t441 = paddle._C_ops.full([], float("49"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t442 = paddle._C_ops.full([], float("3"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t443 = paddle._C_ops.full([], float("6"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t444 = paddle._C_ops.full([], float("32"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t445 = [t438, t441, t442, t443, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t446 = paddle._C_ops.stack(t445, 0)
        del t445

        # pd_op.reshape: (-1x49x3x6x32xf32) <- (-1x49x576xf32, 5xi64)
        t447 = paddle._C_ops.reshape(t440, t446)
        del t446

        # pd_op.transpose: (3x-1x6x49x32xf32) <- (-1x49x3x6x32xf32)
        t448 = paddle._C_ops.transpose(t447, [2, 0, 3, 1, 4])
        del t447

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        t449 = paddle._C_ops.slice(t448, [0], t305, t354, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t450 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        t451 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t452 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t453 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t454 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t455 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t456 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t457 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t458 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t459 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t460 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t461 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t462 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t463 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t464 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t465 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t466 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t467 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t468 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t469 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t470 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t471 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t472 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t473 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t474 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t475 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t476 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t477 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t478 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t479 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t480 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t481 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t482 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t483 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t484 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t485 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t486 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t487 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t488 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t489 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t490 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t491 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t492 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t493 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t494 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t495 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t496 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t497 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t498 = t450

        # pd_op.assign: (1xi64) <- (1xi64)
        t499 = t450

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        t500 = paddle._C_ops.slice(t448, [0], t354, t450, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t501 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        t502 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t503 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t504 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t505 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t506 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t507 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t508 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t509 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t510 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t511 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t512 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t513 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t514 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t515 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t516 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t517 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t518 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t519 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t520 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t521 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t522 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t523 = t501

        # pd_op.assign: (1xi64) <- (1xi64)
        t524 = t501

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        t525 = paddle._C_ops.slice(t448, [0], t450, t501, [1], [0])

        # pd_op.full: (1xf32) <- ()
        t526 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t527 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t528 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t529 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t530 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t531 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t532 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t533 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t534 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t535 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t536 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t537 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t538 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t539 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t540 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t541 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t542 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t543 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t544 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t545 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t546 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t547 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t548 = t526

        # pd_op.assign: (1xf32) <- (1xf32)
        t549 = t526

        # pd_op.scale: (-1x6x49x32xf32) <- (-1x6x49x32xf32, 1xf32)
        t550 = paddle._C_ops.scale(t449, t526, float("0"), True)
        del t449

        # pd_op.transpose: (-1x6x32x49xf32) <- (-1x6x49x32xf32)
        t551 = paddle._C_ops.transpose(t500, [0, 1, 3, 2])
        del t500

        # pd_op.matmul: (-1x6x49x49xf32) <- (-1x6x49x32xf32, -1x6x32x49xf32)
        t552 = paddle._C_ops.matmul(t550, t551, False, False)

        # pd_op.full_int_array: (1xi64) <- ()
        t553 = [-1]

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t554 = paddle._C_ops.reshape(input1, t553)
        del input1

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        t555 = paddle._C_ops.index_select(input2, t554, 0)
        del input2

        # pd_op.full_int_array: (3xi64) <- ()
        t556 = [49, 49, -1]

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        t557 = paddle._C_ops.reshape(t555, t556)

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        t558 = paddle._C_ops.transpose(t557, [2, 0, 1])
        del t557

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        t559 = paddle._C_ops.unsqueeze(t558, t305)

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x49x49xf32, 1x6x49x49xf32)
        t560 = paddle._C_ops.add(t552, t559)

        # pd_op.softmax: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        t561 = paddle._C_ops.softmax(t560, -1)
        del t560

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x49x32xf32)
        t562 = paddle._C_ops.matmul(t561, t525, False, False)

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        t563 = paddle._C_ops.transpose(t562, [0, 2, 1, 3])
        del t562

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t564 = [t438, t441, t421]
        del t438

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t565 = paddle._C_ops.stack(t564, 0)
        del t564

        # pd_op.reshape: (-1x49x192xf32) <- (-1x49x6x32xf32, 3xi64)
        t566 = paddle._C_ops.reshape(t563, t565)
        del t565

        # pd_op.matmul: (-1x49x192xf32) <- (-1x49x192xf32, 192x192xf32)
        t567 = paddle._C_ops.matmul(t566, t8, False, False)
        del t8

        # pd_op.add: (-1x49x192xf32) <- (-1x49x192xf32, 192xf32)
        t568 = paddle._C_ops.add(t567, t9)
        del t9

        # pd_op.reshape: (-1x7x7x192xf32) <- (-1x49x192xf32, 4xi64)
        t569 = paddle._C_ops.reshape(t568, t433)

        # pd_op.full_int_array: (6xi64) <- ()
        t570 = [-1, 8, 8, 7, 7, 192]

        # pd_op.reshape: (-1x8x8x7x7x192xf32) <- (-1x7x7x192xf32, 6xi64)
        t571 = paddle._C_ops.reshape(t569, t570)

        # pd_op.transpose: (-1x8x7x8x7x192xf32) <- (-1x8x8x7x7x192xf32)
        t572 = paddle._C_ops.transpose(t571, [0, 1, 3, 2, 4, 5])
        del t571

        # pd_op.full_int_array: (4xi64) <- ()
        t573 = [-1, 56, 56, 192]

        # pd_op.reshape: (-1x56x56x192xf32) <- (-1x8x7x8x7x192xf32, 4xi64)
        t574 = paddle._C_ops.reshape(t572, t573)

        # pd_op.full: (xi64) <- ()
        t575 = paddle._C_ops.full(
            [], float("3136"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t576 = [t416, t575, t421]
        del t416

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t577 = paddle._C_ops.stack(t576, 0)
        del t576

        # pd_op.reshape: (-1x3136x192xf32) <- (-1x56x56x192xf32, 3xi64)
        t578 = paddle._C_ops.reshape(t574, t577)
        del t577

        # pd_op.add: (-1x3136x192xf32) <- (-1x3136x192xf32, -1x3136x192xf32)
        t579 = paddle._C_ops.add(t412, t578)

        # pd_op.layer_norm: (-1x3136x192xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x192xf32, 192xf32, 192xf32)
        t580, t581, t582 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t579, t10, t11, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t11, t10

        # pd_op.matmul: (-1x3136x768xf32) <- (-1x3136x192xf32, 192x768xf32)
        t583 = paddle._C_ops.matmul(t580, t12, False, False)
        del t12

        # pd_op.add: (-1x3136x768xf32) <- (-1x3136x768xf32, 768xf32)
        t584 = paddle._C_ops.add(t583, t13)
        del t13

        # pd_op.gelu: (-1x3136x768xf32) <- (-1x3136x768xf32)
        t585 = paddle._C_ops.gelu(t584, False)

        # pd_op.matmul: (-1x3136x192xf32) <- (-1x3136x768xf32, 768x192xf32)
        t586 = paddle._C_ops.matmul(t585, t14, False, False)
        del t14

        # pd_op.add: (-1x3136x192xf32) <- (-1x3136x192xf32, 192xf32)
        t587 = paddle._C_ops.add(t586, t15)
        del t15

        # pd_op.add: (-1x3136x192xf32) <- (-1x3136x192xf32, -1x3136x192xf32)
        t588 = paddle._C_ops.add(t579, t587)

        # pd_op.shape64: (3xi64) <- (-1x3136x192xf32)
        t589 = paddle._C_ops.shape64(t588)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t590 = paddle._C_ops.slice(t589, [0], t305, t354, [1], [0])
        del t589

        # pd_op.layer_norm: (-1x3136x192xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x192xf32, 192xf32, 192xf32)
        t591, t592, t593 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t588, t16, t17, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t17, t16

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t594 = [t590, t420, t420, t421]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t595 = paddle._C_ops.stack(t594, 0)
        del t594

        # pd_op.reshape: (-1x56x56x192xf32) <- (-1x3136x192xf32, 4xi64)
        t596 = paddle._C_ops.reshape(t591, t595)
        del t595

        # pd_op.shape64: (4xi64) <- (-1x56x56x192xf32)
        t597 = paddle._C_ops.shape64(t596)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t598 = paddle._C_ops.slice(t597, [0], t305, t354, [1], [0])
        del t597

        # pd_op.full_int_array: (2xi64) <- ()
        t599 = [-3, -3]

        # pd_op.assign: (2xi64) <- (2xi64)
        t600 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t601 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t602 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t603 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t604 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t605 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t606 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t607 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t608 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t609 = t599

        # pd_op.assign: (2xi64) <- (2xi64)
        t610 = t599

        # pd_op.roll: (-1x56x56x192xf32) <- (-1x56x56x192xf32, 2xi64)
        t611 = paddle._C_ops.roll(t596, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x56x56x192xf32)
        t612 = paddle._C_ops.shape64(t611)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t613 = paddle._C_ops.slice(t612, [0], t305, t354, [1], [0])
        del t612

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t614 = [t613, t427, t428, t427, t428, t421]
        del t427, t613

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t615 = paddle._C_ops.stack(t614, 0)
        del t614

        # pd_op.reshape: (-1x8x7x8x7x192xf32) <- (-1x56x56x192xf32, 6xi64)
        t616 = paddle._C_ops.reshape(t611, t615)
        del t615

        # pd_op.transpose: (-1x8x8x7x7x192xf32) <- (-1x8x7x8x7x192xf32)
        t617 = paddle._C_ops.transpose(t616, [0, 1, 3, 2, 4, 5])
        del t616

        # pd_op.reshape: (-1x7x7x192xf32) <- (-1x8x8x7x7x192xf32, 4xi64)
        t618 = paddle._C_ops.reshape(t617, t433)

        # pd_op.reshape: (-1x49x192xf32) <- (-1x7x7x192xf32, 3xi64)
        t619 = paddle._C_ops.reshape(t618, t435)
        del t435

        # pd_op.full: (1x56x56x1xf32) <- ()
        t620 = paddle._C_ops.full(
            [1, 56, 56, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (2xi64) <- ()
        t621 = [0, 0]

        # pd_op.assign: (2xi64) <- (2xi64)
        t622 = t621

        # pd_op.assign: (2xi64) <- (2xi64)
        t623 = t621

        # pd_op.assign: (2xi64) <- (2xi64)
        t624 = t621

        # pd_op.full_int_array: (2xi64) <- ()
        t625 = [-7, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        t626 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        t627 = t626

        # pd_op.assign: (2xi64) <- (2xi64)
        t628 = t626

        # pd_op.assign: (2xi64) <- (2xi64)
        t629 = t626

        # pd_op.assign: (2xi64) <- (2xi64)
        t630 = t626

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        t631 = paddle._C_ops.set_value_(
            t620, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t620

        # pd_op.full_int_array: (2xi64) <- ()
        t632 = [0, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        t633 = [-7, -3]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        t634 = paddle._C_ops.set_value_(
            t631, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t631

        # pd_op.full_int_array: (2xi64) <- ()
        t635 = [0, -3]

        # pd_op.full_int_array: (2xi64) <- ()
        t636 = [-7, 2147483647]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        t637 = paddle._C_ops.set_value_(
            t634, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t634

        # pd_op.full_int_array: (2xi64) <- ()
        t638 = [-7, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        t639 = [-3, -7]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        t640 = paddle._C_ops.set_value_(
            t637, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t637

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        t641 = paddle._C_ops.set_value_(
            t640, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t640

        # pd_op.full_int_array: (2xi64) <- ()
        t642 = [-3, 2147483647]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        t643 = paddle._C_ops.set_value_(
            t641, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t641

        # pd_op.full_int_array: (2xi64) <- ()
        t644 = [-3, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        t645 = [2147483647, -7]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        t646 = paddle._C_ops.set_value_(
            t643, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t643

        # pd_op.full_int_array: (2xi64) <- ()
        t647 = [2147483647, -3]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        t648 = paddle._C_ops.set_value_(
            t646, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t646

        # pd_op.full_int_array: (2xi64) <- ()
        t649 = [2147483647, 2147483647]

        # pd_op.assign: (2xi64) <- (2xi64)
        t650 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t651 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t652 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t653 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t654 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t655 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t656 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t657 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t658 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t659 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t660 = t649

        # pd_op.assign: (2xi64) <- (2xi64)
        t661 = t649

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        t662 = paddle._C_ops.set_value_(
            t648, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t648

        # pd_op.full_int_array: (6xi64) <- ()
        t663 = [1, 8, 7, 8, 7, 1]

        # pd_op.reshape: (1x8x7x8x7x1xf32) <- (1x56x56x1xf32, 6xi64)
        t664 = paddle._C_ops.reshape(t662, t663)
        del t663

        # pd_op.transpose: (1x8x8x7x7x1xf32) <- (1x8x7x8x7x1xf32)
        t665 = paddle._C_ops.transpose(t664, [0, 1, 3, 2, 4, 5])
        del t664

        # pd_op.full_int_array: (4xi64) <- ()
        t666 = [-1, 7, 7, 1]

        # pd_op.reshape: (64x7x7x1xf32) <- (1x8x8x7x7x1xf32, 4xi64)
        t667 = paddle._C_ops.reshape(t665, t666)
        del t665

        # pd_op.full_int_array: (2xi64) <- ()
        t668 = [-1, 49]

        # pd_op.reshape: (64x49xf32) <- (64x7x7x1xf32, 2xi64)
        t669 = paddle._C_ops.reshape(t667, t668)
        del t667

        # pd_op.unsqueeze: (64x1x49xf32) <- (64x49xf32, 1xi64)
        t670 = paddle._C_ops.unsqueeze(t669, t354)

        # pd_op.unsqueeze: (64x49x1xf32) <- (64x49xf32, 1xi64)
        t671 = paddle._C_ops.unsqueeze(t669, t450)
        del t669

        # pd_op.subtract: (64x49x49xf32) <- (64x1x49xf32, 64x49x1xf32)
        t672 = paddle._C_ops.subtract(t670, t671)
        del t670, t671

        # pd_op.full: (xf32) <- ()
        t673 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (64x49x49xb) <- (64x49x49xf32, xf32)
        t674 = paddle._C_ops.not_equal(t672, t673)

        # pd_op.full: (64x49x49xf32) <- ()
        t675 = paddle._C_ops.full(
            [64, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (64x49x49xf32) <- (64x49x49xb, 64x49x49xf32, 64x49x49xf32)
        t676 = paddle._C_ops.where(t674, t675, t672)
        del t675, t674, t672

        # pd_op.equal: (64x49x49xb) <- (64x49x49xf32, xf32)
        t677 = paddle._C_ops.equal(t676, t673)

        # pd_op.full: (64x49x49xf32) <- ()
        t678 = paddle._C_ops.full(
            [64, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (64x49x49xf32) <- (64x49x49xb, 64x49x49xf32, 64x49x49xf32)
        t679 = paddle._C_ops.where(t677, t678, t676)
        del t677, t678, t676

        # pd_op.shape64: (3xi64) <- (-1x49x192xf32)
        t680 = paddle._C_ops.shape64(t619)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t681 = paddle._C_ops.slice(t680, [0], t305, t354, [1], [0])
        del t680

        # pd_op.matmul: (-1x49x576xf32) <- (-1x49x192xf32, 192x576xf32)
        t682 = paddle._C_ops.matmul(t619, t18, False, False)
        del t18

        # pd_op.add: (-1x49x576xf32) <- (-1x49x576xf32, 576xf32)
        t683 = paddle._C_ops.add(t682, t19)
        del t19

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t684 = [t681, t441, t442, t443, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t685 = paddle._C_ops.stack(t684, 0)
        del t684

        # pd_op.reshape: (-1x49x3x6x32xf32) <- (-1x49x576xf32, 5xi64)
        t686 = paddle._C_ops.reshape(t683, t685)
        del t685

        # pd_op.transpose: (3x-1x6x49x32xf32) <- (-1x49x3x6x32xf32)
        t687 = paddle._C_ops.transpose(t686, [2, 0, 3, 1, 4])
        del t686

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        t688 = paddle._C_ops.slice(t687, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        t689 = paddle._C_ops.slice(t687, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        t690 = paddle._C_ops.slice(t687, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x6x49x32xf32) <- (-1x6x49x32xf32, 1xf32)
        t691 = paddle._C_ops.scale(t688, t526, float("0"), True)
        del t688

        # pd_op.transpose: (-1x6x32x49xf32) <- (-1x6x49x32xf32)
        t692 = paddle._C_ops.transpose(t689, [0, 1, 3, 2])
        del t689

        # pd_op.matmul: (-1x6x49x49xf32) <- (-1x6x49x32xf32, -1x6x32x49xf32)
        t693 = paddle._C_ops.matmul(t691, t692, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t694 = paddle._C_ops.reshape(input3, t553)
        del input3

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        t695 = paddle._C_ops.index_select(input4, t694, 0)
        del input4

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        t696 = paddle._C_ops.reshape(t695, t556)

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        t697 = paddle._C_ops.transpose(t696, [2, 0, 1])
        del t696

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        t698 = paddle._C_ops.unsqueeze(t697, t305)

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x49x49xf32, 1x6x49x49xf32)
        t699 = paddle._C_ops.add(t693, t698)

        # pd_op.full: (xi64) <- ()
        t700 = paddle._C_ops.full(
            [], float("64"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t701 = paddle._C_ops.floor_divide(t681, t700)
        del t700

        # pd_op.full: (xi64) <- ()
        t702 = paddle._C_ops.full([], float("64"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t703 = [t701, t702, t443, t441, t441]
        del t701, t702

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t704 = paddle._C_ops.stack(t703, 0)
        del t703

        # pd_op.reshape: (-1x64x6x49x49xf32) <- (-1x6x49x49xf32, 5xi64)
        t705 = paddle._C_ops.reshape(t699, t704)
        del t704

        # pd_op.unsqueeze: (64x1x49x49xf32) <- (64x49x49xf32, 1xi64)
        t706 = paddle._C_ops.unsqueeze(t679, t354)
        del t679

        # pd_op.unsqueeze: (1x64x1x49x49xf32) <- (64x1x49x49xf32, 1xi64)
        t707 = paddle._C_ops.unsqueeze(t706, t305)
        del t706

        # pd_op.add: (-1x64x6x49x49xf32) <- (-1x64x6x49x49xf32, 1x64x1x49x49xf32)
        t708 = paddle._C_ops.add(t705, t707)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t709 = [t681, t443, t441, t441]
        del t443

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t710 = paddle._C_ops.stack(t709, 0)
        del t709

        # pd_op.reshape: (-1x6x49x49xf32) <- (-1x64x6x49x49xf32, 4xi64)
        t711 = paddle._C_ops.reshape(t708, t710)
        del t710

        # pd_op.softmax: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        t712 = paddle._C_ops.softmax(t711, -1)
        del t711

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x49x32xf32)
        t713 = paddle._C_ops.matmul(t712, t690, False, False)

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        t714 = paddle._C_ops.transpose(t713, [0, 2, 1, 3])
        del t713

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t715 = [t681, t441, t421]
        del t681

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t716 = paddle._C_ops.stack(t715, 0)
        del t715

        # pd_op.reshape: (-1x49x192xf32) <- (-1x49x6x32xf32, 3xi64)
        t717 = paddle._C_ops.reshape(t714, t716)
        del t716

        # pd_op.matmul: (-1x49x192xf32) <- (-1x49x192xf32, 192x192xf32)
        t718 = paddle._C_ops.matmul(t717, t20, False, False)
        del t20

        # pd_op.add: (-1x49x192xf32) <- (-1x49x192xf32, 192xf32)
        t719 = paddle._C_ops.add(t718, t21)
        del t21

        # pd_op.reshape: (-1x7x7x192xf32) <- (-1x49x192xf32, 4xi64)
        t720 = paddle._C_ops.reshape(t719, t433)
        del t433

        # pd_op.reshape: (-1x8x8x7x7x192xf32) <- (-1x7x7x192xf32, 6xi64)
        t721 = paddle._C_ops.reshape(t720, t570)
        del t570

        # pd_op.transpose: (-1x8x7x8x7x192xf32) <- (-1x8x8x7x7x192xf32)
        t722 = paddle._C_ops.transpose(t721, [0, 1, 3, 2, 4, 5])
        del t721

        # pd_op.reshape: (-1x56x56x192xf32) <- (-1x8x7x8x7x192xf32, 4xi64)
        t723 = paddle._C_ops.reshape(t722, t573)
        del t573

        # pd_op.full_int_array: (2xi64) <- ()
        t724 = [3, 3]

        # pd_op.assign: (2xi64) <- (2xi64)
        t725 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t726 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t727 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t728 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t729 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t730 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t731 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t732 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t733 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t734 = t724

        # pd_op.assign: (2xi64) <- (2xi64)
        t735 = t724

        # pd_op.roll: (-1x56x56x192xf32) <- (-1x56x56x192xf32, 2xi64)
        t736 = paddle._C_ops.roll(t723, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t737 = [t590, t575, t421]
        del t575, t590

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t738 = paddle._C_ops.stack(t737, 0)
        del t737

        # pd_op.reshape: (-1x3136x192xf32) <- (-1x56x56x192xf32, 3xi64)
        t739 = paddle._C_ops.reshape(t736, t738)
        del t738

        # pd_op.full: (xf32) <- ()
        t740 = paddle._C_ops.full(
            [],
            float("0.995652"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t741 = t740

        # pd_op.shape64: (3xi64) <- (-1x3136x192xf32)
        t742 = paddle._C_ops.shape64(t739)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t743 = paddle._C_ops.slice(t742, [0], t305, t354, [1], [0])
        del t742

        # pd_op.full: (xi64) <- ()
        t744 = paddle._C_ops.full([], float("1"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t745 = [t743, t744, t744]
        del t743

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t746 = paddle._C_ops.stack(t745, 0)
        del t745

        # pd_op.full: (1xf32) <- ()
        t747 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t748 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t749 = paddle._C_ops.uniform(
            t746,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t746

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t750 = paddle._C_ops.add(t740, t749)
        del t749

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t751 = paddle._C_ops.floor(t750)
        del t750

        # pd_op.divide: (-1x3136x192xf32) <- (-1x3136x192xf32, xf32)
        t752 = paddle._C_ops.divide(t739, t740)

        # pd_op.multiply: (-1x3136x192xf32) <- (-1x3136x192xf32, -1x1x1xf32)
        t753 = paddle._C_ops.multiply(t752, t751)

        # pd_op.add: (-1x3136x192xf32) <- (-1x3136x192xf32, -1x3136x192xf32)
        t754 = paddle._C_ops.add(t588, t753)

        # pd_op.layer_norm: (-1x3136x192xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x192xf32, 192xf32, 192xf32)
        t755, t756, t757 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t754, t22, t23, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t23, t22

        # pd_op.matmul: (-1x3136x768xf32) <- (-1x3136x192xf32, 192x768xf32)
        t758 = paddle._C_ops.matmul(t755, t24, False, False)
        del t24

        # pd_op.add: (-1x3136x768xf32) <- (-1x3136x768xf32, 768xf32)
        t759 = paddle._C_ops.add(t758, t25)
        del t25

        # pd_op.gelu: (-1x3136x768xf32) <- (-1x3136x768xf32)
        t760 = paddle._C_ops.gelu(t759, False)

        # pd_op.matmul: (-1x3136x192xf32) <- (-1x3136x768xf32, 768x192xf32)
        t761 = paddle._C_ops.matmul(t760, t26, False, False)
        del t26

        # pd_op.add: (-1x3136x192xf32) <- (-1x3136x192xf32, 192xf32)
        t762 = paddle._C_ops.add(t761, t27)
        del t27

        # pd_op.shape64: (3xi64) <- (-1x3136x192xf32)
        t763 = paddle._C_ops.shape64(t762)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t764 = paddle._C_ops.slice(t763, [0], t305, t354, [1], [0])
        del t763

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t765 = [t764, t744, t744]
        del t764

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t766 = paddle._C_ops.stack(t765, 0)
        del t765

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t767 = paddle._C_ops.uniform(
            t766,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t766

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t768 = paddle._C_ops.add(t740, t767)
        del t767

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t769 = paddle._C_ops.floor(t768)
        del t768

        # pd_op.divide: (-1x3136x192xf32) <- (-1x3136x192xf32, xf32)
        t770 = paddle._C_ops.divide(t762, t740)

        # pd_op.multiply: (-1x3136x192xf32) <- (-1x3136x192xf32, -1x1x1xf32)
        t771 = paddle._C_ops.multiply(t770, t769)

        # pd_op.add: (-1x3136x192xf32) <- (-1x3136x192xf32, -1x3136x192xf32)
        t772 = paddle._C_ops.add(t754, t771)

        # pd_op.shape64: (3xi64) <- (-1x3136x192xf32)
        t773 = paddle._C_ops.shape64(t772)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t774 = paddle._C_ops.slice(t773, [0], t305, t354, [1], [0])
        del t773

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t775 = [t774, t420, t420, t421]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t776 = paddle._C_ops.stack(t775, 0)
        del t775

        # pd_op.reshape: (-1x56x56x192xf32) <- (-1x3136x192xf32, 4xi64)
        t777 = paddle._C_ops.reshape(t772, t776)
        del t776

        # pd_op.full_int_array: (2xi64) <- ()
        t778 = [2, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        t779 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t780 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t781 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t782 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t783 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t784 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t785 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t786 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t787 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t788 = t778

        # pd_op.assign: (2xi64) <- (2xi64)
        t789 = t778

        # pd_op.strided_slice: (-1x28x28x192xf32) <- (-1x56x56x192xf32, 2xi64, 2xi64, 2xi64)
        t790 = paddle._C_ops.strided_slice(t777, [1, 2], t621, t649, t778)

        # pd_op.full_int_array: (2xi64) <- ()
        t791 = [1, 0]

        # pd_op.assign: (2xi64) <- (2xi64)
        t792 = t791

        # pd_op.assign: (2xi64) <- (2xi64)
        t793 = t791

        # pd_op.strided_slice: (-1x28x28x192xf32) <- (-1x56x56x192xf32, 2xi64, 2xi64, 2xi64)
        t794 = paddle._C_ops.strided_slice(t777, [1, 2], t791, t649, t778)

        # pd_op.full_int_array: (2xi64) <- ()
        t795 = [0, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        t796 = t795

        # pd_op.assign: (2xi64) <- (2xi64)
        t797 = t795

        # pd_op.strided_slice: (-1x28x28x192xf32) <- (-1x56x56x192xf32, 2xi64, 2xi64, 2xi64)
        t798 = paddle._C_ops.strided_slice(t777, [1, 2], t795, t649, t778)

        # pd_op.strided_slice: (-1x28x28x192xf32) <- (-1x56x56x192xf32, 2xi64, 2xi64, 2xi64)
        t799 = paddle._C_ops.strided_slice(t777, [1, 2], t626, t649, t778)

        # pd_op.shape64: (4xi64) <- (-1x56x56x192xf32)
        t800 = paddle._C_ops.shape64(t777)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t801 = paddle._C_ops.slice(t800, [0], t305, t354, [1], [0])
        del t800

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t802 = [t801, t420, t420, t421]
        del t420, t421, t801

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t803 = paddle._C_ops.stack(t802, 0)
        del t802

        # pd_op.reshape: (-1x56x56x192xf32) <- (-1x56x56x192xf32, 4xi64)
        t804 = paddle._C_ops.reshape(t777, t803)
        del t803

        # pd_op.full: (1xi32) <- ()
        t805 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        t806 = t805

        # pd_op.assign: (1xi32) <- (1xi32)
        t807 = t805

        # builtin.combine: ([-1x28x28x192xf32, -1x28x28x192xf32, -1x28x28x192xf32, -1x28x28x192xf32]) <- (-1x28x28x192xf32, -1x28x28x192xf32, -1x28x28x192xf32, -1x28x28x192xf32)
        t808 = [t790, t794, t798, t799]

        # pd_op.concat: (-1x28x28x768xf32) <- ([-1x28x28x192xf32, -1x28x28x192xf32, -1x28x28x192xf32, -1x28x28x192xf32], 1xi32)
        t809 = paddle._C_ops.concat(t808, t805)
        del t808

        # pd_op.full: (xi64) <- ()
        t810 = paddle._C_ops.full([], float("-1"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t811 = paddle._C_ops.full(
            [], float("768"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t812 = [t774, t810, t811]
        del t774

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t813 = paddle._C_ops.stack(t812, 0)
        del t812

        # pd_op.reshape: (-1x-1x768xf32) <- (-1x28x28x768xf32, 3xi64)
        t814 = paddle._C_ops.reshape(t809, t813)
        del t813

        # pd_op.layer_norm: (-1x-1x768xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x768xf32, 768xf32, 768xf32)
        t815, t816, t817 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t814, t28, t29, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t29, t28

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        t818 = paddle._C_ops.matmul(t815, t30, False, False)
        del t30

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        t819 = paddle._C_ops.shape64(t818)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t820 = paddle._C_ops.slice(t819, [0], t305, t354, [1], [0])
        del t819

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        t821 = paddle._C_ops.shape64(t818)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t822 = paddle._C_ops.slice(t821, [0], t354, t450, [1], [0])
        del t821

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        t823, t824, t825 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t818, t31, t32, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t32, t31

        # pd_op.full: (xi64) <- ()
        t826 = paddle._C_ops.full([], float("28"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t827 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t828 = [t820, t826, t826, t827]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t829 = paddle._C_ops.stack(t828, 0)
        del t828

        # pd_op.reshape: (-1x28x28x384xf32) <- (-1x-1x384xf32, 4xi64)
        t830 = paddle._C_ops.reshape(t823, t829)
        del t829

        # pd_op.shape64: (4xi64) <- (-1x28x28x384xf32)
        t831 = paddle._C_ops.shape64(t830)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t832 = paddle._C_ops.slice(t831, [0], t305, t354, [1], [0])
        del t831

        # pd_op.full: (xi64) <- ()
        t833 = paddle._C_ops.full([], float("4"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t834 = [t832, t833, t428, t833, t428, t827]
        del t832

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t835 = paddle._C_ops.stack(t834, 0)
        del t834

        # pd_op.reshape: (-1x4x7x4x7x384xf32) <- (-1x28x28x384xf32, 6xi64)
        t836 = paddle._C_ops.reshape(t830, t835)
        del t835

        # pd_op.transpose: (-1x4x4x7x7x384xf32) <- (-1x4x7x4x7x384xf32)
        t837 = paddle._C_ops.transpose(t836, [0, 1, 3, 2, 4, 5])
        del t836

        # pd_op.full_int_array: (4xi64) <- ()
        t838 = [-1, 7, 7, 384]

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x4x4x7x7x384xf32, 4xi64)
        t839 = paddle._C_ops.reshape(t837, t838)

        # pd_op.full_int_array: (3xi64) <- ()
        t840 = [-1, 49, 384]

        # pd_op.reshape: (-1x49x384xf32) <- (-1x7x7x384xf32, 3xi64)
        t841 = paddle._C_ops.reshape(t839, t840)

        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        t842 = paddle._C_ops.shape64(t841)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t843 = paddle._C_ops.slice(t842, [0], t305, t354, [1], [0])
        del t842

        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        t844 = paddle._C_ops.matmul(t841, t33, False, False)
        del t33

        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        t845 = paddle._C_ops.add(t844, t34)
        del t34

        # pd_op.full: (xi64) <- ()
        t846 = paddle._C_ops.full([], float("12"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t847 = [t843, t441, t442, t846, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t848 = paddle._C_ops.stack(t847, 0)
        del t847

        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        t849 = paddle._C_ops.reshape(t845, t848)
        del t848

        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        t850 = paddle._C_ops.transpose(t849, [2, 0, 3, 1, 4])
        del t849

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        t851 = paddle._C_ops.slice(t850, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        t852 = paddle._C_ops.slice(t850, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        t853 = paddle._C_ops.slice(t850, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        t854 = paddle._C_ops.scale(t851, t526, float("0"), True)
        del t851

        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        t855 = paddle._C_ops.transpose(t852, [0, 1, 3, 2])
        del t852

        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        t856 = paddle._C_ops.matmul(t854, t855, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t857 = paddle._C_ops.reshape(input5, t553)
        del input5

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        t858 = paddle._C_ops.index_select(input6, t857, 0)
        del input6

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        t859 = paddle._C_ops.reshape(t858, t556)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        t860 = paddle._C_ops.transpose(t859, [2, 0, 1])
        del t859

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        t861 = paddle._C_ops.unsqueeze(t860, t305)

        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x12x49x49xf32)
        t862 = paddle._C_ops.add(t856, t861)

        # pd_op.softmax: (-1x12x49x49xf32) <- (-1x12x49x49xf32)
        t863 = paddle._C_ops.softmax(t862, -1)
        del t862

        # pd_op.matmul: (-1x12x49x32xf32) <- (-1x12x49x49xf32, -1x12x49x32xf32)
        t864 = paddle._C_ops.matmul(t863, t853, False, False)

        # pd_op.transpose: (-1x49x12x32xf32) <- (-1x12x49x32xf32)
        t865 = paddle._C_ops.transpose(t864, [0, 2, 1, 3])
        del t864

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t866 = [t843, t441, t827]
        del t843

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t867 = paddle._C_ops.stack(t866, 0)
        del t866

        # pd_op.reshape: (-1x49x384xf32) <- (-1x49x12x32xf32, 3xi64)
        t868 = paddle._C_ops.reshape(t865, t867)
        del t867

        # pd_op.matmul: (-1x49x384xf32) <- (-1x49x384xf32, 384x384xf32)
        t869 = paddle._C_ops.matmul(t868, t35, False, False)
        del t35

        # pd_op.add: (-1x49x384xf32) <- (-1x49x384xf32, 384xf32)
        t870 = paddle._C_ops.add(t869, t36)
        del t36

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x49x384xf32, 4xi64)
        t871 = paddle._C_ops.reshape(t870, t838)

        # pd_op.full_int_array: (6xi64) <- ()
        t872 = [-1, 4, 4, 7, 7, 384]

        # pd_op.reshape: (-1x4x4x7x7x384xf32) <- (-1x7x7x384xf32, 6xi64)
        t873 = paddle._C_ops.reshape(t871, t872)

        # pd_op.transpose: (-1x4x7x4x7x384xf32) <- (-1x4x4x7x7x384xf32)
        t874 = paddle._C_ops.transpose(t873, [0, 1, 3, 2, 4, 5])
        del t873

        # pd_op.full_int_array: (4xi64) <- ()
        t875 = [-1, 28, 28, 384]

        # pd_op.reshape: (-1x28x28x384xf32) <- (-1x4x7x4x7x384xf32, 4xi64)
        t876 = paddle._C_ops.reshape(t874, t875)

        # pd_op.full: (xi64) <- ()
        t877 = paddle._C_ops.full(
            [], float("784"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t878 = [t820, t877, t827]
        del t820

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t879 = paddle._C_ops.stack(t878, 0)
        del t878

        # pd_op.reshape: (-1x784x384xf32) <- (-1x28x28x384xf32, 3xi64)
        t880 = paddle._C_ops.reshape(t876, t879)
        del t879

        # pd_op.full: (xf32) <- ()
        t881 = paddle._C_ops.full(
            [],
            float("0.991304"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t882 = t881

        # pd_op.shape64: (3xi64) <- (-1x784x384xf32)
        t883 = paddle._C_ops.shape64(t880)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t884 = paddle._C_ops.slice(t883, [0], t305, t354, [1], [0])
        del t883

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t885 = [t884, t744, t744]
        del t884

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t886 = paddle._C_ops.stack(t885, 0)
        del t885

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t887 = paddle._C_ops.uniform(
            t886,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t886

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t888 = paddle._C_ops.add(t881, t887)
        del t887

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t889 = paddle._C_ops.floor(t888)
        del t888

        # pd_op.divide: (-1x784x384xf32) <- (-1x784x384xf32, xf32)
        t890 = paddle._C_ops.divide(t880, t881)

        # pd_op.multiply: (-1x784x384xf32) <- (-1x784x384xf32, -1x1x1xf32)
        t891 = paddle._C_ops.multiply(t890, t889)

        # pd_op.add: (-1x784x384xf32) <- (-1x-1x384xf32, -1x784x384xf32)
        t892 = paddle._C_ops.add(t818, t891)

        # pd_op.layer_norm: (-1x784x384xf32, -1x784xf32, -1x784xf32) <- (-1x784x384xf32, 384xf32, 384xf32)
        t893, t894, t895 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t892, t37, t38, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t38, t37

        # pd_op.matmul: (-1x784x1536xf32) <- (-1x784x384xf32, 384x1536xf32)
        t896 = paddle._C_ops.matmul(t893, t39, False, False)
        del t39

        # pd_op.add: (-1x784x1536xf32) <- (-1x784x1536xf32, 1536xf32)
        t897 = paddle._C_ops.add(t896, t40)
        del t40

        # pd_op.gelu: (-1x784x1536xf32) <- (-1x784x1536xf32)
        t898 = paddle._C_ops.gelu(t897, False)

        # pd_op.matmul: (-1x784x384xf32) <- (-1x784x1536xf32, 1536x384xf32)
        t899 = paddle._C_ops.matmul(t898, t41, False, False)
        del t41

        # pd_op.add: (-1x784x384xf32) <- (-1x784x384xf32, 384xf32)
        t900 = paddle._C_ops.add(t899, t42)
        del t42

        # pd_op.shape64: (3xi64) <- (-1x784x384xf32)
        t901 = paddle._C_ops.shape64(t900)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t902 = paddle._C_ops.slice(t901, [0], t305, t354, [1], [0])
        del t901

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t903 = [t902, t744, t744]
        del t902

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t904 = paddle._C_ops.stack(t903, 0)
        del t903

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t905 = paddle._C_ops.uniform(
            t904,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t904

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t906 = paddle._C_ops.add(t881, t905)
        del t905

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t907 = paddle._C_ops.floor(t906)
        del t906

        # pd_op.divide: (-1x784x384xf32) <- (-1x784x384xf32, xf32)
        t908 = paddle._C_ops.divide(t900, t881)

        # pd_op.multiply: (-1x784x384xf32) <- (-1x784x384xf32, -1x1x1xf32)
        t909 = paddle._C_ops.multiply(t908, t907)

        # pd_op.add: (-1x784x384xf32) <- (-1x784x384xf32, -1x784x384xf32)
        t910 = paddle._C_ops.add(t892, t909)

        # pd_op.shape64: (3xi64) <- (-1x784x384xf32)
        t911 = paddle._C_ops.shape64(t910)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t912 = paddle._C_ops.slice(t911, [0], t305, t354, [1], [0])
        del t911

        # pd_op.layer_norm: (-1x784x384xf32, -1x784xf32, -1x784xf32) <- (-1x784x384xf32, 384xf32, 384xf32)
        t913, t914, t915 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t910, t43, t44, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t44, t43

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t916 = [t912, t826, t826, t827]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t917 = paddle._C_ops.stack(t916, 0)
        del t916

        # pd_op.reshape: (-1x28x28x384xf32) <- (-1x784x384xf32, 4xi64)
        t918 = paddle._C_ops.reshape(t913, t917)
        del t917

        # pd_op.shape64: (4xi64) <- (-1x28x28x384xf32)
        t919 = paddle._C_ops.shape64(t918)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t920 = paddle._C_ops.slice(t919, [0], t305, t354, [1], [0])
        del t919

        # pd_op.roll: (-1x28x28x384xf32) <- (-1x28x28x384xf32, 2xi64)
        t921 = paddle._C_ops.roll(t918, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x28x28x384xf32)
        t922 = paddle._C_ops.shape64(t921)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t923 = paddle._C_ops.slice(t922, [0], t305, t354, [1], [0])
        del t922

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t924 = [t923, t833, t428, t833, t428, t827]
        del t923

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t925 = paddle._C_ops.stack(t924, 0)
        del t924

        # pd_op.reshape: (-1x4x7x4x7x384xf32) <- (-1x28x28x384xf32, 6xi64)
        t926 = paddle._C_ops.reshape(t921, t925)
        del t925

        # pd_op.transpose: (-1x4x4x7x7x384xf32) <- (-1x4x7x4x7x384xf32)
        t927 = paddle._C_ops.transpose(t926, [0, 1, 3, 2, 4, 5])
        del t926

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x4x4x7x7x384xf32, 4xi64)
        t928 = paddle._C_ops.reshape(t927, t838)

        # pd_op.reshape: (-1x49x384xf32) <- (-1x7x7x384xf32, 3xi64)
        t929 = paddle._C_ops.reshape(t928, t840)
        del t840

        # pd_op.full: (1x28x28x1xf32) <- ()
        t930 = paddle._C_ops.full(
            [1, 28, 28, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        t931 = paddle._C_ops.set_value_(
            t930, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t930

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        t932 = paddle._C_ops.set_value_(
            t931, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t931

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        t933 = paddle._C_ops.set_value_(
            t932, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t932

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        t934 = paddle._C_ops.set_value_(
            t933, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t933

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        t935 = paddle._C_ops.set_value_(
            t934, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t934

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        t936 = paddle._C_ops.set_value_(
            t935, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t935

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        t937 = paddle._C_ops.set_value_(
            t936, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t936

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        t938 = paddle._C_ops.set_value_(
            t937, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t937

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        t939 = paddle._C_ops.set_value_(
            t938, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t938

        # pd_op.full_int_array: (6xi64) <- ()
        t940 = [1, 4, 7, 4, 7, 1]

        # pd_op.reshape: (1x4x7x4x7x1xf32) <- (1x28x28x1xf32, 6xi64)
        t941 = paddle._C_ops.reshape(t939, t940)
        del t940

        # pd_op.transpose: (1x4x4x7x7x1xf32) <- (1x4x7x4x7x1xf32)
        t942 = paddle._C_ops.transpose(t941, [0, 1, 3, 2, 4, 5])
        del t941

        # pd_op.reshape: (16x7x7x1xf32) <- (1x4x4x7x7x1xf32, 4xi64)
        t943 = paddle._C_ops.reshape(t942, t666)
        del t942

        # pd_op.reshape: (16x49xf32) <- (16x7x7x1xf32, 2xi64)
        t944 = paddle._C_ops.reshape(t943, t668)
        del t943

        # pd_op.unsqueeze: (16x1x49xf32) <- (16x49xf32, 1xi64)
        t945 = paddle._C_ops.unsqueeze(t944, t354)

        # pd_op.unsqueeze: (16x49x1xf32) <- (16x49xf32, 1xi64)
        t946 = paddle._C_ops.unsqueeze(t944, t450)
        del t944

        # pd_op.subtract: (16x49x49xf32) <- (16x1x49xf32, 16x49x1xf32)
        t947 = paddle._C_ops.subtract(t945, t946)
        del t945, t946

        # pd_op.not_equal: (16x49x49xb) <- (16x49x49xf32, xf32)
        t948 = paddle._C_ops.not_equal(t947, t673)

        # pd_op.full: (16x49x49xf32) <- ()
        t949 = paddle._C_ops.full(
            [16, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x49x49xf32) <- (16x49x49xb, 16x49x49xf32, 16x49x49xf32)
        t950 = paddle._C_ops.where(t948, t949, t947)
        del t949, t948, t947

        # pd_op.equal: (16x49x49xb) <- (16x49x49xf32, xf32)
        t951 = paddle._C_ops.equal(t950, t673)

        # pd_op.full: (16x49x49xf32) <- ()
        t952 = paddle._C_ops.full(
            [16, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x49x49xf32) <- (16x49x49xb, 16x49x49xf32, 16x49x49xf32)
        t953 = paddle._C_ops.where(t951, t952, t950)
        del t951, t952, t950

        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        t954 = paddle._C_ops.shape64(t929)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t955 = paddle._C_ops.slice(t954, [0], t305, t354, [1], [0])
        del t954

        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        t956 = paddle._C_ops.matmul(t929, t45, False, False)
        del t45

        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        t957 = paddle._C_ops.add(t956, t46)
        del t46

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t958 = [t955, t441, t442, t846, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t959 = paddle._C_ops.stack(t958, 0)
        del t958

        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        t960 = paddle._C_ops.reshape(t957, t959)
        del t959

        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        t961 = paddle._C_ops.transpose(t960, [2, 0, 3, 1, 4])
        del t960

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        t962 = paddle._C_ops.slice(t961, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        t963 = paddle._C_ops.slice(t961, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        t964 = paddle._C_ops.slice(t961, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        t965 = paddle._C_ops.scale(t962, t526, float("0"), True)
        del t962

        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        t966 = paddle._C_ops.transpose(t963, [0, 1, 3, 2])
        del t963

        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        t967 = paddle._C_ops.matmul(t965, t966, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t968 = paddle._C_ops.reshape(input7, t553)
        del input7

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        t969 = paddle._C_ops.index_select(input8, t968, 0)
        del input8

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        t970 = paddle._C_ops.reshape(t969, t556)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        t971 = paddle._C_ops.transpose(t970, [2, 0, 1])
        del t970

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        t972 = paddle._C_ops.unsqueeze(t971, t305)

        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x12x49x49xf32)
        t973 = paddle._C_ops.add(t967, t972)

        # pd_op.full: (xi64) <- ()
        t974 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t975 = paddle._C_ops.floor_divide(t955, t974)
        del t974

        # pd_op.full: (xi64) <- ()
        t976 = paddle._C_ops.full([], float("16"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t977 = [t975, t976, t846, t441, t441]
        del t975, t976

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t978 = paddle._C_ops.stack(t977, 0)
        del t977

        # pd_op.reshape: (-1x16x12x49x49xf32) <- (-1x12x49x49xf32, 5xi64)
        t979 = paddle._C_ops.reshape(t973, t978)
        del t978

        # pd_op.unsqueeze: (16x1x49x49xf32) <- (16x49x49xf32, 1xi64)
        t980 = paddle._C_ops.unsqueeze(t953, t354)
        del t953

        # pd_op.unsqueeze: (1x16x1x49x49xf32) <- (16x1x49x49xf32, 1xi64)
        t981 = paddle._C_ops.unsqueeze(t980, t305)
        del t980

        # pd_op.add: (-1x16x12x49x49xf32) <- (-1x16x12x49x49xf32, 1x16x1x49x49xf32)
        t982 = paddle._C_ops.add(t979, t981)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t983 = [t955, t846, t441, t441]
        del t846

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t984 = paddle._C_ops.stack(t983, 0)
        del t983

        # pd_op.reshape: (-1x12x49x49xf32) <- (-1x16x12x49x49xf32, 4xi64)
        t985 = paddle._C_ops.reshape(t982, t984)
        del t984

        # pd_op.softmax: (-1x12x49x49xf32) <- (-1x12x49x49xf32)
        t986 = paddle._C_ops.softmax(t985, -1)
        del t985

        # pd_op.matmul: (-1x12x49x32xf32) <- (-1x12x49x49xf32, -1x12x49x32xf32)
        t987 = paddle._C_ops.matmul(t986, t964, False, False)

        # pd_op.transpose: (-1x49x12x32xf32) <- (-1x12x49x32xf32)
        t988 = paddle._C_ops.transpose(t987, [0, 2, 1, 3])
        del t987

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t989 = [t955, t441, t827]
        del t955

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t990 = paddle._C_ops.stack(t989, 0)
        del t989

        # pd_op.reshape: (-1x49x384xf32) <- (-1x49x12x32xf32, 3xi64)
        t991 = paddle._C_ops.reshape(t988, t990)
        del t990

        # pd_op.matmul: (-1x49x384xf32) <- (-1x49x384xf32, 384x384xf32)
        t992 = paddle._C_ops.matmul(t991, t47, False, False)
        del t47

        # pd_op.add: (-1x49x384xf32) <- (-1x49x384xf32, 384xf32)
        t993 = paddle._C_ops.add(t992, t48)
        del t48

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x49x384xf32, 4xi64)
        t994 = paddle._C_ops.reshape(t993, t838)
        del t838

        # pd_op.reshape: (-1x4x4x7x7x384xf32) <- (-1x7x7x384xf32, 6xi64)
        t995 = paddle._C_ops.reshape(t994, t872)
        del t872

        # pd_op.transpose: (-1x4x7x4x7x384xf32) <- (-1x4x4x7x7x384xf32)
        t996 = paddle._C_ops.transpose(t995, [0, 1, 3, 2, 4, 5])
        del t995

        # pd_op.reshape: (-1x28x28x384xf32) <- (-1x4x7x4x7x384xf32, 4xi64)
        t997 = paddle._C_ops.reshape(t996, t875)
        del t875

        # pd_op.roll: (-1x28x28x384xf32) <- (-1x28x28x384xf32, 2xi64)
        t998 = paddle._C_ops.roll(t997, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t999 = [t912, t877, t827]
        del t877, t912

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1000 = paddle._C_ops.stack(t999, 0)
        del t999

        # pd_op.reshape: (-1x784x384xf32) <- (-1x28x28x384xf32, 3xi64)
        t1001 = paddle._C_ops.reshape(t998, t1000)
        del t1000

        # pd_op.full: (xf32) <- ()
        t1002 = paddle._C_ops.full(
            [],
            float("0.986957"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1003 = t1002

        # pd_op.shape64: (3xi64) <- (-1x784x384xf32)
        t1004 = paddle._C_ops.shape64(t1001)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1005 = paddle._C_ops.slice(t1004, [0], t305, t354, [1], [0])
        del t1004

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1006 = [t1005, t744, t744]
        del t1005

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1007 = paddle._C_ops.stack(t1006, 0)
        del t1006

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1008 = paddle._C_ops.uniform(
            t1007,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1007

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1009 = paddle._C_ops.add(t1002, t1008)
        del t1008

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1010 = paddle._C_ops.floor(t1009)
        del t1009

        # pd_op.divide: (-1x784x384xf32) <- (-1x784x384xf32, xf32)
        t1011 = paddle._C_ops.divide(t1001, t1002)

        # pd_op.multiply: (-1x784x384xf32) <- (-1x784x384xf32, -1x1x1xf32)
        t1012 = paddle._C_ops.multiply(t1011, t1010)

        # pd_op.add: (-1x784x384xf32) <- (-1x784x384xf32, -1x784x384xf32)
        t1013 = paddle._C_ops.add(t910, t1012)

        # pd_op.layer_norm: (-1x784x384xf32, -1x784xf32, -1x784xf32) <- (-1x784x384xf32, 384xf32, 384xf32)
        t1014, t1015, t1016 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1013, t49, t50, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t50, t49

        # pd_op.matmul: (-1x784x1536xf32) <- (-1x784x384xf32, 384x1536xf32)
        t1017 = paddle._C_ops.matmul(t1014, t51, False, False)
        del t51

        # pd_op.add: (-1x784x1536xf32) <- (-1x784x1536xf32, 1536xf32)
        t1018 = paddle._C_ops.add(t1017, t52)
        del t52

        # pd_op.gelu: (-1x784x1536xf32) <- (-1x784x1536xf32)
        t1019 = paddle._C_ops.gelu(t1018, False)

        # pd_op.matmul: (-1x784x384xf32) <- (-1x784x1536xf32, 1536x384xf32)
        t1020 = paddle._C_ops.matmul(t1019, t53, False, False)
        del t53

        # pd_op.add: (-1x784x384xf32) <- (-1x784x384xf32, 384xf32)
        t1021 = paddle._C_ops.add(t1020, t54)
        del t54

        # pd_op.shape64: (3xi64) <- (-1x784x384xf32)
        t1022 = paddle._C_ops.shape64(t1021)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1023 = paddle._C_ops.slice(t1022, [0], t305, t354, [1], [0])
        del t1022

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1024 = [t1023, t744, t744]
        del t1023

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1025 = paddle._C_ops.stack(t1024, 0)
        del t1024

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1026 = paddle._C_ops.uniform(
            t1025,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1025

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1027 = paddle._C_ops.add(t1002, t1026)
        del t1026

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1028 = paddle._C_ops.floor(t1027)
        del t1027

        # pd_op.divide: (-1x784x384xf32) <- (-1x784x384xf32, xf32)
        t1029 = paddle._C_ops.divide(t1021, t1002)

        # pd_op.multiply: (-1x784x384xf32) <- (-1x784x384xf32, -1x1x1xf32)
        t1030 = paddle._C_ops.multiply(t1029, t1028)

        # pd_op.add: (-1x784x384xf32) <- (-1x784x384xf32, -1x784x384xf32)
        t1031 = paddle._C_ops.add(t1013, t1030)

        # pd_op.shape64: (3xi64) <- (-1x784x384xf32)
        t1032 = paddle._C_ops.shape64(t1031)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1033 = paddle._C_ops.slice(t1032, [0], t305, t354, [1], [0])
        del t1032

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1034 = [t1033, t826, t826, t827]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1035 = paddle._C_ops.stack(t1034, 0)
        del t1034

        # pd_op.reshape: (-1x28x28x384xf32) <- (-1x784x384xf32, 4xi64)
        t1036 = paddle._C_ops.reshape(t1031, t1035)
        del t1035

        # pd_op.strided_slice: (-1x14x14x384xf32) <- (-1x28x28x384xf32, 2xi64, 2xi64, 2xi64)
        t1037 = paddle._C_ops.strided_slice(t1036, [1, 2], t621, t649, t778)

        # pd_op.strided_slice: (-1x14x14x384xf32) <- (-1x28x28x384xf32, 2xi64, 2xi64, 2xi64)
        t1038 = paddle._C_ops.strided_slice(t1036, [1, 2], t791, t649, t778)

        # pd_op.strided_slice: (-1x14x14x384xf32) <- (-1x28x28x384xf32, 2xi64, 2xi64, 2xi64)
        t1039 = paddle._C_ops.strided_slice(t1036, [1, 2], t795, t649, t778)

        # pd_op.strided_slice: (-1x14x14x384xf32) <- (-1x28x28x384xf32, 2xi64, 2xi64, 2xi64)
        t1040 = paddle._C_ops.strided_slice(t1036, [1, 2], t626, t649, t778)

        # pd_op.shape64: (4xi64) <- (-1x28x28x384xf32)
        t1041 = paddle._C_ops.shape64(t1036)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1042 = paddle._C_ops.slice(t1041, [0], t305, t354, [1], [0])
        del t1041

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1043 = [t1042, t826, t826, t827]
        del t826, t827, t1042

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1044 = paddle._C_ops.stack(t1043, 0)
        del t1043

        # pd_op.reshape: (-1x28x28x384xf32) <- (-1x28x28x384xf32, 4xi64)
        t1045 = paddle._C_ops.reshape(t1036, t1044)
        del t1044

        # builtin.combine: ([-1x14x14x384xf32, -1x14x14x384xf32, -1x14x14x384xf32, -1x14x14x384xf32]) <- (-1x14x14x384xf32, -1x14x14x384xf32, -1x14x14x384xf32, -1x14x14x384xf32)
        t1046 = [t1037, t1038, t1039, t1040]

        # pd_op.concat: (-1x14x14x1536xf32) <- ([-1x14x14x384xf32, -1x14x14x384xf32, -1x14x14x384xf32, -1x14x14x384xf32], 1xi32)
        t1047 = paddle._C_ops.concat(t1046, t805)
        del t1046

        # pd_op.full: (xi64) <- ()
        t1048 = paddle._C_ops.full(
            [], float("1536"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1049 = [t1033, t810, t1048]
        del t1033

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1050 = paddle._C_ops.stack(t1049, 0)
        del t1049

        # pd_op.reshape: (-1x-1x1536xf32) <- (-1x14x14x1536xf32, 3xi64)
        t1051 = paddle._C_ops.reshape(t1047, t1050)
        del t1050

        # pd_op.layer_norm: (-1x-1x1536xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x1536xf32, 1536xf32, 1536xf32)
        t1052, t1053, t1054 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1051, t55, t56, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t56, t55

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x1536xf32, 1536x768xf32)
        t1055 = paddle._C_ops.matmul(t1052, t57, False, False)
        del t57

        # pd_op.shape64: (3xi64) <- (-1x-1x768xf32)
        t1056 = paddle._C_ops.shape64(t1055)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1057 = paddle._C_ops.slice(t1056, [0], t305, t354, [1], [0])
        del t1056

        # pd_op.shape64: (3xi64) <- (-1x-1x768xf32)
        t1058 = paddle._C_ops.shape64(t1055)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1059 = paddle._C_ops.slice(t1058, [0], t354, t450, [1], [0])
        del t1058

        # pd_op.layer_norm: (-1x-1x768xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x768xf32, 768xf32, 768xf32)
        t1060, t1061, t1062 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1055, t58, t59, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t59, t58

        # pd_op.full: (xi64) <- ()
        t1063 = paddle._C_ops.full(
            [], float("14"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1064 = [t1057, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1065 = paddle._C_ops.stack(t1064, 0)
        del t1064

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x-1x768xf32, 4xi64)
        t1066 = paddle._C_ops.reshape(t1060, t1065)
        del t1065

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1067 = paddle._C_ops.shape64(t1066)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1068 = paddle._C_ops.slice(t1067, [0], t305, t354, [1], [0])
        del t1067

        # pd_op.full: (xi64) <- ()
        t1069 = paddle._C_ops.full([], float("2"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1070 = [t1068, t1069, t428, t1069, t428, t811]
        del t1068

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1071 = paddle._C_ops.stack(t1070, 0)
        del t1070

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1072 = paddle._C_ops.reshape(t1066, t1071)
        del t1071

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1073 = paddle._C_ops.transpose(t1072, [0, 1, 3, 2, 4, 5])
        del t1072

        # pd_op.full_int_array: (4xi64) <- ()
        t1074 = [-1, 7, 7, 768]

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1075 = paddle._C_ops.reshape(t1073, t1074)

        # pd_op.full_int_array: (3xi64) <- ()
        t1076 = [-1, 49, 768]

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1077 = paddle._C_ops.reshape(t1075, t1076)

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1078 = paddle._C_ops.shape64(t1077)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1079 = paddle._C_ops.slice(t1078, [0], t305, t354, [1], [0])
        del t1078

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1080 = paddle._C_ops.matmul(t1077, t60, False, False)
        del t60

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1081 = paddle._C_ops.add(t1080, t61)
        del t61

        # pd_op.full: (xi64) <- ()
        t1082 = paddle._C_ops.full(
            [], float("24"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1083 = [t1079, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1084 = paddle._C_ops.stack(t1083, 0)
        del t1083

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1085 = paddle._C_ops.reshape(t1081, t1084)
        del t1084

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1086 = paddle._C_ops.transpose(t1085, [2, 0, 3, 1, 4])
        del t1085

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1087 = paddle._C_ops.slice(t1086, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1088 = paddle._C_ops.slice(t1086, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1089 = paddle._C_ops.slice(t1086, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1090 = paddle._C_ops.scale(t1087, t526, float("0"), True)
        del t1087

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1091 = paddle._C_ops.transpose(t1088, [0, 1, 3, 2])
        del t1088

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1092 = paddle._C_ops.matmul(t1090, t1091, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1093 = paddle._C_ops.reshape(input9, t553)
        del input9

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1094 = paddle._C_ops.index_select(input10, t1093, 0)
        del input10

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1095 = paddle._C_ops.reshape(t1094, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1096 = paddle._C_ops.transpose(t1095, [2, 0, 1])
        del t1095

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1097 = paddle._C_ops.unsqueeze(t1096, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1098 = paddle._C_ops.add(t1092, t1097)

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t1099 = paddle._C_ops.softmax(t1098, -1)
        del t1098

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t1100 = paddle._C_ops.matmul(t1099, t1089, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t1101 = paddle._C_ops.transpose(t1100, [0, 2, 1, 3])
        del t1100

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1102 = [t1079, t441, t811]
        del t1079

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1103 = paddle._C_ops.stack(t1102, 0)
        del t1102

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t1104 = paddle._C_ops.reshape(t1101, t1103)
        del t1103

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t1105 = paddle._C_ops.matmul(t1104, t62, False, False)
        del t62

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t1106 = paddle._C_ops.add(t1105, t63)
        del t63

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t1107 = paddle._C_ops.reshape(t1106, t1074)

        # pd_op.full_int_array: (6xi64) <- ()
        t1108 = [-1, 2, 2, 7, 7, 768]

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t1109 = paddle._C_ops.reshape(t1107, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t1110 = paddle._C_ops.transpose(t1109, [0, 1, 3, 2, 4, 5])
        del t1109

        # pd_op.full_int_array: (4xi64) <- ()
        t1111 = [-1, 14, 14, 768]

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t1112 = paddle._C_ops.reshape(t1110, t1111)

        # pd_op.full: (xi64) <- ()
        t1113 = paddle._C_ops.full(
            [], float("196"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1114 = [t1057, t1113, t811]
        del t1057

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1115 = paddle._C_ops.stack(t1114, 0)
        del t1114

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t1116 = paddle._C_ops.reshape(t1112, t1115)
        del t1115

        # pd_op.full: (xf32) <- ()
        t1117 = paddle._C_ops.full(
            [],
            float("0.982609"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1118 = t1117

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1119 = paddle._C_ops.shape64(t1116)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1120 = paddle._C_ops.slice(t1119, [0], t305, t354, [1], [0])
        del t1119

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1121 = [t1120, t744, t744]
        del t1120

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1122 = paddle._C_ops.stack(t1121, 0)
        del t1121

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1123 = paddle._C_ops.uniform(
            t1122,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1122

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1124 = paddle._C_ops.add(t1117, t1123)
        del t1123

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1125 = paddle._C_ops.floor(t1124)
        del t1124

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1126 = paddle._C_ops.divide(t1116, t1117)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1127 = paddle._C_ops.multiply(t1126, t1125)

        # pd_op.add: (-1x196x768xf32) <- (-1x-1x768xf32, -1x196x768xf32)
        t1128 = paddle._C_ops.add(t1055, t1127)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1129, t1130, t1131 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1128, t64, t65, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t65, t64

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t1132 = paddle._C_ops.matmul(t1129, t66, False, False)
        del t66

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t1133 = paddle._C_ops.add(t1132, t67)
        del t67

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t1134 = paddle._C_ops.gelu(t1133, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t1135 = paddle._C_ops.matmul(t1134, t68, False, False)
        del t68

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t1136 = paddle._C_ops.add(t1135, t69)
        del t69

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1137 = paddle._C_ops.shape64(t1136)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1138 = paddle._C_ops.slice(t1137, [0], t305, t354, [1], [0])
        del t1137

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1139 = [t1138, t744, t744]
        del t1138

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1140 = paddle._C_ops.stack(t1139, 0)
        del t1139

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1141 = paddle._C_ops.uniform(
            t1140,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1140

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1142 = paddle._C_ops.add(t1117, t1141)
        del t1141

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1143 = paddle._C_ops.floor(t1142)
        del t1142

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1144 = paddle._C_ops.divide(t1136, t1117)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1145 = paddle._C_ops.multiply(t1144, t1143)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1146 = paddle._C_ops.add(t1128, t1145)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1147 = paddle._C_ops.shape64(t1146)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1148 = paddle._C_ops.slice(t1147, [0], t305, t354, [1], [0])
        del t1147

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1149, t1150, t1151 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1146, t70, t71, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t71, t70

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1152 = [t1148, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1153 = paddle._C_ops.stack(t1152, 0)
        del t1152

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t1154 = paddle._C_ops.reshape(t1149, t1153)
        del t1153

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1155 = paddle._C_ops.shape64(t1154)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1156 = paddle._C_ops.slice(t1155, [0], t305, t354, [1], [0])
        del t1155

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t1157 = paddle._C_ops.roll(t1154, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1158 = paddle._C_ops.shape64(t1157)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1159 = paddle._C_ops.slice(t1158, [0], t305, t354, [1], [0])
        del t1158

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1160 = [t1159, t1069, t428, t1069, t428, t811]
        del t1159

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1161 = paddle._C_ops.stack(t1160, 0)
        del t1160

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1162 = paddle._C_ops.reshape(t1157, t1161)
        del t1161

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1163 = paddle._C_ops.transpose(t1162, [0, 1, 3, 2, 4, 5])
        del t1162

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1164 = paddle._C_ops.reshape(t1163, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1165 = paddle._C_ops.reshape(t1164, t1076)

        # pd_op.full: (1x14x14x1xf32) <- ()
        t1166 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1167 = paddle._C_ops.set_value_(
            t1166, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t1166

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1168 = paddle._C_ops.set_value_(
            t1167, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t1167

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1169 = paddle._C_ops.set_value_(
            t1168, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t1168

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1170 = paddle._C_ops.set_value_(
            t1169, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t1169

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1171 = paddle._C_ops.set_value_(
            t1170, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t1170

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1172 = paddle._C_ops.set_value_(
            t1171, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t1171

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1173 = paddle._C_ops.set_value_(
            t1172, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t1172

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1174 = paddle._C_ops.set_value_(
            t1173, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t1173

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1175 = paddle._C_ops.set_value_(
            t1174, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t1174

        # pd_op.full_int_array: (6xi64) <- ()
        t1176 = [1, 2, 7, 2, 7, 1]

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        t1177 = paddle._C_ops.reshape(t1175, t1176)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        t1178 = paddle._C_ops.transpose(t1177, [0, 1, 3, 2, 4, 5])
        del t1177

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        t1179 = paddle._C_ops.reshape(t1178, t666)
        del t1178

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        t1180 = paddle._C_ops.reshape(t1179, t668)
        del t1179

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        t1181 = paddle._C_ops.unsqueeze(t1180, t354)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        t1182 = paddle._C_ops.unsqueeze(t1180, t450)
        del t1180

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        t1183 = paddle._C_ops.subtract(t1181, t1182)
        del t1181, t1182

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1184 = paddle._C_ops.not_equal(t1183, t673)

        # pd_op.full: (4x49x49xf32) <- ()
        t1185 = paddle._C_ops.full(
            [4, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1186 = paddle._C_ops.where(t1184, t1185, t1183)
        del t1184, t1183

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1187 = paddle._C_ops.equal(t1186, t673)

        # pd_op.full: (4x49x49xf32) <- ()
        t1188 = paddle._C_ops.full(
            [4, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1189 = paddle._C_ops.where(t1187, t1188, t1186)
        del t1187, t1186

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1190 = paddle._C_ops.shape64(t1165)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1191 = paddle._C_ops.slice(t1190, [0], t305, t354, [1], [0])
        del t1190

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1192 = paddle._C_ops.matmul(t1165, t72, False, False)
        del t72

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1193 = paddle._C_ops.add(t1192, t73)
        del t73

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1194 = [t1191, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1195 = paddle._C_ops.stack(t1194, 0)
        del t1194

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1196 = paddle._C_ops.reshape(t1193, t1195)
        del t1195

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1197 = paddle._C_ops.transpose(t1196, [2, 0, 3, 1, 4])
        del t1196

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1198 = paddle._C_ops.slice(t1197, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1199 = paddle._C_ops.slice(t1197, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1200 = paddle._C_ops.slice(t1197, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1201 = paddle._C_ops.scale(t1198, t526, float("0"), True)
        del t1198

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1202 = paddle._C_ops.transpose(t1199, [0, 1, 3, 2])
        del t1199

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1203 = paddle._C_ops.matmul(t1201, t1202, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1204 = paddle._C_ops.reshape(input11, t553)
        del input11

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1205 = paddle._C_ops.index_select(input12, t1204, 0)
        del input12

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1206 = paddle._C_ops.reshape(t1205, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1207 = paddle._C_ops.transpose(t1206, [2, 0, 1])
        del t1206

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1208 = paddle._C_ops.unsqueeze(t1207, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1209 = paddle._C_ops.add(t1203, t1208)

        # pd_op.full: (xi64) <- ()
        t1210 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1211 = paddle._C_ops.floor_divide(t1191, t1210)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1212 = [t1211, t833, t1082, t441, t441]
        del t1211

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1213 = paddle._C_ops.stack(t1212, 0)
        del t1212

        # pd_op.reshape: (-1x4x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        t1214 = paddle._C_ops.reshape(t1209, t1213)
        del t1213

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        t1215 = paddle._C_ops.unsqueeze(t1189, t354)
        del t1189

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        t1216 = paddle._C_ops.unsqueeze(t1215, t305)
        del t1215

        # pd_op.add: (-1x4x24x49x49xf32) <- (-1x4x24x49x49xf32, 1x4x1x49x49xf32)
        t1217 = paddle._C_ops.add(t1214, t1216)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1218 = [t1191, t1082, t441, t441]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1219 = paddle._C_ops.stack(t1218, 0)
        del t1218

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x4x24x49x49xf32, 4xi64)
        t1220 = paddle._C_ops.reshape(t1217, t1219)
        del t1219

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t1221 = paddle._C_ops.softmax(t1220, -1)
        del t1220

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t1222 = paddle._C_ops.matmul(t1221, t1200, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t1223 = paddle._C_ops.transpose(t1222, [0, 2, 1, 3])
        del t1222

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1224 = [t1191, t441, t811]
        del t1191

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1225 = paddle._C_ops.stack(t1224, 0)
        del t1224

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t1226 = paddle._C_ops.reshape(t1223, t1225)
        del t1225

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t1227 = paddle._C_ops.matmul(t1226, t74, False, False)
        del t74

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t1228 = paddle._C_ops.add(t1227, t75)
        del t75

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t1229 = paddle._C_ops.reshape(t1228, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t1230 = paddle._C_ops.reshape(t1229, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t1231 = paddle._C_ops.transpose(t1230, [0, 1, 3, 2, 4, 5])
        del t1230

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t1232 = paddle._C_ops.reshape(t1231, t1111)

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t1233 = paddle._C_ops.roll(t1232, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1234 = [t1148, t1113, t811]
        del t1148

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1235 = paddle._C_ops.stack(t1234, 0)
        del t1234

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t1236 = paddle._C_ops.reshape(t1233, t1235)
        del t1235

        # pd_op.full: (xf32) <- ()
        t1237 = paddle._C_ops.full(
            [],
            float("0.978261"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1238 = t1237

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1239 = paddle._C_ops.shape64(t1236)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1240 = paddle._C_ops.slice(t1239, [0], t305, t354, [1], [0])
        del t1239

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1241 = [t1240, t744, t744]
        del t1240

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1242 = paddle._C_ops.stack(t1241, 0)
        del t1241

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1243 = paddle._C_ops.uniform(
            t1242,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1242

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1244 = paddle._C_ops.add(t1237, t1243)
        del t1243

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1245 = paddle._C_ops.floor(t1244)
        del t1244

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1246 = paddle._C_ops.divide(t1236, t1237)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1247 = paddle._C_ops.multiply(t1246, t1245)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1248 = paddle._C_ops.add(t1146, t1247)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1249, t1250, t1251 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1248, t76, t77, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t77, t76

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t1252 = paddle._C_ops.matmul(t1249, t78, False, False)
        del t78

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t1253 = paddle._C_ops.add(t1252, t79)
        del t79

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t1254 = paddle._C_ops.gelu(t1253, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t1255 = paddle._C_ops.matmul(t1254, t80, False, False)
        del t80

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t1256 = paddle._C_ops.add(t1255, t81)
        del t81

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1257 = paddle._C_ops.shape64(t1256)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1258 = paddle._C_ops.slice(t1257, [0], t305, t354, [1], [0])
        del t1257

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1259 = [t1258, t744, t744]
        del t1258

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1260 = paddle._C_ops.stack(t1259, 0)
        del t1259

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1261 = paddle._C_ops.uniform(
            t1260,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1260

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1262 = paddle._C_ops.add(t1237, t1261)
        del t1261

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1263 = paddle._C_ops.floor(t1262)
        del t1262

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1264 = paddle._C_ops.divide(t1256, t1237)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1265 = paddle._C_ops.multiply(t1264, t1263)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1266 = paddle._C_ops.add(t1248, t1265)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1267 = paddle._C_ops.shape64(t1266)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1268 = paddle._C_ops.slice(t1267, [0], t305, t354, [1], [0])
        del t1267

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1269, t1270, t1271 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1266, t82, t83, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t83, t82

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1272 = [t1268, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1273 = paddle._C_ops.stack(t1272, 0)
        del t1272

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t1274 = paddle._C_ops.reshape(t1269, t1273)
        del t1273

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1275 = paddle._C_ops.shape64(t1274)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1276 = paddle._C_ops.slice(t1275, [0], t305, t354, [1], [0])
        del t1275

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1277 = [t1276, t1069, t428, t1069, t428, t811]
        del t1276

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1278 = paddle._C_ops.stack(t1277, 0)
        del t1277

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1279 = paddle._C_ops.reshape(t1274, t1278)
        del t1278

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1280 = paddle._C_ops.transpose(t1279, [0, 1, 3, 2, 4, 5])
        del t1279

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1281 = paddle._C_ops.reshape(t1280, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1282 = paddle._C_ops.reshape(t1281, t1076)

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1283 = paddle._C_ops.shape64(t1282)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1284 = paddle._C_ops.slice(t1283, [0], t305, t354, [1], [0])
        del t1283

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1285 = paddle._C_ops.matmul(t1282, t84, False, False)
        del t84

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1286 = paddle._C_ops.add(t1285, t85)
        del t85

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1287 = [t1284, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1288 = paddle._C_ops.stack(t1287, 0)
        del t1287

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1289 = paddle._C_ops.reshape(t1286, t1288)
        del t1288

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1290 = paddle._C_ops.transpose(t1289, [2, 0, 3, 1, 4])
        del t1289

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1291 = paddle._C_ops.slice(t1290, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1292 = paddle._C_ops.slice(t1290, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1293 = paddle._C_ops.slice(t1290, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1294 = paddle._C_ops.scale(t1291, t526, float("0"), True)
        del t1291

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1295 = paddle._C_ops.transpose(t1292, [0, 1, 3, 2])
        del t1292

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1296 = paddle._C_ops.matmul(t1294, t1295, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1297 = paddle._C_ops.reshape(input13, t553)
        del input13

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1298 = paddle._C_ops.index_select(input14, t1297, 0)
        del input14

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1299 = paddle._C_ops.reshape(t1298, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1300 = paddle._C_ops.transpose(t1299, [2, 0, 1])
        del t1299

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1301 = paddle._C_ops.unsqueeze(t1300, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1302 = paddle._C_ops.add(t1296, t1301)

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t1303 = paddle._C_ops.softmax(t1302, -1)
        del t1302

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t1304 = paddle._C_ops.matmul(t1303, t1293, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t1305 = paddle._C_ops.transpose(t1304, [0, 2, 1, 3])
        del t1304

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1306 = [t1284, t441, t811]
        del t1284

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1307 = paddle._C_ops.stack(t1306, 0)
        del t1306

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t1308 = paddle._C_ops.reshape(t1305, t1307)
        del t1307

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t1309 = paddle._C_ops.matmul(t1308, t86, False, False)
        del t86

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t1310 = paddle._C_ops.add(t1309, t87)
        del t87

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t1311 = paddle._C_ops.reshape(t1310, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t1312 = paddle._C_ops.reshape(t1311, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t1313 = paddle._C_ops.transpose(t1312, [0, 1, 3, 2, 4, 5])
        del t1312

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t1314 = paddle._C_ops.reshape(t1313, t1111)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1315 = [t1268, t1113, t811]
        del t1268

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1316 = paddle._C_ops.stack(t1315, 0)
        del t1315

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t1317 = paddle._C_ops.reshape(t1314, t1316)
        del t1316

        # pd_op.full: (xf32) <- ()
        t1318 = paddle._C_ops.full(
            [],
            float("0.973913"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1319 = t1318

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1320 = paddle._C_ops.shape64(t1317)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1321 = paddle._C_ops.slice(t1320, [0], t305, t354, [1], [0])
        del t1320

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1322 = [t1321, t744, t744]
        del t1321

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1323 = paddle._C_ops.stack(t1322, 0)
        del t1322

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1324 = paddle._C_ops.uniform(
            t1323,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1323

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1325 = paddle._C_ops.add(t1318, t1324)
        del t1324

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1326 = paddle._C_ops.floor(t1325)
        del t1325

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1327 = paddle._C_ops.divide(t1317, t1318)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1328 = paddle._C_ops.multiply(t1327, t1326)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1329 = paddle._C_ops.add(t1266, t1328)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1330, t1331, t1332 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1329, t88, t89, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t89, t88

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t1333 = paddle._C_ops.matmul(t1330, t90, False, False)
        del t90

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t1334 = paddle._C_ops.add(t1333, t91)
        del t91

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t1335 = paddle._C_ops.gelu(t1334, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t1336 = paddle._C_ops.matmul(t1335, t92, False, False)
        del t92

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t1337 = paddle._C_ops.add(t1336, t93)
        del t93

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1338 = paddle._C_ops.shape64(t1337)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1339 = paddle._C_ops.slice(t1338, [0], t305, t354, [1], [0])
        del t1338

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1340 = [t1339, t744, t744]
        del t1339

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1341 = paddle._C_ops.stack(t1340, 0)
        del t1340

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1342 = paddle._C_ops.uniform(
            t1341,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1341

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1343 = paddle._C_ops.add(t1318, t1342)
        del t1342

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1344 = paddle._C_ops.floor(t1343)
        del t1343

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1345 = paddle._C_ops.divide(t1337, t1318)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1346 = paddle._C_ops.multiply(t1345, t1344)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1347 = paddle._C_ops.add(t1329, t1346)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1348 = paddle._C_ops.shape64(t1347)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1349 = paddle._C_ops.slice(t1348, [0], t305, t354, [1], [0])
        del t1348

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1350, t1351, t1352 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1347, t94, t95, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t95, t94

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1353 = [t1349, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1354 = paddle._C_ops.stack(t1353, 0)
        del t1353

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t1355 = paddle._C_ops.reshape(t1350, t1354)
        del t1354

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1356 = paddle._C_ops.shape64(t1355)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1357 = paddle._C_ops.slice(t1356, [0], t305, t354, [1], [0])
        del t1356

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t1358 = paddle._C_ops.roll(t1355, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1359 = paddle._C_ops.shape64(t1358)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1360 = paddle._C_ops.slice(t1359, [0], t305, t354, [1], [0])
        del t1359

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1361 = [t1360, t1069, t428, t1069, t428, t811]
        del t1360

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1362 = paddle._C_ops.stack(t1361, 0)
        del t1361

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1363 = paddle._C_ops.reshape(t1358, t1362)
        del t1362

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1364 = paddle._C_ops.transpose(t1363, [0, 1, 3, 2, 4, 5])
        del t1363

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1365 = paddle._C_ops.reshape(t1364, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1366 = paddle._C_ops.reshape(t1365, t1076)

        # pd_op.full: (1x14x14x1xf32) <- ()
        t1367 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1368 = paddle._C_ops.set_value_(
            t1367, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t1367

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1369 = paddle._C_ops.set_value_(
            t1368, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t1368

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1370 = paddle._C_ops.set_value_(
            t1369, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t1369

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1371 = paddle._C_ops.set_value_(
            t1370, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t1370

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1372 = paddle._C_ops.set_value_(
            t1371, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t1371

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1373 = paddle._C_ops.set_value_(
            t1372, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t1372

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1374 = paddle._C_ops.set_value_(
            t1373, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t1373

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1375 = paddle._C_ops.set_value_(
            t1374, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t1374

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1376 = paddle._C_ops.set_value_(
            t1375, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t1375

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        t1377 = paddle._C_ops.reshape(t1376, t1176)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        t1378 = paddle._C_ops.transpose(t1377, [0, 1, 3, 2, 4, 5])
        del t1377

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        t1379 = paddle._C_ops.reshape(t1378, t666)
        del t1378

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        t1380 = paddle._C_ops.reshape(t1379, t668)
        del t1379

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        t1381 = paddle._C_ops.unsqueeze(t1380, t354)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        t1382 = paddle._C_ops.unsqueeze(t1380, t450)
        del t1380

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        t1383 = paddle._C_ops.subtract(t1381, t1382)
        del t1381, t1382

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1384 = paddle._C_ops.not_equal(t1383, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1385 = paddle._C_ops.where(t1384, t1185, t1383)
        del t1384, t1383

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1386 = paddle._C_ops.equal(t1385, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1387 = paddle._C_ops.where(t1386, t1188, t1385)
        del t1386, t1385

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1388 = paddle._C_ops.shape64(t1366)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1389 = paddle._C_ops.slice(t1388, [0], t305, t354, [1], [0])
        del t1388

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1390 = paddle._C_ops.matmul(t1366, t96, False, False)
        del t96

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1391 = paddle._C_ops.add(t1390, t97)
        del t97

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1392 = [t1389, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1393 = paddle._C_ops.stack(t1392, 0)
        del t1392

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1394 = paddle._C_ops.reshape(t1391, t1393)
        del t1393

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1395 = paddle._C_ops.transpose(t1394, [2, 0, 3, 1, 4])
        del t1394

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1396 = paddle._C_ops.slice(t1395, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1397 = paddle._C_ops.slice(t1395, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1398 = paddle._C_ops.slice(t1395, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1399 = paddle._C_ops.scale(t1396, t526, float("0"), True)
        del t1396

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1400 = paddle._C_ops.transpose(t1397, [0, 1, 3, 2])
        del t1397

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1401 = paddle._C_ops.matmul(t1399, t1400, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1402 = paddle._C_ops.reshape(input15, t553)
        del input15

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1403 = paddle._C_ops.index_select(input16, t1402, 0)
        del input16

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1404 = paddle._C_ops.reshape(t1403, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1405 = paddle._C_ops.transpose(t1404, [2, 0, 1])
        del t1404

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1406 = paddle._C_ops.unsqueeze(t1405, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1407 = paddle._C_ops.add(t1401, t1406)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1408 = paddle._C_ops.floor_divide(t1389, t1210)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1409 = [t1408, t833, t1082, t441, t441]
        del t1408

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1410 = paddle._C_ops.stack(t1409, 0)
        del t1409

        # pd_op.reshape: (-1x4x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        t1411 = paddle._C_ops.reshape(t1407, t1410)
        del t1410

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        t1412 = paddle._C_ops.unsqueeze(t1387, t354)
        del t1387

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        t1413 = paddle._C_ops.unsqueeze(t1412, t305)
        del t1412

        # pd_op.add: (-1x4x24x49x49xf32) <- (-1x4x24x49x49xf32, 1x4x1x49x49xf32)
        t1414 = paddle._C_ops.add(t1411, t1413)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1415 = [t1389, t1082, t441, t441]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1416 = paddle._C_ops.stack(t1415, 0)
        del t1415

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x4x24x49x49xf32, 4xi64)
        t1417 = paddle._C_ops.reshape(t1414, t1416)
        del t1416

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t1418 = paddle._C_ops.softmax(t1417, -1)
        del t1417

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t1419 = paddle._C_ops.matmul(t1418, t1398, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t1420 = paddle._C_ops.transpose(t1419, [0, 2, 1, 3])
        del t1419

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1421 = [t1389, t441, t811]
        del t1389

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1422 = paddle._C_ops.stack(t1421, 0)
        del t1421

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t1423 = paddle._C_ops.reshape(t1420, t1422)
        del t1422

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t1424 = paddle._C_ops.matmul(t1423, t98, False, False)
        del t98

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t1425 = paddle._C_ops.add(t1424, t99)
        del t99

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t1426 = paddle._C_ops.reshape(t1425, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t1427 = paddle._C_ops.reshape(t1426, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t1428 = paddle._C_ops.transpose(t1427, [0, 1, 3, 2, 4, 5])
        del t1427

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t1429 = paddle._C_ops.reshape(t1428, t1111)

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t1430 = paddle._C_ops.roll(t1429, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1431 = [t1349, t1113, t811]
        del t1349

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1432 = paddle._C_ops.stack(t1431, 0)
        del t1431

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t1433 = paddle._C_ops.reshape(t1430, t1432)
        del t1432

        # pd_op.full: (xf32) <- ()
        t1434 = paddle._C_ops.full(
            [],
            float("0.969565"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1435 = t1434

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1436 = paddle._C_ops.shape64(t1433)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1437 = paddle._C_ops.slice(t1436, [0], t305, t354, [1], [0])
        del t1436

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1438 = [t1437, t744, t744]
        del t1437

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1439 = paddle._C_ops.stack(t1438, 0)
        del t1438

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1440 = paddle._C_ops.uniform(
            t1439,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1439

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1441 = paddle._C_ops.add(t1434, t1440)
        del t1440

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1442 = paddle._C_ops.floor(t1441)
        del t1441

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1443 = paddle._C_ops.divide(t1433, t1434)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1444 = paddle._C_ops.multiply(t1443, t1442)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1445 = paddle._C_ops.add(t1347, t1444)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1446, t1447, t1448 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1445, t100, t101, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t101, t100

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t1449 = paddle._C_ops.matmul(t1446, t102, False, False)
        del t102

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t1450 = paddle._C_ops.add(t1449, t103)
        del t103

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t1451 = paddle._C_ops.gelu(t1450, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t1452 = paddle._C_ops.matmul(t1451, t104, False, False)
        del t104

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t1453 = paddle._C_ops.add(t1452, t105)
        del t105

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1454 = paddle._C_ops.shape64(t1453)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1455 = paddle._C_ops.slice(t1454, [0], t305, t354, [1], [0])
        del t1454

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1456 = [t1455, t744, t744]
        del t1455

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1457 = paddle._C_ops.stack(t1456, 0)
        del t1456

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1458 = paddle._C_ops.uniform(
            t1457,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1457

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1459 = paddle._C_ops.add(t1434, t1458)
        del t1458

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1460 = paddle._C_ops.floor(t1459)
        del t1459

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1461 = paddle._C_ops.divide(t1453, t1434)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1462 = paddle._C_ops.multiply(t1461, t1460)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1463 = paddle._C_ops.add(t1445, t1462)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1464 = paddle._C_ops.shape64(t1463)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1465 = paddle._C_ops.slice(t1464, [0], t305, t354, [1], [0])
        del t1464

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1466, t1467, t1468 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1463, t106, t107, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t107, t106

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1469 = [t1465, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1470 = paddle._C_ops.stack(t1469, 0)
        del t1469

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t1471 = paddle._C_ops.reshape(t1466, t1470)
        del t1470

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1472 = paddle._C_ops.shape64(t1471)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1473 = paddle._C_ops.slice(t1472, [0], t305, t354, [1], [0])
        del t1472

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1474 = [t1473, t1069, t428, t1069, t428, t811]
        del t1473

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1475 = paddle._C_ops.stack(t1474, 0)
        del t1474

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1476 = paddle._C_ops.reshape(t1471, t1475)
        del t1475

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1477 = paddle._C_ops.transpose(t1476, [0, 1, 3, 2, 4, 5])
        del t1476

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1478 = paddle._C_ops.reshape(t1477, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1479 = paddle._C_ops.reshape(t1478, t1076)

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1480 = paddle._C_ops.shape64(t1479)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1481 = paddle._C_ops.slice(t1480, [0], t305, t354, [1], [0])
        del t1480

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1482 = paddle._C_ops.matmul(t1479, t108, False, False)
        del t108

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1483 = paddle._C_ops.add(t1482, t109)
        del t109

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1484 = [t1481, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1485 = paddle._C_ops.stack(t1484, 0)
        del t1484

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1486 = paddle._C_ops.reshape(t1483, t1485)
        del t1485

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1487 = paddle._C_ops.transpose(t1486, [2, 0, 3, 1, 4])
        del t1486

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1488 = paddle._C_ops.slice(t1487, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1489 = paddle._C_ops.slice(t1487, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1490 = paddle._C_ops.slice(t1487, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1491 = paddle._C_ops.scale(t1488, t526, float("0"), True)
        del t1488

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1492 = paddle._C_ops.transpose(t1489, [0, 1, 3, 2])
        del t1489

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1493 = paddle._C_ops.matmul(t1491, t1492, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1494 = paddle._C_ops.reshape(input17, t553)
        del input17

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1495 = paddle._C_ops.index_select(input18, t1494, 0)
        del input18

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1496 = paddle._C_ops.reshape(t1495, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1497 = paddle._C_ops.transpose(t1496, [2, 0, 1])
        del t1496

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1498 = paddle._C_ops.unsqueeze(t1497, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1499 = paddle._C_ops.add(t1493, t1498)

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t1500 = paddle._C_ops.softmax(t1499, -1)
        del t1499

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t1501 = paddle._C_ops.matmul(t1500, t1490, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t1502 = paddle._C_ops.transpose(t1501, [0, 2, 1, 3])
        del t1501

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1503 = [t1481, t441, t811]
        del t1481

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1504 = paddle._C_ops.stack(t1503, 0)
        del t1503

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t1505 = paddle._C_ops.reshape(t1502, t1504)
        del t1504

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t1506 = paddle._C_ops.matmul(t1505, t110, False, False)
        del t110

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t1507 = paddle._C_ops.add(t1506, t111)
        del t111

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t1508 = paddle._C_ops.reshape(t1507, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t1509 = paddle._C_ops.reshape(t1508, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t1510 = paddle._C_ops.transpose(t1509, [0, 1, 3, 2, 4, 5])
        del t1509

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t1511 = paddle._C_ops.reshape(t1510, t1111)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1512 = [t1465, t1113, t811]
        del t1465

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1513 = paddle._C_ops.stack(t1512, 0)
        del t1512

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t1514 = paddle._C_ops.reshape(t1511, t1513)
        del t1513

        # pd_op.full: (xf32) <- ()
        t1515 = paddle._C_ops.full(
            [],
            float("0.965217"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1516 = t1515

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1517 = paddle._C_ops.shape64(t1514)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1518 = paddle._C_ops.slice(t1517, [0], t305, t354, [1], [0])
        del t1517

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1519 = [t1518, t744, t744]
        del t1518

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1520 = paddle._C_ops.stack(t1519, 0)
        del t1519

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1521 = paddle._C_ops.uniform(
            t1520,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1520

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1522 = paddle._C_ops.add(t1515, t1521)
        del t1521

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1523 = paddle._C_ops.floor(t1522)
        del t1522

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1524 = paddle._C_ops.divide(t1514, t1515)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1525 = paddle._C_ops.multiply(t1524, t1523)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1526 = paddle._C_ops.add(t1463, t1525)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1527, t1528, t1529 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1526, t112, t113, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t113, t112

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t1530 = paddle._C_ops.matmul(t1527, t114, False, False)
        del t114

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t1531 = paddle._C_ops.add(t1530, t115)
        del t115

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t1532 = paddle._C_ops.gelu(t1531, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t1533 = paddle._C_ops.matmul(t1532, t116, False, False)
        del t116

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t1534 = paddle._C_ops.add(t1533, t117)
        del t117

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1535 = paddle._C_ops.shape64(t1534)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1536 = paddle._C_ops.slice(t1535, [0], t305, t354, [1], [0])
        del t1535

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1537 = [t1536, t744, t744]
        del t1536

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1538 = paddle._C_ops.stack(t1537, 0)
        del t1537

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1539 = paddle._C_ops.uniform(
            t1538,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1538

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1540 = paddle._C_ops.add(t1515, t1539)
        del t1539

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1541 = paddle._C_ops.floor(t1540)
        del t1540

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1542 = paddle._C_ops.divide(t1534, t1515)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1543 = paddle._C_ops.multiply(t1542, t1541)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1544 = paddle._C_ops.add(t1526, t1543)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1545 = paddle._C_ops.shape64(t1544)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1546 = paddle._C_ops.slice(t1545, [0], t305, t354, [1], [0])
        del t1545

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1547, t1548, t1549 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1544, t118, t119, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t119, t118

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1550 = [t1546, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1551 = paddle._C_ops.stack(t1550, 0)
        del t1550

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t1552 = paddle._C_ops.reshape(t1547, t1551)
        del t1551

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1553 = paddle._C_ops.shape64(t1552)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1554 = paddle._C_ops.slice(t1553, [0], t305, t354, [1], [0])
        del t1553

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t1555 = paddle._C_ops.roll(t1552, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1556 = paddle._C_ops.shape64(t1555)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1557 = paddle._C_ops.slice(t1556, [0], t305, t354, [1], [0])
        del t1556

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1558 = [t1557, t1069, t428, t1069, t428, t811]
        del t1557

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1559 = paddle._C_ops.stack(t1558, 0)
        del t1558

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1560 = paddle._C_ops.reshape(t1555, t1559)
        del t1559

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1561 = paddle._C_ops.transpose(t1560, [0, 1, 3, 2, 4, 5])
        del t1560

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1562 = paddle._C_ops.reshape(t1561, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1563 = paddle._C_ops.reshape(t1562, t1076)

        # pd_op.full: (1x14x14x1xf32) <- ()
        t1564 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1565 = paddle._C_ops.set_value_(
            t1564, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t1564

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1566 = paddle._C_ops.set_value_(
            t1565, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t1565

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1567 = paddle._C_ops.set_value_(
            t1566, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t1566

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1568 = paddle._C_ops.set_value_(
            t1567, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t1567

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1569 = paddle._C_ops.set_value_(
            t1568, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t1568

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1570 = paddle._C_ops.set_value_(
            t1569, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t1569

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1571 = paddle._C_ops.set_value_(
            t1570, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t1570

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1572 = paddle._C_ops.set_value_(
            t1571, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t1571

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1573 = paddle._C_ops.set_value_(
            t1572, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t1572

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        t1574 = paddle._C_ops.reshape(t1573, t1176)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        t1575 = paddle._C_ops.transpose(t1574, [0, 1, 3, 2, 4, 5])
        del t1574

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        t1576 = paddle._C_ops.reshape(t1575, t666)
        del t1575

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        t1577 = paddle._C_ops.reshape(t1576, t668)
        del t1576

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        t1578 = paddle._C_ops.unsqueeze(t1577, t354)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        t1579 = paddle._C_ops.unsqueeze(t1577, t450)
        del t1577

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        t1580 = paddle._C_ops.subtract(t1578, t1579)
        del t1578, t1579

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1581 = paddle._C_ops.not_equal(t1580, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1582 = paddle._C_ops.where(t1581, t1185, t1580)
        del t1581, t1580

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1583 = paddle._C_ops.equal(t1582, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1584 = paddle._C_ops.where(t1583, t1188, t1582)
        del t1583, t1582

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1585 = paddle._C_ops.shape64(t1563)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1586 = paddle._C_ops.slice(t1585, [0], t305, t354, [1], [0])
        del t1585

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1587 = paddle._C_ops.matmul(t1563, t120, False, False)
        del t120

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1588 = paddle._C_ops.add(t1587, t121)
        del t121

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1589 = [t1586, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1590 = paddle._C_ops.stack(t1589, 0)
        del t1589

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1591 = paddle._C_ops.reshape(t1588, t1590)
        del t1590

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1592 = paddle._C_ops.transpose(t1591, [2, 0, 3, 1, 4])
        del t1591

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1593 = paddle._C_ops.slice(t1592, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1594 = paddle._C_ops.slice(t1592, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1595 = paddle._C_ops.slice(t1592, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1596 = paddle._C_ops.scale(t1593, t526, float("0"), True)
        del t1593

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1597 = paddle._C_ops.transpose(t1594, [0, 1, 3, 2])
        del t1594

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1598 = paddle._C_ops.matmul(t1596, t1597, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1599 = paddle._C_ops.reshape(input19, t553)
        del input19

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1600 = paddle._C_ops.index_select(input20, t1599, 0)
        del input20

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1601 = paddle._C_ops.reshape(t1600, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1602 = paddle._C_ops.transpose(t1601, [2, 0, 1])
        del t1601

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1603 = paddle._C_ops.unsqueeze(t1602, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1604 = paddle._C_ops.add(t1598, t1603)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1605 = paddle._C_ops.floor_divide(t1586, t1210)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1606 = [t1605, t833, t1082, t441, t441]
        del t1605

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1607 = paddle._C_ops.stack(t1606, 0)
        del t1606

        # pd_op.reshape: (-1x4x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        t1608 = paddle._C_ops.reshape(t1604, t1607)
        del t1607

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        t1609 = paddle._C_ops.unsqueeze(t1584, t354)
        del t1584

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        t1610 = paddle._C_ops.unsqueeze(t1609, t305)
        del t1609

        # pd_op.add: (-1x4x24x49x49xf32) <- (-1x4x24x49x49xf32, 1x4x1x49x49xf32)
        t1611 = paddle._C_ops.add(t1608, t1610)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1612 = [t1586, t1082, t441, t441]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1613 = paddle._C_ops.stack(t1612, 0)
        del t1612

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x4x24x49x49xf32, 4xi64)
        t1614 = paddle._C_ops.reshape(t1611, t1613)
        del t1613

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t1615 = paddle._C_ops.softmax(t1614, -1)
        del t1614

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t1616 = paddle._C_ops.matmul(t1615, t1595, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t1617 = paddle._C_ops.transpose(t1616, [0, 2, 1, 3])
        del t1616

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1618 = [t1586, t441, t811]
        del t1586

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1619 = paddle._C_ops.stack(t1618, 0)
        del t1618

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t1620 = paddle._C_ops.reshape(t1617, t1619)
        del t1619

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t1621 = paddle._C_ops.matmul(t1620, t122, False, False)
        del t122

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t1622 = paddle._C_ops.add(t1621, t123)
        del t123

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t1623 = paddle._C_ops.reshape(t1622, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t1624 = paddle._C_ops.reshape(t1623, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t1625 = paddle._C_ops.transpose(t1624, [0, 1, 3, 2, 4, 5])
        del t1624

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t1626 = paddle._C_ops.reshape(t1625, t1111)

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t1627 = paddle._C_ops.roll(t1626, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1628 = [t1546, t1113, t811]
        del t1546

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1629 = paddle._C_ops.stack(t1628, 0)
        del t1628

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t1630 = paddle._C_ops.reshape(t1627, t1629)
        del t1629

        # pd_op.full: (xf32) <- ()
        t1631 = paddle._C_ops.full(
            [],
            float("0.96087"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1632 = t1631

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1633 = paddle._C_ops.shape64(t1630)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1634 = paddle._C_ops.slice(t1633, [0], t305, t354, [1], [0])
        del t1633

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1635 = [t1634, t744, t744]
        del t1634

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1636 = paddle._C_ops.stack(t1635, 0)
        del t1635

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1637 = paddle._C_ops.uniform(
            t1636,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1636

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1638 = paddle._C_ops.add(t1631, t1637)
        del t1637

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1639 = paddle._C_ops.floor(t1638)
        del t1638

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1640 = paddle._C_ops.divide(t1630, t1631)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1641 = paddle._C_ops.multiply(t1640, t1639)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1642 = paddle._C_ops.add(t1544, t1641)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1643, t1644, t1645 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1642, t124, t125, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t125, t124

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t1646 = paddle._C_ops.matmul(t1643, t126, False, False)
        del t126

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t1647 = paddle._C_ops.add(t1646, t127)
        del t127

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t1648 = paddle._C_ops.gelu(t1647, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t1649 = paddle._C_ops.matmul(t1648, t128, False, False)
        del t128

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t1650 = paddle._C_ops.add(t1649, t129)
        del t129

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1651 = paddle._C_ops.shape64(t1650)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1652 = paddle._C_ops.slice(t1651, [0], t305, t354, [1], [0])
        del t1651

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1653 = [t1652, t744, t744]
        del t1652

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1654 = paddle._C_ops.stack(t1653, 0)
        del t1653

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1655 = paddle._C_ops.uniform(
            t1654,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1654

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1656 = paddle._C_ops.add(t1631, t1655)
        del t1655

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1657 = paddle._C_ops.floor(t1656)
        del t1656

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1658 = paddle._C_ops.divide(t1650, t1631)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1659 = paddle._C_ops.multiply(t1658, t1657)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1660 = paddle._C_ops.add(t1642, t1659)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1661 = paddle._C_ops.shape64(t1660)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1662 = paddle._C_ops.slice(t1661, [0], t305, t354, [1], [0])
        del t1661

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1663, t1664, t1665 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1660, t130, t131, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t131, t130

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1666 = [t1662, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1667 = paddle._C_ops.stack(t1666, 0)
        del t1666

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t1668 = paddle._C_ops.reshape(t1663, t1667)
        del t1667

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1669 = paddle._C_ops.shape64(t1668)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1670 = paddle._C_ops.slice(t1669, [0], t305, t354, [1], [0])
        del t1669

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1671 = [t1670, t1069, t428, t1069, t428, t811]
        del t1670

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1672 = paddle._C_ops.stack(t1671, 0)
        del t1671

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1673 = paddle._C_ops.reshape(t1668, t1672)
        del t1672

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1674 = paddle._C_ops.transpose(t1673, [0, 1, 3, 2, 4, 5])
        del t1673

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1675 = paddle._C_ops.reshape(t1674, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1676 = paddle._C_ops.reshape(t1675, t1076)

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1677 = paddle._C_ops.shape64(t1676)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1678 = paddle._C_ops.slice(t1677, [0], t305, t354, [1], [0])
        del t1677

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1679 = paddle._C_ops.matmul(t1676, t132, False, False)
        del t132

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1680 = paddle._C_ops.add(t1679, t133)
        del t133

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1681 = [t1678, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1682 = paddle._C_ops.stack(t1681, 0)
        del t1681

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1683 = paddle._C_ops.reshape(t1680, t1682)
        del t1682

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1684 = paddle._C_ops.transpose(t1683, [2, 0, 3, 1, 4])
        del t1683

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1685 = paddle._C_ops.slice(t1684, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1686 = paddle._C_ops.slice(t1684, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1687 = paddle._C_ops.slice(t1684, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1688 = paddle._C_ops.scale(t1685, t526, float("0"), True)
        del t1685

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1689 = paddle._C_ops.transpose(t1686, [0, 1, 3, 2])
        del t1686

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1690 = paddle._C_ops.matmul(t1688, t1689, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1691 = paddle._C_ops.reshape(input21, t553)
        del input21

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1692 = paddle._C_ops.index_select(input22, t1691, 0)
        del input22

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1693 = paddle._C_ops.reshape(t1692, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1694 = paddle._C_ops.transpose(t1693, [2, 0, 1])
        del t1693

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1695 = paddle._C_ops.unsqueeze(t1694, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1696 = paddle._C_ops.add(t1690, t1695)

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t1697 = paddle._C_ops.softmax(t1696, -1)
        del t1696

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t1698 = paddle._C_ops.matmul(t1697, t1687, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t1699 = paddle._C_ops.transpose(t1698, [0, 2, 1, 3])
        del t1698

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1700 = [t1678, t441, t811]
        del t1678

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1701 = paddle._C_ops.stack(t1700, 0)
        del t1700

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t1702 = paddle._C_ops.reshape(t1699, t1701)
        del t1701

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t1703 = paddle._C_ops.matmul(t1702, t134, False, False)
        del t134

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t1704 = paddle._C_ops.add(t1703, t135)
        del t135

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t1705 = paddle._C_ops.reshape(t1704, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t1706 = paddle._C_ops.reshape(t1705, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t1707 = paddle._C_ops.transpose(t1706, [0, 1, 3, 2, 4, 5])
        del t1706

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t1708 = paddle._C_ops.reshape(t1707, t1111)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1709 = [t1662, t1113, t811]
        del t1662

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1710 = paddle._C_ops.stack(t1709, 0)
        del t1709

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t1711 = paddle._C_ops.reshape(t1708, t1710)
        del t1710

        # pd_op.full: (xf32) <- ()
        t1712 = paddle._C_ops.full(
            [],
            float("0.956522"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1713 = t1712

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1714 = paddle._C_ops.shape64(t1711)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1715 = paddle._C_ops.slice(t1714, [0], t305, t354, [1], [0])
        del t1714

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1716 = [t1715, t744, t744]
        del t1715

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1717 = paddle._C_ops.stack(t1716, 0)
        del t1716

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1718 = paddle._C_ops.uniform(
            t1717,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1717

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1719 = paddle._C_ops.add(t1712, t1718)
        del t1718

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1720 = paddle._C_ops.floor(t1719)
        del t1719

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1721 = paddle._C_ops.divide(t1711, t1712)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1722 = paddle._C_ops.multiply(t1721, t1720)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1723 = paddle._C_ops.add(t1660, t1722)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1724, t1725, t1726 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1723, t136, t137, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t137, t136

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t1727 = paddle._C_ops.matmul(t1724, t138, False, False)
        del t138

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t1728 = paddle._C_ops.add(t1727, t139)
        del t139

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t1729 = paddle._C_ops.gelu(t1728, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t1730 = paddle._C_ops.matmul(t1729, t140, False, False)
        del t140

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t1731 = paddle._C_ops.add(t1730, t141)
        del t141

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1732 = paddle._C_ops.shape64(t1731)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1733 = paddle._C_ops.slice(t1732, [0], t305, t354, [1], [0])
        del t1732

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1734 = [t1733, t744, t744]
        del t1733

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1735 = paddle._C_ops.stack(t1734, 0)
        del t1734

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1736 = paddle._C_ops.uniform(
            t1735,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1735

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1737 = paddle._C_ops.add(t1712, t1736)
        del t1736

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1738 = paddle._C_ops.floor(t1737)
        del t1737

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1739 = paddle._C_ops.divide(t1731, t1712)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1740 = paddle._C_ops.multiply(t1739, t1738)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1741 = paddle._C_ops.add(t1723, t1740)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1742 = paddle._C_ops.shape64(t1741)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1743 = paddle._C_ops.slice(t1742, [0], t305, t354, [1], [0])
        del t1742

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1744, t1745, t1746 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1741, t142, t143, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t143, t142

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1747 = [t1743, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1748 = paddle._C_ops.stack(t1747, 0)
        del t1747

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t1749 = paddle._C_ops.reshape(t1744, t1748)
        del t1748

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1750 = paddle._C_ops.shape64(t1749)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1751 = paddle._C_ops.slice(t1750, [0], t305, t354, [1], [0])
        del t1750

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t1752 = paddle._C_ops.roll(t1749, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1753 = paddle._C_ops.shape64(t1752)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1754 = paddle._C_ops.slice(t1753, [0], t305, t354, [1], [0])
        del t1753

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1755 = [t1754, t1069, t428, t1069, t428, t811]
        del t1754

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1756 = paddle._C_ops.stack(t1755, 0)
        del t1755

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1757 = paddle._C_ops.reshape(t1752, t1756)
        del t1756

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1758 = paddle._C_ops.transpose(t1757, [0, 1, 3, 2, 4, 5])
        del t1757

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1759 = paddle._C_ops.reshape(t1758, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1760 = paddle._C_ops.reshape(t1759, t1076)

        # pd_op.full: (1x14x14x1xf32) <- ()
        t1761 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1762 = paddle._C_ops.set_value_(
            t1761, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t1761

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1763 = paddle._C_ops.set_value_(
            t1762, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t1762

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1764 = paddle._C_ops.set_value_(
            t1763, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t1763

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1765 = paddle._C_ops.set_value_(
            t1764, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t1764

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1766 = paddle._C_ops.set_value_(
            t1765, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t1765

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1767 = paddle._C_ops.set_value_(
            t1766, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t1766

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1768 = paddle._C_ops.set_value_(
            t1767, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t1767

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1769 = paddle._C_ops.set_value_(
            t1768, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t1768

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1770 = paddle._C_ops.set_value_(
            t1769, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t1769

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        t1771 = paddle._C_ops.reshape(t1770, t1176)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        t1772 = paddle._C_ops.transpose(t1771, [0, 1, 3, 2, 4, 5])
        del t1771

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        t1773 = paddle._C_ops.reshape(t1772, t666)
        del t1772

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        t1774 = paddle._C_ops.reshape(t1773, t668)
        del t1773

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        t1775 = paddle._C_ops.unsqueeze(t1774, t354)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        t1776 = paddle._C_ops.unsqueeze(t1774, t450)
        del t1774

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        t1777 = paddle._C_ops.subtract(t1775, t1776)
        del t1775, t1776

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1778 = paddle._C_ops.not_equal(t1777, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1779 = paddle._C_ops.where(t1778, t1185, t1777)
        del t1778, t1777

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1780 = paddle._C_ops.equal(t1779, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1781 = paddle._C_ops.where(t1780, t1188, t1779)
        del t1780, t1779

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1782 = paddle._C_ops.shape64(t1760)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1783 = paddle._C_ops.slice(t1782, [0], t305, t354, [1], [0])
        del t1782

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1784 = paddle._C_ops.matmul(t1760, t144, False, False)
        del t144

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1785 = paddle._C_ops.add(t1784, t145)
        del t145

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1786 = [t1783, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1787 = paddle._C_ops.stack(t1786, 0)
        del t1786

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1788 = paddle._C_ops.reshape(t1785, t1787)
        del t1787

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1789 = paddle._C_ops.transpose(t1788, [2, 0, 3, 1, 4])
        del t1788

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1790 = paddle._C_ops.slice(t1789, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1791 = paddle._C_ops.slice(t1789, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1792 = paddle._C_ops.slice(t1789, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1793 = paddle._C_ops.scale(t1790, t526, float("0"), True)
        del t1790

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1794 = paddle._C_ops.transpose(t1791, [0, 1, 3, 2])
        del t1791

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1795 = paddle._C_ops.matmul(t1793, t1794, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1796 = paddle._C_ops.reshape(input23, t553)
        del input23

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1797 = paddle._C_ops.index_select(input24, t1796, 0)
        del input24

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1798 = paddle._C_ops.reshape(t1797, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1799 = paddle._C_ops.transpose(t1798, [2, 0, 1])
        del t1798

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1800 = paddle._C_ops.unsqueeze(t1799, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1801 = paddle._C_ops.add(t1795, t1800)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1802 = paddle._C_ops.floor_divide(t1783, t1210)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1803 = [t1802, t833, t1082, t441, t441]
        del t1802

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1804 = paddle._C_ops.stack(t1803, 0)
        del t1803

        # pd_op.reshape: (-1x4x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        t1805 = paddle._C_ops.reshape(t1801, t1804)
        del t1804

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        t1806 = paddle._C_ops.unsqueeze(t1781, t354)
        del t1781

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        t1807 = paddle._C_ops.unsqueeze(t1806, t305)
        del t1806

        # pd_op.add: (-1x4x24x49x49xf32) <- (-1x4x24x49x49xf32, 1x4x1x49x49xf32)
        t1808 = paddle._C_ops.add(t1805, t1807)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1809 = [t1783, t1082, t441, t441]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1810 = paddle._C_ops.stack(t1809, 0)
        del t1809

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x4x24x49x49xf32, 4xi64)
        t1811 = paddle._C_ops.reshape(t1808, t1810)
        del t1810

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t1812 = paddle._C_ops.softmax(t1811, -1)
        del t1811

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t1813 = paddle._C_ops.matmul(t1812, t1792, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t1814 = paddle._C_ops.transpose(t1813, [0, 2, 1, 3])
        del t1813

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1815 = [t1783, t441, t811]
        del t1783

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1816 = paddle._C_ops.stack(t1815, 0)
        del t1815

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t1817 = paddle._C_ops.reshape(t1814, t1816)
        del t1816

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t1818 = paddle._C_ops.matmul(t1817, t146, False, False)
        del t146

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t1819 = paddle._C_ops.add(t1818, t147)
        del t147

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t1820 = paddle._C_ops.reshape(t1819, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t1821 = paddle._C_ops.reshape(t1820, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t1822 = paddle._C_ops.transpose(t1821, [0, 1, 3, 2, 4, 5])
        del t1821

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t1823 = paddle._C_ops.reshape(t1822, t1111)

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t1824 = paddle._C_ops.roll(t1823, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1825 = [t1743, t1113, t811]
        del t1743

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1826 = paddle._C_ops.stack(t1825, 0)
        del t1825

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t1827 = paddle._C_ops.reshape(t1824, t1826)
        del t1826

        # pd_op.full: (xf32) <- ()
        t1828 = paddle._C_ops.full(
            [],
            float("0.952174"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1829 = t1828

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1830 = paddle._C_ops.shape64(t1827)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1831 = paddle._C_ops.slice(t1830, [0], t305, t354, [1], [0])
        del t1830

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1832 = [t1831, t744, t744]
        del t1831

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1833 = paddle._C_ops.stack(t1832, 0)
        del t1832

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1834 = paddle._C_ops.uniform(
            t1833,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1833

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1835 = paddle._C_ops.add(t1828, t1834)
        del t1834

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1836 = paddle._C_ops.floor(t1835)
        del t1835

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1837 = paddle._C_ops.divide(t1827, t1828)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1838 = paddle._C_ops.multiply(t1837, t1836)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1839 = paddle._C_ops.add(t1741, t1838)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1840, t1841, t1842 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1839, t148, t149, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t149, t148

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t1843 = paddle._C_ops.matmul(t1840, t150, False, False)
        del t150

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t1844 = paddle._C_ops.add(t1843, t151)
        del t151

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t1845 = paddle._C_ops.gelu(t1844, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t1846 = paddle._C_ops.matmul(t1845, t152, False, False)
        del t152

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t1847 = paddle._C_ops.add(t1846, t153)
        del t153

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1848 = paddle._C_ops.shape64(t1847)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1849 = paddle._C_ops.slice(t1848, [0], t305, t354, [1], [0])
        del t1848

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1850 = [t1849, t744, t744]
        del t1849

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1851 = paddle._C_ops.stack(t1850, 0)
        del t1850

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1852 = paddle._C_ops.uniform(
            t1851,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1851

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1853 = paddle._C_ops.add(t1828, t1852)
        del t1852

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1854 = paddle._C_ops.floor(t1853)
        del t1853

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1855 = paddle._C_ops.divide(t1847, t1828)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1856 = paddle._C_ops.multiply(t1855, t1854)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1857 = paddle._C_ops.add(t1839, t1856)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1858 = paddle._C_ops.shape64(t1857)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1859 = paddle._C_ops.slice(t1858, [0], t305, t354, [1], [0])
        del t1858

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1860, t1861, t1862 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1857, t154, t155, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t155, t154

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1863 = [t1859, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1864 = paddle._C_ops.stack(t1863, 0)
        del t1863

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t1865 = paddle._C_ops.reshape(t1860, t1864)
        del t1864

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1866 = paddle._C_ops.shape64(t1865)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1867 = paddle._C_ops.slice(t1866, [0], t305, t354, [1], [0])
        del t1866

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1868 = [t1867, t1069, t428, t1069, t428, t811]
        del t1867

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1869 = paddle._C_ops.stack(t1868, 0)
        del t1868

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1870 = paddle._C_ops.reshape(t1865, t1869)
        del t1869

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1871 = paddle._C_ops.transpose(t1870, [0, 1, 3, 2, 4, 5])
        del t1870

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1872 = paddle._C_ops.reshape(t1871, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1873 = paddle._C_ops.reshape(t1872, t1076)

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1874 = paddle._C_ops.shape64(t1873)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1875 = paddle._C_ops.slice(t1874, [0], t305, t354, [1], [0])
        del t1874

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1876 = paddle._C_ops.matmul(t1873, t156, False, False)
        del t156

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1877 = paddle._C_ops.add(t1876, t157)
        del t157

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1878 = [t1875, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1879 = paddle._C_ops.stack(t1878, 0)
        del t1878

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1880 = paddle._C_ops.reshape(t1877, t1879)
        del t1879

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1881 = paddle._C_ops.transpose(t1880, [2, 0, 3, 1, 4])
        del t1880

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1882 = paddle._C_ops.slice(t1881, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1883 = paddle._C_ops.slice(t1881, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1884 = paddle._C_ops.slice(t1881, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1885 = paddle._C_ops.scale(t1882, t526, float("0"), True)
        del t1882

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1886 = paddle._C_ops.transpose(t1883, [0, 1, 3, 2])
        del t1883

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1887 = paddle._C_ops.matmul(t1885, t1886, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1888 = paddle._C_ops.reshape(input25, t553)
        del input25

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1889 = paddle._C_ops.index_select(input26, t1888, 0)
        del input26

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1890 = paddle._C_ops.reshape(t1889, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1891 = paddle._C_ops.transpose(t1890, [2, 0, 1])
        del t1890

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1892 = paddle._C_ops.unsqueeze(t1891, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1893 = paddle._C_ops.add(t1887, t1892)

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t1894 = paddle._C_ops.softmax(t1893, -1)
        del t1893

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t1895 = paddle._C_ops.matmul(t1894, t1884, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t1896 = paddle._C_ops.transpose(t1895, [0, 2, 1, 3])
        del t1895

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1897 = [t1875, t441, t811]
        del t1875

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1898 = paddle._C_ops.stack(t1897, 0)
        del t1897

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t1899 = paddle._C_ops.reshape(t1896, t1898)
        del t1898

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t1900 = paddle._C_ops.matmul(t1899, t158, False, False)
        del t158

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t1901 = paddle._C_ops.add(t1900, t159)
        del t159

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t1902 = paddle._C_ops.reshape(t1901, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t1903 = paddle._C_ops.reshape(t1902, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t1904 = paddle._C_ops.transpose(t1903, [0, 1, 3, 2, 4, 5])
        del t1903

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t1905 = paddle._C_ops.reshape(t1904, t1111)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1906 = [t1859, t1113, t811]
        del t1859

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1907 = paddle._C_ops.stack(t1906, 0)
        del t1906

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t1908 = paddle._C_ops.reshape(t1905, t1907)
        del t1907

        # pd_op.full: (xf32) <- ()
        t1909 = paddle._C_ops.full(
            [],
            float("0.947826"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t1910 = t1909

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1911 = paddle._C_ops.shape64(t1908)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1912 = paddle._C_ops.slice(t1911, [0], t305, t354, [1], [0])
        del t1911

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1913 = [t1912, t744, t744]
        del t1912

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1914 = paddle._C_ops.stack(t1913, 0)
        del t1913

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1915 = paddle._C_ops.uniform(
            t1914,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1914

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1916 = paddle._C_ops.add(t1909, t1915)
        del t1915

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1917 = paddle._C_ops.floor(t1916)
        del t1916

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1918 = paddle._C_ops.divide(t1908, t1909)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1919 = paddle._C_ops.multiply(t1918, t1917)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1920 = paddle._C_ops.add(t1857, t1919)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1921, t1922, t1923 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1920, t160, t161, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t161, t160

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t1924 = paddle._C_ops.matmul(t1921, t162, False, False)
        del t162

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t1925 = paddle._C_ops.add(t1924, t163)
        del t163

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t1926 = paddle._C_ops.gelu(t1925, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t1927 = paddle._C_ops.matmul(t1926, t164, False, False)
        del t164

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t1928 = paddle._C_ops.add(t1927, t165)
        del t165

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1929 = paddle._C_ops.shape64(t1928)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1930 = paddle._C_ops.slice(t1929, [0], t305, t354, [1], [0])
        del t1929

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t1931 = [t1930, t744, t744]
        del t1930

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t1932 = paddle._C_ops.stack(t1931, 0)
        del t1931

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t1933 = paddle._C_ops.uniform(
            t1932,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t1932

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t1934 = paddle._C_ops.add(t1909, t1933)
        del t1933

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t1935 = paddle._C_ops.floor(t1934)
        del t1934

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t1936 = paddle._C_ops.divide(t1928, t1909)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t1937 = paddle._C_ops.multiply(t1936, t1935)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t1938 = paddle._C_ops.add(t1920, t1937)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t1939 = paddle._C_ops.shape64(t1938)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1940 = paddle._C_ops.slice(t1939, [0], t305, t354, [1], [0])
        del t1939

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t1941, t1942, t1943 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1938, t166, t167, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t167, t166

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t1944 = [t1940, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t1945 = paddle._C_ops.stack(t1944, 0)
        del t1944

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t1946 = paddle._C_ops.reshape(t1941, t1945)
        del t1945

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1947 = paddle._C_ops.shape64(t1946)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1948 = paddle._C_ops.slice(t1947, [0], t305, t354, [1], [0])
        del t1947

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t1949 = paddle._C_ops.roll(t1946, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t1950 = paddle._C_ops.shape64(t1949)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t1951 = paddle._C_ops.slice(t1950, [0], t305, t354, [1], [0])
        del t1950

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t1952 = [t1951, t1069, t428, t1069, t428, t811]
        del t1951

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t1953 = paddle._C_ops.stack(t1952, 0)
        del t1952

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t1954 = paddle._C_ops.reshape(t1949, t1953)
        del t1953

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t1955 = paddle._C_ops.transpose(t1954, [0, 1, 3, 2, 4, 5])
        del t1954

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t1956 = paddle._C_ops.reshape(t1955, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t1957 = paddle._C_ops.reshape(t1956, t1076)

        # pd_op.full: (1x14x14x1xf32) <- ()
        t1958 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1959 = paddle._C_ops.set_value_(
            t1958, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t1958

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1960 = paddle._C_ops.set_value_(
            t1959, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t1959

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1961 = paddle._C_ops.set_value_(
            t1960, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t1960

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1962 = paddle._C_ops.set_value_(
            t1961, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t1961

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1963 = paddle._C_ops.set_value_(
            t1962, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t1962

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1964 = paddle._C_ops.set_value_(
            t1963, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t1963

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1965 = paddle._C_ops.set_value_(
            t1964, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t1964

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1966 = paddle._C_ops.set_value_(
            t1965, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t1965

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t1967 = paddle._C_ops.set_value_(
            t1966, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t1966

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        t1968 = paddle._C_ops.reshape(t1967, t1176)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        t1969 = paddle._C_ops.transpose(t1968, [0, 1, 3, 2, 4, 5])
        del t1968

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        t1970 = paddle._C_ops.reshape(t1969, t666)
        del t1969

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        t1971 = paddle._C_ops.reshape(t1970, t668)
        del t1970

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        t1972 = paddle._C_ops.unsqueeze(t1971, t354)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        t1973 = paddle._C_ops.unsqueeze(t1971, t450)
        del t1971

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        t1974 = paddle._C_ops.subtract(t1972, t1973)
        del t1972, t1973

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1975 = paddle._C_ops.not_equal(t1974, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1976 = paddle._C_ops.where(t1975, t1185, t1974)
        del t1975, t1974

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t1977 = paddle._C_ops.equal(t1976, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t1978 = paddle._C_ops.where(t1977, t1188, t1976)
        del t1977, t1976

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t1979 = paddle._C_ops.shape64(t1957)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t1980 = paddle._C_ops.slice(t1979, [0], t305, t354, [1], [0])
        del t1979

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t1981 = paddle._C_ops.matmul(t1957, t168, False, False)
        del t168

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t1982 = paddle._C_ops.add(t1981, t169)
        del t169

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t1983 = [t1980, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t1984 = paddle._C_ops.stack(t1983, 0)
        del t1983

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t1985 = paddle._C_ops.reshape(t1982, t1984)
        del t1984

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t1986 = paddle._C_ops.transpose(t1985, [2, 0, 3, 1, 4])
        del t1985

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1987 = paddle._C_ops.slice(t1986, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1988 = paddle._C_ops.slice(t1986, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t1989 = paddle._C_ops.slice(t1986, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t1990 = paddle._C_ops.scale(t1987, t526, float("0"), True)
        del t1987

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t1991 = paddle._C_ops.transpose(t1988, [0, 1, 3, 2])
        del t1988

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t1992 = paddle._C_ops.matmul(t1990, t1991, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t1993 = paddle._C_ops.reshape(input27, t553)
        del input27

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t1994 = paddle._C_ops.index_select(input28, t1993, 0)
        del input28

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t1995 = paddle._C_ops.reshape(t1994, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t1996 = paddle._C_ops.transpose(t1995, [2, 0, 1])
        del t1995

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t1997 = paddle._C_ops.unsqueeze(t1996, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t1998 = paddle._C_ops.add(t1992, t1997)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t1999 = paddle._C_ops.floor_divide(t1980, t1210)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2000 = [t1999, t833, t1082, t441, t441]
        del t1999

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2001 = paddle._C_ops.stack(t2000, 0)
        del t2000

        # pd_op.reshape: (-1x4x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        t2002 = paddle._C_ops.reshape(t1998, t2001)
        del t2001

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        t2003 = paddle._C_ops.unsqueeze(t1978, t354)
        del t1978

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        t2004 = paddle._C_ops.unsqueeze(t2003, t305)
        del t2003

        # pd_op.add: (-1x4x24x49x49xf32) <- (-1x4x24x49x49xf32, 1x4x1x49x49xf32)
        t2005 = paddle._C_ops.add(t2002, t2004)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2006 = [t1980, t1082, t441, t441]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2007 = paddle._C_ops.stack(t2006, 0)
        del t2006

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x4x24x49x49xf32, 4xi64)
        t2008 = paddle._C_ops.reshape(t2005, t2007)
        del t2007

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t2009 = paddle._C_ops.softmax(t2008, -1)
        del t2008

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t2010 = paddle._C_ops.matmul(t2009, t1989, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t2011 = paddle._C_ops.transpose(t2010, [0, 2, 1, 3])
        del t2010

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2012 = [t1980, t441, t811]
        del t1980

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2013 = paddle._C_ops.stack(t2012, 0)
        del t2012

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t2014 = paddle._C_ops.reshape(t2011, t2013)
        del t2013

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t2015 = paddle._C_ops.matmul(t2014, t170, False, False)
        del t170

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t2016 = paddle._C_ops.add(t2015, t171)
        del t171

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t2017 = paddle._C_ops.reshape(t2016, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t2018 = paddle._C_ops.reshape(t2017, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t2019 = paddle._C_ops.transpose(t2018, [0, 1, 3, 2, 4, 5])
        del t2018

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t2020 = paddle._C_ops.reshape(t2019, t1111)

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t2021 = paddle._C_ops.roll(t2020, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2022 = [t1940, t1113, t811]
        del t1940

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2023 = paddle._C_ops.stack(t2022, 0)
        del t2022

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t2024 = paddle._C_ops.reshape(t2021, t2023)
        del t2023

        # pd_op.full: (xf32) <- ()
        t2025 = paddle._C_ops.full(
            [],
            float("0.943478"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2026 = t2025

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2027 = paddle._C_ops.shape64(t2024)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2028 = paddle._C_ops.slice(t2027, [0], t305, t354, [1], [0])
        del t2027

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2029 = [t2028, t744, t744]
        del t2028

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2030 = paddle._C_ops.stack(t2029, 0)
        del t2029

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2031 = paddle._C_ops.uniform(
            t2030,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2030

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2032 = paddle._C_ops.add(t2025, t2031)
        del t2031

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2033 = paddle._C_ops.floor(t2032)
        del t2032

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2034 = paddle._C_ops.divide(t2024, t2025)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2035 = paddle._C_ops.multiply(t2034, t2033)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2036 = paddle._C_ops.add(t1938, t2035)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2037, t2038, t2039 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2036, t172, t173, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t173, t172

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t2040 = paddle._C_ops.matmul(t2037, t174, False, False)
        del t174

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t2041 = paddle._C_ops.add(t2040, t175)
        del t175

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t2042 = paddle._C_ops.gelu(t2041, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t2043 = paddle._C_ops.matmul(t2042, t176, False, False)
        del t176

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t2044 = paddle._C_ops.add(t2043, t177)
        del t177

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2045 = paddle._C_ops.shape64(t2044)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2046 = paddle._C_ops.slice(t2045, [0], t305, t354, [1], [0])
        del t2045

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2047 = [t2046, t744, t744]
        del t2046

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2048 = paddle._C_ops.stack(t2047, 0)
        del t2047

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2049 = paddle._C_ops.uniform(
            t2048,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2048

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2050 = paddle._C_ops.add(t2025, t2049)
        del t2049

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2051 = paddle._C_ops.floor(t2050)
        del t2050

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2052 = paddle._C_ops.divide(t2044, t2025)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2053 = paddle._C_ops.multiply(t2052, t2051)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2054 = paddle._C_ops.add(t2036, t2053)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2055 = paddle._C_ops.shape64(t2054)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2056 = paddle._C_ops.slice(t2055, [0], t305, t354, [1], [0])
        del t2055

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2057, t2058, t2059 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2054, t178, t179, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t179, t178

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2060 = [t2056, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2061 = paddle._C_ops.stack(t2060, 0)
        del t2060

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t2062 = paddle._C_ops.reshape(t2057, t2061)
        del t2061

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2063 = paddle._C_ops.shape64(t2062)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2064 = paddle._C_ops.slice(t2063, [0], t305, t354, [1], [0])
        del t2063

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2065 = [t2064, t1069, t428, t1069, t428, t811]
        del t2064

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2066 = paddle._C_ops.stack(t2065, 0)
        del t2065

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t2067 = paddle._C_ops.reshape(t2062, t2066)
        del t2066

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t2068 = paddle._C_ops.transpose(t2067, [0, 1, 3, 2, 4, 5])
        del t2067

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t2069 = paddle._C_ops.reshape(t2068, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t2070 = paddle._C_ops.reshape(t2069, t1076)

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t2071 = paddle._C_ops.shape64(t2070)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2072 = paddle._C_ops.slice(t2071, [0], t305, t354, [1], [0])
        del t2071

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t2073 = paddle._C_ops.matmul(t2070, t180, False, False)
        del t180

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t2074 = paddle._C_ops.add(t2073, t181)
        del t181

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2075 = [t2072, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2076 = paddle._C_ops.stack(t2075, 0)
        del t2075

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t2077 = paddle._C_ops.reshape(t2074, t2076)
        del t2076

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t2078 = paddle._C_ops.transpose(t2077, [2, 0, 3, 1, 4])
        del t2077

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2079 = paddle._C_ops.slice(t2078, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2080 = paddle._C_ops.slice(t2078, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2081 = paddle._C_ops.slice(t2078, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t2082 = paddle._C_ops.scale(t2079, t526, float("0"), True)
        del t2079

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t2083 = paddle._C_ops.transpose(t2080, [0, 1, 3, 2])
        del t2080

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t2084 = paddle._C_ops.matmul(t2082, t2083, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t2085 = paddle._C_ops.reshape(input29, t553)
        del input29

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t2086 = paddle._C_ops.index_select(input30, t2085, 0)
        del input30

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t2087 = paddle._C_ops.reshape(t2086, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t2088 = paddle._C_ops.transpose(t2087, [2, 0, 1])
        del t2087

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t2089 = paddle._C_ops.unsqueeze(t2088, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t2090 = paddle._C_ops.add(t2084, t2089)

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t2091 = paddle._C_ops.softmax(t2090, -1)
        del t2090

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t2092 = paddle._C_ops.matmul(t2091, t2081, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t2093 = paddle._C_ops.transpose(t2092, [0, 2, 1, 3])
        del t2092

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2094 = [t2072, t441, t811]
        del t2072

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2095 = paddle._C_ops.stack(t2094, 0)
        del t2094

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t2096 = paddle._C_ops.reshape(t2093, t2095)
        del t2095

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t2097 = paddle._C_ops.matmul(t2096, t182, False, False)
        del t182

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t2098 = paddle._C_ops.add(t2097, t183)
        del t183

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t2099 = paddle._C_ops.reshape(t2098, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t2100 = paddle._C_ops.reshape(t2099, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t2101 = paddle._C_ops.transpose(t2100, [0, 1, 3, 2, 4, 5])
        del t2100

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t2102 = paddle._C_ops.reshape(t2101, t1111)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2103 = [t2056, t1113, t811]
        del t2056

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2104 = paddle._C_ops.stack(t2103, 0)
        del t2103

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t2105 = paddle._C_ops.reshape(t2102, t2104)
        del t2104

        # pd_op.full: (xf32) <- ()
        t2106 = paddle._C_ops.full(
            [],
            float("0.93913"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2107 = t2106

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2108 = paddle._C_ops.shape64(t2105)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2109 = paddle._C_ops.slice(t2108, [0], t305, t354, [1], [0])
        del t2108

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2110 = [t2109, t744, t744]
        del t2109

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2111 = paddle._C_ops.stack(t2110, 0)
        del t2110

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2112 = paddle._C_ops.uniform(
            t2111,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2111

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2113 = paddle._C_ops.add(t2106, t2112)
        del t2112

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2114 = paddle._C_ops.floor(t2113)
        del t2113

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2115 = paddle._C_ops.divide(t2105, t2106)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2116 = paddle._C_ops.multiply(t2115, t2114)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2117 = paddle._C_ops.add(t2054, t2116)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2118, t2119, t2120 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2117, t184, t185, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t185, t184

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t2121 = paddle._C_ops.matmul(t2118, t186, False, False)
        del t186

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t2122 = paddle._C_ops.add(t2121, t187)
        del t187

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t2123 = paddle._C_ops.gelu(t2122, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t2124 = paddle._C_ops.matmul(t2123, t188, False, False)
        del t188

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t2125 = paddle._C_ops.add(t2124, t189)
        del t189

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2126 = paddle._C_ops.shape64(t2125)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2127 = paddle._C_ops.slice(t2126, [0], t305, t354, [1], [0])
        del t2126

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2128 = [t2127, t744, t744]
        del t2127

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2129 = paddle._C_ops.stack(t2128, 0)
        del t2128

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2130 = paddle._C_ops.uniform(
            t2129,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2129

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2131 = paddle._C_ops.add(t2106, t2130)
        del t2130

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2132 = paddle._C_ops.floor(t2131)
        del t2131

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2133 = paddle._C_ops.divide(t2125, t2106)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2134 = paddle._C_ops.multiply(t2133, t2132)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2135 = paddle._C_ops.add(t2117, t2134)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2136 = paddle._C_ops.shape64(t2135)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2137 = paddle._C_ops.slice(t2136, [0], t305, t354, [1], [0])
        del t2136

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2138, t2139, t2140 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2135, t190, t191, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t191, t190

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2141 = [t2137, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2142 = paddle._C_ops.stack(t2141, 0)
        del t2141

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t2143 = paddle._C_ops.reshape(t2138, t2142)
        del t2142

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2144 = paddle._C_ops.shape64(t2143)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2145 = paddle._C_ops.slice(t2144, [0], t305, t354, [1], [0])
        del t2144

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t2146 = paddle._C_ops.roll(t2143, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2147 = paddle._C_ops.shape64(t2146)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2148 = paddle._C_ops.slice(t2147, [0], t305, t354, [1], [0])
        del t2147

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2149 = [t2148, t1069, t428, t1069, t428, t811]
        del t2148

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2150 = paddle._C_ops.stack(t2149, 0)
        del t2149

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t2151 = paddle._C_ops.reshape(t2146, t2150)
        del t2150

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t2152 = paddle._C_ops.transpose(t2151, [0, 1, 3, 2, 4, 5])
        del t2151

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t2153 = paddle._C_ops.reshape(t2152, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t2154 = paddle._C_ops.reshape(t2153, t1076)

        # pd_op.full: (1x14x14x1xf32) <- ()
        t2155 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2156 = paddle._C_ops.set_value_(
            t2155, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t2155

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2157 = paddle._C_ops.set_value_(
            t2156, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t2156

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2158 = paddle._C_ops.set_value_(
            t2157, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t2157

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2159 = paddle._C_ops.set_value_(
            t2158, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t2158

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2160 = paddle._C_ops.set_value_(
            t2159, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t2159

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2161 = paddle._C_ops.set_value_(
            t2160, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t2160

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2162 = paddle._C_ops.set_value_(
            t2161, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t2161

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2163 = paddle._C_ops.set_value_(
            t2162, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t2162

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2164 = paddle._C_ops.set_value_(
            t2163, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t2163

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        t2165 = paddle._C_ops.reshape(t2164, t1176)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        t2166 = paddle._C_ops.transpose(t2165, [0, 1, 3, 2, 4, 5])
        del t2165

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        t2167 = paddle._C_ops.reshape(t2166, t666)
        del t2166

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        t2168 = paddle._C_ops.reshape(t2167, t668)
        del t2167

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        t2169 = paddle._C_ops.unsqueeze(t2168, t354)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        t2170 = paddle._C_ops.unsqueeze(t2168, t450)
        del t2168

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        t2171 = paddle._C_ops.subtract(t2169, t2170)
        del t2169, t2170

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t2172 = paddle._C_ops.not_equal(t2171, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t2173 = paddle._C_ops.where(t2172, t1185, t2171)
        del t2172, t2171

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t2174 = paddle._C_ops.equal(t2173, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t2175 = paddle._C_ops.where(t2174, t1188, t2173)
        del t2174, t2173

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t2176 = paddle._C_ops.shape64(t2154)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2177 = paddle._C_ops.slice(t2176, [0], t305, t354, [1], [0])
        del t2176

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t2178 = paddle._C_ops.matmul(t2154, t192, False, False)
        del t192

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t2179 = paddle._C_ops.add(t2178, t193)
        del t193

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2180 = [t2177, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2181 = paddle._C_ops.stack(t2180, 0)
        del t2180

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t2182 = paddle._C_ops.reshape(t2179, t2181)
        del t2181

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t2183 = paddle._C_ops.transpose(t2182, [2, 0, 3, 1, 4])
        del t2182

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2184 = paddle._C_ops.slice(t2183, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2185 = paddle._C_ops.slice(t2183, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2186 = paddle._C_ops.slice(t2183, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t2187 = paddle._C_ops.scale(t2184, t526, float("0"), True)
        del t2184

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t2188 = paddle._C_ops.transpose(t2185, [0, 1, 3, 2])
        del t2185

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t2189 = paddle._C_ops.matmul(t2187, t2188, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t2190 = paddle._C_ops.reshape(input31, t553)
        del input31

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t2191 = paddle._C_ops.index_select(input32, t2190, 0)
        del input32

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t2192 = paddle._C_ops.reshape(t2191, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t2193 = paddle._C_ops.transpose(t2192, [2, 0, 1])
        del t2192

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t2194 = paddle._C_ops.unsqueeze(t2193, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t2195 = paddle._C_ops.add(t2189, t2194)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t2196 = paddle._C_ops.floor_divide(t2177, t1210)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2197 = [t2196, t833, t1082, t441, t441]
        del t2196

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2198 = paddle._C_ops.stack(t2197, 0)
        del t2197

        # pd_op.reshape: (-1x4x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        t2199 = paddle._C_ops.reshape(t2195, t2198)
        del t2198

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        t2200 = paddle._C_ops.unsqueeze(t2175, t354)
        del t2175

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        t2201 = paddle._C_ops.unsqueeze(t2200, t305)
        del t2200

        # pd_op.add: (-1x4x24x49x49xf32) <- (-1x4x24x49x49xf32, 1x4x1x49x49xf32)
        t2202 = paddle._C_ops.add(t2199, t2201)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2203 = [t2177, t1082, t441, t441]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2204 = paddle._C_ops.stack(t2203, 0)
        del t2203

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x4x24x49x49xf32, 4xi64)
        t2205 = paddle._C_ops.reshape(t2202, t2204)
        del t2204

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t2206 = paddle._C_ops.softmax(t2205, -1)
        del t2205

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t2207 = paddle._C_ops.matmul(t2206, t2186, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t2208 = paddle._C_ops.transpose(t2207, [0, 2, 1, 3])
        del t2207

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2209 = [t2177, t441, t811]
        del t2177

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2210 = paddle._C_ops.stack(t2209, 0)
        del t2209

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t2211 = paddle._C_ops.reshape(t2208, t2210)
        del t2210

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t2212 = paddle._C_ops.matmul(t2211, t194, False, False)
        del t194

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t2213 = paddle._C_ops.add(t2212, t195)
        del t195

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t2214 = paddle._C_ops.reshape(t2213, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t2215 = paddle._C_ops.reshape(t2214, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t2216 = paddle._C_ops.transpose(t2215, [0, 1, 3, 2, 4, 5])
        del t2215

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t2217 = paddle._C_ops.reshape(t2216, t1111)

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t2218 = paddle._C_ops.roll(t2217, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2219 = [t2137, t1113, t811]
        del t2137

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2220 = paddle._C_ops.stack(t2219, 0)
        del t2219

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t2221 = paddle._C_ops.reshape(t2218, t2220)
        del t2220

        # pd_op.full: (xf32) <- ()
        t2222 = paddle._C_ops.full(
            [],
            float("0.934783"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2223 = t2222

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2224 = paddle._C_ops.shape64(t2221)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2225 = paddle._C_ops.slice(t2224, [0], t305, t354, [1], [0])
        del t2224

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2226 = [t2225, t744, t744]
        del t2225

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2227 = paddle._C_ops.stack(t2226, 0)
        del t2226

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2228 = paddle._C_ops.uniform(
            t2227,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2227

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2229 = paddle._C_ops.add(t2222, t2228)
        del t2228

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2230 = paddle._C_ops.floor(t2229)
        del t2229

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2231 = paddle._C_ops.divide(t2221, t2222)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2232 = paddle._C_ops.multiply(t2231, t2230)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2233 = paddle._C_ops.add(t2135, t2232)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2234, t2235, t2236 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2233, t196, t197, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t197, t196

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t2237 = paddle._C_ops.matmul(t2234, t198, False, False)
        del t198

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t2238 = paddle._C_ops.add(t2237, t199)
        del t199

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t2239 = paddle._C_ops.gelu(t2238, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t2240 = paddle._C_ops.matmul(t2239, t200, False, False)
        del t200

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t2241 = paddle._C_ops.add(t2240, t201)
        del t201

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2242 = paddle._C_ops.shape64(t2241)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2243 = paddle._C_ops.slice(t2242, [0], t305, t354, [1], [0])
        del t2242

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2244 = [t2243, t744, t744]
        del t2243

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2245 = paddle._C_ops.stack(t2244, 0)
        del t2244

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2246 = paddle._C_ops.uniform(
            t2245,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2245

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2247 = paddle._C_ops.add(t2222, t2246)
        del t2246

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2248 = paddle._C_ops.floor(t2247)
        del t2247

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2249 = paddle._C_ops.divide(t2241, t2222)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2250 = paddle._C_ops.multiply(t2249, t2248)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2251 = paddle._C_ops.add(t2233, t2250)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2252 = paddle._C_ops.shape64(t2251)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2253 = paddle._C_ops.slice(t2252, [0], t305, t354, [1], [0])
        del t2252

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2254, t2255, t2256 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2251, t202, t203, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t203, t202

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2257 = [t2253, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2258 = paddle._C_ops.stack(t2257, 0)
        del t2257

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t2259 = paddle._C_ops.reshape(t2254, t2258)
        del t2258

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2260 = paddle._C_ops.shape64(t2259)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2261 = paddle._C_ops.slice(t2260, [0], t305, t354, [1], [0])
        del t2260

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2262 = [t2261, t1069, t428, t1069, t428, t811]
        del t2261

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2263 = paddle._C_ops.stack(t2262, 0)
        del t2262

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t2264 = paddle._C_ops.reshape(t2259, t2263)
        del t2263

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t2265 = paddle._C_ops.transpose(t2264, [0, 1, 3, 2, 4, 5])
        del t2264

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t2266 = paddle._C_ops.reshape(t2265, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t2267 = paddle._C_ops.reshape(t2266, t1076)

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t2268 = paddle._C_ops.shape64(t2267)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2269 = paddle._C_ops.slice(t2268, [0], t305, t354, [1], [0])
        del t2268

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t2270 = paddle._C_ops.matmul(t2267, t204, False, False)
        del t204

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t2271 = paddle._C_ops.add(t2270, t205)
        del t205

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2272 = [t2269, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2273 = paddle._C_ops.stack(t2272, 0)
        del t2272

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t2274 = paddle._C_ops.reshape(t2271, t2273)
        del t2273

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t2275 = paddle._C_ops.transpose(t2274, [2, 0, 3, 1, 4])
        del t2274

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2276 = paddle._C_ops.slice(t2275, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2277 = paddle._C_ops.slice(t2275, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2278 = paddle._C_ops.slice(t2275, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t2279 = paddle._C_ops.scale(t2276, t526, float("0"), True)
        del t2276

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t2280 = paddle._C_ops.transpose(t2277, [0, 1, 3, 2])
        del t2277

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t2281 = paddle._C_ops.matmul(t2279, t2280, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t2282 = paddle._C_ops.reshape(input33, t553)
        del input33

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t2283 = paddle._C_ops.index_select(input34, t2282, 0)
        del input34

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t2284 = paddle._C_ops.reshape(t2283, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t2285 = paddle._C_ops.transpose(t2284, [2, 0, 1])
        del t2284

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t2286 = paddle._C_ops.unsqueeze(t2285, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t2287 = paddle._C_ops.add(t2281, t2286)

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t2288 = paddle._C_ops.softmax(t2287, -1)
        del t2287

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t2289 = paddle._C_ops.matmul(t2288, t2278, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t2290 = paddle._C_ops.transpose(t2289, [0, 2, 1, 3])
        del t2289

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2291 = [t2269, t441, t811]
        del t2269

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2292 = paddle._C_ops.stack(t2291, 0)
        del t2291

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t2293 = paddle._C_ops.reshape(t2290, t2292)
        del t2292

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t2294 = paddle._C_ops.matmul(t2293, t206, False, False)
        del t206

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t2295 = paddle._C_ops.add(t2294, t207)
        del t207

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t2296 = paddle._C_ops.reshape(t2295, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t2297 = paddle._C_ops.reshape(t2296, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t2298 = paddle._C_ops.transpose(t2297, [0, 1, 3, 2, 4, 5])
        del t2297

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t2299 = paddle._C_ops.reshape(t2298, t1111)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2300 = [t2253, t1113, t811]
        del t2253

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2301 = paddle._C_ops.stack(t2300, 0)
        del t2300

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t2302 = paddle._C_ops.reshape(t2299, t2301)
        del t2301

        # pd_op.full: (xf32) <- ()
        t2303 = paddle._C_ops.full(
            [],
            float("0.930435"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2304 = t2303

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2305 = paddle._C_ops.shape64(t2302)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2306 = paddle._C_ops.slice(t2305, [0], t305, t354, [1], [0])
        del t2305

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2307 = [t2306, t744, t744]
        del t2306

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2308 = paddle._C_ops.stack(t2307, 0)
        del t2307

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2309 = paddle._C_ops.uniform(
            t2308,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2308

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2310 = paddle._C_ops.add(t2303, t2309)
        del t2309

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2311 = paddle._C_ops.floor(t2310)
        del t2310

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2312 = paddle._C_ops.divide(t2302, t2303)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2313 = paddle._C_ops.multiply(t2312, t2311)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2314 = paddle._C_ops.add(t2251, t2313)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2315, t2316, t2317 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2314, t208, t209, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t209, t208

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t2318 = paddle._C_ops.matmul(t2315, t210, False, False)
        del t210

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t2319 = paddle._C_ops.add(t2318, t211)
        del t211

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t2320 = paddle._C_ops.gelu(t2319, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t2321 = paddle._C_ops.matmul(t2320, t212, False, False)
        del t212

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t2322 = paddle._C_ops.add(t2321, t213)
        del t213

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2323 = paddle._C_ops.shape64(t2322)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2324 = paddle._C_ops.slice(t2323, [0], t305, t354, [1], [0])
        del t2323

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2325 = [t2324, t744, t744]
        del t2324

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2326 = paddle._C_ops.stack(t2325, 0)
        del t2325

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2327 = paddle._C_ops.uniform(
            t2326,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2326

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2328 = paddle._C_ops.add(t2303, t2327)
        del t2327

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2329 = paddle._C_ops.floor(t2328)
        del t2328

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2330 = paddle._C_ops.divide(t2322, t2303)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2331 = paddle._C_ops.multiply(t2330, t2329)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2332 = paddle._C_ops.add(t2314, t2331)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2333 = paddle._C_ops.shape64(t2332)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2334 = paddle._C_ops.slice(t2333, [0], t305, t354, [1], [0])
        del t2333

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2335, t2336, t2337 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2332, t214, t215, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t215, t214

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2338 = [t2334, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2339 = paddle._C_ops.stack(t2338, 0)
        del t2338

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t2340 = paddle._C_ops.reshape(t2335, t2339)
        del t2339

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2341 = paddle._C_ops.shape64(t2340)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2342 = paddle._C_ops.slice(t2341, [0], t305, t354, [1], [0])
        del t2341

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t2343 = paddle._C_ops.roll(t2340, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2344 = paddle._C_ops.shape64(t2343)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2345 = paddle._C_ops.slice(t2344, [0], t305, t354, [1], [0])
        del t2344

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2346 = [t2345, t1069, t428, t1069, t428, t811]
        del t2345

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2347 = paddle._C_ops.stack(t2346, 0)
        del t2346

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t2348 = paddle._C_ops.reshape(t2343, t2347)
        del t2347

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t2349 = paddle._C_ops.transpose(t2348, [0, 1, 3, 2, 4, 5])
        del t2348

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t2350 = paddle._C_ops.reshape(t2349, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t2351 = paddle._C_ops.reshape(t2350, t1076)

        # pd_op.full: (1x14x14x1xf32) <- ()
        t2352 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2353 = paddle._C_ops.set_value_(
            t2352, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t2352

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2354 = paddle._C_ops.set_value_(
            t2353, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t2353

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2355 = paddle._C_ops.set_value_(
            t2354, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t2354

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2356 = paddle._C_ops.set_value_(
            t2355, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t2355

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2357 = paddle._C_ops.set_value_(
            t2356, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t2356

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2358 = paddle._C_ops.set_value_(
            t2357, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t2357

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2359 = paddle._C_ops.set_value_(
            t2358, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t2358

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2360 = paddle._C_ops.set_value_(
            t2359, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t2359

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2361 = paddle._C_ops.set_value_(
            t2360, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t2360

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        t2362 = paddle._C_ops.reshape(t2361, t1176)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        t2363 = paddle._C_ops.transpose(t2362, [0, 1, 3, 2, 4, 5])
        del t2362

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        t2364 = paddle._C_ops.reshape(t2363, t666)
        del t2363

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        t2365 = paddle._C_ops.reshape(t2364, t668)
        del t2364

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        t2366 = paddle._C_ops.unsqueeze(t2365, t354)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        t2367 = paddle._C_ops.unsqueeze(t2365, t450)
        del t2365

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        t2368 = paddle._C_ops.subtract(t2366, t2367)
        del t2366, t2367

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t2369 = paddle._C_ops.not_equal(t2368, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t2370 = paddle._C_ops.where(t2369, t1185, t2368)
        del t2369, t2368

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t2371 = paddle._C_ops.equal(t2370, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t2372 = paddle._C_ops.where(t2371, t1188, t2370)
        del t2371, t2370

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t2373 = paddle._C_ops.shape64(t2351)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2374 = paddle._C_ops.slice(t2373, [0], t305, t354, [1], [0])
        del t2373

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t2375 = paddle._C_ops.matmul(t2351, t216, False, False)
        del t216

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t2376 = paddle._C_ops.add(t2375, t217)
        del t217

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2377 = [t2374, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2378 = paddle._C_ops.stack(t2377, 0)
        del t2377

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t2379 = paddle._C_ops.reshape(t2376, t2378)
        del t2378

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t2380 = paddle._C_ops.transpose(t2379, [2, 0, 3, 1, 4])
        del t2379

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2381 = paddle._C_ops.slice(t2380, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2382 = paddle._C_ops.slice(t2380, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2383 = paddle._C_ops.slice(t2380, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t2384 = paddle._C_ops.scale(t2381, t526, float("0"), True)
        del t2381

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t2385 = paddle._C_ops.transpose(t2382, [0, 1, 3, 2])
        del t2382

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t2386 = paddle._C_ops.matmul(t2384, t2385, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t2387 = paddle._C_ops.reshape(input35, t553)
        del input35

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t2388 = paddle._C_ops.index_select(input36, t2387, 0)
        del input36

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t2389 = paddle._C_ops.reshape(t2388, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t2390 = paddle._C_ops.transpose(t2389, [2, 0, 1])
        del t2389

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t2391 = paddle._C_ops.unsqueeze(t2390, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t2392 = paddle._C_ops.add(t2386, t2391)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t2393 = paddle._C_ops.floor_divide(t2374, t1210)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2394 = [t2393, t833, t1082, t441, t441]
        del t2393

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2395 = paddle._C_ops.stack(t2394, 0)
        del t2394

        # pd_op.reshape: (-1x4x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        t2396 = paddle._C_ops.reshape(t2392, t2395)
        del t2395

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        t2397 = paddle._C_ops.unsqueeze(t2372, t354)
        del t2372

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        t2398 = paddle._C_ops.unsqueeze(t2397, t305)
        del t2397

        # pd_op.add: (-1x4x24x49x49xf32) <- (-1x4x24x49x49xf32, 1x4x1x49x49xf32)
        t2399 = paddle._C_ops.add(t2396, t2398)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2400 = [t2374, t1082, t441, t441]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2401 = paddle._C_ops.stack(t2400, 0)
        del t2400

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x4x24x49x49xf32, 4xi64)
        t2402 = paddle._C_ops.reshape(t2399, t2401)
        del t2401

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t2403 = paddle._C_ops.softmax(t2402, -1)
        del t2402

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t2404 = paddle._C_ops.matmul(t2403, t2383, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t2405 = paddle._C_ops.transpose(t2404, [0, 2, 1, 3])
        del t2404

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2406 = [t2374, t441, t811]
        del t2374

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2407 = paddle._C_ops.stack(t2406, 0)
        del t2406

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t2408 = paddle._C_ops.reshape(t2405, t2407)
        del t2407

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t2409 = paddle._C_ops.matmul(t2408, t218, False, False)
        del t218

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t2410 = paddle._C_ops.add(t2409, t219)
        del t219

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t2411 = paddle._C_ops.reshape(t2410, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t2412 = paddle._C_ops.reshape(t2411, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t2413 = paddle._C_ops.transpose(t2412, [0, 1, 3, 2, 4, 5])
        del t2412

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t2414 = paddle._C_ops.reshape(t2413, t1111)

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t2415 = paddle._C_ops.roll(t2414, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2416 = [t2334, t1113, t811]
        del t2334

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2417 = paddle._C_ops.stack(t2416, 0)
        del t2416

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t2418 = paddle._C_ops.reshape(t2415, t2417)
        del t2417

        # pd_op.full: (xf32) <- ()
        t2419 = paddle._C_ops.full(
            [],
            float("0.926087"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2420 = t2419

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2421 = paddle._C_ops.shape64(t2418)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2422 = paddle._C_ops.slice(t2421, [0], t305, t354, [1], [0])
        del t2421

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2423 = [t2422, t744, t744]
        del t2422

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2424 = paddle._C_ops.stack(t2423, 0)
        del t2423

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2425 = paddle._C_ops.uniform(
            t2424,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2424

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2426 = paddle._C_ops.add(t2419, t2425)
        del t2425

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2427 = paddle._C_ops.floor(t2426)
        del t2426

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2428 = paddle._C_ops.divide(t2418, t2419)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2429 = paddle._C_ops.multiply(t2428, t2427)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2430 = paddle._C_ops.add(t2332, t2429)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2431, t2432, t2433 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2430, t220, t221, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t221, t220

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t2434 = paddle._C_ops.matmul(t2431, t222, False, False)
        del t222

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t2435 = paddle._C_ops.add(t2434, t223)
        del t223

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t2436 = paddle._C_ops.gelu(t2435, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t2437 = paddle._C_ops.matmul(t2436, t224, False, False)
        del t224

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t2438 = paddle._C_ops.add(t2437, t225)
        del t225

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2439 = paddle._C_ops.shape64(t2438)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2440 = paddle._C_ops.slice(t2439, [0], t305, t354, [1], [0])
        del t2439

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2441 = [t2440, t744, t744]
        del t2440

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2442 = paddle._C_ops.stack(t2441, 0)
        del t2441

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2443 = paddle._C_ops.uniform(
            t2442,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2442

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2444 = paddle._C_ops.add(t2419, t2443)
        del t2443

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2445 = paddle._C_ops.floor(t2444)
        del t2444

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2446 = paddle._C_ops.divide(t2438, t2419)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2447 = paddle._C_ops.multiply(t2446, t2445)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2448 = paddle._C_ops.add(t2430, t2447)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2449 = paddle._C_ops.shape64(t2448)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2450 = paddle._C_ops.slice(t2449, [0], t305, t354, [1], [0])
        del t2449

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2451, t2452, t2453 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2448, t226, t227, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t227, t226

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2454 = [t2450, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2455 = paddle._C_ops.stack(t2454, 0)
        del t2454

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t2456 = paddle._C_ops.reshape(t2451, t2455)
        del t2455

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2457 = paddle._C_ops.shape64(t2456)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2458 = paddle._C_ops.slice(t2457, [0], t305, t354, [1], [0])
        del t2457

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2459 = [t2458, t1069, t428, t1069, t428, t811]
        del t2458

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2460 = paddle._C_ops.stack(t2459, 0)
        del t2459

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t2461 = paddle._C_ops.reshape(t2456, t2460)
        del t2460

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t2462 = paddle._C_ops.transpose(t2461, [0, 1, 3, 2, 4, 5])
        del t2461

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t2463 = paddle._C_ops.reshape(t2462, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t2464 = paddle._C_ops.reshape(t2463, t1076)

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t2465 = paddle._C_ops.shape64(t2464)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2466 = paddle._C_ops.slice(t2465, [0], t305, t354, [1], [0])
        del t2465

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t2467 = paddle._C_ops.matmul(t2464, t228, False, False)
        del t228

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t2468 = paddle._C_ops.add(t2467, t229)
        del t229

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2469 = [t2466, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2470 = paddle._C_ops.stack(t2469, 0)
        del t2469

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t2471 = paddle._C_ops.reshape(t2468, t2470)
        del t2470

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t2472 = paddle._C_ops.transpose(t2471, [2, 0, 3, 1, 4])
        del t2471

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2473 = paddle._C_ops.slice(t2472, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2474 = paddle._C_ops.slice(t2472, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2475 = paddle._C_ops.slice(t2472, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t2476 = paddle._C_ops.scale(t2473, t526, float("0"), True)
        del t2473

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t2477 = paddle._C_ops.transpose(t2474, [0, 1, 3, 2])
        del t2474

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t2478 = paddle._C_ops.matmul(t2476, t2477, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t2479 = paddle._C_ops.reshape(input37, t553)
        del input37

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t2480 = paddle._C_ops.index_select(input38, t2479, 0)
        del input38

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t2481 = paddle._C_ops.reshape(t2480, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t2482 = paddle._C_ops.transpose(t2481, [2, 0, 1])
        del t2481

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t2483 = paddle._C_ops.unsqueeze(t2482, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t2484 = paddle._C_ops.add(t2478, t2483)

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t2485 = paddle._C_ops.softmax(t2484, -1)
        del t2484

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t2486 = paddle._C_ops.matmul(t2485, t2475, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t2487 = paddle._C_ops.transpose(t2486, [0, 2, 1, 3])
        del t2486

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2488 = [t2466, t441, t811]
        del t2466

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2489 = paddle._C_ops.stack(t2488, 0)
        del t2488

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t2490 = paddle._C_ops.reshape(t2487, t2489)
        del t2489

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t2491 = paddle._C_ops.matmul(t2490, t230, False, False)
        del t230

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t2492 = paddle._C_ops.add(t2491, t231)
        del t231

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t2493 = paddle._C_ops.reshape(t2492, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t2494 = paddle._C_ops.reshape(t2493, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t2495 = paddle._C_ops.transpose(t2494, [0, 1, 3, 2, 4, 5])
        del t2494

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t2496 = paddle._C_ops.reshape(t2495, t1111)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2497 = [t2450, t1113, t811]
        del t2450

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2498 = paddle._C_ops.stack(t2497, 0)
        del t2497

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t2499 = paddle._C_ops.reshape(t2496, t2498)
        del t2498

        # pd_op.full: (xf32) <- ()
        t2500 = paddle._C_ops.full(
            [],
            float("0.921739"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2501 = t2500

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2502 = paddle._C_ops.shape64(t2499)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2503 = paddle._C_ops.slice(t2502, [0], t305, t354, [1], [0])
        del t2502

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2504 = [t2503, t744, t744]
        del t2503

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2505 = paddle._C_ops.stack(t2504, 0)
        del t2504

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2506 = paddle._C_ops.uniform(
            t2505,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2505

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2507 = paddle._C_ops.add(t2500, t2506)
        del t2506

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2508 = paddle._C_ops.floor(t2507)
        del t2507

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2509 = paddle._C_ops.divide(t2499, t2500)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2510 = paddle._C_ops.multiply(t2509, t2508)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2511 = paddle._C_ops.add(t2448, t2510)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2512, t2513, t2514 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2511, t232, t233, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t233, t232

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t2515 = paddle._C_ops.matmul(t2512, t234, False, False)
        del t234

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t2516 = paddle._C_ops.add(t2515, t235)
        del t235

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t2517 = paddle._C_ops.gelu(t2516, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t2518 = paddle._C_ops.matmul(t2517, t236, False, False)
        del t236

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t2519 = paddle._C_ops.add(t2518, t237)
        del t237

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2520 = paddle._C_ops.shape64(t2519)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2521 = paddle._C_ops.slice(t2520, [0], t305, t354, [1], [0])
        del t2520

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2522 = [t2521, t744, t744]
        del t2521

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2523 = paddle._C_ops.stack(t2522, 0)
        del t2522

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2524 = paddle._C_ops.uniform(
            t2523,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2523

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2525 = paddle._C_ops.add(t2500, t2524)
        del t2524

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2526 = paddle._C_ops.floor(t2525)
        del t2525

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2527 = paddle._C_ops.divide(t2519, t2500)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2528 = paddle._C_ops.multiply(t2527, t2526)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2529 = paddle._C_ops.add(t2511, t2528)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2530 = paddle._C_ops.shape64(t2529)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2531 = paddle._C_ops.slice(t2530, [0], t305, t354, [1], [0])
        del t2530

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2532, t2533, t2534 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2529, t238, t239, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t239, t238

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2535 = [t2531, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2536 = paddle._C_ops.stack(t2535, 0)
        del t2535

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t2537 = paddle._C_ops.reshape(t2532, t2536)
        del t2536

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2538 = paddle._C_ops.shape64(t2537)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2539 = paddle._C_ops.slice(t2538, [0], t305, t354, [1], [0])
        del t2538

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t2540 = paddle._C_ops.roll(t2537, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2541 = paddle._C_ops.shape64(t2540)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2542 = paddle._C_ops.slice(t2541, [0], t305, t354, [1], [0])
        del t2541

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2543 = [t2542, t1069, t428, t1069, t428, t811]
        del t2542

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2544 = paddle._C_ops.stack(t2543, 0)
        del t2543

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t2545 = paddle._C_ops.reshape(t2540, t2544)
        del t2544

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t2546 = paddle._C_ops.transpose(t2545, [0, 1, 3, 2, 4, 5])
        del t2545

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t2547 = paddle._C_ops.reshape(t2546, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t2548 = paddle._C_ops.reshape(t2547, t1076)

        # pd_op.full: (1x14x14x1xf32) <- ()
        t2549 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2550 = paddle._C_ops.set_value_(
            t2549, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t2549

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2551 = paddle._C_ops.set_value_(
            t2550, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t2550

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2552 = paddle._C_ops.set_value_(
            t2551, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t2551

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2553 = paddle._C_ops.set_value_(
            t2552, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t2552

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2554 = paddle._C_ops.set_value_(
            t2553, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t2553

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2555 = paddle._C_ops.set_value_(
            t2554, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t2554

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2556 = paddle._C_ops.set_value_(
            t2555, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t2555

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2557 = paddle._C_ops.set_value_(
            t2556, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t2556

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2558 = paddle._C_ops.set_value_(
            t2557, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t2557

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        t2559 = paddle._C_ops.reshape(t2558, t1176)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        t2560 = paddle._C_ops.transpose(t2559, [0, 1, 3, 2, 4, 5])
        del t2559

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        t2561 = paddle._C_ops.reshape(t2560, t666)
        del t2560

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        t2562 = paddle._C_ops.reshape(t2561, t668)
        del t2561

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        t2563 = paddle._C_ops.unsqueeze(t2562, t354)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        t2564 = paddle._C_ops.unsqueeze(t2562, t450)
        del t2562

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        t2565 = paddle._C_ops.subtract(t2563, t2564)
        del t2563, t2564

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t2566 = paddle._C_ops.not_equal(t2565, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t2567 = paddle._C_ops.where(t2566, t1185, t2565)
        del t2566, t2565

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t2568 = paddle._C_ops.equal(t2567, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t2569 = paddle._C_ops.where(t2568, t1188, t2567)
        del t2568, t2567

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t2570 = paddle._C_ops.shape64(t2548)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2571 = paddle._C_ops.slice(t2570, [0], t305, t354, [1], [0])
        del t2570

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t2572 = paddle._C_ops.matmul(t2548, t240, False, False)
        del t240

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t2573 = paddle._C_ops.add(t2572, t241)
        del t241

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2574 = [t2571, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2575 = paddle._C_ops.stack(t2574, 0)
        del t2574

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t2576 = paddle._C_ops.reshape(t2573, t2575)
        del t2575

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t2577 = paddle._C_ops.transpose(t2576, [2, 0, 3, 1, 4])
        del t2576

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2578 = paddle._C_ops.slice(t2577, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2579 = paddle._C_ops.slice(t2577, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2580 = paddle._C_ops.slice(t2577, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t2581 = paddle._C_ops.scale(t2578, t526, float("0"), True)
        del t2578

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t2582 = paddle._C_ops.transpose(t2579, [0, 1, 3, 2])
        del t2579

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t2583 = paddle._C_ops.matmul(t2581, t2582, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t2584 = paddle._C_ops.reshape(input39, t553)
        del input39

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t2585 = paddle._C_ops.index_select(input40, t2584, 0)
        del input40

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t2586 = paddle._C_ops.reshape(t2585, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t2587 = paddle._C_ops.transpose(t2586, [2, 0, 1])
        del t2586

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t2588 = paddle._C_ops.unsqueeze(t2587, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t2589 = paddle._C_ops.add(t2583, t2588)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t2590 = paddle._C_ops.floor_divide(t2571, t1210)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2591 = [t2590, t833, t1082, t441, t441]
        del t2590

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2592 = paddle._C_ops.stack(t2591, 0)
        del t2591

        # pd_op.reshape: (-1x4x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        t2593 = paddle._C_ops.reshape(t2589, t2592)
        del t2592

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        t2594 = paddle._C_ops.unsqueeze(t2569, t354)
        del t2569

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        t2595 = paddle._C_ops.unsqueeze(t2594, t305)
        del t2594

        # pd_op.add: (-1x4x24x49x49xf32) <- (-1x4x24x49x49xf32, 1x4x1x49x49xf32)
        t2596 = paddle._C_ops.add(t2593, t2595)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2597 = [t2571, t1082, t441, t441]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2598 = paddle._C_ops.stack(t2597, 0)
        del t2597

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x4x24x49x49xf32, 4xi64)
        t2599 = paddle._C_ops.reshape(t2596, t2598)
        del t2598

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t2600 = paddle._C_ops.softmax(t2599, -1)
        del t2599

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t2601 = paddle._C_ops.matmul(t2600, t2580, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t2602 = paddle._C_ops.transpose(t2601, [0, 2, 1, 3])
        del t2601

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2603 = [t2571, t441, t811]
        del t2571

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2604 = paddle._C_ops.stack(t2603, 0)
        del t2603

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t2605 = paddle._C_ops.reshape(t2602, t2604)
        del t2604

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t2606 = paddle._C_ops.matmul(t2605, t242, False, False)
        del t242

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t2607 = paddle._C_ops.add(t2606, t243)
        del t243

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t2608 = paddle._C_ops.reshape(t2607, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t2609 = paddle._C_ops.reshape(t2608, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t2610 = paddle._C_ops.transpose(t2609, [0, 1, 3, 2, 4, 5])
        del t2609

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t2611 = paddle._C_ops.reshape(t2610, t1111)

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t2612 = paddle._C_ops.roll(t2611, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2613 = [t2531, t1113, t811]
        del t2531

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2614 = paddle._C_ops.stack(t2613, 0)
        del t2613

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t2615 = paddle._C_ops.reshape(t2612, t2614)
        del t2614

        # pd_op.full: (xf32) <- ()
        t2616 = paddle._C_ops.full(
            [],
            float("0.917391"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2617 = t2616

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2618 = paddle._C_ops.shape64(t2615)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2619 = paddle._C_ops.slice(t2618, [0], t305, t354, [1], [0])
        del t2618

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2620 = [t2619, t744, t744]
        del t2619

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2621 = paddle._C_ops.stack(t2620, 0)
        del t2620

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2622 = paddle._C_ops.uniform(
            t2621,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2621

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2623 = paddle._C_ops.add(t2616, t2622)
        del t2622

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2624 = paddle._C_ops.floor(t2623)
        del t2623

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2625 = paddle._C_ops.divide(t2615, t2616)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2626 = paddle._C_ops.multiply(t2625, t2624)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2627 = paddle._C_ops.add(t2529, t2626)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2628, t2629, t2630 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2627, t244, t245, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t245, t244

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t2631 = paddle._C_ops.matmul(t2628, t246, False, False)
        del t246

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t2632 = paddle._C_ops.add(t2631, t247)
        del t247

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t2633 = paddle._C_ops.gelu(t2632, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t2634 = paddle._C_ops.matmul(t2633, t248, False, False)
        del t248

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t2635 = paddle._C_ops.add(t2634, t249)
        del t249

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2636 = paddle._C_ops.shape64(t2635)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2637 = paddle._C_ops.slice(t2636, [0], t305, t354, [1], [0])
        del t2636

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2638 = [t2637, t744, t744]
        del t2637

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2639 = paddle._C_ops.stack(t2638, 0)
        del t2638

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2640 = paddle._C_ops.uniform(
            t2639,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2639

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2641 = paddle._C_ops.add(t2616, t2640)
        del t2640

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2642 = paddle._C_ops.floor(t2641)
        del t2641

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2643 = paddle._C_ops.divide(t2635, t2616)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2644 = paddle._C_ops.multiply(t2643, t2642)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2645 = paddle._C_ops.add(t2627, t2644)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2646 = paddle._C_ops.shape64(t2645)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2647 = paddle._C_ops.slice(t2646, [0], t305, t354, [1], [0])
        del t2646

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2648, t2649, t2650 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2645, t250, t251, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t251, t250

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2651 = [t2647, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2652 = paddle._C_ops.stack(t2651, 0)
        del t2651

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t2653 = paddle._C_ops.reshape(t2648, t2652)
        del t2652

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2654 = paddle._C_ops.shape64(t2653)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2655 = paddle._C_ops.slice(t2654, [0], t305, t354, [1], [0])
        del t2654

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2656 = [t2655, t1069, t428, t1069, t428, t811]
        del t2655

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2657 = paddle._C_ops.stack(t2656, 0)
        del t2656

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t2658 = paddle._C_ops.reshape(t2653, t2657)
        del t2657

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t2659 = paddle._C_ops.transpose(t2658, [0, 1, 3, 2, 4, 5])
        del t2658

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t2660 = paddle._C_ops.reshape(t2659, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t2661 = paddle._C_ops.reshape(t2660, t1076)

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t2662 = paddle._C_ops.shape64(t2661)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2663 = paddle._C_ops.slice(t2662, [0], t305, t354, [1], [0])
        del t2662

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t2664 = paddle._C_ops.matmul(t2661, t252, False, False)
        del t252

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t2665 = paddle._C_ops.add(t2664, t253)
        del t253

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2666 = [t2663, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2667 = paddle._C_ops.stack(t2666, 0)
        del t2666

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t2668 = paddle._C_ops.reshape(t2665, t2667)
        del t2667

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t2669 = paddle._C_ops.transpose(t2668, [2, 0, 3, 1, 4])
        del t2668

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2670 = paddle._C_ops.slice(t2669, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2671 = paddle._C_ops.slice(t2669, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2672 = paddle._C_ops.slice(t2669, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t2673 = paddle._C_ops.scale(t2670, t526, float("0"), True)
        del t2670

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t2674 = paddle._C_ops.transpose(t2671, [0, 1, 3, 2])
        del t2671

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t2675 = paddle._C_ops.matmul(t2673, t2674, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t2676 = paddle._C_ops.reshape(input41, t553)
        del input41

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t2677 = paddle._C_ops.index_select(input42, t2676, 0)
        del input42

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t2678 = paddle._C_ops.reshape(t2677, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t2679 = paddle._C_ops.transpose(t2678, [2, 0, 1])
        del t2678

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t2680 = paddle._C_ops.unsqueeze(t2679, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t2681 = paddle._C_ops.add(t2675, t2680)

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t2682 = paddle._C_ops.softmax(t2681, -1)
        del t2681

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t2683 = paddle._C_ops.matmul(t2682, t2672, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t2684 = paddle._C_ops.transpose(t2683, [0, 2, 1, 3])
        del t2683

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2685 = [t2663, t441, t811]
        del t2663

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2686 = paddle._C_ops.stack(t2685, 0)
        del t2685

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t2687 = paddle._C_ops.reshape(t2684, t2686)
        del t2686

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t2688 = paddle._C_ops.matmul(t2687, t254, False, False)
        del t254

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t2689 = paddle._C_ops.add(t2688, t255)
        del t255

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t2690 = paddle._C_ops.reshape(t2689, t1074)

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t2691 = paddle._C_ops.reshape(t2690, t1108)

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t2692 = paddle._C_ops.transpose(t2691, [0, 1, 3, 2, 4, 5])
        del t2691

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t2693 = paddle._C_ops.reshape(t2692, t1111)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2694 = [t2647, t1113, t811]
        del t2647

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2695 = paddle._C_ops.stack(t2694, 0)
        del t2694

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t2696 = paddle._C_ops.reshape(t2693, t2695)
        del t2695

        # pd_op.full: (xf32) <- ()
        t2697 = paddle._C_ops.full(
            [],
            float("0.913043"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2698 = t2697

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2699 = paddle._C_ops.shape64(t2696)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2700 = paddle._C_ops.slice(t2699, [0], t305, t354, [1], [0])
        del t2699

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2701 = [t2700, t744, t744]
        del t2700

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2702 = paddle._C_ops.stack(t2701, 0)
        del t2701

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2703 = paddle._C_ops.uniform(
            t2702,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2702

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2704 = paddle._C_ops.add(t2697, t2703)
        del t2703

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2705 = paddle._C_ops.floor(t2704)
        del t2704

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2706 = paddle._C_ops.divide(t2696, t2697)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2707 = paddle._C_ops.multiply(t2706, t2705)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2708 = paddle._C_ops.add(t2645, t2707)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2709, t2710, t2711 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2708, t256, t257, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t257, t256

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t2712 = paddle._C_ops.matmul(t2709, t258, False, False)
        del t258

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t2713 = paddle._C_ops.add(t2712, t259)
        del t259

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t2714 = paddle._C_ops.gelu(t2713, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t2715 = paddle._C_ops.matmul(t2714, t260, False, False)
        del t260

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t2716 = paddle._C_ops.add(t2715, t261)
        del t261

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2717 = paddle._C_ops.shape64(t2716)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2718 = paddle._C_ops.slice(t2717, [0], t305, t354, [1], [0])
        del t2717

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2719 = [t2718, t744, t744]
        del t2718

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2720 = paddle._C_ops.stack(t2719, 0)
        del t2719

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2721 = paddle._C_ops.uniform(
            t2720,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2720

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2722 = paddle._C_ops.add(t2697, t2721)
        del t2721

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2723 = paddle._C_ops.floor(t2722)
        del t2722

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2724 = paddle._C_ops.divide(t2716, t2697)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2725 = paddle._C_ops.multiply(t2724, t2723)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2726 = paddle._C_ops.add(t2708, t2725)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2727 = paddle._C_ops.shape64(t2726)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2728 = paddle._C_ops.slice(t2727, [0], t305, t354, [1], [0])
        del t2727

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2729, t2730, t2731 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2726, t262, t263, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t263, t262

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2732 = [t2728, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2733 = paddle._C_ops.stack(t2732, 0)
        del t2732

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t2734 = paddle._C_ops.reshape(t2729, t2733)
        del t2733

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2735 = paddle._C_ops.shape64(t2734)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2736 = paddle._C_ops.slice(t2735, [0], t305, t354, [1], [0])
        del t2735

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t2737 = paddle._C_ops.roll(t2734, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2738 = paddle._C_ops.shape64(t2737)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2739 = paddle._C_ops.slice(t2738, [0], t305, t354, [1], [0])
        del t2738

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2740 = [t2739, t1069, t428, t1069, t428, t811]
        del t1069, t2739

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2741 = paddle._C_ops.stack(t2740, 0)
        del t2740

        # pd_op.reshape: (-1x2x7x2x7x768xf32) <- (-1x14x14x768xf32, 6xi64)
        t2742 = paddle._C_ops.reshape(t2737, t2741)
        del t2741

        # pd_op.transpose: (-1x2x2x7x7x768xf32) <- (-1x2x7x2x7x768xf32)
        t2743 = paddle._C_ops.transpose(t2742, [0, 1, 3, 2, 4, 5])
        del t2742

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x2x2x7x7x768xf32, 4xi64)
        t2744 = paddle._C_ops.reshape(t2743, t1074)

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        t2745 = paddle._C_ops.reshape(t2744, t1076)
        del t1076

        # pd_op.full: (1x14x14x1xf32) <- ()
        t2746 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2747 = paddle._C_ops.set_value_(
            t2746, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t2746

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2748 = paddle._C_ops.set_value_(
            t2747, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t2747

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2749 = paddle._C_ops.set_value_(
            t2748, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t2748

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2750 = paddle._C_ops.set_value_(
            t2749, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t2749

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2751 = paddle._C_ops.set_value_(
            t2750, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t2750

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2752 = paddle._C_ops.set_value_(
            t2751, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t2751

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2753 = paddle._C_ops.set_value_(
            t2752, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t2752

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2754 = paddle._C_ops.set_value_(
            t2753, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t2753

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        t2755 = paddle._C_ops.set_value_(
            t2754, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t2754

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        t2756 = paddle._C_ops.reshape(t2755, t1176)
        del t1176

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        t2757 = paddle._C_ops.transpose(t2756, [0, 1, 3, 2, 4, 5])
        del t2756

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        t2758 = paddle._C_ops.reshape(t2757, t666)
        del t2757

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        t2759 = paddle._C_ops.reshape(t2758, t668)
        del t2758

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        t2760 = paddle._C_ops.unsqueeze(t2759, t354)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        t2761 = paddle._C_ops.unsqueeze(t2759, t450)
        del t2759

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        t2762 = paddle._C_ops.subtract(t2760, t2761)
        del t2760, t2761

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t2763 = paddle._C_ops.not_equal(t2762, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t2764 = paddle._C_ops.where(t2763, t1185, t2762)
        del t1185, t2763, t2762

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        t2765 = paddle._C_ops.equal(t2764, t673)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        t2766 = paddle._C_ops.where(t2765, t1188, t2764)
        del t2765, t1188, t2764

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        t2767 = paddle._C_ops.shape64(t2745)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2768 = paddle._C_ops.slice(t2767, [0], t305, t354, [1], [0])
        del t2767

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        t2769 = paddle._C_ops.matmul(t2745, t264, False, False)
        del t264

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        t2770 = paddle._C_ops.add(t2769, t265)
        del t265

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2771 = [t2768, t441, t442, t1082, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2772 = paddle._C_ops.stack(t2771, 0)
        del t2771

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        t2773 = paddle._C_ops.reshape(t2770, t2772)
        del t2772

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        t2774 = paddle._C_ops.transpose(t2773, [2, 0, 3, 1, 4])
        del t2773

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2775 = paddle._C_ops.slice(t2774, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2776 = paddle._C_ops.slice(t2774, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        t2777 = paddle._C_ops.slice(t2774, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        t2778 = paddle._C_ops.scale(t2775, t526, float("0"), True)
        del t2775

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        t2779 = paddle._C_ops.transpose(t2776, [0, 1, 3, 2])
        del t2776

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        t2780 = paddle._C_ops.matmul(t2778, t2779, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t2781 = paddle._C_ops.reshape(input43, t553)
        del input43

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        t2782 = paddle._C_ops.index_select(input44, t2781, 0)
        del input44

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        t2783 = paddle._C_ops.reshape(t2782, t556)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        t2784 = paddle._C_ops.transpose(t2783, [2, 0, 1])
        del t2783

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        t2785 = paddle._C_ops.unsqueeze(t2784, t305)

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        t2786 = paddle._C_ops.add(t2780, t2785)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t2787 = paddle._C_ops.floor_divide(t2768, t1210)
        del t1210

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2788 = [t2787, t833, t1082, t441, t441]
        del t2787, t833

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2789 = paddle._C_ops.stack(t2788, 0)
        del t2788

        # pd_op.reshape: (-1x4x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        t2790 = paddle._C_ops.reshape(t2786, t2789)
        del t2789

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        t2791 = paddle._C_ops.unsqueeze(t2766, t354)
        del t2766

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        t2792 = paddle._C_ops.unsqueeze(t2791, t305)
        del t2791

        # pd_op.add: (-1x4x24x49x49xf32) <- (-1x4x24x49x49xf32, 1x4x1x49x49xf32)
        t2793 = paddle._C_ops.add(t2790, t2792)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2794 = [t2768, t1082, t441, t441]
        del t1082

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2795 = paddle._C_ops.stack(t2794, 0)
        del t2794

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x4x24x49x49xf32, 4xi64)
        t2796 = paddle._C_ops.reshape(t2793, t2795)
        del t2795

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        t2797 = paddle._C_ops.softmax(t2796, -1)
        del t2796

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        t2798 = paddle._C_ops.matmul(t2797, t2777, False, False)

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        t2799 = paddle._C_ops.transpose(t2798, [0, 2, 1, 3])
        del t2798

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2800 = [t2768, t441, t811]
        del t2768

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2801 = paddle._C_ops.stack(t2800, 0)
        del t2800

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        t2802 = paddle._C_ops.reshape(t2799, t2801)
        del t2801

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        t2803 = paddle._C_ops.matmul(t2802, t266, False, False)
        del t266

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        t2804 = paddle._C_ops.add(t2803, t267)
        del t267

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        t2805 = paddle._C_ops.reshape(t2804, t1074)
        del t1074

        # pd_op.reshape: (-1x2x2x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        t2806 = paddle._C_ops.reshape(t2805, t1108)
        del t1108

        # pd_op.transpose: (-1x2x7x2x7x768xf32) <- (-1x2x2x7x7x768xf32)
        t2807 = paddle._C_ops.transpose(t2806, [0, 1, 3, 2, 4, 5])
        del t2806

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x2x7x2x7x768xf32, 4xi64)
        t2808 = paddle._C_ops.reshape(t2807, t1111)
        del t1111

        # pd_op.roll: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 2xi64)
        t2809 = paddle._C_ops.roll(t2808, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2810 = [t2728, t1113, t811]
        del t1113, t2728

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2811 = paddle._C_ops.stack(t2810, 0)
        del t2810

        # pd_op.reshape: (-1x196x768xf32) <- (-1x14x14x768xf32, 3xi64)
        t2812 = paddle._C_ops.reshape(t2809, t2811)
        del t2811

        # pd_op.full: (xf32) <- ()
        t2813 = paddle._C_ops.full(
            [],
            float("0.908696"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2814 = t2813

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2815 = paddle._C_ops.shape64(t2812)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2816 = paddle._C_ops.slice(t2815, [0], t305, t354, [1], [0])
        del t2815

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2817 = [t2816, t744, t744]
        del t2816

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2818 = paddle._C_ops.stack(t2817, 0)
        del t2817

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2819 = paddle._C_ops.uniform(
            t2818,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2818

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2820 = paddle._C_ops.add(t2813, t2819)
        del t2819

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2821 = paddle._C_ops.floor(t2820)
        del t2820

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2822 = paddle._C_ops.divide(t2812, t2813)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2823 = paddle._C_ops.multiply(t2822, t2821)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2824 = paddle._C_ops.add(t2726, t2823)

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        t2825, t2826, t2827 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2824, t268, t269, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t269, t268

        # pd_op.matmul: (-1x196x3072xf32) <- (-1x196x768xf32, 768x3072xf32)
        t2828 = paddle._C_ops.matmul(t2825, t270, False, False)
        del t270

        # pd_op.add: (-1x196x3072xf32) <- (-1x196x3072xf32, 3072xf32)
        t2829 = paddle._C_ops.add(t2828, t271)
        del t271

        # pd_op.gelu: (-1x196x3072xf32) <- (-1x196x3072xf32)
        t2830 = paddle._C_ops.gelu(t2829, False)

        # pd_op.matmul: (-1x196x768xf32) <- (-1x196x3072xf32, 3072x768xf32)
        t2831 = paddle._C_ops.matmul(t2830, t272, False, False)
        del t272

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, 768xf32)
        t2832 = paddle._C_ops.add(t2831, t273)
        del t273

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2833 = paddle._C_ops.shape64(t2832)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2834 = paddle._C_ops.slice(t2833, [0], t305, t354, [1], [0])
        del t2833

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2835 = [t2834, t744, t744]
        del t2834

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2836 = paddle._C_ops.stack(t2835, 0)
        del t2835

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2837 = paddle._C_ops.uniform(
            t2836,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2836

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2838 = paddle._C_ops.add(t2813, t2837)
        del t2837

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2839 = paddle._C_ops.floor(t2838)
        del t2838

        # pd_op.divide: (-1x196x768xf32) <- (-1x196x768xf32, xf32)
        t2840 = paddle._C_ops.divide(t2832, t2813)

        # pd_op.multiply: (-1x196x768xf32) <- (-1x196x768xf32, -1x1x1xf32)
        t2841 = paddle._C_ops.multiply(t2840, t2839)

        # pd_op.add: (-1x196x768xf32) <- (-1x196x768xf32, -1x196x768xf32)
        t2842 = paddle._C_ops.add(t2824, t2841)

        # pd_op.shape64: (3xi64) <- (-1x196x768xf32)
        t2843 = paddle._C_ops.shape64(t2842)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2844 = paddle._C_ops.slice(t2843, [0], t305, t354, [1], [0])
        del t2843

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2845 = [t2844, t1063, t1063, t811]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2846 = paddle._C_ops.stack(t2845, 0)
        del t2845

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x196x768xf32, 4xi64)
        t2847 = paddle._C_ops.reshape(t2842, t2846)
        del t2846

        # pd_op.strided_slice: (-1x7x7x768xf32) <- (-1x14x14x768xf32, 2xi64, 2xi64, 2xi64)
        t2848 = paddle._C_ops.strided_slice(t2847, [1, 2], t621, t649, t778)

        # pd_op.strided_slice: (-1x7x7x768xf32) <- (-1x14x14x768xf32, 2xi64, 2xi64, 2xi64)
        t2849 = paddle._C_ops.strided_slice(t2847, [1, 2], t791, t649, t778)

        # pd_op.strided_slice: (-1x7x7x768xf32) <- (-1x14x14x768xf32, 2xi64, 2xi64, 2xi64)
        t2850 = paddle._C_ops.strided_slice(t2847, [1, 2], t795, t649, t778)

        # pd_op.strided_slice: (-1x7x7x768xf32) <- (-1x14x14x768xf32, 2xi64, 2xi64, 2xi64)
        t2851 = paddle._C_ops.strided_slice(t2847, [1, 2], t626, t649, t778)

        # pd_op.shape64: (4xi64) <- (-1x14x14x768xf32)
        t2852 = paddle._C_ops.shape64(t2847)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2853 = paddle._C_ops.slice(t2852, [0], t305, t354, [1], [0])
        del t2852

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2854 = [t2853, t1063, t1063, t811]
        del t811, t1063, t2853

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2855 = paddle._C_ops.stack(t2854, 0)
        del t2854

        # pd_op.reshape: (-1x14x14x768xf32) <- (-1x14x14x768xf32, 4xi64)
        t2856 = paddle._C_ops.reshape(t2847, t2855)
        del t2855

        # builtin.combine: ([-1x7x7x768xf32, -1x7x7x768xf32, -1x7x7x768xf32, -1x7x7x768xf32]) <- (-1x7x7x768xf32, -1x7x7x768xf32, -1x7x7x768xf32, -1x7x7x768xf32)
        t2857 = [t2848, t2849, t2850, t2851]

        # pd_op.concat: (-1x7x7x3072xf32) <- ([-1x7x7x768xf32, -1x7x7x768xf32, -1x7x7x768xf32, -1x7x7x768xf32], 1xi32)
        t2858 = paddle._C_ops.concat(t2857, t805)
        del t2857

        # pd_op.full: (xi64) <- ()
        t2859 = paddle._C_ops.full(
            [], float("3072"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2860 = [t2844, t810, t2859]
        del t810, t2859, t2844

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2861 = paddle._C_ops.stack(t2860, 0)
        del t2860

        # pd_op.reshape: (-1x-1x3072xf32) <- (-1x7x7x3072xf32, 3xi64)
        t2862 = paddle._C_ops.reshape(t2858, t2861)
        del t2861

        # pd_op.layer_norm: (-1x-1x3072xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x3072xf32, 3072xf32, 3072xf32)
        t2863, t2864, t2865 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2862, t274, t275, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t275, t274

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x3072xf32, 3072x1536xf32)
        t2866 = paddle._C_ops.matmul(t2863, t276, False, False)
        del t276

        # pd_op.shape64: (3xi64) <- (-1x-1x1536xf32)
        t2867 = paddle._C_ops.shape64(t2866)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2868 = paddle._C_ops.slice(t2867, [0], t305, t354, [1], [0])
        del t2867

        # pd_op.shape64: (3xi64) <- (-1x-1x1536xf32)
        t2869 = paddle._C_ops.shape64(t2866)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2870 = paddle._C_ops.slice(t2869, [0], t354, t450, [1], [0])
        del t2869

        # pd_op.layer_norm: (-1x-1x1536xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x1536xf32, 1536xf32, 1536xf32)
        t2871, t2872, t2873 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2866, t277, t278, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t278, t277

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2874 = [t2868, t428, t428, t1048]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2875 = paddle._C_ops.stack(t2874, 0)
        del t2874

        # pd_op.reshape: (-1x7x7x1536xf32) <- (-1x-1x1536xf32, 4xi64)
        t2876 = paddle._C_ops.reshape(t2871, t2875)
        del t2875

        # pd_op.shape64: (4xi64) <- (-1x7x7x1536xf32)
        t2877 = paddle._C_ops.shape64(t2876)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2878 = paddle._C_ops.slice(t2877, [0], t305, t354, [1], [0])
        del t2877

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2879 = [t2878, t744, t428, t744, t428, t1048]
        del t2878

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2880 = paddle._C_ops.stack(t2879, 0)
        del t2879

        # pd_op.reshape: (-1x1x7x1x7x1536xf32) <- (-1x7x7x1536xf32, 6xi64)
        t2881 = paddle._C_ops.reshape(t2876, t2880)
        del t2880

        # pd_op.transpose: (-1x1x1x7x7x1536xf32) <- (-1x1x7x1x7x1536xf32)
        t2882 = paddle._C_ops.transpose(t2881, [0, 1, 3, 2, 4, 5])
        del t2881

        # pd_op.full_int_array: (4xi64) <- ()
        t2883 = [-1, 7, 7, 1536]

        # pd_op.reshape: (-1x7x7x1536xf32) <- (-1x1x1x7x7x1536xf32, 4xi64)
        t2884 = paddle._C_ops.reshape(t2882, t2883)

        # pd_op.full_int_array: (3xi64) <- ()
        t2885 = [-1, 49, 1536]

        # pd_op.reshape: (-1x49x1536xf32) <- (-1x7x7x1536xf32, 3xi64)
        t2886 = paddle._C_ops.reshape(t2884, t2885)

        # pd_op.shape64: (3xi64) <- (-1x49x1536xf32)
        t2887 = paddle._C_ops.shape64(t2886)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2888 = paddle._C_ops.slice(t2887, [0], t305, t354, [1], [0])
        del t2887

        # pd_op.matmul: (-1x49x4608xf32) <- (-1x49x1536xf32, 1536x4608xf32)
        t2889 = paddle._C_ops.matmul(t2886, t279, False, False)
        del t279

        # pd_op.add: (-1x49x4608xf32) <- (-1x49x4608xf32, 4608xf32)
        t2890 = paddle._C_ops.add(t2889, t280)
        del t280

        # pd_op.full: (xi64) <- ()
        t2891 = paddle._C_ops.full(
            [], float("48"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t2892 = [t2888, t441, t442, t2891, t444]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t2893 = paddle._C_ops.stack(t2892, 0)
        del t2892

        # pd_op.reshape: (-1x49x3x48x32xf32) <- (-1x49x4608xf32, 5xi64)
        t2894 = paddle._C_ops.reshape(t2890, t2893)
        del t2893

        # pd_op.transpose: (3x-1x48x49x32xf32) <- (-1x49x3x48x32xf32)
        t2895 = paddle._C_ops.transpose(t2894, [2, 0, 3, 1, 4])
        del t2894

        # pd_op.slice: (-1x48x49x32xf32) <- (3x-1x48x49x32xf32, 1xi64, 1xi64)
        t2896 = paddle._C_ops.slice(t2895, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x48x49x32xf32) <- (3x-1x48x49x32xf32, 1xi64, 1xi64)
        t2897 = paddle._C_ops.slice(t2895, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x48x49x32xf32) <- (3x-1x48x49x32xf32, 1xi64, 1xi64)
        t2898 = paddle._C_ops.slice(t2895, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x48x49x32xf32) <- (-1x48x49x32xf32, 1xf32)
        t2899 = paddle._C_ops.scale(t2896, t526, float("0"), True)
        del t2896

        # pd_op.transpose: (-1x48x32x49xf32) <- (-1x48x49x32xf32)
        t2900 = paddle._C_ops.transpose(t2897, [0, 1, 3, 2])
        del t2897

        # pd_op.matmul: (-1x48x49x49xf32) <- (-1x48x49x32xf32, -1x48x32x49xf32)
        t2901 = paddle._C_ops.matmul(t2899, t2900, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t2902 = paddle._C_ops.reshape(input45, t553)
        del input45

        # pd_op.index_select: (2401x48xf32) <- (169x48xf32, 2401xi64)
        t2903 = paddle._C_ops.index_select(input46, t2902, 0)
        del input46

        # pd_op.reshape: (49x49x48xf32) <- (2401x48xf32, 3xi64)
        t2904 = paddle._C_ops.reshape(t2903, t556)

        # pd_op.transpose: (48x49x49xf32) <- (49x49x48xf32)
        t2905 = paddle._C_ops.transpose(t2904, [2, 0, 1])
        del t2904

        # pd_op.unsqueeze: (1x48x49x49xf32) <- (48x49x49xf32, 1xi64)
        t2906 = paddle._C_ops.unsqueeze(t2905, t305)

        # pd_op.add: (-1x48x49x49xf32) <- (-1x48x49x49xf32, 1x48x49x49xf32)
        t2907 = paddle._C_ops.add(t2901, t2906)

        # pd_op.softmax: (-1x48x49x49xf32) <- (-1x48x49x49xf32)
        t2908 = paddle._C_ops.softmax(t2907, -1)
        del t2907

        # pd_op.matmul: (-1x48x49x32xf32) <- (-1x48x49x49xf32, -1x48x49x32xf32)
        t2909 = paddle._C_ops.matmul(t2908, t2898, False, False)

        # pd_op.transpose: (-1x49x48x32xf32) <- (-1x48x49x32xf32)
        t2910 = paddle._C_ops.transpose(t2909, [0, 2, 1, 3])
        del t2909

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2911 = [t2888, t441, t1048]
        del t2888

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2912 = paddle._C_ops.stack(t2911, 0)
        del t2911

        # pd_op.reshape: (-1x49x1536xf32) <- (-1x49x48x32xf32, 3xi64)
        t2913 = paddle._C_ops.reshape(t2910, t2912)
        del t2912

        # pd_op.matmul: (-1x49x1536xf32) <- (-1x49x1536xf32, 1536x1536xf32)
        t2914 = paddle._C_ops.matmul(t2913, t281, False, False)
        del t281

        # pd_op.add: (-1x49x1536xf32) <- (-1x49x1536xf32, 1536xf32)
        t2915 = paddle._C_ops.add(t2914, t282)
        del t282

        # pd_op.reshape: (-1x7x7x1536xf32) <- (-1x49x1536xf32, 4xi64)
        t2916 = paddle._C_ops.reshape(t2915, t2883)

        # pd_op.full_int_array: (6xi64) <- ()
        t2917 = [-1, 1, 1, 7, 7, 1536]

        # pd_op.reshape: (-1x1x1x7x7x1536xf32) <- (-1x7x7x1536xf32, 6xi64)
        t2918 = paddle._C_ops.reshape(t2916, t2917)

        # pd_op.transpose: (-1x1x7x1x7x1536xf32) <- (-1x1x1x7x7x1536xf32)
        t2919 = paddle._C_ops.transpose(t2918, [0, 1, 3, 2, 4, 5])
        del t2918

        # pd_op.reshape: (-1x7x7x1536xf32) <- (-1x1x7x1x7x1536xf32, 4xi64)
        t2920 = paddle._C_ops.reshape(t2919, t2883)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2921 = [t2868, t441, t1048]
        del t2868

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2922 = paddle._C_ops.stack(t2921, 0)
        del t2921

        # pd_op.reshape: (-1x49x1536xf32) <- (-1x7x7x1536xf32, 3xi64)
        t2923 = paddle._C_ops.reshape(t2920, t2922)
        del t2922

        # pd_op.full: (xf32) <- ()
        t2924 = paddle._C_ops.full(
            [],
            float("0.904348"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        t2925 = t2924

        # pd_op.shape64: (3xi64) <- (-1x49x1536xf32)
        t2926 = paddle._C_ops.shape64(t2923)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2927 = paddle._C_ops.slice(t2926, [0], t305, t354, [1], [0])
        del t2926

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2928 = [t2927, t744, t744]
        del t2927

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2929 = paddle._C_ops.stack(t2928, 0)
        del t2928

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2930 = paddle._C_ops.uniform(
            t2929,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2929

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2931 = paddle._C_ops.add(t2924, t2930)
        del t2930

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2932 = paddle._C_ops.floor(t2931)
        del t2931

        # pd_op.divide: (-1x49x1536xf32) <- (-1x49x1536xf32, xf32)
        t2933 = paddle._C_ops.divide(t2923, t2924)

        # pd_op.multiply: (-1x49x1536xf32) <- (-1x49x1536xf32, -1x1x1xf32)
        t2934 = paddle._C_ops.multiply(t2933, t2932)

        # pd_op.add: (-1x49x1536xf32) <- (-1x-1x1536xf32, -1x49x1536xf32)
        t2935 = paddle._C_ops.add(t2866, t2934)

        # pd_op.layer_norm: (-1x49x1536xf32, -1x49xf32, -1x49xf32) <- (-1x49x1536xf32, 1536xf32, 1536xf32)
        t2936, t2937, t2938 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2935, t283, t284, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t284, t283

        # pd_op.matmul: (-1x49x6144xf32) <- (-1x49x1536xf32, 1536x6144xf32)
        t2939 = paddle._C_ops.matmul(t2936, t285, False, False)
        del t285

        # pd_op.add: (-1x49x6144xf32) <- (-1x49x6144xf32, 6144xf32)
        t2940 = paddle._C_ops.add(t2939, t286)
        del t286

        # pd_op.gelu: (-1x49x6144xf32) <- (-1x49x6144xf32)
        t2941 = paddle._C_ops.gelu(t2940, False)

        # pd_op.matmul: (-1x49x1536xf32) <- (-1x49x6144xf32, 6144x1536xf32)
        t2942 = paddle._C_ops.matmul(t2941, t287, False, False)
        del t287

        # pd_op.add: (-1x49x1536xf32) <- (-1x49x1536xf32, 1536xf32)
        t2943 = paddle._C_ops.add(t2942, t288)
        del t288

        # pd_op.shape64: (3xi64) <- (-1x49x1536xf32)
        t2944 = paddle._C_ops.shape64(t2943)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2945 = paddle._C_ops.slice(t2944, [0], t305, t354, [1], [0])
        del t2944

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t2946 = [t2945, t744, t744]
        del t2945

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t2947 = paddle._C_ops.stack(t2946, 0)
        del t2946

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t2948 = paddle._C_ops.uniform(
            t2947,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t2947

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t2949 = paddle._C_ops.add(t2924, t2948)
        del t2948

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t2950 = paddle._C_ops.floor(t2949)
        del t2949

        # pd_op.divide: (-1x49x1536xf32) <- (-1x49x1536xf32, xf32)
        t2951 = paddle._C_ops.divide(t2943, t2924)

        # pd_op.multiply: (-1x49x1536xf32) <- (-1x49x1536xf32, -1x1x1xf32)
        t2952 = paddle._C_ops.multiply(t2951, t2950)

        # pd_op.add: (-1x49x1536xf32) <- (-1x49x1536xf32, -1x49x1536xf32)
        t2953 = paddle._C_ops.add(t2935, t2952)

        # pd_op.shape64: (3xi64) <- (-1x49x1536xf32)
        t2954 = paddle._C_ops.shape64(t2953)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2955 = paddle._C_ops.slice(t2954, [0], t305, t354, [1], [0])
        del t2954

        # pd_op.layer_norm: (-1x49x1536xf32, -1x49xf32, -1x49xf32) <- (-1x49x1536xf32, 1536xf32, 1536xf32)
        t2956, t2957, t2958 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t2953, t289, t290, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t290, t289

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t2959 = [t2955, t428, t428, t1048]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t2960 = paddle._C_ops.stack(t2959, 0)
        del t2959

        # pd_op.reshape: (-1x7x7x1536xf32) <- (-1x49x1536xf32, 4xi64)
        t2961 = paddle._C_ops.reshape(t2956, t2960)
        del t2960

        # pd_op.shape64: (4xi64) <- (-1x7x7x1536xf32)
        t2962 = paddle._C_ops.shape64(t2961)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2963 = paddle._C_ops.slice(t2962, [0], t305, t354, [1], [0])
        del t2962

        # pd_op.roll: (-1x7x7x1536xf32) <- (-1x7x7x1536xf32, 2xi64)
        t2964 = paddle._C_ops.roll(t2961, t599, [1, 2])

        # pd_op.shape64: (4xi64) <- (-1x7x7x1536xf32)
        t2965 = paddle._C_ops.shape64(t2964)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t2966 = paddle._C_ops.slice(t2965, [0], t305, t354, [1], [0])
        del t2965

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        t2967 = [t2966, t744, t428, t744, t428, t1048]
        del t428, t2966

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        t2968 = paddle._C_ops.stack(t2967, 0)
        del t2967

        # pd_op.reshape: (-1x1x7x1x7x1536xf32) <- (-1x7x7x1536xf32, 6xi64)
        t2969 = paddle._C_ops.reshape(t2964, t2968)
        del t2968

        # pd_op.transpose: (-1x1x1x7x7x1536xf32) <- (-1x1x7x1x7x1536xf32)
        t2970 = paddle._C_ops.transpose(t2969, [0, 1, 3, 2, 4, 5])
        del t2969

        # pd_op.reshape: (-1x7x7x1536xf32) <- (-1x1x1x7x7x1536xf32, 4xi64)
        t2971 = paddle._C_ops.reshape(t2970, t2883)

        # pd_op.reshape: (-1x49x1536xf32) <- (-1x7x7x1536xf32, 3xi64)
        t2972 = paddle._C_ops.reshape(t2971, t2885)
        del t2885

        # pd_op.full: (1x7x7x1xf32) <- ()
        t2973 = paddle._C_ops.full(
            [1, 7, 7, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        t2974 = paddle._C_ops.set_value_(
            t2973, t621, t625, t626, [1, 2], [], [], [1], [float("0")]
        )
        del t2973, t621

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        t2975 = paddle._C_ops.set_value_(
            t2974, t632, t633, t626, [1, 2], [], [], [1], [float("1")]
        )
        del t632, t2974

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        t2976 = paddle._C_ops.set_value_(
            t2975, t635, t636, t626, [1, 2], [], [], [1], [float("2")]
        )
        del t635, t636, t2975

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        t2977 = paddle._C_ops.set_value_(
            t2976, t638, t639, t626, [1, 2], [], [], [1], [float("3")]
        )
        del t638, t2976

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        t2978 = paddle._C_ops.set_value_(
            t2977, t625, t599, t626, [1, 2], [], [], [1], [float("4")]
        )
        del t625, t2977

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        t2979 = paddle._C_ops.set_value_(
            t2978, t633, t642, t626, [1, 2], [], [], [1], [float("5")]
        )
        del t633, t642, t2978

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        t2980 = paddle._C_ops.set_value_(
            t2979, t644, t645, t626, [1, 2], [], [], [1], [float("6")]
        )
        del t644, t645, t2979

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        t2981 = paddle._C_ops.set_value_(
            t2980, t639, t647, t626, [1, 2], [], [], [1], [float("7")]
        )
        del t639, t647, t2980

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        t2982 = paddle._C_ops.set_value_(
            t2981, t599, t649, t626, [1, 2], [], [], [1], [float("8")]
        )
        del t649, t2981

        # pd_op.full_int_array: (6xi64) <- ()
        t2983 = [1, 1, 7, 1, 7, 1]

        # pd_op.reshape: (1x1x7x1x7x1xf32) <- (1x7x7x1xf32, 6xi64)
        t2984 = paddle._C_ops.reshape(t2982, t2983)
        del t2983

        # pd_op.transpose: (1x1x1x7x7x1xf32) <- (1x1x7x1x7x1xf32)
        t2985 = paddle._C_ops.transpose(t2984, [0, 1, 3, 2, 4, 5])
        del t2984

        # pd_op.reshape: (1x7x7x1xf32) <- (1x1x1x7x7x1xf32, 4xi64)
        t2986 = paddle._C_ops.reshape(t2985, t666)
        del t666, t2985

        # pd_op.reshape: (1x49xf32) <- (1x7x7x1xf32, 2xi64)
        t2987 = paddle._C_ops.reshape(t2986, t668)
        del t668, t2986

        # pd_op.unsqueeze: (1x1x49xf32) <- (1x49xf32, 1xi64)
        t2988 = paddle._C_ops.unsqueeze(t2987, t354)

        # pd_op.unsqueeze: (1x49x1xf32) <- (1x49xf32, 1xi64)
        t2989 = paddle._C_ops.unsqueeze(t2987, t450)
        del t2987

        # pd_op.subtract: (1x49x49xf32) <- (1x1x49xf32, 1x49x1xf32)
        t2990 = paddle._C_ops.subtract(t2988, t2989)
        del t2988, t2989

        # pd_op.not_equal: (1x49x49xb) <- (1x49x49xf32, xf32)
        t2991 = paddle._C_ops.not_equal(t2990, t673)

        # pd_op.full: (1x49x49xf32) <- ()
        t2992 = paddle._C_ops.full(
            [1, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x49x49xf32) <- (1x49x49xb, 1x49x49xf32, 1x49x49xf32)
        t2993 = paddle._C_ops.where(t2991, t2992, t2990)
        del t2992, t2991, t2990

        # pd_op.equal: (1x49x49xb) <- (1x49x49xf32, xf32)
        t2994 = paddle._C_ops.equal(t2993, t673)
        del t673

        # pd_op.full: (1x49x49xf32) <- ()
        t2995 = paddle._C_ops.full(
            [1, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x49x49xf32) <- (1x49x49xb, 1x49x49xf32, 1x49x49xf32)
        t2996 = paddle._C_ops.where(t2994, t2995, t2993)
        del t2994, t2995, t2993

        # pd_op.shape64: (3xi64) <- (-1x49x1536xf32)
        t2997 = paddle._C_ops.shape64(t2972)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t2998 = paddle._C_ops.slice(t2997, [0], t305, t354, [1], [0])
        del t2997

        # pd_op.matmul: (-1x49x4608xf32) <- (-1x49x1536xf32, 1536x4608xf32)
        t2999 = paddle._C_ops.matmul(t2972, t291, False, False)
        del t291

        # pd_op.add: (-1x49x4608xf32) <- (-1x49x4608xf32, 4608xf32)
        t3000 = paddle._C_ops.add(t2999, t292)
        del t292

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t3001 = [t2998, t441, t442, t2891, t444]
        del t442, t444

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t3002 = paddle._C_ops.stack(t3001, 0)
        del t3001

        # pd_op.reshape: (-1x49x3x48x32xf32) <- (-1x49x4608xf32, 5xi64)
        t3003 = paddle._C_ops.reshape(t3000, t3002)
        del t3002

        # pd_op.transpose: (3x-1x48x49x32xf32) <- (-1x49x3x48x32xf32)
        t3004 = paddle._C_ops.transpose(t3003, [2, 0, 3, 1, 4])
        del t3003

        # pd_op.slice: (-1x48x49x32xf32) <- (3x-1x48x49x32xf32, 1xi64, 1xi64)
        t3005 = paddle._C_ops.slice(t3004, [0], t305, t354, [1], [0])

        # pd_op.slice: (-1x48x49x32xf32) <- (3x-1x48x49x32xf32, 1xi64, 1xi64)
        t3006 = paddle._C_ops.slice(t3004, [0], t354, t450, [1], [0])

        # pd_op.slice: (-1x48x49x32xf32) <- (3x-1x48x49x32xf32, 1xi64, 1xi64)
        t3007 = paddle._C_ops.slice(t3004, [0], t450, t501, [1], [0])

        # pd_op.scale: (-1x48x49x32xf32) <- (-1x48x49x32xf32, 1xf32)
        t3008 = paddle._C_ops.scale(t3005, t526, float("0"), True)
        del t3005

        # pd_op.transpose: (-1x48x32x49xf32) <- (-1x48x49x32xf32)
        t3009 = paddle._C_ops.transpose(t3006, [0, 1, 3, 2])
        del t3006

        # pd_op.matmul: (-1x48x49x49xf32) <- (-1x48x49x32xf32, -1x48x32x49xf32)
        t3010 = paddle._C_ops.matmul(t3008, t3009, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        t3011 = paddle._C_ops.reshape(input47, t553)
        del input47, t553

        # pd_op.index_select: (2401x48xf32) <- (169x48xf32, 2401xi64)
        t3012 = paddle._C_ops.index_select(input48, t3011, 0)
        del input48

        # pd_op.reshape: (49x49x48xf32) <- (2401x48xf32, 3xi64)
        t3013 = paddle._C_ops.reshape(t3012, t556)
        del t556

        # pd_op.transpose: (48x49x49xf32) <- (49x49x48xf32)
        t3014 = paddle._C_ops.transpose(t3013, [2, 0, 1])
        del t3013

        # pd_op.unsqueeze: (1x48x49x49xf32) <- (48x49x49xf32, 1xi64)
        t3015 = paddle._C_ops.unsqueeze(t3014, t305)

        # pd_op.add: (-1x48x49x49xf32) <- (-1x48x49x49xf32, 1x48x49x49xf32)
        t3016 = paddle._C_ops.add(t3010, t3015)

        # pd_op.full: (xi64) <- ()
        t3017 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        t3018 = paddle._C_ops.floor_divide(t2998, t3017)
        del t3017

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t3019 = [t3018, t744, t2891, t441, t441]
        del t3018

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t3020 = paddle._C_ops.stack(t3019, 0)
        del t3019

        # pd_op.reshape: (-1x1x48x49x49xf32) <- (-1x48x49x49xf32, 5xi64)
        t3021 = paddle._C_ops.reshape(t3016, t3020)
        del t3020

        # pd_op.unsqueeze: (1x1x49x49xf32) <- (1x49x49xf32, 1xi64)
        t3022 = paddle._C_ops.unsqueeze(t2996, t354)
        del t2996

        # pd_op.unsqueeze: (1x1x1x49x49xf32) <- (1x1x49x49xf32, 1xi64)
        t3023 = paddle._C_ops.unsqueeze(t3022, t305)
        del t3022

        # pd_op.add: (-1x1x48x49x49xf32) <- (-1x1x48x49x49xf32, 1x1x1x49x49xf32)
        t3024 = paddle._C_ops.add(t3021, t3023)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t3025 = [t2998, t2891, t441, t441]
        del t2891

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t3026 = paddle._C_ops.stack(t3025, 0)
        del t3025

        # pd_op.reshape: (-1x48x49x49xf32) <- (-1x1x48x49x49xf32, 4xi64)
        t3027 = paddle._C_ops.reshape(t3024, t3026)
        del t3026

        # pd_op.softmax: (-1x48x49x49xf32) <- (-1x48x49x49xf32)
        t3028 = paddle._C_ops.softmax(t3027, -1)
        del t3027

        # pd_op.matmul: (-1x48x49x32xf32) <- (-1x48x49x49xf32, -1x48x49x32xf32)
        t3029 = paddle._C_ops.matmul(t3028, t3007, False, False)

        # pd_op.transpose: (-1x49x48x32xf32) <- (-1x48x49x32xf32)
        t3030 = paddle._C_ops.transpose(t3029, [0, 2, 1, 3])
        del t3029

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t3031 = [t2998, t441, t1048]
        del t2998

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t3032 = paddle._C_ops.stack(t3031, 0)
        del t3031

        # pd_op.reshape: (-1x49x1536xf32) <- (-1x49x48x32xf32, 3xi64)
        t3033 = paddle._C_ops.reshape(t3030, t3032)
        del t3032

        # pd_op.matmul: (-1x49x1536xf32) <- (-1x49x1536xf32, 1536x1536xf32)
        t3034 = paddle._C_ops.matmul(t3033, t293, False, False)
        del t293

        # pd_op.add: (-1x49x1536xf32) <- (-1x49x1536xf32, 1536xf32)
        t3035 = paddle._C_ops.add(t3034, t294)
        del t294

        # pd_op.reshape: (-1x7x7x1536xf32) <- (-1x49x1536xf32, 4xi64)
        t3036 = paddle._C_ops.reshape(t3035, t2883)

        # pd_op.reshape: (-1x1x1x7x7x1536xf32) <- (-1x7x7x1536xf32, 6xi64)
        t3037 = paddle._C_ops.reshape(t3036, t2917)
        del t2917

        # pd_op.transpose: (-1x1x7x1x7x1536xf32) <- (-1x1x1x7x7x1536xf32)
        t3038 = paddle._C_ops.transpose(t3037, [0, 1, 3, 2, 4, 5])
        del t3037

        # pd_op.reshape: (-1x7x7x1536xf32) <- (-1x1x7x1x7x1536xf32, 4xi64)
        t3039 = paddle._C_ops.reshape(t3038, t2883)
        del t2883

        # pd_op.roll: (-1x7x7x1536xf32) <- (-1x7x7x1536xf32, 2xi64)
        t3040 = paddle._C_ops.roll(t3039, t724, [1, 2])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t3041 = [t2955, t441, t1048]
        del t441, t1048, t2955

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t3042 = paddle._C_ops.stack(t3041, 0)
        del t3041

        # pd_op.reshape: (-1x49x1536xf32) <- (-1x7x7x1536xf32, 3xi64)
        t3043 = paddle._C_ops.reshape(t3040, t3042)
        del t3042

        # pd_op.full: (xf32) <- ()
        t3044 = paddle._C_ops.full(
            [], float("0.9"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign: (xf32) <- (xf32)
        t3045 = t3044

        # pd_op.shape64: (3xi64) <- (-1x49x1536xf32)
        t3046 = paddle._C_ops.shape64(t3043)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t3047 = paddle._C_ops.slice(t3046, [0], t305, t354, [1], [0])
        del t3046

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t3048 = [t3047, t744, t744]
        del t3047

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t3049 = paddle._C_ops.stack(t3048, 0)
        del t3048

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t3050 = paddle._C_ops.uniform(
            t3049,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t3049

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t3051 = paddle._C_ops.add(t3044, t3050)
        del t3050

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t3052 = paddle._C_ops.floor(t3051)
        del t3051

        # pd_op.divide: (-1x49x1536xf32) <- (-1x49x1536xf32, xf32)
        t3053 = paddle._C_ops.divide(t3043, t3044)

        # pd_op.multiply: (-1x49x1536xf32) <- (-1x49x1536xf32, -1x1x1xf32)
        t3054 = paddle._C_ops.multiply(t3053, t3052)

        # pd_op.add: (-1x49x1536xf32) <- (-1x49x1536xf32, -1x49x1536xf32)
        t3055 = paddle._C_ops.add(t2953, t3054)

        # pd_op.layer_norm: (-1x49x1536xf32, -1x49xf32, -1x49xf32) <- (-1x49x1536xf32, 1536xf32, 1536xf32)
        t3056, t3057, t3058 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t3055, t295, t296, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t296, t295

        # pd_op.matmul: (-1x49x6144xf32) <- (-1x49x1536xf32, 1536x6144xf32)
        t3059 = paddle._C_ops.matmul(t3056, t297, False, False)
        del t297

        # pd_op.add: (-1x49x6144xf32) <- (-1x49x6144xf32, 6144xf32)
        t3060 = paddle._C_ops.add(t3059, t298)
        del t298

        # pd_op.gelu: (-1x49x6144xf32) <- (-1x49x6144xf32)
        t3061 = paddle._C_ops.gelu(t3060, False)

        # pd_op.matmul: (-1x49x1536xf32) <- (-1x49x6144xf32, 6144x1536xf32)
        t3062 = paddle._C_ops.matmul(t3061, t299, False, False)
        del t299

        # pd_op.add: (-1x49x1536xf32) <- (-1x49x1536xf32, 1536xf32)
        t3063 = paddle._C_ops.add(t3062, t300)
        del t300

        # pd_op.shape64: (3xi64) <- (-1x49x1536xf32)
        t3064 = paddle._C_ops.shape64(t3063)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t3065 = paddle._C_ops.slice(t3064, [0], t305, t354, [1], [0])
        del t305, t354, t3064

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t3066 = [t3065, t744, t744]
        del t744, t3065

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t3067 = paddle._C_ops.stack(t3066, 0)
        del t3066

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        t3068 = paddle._C_ops.uniform(
            t3067,
            paddle.float32,
            t747,
            t748,
            0,
            paddle.framework._current_expected_place(),
        )
        del t747, t748, t3067

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        t3069 = paddle._C_ops.add(t3044, t3068)
        del t3068

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        t3070 = paddle._C_ops.floor(t3069)
        del t3069

        # pd_op.divide: (-1x49x1536xf32) <- (-1x49x1536xf32, xf32)
        t3071 = paddle._C_ops.divide(t3063, t3044)

        # pd_op.multiply: (-1x49x1536xf32) <- (-1x49x1536xf32, -1x1x1xf32)
        t3072 = paddle._C_ops.multiply(t3071, t3070)

        # pd_op.add: (-1x49x1536xf32) <- (-1x49x1536xf32, -1x49x1536xf32)
        t3073 = paddle._C_ops.add(t3055, t3072)

        # pd_op.layer_norm: (-1x49x1536xf32, -1x49xf32, -1x49xf32) <- (-1x49x1536xf32, 1536xf32, 1536xf32)
        t3074, t3075, t3076 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t3073, t301, t302, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t302, t301

        # pd_op.transpose: (-1x1536x49xf32) <- (-1x49x1536xf32)
        t3077 = paddle._C_ops.transpose(t3074, [0, 2, 1])
        del t3074

        # pd_op.unsqueeze: (-1x1536x1x49xf32) <- (-1x1536x49xf32, 1xi64)
        t3078 = paddle._C_ops.unsqueeze(t3077, t450)

        # pd_op.pool2d: (-1x1536x1x1xf32) <- (-1x1536x1x49xf32, 2xi64)
        t3079 = paddle._C_ops.pool2d(
            t3078,
            t626,
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
        del t626

        # pd_op.squeeze: (-1x1536x1xf32) <- (-1x1536x1x1xf32, 1xi64)
        t3080 = paddle._C_ops.squeeze(t3079, t450)

        # pd_op.flatten: (-1x1536xf32) <- (-1x1536x1xf32)
        t3081 = paddle._C_ops.flatten(t3080, 1, 2)

        # pd_op.matmul: (-1x102xf32) <- (-1x1536xf32, 1536x102xf32)
        t3082 = paddle._C_ops.matmul(t3081, t303, False, False)
        del t303

        return (
            t306,
            t307,
            t308,
            t309,
            t310,
            t311,
            t312,
            t313,
            t314,
            t315,
            t316,
            t317,
            t318,
            t319,
            t320,
            t321,
            t322,
            t323,
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
            t343,
            t344,
            t345,
            t346,
            t347,
            t348,
            t349,
            t350,
            t351,
            t352,
            t353,
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
            t372,
            t373,
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
            t401,
            t402,
            t404,
            t406,
            t407,
            t411,
            t412,
            t413,
            t414,
            t417,
            t418,
            t419,
            t424,
            t432,
            t434,
            t436,
            t439,
            t440,
            t448,
            t450,
            t451,
            t452,
            t453,
            t454,
            t455,
            t456,
            t457,
            t458,
            t459,
            t460,
            t461,
            t462,
            t463,
            t464,
            t465,
            t466,
            t467,
            t468,
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
            t488,
            t489,
            t490,
            t491,
            t492,
            t493,
            t494,
            t495,
            t496,
            t497,
            t498,
            t499,
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
            t517,
            t518,
            t519,
            t520,
            t521,
            t522,
            t523,
            t524,
            t525,
            t526,
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
            t546,
            t547,
            t548,
            t549,
            t550,
            t551,
            t552,
            t554,
            t555,
            t558,
            t559,
            t561,
            t563,
            t566,
            t567,
            t568,
            t569,
            t572,
            t574,
            t578,
            t579,
            t580,
            t581,
            t582,
            t583,
            t584,
            t585,
            t586,
            t587,
            t588,
            t591,
            t592,
            t593,
            t596,
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
            t617,
            t618,
            t619,
            t622,
            t623,
            t624,
            t627,
            t628,
            t629,
            t630,
            t650,
            t651,
            t652,
            t653,
            t654,
            t655,
            t656,
            t657,
            t658,
            t659,
            t660,
            t661,
            t662,
            t682,
            t683,
            t687,
            t690,
            t691,
            t692,
            t693,
            t694,
            t695,
            t697,
            t698,
            t699,
            t705,
            t707,
            t708,
            t712,
            t714,
            t717,
            t718,
            t719,
            t720,
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
            t739,
            t740,
            t741,
            t751,
            t752,
            t753,
            t754,
            t755,
            t756,
            t757,
            t758,
            t759,
            t760,
            t761,
            t762,
            t769,
            t770,
            t771,
            t772,
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
            t794,
            t795,
            t796,
            t797,
            t798,
            t799,
            t805,
            t806,
            t807,
            t809,
            t814,
            t815,
            t816,
            t817,
            t818,
            t823,
            t824,
            t825,
            t830,
            t837,
            t839,
            t841,
            t844,
            t845,
            t850,
            t853,
            t854,
            t855,
            t856,
            t857,
            t858,
            t860,
            t861,
            t863,
            t865,
            t868,
            t869,
            t870,
            t871,
            t874,
            t876,
            t880,
            t881,
            t882,
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
            t907,
            t908,
            t909,
            t910,
            t913,
            t914,
            t915,
            t918,
            t921,
            t927,
            t928,
            t929,
            t939,
            t956,
            t957,
            t961,
            t964,
            t965,
            t966,
            t967,
            t968,
            t969,
            t971,
            t972,
            t973,
            t979,
            t981,
            t982,
            t986,
            t988,
            t991,
            t992,
            t993,
            t994,
            t996,
            t997,
            t998,
            t1001,
            t1002,
            t1003,
            t1010,
            t1011,
            t1012,
            t1013,
            t1014,
            t1015,
            t1016,
            t1017,
            t1018,
            t1019,
            t1020,
            t1021,
            t1028,
            t1029,
            t1030,
            t1031,
            t1036,
            t1037,
            t1038,
            t1039,
            t1040,
            t1047,
            t1051,
            t1052,
            t1053,
            t1054,
            t1055,
            t1060,
            t1061,
            t1062,
            t1066,
            t1073,
            t1075,
            t1077,
            t1080,
            t1081,
            t1086,
            t1089,
            t1090,
            t1091,
            t1092,
            t1093,
            t1094,
            t1096,
            t1097,
            t1099,
            t1101,
            t1104,
            t1105,
            t1106,
            t1107,
            t1110,
            t1112,
            t1116,
            t1117,
            t1118,
            t1125,
            t1126,
            t1127,
            t1128,
            t1129,
            t1130,
            t1131,
            t1132,
            t1133,
            t1134,
            t1135,
            t1136,
            t1143,
            t1144,
            t1145,
            t1146,
            t1149,
            t1150,
            t1151,
            t1154,
            t1157,
            t1163,
            t1164,
            t1165,
            t1175,
            t1192,
            t1193,
            t1197,
            t1200,
            t1201,
            t1202,
            t1203,
            t1204,
            t1205,
            t1207,
            t1208,
            t1209,
            t1214,
            t1216,
            t1217,
            t1221,
            t1223,
            t1226,
            t1227,
            t1228,
            t1229,
            t1231,
            t1232,
            t1233,
            t1236,
            t1237,
            t1238,
            t1245,
            t1246,
            t1247,
            t1248,
            t1249,
            t1250,
            t1251,
            t1252,
            t1253,
            t1254,
            t1255,
            t1256,
            t1263,
            t1264,
            t1265,
            t1266,
            t1269,
            t1270,
            t1271,
            t1274,
            t1280,
            t1281,
            t1282,
            t1285,
            t1286,
            t1290,
            t1293,
            t1294,
            t1295,
            t1296,
            t1297,
            t1298,
            t1300,
            t1301,
            t1303,
            t1305,
            t1308,
            t1309,
            t1310,
            t1311,
            t1313,
            t1314,
            t1317,
            t1318,
            t1319,
            t1326,
            t1327,
            t1328,
            t1329,
            t1330,
            t1331,
            t1332,
            t1333,
            t1334,
            t1335,
            t1336,
            t1337,
            t1344,
            t1345,
            t1346,
            t1347,
            t1350,
            t1351,
            t1352,
            t1355,
            t1358,
            t1364,
            t1365,
            t1366,
            t1376,
            t1390,
            t1391,
            t1395,
            t1398,
            t1399,
            t1400,
            t1401,
            t1402,
            t1403,
            t1405,
            t1406,
            t1407,
            t1411,
            t1413,
            t1414,
            t1418,
            t1420,
            t1423,
            t1424,
            t1425,
            t1426,
            t1428,
            t1429,
            t1430,
            t1433,
            t1434,
            t1435,
            t1442,
            t1443,
            t1444,
            t1445,
            t1446,
            t1447,
            t1448,
            t1449,
            t1450,
            t1451,
            t1452,
            t1453,
            t1460,
            t1461,
            t1462,
            t1463,
            t1466,
            t1467,
            t1468,
            t1471,
            t1477,
            t1478,
            t1479,
            t1482,
            t1483,
            t1487,
            t1490,
            t1491,
            t1492,
            t1493,
            t1494,
            t1495,
            t1497,
            t1498,
            t1500,
            t1502,
            t1505,
            t1506,
            t1507,
            t1508,
            t1510,
            t1511,
            t1514,
            t1515,
            t1516,
            t1523,
            t1524,
            t1525,
            t1526,
            t1527,
            t1528,
            t1529,
            t1530,
            t1531,
            t1532,
            t1533,
            t1534,
            t1541,
            t1542,
            t1543,
            t1544,
            t1547,
            t1548,
            t1549,
            t1552,
            t1555,
            t1561,
            t1562,
            t1563,
            t1573,
            t1587,
            t1588,
            t1592,
            t1595,
            t1596,
            t1597,
            t1598,
            t1599,
            t1600,
            t1602,
            t1603,
            t1604,
            t1608,
            t1610,
            t1611,
            t1615,
            t1617,
            t1620,
            t1621,
            t1622,
            t1623,
            t1625,
            t1626,
            t1627,
            t1630,
            t1631,
            t1632,
            t1639,
            t1640,
            t1641,
            t1642,
            t1643,
            t1644,
            t1645,
            t1646,
            t1647,
            t1648,
            t1649,
            t1650,
            t1657,
            t1658,
            t1659,
            t1660,
            t1663,
            t1664,
            t1665,
            t1668,
            t1674,
            t1675,
            t1676,
            t1679,
            t1680,
            t1684,
            t1687,
            t1688,
            t1689,
            t1690,
            t1691,
            t1692,
            t1694,
            t1695,
            t1697,
            t1699,
            t1702,
            t1703,
            t1704,
            t1705,
            t1707,
            t1708,
            t1711,
            t1712,
            t1713,
            t1720,
            t1721,
            t1722,
            t1723,
            t1724,
            t1725,
            t1726,
            t1727,
            t1728,
            t1729,
            t1730,
            t1731,
            t1738,
            t1739,
            t1740,
            t1741,
            t1744,
            t1745,
            t1746,
            t1749,
            t1752,
            t1758,
            t1759,
            t1760,
            t1770,
            t1784,
            t1785,
            t1789,
            t1792,
            t1793,
            t1794,
            t1795,
            t1796,
            t1797,
            t1799,
            t1800,
            t1801,
            t1805,
            t1807,
            t1808,
            t1812,
            t1814,
            t1817,
            t1818,
            t1819,
            t1820,
            t1822,
            t1823,
            t1824,
            t1827,
            t1828,
            t1829,
            t1836,
            t1837,
            t1838,
            t1839,
            t1840,
            t1841,
            t1842,
            t1843,
            t1844,
            t1845,
            t1846,
            t1847,
            t1854,
            t1855,
            t1856,
            t1857,
            t1860,
            t1861,
            t1862,
            t1865,
            t1871,
            t1872,
            t1873,
            t1876,
            t1877,
            t1881,
            t1884,
            t1885,
            t1886,
            t1887,
            t1888,
            t1889,
            t1891,
            t1892,
            t1894,
            t1896,
            t1899,
            t1900,
            t1901,
            t1902,
            t1904,
            t1905,
            t1908,
            t1909,
            t1910,
            t1917,
            t1918,
            t1919,
            t1920,
            t1921,
            t1922,
            t1923,
            t1924,
            t1925,
            t1926,
            t1927,
            t1928,
            t1935,
            t1936,
            t1937,
            t1938,
            t1941,
            t1942,
            t1943,
            t1946,
            t1949,
            t1955,
            t1956,
            t1957,
            t1967,
            t1981,
            t1982,
            t1986,
            t1989,
            t1990,
            t1991,
            t1992,
            t1993,
            t1994,
            t1996,
            t1997,
            t1998,
            t2002,
            t2004,
            t2005,
            t2009,
            t2011,
            t2014,
            t2015,
            t2016,
            t2017,
            t2019,
            t2020,
            t2021,
            t2024,
            t2025,
            t2026,
            t2033,
            t2034,
            t2035,
            t2036,
            t2037,
            t2038,
            t2039,
            t2040,
            t2041,
            t2042,
            t2043,
            t2044,
            t2051,
            t2052,
            t2053,
            t2054,
            t2057,
            t2058,
            t2059,
            t2062,
            t2068,
            t2069,
            t2070,
            t2073,
            t2074,
            t2078,
            t2081,
            t2082,
            t2083,
            t2084,
            t2085,
            t2086,
            t2088,
            t2089,
            t2091,
            t2093,
            t2096,
            t2097,
            t2098,
            t2099,
            t2101,
            t2102,
            t2105,
            t2106,
            t2107,
            t2114,
            t2115,
            t2116,
            t2117,
            t2118,
            t2119,
            t2120,
            t2121,
            t2122,
            t2123,
            t2124,
            t2125,
            t2132,
            t2133,
            t2134,
            t2135,
            t2138,
            t2139,
            t2140,
            t2143,
            t2146,
            t2152,
            t2153,
            t2154,
            t2164,
            t2178,
            t2179,
            t2183,
            t2186,
            t2187,
            t2188,
            t2189,
            t2190,
            t2191,
            t2193,
            t2194,
            t2195,
            t2199,
            t2201,
            t2202,
            t2206,
            t2208,
            t2211,
            t2212,
            t2213,
            t2214,
            t2216,
            t2217,
            t2218,
            t2221,
            t2222,
            t2223,
            t2230,
            t2231,
            t2232,
            t2233,
            t2234,
            t2235,
            t2236,
            t2237,
            t2238,
            t2239,
            t2240,
            t2241,
            t2248,
            t2249,
            t2250,
            t2251,
            t2254,
            t2255,
            t2256,
            t2259,
            t2265,
            t2266,
            t2267,
            t2270,
            t2271,
            t2275,
            t2278,
            t2279,
            t2280,
            t2281,
            t2282,
            t2283,
            t2285,
            t2286,
            t2288,
            t2290,
            t2293,
            t2294,
            t2295,
            t2296,
            t2298,
            t2299,
            t2302,
            t2303,
            t2304,
            t2311,
            t2312,
            t2313,
            t2314,
            t2315,
            t2316,
            t2317,
            t2318,
            t2319,
            t2320,
            t2321,
            t2322,
            t2329,
            t2330,
            t2331,
            t2332,
            t2335,
            t2336,
            t2337,
            t2340,
            t2343,
            t2349,
            t2350,
            t2351,
            t2361,
            t2375,
            t2376,
            t2380,
            t2383,
            t2384,
            t2385,
            t2386,
            t2387,
            t2388,
            t2390,
            t2391,
            t2392,
            t2396,
            t2398,
            t2399,
            t2403,
            t2405,
            t2408,
            t2409,
            t2410,
            t2411,
            t2413,
            t2414,
            t2415,
            t2418,
            t2419,
            t2420,
            t2427,
            t2428,
            t2429,
            t2430,
            t2431,
            t2432,
            t2433,
            t2434,
            t2435,
            t2436,
            t2437,
            t2438,
            t2445,
            t2446,
            t2447,
            t2448,
            t2451,
            t2452,
            t2453,
            t2456,
            t2462,
            t2463,
            t2464,
            t2467,
            t2468,
            t2472,
            t2475,
            t2476,
            t2477,
            t2478,
            t2479,
            t2480,
            t2482,
            t2483,
            t2485,
            t2487,
            t2490,
            t2491,
            t2492,
            t2493,
            t2495,
            t2496,
            t2499,
            t2500,
            t2501,
            t2508,
            t2509,
            t2510,
            t2511,
            t2512,
            t2513,
            t2514,
            t2515,
            t2516,
            t2517,
            t2518,
            t2519,
            t2526,
            t2527,
            t2528,
            t2529,
            t2532,
            t2533,
            t2534,
            t2537,
            t2540,
            t2546,
            t2547,
            t2548,
            t2558,
            t2572,
            t2573,
            t2577,
            t2580,
            t2581,
            t2582,
            t2583,
            t2584,
            t2585,
            t2587,
            t2588,
            t2589,
            t2593,
            t2595,
            t2596,
            t2600,
            t2602,
            t2605,
            t2606,
            t2607,
            t2608,
            t2610,
            t2611,
            t2612,
            t2615,
            t2616,
            t2617,
            t2624,
            t2625,
            t2626,
            t2627,
            t2628,
            t2629,
            t2630,
            t2631,
            t2632,
            t2633,
            t2634,
            t2635,
            t2642,
            t2643,
            t2644,
            t2645,
            t2648,
            t2649,
            t2650,
            t2653,
            t2659,
            t2660,
            t2661,
            t2664,
            t2665,
            t2669,
            t2672,
            t2673,
            t2674,
            t2675,
            t2676,
            t2677,
            t2679,
            t2680,
            t2682,
            t2684,
            t2687,
            t2688,
            t2689,
            t2690,
            t2692,
            t2693,
            t2696,
            t2697,
            t2698,
            t2705,
            t2706,
            t2707,
            t2708,
            t2709,
            t2710,
            t2711,
            t2712,
            t2713,
            t2714,
            t2715,
            t2716,
            t2723,
            t2724,
            t2725,
            t2726,
            t2729,
            t2730,
            t2731,
            t2734,
            t2737,
            t2743,
            t2744,
            t2745,
            t2755,
            t2769,
            t2770,
            t2774,
            t2777,
            t2778,
            t2779,
            t2780,
            t2781,
            t2782,
            t2784,
            t2785,
            t2786,
            t2790,
            t2792,
            t2793,
            t2797,
            t2799,
            t2802,
            t2803,
            t2804,
            t2805,
            t2807,
            t2808,
            t2809,
            t2812,
            t2813,
            t2814,
            t2821,
            t2822,
            t2823,
            t2824,
            t2825,
            t2826,
            t2827,
            t2828,
            t2829,
            t2830,
            t2831,
            t2832,
            t2839,
            t2840,
            t2841,
            t2842,
            t2847,
            t2848,
            t2849,
            t2850,
            t2851,
            t2858,
            t2862,
            t2863,
            t2864,
            t2865,
            t2866,
            t2871,
            t2872,
            t2873,
            t2876,
            t2882,
            t2884,
            t2886,
            t2889,
            t2890,
            t2895,
            t2898,
            t2899,
            t2900,
            t2901,
            t2902,
            t2903,
            t2905,
            t2906,
            t2908,
            t2910,
            t2913,
            t2914,
            t2915,
            t2916,
            t2919,
            t2920,
            t2923,
            t2924,
            t2925,
            t2932,
            t2933,
            t2934,
            t2935,
            t2936,
            t2937,
            t2938,
            t2939,
            t2940,
            t2941,
            t2942,
            t2943,
            t2950,
            t2951,
            t2952,
            t2953,
            t2956,
            t2957,
            t2958,
            t2961,
            t2964,
            t2970,
            t2971,
            t2972,
            t2982,
            t2999,
            t3000,
            t3004,
            t3007,
            t3008,
            t3009,
            t3010,
            t3011,
            t3012,
            t3014,
            t3015,
            t3016,
            t3021,
            t3023,
            t3024,
            t3028,
            t3030,
            t3033,
            t3034,
            t3035,
            t3036,
            t3038,
            t3039,
            t3040,
            t3043,
            t3044,
            t3045,
            t3052,
            t3053,
            t3054,
            t3055,
            t3056,
            t3057,
            t3058,
            t3059,
            t3060,
            t3061,
            t3062,
            t3063,
            t3070,
            t3071,
            t3072,
            t3073,
            t3075,
            t3076,
            t3077,
            t3078,
            t3079,
            t3080,
            t3081,
            t3082,
        )
