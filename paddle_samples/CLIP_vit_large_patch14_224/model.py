import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input0: paddle.Tensor,
        t0: paddle.Tensor,
        input1: paddle.Tensor,
        input2: paddle.Tensor,
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
    ):
        # pd_op.conv2d: (16x1024x16x16xf32) <- (16x3x224x224xf32, 1024x3x14x14xf32)
        t294 = paddle._C_ops.conv2d(
            input0, t0, [14, 14], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.flatten: (16x1024x256xf32) <- (16x1024x16x16xf32)
        t295 = paddle._C_ops.flatten(t294, 2, 3)

        # pd_op.transpose: (16x256x1024xf32) <- (16x1024x256xf32)
        t296 = paddle._C_ops.transpose(t295, [0, 2, 1])
        del t295

        # pd_op.full_int_array: (3xi64) <- ()
        t297 = [16, -1, -1]

        # pd_op.expand: (16x1x1024xf32) <- (1x1x1024xf32, 3xi64)
        t298 = paddle._C_ops.expand(input1, t297)
        del input1

        # pd_op.full: (1xi32) <- ()
        t299 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([16x1x1024xf32, 16x256x1024xf32]) <- (16x1x1024xf32, 16x256x1024xf32)
        t300 = [t298, t296]

        # pd_op.concat: (16x257x1024xf32) <- ([16x1x1024xf32, 16x256x1024xf32], 1xi32)
        t301 = paddle._C_ops.concat(t300, t299)
        del t300

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1x257x1024xf32)
        t302 = paddle._C_ops.add(t301, input2)
        del input2

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t303, t304, t305 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t302, t1, t2, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t2, t1

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t306, t307, t308 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t303, t3, t4, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t4, t3

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t309 = paddle._C_ops.matmul(t306, t5, False, False)
        del t5

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t310 = paddle._C_ops.add(t309, t6)
        del t6

        # pd_op.full_int_array: (5xi64) <- ()
        t311 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t312 = paddle._C_ops.reshape(t310, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t313 = paddle._C_ops.transpose(t312, [2, 0, 3, 1, 4])
        del t312

        # pd_op.full_int_array: (1xi64) <- ()
        t314 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        t315 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t316 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t317 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t318 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t319 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t320 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t321 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t322 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t323 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t324 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t325 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t326 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t327 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t328 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t329 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t330 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t331 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t332 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t333 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t334 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t335 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t336 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t337 = t314

        # pd_op.assign: (1xi64) <- (1xi64)
        t338 = t314

        # pd_op.full_int_array: (1xi64) <- ()
        t339 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t340 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t341 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t342 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t343 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t344 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t345 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t346 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t347 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t348 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t349 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t350 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t351 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t352 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t353 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t354 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t355 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t356 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t357 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t358 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t359 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t360 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t361 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t362 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t363 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t364 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t365 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t366 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t367 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t368 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t369 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t370 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t371 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t372 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t373 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t374 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t375 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t376 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t377 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t378 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t379 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t380 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t381 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t382 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t383 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t384 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t385 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t386 = t339

        # pd_op.assign: (1xi64) <- (1xi64)
        t387 = t339

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t388 = paddle._C_ops.slice(t313, [0], t314, t339, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t389 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        t390 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t391 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t392 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t393 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t394 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t395 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t396 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t397 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t398 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t399 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t400 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t401 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t402 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t403 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t404 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t405 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t406 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t407 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t408 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t409 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t410 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t411 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t412 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t413 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t414 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t415 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t416 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t417 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t418 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t419 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t420 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t421 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t422 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t423 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t424 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t425 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t426 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t427 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t428 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t429 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t430 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t431 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t432 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t433 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t434 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t435 = t389

        # pd_op.assign: (1xi64) <- (1xi64)
        t436 = t389

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t437 = paddle._C_ops.slice(t313, [0], t339, t389, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t438 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        t439 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t440 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t441 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t442 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t443 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t444 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t445 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t446 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t447 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t448 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t449 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t450 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t451 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t452 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t453 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t454 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t455 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t456 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t457 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t458 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t459 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t460 = t438

        # pd_op.assign: (1xi64) <- (1xi64)
        t461 = t438

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t462 = paddle._C_ops.slice(t313, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t463 = paddle._C_ops.transpose(t437, [0, 1, 3, 2])
        del t437

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t464 = paddle._C_ops.matmul(t388, t463, False, False)

        # pd_op.full: (1xf32) <- ()
        t465 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t466 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t467 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t468 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t469 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t470 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t471 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t472 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t473 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t474 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t475 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t476 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t477 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t478 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t479 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t480 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t481 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t482 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t483 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t484 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t485 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t486 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t487 = t465

        # pd_op.assign: (1xf32) <- (1xf32)
        t488 = t465

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t489 = paddle._C_ops.scale(t464, t465, float("0"), True)
        del t464

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t490 = paddle._C_ops.softmax(t489, -1)
        del t489

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t491 = paddle._C_ops.matmul(t490, t462, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t492 = paddle._C_ops.transpose(t491, [0, 2, 1, 3])
        del t491

        # pd_op.full_int_array: (3xi64) <- ()
        t493 = [-1, 257, 1024]

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t494 = paddle._C_ops.reshape(t492, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t495 = paddle._C_ops.matmul(t494, t7, False, False)
        del t7

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t496 = paddle._C_ops.add(t495, t8)
        del t8

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t497 = paddle._C_ops.add(t303, t496)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t498, t499, t500 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t497, t9, t10, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t10, t9

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t501 = paddle._C_ops.matmul(t498, t11, False, False)
        del t11

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t502 = paddle._C_ops.add(t501, t12)
        del t12

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t503 = paddle._C_ops.gelu(t502, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t504 = paddle._C_ops.matmul(t503, t13, False, False)
        del t13

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t505 = paddle._C_ops.add(t504, t14)
        del t14

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t506 = paddle._C_ops.add(t497, t505)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t507, t508, t509 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t506, t15, t16, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t16, t15

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t510 = paddle._C_ops.matmul(t507, t17, False, False)
        del t17

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t511 = paddle._C_ops.add(t510, t18)
        del t18

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t512 = paddle._C_ops.reshape(t511, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t513 = paddle._C_ops.transpose(t512, [2, 0, 3, 1, 4])
        del t512

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t514 = paddle._C_ops.slice(t513, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t515 = paddle._C_ops.slice(t513, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t516 = paddle._C_ops.slice(t513, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t517 = paddle._C_ops.transpose(t515, [0, 1, 3, 2])
        del t515

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t518 = paddle._C_ops.matmul(t514, t517, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t519 = paddle._C_ops.scale(t518, t465, float("0"), True)
        del t518

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t520 = paddle._C_ops.softmax(t519, -1)
        del t519

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t521 = paddle._C_ops.matmul(t520, t516, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t522 = paddle._C_ops.transpose(t521, [0, 2, 1, 3])
        del t521

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t523 = paddle._C_ops.reshape(t522, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t524 = paddle._C_ops.matmul(t523, t19, False, False)
        del t19

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t525 = paddle._C_ops.add(t524, t20)
        del t20

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t526 = paddle._C_ops.add(t506, t525)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t527, t528, t529 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t526, t21, t22, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t22, t21

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t530 = paddle._C_ops.matmul(t527, t23, False, False)
        del t23

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t531 = paddle._C_ops.add(t530, t24)
        del t24

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t532 = paddle._C_ops.gelu(t531, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t533 = paddle._C_ops.matmul(t532, t25, False, False)
        del t25

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t534 = paddle._C_ops.add(t533, t26)
        del t26

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t535 = paddle._C_ops.add(t526, t534)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t536, t537, t538 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t535, t27, t28, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t28, t27

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t539 = paddle._C_ops.matmul(t536, t29, False, False)
        del t29

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t540 = paddle._C_ops.add(t539, t30)
        del t30

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t541 = paddle._C_ops.reshape(t540, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t542 = paddle._C_ops.transpose(t541, [2, 0, 3, 1, 4])
        del t541

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t543 = paddle._C_ops.slice(t542, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t544 = paddle._C_ops.slice(t542, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t545 = paddle._C_ops.slice(t542, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t546 = paddle._C_ops.transpose(t544, [0, 1, 3, 2])
        del t544

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t547 = paddle._C_ops.matmul(t543, t546, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t548 = paddle._C_ops.scale(t547, t465, float("0"), True)
        del t547

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t549 = paddle._C_ops.softmax(t548, -1)
        del t548

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t550 = paddle._C_ops.matmul(t549, t545, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t551 = paddle._C_ops.transpose(t550, [0, 2, 1, 3])
        del t550

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t552 = paddle._C_ops.reshape(t551, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t553 = paddle._C_ops.matmul(t552, t31, False, False)
        del t31

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t554 = paddle._C_ops.add(t553, t32)
        del t32

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t555 = paddle._C_ops.add(t535, t554)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t556, t557, t558 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t555, t33, t34, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t34, t33

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t559 = paddle._C_ops.matmul(t556, t35, False, False)
        del t35

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t560 = paddle._C_ops.add(t559, t36)
        del t36

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t561 = paddle._C_ops.gelu(t560, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t562 = paddle._C_ops.matmul(t561, t37, False, False)
        del t37

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t563 = paddle._C_ops.add(t562, t38)
        del t38

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t564 = paddle._C_ops.add(t555, t563)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t565, t566, t567 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t564, t39, t40, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t40, t39

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t568 = paddle._C_ops.matmul(t565, t41, False, False)
        del t41

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t569 = paddle._C_ops.add(t568, t42)
        del t42

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t570 = paddle._C_ops.reshape(t569, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t571 = paddle._C_ops.transpose(t570, [2, 0, 3, 1, 4])
        del t570

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t572 = paddle._C_ops.slice(t571, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t573 = paddle._C_ops.slice(t571, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t574 = paddle._C_ops.slice(t571, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t575 = paddle._C_ops.transpose(t573, [0, 1, 3, 2])
        del t573

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t576 = paddle._C_ops.matmul(t572, t575, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t577 = paddle._C_ops.scale(t576, t465, float("0"), True)
        del t576

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t578 = paddle._C_ops.softmax(t577, -1)
        del t577

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t579 = paddle._C_ops.matmul(t578, t574, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t580 = paddle._C_ops.transpose(t579, [0, 2, 1, 3])
        del t579

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t581 = paddle._C_ops.reshape(t580, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t582 = paddle._C_ops.matmul(t581, t43, False, False)
        del t43

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t583 = paddle._C_ops.add(t582, t44)
        del t44

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t584 = paddle._C_ops.add(t564, t583)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t585, t586, t587 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t584, t45, t46, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t46, t45

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t588 = paddle._C_ops.matmul(t585, t47, False, False)
        del t47

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t589 = paddle._C_ops.add(t588, t48)
        del t48

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t590 = paddle._C_ops.gelu(t589, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t591 = paddle._C_ops.matmul(t590, t49, False, False)
        del t49

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t592 = paddle._C_ops.add(t591, t50)
        del t50

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t593 = paddle._C_ops.add(t584, t592)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t594, t595, t596 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t593, t51, t52, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t52, t51

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t597 = paddle._C_ops.matmul(t594, t53, False, False)
        del t53

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t598 = paddle._C_ops.add(t597, t54)
        del t54

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t599 = paddle._C_ops.reshape(t598, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t600 = paddle._C_ops.transpose(t599, [2, 0, 3, 1, 4])
        del t599

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t601 = paddle._C_ops.slice(t600, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t602 = paddle._C_ops.slice(t600, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t603 = paddle._C_ops.slice(t600, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t604 = paddle._C_ops.transpose(t602, [0, 1, 3, 2])
        del t602

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t605 = paddle._C_ops.matmul(t601, t604, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t606 = paddle._C_ops.scale(t605, t465, float("0"), True)
        del t605

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t607 = paddle._C_ops.softmax(t606, -1)
        del t606

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t608 = paddle._C_ops.matmul(t607, t603, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t609 = paddle._C_ops.transpose(t608, [0, 2, 1, 3])
        del t608

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t610 = paddle._C_ops.reshape(t609, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t611 = paddle._C_ops.matmul(t610, t55, False, False)
        del t55

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t612 = paddle._C_ops.add(t611, t56)
        del t56

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t613 = paddle._C_ops.add(t593, t612)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t614, t615, t616 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t613, t57, t58, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t58, t57

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t617 = paddle._C_ops.matmul(t614, t59, False, False)
        del t59

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t618 = paddle._C_ops.add(t617, t60)
        del t60

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t619 = paddle._C_ops.gelu(t618, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t620 = paddle._C_ops.matmul(t619, t61, False, False)
        del t61

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t621 = paddle._C_ops.add(t620, t62)
        del t62

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t622 = paddle._C_ops.add(t613, t621)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t623, t624, t625 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t622, t63, t64, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t64, t63

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t626 = paddle._C_ops.matmul(t623, t65, False, False)
        del t65

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t627 = paddle._C_ops.add(t626, t66)
        del t66

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t628 = paddle._C_ops.reshape(t627, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t629 = paddle._C_ops.transpose(t628, [2, 0, 3, 1, 4])
        del t628

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t630 = paddle._C_ops.slice(t629, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t631 = paddle._C_ops.slice(t629, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t632 = paddle._C_ops.slice(t629, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t633 = paddle._C_ops.transpose(t631, [0, 1, 3, 2])
        del t631

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t634 = paddle._C_ops.matmul(t630, t633, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t635 = paddle._C_ops.scale(t634, t465, float("0"), True)
        del t634

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t636 = paddle._C_ops.softmax(t635, -1)
        del t635

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t637 = paddle._C_ops.matmul(t636, t632, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t638 = paddle._C_ops.transpose(t637, [0, 2, 1, 3])
        del t637

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t639 = paddle._C_ops.reshape(t638, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t640 = paddle._C_ops.matmul(t639, t67, False, False)
        del t67

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t641 = paddle._C_ops.add(t640, t68)
        del t68

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t642 = paddle._C_ops.add(t622, t641)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t643, t644, t645 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t642, t69, t70, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t70, t69

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t646 = paddle._C_ops.matmul(t643, t71, False, False)
        del t71

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t647 = paddle._C_ops.add(t646, t72)
        del t72

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t648 = paddle._C_ops.gelu(t647, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t649 = paddle._C_ops.matmul(t648, t73, False, False)
        del t73

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t650 = paddle._C_ops.add(t649, t74)
        del t74

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t651 = paddle._C_ops.add(t642, t650)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t652, t653, t654 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t651, t75, t76, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t76, t75

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t655 = paddle._C_ops.matmul(t652, t77, False, False)
        del t77

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t656 = paddle._C_ops.add(t655, t78)
        del t78

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t657 = paddle._C_ops.reshape(t656, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t658 = paddle._C_ops.transpose(t657, [2, 0, 3, 1, 4])
        del t657

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t659 = paddle._C_ops.slice(t658, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t660 = paddle._C_ops.slice(t658, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t661 = paddle._C_ops.slice(t658, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t662 = paddle._C_ops.transpose(t660, [0, 1, 3, 2])
        del t660

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t663 = paddle._C_ops.matmul(t659, t662, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t664 = paddle._C_ops.scale(t663, t465, float("0"), True)
        del t663

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t665 = paddle._C_ops.softmax(t664, -1)
        del t664

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t666 = paddle._C_ops.matmul(t665, t661, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t667 = paddle._C_ops.transpose(t666, [0, 2, 1, 3])
        del t666

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t668 = paddle._C_ops.reshape(t667, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t669 = paddle._C_ops.matmul(t668, t79, False, False)
        del t79

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t670 = paddle._C_ops.add(t669, t80)
        del t80

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t671 = paddle._C_ops.add(t651, t670)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t672, t673, t674 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t671, t81, t82, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t82, t81

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t675 = paddle._C_ops.matmul(t672, t83, False, False)
        del t83

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t676 = paddle._C_ops.add(t675, t84)
        del t84

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t677 = paddle._C_ops.gelu(t676, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t678 = paddle._C_ops.matmul(t677, t85, False, False)
        del t85

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t679 = paddle._C_ops.add(t678, t86)
        del t86

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t680 = paddle._C_ops.add(t671, t679)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t681, t682, t683 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t680, t87, t88, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t88, t87

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t684 = paddle._C_ops.matmul(t681, t89, False, False)
        del t89

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t685 = paddle._C_ops.add(t684, t90)
        del t90

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t686 = paddle._C_ops.reshape(t685, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t687 = paddle._C_ops.transpose(t686, [2, 0, 3, 1, 4])
        del t686

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t688 = paddle._C_ops.slice(t687, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t689 = paddle._C_ops.slice(t687, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t690 = paddle._C_ops.slice(t687, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t691 = paddle._C_ops.transpose(t689, [0, 1, 3, 2])
        del t689

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t692 = paddle._C_ops.matmul(t688, t691, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t693 = paddle._C_ops.scale(t692, t465, float("0"), True)
        del t692

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t694 = paddle._C_ops.softmax(t693, -1)
        del t693

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t695 = paddle._C_ops.matmul(t694, t690, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t696 = paddle._C_ops.transpose(t695, [0, 2, 1, 3])
        del t695

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t697 = paddle._C_ops.reshape(t696, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t698 = paddle._C_ops.matmul(t697, t91, False, False)
        del t91

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t699 = paddle._C_ops.add(t698, t92)
        del t92

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t700 = paddle._C_ops.add(t680, t699)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t701, t702, t703 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t700, t93, t94, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t94, t93

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t704 = paddle._C_ops.matmul(t701, t95, False, False)
        del t95

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t705 = paddle._C_ops.add(t704, t96)
        del t96

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t706 = paddle._C_ops.gelu(t705, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t707 = paddle._C_ops.matmul(t706, t97, False, False)
        del t97

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t708 = paddle._C_ops.add(t707, t98)
        del t98

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t709 = paddle._C_ops.add(t700, t708)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t710, t711, t712 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t709, t99, t100, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t100, t99

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t713 = paddle._C_ops.matmul(t710, t101, False, False)
        del t101

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t714 = paddle._C_ops.add(t713, t102)
        del t102

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t715 = paddle._C_ops.reshape(t714, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t716 = paddle._C_ops.transpose(t715, [2, 0, 3, 1, 4])
        del t715

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t717 = paddle._C_ops.slice(t716, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t718 = paddle._C_ops.slice(t716, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t719 = paddle._C_ops.slice(t716, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t720 = paddle._C_ops.transpose(t718, [0, 1, 3, 2])
        del t718

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t721 = paddle._C_ops.matmul(t717, t720, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t722 = paddle._C_ops.scale(t721, t465, float("0"), True)
        del t721

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t723 = paddle._C_ops.softmax(t722, -1)
        del t722

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t724 = paddle._C_ops.matmul(t723, t719, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t725 = paddle._C_ops.transpose(t724, [0, 2, 1, 3])
        del t724

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t726 = paddle._C_ops.reshape(t725, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t727 = paddle._C_ops.matmul(t726, t103, False, False)
        del t103

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t728 = paddle._C_ops.add(t727, t104)
        del t104

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t729 = paddle._C_ops.add(t709, t728)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t730, t731, t732 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t729, t105, t106, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t106, t105

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t733 = paddle._C_ops.matmul(t730, t107, False, False)
        del t107

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t734 = paddle._C_ops.add(t733, t108)
        del t108

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t735 = paddle._C_ops.gelu(t734, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t736 = paddle._C_ops.matmul(t735, t109, False, False)
        del t109

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t737 = paddle._C_ops.add(t736, t110)
        del t110

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t738 = paddle._C_ops.add(t729, t737)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t739, t740, t741 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t738, t111, t112, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t112, t111

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t742 = paddle._C_ops.matmul(t739, t113, False, False)
        del t113

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t743 = paddle._C_ops.add(t742, t114)
        del t114

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t744 = paddle._C_ops.reshape(t743, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t745 = paddle._C_ops.transpose(t744, [2, 0, 3, 1, 4])
        del t744

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t746 = paddle._C_ops.slice(t745, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t747 = paddle._C_ops.slice(t745, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t748 = paddle._C_ops.slice(t745, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t749 = paddle._C_ops.transpose(t747, [0, 1, 3, 2])
        del t747

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t750 = paddle._C_ops.matmul(t746, t749, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t751 = paddle._C_ops.scale(t750, t465, float("0"), True)
        del t750

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t752 = paddle._C_ops.softmax(t751, -1)
        del t751

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t753 = paddle._C_ops.matmul(t752, t748, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t754 = paddle._C_ops.transpose(t753, [0, 2, 1, 3])
        del t753

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t755 = paddle._C_ops.reshape(t754, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t756 = paddle._C_ops.matmul(t755, t115, False, False)
        del t115

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t757 = paddle._C_ops.add(t756, t116)
        del t116

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t758 = paddle._C_ops.add(t738, t757)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t759, t760, t761 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t758, t117, t118, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t118, t117

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t762 = paddle._C_ops.matmul(t759, t119, False, False)
        del t119

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t763 = paddle._C_ops.add(t762, t120)
        del t120

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t764 = paddle._C_ops.gelu(t763, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t765 = paddle._C_ops.matmul(t764, t121, False, False)
        del t121

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t766 = paddle._C_ops.add(t765, t122)
        del t122

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t767 = paddle._C_ops.add(t758, t766)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t768, t769, t770 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t767, t123, t124, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t124, t123

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t771 = paddle._C_ops.matmul(t768, t125, False, False)
        del t125

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t772 = paddle._C_ops.add(t771, t126)
        del t126

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t773 = paddle._C_ops.reshape(t772, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t774 = paddle._C_ops.transpose(t773, [2, 0, 3, 1, 4])
        del t773

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t775 = paddle._C_ops.slice(t774, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t776 = paddle._C_ops.slice(t774, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t777 = paddle._C_ops.slice(t774, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t778 = paddle._C_ops.transpose(t776, [0, 1, 3, 2])
        del t776

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t779 = paddle._C_ops.matmul(t775, t778, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t780 = paddle._C_ops.scale(t779, t465, float("0"), True)
        del t779

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t781 = paddle._C_ops.softmax(t780, -1)
        del t780

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t782 = paddle._C_ops.matmul(t781, t777, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t783 = paddle._C_ops.transpose(t782, [0, 2, 1, 3])
        del t782

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t784 = paddle._C_ops.reshape(t783, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t785 = paddle._C_ops.matmul(t784, t127, False, False)
        del t127

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t786 = paddle._C_ops.add(t785, t128)
        del t128

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t787 = paddle._C_ops.add(t767, t786)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t788, t789, t790 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t787, t129, t130, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t130, t129

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t791 = paddle._C_ops.matmul(t788, t131, False, False)
        del t131

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t792 = paddle._C_ops.add(t791, t132)
        del t132

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t793 = paddle._C_ops.gelu(t792, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t794 = paddle._C_ops.matmul(t793, t133, False, False)
        del t133

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t795 = paddle._C_ops.add(t794, t134)
        del t134

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t796 = paddle._C_ops.add(t787, t795)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t797, t798, t799 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t796, t135, t136, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t136, t135

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t800 = paddle._C_ops.matmul(t797, t137, False, False)
        del t137

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t801 = paddle._C_ops.add(t800, t138)
        del t138

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t802 = paddle._C_ops.reshape(t801, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t803 = paddle._C_ops.transpose(t802, [2, 0, 3, 1, 4])
        del t802

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t804 = paddle._C_ops.slice(t803, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t805 = paddle._C_ops.slice(t803, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t806 = paddle._C_ops.slice(t803, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t807 = paddle._C_ops.transpose(t805, [0, 1, 3, 2])
        del t805

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t808 = paddle._C_ops.matmul(t804, t807, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t809 = paddle._C_ops.scale(t808, t465, float("0"), True)
        del t808

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t810 = paddle._C_ops.softmax(t809, -1)
        del t809

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t811 = paddle._C_ops.matmul(t810, t806, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t812 = paddle._C_ops.transpose(t811, [0, 2, 1, 3])
        del t811

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t813 = paddle._C_ops.reshape(t812, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t814 = paddle._C_ops.matmul(t813, t139, False, False)
        del t139

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t815 = paddle._C_ops.add(t814, t140)
        del t140

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t816 = paddle._C_ops.add(t796, t815)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t817, t818, t819 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t816, t141, t142, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t142, t141

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t820 = paddle._C_ops.matmul(t817, t143, False, False)
        del t143

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t821 = paddle._C_ops.add(t820, t144)
        del t144

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t822 = paddle._C_ops.gelu(t821, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t823 = paddle._C_ops.matmul(t822, t145, False, False)
        del t145

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t824 = paddle._C_ops.add(t823, t146)
        del t146

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t825 = paddle._C_ops.add(t816, t824)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t826, t827, t828 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t825, t147, t148, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t148, t147

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t829 = paddle._C_ops.matmul(t826, t149, False, False)
        del t149

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t830 = paddle._C_ops.add(t829, t150)
        del t150

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t831 = paddle._C_ops.reshape(t830, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t832 = paddle._C_ops.transpose(t831, [2, 0, 3, 1, 4])
        del t831

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t833 = paddle._C_ops.slice(t832, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t834 = paddle._C_ops.slice(t832, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t835 = paddle._C_ops.slice(t832, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t836 = paddle._C_ops.transpose(t834, [0, 1, 3, 2])
        del t834

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t837 = paddle._C_ops.matmul(t833, t836, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t838 = paddle._C_ops.scale(t837, t465, float("0"), True)
        del t837

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t839 = paddle._C_ops.softmax(t838, -1)
        del t838

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t840 = paddle._C_ops.matmul(t839, t835, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t841 = paddle._C_ops.transpose(t840, [0, 2, 1, 3])
        del t840

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t842 = paddle._C_ops.reshape(t841, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t843 = paddle._C_ops.matmul(t842, t151, False, False)
        del t151

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t844 = paddle._C_ops.add(t843, t152)
        del t152

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t845 = paddle._C_ops.add(t825, t844)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t846, t847, t848 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t845, t153, t154, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t154, t153

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t849 = paddle._C_ops.matmul(t846, t155, False, False)
        del t155

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t850 = paddle._C_ops.add(t849, t156)
        del t156

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t851 = paddle._C_ops.gelu(t850, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t852 = paddle._C_ops.matmul(t851, t157, False, False)
        del t157

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t853 = paddle._C_ops.add(t852, t158)
        del t158

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t854 = paddle._C_ops.add(t845, t853)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t855, t856, t857 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t854, t159, t160, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t160, t159

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t858 = paddle._C_ops.matmul(t855, t161, False, False)
        del t161

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t859 = paddle._C_ops.add(t858, t162)
        del t162

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t860 = paddle._C_ops.reshape(t859, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t861 = paddle._C_ops.transpose(t860, [2, 0, 3, 1, 4])
        del t860

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t862 = paddle._C_ops.slice(t861, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t863 = paddle._C_ops.slice(t861, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t864 = paddle._C_ops.slice(t861, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t865 = paddle._C_ops.transpose(t863, [0, 1, 3, 2])
        del t863

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t866 = paddle._C_ops.matmul(t862, t865, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t867 = paddle._C_ops.scale(t866, t465, float("0"), True)
        del t866

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t868 = paddle._C_ops.softmax(t867, -1)
        del t867

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t869 = paddle._C_ops.matmul(t868, t864, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t870 = paddle._C_ops.transpose(t869, [0, 2, 1, 3])
        del t869

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t871 = paddle._C_ops.reshape(t870, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t872 = paddle._C_ops.matmul(t871, t163, False, False)
        del t163

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t873 = paddle._C_ops.add(t872, t164)
        del t164

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t874 = paddle._C_ops.add(t854, t873)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t875, t876, t877 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t874, t165, t166, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t166, t165

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t878 = paddle._C_ops.matmul(t875, t167, False, False)
        del t167

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t879 = paddle._C_ops.add(t878, t168)
        del t168

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t880 = paddle._C_ops.gelu(t879, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t881 = paddle._C_ops.matmul(t880, t169, False, False)
        del t169

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t882 = paddle._C_ops.add(t881, t170)
        del t170

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t883 = paddle._C_ops.add(t874, t882)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t884, t885, t886 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t883, t171, t172, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t172, t171

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t887 = paddle._C_ops.matmul(t884, t173, False, False)
        del t173

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t888 = paddle._C_ops.add(t887, t174)
        del t174

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t889 = paddle._C_ops.reshape(t888, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t890 = paddle._C_ops.transpose(t889, [2, 0, 3, 1, 4])
        del t889

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t891 = paddle._C_ops.slice(t890, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t892 = paddle._C_ops.slice(t890, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t893 = paddle._C_ops.slice(t890, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t894 = paddle._C_ops.transpose(t892, [0, 1, 3, 2])
        del t892

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t895 = paddle._C_ops.matmul(t891, t894, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t896 = paddle._C_ops.scale(t895, t465, float("0"), True)
        del t895

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t897 = paddle._C_ops.softmax(t896, -1)
        del t896

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t898 = paddle._C_ops.matmul(t897, t893, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t899 = paddle._C_ops.transpose(t898, [0, 2, 1, 3])
        del t898

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t900 = paddle._C_ops.reshape(t899, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t901 = paddle._C_ops.matmul(t900, t175, False, False)
        del t175

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t902 = paddle._C_ops.add(t901, t176)
        del t176

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t903 = paddle._C_ops.add(t883, t902)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t904, t905, t906 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t903, t177, t178, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t178, t177

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t907 = paddle._C_ops.matmul(t904, t179, False, False)
        del t179

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t908 = paddle._C_ops.add(t907, t180)
        del t180

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t909 = paddle._C_ops.gelu(t908, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t910 = paddle._C_ops.matmul(t909, t181, False, False)
        del t181

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t911 = paddle._C_ops.add(t910, t182)
        del t182

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t912 = paddle._C_ops.add(t903, t911)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t913, t914, t915 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t912, t183, t184, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t184, t183

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t916 = paddle._C_ops.matmul(t913, t185, False, False)
        del t185

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t917 = paddle._C_ops.add(t916, t186)
        del t186

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t918 = paddle._C_ops.reshape(t917, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t919 = paddle._C_ops.transpose(t918, [2, 0, 3, 1, 4])
        del t918

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t920 = paddle._C_ops.slice(t919, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t921 = paddle._C_ops.slice(t919, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t922 = paddle._C_ops.slice(t919, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t923 = paddle._C_ops.transpose(t921, [0, 1, 3, 2])
        del t921

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t924 = paddle._C_ops.matmul(t920, t923, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t925 = paddle._C_ops.scale(t924, t465, float("0"), True)
        del t924

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t926 = paddle._C_ops.softmax(t925, -1)
        del t925

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t927 = paddle._C_ops.matmul(t926, t922, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t928 = paddle._C_ops.transpose(t927, [0, 2, 1, 3])
        del t927

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t929 = paddle._C_ops.reshape(t928, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t930 = paddle._C_ops.matmul(t929, t187, False, False)
        del t187

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t931 = paddle._C_ops.add(t930, t188)
        del t188

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t932 = paddle._C_ops.add(t912, t931)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t933, t934, t935 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t932, t189, t190, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t190, t189

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t936 = paddle._C_ops.matmul(t933, t191, False, False)
        del t191

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t937 = paddle._C_ops.add(t936, t192)
        del t192

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t938 = paddle._C_ops.gelu(t937, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t939 = paddle._C_ops.matmul(t938, t193, False, False)
        del t193

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t940 = paddle._C_ops.add(t939, t194)
        del t194

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t941 = paddle._C_ops.add(t932, t940)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t942, t943, t944 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t941, t195, t196, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t196, t195

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t945 = paddle._C_ops.matmul(t942, t197, False, False)
        del t197

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t946 = paddle._C_ops.add(t945, t198)
        del t198

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t947 = paddle._C_ops.reshape(t946, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t948 = paddle._C_ops.transpose(t947, [2, 0, 3, 1, 4])
        del t947

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t949 = paddle._C_ops.slice(t948, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t950 = paddle._C_ops.slice(t948, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t951 = paddle._C_ops.slice(t948, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t952 = paddle._C_ops.transpose(t950, [0, 1, 3, 2])
        del t950

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t953 = paddle._C_ops.matmul(t949, t952, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t954 = paddle._C_ops.scale(t953, t465, float("0"), True)
        del t953

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t955 = paddle._C_ops.softmax(t954, -1)
        del t954

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t956 = paddle._C_ops.matmul(t955, t951, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t957 = paddle._C_ops.transpose(t956, [0, 2, 1, 3])
        del t956

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t958 = paddle._C_ops.reshape(t957, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t959 = paddle._C_ops.matmul(t958, t199, False, False)
        del t199

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t960 = paddle._C_ops.add(t959, t200)
        del t200

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t961 = paddle._C_ops.add(t941, t960)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t962, t963, t964 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t961, t201, t202, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t202, t201

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t965 = paddle._C_ops.matmul(t962, t203, False, False)
        del t203

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t966 = paddle._C_ops.add(t965, t204)
        del t204

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t967 = paddle._C_ops.gelu(t966, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t968 = paddle._C_ops.matmul(t967, t205, False, False)
        del t205

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t969 = paddle._C_ops.add(t968, t206)
        del t206

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t970 = paddle._C_ops.add(t961, t969)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t971, t972, t973 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t970, t207, t208, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t208, t207

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t974 = paddle._C_ops.matmul(t971, t209, False, False)
        del t209

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t975 = paddle._C_ops.add(t974, t210)
        del t210

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t976 = paddle._C_ops.reshape(t975, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t977 = paddle._C_ops.transpose(t976, [2, 0, 3, 1, 4])
        del t976

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t978 = paddle._C_ops.slice(t977, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t979 = paddle._C_ops.slice(t977, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t980 = paddle._C_ops.slice(t977, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t981 = paddle._C_ops.transpose(t979, [0, 1, 3, 2])
        del t979

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t982 = paddle._C_ops.matmul(t978, t981, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t983 = paddle._C_ops.scale(t982, t465, float("0"), True)
        del t982

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t984 = paddle._C_ops.softmax(t983, -1)
        del t983

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t985 = paddle._C_ops.matmul(t984, t980, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t986 = paddle._C_ops.transpose(t985, [0, 2, 1, 3])
        del t985

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t987 = paddle._C_ops.reshape(t986, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t988 = paddle._C_ops.matmul(t987, t211, False, False)
        del t211

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t989 = paddle._C_ops.add(t988, t212)
        del t212

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t990 = paddle._C_ops.add(t970, t989)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t991, t992, t993 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t990, t213, t214, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t214, t213

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t994 = paddle._C_ops.matmul(t991, t215, False, False)
        del t215

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t995 = paddle._C_ops.add(t994, t216)
        del t216

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t996 = paddle._C_ops.gelu(t995, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t997 = paddle._C_ops.matmul(t996, t217, False, False)
        del t217

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t998 = paddle._C_ops.add(t997, t218)
        del t218

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t999 = paddle._C_ops.add(t990, t998)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1000, t1001, t1002 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t999, t219, t220, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t220, t219

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t1003 = paddle._C_ops.matmul(t1000, t221, False, False)
        del t221

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t1004 = paddle._C_ops.add(t1003, t222)
        del t222

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t1005 = paddle._C_ops.reshape(t1004, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t1006 = paddle._C_ops.transpose(t1005, [2, 0, 3, 1, 4])
        del t1005

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1007 = paddle._C_ops.slice(t1006, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1008 = paddle._C_ops.slice(t1006, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1009 = paddle._C_ops.slice(t1006, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t1010 = paddle._C_ops.transpose(t1008, [0, 1, 3, 2])
        del t1008

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t1011 = paddle._C_ops.matmul(t1007, t1010, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t1012 = paddle._C_ops.scale(t1011, t465, float("0"), True)
        del t1011

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t1013 = paddle._C_ops.softmax(t1012, -1)
        del t1012

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t1014 = paddle._C_ops.matmul(t1013, t1009, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t1015 = paddle._C_ops.transpose(t1014, [0, 2, 1, 3])
        del t1014

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t1016 = paddle._C_ops.reshape(t1015, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t1017 = paddle._C_ops.matmul(t1016, t223, False, False)
        del t223

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1018 = paddle._C_ops.add(t1017, t224)
        del t224

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1019 = paddle._C_ops.add(t999, t1018)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1020, t1021, t1022 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1019, t225, t226, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t226, t225

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t1023 = paddle._C_ops.matmul(t1020, t227, False, False)
        del t227

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t1024 = paddle._C_ops.add(t1023, t228)
        del t228

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t1025 = paddle._C_ops.gelu(t1024, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t1026 = paddle._C_ops.matmul(t1025, t229, False, False)
        del t229

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1027 = paddle._C_ops.add(t1026, t230)
        del t230

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1028 = paddle._C_ops.add(t1019, t1027)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1029, t1030, t1031 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1028, t231, t232, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t232, t231

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t1032 = paddle._C_ops.matmul(t1029, t233, False, False)
        del t233

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t1033 = paddle._C_ops.add(t1032, t234)
        del t234

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t1034 = paddle._C_ops.reshape(t1033, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t1035 = paddle._C_ops.transpose(t1034, [2, 0, 3, 1, 4])
        del t1034

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1036 = paddle._C_ops.slice(t1035, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1037 = paddle._C_ops.slice(t1035, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1038 = paddle._C_ops.slice(t1035, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t1039 = paddle._C_ops.transpose(t1037, [0, 1, 3, 2])
        del t1037

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t1040 = paddle._C_ops.matmul(t1036, t1039, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t1041 = paddle._C_ops.scale(t1040, t465, float("0"), True)
        del t1040

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t1042 = paddle._C_ops.softmax(t1041, -1)
        del t1041

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t1043 = paddle._C_ops.matmul(t1042, t1038, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t1044 = paddle._C_ops.transpose(t1043, [0, 2, 1, 3])
        del t1043

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t1045 = paddle._C_ops.reshape(t1044, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t1046 = paddle._C_ops.matmul(t1045, t235, False, False)
        del t235

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1047 = paddle._C_ops.add(t1046, t236)
        del t236

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1048 = paddle._C_ops.add(t1028, t1047)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1049, t1050, t1051 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1048, t237, t238, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t238, t237

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t1052 = paddle._C_ops.matmul(t1049, t239, False, False)
        del t239

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t1053 = paddle._C_ops.add(t1052, t240)
        del t240

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t1054 = paddle._C_ops.gelu(t1053, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t1055 = paddle._C_ops.matmul(t1054, t241, False, False)
        del t241

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1056 = paddle._C_ops.add(t1055, t242)
        del t242

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1057 = paddle._C_ops.add(t1048, t1056)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1058, t1059, t1060 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1057, t243, t244, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t244, t243

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t1061 = paddle._C_ops.matmul(t1058, t245, False, False)
        del t245

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t1062 = paddle._C_ops.add(t1061, t246)
        del t246

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t1063 = paddle._C_ops.reshape(t1062, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t1064 = paddle._C_ops.transpose(t1063, [2, 0, 3, 1, 4])
        del t1063

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1065 = paddle._C_ops.slice(t1064, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1066 = paddle._C_ops.slice(t1064, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1067 = paddle._C_ops.slice(t1064, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t1068 = paddle._C_ops.transpose(t1066, [0, 1, 3, 2])
        del t1066

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t1069 = paddle._C_ops.matmul(t1065, t1068, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t1070 = paddle._C_ops.scale(t1069, t465, float("0"), True)
        del t1069

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t1071 = paddle._C_ops.softmax(t1070, -1)
        del t1070

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t1072 = paddle._C_ops.matmul(t1071, t1067, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t1073 = paddle._C_ops.transpose(t1072, [0, 2, 1, 3])
        del t1072

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t1074 = paddle._C_ops.reshape(t1073, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t1075 = paddle._C_ops.matmul(t1074, t247, False, False)
        del t247

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1076 = paddle._C_ops.add(t1075, t248)
        del t248

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1077 = paddle._C_ops.add(t1057, t1076)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1078, t1079, t1080 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1077, t249, t250, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t250, t249

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t1081 = paddle._C_ops.matmul(t1078, t251, False, False)
        del t251

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t1082 = paddle._C_ops.add(t1081, t252)
        del t252

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t1083 = paddle._C_ops.gelu(t1082, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t1084 = paddle._C_ops.matmul(t1083, t253, False, False)
        del t253

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1085 = paddle._C_ops.add(t1084, t254)
        del t254

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1086 = paddle._C_ops.add(t1077, t1085)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1087, t1088, t1089 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1086, t255, t256, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t256, t255

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t1090 = paddle._C_ops.matmul(t1087, t257, False, False)
        del t257

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t1091 = paddle._C_ops.add(t1090, t258)
        del t258

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t1092 = paddle._C_ops.reshape(t1091, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t1093 = paddle._C_ops.transpose(t1092, [2, 0, 3, 1, 4])
        del t1092

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1094 = paddle._C_ops.slice(t1093, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1095 = paddle._C_ops.slice(t1093, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1096 = paddle._C_ops.slice(t1093, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t1097 = paddle._C_ops.transpose(t1095, [0, 1, 3, 2])
        del t1095

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t1098 = paddle._C_ops.matmul(t1094, t1097, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t1099 = paddle._C_ops.scale(t1098, t465, float("0"), True)
        del t1098

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t1100 = paddle._C_ops.softmax(t1099, -1)
        del t1099

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t1101 = paddle._C_ops.matmul(t1100, t1096, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t1102 = paddle._C_ops.transpose(t1101, [0, 2, 1, 3])
        del t1101

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t1103 = paddle._C_ops.reshape(t1102, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t1104 = paddle._C_ops.matmul(t1103, t259, False, False)
        del t259

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1105 = paddle._C_ops.add(t1104, t260)
        del t260

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1106 = paddle._C_ops.add(t1086, t1105)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1107, t1108, t1109 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1106, t261, t262, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t262, t261

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t1110 = paddle._C_ops.matmul(t1107, t263, False, False)
        del t263

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t1111 = paddle._C_ops.add(t1110, t264)
        del t264

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t1112 = paddle._C_ops.gelu(t1111, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t1113 = paddle._C_ops.matmul(t1112, t265, False, False)
        del t265

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1114 = paddle._C_ops.add(t1113, t266)
        del t266

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1115 = paddle._C_ops.add(t1106, t1114)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1116, t1117, t1118 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1115, t267, t268, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t268, t267

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t1119 = paddle._C_ops.matmul(t1116, t269, False, False)
        del t269

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t1120 = paddle._C_ops.add(t1119, t270)
        del t270

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t1121 = paddle._C_ops.reshape(t1120, t311)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t1122 = paddle._C_ops.transpose(t1121, [2, 0, 3, 1, 4])
        del t1121

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1123 = paddle._C_ops.slice(t1122, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1124 = paddle._C_ops.slice(t1122, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1125 = paddle._C_ops.slice(t1122, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t1126 = paddle._C_ops.transpose(t1124, [0, 1, 3, 2])
        del t1124

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t1127 = paddle._C_ops.matmul(t1123, t1126, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t1128 = paddle._C_ops.scale(t1127, t465, float("0"), True)
        del t1127

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t1129 = paddle._C_ops.softmax(t1128, -1)
        del t1128

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t1130 = paddle._C_ops.matmul(t1129, t1125, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t1131 = paddle._C_ops.transpose(t1130, [0, 2, 1, 3])
        del t1130

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t1132 = paddle._C_ops.reshape(t1131, t493)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t1133 = paddle._C_ops.matmul(t1132, t271, False, False)
        del t271

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1134 = paddle._C_ops.add(t1133, t272)
        del t272

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1135 = paddle._C_ops.add(t1115, t1134)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1136, t1137, t1138 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1135, t273, t274, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t274, t273

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t1139 = paddle._C_ops.matmul(t1136, t275, False, False)
        del t275

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t1140 = paddle._C_ops.add(t1139, t276)
        del t276

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t1141 = paddle._C_ops.gelu(t1140, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t1142 = paddle._C_ops.matmul(t1141, t277, False, False)
        del t277

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1143 = paddle._C_ops.add(t1142, t278)
        del t278

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1144 = paddle._C_ops.add(t1135, t1143)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1145, t1146, t1147 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1144, t279, t280, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t280, t279

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        t1148 = paddle._C_ops.matmul(t1145, t281, False, False)
        del t281

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        t1149 = paddle._C_ops.add(t1148, t282)
        del t282

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        t1150 = paddle._C_ops.reshape(t1149, t311)
        del t311

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        t1151 = paddle._C_ops.transpose(t1150, [2, 0, 3, 1, 4])
        del t1150

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1152 = paddle._C_ops.slice(t1151, [0], t314, t339, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1153 = paddle._C_ops.slice(t1151, [0], t339, t389, [1], [0])

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        t1154 = paddle._C_ops.slice(t1151, [0], t389, t438, [1], [0])

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        t1155 = paddle._C_ops.transpose(t1153, [0, 1, 3, 2])
        del t1153

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        t1156 = paddle._C_ops.matmul(t1152, t1155, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        t1157 = paddle._C_ops.scale(t1156, t465, float("0"), True)
        del t1156

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        t1158 = paddle._C_ops.softmax(t1157, -1)
        del t1157

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        t1159 = paddle._C_ops.matmul(t1158, t1154, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        t1160 = paddle._C_ops.transpose(t1159, [0, 2, 1, 3])
        del t1159

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        t1161 = paddle._C_ops.reshape(t1160, t493)
        del t493

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        t1162 = paddle._C_ops.matmul(t1161, t283, False, False)
        del t283

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1163 = paddle._C_ops.add(t1162, t284)
        del t284

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1164 = paddle._C_ops.add(t1144, t1163)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        t1165, t1166, t1167 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1164, t285, t286, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t286, t285

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        t1168 = paddle._C_ops.matmul(t1165, t287, False, False)
        del t287

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        t1169 = paddle._C_ops.add(t1168, t288)
        del t288

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        t1170 = paddle._C_ops.gelu(t1169, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        t1171 = paddle._C_ops.matmul(t1170, t289, False, False)
        del t289

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        t1172 = paddle._C_ops.add(t1171, t290)
        del t290

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        t1173 = paddle._C_ops.add(t1164, t1172)

        # pd_op.slice: (16x1024xf32) <- (16x257x1024xf32, 1xi64, 1xi64)
        t1174 = paddle._C_ops.slice(t1173, [1], t314, t339, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        t1175 = [2147483647]

        # pd_op.slice: (16x256x1024xf32) <- (16x257x1024xf32, 1xi64, 1xi64)
        t1176 = paddle._C_ops.slice(t1173, [1], t339, t1175, [1], [])
        del t1175

        # pd_op.layer_norm: (16x1024xf32, 16xf32, 16xf32) <- (16x1024xf32, 1024xf32, 1024xf32)
        t1177, t1178, t1179 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t1174, t291, t292, float("1e-05"), 1),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t292, t291

        # pd_op.matmul: (16x102xf32) <- (16x1024xf32, 1024x102xf32)
        t1180 = paddle._C_ops.matmul(t1177, t293, False, False)
        del t293

        return (
            t294,
            t296,
            t297,
            t298,
            t299,
            t301,
            t302,
            t303,
            t304,
            t305,
            t306,
            t307,
            t308,
            t309,
            t310,
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
            t403,
            t404,
            t405,
            t406,
            t407,
            t408,
            t409,
            t410,
            t411,
            t412,
            t413,
            t414,
            t415,
            t416,
            t417,
            t418,
            t419,
            t420,
            t421,
            t422,
            t423,
            t424,
            t425,
            t426,
            t427,
            t428,
            t429,
            t430,
            t431,
            t432,
            t433,
            t434,
            t435,
            t436,
            t438,
            t439,
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
            t490,
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
            t503,
            t504,
            t505,
            t506,
            t507,
            t508,
            t509,
            t510,
            t511,
            t513,
            t514,
            t516,
            t517,
            t520,
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
            t542,
            t543,
            t545,
            t546,
            t549,
            t551,
            t552,
            t553,
            t554,
            t555,
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
            t571,
            t572,
            t574,
            t575,
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
            t596,
            t597,
            t598,
            t600,
            t601,
            t603,
            t604,
            t607,
            t609,
            t610,
            t611,
            t612,
            t613,
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
            t629,
            t630,
            t632,
            t633,
            t636,
            t638,
            t639,
            t640,
            t641,
            t642,
            t643,
            t644,
            t645,
            t646,
            t647,
            t648,
            t649,
            t650,
            t651,
            t652,
            t653,
            t654,
            t655,
            t656,
            t658,
            t659,
            t661,
            t662,
            t665,
            t667,
            t668,
            t669,
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
            t690,
            t691,
            t694,
            t696,
            t697,
            t698,
            t699,
            t700,
            t701,
            t702,
            t703,
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
            t716,
            t717,
            t719,
            t720,
            t723,
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
            t737,
            t738,
            t739,
            t740,
            t741,
            t742,
            t743,
            t745,
            t746,
            t748,
            t749,
            t752,
            t754,
            t755,
            t756,
            t757,
            t758,
            t759,
            t760,
            t761,
            t762,
            t763,
            t764,
            t765,
            t766,
            t767,
            t768,
            t769,
            t770,
            t771,
            t772,
            t774,
            t775,
            t777,
            t778,
            t781,
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
            t800,
            t801,
            t803,
            t804,
            t806,
            t807,
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
            t828,
            t829,
            t830,
            t832,
            t833,
            t835,
            t836,
            t839,
            t841,
            t842,
            t843,
            t844,
            t845,
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
            t861,
            t862,
            t864,
            t865,
            t868,
            t870,
            t871,
            t872,
            t873,
            t874,
            t875,
            t876,
            t877,
            t878,
            t879,
            t880,
            t881,
            t882,
            t883,
            t884,
            t885,
            t886,
            t887,
            t888,
            t890,
            t891,
            t893,
            t894,
            t897,
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
            t909,
            t910,
            t911,
            t912,
            t913,
            t914,
            t915,
            t916,
            t917,
            t919,
            t920,
            t922,
            t923,
            t926,
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
            t943,
            t944,
            t945,
            t946,
            t948,
            t949,
            t951,
            t952,
            t955,
            t957,
            t958,
            t959,
            t960,
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
            t977,
            t978,
            t980,
            t981,
            t984,
            t986,
            t987,
            t988,
            t989,
            t990,
            t991,
            t992,
            t993,
            t994,
            t995,
            t996,
            t997,
            t998,
            t999,
            t1000,
            t1001,
            t1002,
            t1003,
            t1004,
            t1006,
            t1007,
            t1009,
            t1010,
            t1013,
            t1015,
            t1016,
            t1017,
            t1018,
            t1019,
            t1020,
            t1021,
            t1022,
            t1023,
            t1024,
            t1025,
            t1026,
            t1027,
            t1028,
            t1029,
            t1030,
            t1031,
            t1032,
            t1033,
            t1035,
            t1036,
            t1038,
            t1039,
            t1042,
            t1044,
            t1045,
            t1046,
            t1047,
            t1048,
            t1049,
            t1050,
            t1051,
            t1052,
            t1053,
            t1054,
            t1055,
            t1056,
            t1057,
            t1058,
            t1059,
            t1060,
            t1061,
            t1062,
            t1064,
            t1065,
            t1067,
            t1068,
            t1071,
            t1073,
            t1074,
            t1075,
            t1076,
            t1077,
            t1078,
            t1079,
            t1080,
            t1081,
            t1082,
            t1083,
            t1084,
            t1085,
            t1086,
            t1087,
            t1088,
            t1089,
            t1090,
            t1091,
            t1093,
            t1094,
            t1096,
            t1097,
            t1100,
            t1102,
            t1103,
            t1104,
            t1105,
            t1106,
            t1107,
            t1108,
            t1109,
            t1110,
            t1111,
            t1112,
            t1113,
            t1114,
            t1115,
            t1116,
            t1117,
            t1118,
            t1119,
            t1120,
            t1122,
            t1123,
            t1125,
            t1126,
            t1129,
            t1131,
            t1132,
            t1133,
            t1134,
            t1135,
            t1136,
            t1137,
            t1138,
            t1139,
            t1140,
            t1141,
            t1142,
            t1143,
            t1144,
            t1145,
            t1146,
            t1147,
            t1148,
            t1149,
            t1151,
            t1152,
            t1154,
            t1155,
            t1158,
            t1160,
            t1161,
            t1162,
            t1163,
            t1164,
            t1165,
            t1166,
            t1167,
            t1168,
            t1169,
            t1170,
            t1171,
            t1172,
            t1173,
            t1174,
            t1177,
            t1178,
            t1179,
            t1180,
        )
