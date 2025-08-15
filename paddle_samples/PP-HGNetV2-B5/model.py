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
        t305: paddle.Tensor,
        t306: paddle.Tensor,
        t307: paddle.Tensor,
        t308: paddle.Tensor,
        t309: paddle.Tensor,
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
        t340: paddle.Tensor,
        t341: paddle.Tensor,
        t342: paddle.Tensor,
        t343: paddle.Tensor,
        t344: paddle.Tensor,
        t345: paddle.Tensor,
        t346: paddle.Tensor,
        t347: paddle.Tensor,
        t348: paddle.Tensor,
        t349: paddle.Tensor,
        t350: paddle.Tensor,
        t351: paddle.Tensor,
        t352: paddle.Tensor,
        t353: paddle.Tensor,
        t354: paddle.Tensor,
        t355: paddle.Tensor,
        t356: paddle.Tensor,
        t357: paddle.Tensor,
        t358: paddle.Tensor,
        t359: paddle.Tensor,
        t360: paddle.Tensor,
        t361: paddle.Tensor,
        t362: paddle.Tensor,
        t363: paddle.Tensor,
        t364: paddle.Tensor,
        t365: paddle.Tensor,
        t366: paddle.Tensor,
        t367: paddle.Tensor,
        t368: paddle.Tensor,
        t369: paddle.Tensor,
        t370: paddle.Tensor,
        t371: paddle.Tensor,
        t372: paddle.Tensor,
        t373: paddle.Tensor,
        t374: paddle.Tensor,
        t375: paddle.Tensor,
        t376: paddle.Tensor,
        t377: paddle.Tensor,
        t378: paddle.Tensor,
        t379: paddle.Tensor,
        t380: paddle.Tensor,
        t381: paddle.Tensor,
        t382: paddle.Tensor,
        t383: paddle.Tensor,
        t384: paddle.Tensor,
        t385: paddle.Tensor,
        t386: paddle.Tensor,
        t387: paddle.Tensor,
        t388: paddle.Tensor,
        t389: paddle.Tensor,
        t390: paddle.Tensor,
        t391: paddle.Tensor,
        t392: paddle.Tensor,
        t393: paddle.Tensor,
        t394: paddle.Tensor,
        t395: paddle.Tensor,
        t396: paddle.Tensor,
        t397: paddle.Tensor,
        t398: paddle.Tensor,
        t399: paddle.Tensor,
        t400: paddle.Tensor,
        t401: paddle.Tensor,
        t402: paddle.Tensor,
        t403: paddle.Tensor,
        t404: paddle.Tensor,
        t405: paddle.Tensor,
        t406: paddle.Tensor,
        t407: paddle.Tensor,
        t408: paddle.Tensor,
        t409: paddle.Tensor,
        t410: paddle.Tensor,
        t411: paddle.Tensor,
        t412: paddle.Tensor,
        t413: paddle.Tensor,
        t414: paddle.Tensor,
        t415: paddle.Tensor,
        t416: paddle.Tensor,
        t417: paddle.Tensor,
        t418: paddle.Tensor,
        t419: paddle.Tensor,
        t420: paddle.Tensor,
        t421: paddle.Tensor,
        t422: paddle.Tensor,
        t423: paddle.Tensor,
        t424: paddle.Tensor,
        t425: paddle.Tensor,
        t426: paddle.Tensor,
        t427: paddle.Tensor,
        t428: paddle.Tensor,
        t429: paddle.Tensor,
        t430: paddle.Tensor,
        t431: paddle.Tensor,
        t432: paddle.Tensor,
        t433: paddle.Tensor,
        t434: paddle.Tensor,
        t435: paddle.Tensor,
        t436: paddle.Tensor,
        t437: paddle.Tensor,
        t438: paddle.Tensor,
        t439: paddle.Tensor,
        t440: paddle.Tensor,
        t441: paddle.Tensor,
        t442: paddle.Tensor,
        t443: paddle.Tensor,
        t444: paddle.Tensor,
        t445: paddle.Tensor,
        t446: paddle.Tensor,
        t447: paddle.Tensor,
        t448: paddle.Tensor,
        t449: paddle.Tensor,
        t450: paddle.Tensor,
        t451: paddle.Tensor,
        t452: paddle.Tensor,
        t453: paddle.Tensor,
        t454: paddle.Tensor,
        t455: paddle.Tensor,
        t456: paddle.Tensor,
        t457: paddle.Tensor,
        t458: paddle.Tensor,
        t459: paddle.Tensor,
        t460: paddle.Tensor,
        t461: paddle.Tensor,
        t462: paddle.Tensor,
        t463: paddle.Tensor,
        t464: paddle.Tensor,
        t465: paddle.Tensor,
        t466: paddle.Tensor,
        t467: paddle.Tensor,
        t468: paddle.Tensor,
        t469: paddle.Tensor,
        t470: paddle.Tensor,
        t471: paddle.Tensor,
        t472: paddle.Tensor,
        t473: paddle.Tensor,
        t474: paddle.Tensor,
        t475: paddle.Tensor,
        t476: paddle.Tensor,
        t477: paddle.Tensor,
        t478: paddle.Tensor,
        t479: paddle.Tensor,
        t480: paddle.Tensor,
        t481: paddle.Tensor,
        t482: paddle.Tensor,
        t483: paddle.Tensor,
        t484: paddle.Tensor,
        t485: paddle.Tensor,
        t486: paddle.Tensor,
        t487: paddle.Tensor,
        t488: paddle.Tensor,
        t489: paddle.Tensor,
        t490: paddle.Tensor,
        t491: paddle.Tensor,
        t492: paddle.Tensor,
        t493: paddle.Tensor,
        t494: paddle.Tensor,
        t495: paddle.Tensor,
        t496: paddle.Tensor,
        t497: paddle.Tensor,
        t498: paddle.Tensor,
        t499: paddle.Tensor,
        t500: paddle.Tensor,
        t501: paddle.Tensor,
        t502: paddle.Tensor,
        t503: paddle.Tensor,
        t504: paddle.Tensor,
        t505: paddle.Tensor,
        t506: paddle.Tensor,
        t507: paddle.Tensor,
        t508: paddle.Tensor,
        t509: paddle.Tensor,
        t510: paddle.Tensor,
        t511: paddle.Tensor,
        t512: paddle.Tensor,
        t513: paddle.Tensor,
        t514: paddle.Tensor,
        t515: paddle.Tensor,
        t516: paddle.Tensor,
        t517: paddle.Tensor,
        t518: paddle.Tensor,
        t519: paddle.Tensor,
        t520: paddle.Tensor,
        t521: paddle.Tensor,
        t522: paddle.Tensor,
        t523: paddle.Tensor,
        t524: paddle.Tensor,
        t525: paddle.Tensor,
        t526: paddle.Tensor,
        t527: paddle.Tensor,
        t528: paddle.Tensor,
        t529: paddle.Tensor,
        t530: paddle.Tensor,
        t531: paddle.Tensor,
        t532: paddle.Tensor,
        t533: paddle.Tensor,
        t534: paddle.Tensor,
        t535: paddle.Tensor,
        t536: paddle.Tensor,
        t537: paddle.Tensor,
        t538: paddle.Tensor,
        t539: paddle.Tensor,
        t540: paddle.Tensor,
        t541: paddle.Tensor,
        t542: paddle.Tensor,
        t543: paddle.Tensor,
        t544: paddle.Tensor,
        t545: paddle.Tensor,
        t546: paddle.Tensor,
        t547: paddle.Tensor,
        t548: paddle.Tensor,
        t549: paddle.Tensor,
        t550: paddle.Tensor,
        t551: paddle.Tensor,
        t552: paddle.Tensor,
        t553: paddle.Tensor,
        t554: paddle.Tensor,
        t555: paddle.Tensor,
        t556: paddle.Tensor,
        t557: paddle.Tensor,
        t558: paddle.Tensor,
        t559: paddle.Tensor,
        t560: paddle.Tensor,
        t561: paddle.Tensor,
        t562: paddle.Tensor,
        t563: paddle.Tensor,
        t564: paddle.Tensor,
        t565: paddle.Tensor,
        t566: paddle.Tensor,
        t567: paddle.Tensor,
        t568: paddle.Tensor,
        t569: paddle.Tensor,
        t570: paddle.Tensor,
        t571: paddle.Tensor,
        t572: paddle.Tensor,
        t573: paddle.Tensor,
        t574: paddle.Tensor,
        t575: paddle.Tensor,
        t576: paddle.Tensor,
        t577: paddle.Tensor,
        t578: paddle.Tensor,
        t579: paddle.Tensor,
        t580: paddle.Tensor,
        t581: paddle.Tensor,
        t582: paddle.Tensor,
        t583: paddle.Tensor,
        t584: paddle.Tensor,
        t585: paddle.Tensor,
        t586: paddle.Tensor,
        t587: paddle.Tensor,
        t588: paddle.Tensor,
        t589: paddle.Tensor,
        t590: paddle.Tensor,
        t591: paddle.Tensor,
        t592: paddle.Tensor,
        t593: paddle.Tensor,
        t594: paddle.Tensor,
        t595: paddle.Tensor,
        t596: paddle.Tensor,
        t597: paddle.Tensor,
        t598: paddle.Tensor,
        t599: paddle.Tensor,
        t600: paddle.Tensor,
        t601: paddle.Tensor,
        t602: paddle.Tensor,
        t603: paddle.Tensor,
        t604: paddle.Tensor,
        t605: paddle.Tensor,
        t606: paddle.Tensor,
        t607: paddle.Tensor,
        t608: paddle.Tensor,
        t609: paddle.Tensor,
        t610: paddle.Tensor,
        t611: paddle.Tensor,
        t612: paddle.Tensor,
        t613: paddle.Tensor,
        t614: paddle.Tensor,
        t615: paddle.Tensor,
        t616: paddle.Tensor,
        t617: paddle.Tensor,
        t618: paddle.Tensor,
        t619: paddle.Tensor,
        t620: paddle.Tensor,
        t621: paddle.Tensor,
        t622: paddle.Tensor,
        t623: paddle.Tensor,
        t624: paddle.Tensor,
        t625: paddle.Tensor,
        t626: paddle.Tensor,
        t627: paddle.Tensor,
        t628: paddle.Tensor,
        t629: paddle.Tensor,
        t630: paddle.Tensor,
        t631: paddle.Tensor,
        t632: paddle.Tensor,
        t633: paddle.Tensor,
        t634: paddle.Tensor,
        t635: paddle.Tensor,
        t636: paddle.Tensor,
        t637: paddle.Tensor,
        t638: paddle.Tensor,
        t639: paddle.Tensor,
        t640: paddle.Tensor,
        t641: paddle.Tensor,
        t642: paddle.Tensor,
        t643: paddle.Tensor,
        t644: paddle.Tensor,
        t645: paddle.Tensor,
        t646: paddle.Tensor,
        t647: paddle.Tensor,
        t648: paddle.Tensor,
        t649: paddle.Tensor,
        t650: paddle.Tensor,
        t651: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        t652 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t653, t654, t655, t656, t657, t658 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t652,
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

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t659 = paddle._C_ops.relu(t653)
        del t653

        # pd_op.conv2d: (-1x16x112x112xf32) <- (-1x32x112x112xf32, 16x32x2x2xf32)
        t660 = paddle._C_ops.conv2d(t659, t5, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW")
        del t5

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t661, t662, t663, t664, t665, t666 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t660,
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

        # pd_op.relu: (-1x16x112x112xf32) <- (-1x16x112x112xf32)
        t667 = paddle._C_ops.relu(t661)
        del t661

        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x16x112x112xf32, 32x16x2x2xf32)
        t668 = paddle._C_ops.conv2d(
            t667, t10, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t669, t670, t671, t672, t673, t674 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t668,
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

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t675 = paddle._C_ops.relu(t669)
        del t669

        # pd_op.full_int_array: (2xi64) <- ()
        t676 = [2, 2]

        # pd_op.pool2d: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 2xi64)
        t677 = paddle._C_ops.pool2d(
            t659, t676, [1, 1], [0, 0], True, True, "NCHW", "max", False, False, "SAME"
        )

        # pd_op.full: (1xi32) <- ()
        t678 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t679 = t678

        # pd_op.assign: (1xi32) <- (1xi32)
        t680 = t678

        # pd_op.assign: (1xi32) <- (1xi32)
        t681 = t678

        # pd_op.assign: (1xi32) <- (1xi32)
        t682 = t678

        # pd_op.assign: (1xi32) <- (1xi32)
        t683 = t678

        # pd_op.assign: (1xi32) <- (1xi32)
        t684 = t678

        # pd_op.assign: (1xi32) <- (1xi32)
        t685 = t678

        # pd_op.assign: (1xi32) <- (1xi32)
        t686 = t678

        # pd_op.assign: (1xi32) <- (1xi32)
        t687 = t678

        # pd_op.assign: (1xi32) <- (1xi32)
        t688 = t678

        # builtin.combine: ([-1x32x112x112xf32, -1x32x112x112xf32]) <- (-1x32x112x112xf32, -1x32x112x112xf32)
        t689 = [t677, t675]

        # pd_op.concat: (-1x64x112x112xf32) <- ([-1x32x112x112xf32, -1x32x112x112xf32], 1xi32)
        t690 = paddle._C_ops.concat(t689, t678)
        del t689

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x64x112x112xf32, 32x64x3x3xf32)
        t691 = paddle._C_ops.conv2d(
            t690, t15, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t692, t693, t694, t695, t696, t697 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t691,
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

        # pd_op.relu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        t698 = paddle._C_ops.relu(t692)
        del t692

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x32x56x56xf32, 64x32x1x1xf32)
        t699 = paddle._C_ops.conv2d(
            t698, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t700, t701, t702, t703, t704, t705 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t699,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t706 = paddle._C_ops.relu(t700)
        del t700

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t707 = paddle._C_ops.conv2d(
            t706, t25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t708, t709, t710, t711, t712, t713 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t707,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t714 = paddle._C_ops.relu(t708)
        del t708

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t715 = paddle._C_ops.conv2d(
            t714, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t716, t717, t718, t719, t720, t721 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t715,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t722 = paddle._C_ops.relu(t716)
        del t716

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t723 = paddle._C_ops.conv2d(
            t722, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t724, t725, t726, t727, t728, t729 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t723,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t730 = paddle._C_ops.relu(t724)
        del t724

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t731 = paddle._C_ops.conv2d(
            t730, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t732, t733, t734, t735, t736, t737 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t731,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t738 = paddle._C_ops.relu(t732)
        del t732

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t739 = paddle._C_ops.conv2d(
            t738, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t740, t741, t742, t743, t744, t745 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t739,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t746 = paddle._C_ops.relu(t740)
        del t740

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t747 = paddle._C_ops.conv2d(
            t746, t50, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t748, t749, t750, t751, t752, t753 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t747,
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
        t754 = paddle._C_ops.relu(t748)
        del t748

        # builtin.combine: ([-1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32]) <- (-1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32)
        t755 = [t706, t714, t722, t730, t738, t746, t754]

        # pd_op.concat: (-1x448x56x56xf32) <- ([-1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32, -1x64x56x56xf32], 1xi32)
        t756 = paddle._C_ops.concat(t755, t678)
        del t755

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x448x56x56xf32, 64x448x1x1xf32)
        t757 = paddle._C_ops.conv2d(
            t756, t55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t758, t759, t760, t761, t762, t763 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t757,
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

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t764 = paddle._C_ops.relu(t758)
        del t758

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x64x56x56xf32, 128x64x1x1xf32)
        t765 = paddle._C_ops.conv2d(
            t764, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t766, t767, t768, t769, t770, t771 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t765,
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

        # pd_op.relu: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        t772 = paddle._C_ops.relu(t766)
        del t766

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x56x56xf32, 128x1x3x3xf32)
        t773 = paddle._C_ops.depthwise_conv2d(
            t772, t65, [2, 2], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t774, t775, t776, t777, t778, t779 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t773,
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

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t780 = paddle._C_ops.conv2d(
            t774, t70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t781, t782, t783, t784, t785, t786 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t780,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t787 = paddle._C_ops.relu(t781)
        del t781

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t788 = paddle._C_ops.conv2d(
            t787, t75, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t789, t790, t791, t792, t793, t794 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t788,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t795 = paddle._C_ops.relu(t789)
        del t789

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t796 = paddle._C_ops.conv2d(
            t795, t80, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t797, t798, t799, t800, t801, t802 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t796,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t803 = paddle._C_ops.relu(t797)
        del t797

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t804 = paddle._C_ops.conv2d(
            t803, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t805, t806, t807, t808, t809, t810 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t804,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t811 = paddle._C_ops.relu(t805)
        del t805

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t812 = paddle._C_ops.conv2d(
            t811, t90, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t813, t814, t815, t816, t817, t818 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t812,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t819 = paddle._C_ops.relu(t813)
        del t813

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t820 = paddle._C_ops.conv2d(
            t819, t95, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t821, t822, t823, t824, t825, t826 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t820,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t827 = paddle._C_ops.relu(t821)
        del t821

        # builtin.combine: ([-1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32]) <- (-1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32)
        t828 = [t774, t787, t795, t803, t811, t819, t827]

        # pd_op.concat: (-1x896x28x28xf32) <- ([-1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32], 1xi32)
        t829 = paddle._C_ops.concat(t828, t678)
        del t828

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x896x28x28xf32, 256x896x1x1xf32)
        t830 = paddle._C_ops.conv2d(
            t829, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t831, t832, t833, t834, t835, t836 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t830,
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

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t837 = paddle._C_ops.relu(t831)
        del t831

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        t838 = paddle._C_ops.conv2d(
            t837, t105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t839, t840, t841, t842, t843, t844 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t838,
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

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t845 = paddle._C_ops.relu(t839)
        del t839

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x3x3xf32)
        t846 = paddle._C_ops.conv2d(
            t845, t110, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t847, t848, t849, t850, t851, t852 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t846,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t853 = paddle._C_ops.relu(t847)
        del t847

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t854 = paddle._C_ops.conv2d(
            t853, t115, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t855, t856, t857, t858, t859, t860 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t854,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t861 = paddle._C_ops.relu(t855)
        del t855

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t862 = paddle._C_ops.conv2d(
            t861, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t863, t864, t865, t866, t867, t868 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t862,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t869 = paddle._C_ops.relu(t863)
        del t863

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t870 = paddle._C_ops.conv2d(
            t869, t125, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t871, t872, t873, t874, t875, t876 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t870,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t877 = paddle._C_ops.relu(t871)
        del t871

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t878 = paddle._C_ops.conv2d(
            t877, t130, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t879, t880, t881, t882, t883, t884 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t878,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t885 = paddle._C_ops.relu(t879)
        del t879

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t886 = paddle._C_ops.conv2d(
            t885, t135, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t887, t888, t889, t890, t891, t892 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t886,
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

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t893 = paddle._C_ops.relu(t887)
        del t887

        # builtin.combine: ([-1x512x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32]) <- (-1x512x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32)
        t894 = [t845, t853, t861, t869, t877, t885, t893]

        # pd_op.concat: (-1x1280x28x28xf32) <- ([-1x512x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32], 1xi32)
        t895 = paddle._C_ops.concat(t894, t678)
        del t894

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x1280x28x28xf32, 256x1280x1x1xf32)
        t896 = paddle._C_ops.conv2d(
            t895, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t897, t898, t899, t900, t901, t902 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t896,
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

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t903 = paddle._C_ops.relu(t897)
        del t897

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        t904 = paddle._C_ops.conv2d(
            t903, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t905, t906, t907, t908, t909, t910 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t904,
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

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t911 = paddle._C_ops.relu(t905)
        del t905

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t912 = paddle._C_ops.add(t911, t845)

        # pd_op.depthwise_conv2d: (-1x512x14x14xf32) <- (-1x512x28x28xf32, 512x1x3x3xf32)
        t913 = paddle._C_ops.depthwise_conv2d(
            t912, t150, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t914, t915, t916, t917, t918, t919 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t913,
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

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x512x14x14xf32, 256x512x1x1xf32)
        t920 = paddle._C_ops.conv2d(
            t914, t155, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t921, t922, t923, t924, t925, t926 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t920,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t927 = paddle._C_ops.depthwise_conv2d(
            t921, t160, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t928, t929, t930, t931, t932, t933 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t927,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t934 = paddle._C_ops.relu(t928)
        del t928

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t935 = paddle._C_ops.conv2d(
            t934, t165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t936, t937, t938, t939, t940, t941 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t935,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t942 = paddle._C_ops.depthwise_conv2d(
            t936, t170, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t943, t944, t945, t946, t947, t948 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t942,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t949 = paddle._C_ops.relu(t943)
        del t943

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t950 = paddle._C_ops.conv2d(
            t949, t175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t951, t952, t953, t954, t955, t956 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t950,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t957 = paddle._C_ops.depthwise_conv2d(
            t951, t180, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t180

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t958, t959, t960, t961, t962, t963 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t957,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t964 = paddle._C_ops.relu(t958)
        del t958

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t965 = paddle._C_ops.conv2d(
            t964, t185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t966, t967, t968, t969, t970, t971 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t965,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t972 = paddle._C_ops.depthwise_conv2d(
            t966, t190, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t973, t974, t975, t976, t977, t978 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t972,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t979 = paddle._C_ops.relu(t973)
        del t973

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t980 = paddle._C_ops.conv2d(
            t979, t195, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t981, t982, t983, t984, t985, t986 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t980,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t987 = paddle._C_ops.depthwise_conv2d(
            t981, t200, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t988, t989, t990, t991, t992, t993 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t987,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t994 = paddle._C_ops.relu(t988)
        del t988

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t995 = paddle._C_ops.conv2d(
            t994, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t996, t997, t998, t999, t1000, t1001 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t995,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1002 = paddle._C_ops.depthwise_conv2d(
            t996, t210, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1003, t1004, t1005, t1006, t1007, t1008 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1002,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1009 = paddle._C_ops.relu(t1003)
        del t1003

        # builtin.combine: ([-1x512x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x512x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32)
        t1010 = [t914, t934, t949, t964, t979, t994, t1009]

        # pd_op.concat: (-1x2048x14x14xf32) <- ([-1x512x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32], 1xi32)
        t1011 = paddle._C_ops.concat(t1010, t678)
        del t1010

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x2048x14x14xf32, 512x2048x1x1xf32)
        t1012 = paddle._C_ops.conv2d(
            t1011, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1013, t1014, t1015, t1016, t1017, t1018 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1012,
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
        t1019 = paddle._C_ops.relu(t1013)
        del t1013

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1020 = paddle._C_ops.conv2d(
            t1019, t220, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t220

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1021, t1022, t1023, t1024, t1025, t1026 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1020,
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
        t1027 = paddle._C_ops.relu(t1021)
        del t1021

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1028 = paddle._C_ops.conv2d(
            t1027, t225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1029, t1030, t1031, t1032, t1033, t1034 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1028,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1035 = paddle._C_ops.depthwise_conv2d(
            t1029, t230, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t230

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1036, t1037, t1038, t1039, t1040, t1041 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1035,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1042 = paddle._C_ops.relu(t1036)
        del t1036

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1043 = paddle._C_ops.conv2d(
            t1042, t235, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1044, t1045, t1046, t1047, t1048, t1049 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1043,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1050 = paddle._C_ops.depthwise_conv2d(
            t1044, t240, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t240

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1051, t1052, t1053, t1054, t1055, t1056 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1050,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1057 = paddle._C_ops.relu(t1051)
        del t1051

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1058 = paddle._C_ops.conv2d(
            t1057, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1059, t1060, t1061, t1062, t1063, t1064 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1058,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1065 = paddle._C_ops.depthwise_conv2d(
            t1059, t250, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t250

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1066, t1067, t1068, t1069, t1070, t1071 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1065,
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
        del t254, t253, t252, t251

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1072 = paddle._C_ops.relu(t1066)
        del t1066

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1073 = paddle._C_ops.conv2d(
            t1072, t255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1074, t1075, t1076, t1077, t1078, t1079 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1073,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1080 = paddle._C_ops.depthwise_conv2d(
            t1074, t260, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t260

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1081, t1082, t1083, t1084, t1085, t1086 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1080,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1087 = paddle._C_ops.relu(t1081)
        del t1081

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1088 = paddle._C_ops.conv2d(
            t1087, t265, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t265

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1089, t1090, t1091, t1092, t1093, t1094 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1088,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1095 = paddle._C_ops.depthwise_conv2d(
            t1089, t270, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t270

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1096, t1097, t1098, t1099, t1100, t1101 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1095,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1102 = paddle._C_ops.relu(t1096)
        del t1096

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1103 = paddle._C_ops.conv2d(
            t1102, t275, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t275

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1104, t1105, t1106, t1107, t1108, t1109 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1103,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1110 = paddle._C_ops.depthwise_conv2d(
            t1104, t280, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t280

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1111, t1112, t1113, t1114, t1115, t1116 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1110,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1117 = paddle._C_ops.relu(t1111)
        del t1111

        # builtin.combine: ([-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32)
        t1118 = [t1027, t1042, t1057, t1072, t1087, t1102, t1117]

        # pd_op.concat: (-1x2560x14x14xf32) <- ([-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32], 1xi32)
        t1119 = paddle._C_ops.concat(t1118, t678)
        del t1118

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x2560x14x14xf32, 512x2560x1x1xf32)
        t1120 = paddle._C_ops.conv2d(
            t1119, t285, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t285

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1121, t1122, t1123, t1124, t1125, t1126 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1120,
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

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1127 = paddle._C_ops.relu(t1121)
        del t1121

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1128 = paddle._C_ops.conv2d(
            t1127, t290, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t290

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1129, t1130, t1131, t1132, t1133, t1134 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1128,
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

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1135 = paddle._C_ops.relu(t1129)
        del t1129

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1136 = paddle._C_ops.add(t1135, t1027)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1137 = paddle._C_ops.conv2d(
            t1136, t295, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t295

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1138, t1139, t1140, t1141, t1142, t1143 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1137,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1144 = paddle._C_ops.depthwise_conv2d(
            t1138, t300, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t300

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1145, t1146, t1147, t1148, t1149, t1150 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1144,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1151 = paddle._C_ops.relu(t1145)
        del t1145

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1152 = paddle._C_ops.conv2d(
            t1151, t305, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t305

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1153, t1154, t1155, t1156, t1157, t1158 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1152,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1159 = paddle._C_ops.depthwise_conv2d(
            t1153, t310, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t310

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1160, t1161, t1162, t1163, t1164, t1165 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1159,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1166 = paddle._C_ops.relu(t1160)
        del t1160

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1167 = paddle._C_ops.conv2d(
            t1166, t315, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t315

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1168, t1169, t1170, t1171, t1172, t1173 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1167,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1174 = paddle._C_ops.depthwise_conv2d(
            t1168, t320, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t320

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1175, t1176, t1177, t1178, t1179, t1180 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1174,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1181 = paddle._C_ops.relu(t1175)
        del t1175

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1182 = paddle._C_ops.conv2d(
            t1181, t325, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t325

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1183, t1184, t1185, t1186, t1187, t1188 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1182,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1189 = paddle._C_ops.depthwise_conv2d(
            t1183, t330, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t330

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1190, t1191, t1192, t1193, t1194, t1195 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1189,
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

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1196 = paddle._C_ops.relu(t1190)
        del t1190

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1197 = paddle._C_ops.conv2d(
            t1196, t335, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t335

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1198, t1199, t1200, t1201, t1202, t1203 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1197,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1204 = paddle._C_ops.depthwise_conv2d(
            t1198, t340, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t340

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1205, t1206, t1207, t1208, t1209, t1210 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1204,
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
        del t344, t343, t342, t341

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1211 = paddle._C_ops.relu(t1205)
        del t1205

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1212 = paddle._C_ops.conv2d(
            t1211, t345, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t345

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1213, t1214, t1215, t1216, t1217, t1218 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1212,
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

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1219 = paddle._C_ops.depthwise_conv2d(
            t1213, t350, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t350

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1220, t1221, t1222, t1223, t1224, t1225 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1219,
                t351,
                t352,
                t353,
                t354,
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
        del t354, t353, t352, t351

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1226 = paddle._C_ops.relu(t1220)
        del t1220

        # builtin.combine: ([-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32)
        t1227 = [t1136, t1151, t1166, t1181, t1196, t1211, t1226]

        # pd_op.concat: (-1x2560x14x14xf32) <- ([-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32], 1xi32)
        t1228 = paddle._C_ops.concat(t1227, t678)
        del t1227

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x2560x14x14xf32, 512x2560x1x1xf32)
        t1229 = paddle._C_ops.conv2d(
            t1228, t355, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t355

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1230, t1231, t1232, t1233, t1234, t1235 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1229,
                t356,
                t357,
                t358,
                t359,
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
        del t359, t358, t357, t356

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1236 = paddle._C_ops.relu(t1230)
        del t1230

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1237 = paddle._C_ops.conv2d(
            t1236, t360, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t360

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1238, t1239, t1240, t1241, t1242, t1243 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1237,
                t361,
                t362,
                t363,
                t364,
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
        del t364, t363, t362, t361

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1244 = paddle._C_ops.relu(t1238)
        del t1238

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1245 = paddle._C_ops.add(t1244, t1136)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1246 = paddle._C_ops.conv2d(
            t1245, t365, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t365

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1247, t1248, t1249, t1250, t1251, t1252 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1246,
                t366,
                t367,
                t368,
                t369,
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
        del t369, t368, t367, t366

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1253 = paddle._C_ops.depthwise_conv2d(
            t1247, t370, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t370

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1254, t1255, t1256, t1257, t1258, t1259 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1253,
                t371,
                t372,
                t373,
                t374,
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
        del t374, t373, t372, t371

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1260 = paddle._C_ops.relu(t1254)
        del t1254

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1261 = paddle._C_ops.conv2d(
            t1260, t375, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t375

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1262, t1263, t1264, t1265, t1266, t1267 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1261,
                t376,
                t377,
                t378,
                t379,
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
        del t379, t378, t377, t376

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1268 = paddle._C_ops.depthwise_conv2d(
            t1262, t380, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t380

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1269, t1270, t1271, t1272, t1273, t1274 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1268,
                t381,
                t382,
                t383,
                t384,
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
        del t384, t383, t382, t381

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1275 = paddle._C_ops.relu(t1269)
        del t1269

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1276 = paddle._C_ops.conv2d(
            t1275, t385, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t385

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1277, t1278, t1279, t1280, t1281, t1282 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1276,
                t386,
                t387,
                t388,
                t389,
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
        del t389, t388, t387, t386

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1283 = paddle._C_ops.depthwise_conv2d(
            t1277, t390, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t390

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1284, t1285, t1286, t1287, t1288, t1289 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1283,
                t391,
                t392,
                t393,
                t394,
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
        del t394, t393, t392, t391

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1290 = paddle._C_ops.relu(t1284)
        del t1284

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1291 = paddle._C_ops.conv2d(
            t1290, t395, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t395

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1292, t1293, t1294, t1295, t1296, t1297 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1291,
                t396,
                t397,
                t398,
                t399,
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
        del t399, t398, t397, t396

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1298 = paddle._C_ops.depthwise_conv2d(
            t1292, t400, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t400

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1299, t1300, t1301, t1302, t1303, t1304 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1298,
                t401,
                t402,
                t403,
                t404,
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
        del t404, t403, t402, t401

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1305 = paddle._C_ops.relu(t1299)
        del t1299

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1306 = paddle._C_ops.conv2d(
            t1305, t405, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t405

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1307, t1308, t1309, t1310, t1311, t1312 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1306,
                t406,
                t407,
                t408,
                t409,
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
        del t409, t408, t407, t406

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1313 = paddle._C_ops.depthwise_conv2d(
            t1307, t410, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t410

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1314, t1315, t1316, t1317, t1318, t1319 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1313,
                t411,
                t412,
                t413,
                t414,
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
        del t414, t413, t412, t411

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1320 = paddle._C_ops.relu(t1314)
        del t1314

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1321 = paddle._C_ops.conv2d(
            t1320, t415, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t415

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1322, t1323, t1324, t1325, t1326, t1327 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1321,
                t416,
                t417,
                t418,
                t419,
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
        del t419, t418, t417, t416

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1328 = paddle._C_ops.depthwise_conv2d(
            t1322, t420, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t420

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1329, t1330, t1331, t1332, t1333, t1334 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1328,
                t421,
                t422,
                t423,
                t424,
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
        del t424, t423, t422, t421

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1335 = paddle._C_ops.relu(t1329)
        del t1329

        # builtin.combine: ([-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32)
        t1336 = [t1245, t1260, t1275, t1290, t1305, t1320, t1335]

        # pd_op.concat: (-1x2560x14x14xf32) <- ([-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32], 1xi32)
        t1337 = paddle._C_ops.concat(t1336, t678)
        del t1336

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x2560x14x14xf32, 512x2560x1x1xf32)
        t1338 = paddle._C_ops.conv2d(
            t1337, t425, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t425

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1339, t1340, t1341, t1342, t1343, t1344 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1338,
                t426,
                t427,
                t428,
                t429,
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
        del t429, t428, t427, t426

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1345 = paddle._C_ops.relu(t1339)
        del t1339

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1346 = paddle._C_ops.conv2d(
            t1345, t430, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t430

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1347, t1348, t1349, t1350, t1351, t1352 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1346,
                t431,
                t432,
                t433,
                t434,
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
        del t434, t433, t432, t431

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1353 = paddle._C_ops.relu(t1347)
        del t1347

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1354 = paddle._C_ops.add(t1353, t1245)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1355 = paddle._C_ops.conv2d(
            t1354, t435, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t435

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1356, t1357, t1358, t1359, t1360, t1361 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1355,
                t436,
                t437,
                t438,
                t439,
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
        del t439, t438, t437, t436

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1362 = paddle._C_ops.depthwise_conv2d(
            t1356, t440, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t440

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1363, t1364, t1365, t1366, t1367, t1368 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1362,
                t441,
                t442,
                t443,
                t444,
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
        del t444, t443, t442, t441

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1369 = paddle._C_ops.relu(t1363)
        del t1363

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1370 = paddle._C_ops.conv2d(
            t1369, t445, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t445

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1371, t1372, t1373, t1374, t1375, t1376 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1370,
                t446,
                t447,
                t448,
                t449,
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
        del t449, t448, t447, t446

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1377 = paddle._C_ops.depthwise_conv2d(
            t1371, t450, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t450

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1378, t1379, t1380, t1381, t1382, t1383 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1377,
                t451,
                t452,
                t453,
                t454,
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
        del t454, t453, t452, t451

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1384 = paddle._C_ops.relu(t1378)
        del t1378

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1385 = paddle._C_ops.conv2d(
            t1384, t455, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t455

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1386, t1387, t1388, t1389, t1390, t1391 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1385,
                t456,
                t457,
                t458,
                t459,
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
        del t459, t458, t457, t456

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1392 = paddle._C_ops.depthwise_conv2d(
            t1386, t460, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t460

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1393, t1394, t1395, t1396, t1397, t1398 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1392,
                t461,
                t462,
                t463,
                t464,
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
        del t464, t463, t462, t461

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1399 = paddle._C_ops.relu(t1393)
        del t1393

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1400 = paddle._C_ops.conv2d(
            t1399, t465, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t465

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1401, t1402, t1403, t1404, t1405, t1406 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1400,
                t466,
                t467,
                t468,
                t469,
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
        del t469, t468, t467, t466

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1407 = paddle._C_ops.depthwise_conv2d(
            t1401, t470, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t470

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1408, t1409, t1410, t1411, t1412, t1413 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1407,
                t471,
                t472,
                t473,
                t474,
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
        del t474, t473, t472, t471

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1414 = paddle._C_ops.relu(t1408)
        del t1408

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1415 = paddle._C_ops.conv2d(
            t1414, t475, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t475

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1416, t1417, t1418, t1419, t1420, t1421 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1415,
                t476,
                t477,
                t478,
                t479,
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
        del t479, t478, t477, t476

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1422 = paddle._C_ops.depthwise_conv2d(
            t1416, t480, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t480

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1423, t1424, t1425, t1426, t1427, t1428 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1422,
                t481,
                t482,
                t483,
                t484,
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
        del t484, t483, t482, t481

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1429 = paddle._C_ops.relu(t1423)
        del t1423

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x1x1xf32)
        t1430 = paddle._C_ops.conv2d(
            t1429, t485, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t485

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1431, t1432, t1433, t1434, t1435, t1436 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1430,
                t486,
                t487,
                t488,
                t489,
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
        del t489, t488, t487, t486

        # pd_op.depthwise_conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x1x5x5xf32)
        t1437 = paddle._C_ops.depthwise_conv2d(
            t1431, t490, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del t490

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1438, t1439, t1440, t1441, t1442, t1443 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1437,
                t491,
                t492,
                t493,
                t494,
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
        del t494, t493, t492, t491

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1444 = paddle._C_ops.relu(t1438)
        del t1438

        # builtin.combine: ([-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32)
        t1445 = [t1354, t1369, t1384, t1399, t1414, t1429, t1444]

        # pd_op.concat: (-1x2560x14x14xf32) <- ([-1x1024x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32, -1x256x14x14xf32], 1xi32)
        t1446 = paddle._C_ops.concat(t1445, t678)
        del t1445

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x2560x14x14xf32, 512x2560x1x1xf32)
        t1447 = paddle._C_ops.conv2d(
            t1446, t495, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t495

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1448, t1449, t1450, t1451, t1452, t1453 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1447,
                t496,
                t497,
                t498,
                t499,
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
        del t499, t498, t497, t496

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1454 = paddle._C_ops.relu(t1448)
        del t1448

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1455 = paddle._C_ops.conv2d(
            t1454, t500, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t500

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1456, t1457, t1458, t1459, t1460, t1461 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1455,
                t501,
                t502,
                t503,
                t504,
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
        del t504, t503, t502, t501

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1462 = paddle._C_ops.relu(t1456)
        del t1456

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1463 = paddle._C_ops.add(t1462, t1354)

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf32) <- (-1x1024x14x14xf32, 1024x1x3x3xf32)
        t1464 = paddle._C_ops.depthwise_conv2d(
            t1463, t505, [2, 2], [1, 1], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del t505

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1465, t1466, t1467, t1468, t1469, t1470 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1464,
                t506,
                t507,
                t508,
                t509,
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
        del t509, t508, t507, t506

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x1024x7x7xf32, 512x1024x1x1xf32)
        t1471 = paddle._C_ops.conv2d(
            t1465, t510, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t510

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1472, t1473, t1474, t1475, t1476, t1477 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1471,
                t511,
                t512,
                t513,
                t514,
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
        del t514, t513, t512, t511

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1478 = paddle._C_ops.depthwise_conv2d(
            t1472, t515, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t515

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1479, t1480, t1481, t1482, t1483, t1484 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1478,
                t516,
                t517,
                t518,
                t519,
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
        del t519, t518, t517, t516

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1485 = paddle._C_ops.relu(t1479)
        del t1479

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1486 = paddle._C_ops.conv2d(
            t1485, t520, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t520

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1487, t1488, t1489, t1490, t1491, t1492 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1486,
                t521,
                t522,
                t523,
                t524,
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
        del t524, t523, t522, t521

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1493 = paddle._C_ops.depthwise_conv2d(
            t1487, t525, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t525

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1494, t1495, t1496, t1497, t1498, t1499 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1493,
                t526,
                t527,
                t528,
                t529,
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
        del t529, t528, t527, t526

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1500 = paddle._C_ops.relu(t1494)
        del t1494

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1501 = paddle._C_ops.conv2d(
            t1500, t530, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t530

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1502, t1503, t1504, t1505, t1506, t1507 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1501,
                t531,
                t532,
                t533,
                t534,
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
        del t534, t533, t532, t531

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1508 = paddle._C_ops.depthwise_conv2d(
            t1502, t535, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t535

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1509, t1510, t1511, t1512, t1513, t1514 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1508,
                t536,
                t537,
                t538,
                t539,
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
        del t539, t538, t537, t536

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1515 = paddle._C_ops.relu(t1509)
        del t1509

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1516 = paddle._C_ops.conv2d(
            t1515, t540, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t540

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1517, t1518, t1519, t1520, t1521, t1522 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1516,
                t541,
                t542,
                t543,
                t544,
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
        del t544, t543, t542, t541

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1523 = paddle._C_ops.depthwise_conv2d(
            t1517, t545, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t545

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1524, t1525, t1526, t1527, t1528, t1529 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1523,
                t546,
                t547,
                t548,
                t549,
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
        del t549, t548, t547, t546

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1530 = paddle._C_ops.relu(t1524)
        del t1524

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1531 = paddle._C_ops.conv2d(
            t1530, t550, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t550

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1532, t1533, t1534, t1535, t1536, t1537 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1531,
                t551,
                t552,
                t553,
                t554,
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
        del t552, t551, t554, t553

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1538 = paddle._C_ops.depthwise_conv2d(
            t1532, t555, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t555

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1539, t1540, t1541, t1542, t1543, t1544 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1538,
                t556,
                t557,
                t558,
                t559,
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
        del t559, t558, t557, t556

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1545 = paddle._C_ops.relu(t1539)
        del t1539

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1546 = paddle._C_ops.conv2d(
            t1545, t560, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t560

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1547, t1548, t1549, t1550, t1551, t1552 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1546,
                t561,
                t562,
                t563,
                t564,
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
        del t564, t563, t562, t561

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1553 = paddle._C_ops.depthwise_conv2d(
            t1547, t565, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t565

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1554, t1555, t1556, t1557, t1558, t1559 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1553,
                t566,
                t567,
                t568,
                t569,
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
        del t569, t568, t567, t566

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1560 = paddle._C_ops.relu(t1554)
        del t1554

        # builtin.combine: ([-1x1024x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32]) <- (-1x1024x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32)
        t1561 = [t1465, t1485, t1500, t1515, t1530, t1545, t1560]

        # pd_op.concat: (-1x4096x7x7xf32) <- ([-1x1024x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32], 1xi32)
        t1562 = paddle._C_ops.concat(t1561, t678)
        del t1561

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x4096x7x7xf32, 1024x4096x1x1xf32)
        t1563 = paddle._C_ops.conv2d(
            t1562, t570, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t570

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1564, t1565, t1566, t1567, t1568, t1569 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1563,
                t571,
                t572,
                t573,
                t574,
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
        del t574, t573, t572, t571

        # pd_op.relu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t1570 = paddle._C_ops.relu(t1564)
        del t1564

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        t1571 = paddle._C_ops.conv2d(
            t1570, t575, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t575

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t1572, t1573, t1574, t1575, t1576, t1577 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1571,
                t576,
                t577,
                t578,
                t579,
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
        del t579, t578, t577, t576

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t1578 = paddle._C_ops.relu(t1572)
        del t1572

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x2048x7x7xf32, 512x2048x1x1xf32)
        t1579 = paddle._C_ops.conv2d(
            t1578, t580, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t580

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1580, t1581, t1582, t1583, t1584, t1585 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1579,
                t581,
                t582,
                t583,
                t584,
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
        del t584, t583, t582, t581

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1586 = paddle._C_ops.depthwise_conv2d(
            t1580, t585, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t585

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1587, t1588, t1589, t1590, t1591, t1592 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1586,
                t586,
                t587,
                t588,
                t589,
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
        del t589, t588, t587, t586

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1593 = paddle._C_ops.relu(t1587)
        del t1587

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1594 = paddle._C_ops.conv2d(
            t1593, t590, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t590

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1595, t1596, t1597, t1598, t1599, t1600 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1594,
                t591,
                t592,
                t593,
                t594,
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
        del t594, t593, t592, t591

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1601 = paddle._C_ops.depthwise_conv2d(
            t1595, t595, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t595

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1602, t1603, t1604, t1605, t1606, t1607 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1601,
                t596,
                t597,
                t598,
                t599,
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
        del t599, t598, t597, t596

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1608 = paddle._C_ops.relu(t1602)
        del t1602

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1609 = paddle._C_ops.conv2d(
            t1608, t600, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t600

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1610, t1611, t1612, t1613, t1614, t1615 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1609,
                t601,
                t602,
                t603,
                t604,
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
        del t604, t603, t602, t601

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1616 = paddle._C_ops.depthwise_conv2d(
            t1610, t605, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t605

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1617, t1618, t1619, t1620, t1621, t1622 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1616,
                t606,
                t607,
                t608,
                t609,
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
        del t609, t608, t607, t606

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1623 = paddle._C_ops.relu(t1617)
        del t1617

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1624 = paddle._C_ops.conv2d(
            t1623, t610, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t610

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1625, t1626, t1627, t1628, t1629, t1630 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1624,
                t611,
                t612,
                t613,
                t614,
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
        del t614, t613, t612, t611

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1631 = paddle._C_ops.depthwise_conv2d(
            t1625, t615, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t615

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1632, t1633, t1634, t1635, t1636, t1637 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1631,
                t616,
                t617,
                t618,
                t619,
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
        del t619, t618, t617, t616

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1638 = paddle._C_ops.relu(t1632)
        del t1632

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1639 = paddle._C_ops.conv2d(
            t1638, t620, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t620

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1640, t1641, t1642, t1643, t1644, t1645 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1639,
                t621,
                t622,
                t623,
                t624,
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
        del t624, t623, t622, t621

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1646 = paddle._C_ops.depthwise_conv2d(
            t1640, t625, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t625

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1647, t1648, t1649, t1650, t1651, t1652 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1646,
                t626,
                t627,
                t628,
                t629,
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
        del t629, t628, t627, t626

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1653 = paddle._C_ops.relu(t1647)
        del t1647

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x1x1xf32)
        t1654 = paddle._C_ops.conv2d(
            t1653, t630, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t630

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1655, t1656, t1657, t1658, t1659, t1660 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1654,
                t631,
                t632,
                t633,
                t634,
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
        del t634, t633, t632, t631

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x5x5xf32)
        t1661 = paddle._C_ops.depthwise_conv2d(
            t1655, t635, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t635

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1662, t1663, t1664, t1665, t1666, t1667 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1661,
                t636,
                t637,
                t638,
                t639,
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
        del t639, t638, t637, t636

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t1668 = paddle._C_ops.relu(t1662)
        del t1662

        # builtin.combine: ([-1x2048x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32]) <- (-1x2048x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32)
        t1669 = [t1578, t1593, t1608, t1623, t1638, t1653, t1668]

        # pd_op.concat: (-1x5120x7x7xf32) <- ([-1x2048x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32, -1x512x7x7xf32], 1xi32)
        t1670 = paddle._C_ops.concat(t1669, t678)
        del t1669

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x5120x7x7xf32, 1024x5120x1x1xf32)
        t1671 = paddle._C_ops.conv2d(
            t1670, t640, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t640

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1672, t1673, t1674, t1675, t1676, t1677 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1671,
                t641,
                t642,
                t643,
                t644,
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
        del t642, t641, t644, t643

        # pd_op.relu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t1678 = paddle._C_ops.relu(t1672)
        del t1672

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        t1679 = paddle._C_ops.conv2d(
            t1678, t645, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t645

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t1680, t1681, t1682, t1683, t1684, t1685 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1679,
                t646,
                t647,
                t648,
                t649,
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
        del t649, t648, t647, t646

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t1686 = paddle._C_ops.relu(t1680)
        del t1680

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t1687 = paddle._C_ops.add(t1686, t1578)

        # pd_op.full_int_array: (2xi64) <- ()
        t1688 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        t1689 = paddle._C_ops.pool2d(
            t1687,
            t1688,
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
        t1690 = paddle._C_ops.conv2d(
            t1689, t650, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t650

        # pd_op.relu: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32)
        t1691 = paddle._C_ops.relu(t1690)
        del t1690

        # pd_op.flatten: (-1x2048xf32) <- (-1x2048x1x1xf32)
        t1692 = paddle._C_ops.flatten(t1691, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x2048xf32, 2048x102xf32)
        t1693 = paddle._C_ops.matmul(t1692, t651, False, False)
        del t651

        return (
            t652,
            t654,
            t655,
            t656,
            t657,
            t658,
            t659,
            t660,
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
            t686,
            t687,
            t688,
            t690,
            t691,
            t693,
            t694,
            t695,
            t696,
            t697,
            t698,
            t699,
            t701,
            t702,
            t703,
            t704,
            t705,
            t706,
            t707,
            t709,
            t710,
            t711,
            t712,
            t713,
            t714,
            t715,
            t717,
            t718,
            t719,
            t720,
            t721,
            t722,
            t723,
            t725,
            t726,
            t727,
            t728,
            t729,
            t730,
            t731,
            t733,
            t734,
            t735,
            t736,
            t737,
            t738,
            t739,
            t741,
            t742,
            t743,
            t744,
            t745,
            t746,
            t747,
            t749,
            t750,
            t751,
            t752,
            t753,
            t754,
            t756,
            t757,
            t759,
            t760,
            t761,
            t762,
            t763,
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
            t776,
            t777,
            t778,
            t779,
            t780,
            t782,
            t783,
            t784,
            t785,
            t786,
            t787,
            t788,
            t790,
            t791,
            t792,
            t793,
            t794,
            t795,
            t796,
            t798,
            t799,
            t800,
            t801,
            t802,
            t803,
            t804,
            t806,
            t807,
            t808,
            t809,
            t810,
            t811,
            t812,
            t814,
            t815,
            t816,
            t817,
            t818,
            t819,
            t820,
            t822,
            t823,
            t824,
            t825,
            t826,
            t827,
            t829,
            t830,
            t832,
            t833,
            t834,
            t835,
            t836,
            t837,
            t838,
            t840,
            t841,
            t842,
            t843,
            t844,
            t845,
            t846,
            t848,
            t849,
            t850,
            t851,
            t852,
            t853,
            t854,
            t856,
            t857,
            t858,
            t859,
            t860,
            t861,
            t862,
            t864,
            t865,
            t866,
            t867,
            t868,
            t869,
            t870,
            t872,
            t873,
            t874,
            t875,
            t876,
            t877,
            t878,
            t880,
            t881,
            t882,
            t883,
            t884,
            t885,
            t886,
            t888,
            t889,
            t890,
            t891,
            t892,
            t893,
            t895,
            t896,
            t898,
            t899,
            t900,
            t901,
            t902,
            t903,
            t904,
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
            t918,
            t919,
            t920,
            t921,
            t922,
            t923,
            t924,
            t925,
            t926,
            t927,
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
            t974,
            t975,
            t976,
            t977,
            t978,
            t979,
            t980,
            t981,
            t982,
            t983,
            t984,
            t985,
            t986,
            t987,
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
            t1004,
            t1005,
            t1006,
            t1007,
            t1008,
            t1009,
            t1011,
            t1012,
            t1014,
            t1015,
            t1016,
            t1017,
            t1018,
            t1019,
            t1020,
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
            t1034,
            t1035,
            t1037,
            t1038,
            t1039,
            t1040,
            t1041,
            t1042,
            t1043,
            t1044,
            t1045,
            t1046,
            t1047,
            t1048,
            t1049,
            t1050,
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
            t1063,
            t1064,
            t1065,
            t1067,
            t1068,
            t1069,
            t1070,
            t1071,
            t1072,
            t1073,
            t1074,
            t1075,
            t1076,
            t1077,
            t1078,
            t1079,
            t1080,
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
            t1092,
            t1093,
            t1094,
            t1095,
            t1097,
            t1098,
            t1099,
            t1100,
            t1101,
            t1102,
            t1103,
            t1104,
            t1105,
            t1106,
            t1107,
            t1108,
            t1109,
            t1110,
            t1112,
            t1113,
            t1114,
            t1115,
            t1116,
            t1117,
            t1119,
            t1120,
            t1122,
            t1123,
            t1124,
            t1125,
            t1126,
            t1127,
            t1128,
            t1130,
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
            t1146,
            t1147,
            t1148,
            t1149,
            t1150,
            t1151,
            t1152,
            t1153,
            t1154,
            t1155,
            t1156,
            t1157,
            t1158,
            t1159,
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
            t1176,
            t1177,
            t1178,
            t1179,
            t1180,
            t1181,
            t1182,
            t1183,
            t1184,
            t1185,
            t1186,
            t1187,
            t1188,
            t1189,
            t1191,
            t1192,
            t1193,
            t1194,
            t1195,
            t1196,
            t1197,
            t1198,
            t1199,
            t1200,
            t1201,
            t1202,
            t1203,
            t1204,
            t1206,
            t1207,
            t1208,
            t1209,
            t1210,
            t1211,
            t1212,
            t1213,
            t1214,
            t1215,
            t1216,
            t1217,
            t1218,
            t1219,
            t1221,
            t1222,
            t1223,
            t1224,
            t1225,
            t1226,
            t1228,
            t1229,
            t1231,
            t1232,
            t1233,
            t1234,
            t1235,
            t1236,
            t1237,
            t1239,
            t1240,
            t1241,
            t1242,
            t1243,
            t1244,
            t1245,
            t1246,
            t1247,
            t1248,
            t1249,
            t1250,
            t1251,
            t1252,
            t1253,
            t1255,
            t1256,
            t1257,
            t1258,
            t1259,
            t1260,
            t1261,
            t1262,
            t1263,
            t1264,
            t1265,
            t1266,
            t1267,
            t1268,
            t1270,
            t1271,
            t1272,
            t1273,
            t1274,
            t1275,
            t1276,
            t1277,
            t1278,
            t1279,
            t1280,
            t1281,
            t1282,
            t1283,
            t1285,
            t1286,
            t1287,
            t1288,
            t1289,
            t1290,
            t1291,
            t1292,
            t1293,
            t1294,
            t1295,
            t1296,
            t1297,
            t1298,
            t1300,
            t1301,
            t1302,
            t1303,
            t1304,
            t1305,
            t1306,
            t1307,
            t1308,
            t1309,
            t1310,
            t1311,
            t1312,
            t1313,
            t1315,
            t1316,
            t1317,
            t1318,
            t1319,
            t1320,
            t1321,
            t1322,
            t1323,
            t1324,
            t1325,
            t1326,
            t1327,
            t1328,
            t1330,
            t1331,
            t1332,
            t1333,
            t1334,
            t1335,
            t1337,
            t1338,
            t1340,
            t1341,
            t1342,
            t1343,
            t1344,
            t1345,
            t1346,
            t1348,
            t1349,
            t1350,
            t1351,
            t1352,
            t1353,
            t1354,
            t1355,
            t1356,
            t1357,
            t1358,
            t1359,
            t1360,
            t1361,
            t1362,
            t1364,
            t1365,
            t1366,
            t1367,
            t1368,
            t1369,
            t1370,
            t1371,
            t1372,
            t1373,
            t1374,
            t1375,
            t1376,
            t1377,
            t1379,
            t1380,
            t1381,
            t1382,
            t1383,
            t1384,
            t1385,
            t1386,
            t1387,
            t1388,
            t1389,
            t1390,
            t1391,
            t1392,
            t1394,
            t1395,
            t1396,
            t1397,
            t1398,
            t1399,
            t1400,
            t1401,
            t1402,
            t1403,
            t1404,
            t1405,
            t1406,
            t1407,
            t1409,
            t1410,
            t1411,
            t1412,
            t1413,
            t1414,
            t1415,
            t1416,
            t1417,
            t1418,
            t1419,
            t1420,
            t1421,
            t1422,
            t1424,
            t1425,
            t1426,
            t1427,
            t1428,
            t1429,
            t1430,
            t1431,
            t1432,
            t1433,
            t1434,
            t1435,
            t1436,
            t1437,
            t1439,
            t1440,
            t1441,
            t1442,
            t1443,
            t1444,
            t1446,
            t1447,
            t1449,
            t1450,
            t1451,
            t1452,
            t1453,
            t1454,
            t1455,
            t1457,
            t1458,
            t1459,
            t1460,
            t1461,
            t1462,
            t1463,
            t1464,
            t1465,
            t1466,
            t1467,
            t1468,
            t1469,
            t1470,
            t1471,
            t1472,
            t1473,
            t1474,
            t1475,
            t1476,
            t1477,
            t1478,
            t1480,
            t1481,
            t1482,
            t1483,
            t1484,
            t1485,
            t1486,
            t1487,
            t1488,
            t1489,
            t1490,
            t1491,
            t1492,
            t1493,
            t1495,
            t1496,
            t1497,
            t1498,
            t1499,
            t1500,
            t1501,
            t1502,
            t1503,
            t1504,
            t1505,
            t1506,
            t1507,
            t1508,
            t1510,
            t1511,
            t1512,
            t1513,
            t1514,
            t1515,
            t1516,
            t1517,
            t1518,
            t1519,
            t1520,
            t1521,
            t1522,
            t1523,
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
            t1535,
            t1536,
            t1537,
            t1538,
            t1540,
            t1541,
            t1542,
            t1543,
            t1544,
            t1545,
            t1546,
            t1547,
            t1548,
            t1549,
            t1550,
            t1551,
            t1552,
            t1553,
            t1555,
            t1556,
            t1557,
            t1558,
            t1559,
            t1560,
            t1562,
            t1563,
            t1565,
            t1566,
            t1567,
            t1568,
            t1569,
            t1570,
            t1571,
            t1573,
            t1574,
            t1575,
            t1576,
            t1577,
            t1578,
            t1579,
            t1580,
            t1581,
            t1582,
            t1583,
            t1584,
            t1585,
            t1586,
            t1588,
            t1589,
            t1590,
            t1591,
            t1592,
            t1593,
            t1594,
            t1595,
            t1596,
            t1597,
            t1598,
            t1599,
            t1600,
            t1601,
            t1603,
            t1604,
            t1605,
            t1606,
            t1607,
            t1608,
            t1609,
            t1610,
            t1611,
            t1612,
            t1613,
            t1614,
            t1615,
            t1616,
            t1618,
            t1619,
            t1620,
            t1621,
            t1622,
            t1623,
            t1624,
            t1625,
            t1626,
            t1627,
            t1628,
            t1629,
            t1630,
            t1631,
            t1633,
            t1634,
            t1635,
            t1636,
            t1637,
            t1638,
            t1639,
            t1640,
            t1641,
            t1642,
            t1643,
            t1644,
            t1645,
            t1646,
            t1648,
            t1649,
            t1650,
            t1651,
            t1652,
            t1653,
            t1654,
            t1655,
            t1656,
            t1657,
            t1658,
            t1659,
            t1660,
            t1661,
            t1663,
            t1664,
            t1665,
            t1666,
            t1667,
            t1668,
            t1670,
            t1671,
            t1673,
            t1674,
            t1675,
            t1676,
            t1677,
            t1678,
            t1679,
            t1681,
            t1682,
            t1683,
            t1684,
            t1685,
            t1686,
            t1687,
            t1688,
            t1689,
            t1691,
            t1692,
            t1693,
        )
