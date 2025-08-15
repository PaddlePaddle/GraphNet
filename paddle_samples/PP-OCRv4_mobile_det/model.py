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
        input1: paddle.Tensor,
        input2: paddle.Tensor,
        input3: paddle.Tensor,
        input4: paddle.Tensor,
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
        input5: paddle.Tensor,
        input6: paddle.Tensor,
        input7: paddle.Tensor,
        input8: paddle.Tensor,
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
        input9: paddle.Tensor,
        input10: paddle.Tensor,
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
        input11: paddle.Tensor,
        input12: paddle.Tensor,
        input13: paddle.Tensor,
        input14: paddle.Tensor,
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
        input15: paddle.Tensor,
        input16: paddle.Tensor,
        input17: paddle.Tensor,
        input18: paddle.Tensor,
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
        input19: paddle.Tensor,
        input20: paddle.Tensor,
        input21: paddle.Tensor,
        input22: paddle.Tensor,
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
        input23: paddle.Tensor,
        input24: paddle.Tensor,
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
        input25: paddle.Tensor,
        input26: paddle.Tensor,
        input27: paddle.Tensor,
        input28: paddle.Tensor,
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
        input29: paddle.Tensor,
        input30: paddle.Tensor,
        input31: paddle.Tensor,
        input32: paddle.Tensor,
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
        input33: paddle.Tensor,
        input34: paddle.Tensor,
        input35: paddle.Tensor,
        input36: paddle.Tensor,
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
        input37: paddle.Tensor,
        input38: paddle.Tensor,
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
        input39: paddle.Tensor,
        input40: paddle.Tensor,
        input41: paddle.Tensor,
        input42: paddle.Tensor,
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
        input43: paddle.Tensor,
        input44: paddle.Tensor,
        input45: paddle.Tensor,
        input46: paddle.Tensor,
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
        input47: paddle.Tensor,
        input48: paddle.Tensor,
        input49: paddle.Tensor,
        input50: paddle.Tensor,
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
        input51: paddle.Tensor,
        input52: paddle.Tensor,
        input53: paddle.Tensor,
        input54: paddle.Tensor,
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
        input55: paddle.Tensor,
        input56: paddle.Tensor,
        input57: paddle.Tensor,
        input58: paddle.Tensor,
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
        input59: paddle.Tensor,
        input60: paddle.Tensor,
        input61: paddle.Tensor,
        input62: paddle.Tensor,
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
        input63: paddle.Tensor,
        input64: paddle.Tensor,
        input65: paddle.Tensor,
        input66: paddle.Tensor,
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
        input67: paddle.Tensor,
        input68: paddle.Tensor,
        input69: paddle.Tensor,
        input70: paddle.Tensor,
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
        input71: paddle.Tensor,
        input72: paddle.Tensor,
        input73: paddle.Tensor,
        input74: paddle.Tensor,
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
        input75: paddle.Tensor,
        input76: paddle.Tensor,
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
        input77: paddle.Tensor,
        input78: paddle.Tensor,
        input79: paddle.Tensor,
        input80: paddle.Tensor,
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
        input81: paddle.Tensor,
        input82: paddle.Tensor,
        input83: paddle.Tensor,
        input84: paddle.Tensor,
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
        input85: paddle.Tensor,
        input86: paddle.Tensor,
        input87: paddle.Tensor,
        input88: paddle.Tensor,
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
        input89: paddle.Tensor,
        input90: paddle.Tensor,
        input91: paddle.Tensor,
        input92: paddle.Tensor,
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
        t652: paddle.Tensor,
        t653: paddle.Tensor,
        t654: paddle.Tensor,
        t655: paddle.Tensor,
        t656: paddle.Tensor,
        t657: paddle.Tensor,
        t658: paddle.Tensor,
        t659: paddle.Tensor,
        t660: paddle.Tensor,
        t661: paddle.Tensor,
        t662: paddle.Tensor,
        t663: paddle.Tensor,
        t664: paddle.Tensor,
        t665: paddle.Tensor,
        input93: paddle.Tensor,
        input94: paddle.Tensor,
        input95: paddle.Tensor,
        input96: paddle.Tensor,
        t666: paddle.Tensor,
        t667: paddle.Tensor,
        t668: paddle.Tensor,
        t669: paddle.Tensor,
        t670: paddle.Tensor,
        t671: paddle.Tensor,
        t672: paddle.Tensor,
        t673: paddle.Tensor,
        t674: paddle.Tensor,
        t675: paddle.Tensor,
        t676: paddle.Tensor,
        t677: paddle.Tensor,
        t678: paddle.Tensor,
        t679: paddle.Tensor,
        t680: paddle.Tensor,
        t681: paddle.Tensor,
        t682: paddle.Tensor,
        t683: paddle.Tensor,
        t684: paddle.Tensor,
        t685: paddle.Tensor,
        t686: paddle.Tensor,
        t687: paddle.Tensor,
        t688: paddle.Tensor,
        t689: paddle.Tensor,
        t690: paddle.Tensor,
        t691: paddle.Tensor,
        t692: paddle.Tensor,
        t693: paddle.Tensor,
        t694: paddle.Tensor,
        input97: paddle.Tensor,
        input98: paddle.Tensor,
        input99: paddle.Tensor,
        input100: paddle.Tensor,
        t695: paddle.Tensor,
        t696: paddle.Tensor,
        t697: paddle.Tensor,
        t698: paddle.Tensor,
        t699: paddle.Tensor,
        t700: paddle.Tensor,
        t701: paddle.Tensor,
        t702: paddle.Tensor,
        t703: paddle.Tensor,
        t704: paddle.Tensor,
        t705: paddle.Tensor,
        t706: paddle.Tensor,
        t707: paddle.Tensor,
        t708: paddle.Tensor,
        t709: paddle.Tensor,
        t710: paddle.Tensor,
        t711: paddle.Tensor,
        t712: paddle.Tensor,
        t713: paddle.Tensor,
        t714: paddle.Tensor,
        t715: paddle.Tensor,
        t716: paddle.Tensor,
        t717: paddle.Tensor,
        t718: paddle.Tensor,
        input101: paddle.Tensor,
        input102: paddle.Tensor,
        input103: paddle.Tensor,
        input104: paddle.Tensor,
        t719: paddle.Tensor,
        t720: paddle.Tensor,
        t721: paddle.Tensor,
        t722: paddle.Tensor,
        t723: paddle.Tensor,
        t724: paddle.Tensor,
        t725: paddle.Tensor,
        t726: paddle.Tensor,
        t727: paddle.Tensor,
        t728: paddle.Tensor,
        t729: paddle.Tensor,
        t730: paddle.Tensor,
        t731: paddle.Tensor,
        t732: paddle.Tensor,
        t733: paddle.Tensor,
        t734: paddle.Tensor,
        t735: paddle.Tensor,
        t736: paddle.Tensor,
        t737: paddle.Tensor,
        t738: paddle.Tensor,
        t739: paddle.Tensor,
        t740: paddle.Tensor,
        t741: paddle.Tensor,
        t742: paddle.Tensor,
        t743: paddle.Tensor,
        t744: paddle.Tensor,
        t745: paddle.Tensor,
        t746: paddle.Tensor,
        t747: paddle.Tensor,
        t748: paddle.Tensor,
        t749: paddle.Tensor,
        t750: paddle.Tensor,
        t751: paddle.Tensor,
        t752: paddle.Tensor,
        t753: paddle.Tensor,
        t754: paddle.Tensor,
        t755: paddle.Tensor,
        t756: paddle.Tensor,
        t757: paddle.Tensor,
        t758: paddle.Tensor,
        t759: paddle.Tensor,
        t760: paddle.Tensor,
        t761: paddle.Tensor,
        t762: paddle.Tensor,
        t763: paddle.Tensor,
        t764: paddle.Tensor,
        t765: paddle.Tensor,
        t766: paddle.Tensor,
        t767: paddle.Tensor,
        t768: paddle.Tensor,
        t769: paddle.Tensor,
        t770: paddle.Tensor,
        t771: paddle.Tensor,
        t772: paddle.Tensor,
        t773: paddle.Tensor,
        t774: paddle.Tensor,
        t775: paddle.Tensor,
        t776: paddle.Tensor,
        t777: paddle.Tensor,
        t778: paddle.Tensor,
        t779: paddle.Tensor,
        t780: paddle.Tensor,
        t781: paddle.Tensor,
        t782: paddle.Tensor,
        t783: paddle.Tensor,
        t784: paddle.Tensor,
        t785: paddle.Tensor,
        t786: paddle.Tensor,
        t787: paddle.Tensor,
        t788: paddle.Tensor,
        t789: paddle.Tensor,
        t790: paddle.Tensor,
        t791: paddle.Tensor,
        t792: paddle.Tensor,
    ):
        t793 = None
        # pd_op.conv2d: (8x16x320x320xf32) <- (8x3x640x640xf32, 16x3x3x3xf32)
        t794 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t795, t796, t797, t798, t799, t800 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t794,
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

        # pd_op.batch_norm_: (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t801, t802, t803, t804, t805, t806 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t795,
                t5,
                t6,
                t7,
                t8,
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
        del t8, t7, t6, t5

        # pd_op.full: (1xf32) <- ()
        t807 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t808 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t809 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t810 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t811 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t812 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t813 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t814 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t815 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t816 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t817 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t818 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t819 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t820 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t821 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t822 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t823 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t824 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t825 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t826 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t827 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t828 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t829 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t830 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t831 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t832 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t833 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t834 = t807

        # pd_op.assign: (1xf32) <- (1xf32)
        t835 = t807

        # pd_op.scale: (8x16x320x320xf32) <- (8x16x320x320xf32, 1xf32)
        t836 = paddle._C_ops.scale(t801, t807, float("0"), True)
        del t801

        # pd_op.depthwise_conv2d: (8x16x320x320xf32) <- (8x16x320x320xf32, 16x1x1x1xf32)
        t837 = paddle._C_ops.depthwise_conv2d(
            t795, t9, [1, 1], [0, 0], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t9

        # pd_op.batch_norm_: (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t838, t839, t840, t841, t842, t843 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t837,
                t10,
                t11,
                t12,
                t13,
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
        del t13, t12, t11, t10

        # pd_op.add: (8x16x320x320xf32) <- (8x16x320x320xf32, 8x16x320x320xf32)
        t844 = paddle._C_ops.add(t836, t838)

        # pd_op.depthwise_conv2d: (8x16x320x320xf32) <- (8x16x320x320xf32, 16x1x3x3xf32)
        t845 = paddle._C_ops.depthwise_conv2d(
            t795, t14, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t14

        # pd_op.batch_norm_: (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t846, t847, t848, t849, t850, t851 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t845,
                t15,
                t16,
                t17,
                t18,
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
        del t18, t17, t16, t15

        # pd_op.add: (8x16x320x320xf32) <- (8x16x320x320xf32, 8x16x320x320xf32)
        t852 = paddle._C_ops.add(t844, t846)

        # pd_op.depthwise_conv2d: (8x16x320x320xf32) <- (8x16x320x320xf32, 16x1x3x3xf32)
        t853 = paddle._C_ops.depthwise_conv2d(
            t795, t19, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t19

        # pd_op.batch_norm_: (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t854, t855, t856, t857, t858, t859 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t853,
                t20,
                t21,
                t22,
                t23,
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
        del t23, t22, t21, t20

        # pd_op.add: (8x16x320x320xf32) <- (8x16x320x320xf32, 8x16x320x320xf32)
        t860 = paddle._C_ops.add(t852, t854)

        # pd_op.depthwise_conv2d: (8x16x320x320xf32) <- (8x16x320x320xf32, 16x1x3x3xf32)
        t861 = paddle._C_ops.depthwise_conv2d(
            t795, t24, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t24

        # pd_op.batch_norm_: (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t862, t863, t864, t865, t866, t867 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t861,
                t25,
                t26,
                t27,
                t28,
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
        del t28, t27, t26, t25

        # pd_op.add: (8x16x320x320xf32) <- (8x16x320x320xf32, 8x16x320x320xf32)
        t868 = paddle._C_ops.add(t860, t862)

        # pd_op.depthwise_conv2d: (8x16x320x320xf32) <- (8x16x320x320xf32, 16x1x3x3xf32)
        t869 = paddle._C_ops.depthwise_conv2d(
            t795, t29, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del t29

        # pd_op.batch_norm_: (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        t870, t871, t872, t873, t874, t875 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t869,
                t30,
                t31,
                t32,
                t33,
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
        del t33, t32, t31, t30

        # pd_op.add: (8x16x320x320xf32) <- (8x16x320x320xf32, 8x16x320x320xf32)
        t876 = paddle._C_ops.add(t868, t870)

        # pd_op.multiply: (8x16x320x320xf32) <- (1xf32, 8x16x320x320xf32)
        t877 = paddle._C_ops.multiply(input1, t876)
        del input1

        # pd_op.add: (8x16x320x320xf32) <- (8x16x320x320xf32, 1xf32)
        t878 = paddle._C_ops.add(t877, input2)
        del input2

        # pd_op.hardswish: (8x16x320x320xf32) <- (8x16x320x320xf32)
        t879 = paddle._C_ops.hardswish(t878)

        # pd_op.multiply: (8x16x320x320xf32) <- (1xf32, 8x16x320x320xf32)
        t880 = paddle._C_ops.multiply(input3, t879)
        del input3

        # pd_op.add: (8x16x320x320xf32) <- (8x16x320x320xf32, 1xf32)
        t881 = paddle._C_ops.add(t880, input4)
        del input4

        # pd_op.conv2d: (8x32x320x320xf32) <- (8x16x320x320xf32, 32x16x1x1xf32)
        t882 = paddle._C_ops.conv2d(
            t881, t34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t34

        # pd_op.batch_norm_: (8x32x320x320xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x320x320xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t883, t884, t885, t886, t887, t888 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t882,
                t35,
                t36,
                t37,
                t38,
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
        del t38, t37, t36, t35

        # pd_op.scale: (8x32x320x320xf32) <- (8x32x320x320xf32, 1xf32)
        t889 = paddle._C_ops.scale(t883, t807, float("0"), True)
        del t883

        # pd_op.conv2d: (8x32x320x320xf32) <- (8x16x320x320xf32, 32x16x1x1xf32)
        t890 = paddle._C_ops.conv2d(
            t881, t39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t39

        # pd_op.batch_norm_: (8x32x320x320xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x320x320xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t891, t892, t893, t894, t895, t896 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t890,
                t40,
                t41,
                t42,
                t43,
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
        del t43, t42, t41, t40

        # pd_op.add: (8x32x320x320xf32) <- (8x32x320x320xf32, 8x32x320x320xf32)
        t897 = paddle._C_ops.add(t889, t891)

        # pd_op.conv2d: (8x32x320x320xf32) <- (8x16x320x320xf32, 32x16x1x1xf32)
        t898 = paddle._C_ops.conv2d(
            t881, t44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t44

        # pd_op.batch_norm_: (8x32x320x320xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x320x320xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t899, t900, t901, t902, t903, t904 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t898,
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

        # pd_op.add: (8x32x320x320xf32) <- (8x32x320x320xf32, 8x32x320x320xf32)
        t905 = paddle._C_ops.add(t897, t899)

        # pd_op.conv2d: (8x32x320x320xf32) <- (8x16x320x320xf32, 32x16x1x1xf32)
        t906 = paddle._C_ops.conv2d(
            t881, t49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t49

        # pd_op.batch_norm_: (8x32x320x320xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x320x320xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t907, t908, t909, t910, t911, t912 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t906,
                t50,
                t51,
                t52,
                t53,
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
        del t53, t52, t51, t50

        # pd_op.add: (8x32x320x320xf32) <- (8x32x320x320xf32, 8x32x320x320xf32)
        t913 = paddle._C_ops.add(t905, t907)

        # pd_op.multiply: (8x32x320x320xf32) <- (1xf32, 8x32x320x320xf32)
        t914 = paddle._C_ops.multiply(input5, t913)
        del input5

        # pd_op.add: (8x32x320x320xf32) <- (8x32x320x320xf32, 1xf32)
        t915 = paddle._C_ops.add(t914, input6)
        del input6

        # pd_op.hardswish: (8x32x320x320xf32) <- (8x32x320x320xf32)
        t916 = paddle._C_ops.hardswish(t915)

        # pd_op.multiply: (8x32x320x320xf32) <- (1xf32, 8x32x320x320xf32)
        t917 = paddle._C_ops.multiply(input7, t916)
        del input7

        # pd_op.add: (8x32x320x320xf32) <- (8x32x320x320xf32, 1xf32)
        t918 = paddle._C_ops.add(t917, input8)
        del input8

        # pd_op.depthwise_conv2d: (8x32x160x160xf32) <- (8x32x320x320xf32, 32x1x1x1xf32)
        t919 = paddle._C_ops.depthwise_conv2d(
            t918, t54, [2, 2], [0, 0], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t54

        # pd_op.batch_norm_: (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t920, t921, t922, t923, t924, t925 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t919,
                t55,
                t56,
                t57,
                t58,
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
        del t58, t57, t56, t55

        # pd_op.scale: (8x32x160x160xf32) <- (8x32x160x160xf32, 1xf32)
        t926 = paddle._C_ops.scale(t920, t807, float("0"), True)
        del t920

        # pd_op.depthwise_conv2d: (8x32x160x160xf32) <- (8x32x320x320xf32, 32x1x3x3xf32)
        t927 = paddle._C_ops.depthwise_conv2d(
            t918, t59, [2, 2], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t59

        # pd_op.batch_norm_: (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t928, t929, t930, t931, t932, t933 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t927,
                t60,
                t61,
                t62,
                t63,
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
        del t63, t62, t61, t60

        # pd_op.add: (8x32x160x160xf32) <- (8x32x160x160xf32, 8x32x160x160xf32)
        t934 = paddle._C_ops.add(t926, t928)

        # pd_op.depthwise_conv2d: (8x32x160x160xf32) <- (8x32x320x320xf32, 32x1x3x3xf32)
        t935 = paddle._C_ops.depthwise_conv2d(
            t918, t64, [2, 2], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t64

        # pd_op.batch_norm_: (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t936, t937, t938, t939, t940, t941 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t935,
                t65,
                t66,
                t67,
                t68,
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
        del t68, t67, t66, t65

        # pd_op.add: (8x32x160x160xf32) <- (8x32x160x160xf32, 8x32x160x160xf32)
        t942 = paddle._C_ops.add(t934, t936)

        # pd_op.depthwise_conv2d: (8x32x160x160xf32) <- (8x32x320x320xf32, 32x1x3x3xf32)
        t943 = paddle._C_ops.depthwise_conv2d(
            t918, t69, [2, 2], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t69

        # pd_op.batch_norm_: (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t944, t945, t946, t947, t948, t949 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t943,
                t70,
                t71,
                t72,
                t73,
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
        del t73, t72, t71, t70

        # pd_op.add: (8x32x160x160xf32) <- (8x32x160x160xf32, 8x32x160x160xf32)
        t950 = paddle._C_ops.add(t942, t944)

        # pd_op.depthwise_conv2d: (8x32x160x160xf32) <- (8x32x320x320xf32, 32x1x3x3xf32)
        t951 = paddle._C_ops.depthwise_conv2d(
            t918, t74, [2, 2], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del t74

        # pd_op.batch_norm_: (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x160x160xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t952, t953, t954, t955, t956, t957 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t951,
                t75,
                t76,
                t77,
                t78,
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
        del t78, t77, t76, t75

        # pd_op.add: (8x32x160x160xf32) <- (8x32x160x160xf32, 8x32x160x160xf32)
        t958 = paddle._C_ops.add(t950, t952)

        # pd_op.multiply: (8x32x160x160xf32) <- (1xf32, 8x32x160x160xf32)
        t959 = paddle._C_ops.multiply(input9, t958)
        del input9

        # pd_op.add: (8x32x160x160xf32) <- (8x32x160x160xf32, 1xf32)
        t960 = paddle._C_ops.add(t959, input10)
        del input10

        # pd_op.conv2d: (8x48x160x160xf32) <- (8x32x160x160xf32, 48x32x1x1xf32)
        t961 = paddle._C_ops.conv2d(
            t960, t79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t79

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t962, t963, t964, t965, t966, t967 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t961,
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

        # pd_op.scale: (8x48x160x160xf32) <- (8x48x160x160xf32, 1xf32)
        t968 = paddle._C_ops.scale(t962, t807, float("0"), True)
        del t962

        # pd_op.conv2d: (8x48x160x160xf32) <- (8x32x160x160xf32, 48x32x1x1xf32)
        t969 = paddle._C_ops.conv2d(
            t960, t84, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t84

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t970, t971, t972, t973, t974, t975 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t969,
                t85,
                t86,
                t87,
                t88,
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
        del t88, t87, t86, t85

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t976 = paddle._C_ops.add(t968, t970)

        # pd_op.conv2d: (8x48x160x160xf32) <- (8x32x160x160xf32, 48x32x1x1xf32)
        t977 = paddle._C_ops.conv2d(
            t960, t89, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t89

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t978, t979, t980, t981, t982, t983 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t977,
                t90,
                t91,
                t92,
                t93,
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
        del t93, t92, t91, t90

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t984 = paddle._C_ops.add(t976, t978)

        # pd_op.conv2d: (8x48x160x160xf32) <- (8x32x160x160xf32, 48x32x1x1xf32)
        t985 = paddle._C_ops.conv2d(
            t960, t94, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t94

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t986, t987, t988, t989, t990, t991 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t985,
                t95,
                t96,
                t97,
                t98,
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
        del t98, t97, t96, t95

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t992 = paddle._C_ops.add(t984, t986)

        # pd_op.multiply: (8x48x160x160xf32) <- (1xf32, 8x48x160x160xf32)
        t993 = paddle._C_ops.multiply(input11, t992)
        del input11

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 1xf32)
        t994 = paddle._C_ops.add(t993, input12)
        del input12

        # pd_op.hardswish: (8x48x160x160xf32) <- (8x48x160x160xf32)
        t995 = paddle._C_ops.hardswish(t994)

        # pd_op.multiply: (8x48x160x160xf32) <- (1xf32, 8x48x160x160xf32)
        t996 = paddle._C_ops.multiply(input13, t995)
        del input13

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 1xf32)
        t997 = paddle._C_ops.add(t996, input14)
        del input14

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t998, t999, t1000, t1001, t1002, t1003 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t997,
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

        # pd_op.scale: (8x48x160x160xf32) <- (8x48x160x160xf32, 1xf32)
        t1004 = paddle._C_ops.scale(t998, t807, float("0"), True)
        del t998

        # pd_op.depthwise_conv2d: (8x48x160x160xf32) <- (8x48x160x160xf32, 48x1x1x1xf32)
        t1005 = paddle._C_ops.depthwise_conv2d(
            t997, t103, [1, 1], [0, 0], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t103

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1006, t1007, t1008, t1009, t1010, t1011 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1005,
                t104,
                t105,
                t106,
                t107,
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
        del t107, t106, t105, t104

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t1012 = paddle._C_ops.add(t1004, t1006)

        # pd_op.depthwise_conv2d: (8x48x160x160xf32) <- (8x48x160x160xf32, 48x1x3x3xf32)
        t1013 = paddle._C_ops.depthwise_conv2d(
            t997, t108, [1, 1], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t108

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1014, t1015, t1016, t1017, t1018, t1019 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1013,
                t109,
                t110,
                t111,
                t112,
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
        del t112, t111, t110, t109

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t1020 = paddle._C_ops.add(t1012, t1014)

        # pd_op.depthwise_conv2d: (8x48x160x160xf32) <- (8x48x160x160xf32, 48x1x3x3xf32)
        t1021 = paddle._C_ops.depthwise_conv2d(
            t997, t113, [1, 1], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t113

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1022, t1023, t1024, t1025, t1026, t1027 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1021,
                t114,
                t115,
                t116,
                t117,
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
        del t117, t116, t115, t114

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t1028 = paddle._C_ops.add(t1020, t1022)

        # pd_op.depthwise_conv2d: (8x48x160x160xf32) <- (8x48x160x160xf32, 48x1x3x3xf32)
        t1029 = paddle._C_ops.depthwise_conv2d(
            t997, t118, [1, 1], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t118

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1030, t1031, t1032, t1033, t1034, t1035 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1029,
                t119,
                t120,
                t121,
                t122,
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
        del t122, t121, t120, t119

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t1036 = paddle._C_ops.add(t1028, t1030)

        # pd_op.depthwise_conv2d: (8x48x160x160xf32) <- (8x48x160x160xf32, 48x1x3x3xf32)
        t1037 = paddle._C_ops.depthwise_conv2d(
            t997, t123, [1, 1], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t123

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1038, t1039, t1040, t1041, t1042, t1043 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1037,
                t124,
                t125,
                t126,
                t127,
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
        del t127, t126, t125, t124

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t1044 = paddle._C_ops.add(t1036, t1038)

        # pd_op.multiply: (8x48x160x160xf32) <- (1xf32, 8x48x160x160xf32)
        t1045 = paddle._C_ops.multiply(input15, t1044)
        del input15

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 1xf32)
        t1046 = paddle._C_ops.add(t1045, input16)
        del input16

        # pd_op.hardswish: (8x48x160x160xf32) <- (8x48x160x160xf32)
        t1047 = paddle._C_ops.hardswish(t1046)

        # pd_op.multiply: (8x48x160x160xf32) <- (1xf32, 8x48x160x160xf32)
        t1048 = paddle._C_ops.multiply(input17, t1047)
        del input17

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 1xf32)
        t1049 = paddle._C_ops.add(t1048, input18)
        del input18

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1050, t1051, t1052, t1053, t1054, t1055 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1049,
                t128,
                t129,
                t130,
                t131,
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
        del t131, t130, t129, t128

        # pd_op.scale: (8x48x160x160xf32) <- (8x48x160x160xf32, 1xf32)
        t1056 = paddle._C_ops.scale(t1050, t807, float("0"), True)
        del t1050

        # pd_op.conv2d: (8x48x160x160xf32) <- (8x48x160x160xf32, 48x48x1x1xf32)
        t1057 = paddle._C_ops.conv2d(
            t1049, t132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t132

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1058, t1059, t1060, t1061, t1062, t1063 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1057,
                t133,
                t134,
                t135,
                t136,
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
        del t136, t135, t134, t133

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t1064 = paddle._C_ops.add(t1056, t1058)

        # pd_op.conv2d: (8x48x160x160xf32) <- (8x48x160x160xf32, 48x48x1x1xf32)
        t1065 = paddle._C_ops.conv2d(
            t1049, t137, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t137

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1066, t1067, t1068, t1069, t1070, t1071 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1065,
                t138,
                t139,
                t140,
                t141,
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
        del t141, t140, t139, t138

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t1072 = paddle._C_ops.add(t1064, t1066)

        # pd_op.conv2d: (8x48x160x160xf32) <- (8x48x160x160xf32, 48x48x1x1xf32)
        t1073 = paddle._C_ops.conv2d(
            t1049, t142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t142

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1074, t1075, t1076, t1077, t1078, t1079 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1073,
                t143,
                t144,
                t145,
                t146,
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
        del t146, t145, t144, t143

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t1080 = paddle._C_ops.add(t1072, t1074)

        # pd_op.conv2d: (8x48x160x160xf32) <- (8x48x160x160xf32, 48x48x1x1xf32)
        t1081 = paddle._C_ops.conv2d(
            t1049, t147, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t147

        # pd_op.batch_norm_: (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1082, t1083, t1084, t1085, t1086, t1087 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1081,
                t148,
                t149,
                t150,
                t151,
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
        del t151, t150, t149, t148

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 8x48x160x160xf32)
        t1088 = paddle._C_ops.add(t1080, t1082)

        # pd_op.multiply: (8x48x160x160xf32) <- (1xf32, 8x48x160x160xf32)
        t1089 = paddle._C_ops.multiply(input19, t1088)
        del input19

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 1xf32)
        t1090 = paddle._C_ops.add(t1089, input20)
        del input20

        # pd_op.hardswish: (8x48x160x160xf32) <- (8x48x160x160xf32)
        t1091 = paddle._C_ops.hardswish(t1090)

        # pd_op.multiply: (8x48x160x160xf32) <- (1xf32, 8x48x160x160xf32)
        t1092 = paddle._C_ops.multiply(input21, t1091)
        del input21

        # pd_op.add: (8x48x160x160xf32) <- (8x48x160x160xf32, 1xf32)
        t1093 = paddle._C_ops.add(t1092, input22)
        del input22

        # pd_op.depthwise_conv2d: (8x48x80x80xf32) <- (8x48x160x160xf32, 48x1x1x1xf32)
        t1094 = paddle._C_ops.depthwise_conv2d(
            t1093, t152, [2, 2], [0, 0], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t152

        # pd_op.batch_norm_: (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1095, t1096, t1097, t1098, t1099, t1100 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1094,
                t153,
                t154,
                t155,
                t156,
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
        del t156, t155, t154, t153

        # pd_op.scale: (8x48x80x80xf32) <- (8x48x80x80xf32, 1xf32)
        t1101 = paddle._C_ops.scale(t1095, t807, float("0"), True)
        del t1095

        # pd_op.depthwise_conv2d: (8x48x80x80xf32) <- (8x48x160x160xf32, 48x1x3x3xf32)
        t1102 = paddle._C_ops.depthwise_conv2d(
            t1093, t157, [2, 2], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t157

        # pd_op.batch_norm_: (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1103, t1104, t1105, t1106, t1107, t1108 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1102,
                t158,
                t159,
                t160,
                t161,
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
        del t161, t160, t159, t158

        # pd_op.add: (8x48x80x80xf32) <- (8x48x80x80xf32, 8x48x80x80xf32)
        t1109 = paddle._C_ops.add(t1101, t1103)

        # pd_op.depthwise_conv2d: (8x48x80x80xf32) <- (8x48x160x160xf32, 48x1x3x3xf32)
        t1110 = paddle._C_ops.depthwise_conv2d(
            t1093, t162, [2, 2], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t162

        # pd_op.batch_norm_: (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1111, t1112, t1113, t1114, t1115, t1116 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1110,
                t163,
                t164,
                t165,
                t166,
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
        del t166, t165, t164, t163

        # pd_op.add: (8x48x80x80xf32) <- (8x48x80x80xf32, 8x48x80x80xf32)
        t1117 = paddle._C_ops.add(t1109, t1111)

        # pd_op.depthwise_conv2d: (8x48x80x80xf32) <- (8x48x160x160xf32, 48x1x3x3xf32)
        t1118 = paddle._C_ops.depthwise_conv2d(
            t1093, t167, [2, 2], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t167

        # pd_op.batch_norm_: (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1119, t1120, t1121, t1122, t1123, t1124 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1118,
                t168,
                t169,
                t170,
                t171,
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
        del t171, t170, t169, t168

        # pd_op.add: (8x48x80x80xf32) <- (8x48x80x80xf32, 8x48x80x80xf32)
        t1125 = paddle._C_ops.add(t1117, t1119)

        # pd_op.depthwise_conv2d: (8x48x80x80xf32) <- (8x48x160x160xf32, 48x1x3x3xf32)
        t1126 = paddle._C_ops.depthwise_conv2d(
            t1093, t172, [2, 2], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del t172

        # pd_op.batch_norm_: (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (8x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t1127, t1128, t1129, t1130, t1131, t1132 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1126,
                t173,
                t174,
                t175,
                t176,
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
        del t176, t175, t174, t173

        # pd_op.add: (8x48x80x80xf32) <- (8x48x80x80xf32, 8x48x80x80xf32)
        t1133 = paddle._C_ops.add(t1125, t1127)

        # pd_op.multiply: (8x48x80x80xf32) <- (1xf32, 8x48x80x80xf32)
        t1134 = paddle._C_ops.multiply(input23, t1133)
        del input23

        # pd_op.add: (8x48x80x80xf32) <- (8x48x80x80xf32, 1xf32)
        t1135 = paddle._C_ops.add(t1134, input24)
        del input24

        # pd_op.conv2d: (8x96x80x80xf32) <- (8x48x80x80xf32, 96x48x1x1xf32)
        t1136 = paddle._C_ops.conv2d(
            t1135, t177, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t177

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1137, t1138, t1139, t1140, t1141, t1142 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1136,
                t178,
                t179,
                t180,
                t181,
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
        del t181, t180, t179, t178

        # pd_op.scale: (8x96x80x80xf32) <- (8x96x80x80xf32, 1xf32)
        t1143 = paddle._C_ops.scale(t1137, t807, float("0"), True)
        del t1137

        # pd_op.conv2d: (8x96x80x80xf32) <- (8x48x80x80xf32, 96x48x1x1xf32)
        t1144 = paddle._C_ops.conv2d(
            t1135, t182, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t182

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1145, t1146, t1147, t1148, t1149, t1150 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1144,
                t183,
                t184,
                t185,
                t186,
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
        del t186, t185, t184, t183

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1151 = paddle._C_ops.add(t1143, t1145)

        # pd_op.conv2d: (8x96x80x80xf32) <- (8x48x80x80xf32, 96x48x1x1xf32)
        t1152 = paddle._C_ops.conv2d(
            t1135, t187, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t187

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1153, t1154, t1155, t1156, t1157, t1158 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1152,
                t188,
                t189,
                t190,
                t191,
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
        del t191, t190, t189, t188

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1159 = paddle._C_ops.add(t1151, t1153)

        # pd_op.conv2d: (8x96x80x80xf32) <- (8x48x80x80xf32, 96x48x1x1xf32)
        t1160 = paddle._C_ops.conv2d(
            t1135, t192, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t192

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1161, t1162, t1163, t1164, t1165, t1166 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1160,
                t193,
                t194,
                t195,
                t196,
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
        del t196, t195, t194, t193

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1167 = paddle._C_ops.add(t1159, t1161)

        # pd_op.multiply: (8x96x80x80xf32) <- (1xf32, 8x96x80x80xf32)
        t1168 = paddle._C_ops.multiply(input25, t1167)
        del input25

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 1xf32)
        t1169 = paddle._C_ops.add(t1168, input26)
        del input26

        # pd_op.hardswish: (8x96x80x80xf32) <- (8x96x80x80xf32)
        t1170 = paddle._C_ops.hardswish(t1169)

        # pd_op.multiply: (8x96x80x80xf32) <- (1xf32, 8x96x80x80xf32)
        t1171 = paddle._C_ops.multiply(input27, t1170)
        del input27

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 1xf32)
        t1172 = paddle._C_ops.add(t1171, input28)
        del input28

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1173, t1174, t1175, t1176, t1177, t1178 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1172,
                t197,
                t198,
                t199,
                t200,
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
        del t200, t199, t198, t197

        # pd_op.scale: (8x96x80x80xf32) <- (8x96x80x80xf32, 1xf32)
        t1179 = paddle._C_ops.scale(t1173, t807, float("0"), True)
        del t1173

        # pd_op.depthwise_conv2d: (8x96x80x80xf32) <- (8x96x80x80xf32, 96x1x1x1xf32)
        t1180 = paddle._C_ops.depthwise_conv2d(
            t1172, t201, [1, 1], [0, 0], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t201

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1181, t1182, t1183, t1184, t1185, t1186 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1180,
                t202,
                t203,
                t204,
                t205,
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
        del t205, t204, t203, t202

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1187 = paddle._C_ops.add(t1179, t1181)

        # pd_op.depthwise_conv2d: (8x96x80x80xf32) <- (8x96x80x80xf32, 96x1x3x3xf32)
        t1188 = paddle._C_ops.depthwise_conv2d(
            t1172, t206, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t206

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1189, t1190, t1191, t1192, t1193, t1194 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1188,
                t207,
                t208,
                t209,
                t210,
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
        del t210, t209, t208, t207

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1195 = paddle._C_ops.add(t1187, t1189)

        # pd_op.depthwise_conv2d: (8x96x80x80xf32) <- (8x96x80x80xf32, 96x1x3x3xf32)
        t1196 = paddle._C_ops.depthwise_conv2d(
            t1172, t211, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t211

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1197, t1198, t1199, t1200, t1201, t1202 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1196,
                t212,
                t213,
                t214,
                t215,
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
        del t215, t214, t213, t212

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1203 = paddle._C_ops.add(t1195, t1197)

        # pd_op.depthwise_conv2d: (8x96x80x80xf32) <- (8x96x80x80xf32, 96x1x3x3xf32)
        t1204 = paddle._C_ops.depthwise_conv2d(
            t1172, t216, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t216

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1205, t1206, t1207, t1208, t1209, t1210 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1204,
                t217,
                t218,
                t219,
                t220,
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
        del t220, t219, t218, t217

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1211 = paddle._C_ops.add(t1203, t1205)

        # pd_op.depthwise_conv2d: (8x96x80x80xf32) <- (8x96x80x80xf32, 96x1x3x3xf32)
        t1212 = paddle._C_ops.depthwise_conv2d(
            t1172, t221, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t221

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1213, t1214, t1215, t1216, t1217, t1218 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1212,
                t222,
                t223,
                t224,
                t225,
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
        del t225, t224, t223, t222

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1219 = paddle._C_ops.add(t1211, t1213)

        # pd_op.multiply: (8x96x80x80xf32) <- (1xf32, 8x96x80x80xf32)
        t1220 = paddle._C_ops.multiply(input29, t1219)
        del input29

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 1xf32)
        t1221 = paddle._C_ops.add(t1220, input30)
        del input30

        # pd_op.hardswish: (8x96x80x80xf32) <- (8x96x80x80xf32)
        t1222 = paddle._C_ops.hardswish(t1221)

        # pd_op.multiply: (8x96x80x80xf32) <- (1xf32, 8x96x80x80xf32)
        t1223 = paddle._C_ops.multiply(input31, t1222)
        del input31

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 1xf32)
        t1224 = paddle._C_ops.add(t1223, input32)
        del input32

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1225, t1226, t1227, t1228, t1229, t1230 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1224,
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

        # pd_op.scale: (8x96x80x80xf32) <- (8x96x80x80xf32, 1xf32)
        t1231 = paddle._C_ops.scale(t1225, t807, float("0"), True)
        del t1225

        # pd_op.conv2d: (8x96x80x80xf32) <- (8x96x80x80xf32, 96x96x1x1xf32)
        t1232 = paddle._C_ops.conv2d(
            t1224, t230, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t230

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1233, t1234, t1235, t1236, t1237, t1238 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1232,
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

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1239 = paddle._C_ops.add(t1231, t1233)

        # pd_op.conv2d: (8x96x80x80xf32) <- (8x96x80x80xf32, 96x96x1x1xf32)
        t1240 = paddle._C_ops.conv2d(
            t1224, t235, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1241, t1242, t1243, t1244, t1245, t1246 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1240,
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

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1247 = paddle._C_ops.add(t1239, t1241)

        # pd_op.conv2d: (8x96x80x80xf32) <- (8x96x80x80xf32, 96x96x1x1xf32)
        t1248 = paddle._C_ops.conv2d(
            t1224, t240, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t240

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1249, t1250, t1251, t1252, t1253, t1254 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1248,
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

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1255 = paddle._C_ops.add(t1247, t1249)

        # pd_op.conv2d: (8x96x80x80xf32) <- (8x96x80x80xf32, 96x96x1x1xf32)
        t1256 = paddle._C_ops.conv2d(
            t1224, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1257, t1258, t1259, t1260, t1261, t1262 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1256,
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

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t1263 = paddle._C_ops.add(t1255, t1257)

        # pd_op.multiply: (8x96x80x80xf32) <- (1xf32, 8x96x80x80xf32)
        t1264 = paddle._C_ops.multiply(input33, t1263)
        del input33

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 1xf32)
        t1265 = paddle._C_ops.add(t1264, input34)
        del input34

        # pd_op.hardswish: (8x96x80x80xf32) <- (8x96x80x80xf32)
        t1266 = paddle._C_ops.hardswish(t1265)

        # pd_op.multiply: (8x96x80x80xf32) <- (1xf32, 8x96x80x80xf32)
        t1267 = paddle._C_ops.multiply(input35, t1266)
        del input35

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 1xf32)
        t1268 = paddle._C_ops.add(t1267, input36)
        del input36

        # pd_op.depthwise_conv2d: (8x96x40x40xf32) <- (8x96x80x80xf32, 96x1x1x1xf32)
        t1269 = paddle._C_ops.depthwise_conv2d(
            t1268, t250, [2, 2], [0, 0], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t250

        # pd_op.batch_norm_: (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1270, t1271, t1272, t1273, t1274, t1275 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1269,
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

        # pd_op.scale: (8x96x40x40xf32) <- (8x96x40x40xf32, 1xf32)
        t1276 = paddle._C_ops.scale(t1270, t807, float("0"), True)
        del t1270

        # pd_op.depthwise_conv2d: (8x96x40x40xf32) <- (8x96x80x80xf32, 96x1x3x3xf32)
        t1277 = paddle._C_ops.depthwise_conv2d(
            t1268, t255, [2, 2], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t255

        # pd_op.batch_norm_: (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1278, t1279, t1280, t1281, t1282, t1283 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1277,
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

        # pd_op.add: (8x96x40x40xf32) <- (8x96x40x40xf32, 8x96x40x40xf32)
        t1284 = paddle._C_ops.add(t1276, t1278)

        # pd_op.depthwise_conv2d: (8x96x40x40xf32) <- (8x96x80x80xf32, 96x1x3x3xf32)
        t1285 = paddle._C_ops.depthwise_conv2d(
            t1268, t260, [2, 2], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t260

        # pd_op.batch_norm_: (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1286, t1287, t1288, t1289, t1290, t1291 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1285,
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

        # pd_op.add: (8x96x40x40xf32) <- (8x96x40x40xf32, 8x96x40x40xf32)
        t1292 = paddle._C_ops.add(t1284, t1286)

        # pd_op.depthwise_conv2d: (8x96x40x40xf32) <- (8x96x80x80xf32, 96x1x3x3xf32)
        t1293 = paddle._C_ops.depthwise_conv2d(
            t1268, t265, [2, 2], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t265

        # pd_op.batch_norm_: (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1294, t1295, t1296, t1297, t1298, t1299 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1293,
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

        # pd_op.add: (8x96x40x40xf32) <- (8x96x40x40xf32, 8x96x40x40xf32)
        t1300 = paddle._C_ops.add(t1292, t1294)

        # pd_op.depthwise_conv2d: (8x96x40x40xf32) <- (8x96x80x80xf32, 96x1x3x3xf32)
        t1301 = paddle._C_ops.depthwise_conv2d(
            t1268, t270, [2, 2], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del t270

        # pd_op.batch_norm_: (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (8x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1302, t1303, t1304, t1305, t1306, t1307 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1301,
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

        # pd_op.add: (8x96x40x40xf32) <- (8x96x40x40xf32, 8x96x40x40xf32)
        t1308 = paddle._C_ops.add(t1300, t1302)

        # pd_op.multiply: (8x96x40x40xf32) <- (1xf32, 8x96x40x40xf32)
        t1309 = paddle._C_ops.multiply(input37, t1308)
        del input37

        # pd_op.add: (8x96x40x40xf32) <- (8x96x40x40xf32, 1xf32)
        t1310 = paddle._C_ops.add(t1309, input38)
        del input38

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x96x40x40xf32, 192x96x1x1xf32)
        t1311 = paddle._C_ops.conv2d(
            t1310, t275, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t275

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1312, t1313, t1314, t1315, t1316, t1317 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1311,
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

        # pd_op.scale: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1318 = paddle._C_ops.scale(t1312, t807, float("0"), True)
        del t1312

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x96x40x40xf32, 192x96x1x1xf32)
        t1319 = paddle._C_ops.conv2d(
            t1310, t280, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t280

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1320, t1321, t1322, t1323, t1324, t1325 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1319,
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

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1326 = paddle._C_ops.add(t1318, t1320)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x96x40x40xf32, 192x96x1x1xf32)
        t1327 = paddle._C_ops.conv2d(
            t1310, t285, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t285

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1328, t1329, t1330, t1331, t1332, t1333 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1327,
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

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1334 = paddle._C_ops.add(t1326, t1328)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x96x40x40xf32, 192x96x1x1xf32)
        t1335 = paddle._C_ops.conv2d(
            t1310, t290, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t290

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1336, t1337, t1338, t1339, t1340, t1341 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1335,
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

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1342 = paddle._C_ops.add(t1334, t1336)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1343 = paddle._C_ops.multiply(input39, t1342)
        del input39

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1344 = paddle._C_ops.add(t1343, input40)
        del input40

        # pd_op.hardswish: (8x192x40x40xf32) <- (8x192x40x40xf32)
        t1345 = paddle._C_ops.hardswish(t1344)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1346 = paddle._C_ops.multiply(input41, t1345)
        del input41

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1347 = paddle._C_ops.add(t1346, input42)
        del input42

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1348, t1349, t1350, t1351, t1352, t1353 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1347,
                t295,
                t296,
                t297,
                t298,
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
        del t298, t297, t296, t295

        # pd_op.scale: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1354 = paddle._C_ops.scale(t1348, t807, float("0"), True)
        del t1348

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x1x1xf32)
        t1355 = paddle._C_ops.depthwise_conv2d(
            t1347, t299, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t299

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1356, t1357, t1358, t1359, t1360, t1361 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1355,
                t300,
                t301,
                t302,
                t303,
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
        del t303, t302, t301, t300

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1362 = paddle._C_ops.add(t1354, t1356)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1363 = paddle._C_ops.depthwise_conv2d(
            t1347, t304, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t304

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1364, t1365, t1366, t1367, t1368, t1369 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1363,
                t305,
                t306,
                t307,
                t308,
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
        del t308, t307, t306, t305

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1370 = paddle._C_ops.add(t1362, t1364)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1371 = paddle._C_ops.depthwise_conv2d(
            t1347, t309, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t309

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1372, t1373, t1374, t1375, t1376, t1377 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1371,
                t310,
                t311,
                t312,
                t313,
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
        del t313, t312, t311, t310

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1378 = paddle._C_ops.add(t1370, t1372)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1379 = paddle._C_ops.depthwise_conv2d(
            t1347, t314, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t314

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1380, t1381, t1382, t1383, t1384, t1385 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1379,
                t315,
                t316,
                t317,
                t318,
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
        del t318, t317, t316, t315

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1386 = paddle._C_ops.add(t1378, t1380)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1387 = paddle._C_ops.depthwise_conv2d(
            t1347, t319, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t319

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1388, t1389, t1390, t1391, t1392, t1393 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1387,
                t320,
                t321,
                t322,
                t323,
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
        del t323, t322, t321, t320

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1394 = paddle._C_ops.add(t1386, t1388)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1395 = paddle._C_ops.multiply(input43, t1394)
        del input43

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1396 = paddle._C_ops.add(t1395, input44)
        del input44

        # pd_op.hardswish: (8x192x40x40xf32) <- (8x192x40x40xf32)
        t1397 = paddle._C_ops.hardswish(t1396)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1398 = paddle._C_ops.multiply(input45, t1397)
        del input45

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1399 = paddle._C_ops.add(t1398, input46)
        del input46

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1400, t1401, t1402, t1403, t1404, t1405 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1399,
                t324,
                t325,
                t326,
                t327,
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
        del t327, t326, t325, t324

        # pd_op.scale: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1406 = paddle._C_ops.scale(t1400, t807, float("0"), True)
        del t1400

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1407 = paddle._C_ops.conv2d(
            t1399, t328, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t328

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1408, t1409, t1410, t1411, t1412, t1413 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1407,
                t329,
                t330,
                t331,
                t332,
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
        del t332, t331, t330, t329

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1414 = paddle._C_ops.add(t1406, t1408)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1415 = paddle._C_ops.conv2d(
            t1399, t333, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t333

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1416, t1417, t1418, t1419, t1420, t1421 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1415,
                t334,
                t335,
                t336,
                t337,
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
        del t337, t336, t335, t334

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1422 = paddle._C_ops.add(t1414, t1416)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1423 = paddle._C_ops.conv2d(
            t1399, t338, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t338

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1424, t1425, t1426, t1427, t1428, t1429 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1423,
                t339,
                t340,
                t341,
                t342,
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
        del t342, t341, t340, t339

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1430 = paddle._C_ops.add(t1422, t1424)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1431 = paddle._C_ops.conv2d(
            t1399, t343, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t343

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1432, t1433, t1434, t1435, t1436, t1437 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1431,
                t344,
                t345,
                t346,
                t347,
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
        del t347, t346, t345, t344

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1438 = paddle._C_ops.add(t1430, t1432)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1439 = paddle._C_ops.multiply(input47, t1438)
        del input47

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1440 = paddle._C_ops.add(t1439, input48)
        del input48

        # pd_op.hardswish: (8x192x40x40xf32) <- (8x192x40x40xf32)
        t1441 = paddle._C_ops.hardswish(t1440)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1442 = paddle._C_ops.multiply(input49, t1441)
        del input49

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1443 = paddle._C_ops.add(t1442, input50)
        del input50

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1444, t1445, t1446, t1447, t1448, t1449 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1443,
                t348,
                t349,
                t350,
                t351,
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
        del t351, t350, t349, t348

        # pd_op.scale: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1450 = paddle._C_ops.scale(t1444, t807, float("0"), True)
        del t1444

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x1x1xf32)
        t1451 = paddle._C_ops.depthwise_conv2d(
            t1443, t352, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t352

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1452, t1453, t1454, t1455, t1456, t1457 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1451,
                t353,
                t354,
                t355,
                t356,
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
        del t356, t355, t354, t353

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1458 = paddle._C_ops.add(t1450, t1452)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1459 = paddle._C_ops.depthwise_conv2d(
            t1443, t357, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t357

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1460, t1461, t1462, t1463, t1464, t1465 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1459,
                t358,
                t359,
                t360,
                t361,
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
        del t361, t360, t359, t358

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1466 = paddle._C_ops.add(t1458, t1460)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1467 = paddle._C_ops.depthwise_conv2d(
            t1443, t362, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t362

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1468, t1469, t1470, t1471, t1472, t1473 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1467,
                t363,
                t364,
                t365,
                t366,
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
        del t366, t365, t364, t363

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1474 = paddle._C_ops.add(t1466, t1468)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1475 = paddle._C_ops.depthwise_conv2d(
            t1443, t367, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t367

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1476, t1477, t1478, t1479, t1480, t1481 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1475,
                t368,
                t369,
                t370,
                t371,
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
        del t371, t370, t369, t368

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1482 = paddle._C_ops.add(t1474, t1476)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1483 = paddle._C_ops.depthwise_conv2d(
            t1443, t372, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t372

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1484, t1485, t1486, t1487, t1488, t1489 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1483,
                t373,
                t374,
                t375,
                t376,
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
        del t376, t375, t374, t373

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1490 = paddle._C_ops.add(t1482, t1484)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1491 = paddle._C_ops.multiply(input51, t1490)
        del input51

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1492 = paddle._C_ops.add(t1491, input52)
        del input52

        # pd_op.hardswish: (8x192x40x40xf32) <- (8x192x40x40xf32)
        t1493 = paddle._C_ops.hardswish(t1492)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1494 = paddle._C_ops.multiply(input53, t1493)
        del input53

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1495 = paddle._C_ops.add(t1494, input54)
        del input54

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1496, t1497, t1498, t1499, t1500, t1501 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1495,
                t377,
                t378,
                t379,
                t380,
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
        del t380, t379, t378, t377

        # pd_op.scale: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1502 = paddle._C_ops.scale(t1496, t807, float("0"), True)
        del t1496

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1503 = paddle._C_ops.conv2d(
            t1495, t381, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t381

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1504, t1505, t1506, t1507, t1508, t1509 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1503,
                t382,
                t383,
                t384,
                t385,
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
        del t385, t384, t383, t382

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1510 = paddle._C_ops.add(t1502, t1504)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1511 = paddle._C_ops.conv2d(
            t1495, t386, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t386

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1512, t1513, t1514, t1515, t1516, t1517 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1511,
                t387,
                t388,
                t389,
                t390,
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
        del t390, t389, t388, t387

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1518 = paddle._C_ops.add(t1510, t1512)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1519 = paddle._C_ops.conv2d(
            t1495, t391, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t391

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1520, t1521, t1522, t1523, t1524, t1525 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1519,
                t392,
                t393,
                t394,
                t395,
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
        del t395, t394, t393, t392

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1526 = paddle._C_ops.add(t1518, t1520)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1527 = paddle._C_ops.conv2d(
            t1495, t396, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t396

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1528, t1529, t1530, t1531, t1532, t1533 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1527,
                t397,
                t398,
                t399,
                t400,
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
        del t400, t399, t398, t397

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1534 = paddle._C_ops.add(t1526, t1528)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1535 = paddle._C_ops.multiply(input55, t1534)
        del input55

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1536 = paddle._C_ops.add(t1535, input56)
        del input56

        # pd_op.hardswish: (8x192x40x40xf32) <- (8x192x40x40xf32)
        t1537 = paddle._C_ops.hardswish(t1536)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1538 = paddle._C_ops.multiply(input57, t1537)
        del input57

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1539 = paddle._C_ops.add(t1538, input58)
        del input58

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1540, t1541, t1542, t1543, t1544, t1545 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1539,
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

        # pd_op.scale: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1546 = paddle._C_ops.scale(t1540, t807, float("0"), True)
        del t1540

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x1x1xf32)
        t1547 = paddle._C_ops.depthwise_conv2d(
            t1539, t405, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t405

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1548, t1549, t1550, t1551, t1552, t1553 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1547,
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

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1554 = paddle._C_ops.add(t1546, t1548)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1555 = paddle._C_ops.depthwise_conv2d(
            t1539, t410, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t410

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1556, t1557, t1558, t1559, t1560, t1561 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1555,
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

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1562 = paddle._C_ops.add(t1554, t1556)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1563 = paddle._C_ops.depthwise_conv2d(
            t1539, t415, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t415

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1564, t1565, t1566, t1567, t1568, t1569 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1563,
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

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1570 = paddle._C_ops.add(t1562, t1564)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1571 = paddle._C_ops.depthwise_conv2d(
            t1539, t420, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t420

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1572, t1573, t1574, t1575, t1576, t1577 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1571,
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

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1578 = paddle._C_ops.add(t1570, t1572)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1579 = paddle._C_ops.depthwise_conv2d(
            t1539, t425, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t425

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1580, t1581, t1582, t1583, t1584, t1585 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1579,
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

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1586 = paddle._C_ops.add(t1578, t1580)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1587 = paddle._C_ops.multiply(input59, t1586)
        del input59

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1588 = paddle._C_ops.add(t1587, input60)
        del input60

        # pd_op.hardswish: (8x192x40x40xf32) <- (8x192x40x40xf32)
        t1589 = paddle._C_ops.hardswish(t1588)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1590 = paddle._C_ops.multiply(input61, t1589)
        del input61

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1591 = paddle._C_ops.add(t1590, input62)
        del input62

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1592, t1593, t1594, t1595, t1596, t1597 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1591,
                t430,
                t431,
                t432,
                t433,
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
        del t433, t432, t431, t430

        # pd_op.scale: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1598 = paddle._C_ops.scale(t1592, t807, float("0"), True)
        del t1592

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1599 = paddle._C_ops.conv2d(
            t1591, t434, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t434

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1600, t1601, t1602, t1603, t1604, t1605 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1599,
                t435,
                t436,
                t437,
                t438,
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
        del t438, t437, t436, t435

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1606 = paddle._C_ops.add(t1598, t1600)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1607 = paddle._C_ops.conv2d(
            t1591, t439, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t439

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1608, t1609, t1610, t1611, t1612, t1613 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1607,
                t440,
                t441,
                t442,
                t443,
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
        del t443, t442, t441, t440

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1614 = paddle._C_ops.add(t1606, t1608)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1615 = paddle._C_ops.conv2d(
            t1591, t444, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t444

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1616, t1617, t1618, t1619, t1620, t1621 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1615,
                t445,
                t446,
                t447,
                t448,
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
        del t448, t447, t446, t445

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1622 = paddle._C_ops.add(t1614, t1616)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1623 = paddle._C_ops.conv2d(
            t1591, t449, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t449

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1624, t1625, t1626, t1627, t1628, t1629 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1623,
                t450,
                t451,
                t452,
                t453,
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
        del t453, t452, t451, t450

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1630 = paddle._C_ops.add(t1622, t1624)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1631 = paddle._C_ops.multiply(input63, t1630)
        del input63

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1632 = paddle._C_ops.add(t1631, input64)
        del input64

        # pd_op.hardswish: (8x192x40x40xf32) <- (8x192x40x40xf32)
        t1633 = paddle._C_ops.hardswish(t1632)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1634 = paddle._C_ops.multiply(input65, t1633)
        del input65

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1635 = paddle._C_ops.add(t1634, input66)
        del input66

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1636, t1637, t1638, t1639, t1640, t1641 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1635,
                t454,
                t455,
                t456,
                t457,
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
        del t457, t456, t455, t454

        # pd_op.scale: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1642 = paddle._C_ops.scale(t1636, t807, float("0"), True)
        del t1636

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x1x1xf32)
        t1643 = paddle._C_ops.depthwise_conv2d(
            t1635, t458, [1, 1], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t458

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1644, t1645, t1646, t1647, t1648, t1649 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1643,
                t459,
                t460,
                t461,
                t462,
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
        del t462, t461, t460, t459

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1650 = paddle._C_ops.add(t1642, t1644)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1651 = paddle._C_ops.depthwise_conv2d(
            t1635, t463, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t463

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1652, t1653, t1654, t1655, t1656, t1657 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1651,
                t464,
                t465,
                t466,
                t467,
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
        del t467, t466, t465, t464

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1658 = paddle._C_ops.add(t1650, t1652)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1659 = paddle._C_ops.depthwise_conv2d(
            t1635, t468, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t468

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1660, t1661, t1662, t1663, t1664, t1665 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1659,
                t469,
                t470,
                t471,
                t472,
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
        del t472, t471, t470, t469

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1666 = paddle._C_ops.add(t1658, t1660)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1667 = paddle._C_ops.depthwise_conv2d(
            t1635, t473, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t473

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1668, t1669, t1670, t1671, t1672, t1673 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1667,
                t474,
                t475,
                t476,
                t477,
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
        del t477, t476, t475, t474

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1674 = paddle._C_ops.add(t1666, t1668)

        # pd_op.depthwise_conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1675 = paddle._C_ops.depthwise_conv2d(
            t1635, t478, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t478

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1676, t1677, t1678, t1679, t1680, t1681 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1675,
                t479,
                t480,
                t481,
                t482,
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
        del t482, t481, t480, t479

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1682 = paddle._C_ops.add(t1674, t1676)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1683 = paddle._C_ops.multiply(input67, t1682)
        del input67

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1684 = paddle._C_ops.add(t1683, input68)
        del input68

        # pd_op.hardswish: (8x192x40x40xf32) <- (8x192x40x40xf32)
        t1685 = paddle._C_ops.hardswish(t1684)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1686 = paddle._C_ops.multiply(input69, t1685)
        del input69

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1687 = paddle._C_ops.add(t1686, input70)
        del input70

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1688, t1689, t1690, t1691, t1692, t1693 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1687,
                t483,
                t484,
                t485,
                t486,
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
        del t486, t485, t484, t483

        # pd_op.scale: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1694 = paddle._C_ops.scale(t1688, t807, float("0"), True)
        del t1688

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1695 = paddle._C_ops.conv2d(
            t1687, t487, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t487

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1696, t1697, t1698, t1699, t1700, t1701 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1695,
                t488,
                t489,
                t490,
                t491,
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
        del t491, t490, t489, t488

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1702 = paddle._C_ops.add(t1694, t1696)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1703 = paddle._C_ops.conv2d(
            t1687, t492, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t492

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1704, t1705, t1706, t1707, t1708, t1709 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1703,
                t493,
                t494,
                t495,
                t496,
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
        del t496, t495, t494, t493

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1710 = paddle._C_ops.add(t1702, t1704)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1711 = paddle._C_ops.conv2d(
            t1687, t497, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t497

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1712, t1713, t1714, t1715, t1716, t1717 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1711,
                t498,
                t499,
                t500,
                t501,
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
        del t501, t500, t499, t498

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1718 = paddle._C_ops.add(t1710, t1712)

        # pd_op.conv2d: (8x192x40x40xf32) <- (8x192x40x40xf32, 192x192x1x1xf32)
        t1719 = paddle._C_ops.conv2d(
            t1687, t502, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t502

        # pd_op.batch_norm_: (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1720, t1721, t1722, t1723, t1724, t1725 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1719,
                t503,
                t504,
                t505,
                t506,
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
        del t506, t505, t504, t503

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 8x192x40x40xf32)
        t1726 = paddle._C_ops.add(t1718, t1720)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1727 = paddle._C_ops.multiply(input71, t1726)
        del input71

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1728 = paddle._C_ops.add(t1727, input72)
        del input72

        # pd_op.hardswish: (8x192x40x40xf32) <- (8x192x40x40xf32)
        t1729 = paddle._C_ops.hardswish(t1728)

        # pd_op.multiply: (8x192x40x40xf32) <- (1xf32, 8x192x40x40xf32)
        t1730 = paddle._C_ops.multiply(input73, t1729)
        del input73

        # pd_op.add: (8x192x40x40xf32) <- (8x192x40x40xf32, 1xf32)
        t1731 = paddle._C_ops.add(t1730, input74)
        del input74

        # pd_op.depthwise_conv2d: (8x192x20x20xf32) <- (8x192x40x40xf32, 192x1x1x1xf32)
        t1732 = paddle._C_ops.depthwise_conv2d(
            t1731, t507, [2, 2], [0, 0], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t507

        # pd_op.batch_norm_: (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1733, t1734, t1735, t1736, t1737, t1738 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1732,
                t508,
                t509,
                t510,
                t511,
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
        del t511, t510, t509, t508

        # pd_op.scale: (8x192x20x20xf32) <- (8x192x20x20xf32, 1xf32)
        t1739 = paddle._C_ops.scale(t1733, t807, float("0"), True)
        del t1733

        # pd_op.depthwise_conv2d: (8x192x20x20xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1740 = paddle._C_ops.depthwise_conv2d(
            t1731, t512, [2, 2], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t512

        # pd_op.batch_norm_: (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1741, t1742, t1743, t1744, t1745, t1746 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1740,
                t513,
                t514,
                t515,
                t516,
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
        del t516, t515, t514, t513

        # pd_op.add: (8x192x20x20xf32) <- (8x192x20x20xf32, 8x192x20x20xf32)
        t1747 = paddle._C_ops.add(t1739, t1741)

        # pd_op.depthwise_conv2d: (8x192x20x20xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1748 = paddle._C_ops.depthwise_conv2d(
            t1731, t517, [2, 2], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t517

        # pd_op.batch_norm_: (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1749, t1750, t1751, t1752, t1753, t1754 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1748,
                t518,
                t519,
                t520,
                t521,
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
        del t521, t520, t519, t518

        # pd_op.add: (8x192x20x20xf32) <- (8x192x20x20xf32, 8x192x20x20xf32)
        t1755 = paddle._C_ops.add(t1747, t1749)

        # pd_op.depthwise_conv2d: (8x192x20x20xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1756 = paddle._C_ops.depthwise_conv2d(
            t1731, t522, [2, 2], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t522

        # pd_op.batch_norm_: (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1757, t1758, t1759, t1760, t1761, t1762 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1756,
                t523,
                t524,
                t525,
                t526,
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
        del t526, t525, t524, t523

        # pd_op.add: (8x192x20x20xf32) <- (8x192x20x20xf32, 8x192x20x20xf32)
        t1763 = paddle._C_ops.add(t1755, t1757)

        # pd_op.depthwise_conv2d: (8x192x20x20xf32) <- (8x192x40x40xf32, 192x1x5x5xf32)
        t1764 = paddle._C_ops.depthwise_conv2d(
            t1731, t527, [2, 2], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t527

        # pd_op.batch_norm_: (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (8x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1765, t1766, t1767, t1768, t1769, t1770 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1764,
                t528,
                t529,
                t530,
                t531,
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
        del t531, t530, t529, t528

        # pd_op.add: (8x192x20x20xf32) <- (8x192x20x20xf32, 8x192x20x20xf32)
        t1771 = paddle._C_ops.add(t1763, t1765)

        # pd_op.multiply: (8x192x20x20xf32) <- (1xf32, 8x192x20x20xf32)
        t1772 = paddle._C_ops.multiply(input75, t1771)
        del input75

        # pd_op.add: (8x192x20x20xf32) <- (8x192x20x20xf32, 1xf32)
        t1773 = paddle._C_ops.add(t1772, input76)
        del input76

        # pd_op.full_int_array: (2xi64) <- ()
        t1774 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        t1775 = t1774

        # pd_op.assign: (2xi64) <- (2xi64)
        t1776 = t1774

        # pd_op.assign: (2xi64) <- (2xi64)
        t1777 = t1774

        # pd_op.assign: (2xi64) <- (2xi64)
        t1778 = t1774

        # pd_op.assign: (2xi64) <- (2xi64)
        t1779 = t1774

        # pd_op.assign: (2xi64) <- (2xi64)
        t1780 = t1774

        # pd_op.assign: (2xi64) <- (2xi64)
        t1781 = t1774

        # pd_op.assign: (2xi64) <- (2xi64)
        t1782 = t1774

        # pd_op.assign: (2xi64) <- (2xi64)
        t1783 = t1774

        # pd_op.pool2d: (8x192x1x1xf32) <- (8x192x20x20xf32, 2xi64)
        t1784 = paddle._C_ops.pool2d(
            t1773,
            t1774,
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

        # pd_op.conv2d: (8x48x1x1xf32) <- (8x192x1x1xf32, 48x192x1x1xf32)
        t1785 = paddle._C_ops.conv2d(
            t1784, t532, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t532

        # pd_op.full_int_array: (4xi64) <- ()
        t1786 = [1, -1, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        t1787 = paddle._C_ops.reshape(t533, t1786)
        del t533

        # pd_op.add: (8x48x1x1xf32) <- (8x48x1x1xf32, 1x48x1x1xf32)
        t1788 = paddle._C_ops.add(t1785, t1787)

        # pd_op.relu: (8x48x1x1xf32) <- (8x48x1x1xf32)
        t1789 = paddle._C_ops.relu(t1788)
        del t1788

        # pd_op.conv2d: (8x192x1x1xf32) <- (8x48x1x1xf32, 192x48x1x1xf32)
        t1790 = paddle._C_ops.conv2d(
            t1789, t534, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t534

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        t1791 = paddle._C_ops.reshape(t535, t1786)
        del t535

        # pd_op.add: (8x192x1x1xf32) <- (8x192x1x1xf32, 1x192x1x1xf32)
        t1792 = paddle._C_ops.add(t1790, t1791)

        # pd_op.hardsigmoid: (8x192x1x1xf32) <- (8x192x1x1xf32)
        t1793 = paddle._C_ops.hardsigmoid(t1792, float("0.166667"), float("0.5"))
        del t1792

        # pd_op.multiply: (8x192x20x20xf32) <- (8x192x20x20xf32, 8x192x1x1xf32)
        t1794 = paddle._C_ops.multiply(t1773, t1793)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x192x20x20xf32, 384x192x1x1xf32)
        t1795 = paddle._C_ops.conv2d(
            t1794, t536, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t536

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1796, t1797, t1798, t1799, t1800, t1801 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1795,
                t537,
                t538,
                t539,
                t540,
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
        del t540, t539, t538, t537

        # pd_op.scale: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1802 = paddle._C_ops.scale(t1796, t807, float("0"), True)
        del t1796

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x192x20x20xf32, 384x192x1x1xf32)
        t1803 = paddle._C_ops.conv2d(
            t1794, t541, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t541

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1804, t1805, t1806, t1807, t1808, t1809 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1803,
                t542,
                t543,
                t544,
                t545,
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
        del t545, t544, t543, t542

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1810 = paddle._C_ops.add(t1802, t1804)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x192x20x20xf32, 384x192x1x1xf32)
        t1811 = paddle._C_ops.conv2d(
            t1794, t546, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t546

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1812, t1813, t1814, t1815, t1816, t1817 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1811,
                t547,
                t548,
                t549,
                t550,
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
        del t550, t549, t548, t547

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1818 = paddle._C_ops.add(t1810, t1812)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x192x20x20xf32, 384x192x1x1xf32)
        t1819 = paddle._C_ops.conv2d(
            t1794, t551, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t551

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1820, t1821, t1822, t1823, t1824, t1825 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1819,
                t552,
                t553,
                t554,
                t555,
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
        del t555, t554, t553, t552

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1826 = paddle._C_ops.add(t1818, t1820)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t1827 = paddle._C_ops.multiply(input77, t1826)
        del input77

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1828 = paddle._C_ops.add(t1827, input78)
        del input78

        # pd_op.hardswish: (8x384x20x20xf32) <- (8x384x20x20xf32)
        t1829 = paddle._C_ops.hardswish(t1828)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t1830 = paddle._C_ops.multiply(input79, t1829)
        del input79

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1831 = paddle._C_ops.add(t1830, input80)
        del input80

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1832, t1833, t1834, t1835, t1836, t1837 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1831,
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

        # pd_op.scale: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1838 = paddle._C_ops.scale(t1832, t807, float("0"), True)
        del t1832

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x1x1xf32)
        t1839 = paddle._C_ops.depthwise_conv2d(
            t1831, t560, [1, 1], [0, 0], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t560

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1840, t1841, t1842, t1843, t1844, t1845 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1839,
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

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1846 = paddle._C_ops.add(t1838, t1840)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t1847 = paddle._C_ops.depthwise_conv2d(
            t1831, t565, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t565

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1848, t1849, t1850, t1851, t1852, t1853 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1847,
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

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1854 = paddle._C_ops.add(t1846, t1848)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t1855 = paddle._C_ops.depthwise_conv2d(
            t1831, t570, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t570

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1856, t1857, t1858, t1859, t1860, t1861 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1855,
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

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1862 = paddle._C_ops.add(t1854, t1856)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t1863 = paddle._C_ops.depthwise_conv2d(
            t1831, t575, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t575

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1864, t1865, t1866, t1867, t1868, t1869 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1863,
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

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1870 = paddle._C_ops.add(t1862, t1864)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t1871 = paddle._C_ops.depthwise_conv2d(
            t1831, t580, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t580

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1872, t1873, t1874, t1875, t1876, t1877 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1871,
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

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1878 = paddle._C_ops.add(t1870, t1872)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t1879 = paddle._C_ops.multiply(input81, t1878)
        del input81

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1880 = paddle._C_ops.add(t1879, input82)
        del input82

        # pd_op.hardswish: (8x384x20x20xf32) <- (8x384x20x20xf32)
        t1881 = paddle._C_ops.hardswish(t1880)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t1882 = paddle._C_ops.multiply(input83, t1881)
        del input83

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1883 = paddle._C_ops.add(t1882, input84)
        del input84

        # pd_op.pool2d: (8x384x1x1xf32) <- (8x384x20x20xf32, 2xi64)
        t1884 = paddle._C_ops.pool2d(
            t1883,
            t1774,
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

        # pd_op.conv2d: (8x96x1x1xf32) <- (8x384x1x1xf32, 96x384x1x1xf32)
        t1885 = paddle._C_ops.conv2d(
            t1884, t585, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t585

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t1886 = paddle._C_ops.reshape(t586, t1786)
        del t586

        # pd_op.add: (8x96x1x1xf32) <- (8x96x1x1xf32, 1x96x1x1xf32)
        t1887 = paddle._C_ops.add(t1885, t1886)

        # pd_op.relu: (8x96x1x1xf32) <- (8x96x1x1xf32)
        t1888 = paddle._C_ops.relu(t1887)
        del t1887

        # pd_op.conv2d: (8x384x1x1xf32) <- (8x96x1x1xf32, 384x96x1x1xf32)
        t1889 = paddle._C_ops.conv2d(
            t1888, t587, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t587

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        t1890 = paddle._C_ops.reshape(t588, t1786)
        del t588

        # pd_op.add: (8x384x1x1xf32) <- (8x384x1x1xf32, 1x384x1x1xf32)
        t1891 = paddle._C_ops.add(t1889, t1890)

        # pd_op.hardsigmoid: (8x384x1x1xf32) <- (8x384x1x1xf32)
        t1892 = paddle._C_ops.hardsigmoid(t1891, float("0.166667"), float("0.5"))
        del t1891

        # pd_op.multiply: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x1x1xf32)
        t1893 = paddle._C_ops.multiply(t1883, t1892)

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1894, t1895, t1896, t1897, t1898, t1899 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1893,
                t589,
                t590,
                t591,
                t592,
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
        del t592, t591, t590, t589

        # pd_op.scale: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1900 = paddle._C_ops.scale(t1894, t807, float("0"), True)
        del t1894

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t1901 = paddle._C_ops.conv2d(
            t1893, t593, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t593

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1902, t1903, t1904, t1905, t1906, t1907 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1901,
                t594,
                t595,
                t596,
                t597,
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
        del t597, t596, t595, t594

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1908 = paddle._C_ops.add(t1900, t1902)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t1909 = paddle._C_ops.conv2d(
            t1893, t598, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t598

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1910, t1911, t1912, t1913, t1914, t1915 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1909,
                t599,
                t600,
                t601,
                t602,
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
        del t602, t601, t600, t599

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1916 = paddle._C_ops.add(t1908, t1910)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t1917 = paddle._C_ops.conv2d(
            t1893, t603, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t603

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1918, t1919, t1920, t1921, t1922, t1923 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1917,
                t604,
                t605,
                t606,
                t607,
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
        del t607, t606, t605, t604

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1924 = paddle._C_ops.add(t1916, t1918)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t1925 = paddle._C_ops.conv2d(
            t1893, t608, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t608

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1926, t1927, t1928, t1929, t1930, t1931 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1925,
                t609,
                t610,
                t611,
                t612,
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
        del t612, t611, t610, t609

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1932 = paddle._C_ops.add(t1924, t1926)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t1933 = paddle._C_ops.multiply(input85, t1932)
        del input85

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1934 = paddle._C_ops.add(t1933, input86)
        del input86

        # pd_op.hardswish: (8x384x20x20xf32) <- (8x384x20x20xf32)
        t1935 = paddle._C_ops.hardswish(t1934)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t1936 = paddle._C_ops.multiply(input87, t1935)
        del input87

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1937 = paddle._C_ops.add(t1936, input88)
        del input88

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1938, t1939, t1940, t1941, t1942, t1943 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1937,
                t613,
                t614,
                t615,
                t616,
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
        del t616, t615, t614, t613

        # pd_op.scale: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1944 = paddle._C_ops.scale(t1938, t807, float("0"), True)
        del t1938

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x1x1xf32)
        t1945 = paddle._C_ops.depthwise_conv2d(
            t1937, t617, [1, 1], [0, 0], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t617

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1946, t1947, t1948, t1949, t1950, t1951 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1945,
                t618,
                t619,
                t620,
                t621,
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
        del t621, t620, t619, t618

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1952 = paddle._C_ops.add(t1944, t1946)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t1953 = paddle._C_ops.depthwise_conv2d(
            t1937, t622, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t622

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1954, t1955, t1956, t1957, t1958, t1959 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1953,
                t623,
                t624,
                t625,
                t626,
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
        del t626, t625, t624, t623

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1960 = paddle._C_ops.add(t1952, t1954)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t1961 = paddle._C_ops.depthwise_conv2d(
            t1937, t627, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t627

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1962, t1963, t1964, t1965, t1966, t1967 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1961,
                t628,
                t629,
                t630,
                t631,
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
        del t631, t630, t629, t628

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1968 = paddle._C_ops.add(t1960, t1962)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t1969 = paddle._C_ops.depthwise_conv2d(
            t1937, t632, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t632

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1970, t1971, t1972, t1973, t1974, t1975 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1969,
                t633,
                t634,
                t635,
                t636,
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
        del t636, t635, t634, t633

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1976 = paddle._C_ops.add(t1968, t1970)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t1977 = paddle._C_ops.depthwise_conv2d(
            t1937, t637, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t637

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1978, t1979, t1980, t1981, t1982, t1983 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1977,
                t638,
                t639,
                t640,
                t641,
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
        del t641, t640, t639, t638

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t1984 = paddle._C_ops.add(t1976, t1978)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t1985 = paddle._C_ops.multiply(input89, t1984)
        del input89

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1986 = paddle._C_ops.add(t1985, input90)
        del input90

        # pd_op.hardswish: (8x384x20x20xf32) <- (8x384x20x20xf32)
        t1987 = paddle._C_ops.hardswish(t1986)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t1988 = paddle._C_ops.multiply(input91, t1987)
        del input91

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1989 = paddle._C_ops.add(t1988, input92)
        del input92

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1990, t1991, t1992, t1993, t1994, t1995 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1989,
                t642,
                t643,
                t644,
                t645,
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
        del t645, t644, t643, t642

        # pd_op.scale: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t1996 = paddle._C_ops.scale(t1990, t807, float("0"), True)
        del t1990

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t1997 = paddle._C_ops.conv2d(
            t1989, t646, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t646

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1998, t1999, t2000, t2001, t2002, t2003 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1997,
                t647,
                t648,
                t649,
                t650,
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
        del t650, t649, t648, t647

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2004 = paddle._C_ops.add(t1996, t1998)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t2005 = paddle._C_ops.conv2d(
            t1989, t651, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t651

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2006, t2007, t2008, t2009, t2010, t2011 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2005,
                t652,
                t653,
                t654,
                t655,
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
        del t655, t654, t653, t652

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2012 = paddle._C_ops.add(t2004, t2006)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t2013 = paddle._C_ops.conv2d(
            t1989, t656, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t656

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2014, t2015, t2016, t2017, t2018, t2019 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2013,
                t657,
                t658,
                t659,
                t660,
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
        del t660, t659, t658, t657

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2020 = paddle._C_ops.add(t2012, t2014)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t2021 = paddle._C_ops.conv2d(
            t1989, t661, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t661

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2022, t2023, t2024, t2025, t2026, t2027 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2021,
                t662,
                t663,
                t664,
                t665,
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
        del t665, t664, t663, t662

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2028 = paddle._C_ops.add(t2020, t2022)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t2029 = paddle._C_ops.multiply(input93, t2028)
        del input93

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t2030 = paddle._C_ops.add(t2029, input94)
        del input94

        # pd_op.hardswish: (8x384x20x20xf32) <- (8x384x20x20xf32)
        t2031 = paddle._C_ops.hardswish(t2030)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t2032 = paddle._C_ops.multiply(input95, t2031)
        del input95

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t2033 = paddle._C_ops.add(t2032, input96)
        del input96

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2034, t2035, t2036, t2037, t2038, t2039 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2033,
                t666,
                t667,
                t668,
                t669,
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
        del t669, t668, t667, t666

        # pd_op.scale: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t2040 = paddle._C_ops.scale(t2034, t807, float("0"), True)
        del t2034

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x1x1xf32)
        t2041 = paddle._C_ops.depthwise_conv2d(
            t2033, t670, [1, 1], [0, 0], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t670

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2042, t2043, t2044, t2045, t2046, t2047 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2041,
                t671,
                t672,
                t673,
                t674,
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
        del t674, t673, t672, t671

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2048 = paddle._C_ops.add(t2040, t2042)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t2049 = paddle._C_ops.depthwise_conv2d(
            t2033, t675, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t675

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2050, t2051, t2052, t2053, t2054, t2055 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2049,
                t676,
                t677,
                t678,
                t679,
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
        del t679, t678, t677, t676

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2056 = paddle._C_ops.add(t2048, t2050)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t2057 = paddle._C_ops.depthwise_conv2d(
            t2033, t680, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t680

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2058, t2059, t2060, t2061, t2062, t2063 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2057,
                t681,
                t682,
                t683,
                t684,
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
        del t684, t683, t682, t681

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2064 = paddle._C_ops.add(t2056, t2058)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t2065 = paddle._C_ops.depthwise_conv2d(
            t2033, t685, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t685

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2066, t2067, t2068, t2069, t2070, t2071 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2065,
                t686,
                t687,
                t688,
                t689,
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
        del t689, t688, t687, t686

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2072 = paddle._C_ops.add(t2064, t2066)

        # pd_op.depthwise_conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x1x5x5xf32)
        t2073 = paddle._C_ops.depthwise_conv2d(
            t2033, t690, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t690

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2074, t2075, t2076, t2077, t2078, t2079 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2073,
                t691,
                t692,
                t693,
                t694,
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
        del t692, t691, t694, t693

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2080 = paddle._C_ops.add(t2072, t2074)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t2081 = paddle._C_ops.multiply(input97, t2080)
        del input97

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t2082 = paddle._C_ops.add(t2081, input98)
        del input98

        # pd_op.hardswish: (8x384x20x20xf32) <- (8x384x20x20xf32)
        t2083 = paddle._C_ops.hardswish(t2082)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t2084 = paddle._C_ops.multiply(input99, t2083)
        del input99

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t2085 = paddle._C_ops.add(t2084, input100)
        del input100

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2086, t2087, t2088, t2089, t2090, t2091 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2085,
                t695,
                t696,
                t697,
                t698,
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
        del t698, t697, t696, t695

        # pd_op.scale: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t2092 = paddle._C_ops.scale(t2086, t807, float("0"), True)
        del t2086

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t2093 = paddle._C_ops.conv2d(
            t2085, t699, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t699

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2094, t2095, t2096, t2097, t2098, t2099 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2093,
                t700,
                t701,
                t702,
                t703,
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
        del t703, t702, t701, t700

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2100 = paddle._C_ops.add(t2092, t2094)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t2101 = paddle._C_ops.conv2d(
            t2085, t704, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t704

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2102, t2103, t2104, t2105, t2106, t2107 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2101,
                t705,
                t706,
                t707,
                t708,
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
        del t708, t707, t706, t705

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2108 = paddle._C_ops.add(t2100, t2102)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t2109 = paddle._C_ops.conv2d(
            t2085, t709, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t709

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2110, t2111, t2112, t2113, t2114, t2115 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2109,
                t710,
                t711,
                t712,
                t713,
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
        del t713, t712, t711, t710

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2116 = paddle._C_ops.add(t2108, t2110)

        # pd_op.conv2d: (8x384x20x20xf32) <- (8x384x20x20xf32, 384x384x1x1xf32)
        t2117 = paddle._C_ops.conv2d(
            t2085, t714, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t714

        # pd_op.batch_norm_: (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (8x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t2118, t2119, t2120, t2121, t2122, t2123 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2117,
                t715,
                t716,
                t717,
                t718,
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
        del t718, t717, t716, t715

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 8x384x20x20xf32)
        t2124 = paddle._C_ops.add(t2116, t2118)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t2125 = paddle._C_ops.multiply(input101, t2124)
        del input101

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t2126 = paddle._C_ops.add(t2125, input102)
        del input102

        # pd_op.hardswish: (8x384x20x20xf32) <- (8x384x20x20xf32)
        t2127 = paddle._C_ops.hardswish(t2126)

        # pd_op.multiply: (8x384x20x20xf32) <- (1xf32, 8x384x20x20xf32)
        t2128 = paddle._C_ops.multiply(input103, t2127)
        del input103

        # pd_op.add: (8x384x20x20xf32) <- (8x384x20x20xf32, 1xf32)
        t2129 = paddle._C_ops.add(t2128, input104)
        del input104

        # pd_op.conv2d: (8x12x160x160xf32) <- (8x48x160x160xf32, 12x48x1x1xf32)
        t2130 = paddle._C_ops.conv2d(
            t1093, t719, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t719

        # pd_op.reshape: (1x12x1x1xf32) <- (12xf32, 4xi64)
        t2131 = paddle._C_ops.reshape(t720, t1786)
        del t720

        # pd_op.add: (8x12x160x160xf32) <- (8x12x160x160xf32, 1x12x1x1xf32)
        t2132 = paddle._C_ops.add(t2130, t2131)

        # pd_op.conv2d: (8x18x80x80xf32) <- (8x96x80x80xf32, 18x96x1x1xf32)
        t2133 = paddle._C_ops.conv2d(
            t1268, t721, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t721

        # pd_op.reshape: (1x18x1x1xf32) <- (18xf32, 4xi64)
        t2134 = paddle._C_ops.reshape(t722, t1786)
        del t722

        # pd_op.add: (8x18x80x80xf32) <- (8x18x80x80xf32, 1x18x1x1xf32)
        t2135 = paddle._C_ops.add(t2133, t2134)

        # pd_op.conv2d: (8x42x40x40xf32) <- (8x192x40x40xf32, 42x192x1x1xf32)
        t2136 = paddle._C_ops.conv2d(
            t1731, t723, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t723

        # pd_op.reshape: (1x42x1x1xf32) <- (42xf32, 4xi64)
        t2137 = paddle._C_ops.reshape(t724, t1786)
        del t724

        # pd_op.add: (8x42x40x40xf32) <- (8x42x40x40xf32, 1x42x1x1xf32)
        t2138 = paddle._C_ops.add(t2136, t2137)

        # pd_op.conv2d: (8x360x20x20xf32) <- (8x384x20x20xf32, 360x384x1x1xf32)
        t2139 = paddle._C_ops.conv2d(
            t2129, t725, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t725

        # pd_op.reshape: (1x360x1x1xf32) <- (360xf32, 4xi64)
        t2140 = paddle._C_ops.reshape(t726, t1786)
        del t726

        # pd_op.add: (8x360x20x20xf32) <- (8x360x20x20xf32, 1x360x1x1xf32)
        t2141 = paddle._C_ops.add(t2139, t2140)

        # pd_op.conv2d: (8x96x20x20xf32) <- (8x360x20x20xf32, 96x360x1x1xf32)
        t2142 = paddle._C_ops.conv2d(
            t2141, t727, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t727

        # pd_op.pool2d: (8x96x1x1xf32) <- (8x96x20x20xf32, 2xi64)
        t2143 = paddle._C_ops.pool2d(
            t2142,
            t1774,
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

        # pd_op.conv2d: (8x24x1x1xf32) <- (8x96x1x1xf32, 24x96x1x1xf32)
        t2144 = paddle._C_ops.conv2d(
            t2143, t728, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t728

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2145 = paddle._C_ops.reshape(t729, t1786)
        del t729

        # pd_op.add: (8x24x1x1xf32) <- (8x24x1x1xf32, 1x24x1x1xf32)
        t2146 = paddle._C_ops.add(t2144, t2145)

        # pd_op.relu: (8x24x1x1xf32) <- (8x24x1x1xf32)
        t2147 = paddle._C_ops.relu(t2146)
        del t2146

        # pd_op.conv2d: (8x96x1x1xf32) <- (8x24x1x1xf32, 96x24x1x1xf32)
        t2148 = paddle._C_ops.conv2d(
            t2147, t730, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t730

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t2149 = paddle._C_ops.reshape(t731, t1786)
        del t731

        # pd_op.add: (8x96x1x1xf32) <- (8x96x1x1xf32, 1x96x1x1xf32)
        t2150 = paddle._C_ops.add(t2148, t2149)

        # pd_op.hardsigmoid: (8x96x1x1xf32) <- (8x96x1x1xf32)
        t2151 = paddle._C_ops.hardsigmoid(t2150, float("0.2"), float("0.5"))
        del t2150

        # pd_op.multiply: (8x96x20x20xf32) <- (8x96x20x20xf32, 8x96x1x1xf32)
        t2152 = paddle._C_ops.multiply(t2142, t2151)

        # pd_op.add: (8x96x20x20xf32) <- (8x96x20x20xf32, 8x96x20x20xf32)
        t2153 = paddle._C_ops.add(t2142, t2152)

        # pd_op.conv2d: (8x96x40x40xf32) <- (8x42x40x40xf32, 96x42x1x1xf32)
        t2154 = paddle._C_ops.conv2d(
            t2138, t732, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t732

        # pd_op.pool2d: (8x96x1x1xf32) <- (8x96x40x40xf32, 2xi64)
        t2155 = paddle._C_ops.pool2d(
            t2154,
            t1774,
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

        # pd_op.conv2d: (8x24x1x1xf32) <- (8x96x1x1xf32, 24x96x1x1xf32)
        t2156 = paddle._C_ops.conv2d(
            t2155, t733, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t733

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2157 = paddle._C_ops.reshape(t734, t1786)
        del t734

        # pd_op.add: (8x24x1x1xf32) <- (8x24x1x1xf32, 1x24x1x1xf32)
        t2158 = paddle._C_ops.add(t2156, t2157)

        # pd_op.relu: (8x24x1x1xf32) <- (8x24x1x1xf32)
        t2159 = paddle._C_ops.relu(t2158)
        del t2158

        # pd_op.conv2d: (8x96x1x1xf32) <- (8x24x1x1xf32, 96x24x1x1xf32)
        t2160 = paddle._C_ops.conv2d(
            t2159, t735, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t735

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t2161 = paddle._C_ops.reshape(t736, t1786)
        del t736

        # pd_op.add: (8x96x1x1xf32) <- (8x96x1x1xf32, 1x96x1x1xf32)
        t2162 = paddle._C_ops.add(t2160, t2161)

        # pd_op.hardsigmoid: (8x96x1x1xf32) <- (8x96x1x1xf32)
        t2163 = paddle._C_ops.hardsigmoid(t2162, float("0.2"), float("0.5"))
        del t2162

        # pd_op.multiply: (8x96x40x40xf32) <- (8x96x40x40xf32, 8x96x1x1xf32)
        t2164 = paddle._C_ops.multiply(t2154, t2163)

        # pd_op.add: (8x96x40x40xf32) <- (8x96x40x40xf32, 8x96x40x40xf32)
        t2165 = paddle._C_ops.add(t2154, t2164)

        # pd_op.conv2d: (8x96x80x80xf32) <- (8x18x80x80xf32, 96x18x1x1xf32)
        t2166 = paddle._C_ops.conv2d(
            t2135, t737, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t737

        # pd_op.pool2d: (8x96x1x1xf32) <- (8x96x80x80xf32, 2xi64)
        t2167 = paddle._C_ops.pool2d(
            t2166,
            t1774,
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

        # pd_op.conv2d: (8x24x1x1xf32) <- (8x96x1x1xf32, 24x96x1x1xf32)
        t2168 = paddle._C_ops.conv2d(
            t2167, t738, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t738

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2169 = paddle._C_ops.reshape(t739, t1786)
        del t739

        # pd_op.add: (8x24x1x1xf32) <- (8x24x1x1xf32, 1x24x1x1xf32)
        t2170 = paddle._C_ops.add(t2168, t2169)

        # pd_op.relu: (8x24x1x1xf32) <- (8x24x1x1xf32)
        t2171 = paddle._C_ops.relu(t2170)
        del t2170

        # pd_op.conv2d: (8x96x1x1xf32) <- (8x24x1x1xf32, 96x24x1x1xf32)
        t2172 = paddle._C_ops.conv2d(
            t2171, t740, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t740

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t2173 = paddle._C_ops.reshape(t741, t1786)
        del t741

        # pd_op.add: (8x96x1x1xf32) <- (8x96x1x1xf32, 1x96x1x1xf32)
        t2174 = paddle._C_ops.add(t2172, t2173)

        # pd_op.hardsigmoid: (8x96x1x1xf32) <- (8x96x1x1xf32)
        t2175 = paddle._C_ops.hardsigmoid(t2174, float("0.2"), float("0.5"))
        del t2174

        # pd_op.multiply: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x1x1xf32)
        t2176 = paddle._C_ops.multiply(t2166, t2175)

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t2177 = paddle._C_ops.add(t2166, t2176)

        # pd_op.conv2d: (8x96x160x160xf32) <- (8x12x160x160xf32, 96x12x1x1xf32)
        t2178 = paddle._C_ops.conv2d(
            t2132, t742, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t742

        # pd_op.pool2d: (8x96x1x1xf32) <- (8x96x160x160xf32, 2xi64)
        t2179 = paddle._C_ops.pool2d(
            t2178,
            t1774,
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

        # pd_op.conv2d: (8x24x1x1xf32) <- (8x96x1x1xf32, 24x96x1x1xf32)
        t2180 = paddle._C_ops.conv2d(
            t2179, t743, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t743

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2181 = paddle._C_ops.reshape(t744, t1786)
        del t744

        # pd_op.add: (8x24x1x1xf32) <- (8x24x1x1xf32, 1x24x1x1xf32)
        t2182 = paddle._C_ops.add(t2180, t2181)

        # pd_op.relu: (8x24x1x1xf32) <- (8x24x1x1xf32)
        t2183 = paddle._C_ops.relu(t2182)
        del t2182

        # pd_op.conv2d: (8x96x1x1xf32) <- (8x24x1x1xf32, 96x24x1x1xf32)
        t2184 = paddle._C_ops.conv2d(
            t2183, t745, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t745

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        t2185 = paddle._C_ops.reshape(t746, t1786)
        del t746

        # pd_op.add: (8x96x1x1xf32) <- (8x96x1x1xf32, 1x96x1x1xf32)
        t2186 = paddle._C_ops.add(t2184, t2185)

        # pd_op.hardsigmoid: (8x96x1x1xf32) <- (8x96x1x1xf32)
        t2187 = paddle._C_ops.hardsigmoid(t2186, float("0.2"), float("0.5"))
        del t2186

        # pd_op.multiply: (8x96x160x160xf32) <- (8x96x160x160xf32, 8x96x1x1xf32)
        t2188 = paddle._C_ops.multiply(t2178, t2187)

        # pd_op.add: (8x96x160x160xf32) <- (8x96x160x160xf32, 8x96x160x160xf32)
        t2189 = paddle._C_ops.add(t2178, t2188)

        # pd_op.nearest_interp: (8x96x40x40xf32) <- (8x96x20x20xf32, None, None, None)
        t2190 = paddle._C_ops.nearest_interp(
            t2153,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            1,
        )

        # pd_op.add: (8x96x40x40xf32) <- (8x96x40x40xf32, 8x96x40x40xf32)
        t2191 = paddle._C_ops.add(t2165, t2190)

        # pd_op.nearest_interp: (8x96x80x80xf32) <- (8x96x40x40xf32, None, None, None)
        t2192 = paddle._C_ops.nearest_interp(
            t2191,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            1,
        )

        # pd_op.add: (8x96x80x80xf32) <- (8x96x80x80xf32, 8x96x80x80xf32)
        t2193 = paddle._C_ops.add(t2177, t2192)

        # pd_op.nearest_interp: (8x96x160x160xf32) <- (8x96x80x80xf32, None, None, None)
        t2194 = paddle._C_ops.nearest_interp(
            t2193,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            1,
        )

        # pd_op.add: (8x96x160x160xf32) <- (8x96x160x160xf32, 8x96x160x160xf32)
        t2195 = paddle._C_ops.add(t2189, t2194)

        # pd_op.conv2d: (8x24x20x20xf32) <- (8x96x20x20xf32, 24x96x3x3xf32)
        t2196 = paddle._C_ops.conv2d(
            t2153, t747, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t747

        # pd_op.pool2d: (8x24x1x1xf32) <- (8x24x20x20xf32, 2xi64)
        t2197 = paddle._C_ops.pool2d(
            t2196,
            t1774,
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

        # pd_op.conv2d: (8x6x1x1xf32) <- (8x24x1x1xf32, 6x24x1x1xf32)
        t2198 = paddle._C_ops.conv2d(
            t2197, t748, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t748

        # pd_op.reshape: (1x6x1x1xf32) <- (6xf32, 4xi64)
        t2199 = paddle._C_ops.reshape(t749, t1786)
        del t749

        # pd_op.add: (8x6x1x1xf32) <- (8x6x1x1xf32, 1x6x1x1xf32)
        t2200 = paddle._C_ops.add(t2198, t2199)

        # pd_op.relu: (8x6x1x1xf32) <- (8x6x1x1xf32)
        t2201 = paddle._C_ops.relu(t2200)
        del t2200

        # pd_op.conv2d: (8x24x1x1xf32) <- (8x6x1x1xf32, 24x6x1x1xf32)
        t2202 = paddle._C_ops.conv2d(
            t2201, t750, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t750

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2203 = paddle._C_ops.reshape(t751, t1786)
        del t751

        # pd_op.add: (8x24x1x1xf32) <- (8x24x1x1xf32, 1x24x1x1xf32)
        t2204 = paddle._C_ops.add(t2202, t2203)

        # pd_op.hardsigmoid: (8x24x1x1xf32) <- (8x24x1x1xf32)
        t2205 = paddle._C_ops.hardsigmoid(t2204, float("0.2"), float("0.5"))
        del t2204

        # pd_op.multiply: (8x24x20x20xf32) <- (8x24x20x20xf32, 8x24x1x1xf32)
        t2206 = paddle._C_ops.multiply(t2196, t2205)

        # pd_op.add: (8x24x20x20xf32) <- (8x24x20x20xf32, 8x24x20x20xf32)
        t2207 = paddle._C_ops.add(t2196, t2206)

        # pd_op.conv2d: (8x24x40x40xf32) <- (8x96x40x40xf32, 24x96x3x3xf32)
        t2208 = paddle._C_ops.conv2d(
            t2191, t752, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t752

        # pd_op.pool2d: (8x24x1x1xf32) <- (8x24x40x40xf32, 2xi64)
        t2209 = paddle._C_ops.pool2d(
            t2208,
            t1774,
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

        # pd_op.conv2d: (8x6x1x1xf32) <- (8x24x1x1xf32, 6x24x1x1xf32)
        t2210 = paddle._C_ops.conv2d(
            t2209, t753, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t753

        # pd_op.reshape: (1x6x1x1xf32) <- (6xf32, 4xi64)
        t2211 = paddle._C_ops.reshape(t754, t1786)
        del t754

        # pd_op.add: (8x6x1x1xf32) <- (8x6x1x1xf32, 1x6x1x1xf32)
        t2212 = paddle._C_ops.add(t2210, t2211)

        # pd_op.relu: (8x6x1x1xf32) <- (8x6x1x1xf32)
        t2213 = paddle._C_ops.relu(t2212)
        del t2212

        # pd_op.conv2d: (8x24x1x1xf32) <- (8x6x1x1xf32, 24x6x1x1xf32)
        t2214 = paddle._C_ops.conv2d(
            t2213, t755, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t755

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2215 = paddle._C_ops.reshape(t756, t1786)
        del t756

        # pd_op.add: (8x24x1x1xf32) <- (8x24x1x1xf32, 1x24x1x1xf32)
        t2216 = paddle._C_ops.add(t2214, t2215)

        # pd_op.hardsigmoid: (8x24x1x1xf32) <- (8x24x1x1xf32)
        t2217 = paddle._C_ops.hardsigmoid(t2216, float("0.2"), float("0.5"))
        del t2216

        # pd_op.multiply: (8x24x40x40xf32) <- (8x24x40x40xf32, 8x24x1x1xf32)
        t2218 = paddle._C_ops.multiply(t2208, t2217)

        # pd_op.add: (8x24x40x40xf32) <- (8x24x40x40xf32, 8x24x40x40xf32)
        t2219 = paddle._C_ops.add(t2208, t2218)

        # pd_op.conv2d: (8x24x80x80xf32) <- (8x96x80x80xf32, 24x96x3x3xf32)
        t2220 = paddle._C_ops.conv2d(
            t2193, t757, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t757

        # pd_op.pool2d: (8x24x1x1xf32) <- (8x24x80x80xf32, 2xi64)
        t2221 = paddle._C_ops.pool2d(
            t2220,
            t1774,
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

        # pd_op.conv2d: (8x6x1x1xf32) <- (8x24x1x1xf32, 6x24x1x1xf32)
        t2222 = paddle._C_ops.conv2d(
            t2221, t758, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t758

        # pd_op.reshape: (1x6x1x1xf32) <- (6xf32, 4xi64)
        t2223 = paddle._C_ops.reshape(t759, t1786)
        del t759

        # pd_op.add: (8x6x1x1xf32) <- (8x6x1x1xf32, 1x6x1x1xf32)
        t2224 = paddle._C_ops.add(t2222, t2223)

        # pd_op.relu: (8x6x1x1xf32) <- (8x6x1x1xf32)
        t2225 = paddle._C_ops.relu(t2224)
        del t2224

        # pd_op.conv2d: (8x24x1x1xf32) <- (8x6x1x1xf32, 24x6x1x1xf32)
        t2226 = paddle._C_ops.conv2d(
            t2225, t760, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t760

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2227 = paddle._C_ops.reshape(t761, t1786)
        del t761

        # pd_op.add: (8x24x1x1xf32) <- (8x24x1x1xf32, 1x24x1x1xf32)
        t2228 = paddle._C_ops.add(t2226, t2227)

        # pd_op.hardsigmoid: (8x24x1x1xf32) <- (8x24x1x1xf32)
        t2229 = paddle._C_ops.hardsigmoid(t2228, float("0.2"), float("0.5"))
        del t2228

        # pd_op.multiply: (8x24x80x80xf32) <- (8x24x80x80xf32, 8x24x1x1xf32)
        t2230 = paddle._C_ops.multiply(t2220, t2229)

        # pd_op.add: (8x24x80x80xf32) <- (8x24x80x80xf32, 8x24x80x80xf32)
        t2231 = paddle._C_ops.add(t2220, t2230)

        # pd_op.conv2d: (8x24x160x160xf32) <- (8x96x160x160xf32, 24x96x3x3xf32)
        t2232 = paddle._C_ops.conv2d(
            t2195, t762, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t762

        # pd_op.pool2d: (8x24x1x1xf32) <- (8x24x160x160xf32, 2xi64)
        t2233 = paddle._C_ops.pool2d(
            t2232,
            t1774,
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

        # pd_op.conv2d: (8x6x1x1xf32) <- (8x24x1x1xf32, 6x24x1x1xf32)
        t2234 = paddle._C_ops.conv2d(
            t2233, t763, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t763

        # pd_op.reshape: (1x6x1x1xf32) <- (6xf32, 4xi64)
        t2235 = paddle._C_ops.reshape(t764, t1786)
        del t764

        # pd_op.add: (8x6x1x1xf32) <- (8x6x1x1xf32, 1x6x1x1xf32)
        t2236 = paddle._C_ops.add(t2234, t2235)

        # pd_op.relu: (8x6x1x1xf32) <- (8x6x1x1xf32)
        t2237 = paddle._C_ops.relu(t2236)
        del t2236

        # pd_op.conv2d: (8x24x1x1xf32) <- (8x6x1x1xf32, 24x6x1x1xf32)
        t2238 = paddle._C_ops.conv2d(
            t2237, t765, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t765

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2239 = paddle._C_ops.reshape(t766, t1786)
        del t1786, t766

        # pd_op.add: (8x24x1x1xf32) <- (8x24x1x1xf32, 1x24x1x1xf32)
        t2240 = paddle._C_ops.add(t2238, t2239)

        # pd_op.hardsigmoid: (8x24x1x1xf32) <- (8x24x1x1xf32)
        t2241 = paddle._C_ops.hardsigmoid(t2240, float("0.2"), float("0.5"))
        del t2240

        # pd_op.multiply: (8x24x160x160xf32) <- (8x24x160x160xf32, 8x24x1x1xf32)
        t2242 = paddle._C_ops.multiply(t2232, t2241)

        # pd_op.add: (8x24x160x160xf32) <- (8x24x160x160xf32, 8x24x160x160xf32)
        t2243 = paddle._C_ops.add(t2232, t2242)

        # pd_op.nearest_interp: (8x24x160x160xf32) <- (8x24x20x20xf32, None, None, None)
        t2244 = paddle._C_ops.nearest_interp(
            t2207,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("8"), float("8")],
            "nearest",
            False,
            1,
        )

        # pd_op.nearest_interp: (8x24x160x160xf32) <- (8x24x40x40xf32, None, None, None)
        t2245 = paddle._C_ops.nearest_interp(
            t2219,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("4"), float("4")],
            "nearest",
            False,
            1,
        )

        # pd_op.nearest_interp: (8x24x160x160xf32) <- (8x24x80x80xf32, None, None, None)
        t2246 = paddle._C_ops.nearest_interp(
            t2231,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            1,
        )

        # pd_op.full: (1xi32) <- ()
        t2247 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        t2248 = t2247

        # builtin.combine: ([8x24x160x160xf32, 8x24x160x160xf32, 8x24x160x160xf32, 8x24x160x160xf32]) <- (8x24x160x160xf32, 8x24x160x160xf32, 8x24x160x160xf32, 8x24x160x160xf32)
        t2249 = [t2244, t2245, t2246, t2243]

        # pd_op.concat: (8x96x160x160xf32) <- ([8x24x160x160xf32, 8x24x160x160xf32, 8x24x160x160xf32, 8x24x160x160xf32], 1xi32)
        t2250 = paddle._C_ops.concat(t2249, t2247)
        del t2249

        # pd_op.conv2d: (8x24x160x160xf32) <- (8x96x160x160xf32, 24x96x3x3xf32)
        t2251 = paddle._C_ops.conv2d(
            t2250, t767, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t767

        # pd_op.batch_norm_: (8x24x160x160xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x160x160xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t2252, t2253, t2254, t2255, t2256, t2257 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2251,
                t768,
                t769,
                t770,
                t771,
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
        del t771, t770, t769, t768

        # pd_op.relu: (8x24x160x160xf32) <- (8x24x160x160xf32)
        t2258 = paddle._C_ops.relu(t2252)
        del t2252

        # pd_op.isnan: (8x24x160x160xb) <- (8x24x160x160xf32)
        t2259 = paddle._C_ops.isnan(t2258)

        # pd_op.full: (1xf32) <- ()
        t2260 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (8x24x160x160xf32) <- (8x24x160x160xf32, 1xf32)
        t2261 = paddle._C_ops.full_like(
            t2258, t2260, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (8x24x160x160xf32) <- (8x24x160x160xb, 8x24x160x160xf32, 8x24x160x160xf32)
        t2262 = paddle._C_ops.where(t2259, t2261, t2258)

        # pd_op.full_int_array: (0xi64) <- ()
        t2263 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        t2264 = t2263

        # pd_op.assign: (0xi64) <- (0xi64)
        t2265 = t2263

        # pd_op.assign: (0xi64) <- (0xi64)
        t2266 = t2263

        # pd_op.conv2d_transpose: (8x24x320x320xf32) <- (8x24x160x160xf32, 24x24x2x2xf32, 0xi64)
        t2267 = paddle._C_ops.conv2d_transpose(
            t2262, t772, [2, 2], [0, 0], [], t2263, "EXPLICIT", 1, [1, 1], "NCHW"
        )
        del t772

        # pd_op.full_int_array: (4xi64) <- ()
        t2268 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2269 = paddle._C_ops.reshape(t773, t2268)
        del t773

        # pd_op.add: (8x24x320x320xf32) <- (8x24x320x320xf32, 1x24x1x1xf32)
        t2270 = paddle._C_ops.add(t2267, t2269)

        # pd_op.batch_norm_: (8x24x320x320xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x320x320xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t2271, t2272, t2273, t2274, t2275, t2276 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2270,
                t774,
                t775,
                t776,
                t777,
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
        del t777, t776, t775, t774

        # pd_op.relu: (8x24x320x320xf32) <- (8x24x320x320xf32)
        t2277 = paddle._C_ops.relu(t2271)
        del t2271

        # pd_op.isnan: (8x24x320x320xb) <- (8x24x320x320xf32)
        t2278 = paddle._C_ops.isnan(t2277)

        # pd_op.full_like: (8x24x320x320xf32) <- (8x24x320x320xf32, 1xf32)
        t2279 = paddle._C_ops.full_like(
            t2277, t2260, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (8x24x320x320xf32) <- (8x24x320x320xb, 8x24x320x320xf32, 8x24x320x320xf32)
        t2280 = paddle._C_ops.where(t2278, t2279, t2277)

        # pd_op.conv2d_transpose: (8x1x640x640xf32) <- (8x24x320x320xf32, 24x1x2x2xf32, 0xi64)
        t2281 = paddle._C_ops.conv2d_transpose(
            t2280, t778, [2, 2], [0, 0], [], t2263, "EXPLICIT", 1, [1, 1], "NCHW"
        )
        del t778

        # pd_op.full_int_array: (4xi64) <- ()
        t2282 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        t2283 = paddle._C_ops.reshape(t779, t2282)
        del t779

        # pd_op.add: (8x1x640x640xf32) <- (8x1x640x640xf32, 1x1x1x1xf32)
        t2284 = paddle._C_ops.add(t2281, t2283)

        # pd_op.sigmoid: (8x1x640x640xf32) <- (8x1x640x640xf32)
        t2285 = paddle._C_ops.sigmoid(t2284)
        del t2284

        # pd_op.conv2d: (8x24x160x160xf32) <- (8x96x160x160xf32, 24x96x3x3xf32)
        t2286 = paddle._C_ops.conv2d(
            t2250, t780, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t780

        # pd_op.batch_norm_: (8x24x160x160xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x160x160xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t2287, t2288, t2289, t2290, t2291, t2292 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2286,
                t781,
                t782,
                t783,
                t784,
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
        del t782, t781, t784, t783

        # pd_op.relu: (8x24x160x160xf32) <- (8x24x160x160xf32)
        t2293 = paddle._C_ops.relu(t2287)
        del t2287

        # pd_op.isnan: (8x24x160x160xb) <- (8x24x160x160xf32)
        t2294 = paddle._C_ops.isnan(t2293)

        # pd_op.full_like: (8x24x160x160xf32) <- (8x24x160x160xf32, 1xf32)
        t2295 = paddle._C_ops.full_like(
            t2293, t2260, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (8x24x160x160xf32) <- (8x24x160x160xb, 8x24x160x160xf32, 8x24x160x160xf32)
        t2296 = paddle._C_ops.where(t2294, t2295, t2293)

        # pd_op.conv2d_transpose: (8x24x320x320xf32) <- (8x24x160x160xf32, 24x24x2x2xf32, 0xi64)
        t2297 = paddle._C_ops.conv2d_transpose(
            t2296, t785, [2, 2], [0, 0], [], t2263, "EXPLICIT", 1, [1, 1], "NCHW"
        )
        del t785

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        t2298 = paddle._C_ops.reshape(t786, t2268)
        del t2268, t786

        # pd_op.add: (8x24x320x320xf32) <- (8x24x320x320xf32, 1x24x1x1xf32)
        t2299 = paddle._C_ops.add(t2297, t2298)

        # pd_op.batch_norm_: (8x24x320x320xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (8x24x320x320xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t2300, t2301, t2302, t2303, t2304, t2305 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2299,
                t787,
                t788,
                t789,
                t790,
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
        del t790, t789, t788, t787

        # pd_op.relu: (8x24x320x320xf32) <- (8x24x320x320xf32)
        t2306 = paddle._C_ops.relu(t2300)
        del t2300

        # pd_op.isnan: (8x24x320x320xb) <- (8x24x320x320xf32)
        t2307 = paddle._C_ops.isnan(t2306)

        # pd_op.full_like: (8x24x320x320xf32) <- (8x24x320x320xf32, 1xf32)
        t2308 = paddle._C_ops.full_like(
            t2306, t2260, paddle.float32, paddle.framework._current_expected_place()
        )
        del t2260

        # pd_op.where: (8x24x320x320xf32) <- (8x24x320x320xb, 8x24x320x320xf32, 8x24x320x320xf32)
        t2309 = paddle._C_ops.where(t2307, t2308, t2306)

        # pd_op.conv2d_transpose: (8x1x640x640xf32) <- (8x24x320x320xf32, 24x1x2x2xf32, 0xi64)
        t2310 = paddle._C_ops.conv2d_transpose(
            t2309, t791, [2, 2], [0, 0], [], t2263, "EXPLICIT", 1, [1, 1], "NCHW"
        )
        del t791

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        t2311 = paddle._C_ops.reshape(t792, t2282)
        del t2282, t792

        # pd_op.add: (8x1x640x640xf32) <- (8x1x640x640xf32, 1x1x1x1xf32)
        t2312 = paddle._C_ops.add(t2310, t2311)

        # pd_op.sigmoid: (8x1x640x640xf32) <- (8x1x640x640xf32)
        t2313 = paddle._C_ops.sigmoid(t2312)
        del t2312

        # pd_op.subtract: (8x1x640x640xf32) <- (8x1x640x640xf32, 8x1x640x640xf32)
        t2314 = paddle._C_ops.subtract(t2285, t2313)

        # pd_op.full: (1xf32) <- ()
        t2315 = paddle._C_ops.full(
            [1], float("-50"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x1x640x640xf32) <- (8x1x640x640xf32, 1xf32)
        t2316 = paddle._C_ops.scale(t2314, t2315, float("0"), True)
        del t2314

        # pd_op.exp: (8x1x640x640xf32) <- (8x1x640x640xf32)
        t2317 = paddle._C_ops.exp(t2316)
        del t2316

        # pd_op.scale: (8x1x640x640xf32) <- (8x1x640x640xf32, 1xf32)
        t2318 = paddle._C_ops.scale(t2317, t807, float("1"), True)

        # pd_op.reciprocal: (8x1x640x640xf32) <- (8x1x640x640xf32)
        t2319 = paddle._C_ops.reciprocal(t2318)
        del t2318

        # builtin.combine: ([8x1x640x640xf32, 8x1x640x640xf32, 8x1x640x640xf32]) <- (8x1x640x640xf32, 8x1x640x640xf32, 8x1x640x640xf32)
        t2320 = [t2285, t2313, t2319]

        return (
            t794,
            t795,
            t796,
            t797,
            t798,
            t799,
            t800,
            t802,
            t803,
            t804,
            t805,
            t806,
            t807,
            t808,
            t809,
            t810,
            t811,
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
            t860,
            t861,
            t862,
            t863,
            t864,
            t865,
            t866,
            t867,
            t868,
            t869,
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
            t884,
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
            t921,
            t922,
            t923,
            t924,
            t925,
            t926,
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
            t943,
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
            t960,
            t961,
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
            t999,
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
            t1036,
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
            t1063,
            t1064,
            t1065,
            t1066,
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
            t1092,
            t1093,
            t1094,
            t1096,
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
            t1121,
            t1122,
            t1123,
            t1124,
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
            t1174,
            t1175,
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
            t1190,
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
            t1205,
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
            t1220,
            t1221,
            t1222,
            t1223,
            t1224,
            t1226,
            t1227,
            t1228,
            t1229,
            t1230,
            t1231,
            t1232,
            t1233,
            t1234,
            t1235,
            t1236,
            t1237,
            t1238,
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
            t1254,
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
            t1269,
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
            t1284,
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
            t1299,
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
            t1313,
            t1314,
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
            t1329,
            t1330,
            t1331,
            t1332,
            t1333,
            t1334,
            t1335,
            t1336,
            t1337,
            t1338,
            t1339,
            t1340,
            t1341,
            t1342,
            t1343,
            t1344,
            t1345,
            t1346,
            t1347,
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
            t1363,
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
            t1378,
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
            t1393,
            t1394,
            t1395,
            t1396,
            t1397,
            t1398,
            t1399,
            t1401,
            t1402,
            t1403,
            t1404,
            t1405,
            t1406,
            t1407,
            t1408,
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
            t1423,
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
            t1438,
            t1439,
            t1440,
            t1441,
            t1442,
            t1443,
            t1445,
            t1446,
            t1447,
            t1448,
            t1449,
            t1450,
            t1451,
            t1452,
            t1453,
            t1454,
            t1455,
            t1456,
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
            t1479,
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
            t1494,
            t1495,
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
            t1509,
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
            t1535,
            t1536,
            t1537,
            t1538,
            t1539,
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
            t1554,
            t1555,
            t1556,
            t1557,
            t1558,
            t1559,
            t1560,
            t1561,
            t1562,
            t1563,
            t1564,
            t1565,
            t1566,
            t1567,
            t1568,
            t1569,
            t1570,
            t1571,
            t1572,
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
            t1587,
            t1588,
            t1589,
            t1590,
            t1591,
            t1593,
            t1594,
            t1595,
            t1596,
            t1597,
            t1598,
            t1599,
            t1600,
            t1601,
            t1602,
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
            t1617,
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
            t1632,
            t1633,
            t1634,
            t1635,
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
            t1647,
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
            t1662,
            t1663,
            t1664,
            t1665,
            t1666,
            t1667,
            t1668,
            t1669,
            t1670,
            t1671,
            t1672,
            t1673,
            t1674,
            t1675,
            t1676,
            t1677,
            t1678,
            t1679,
            t1680,
            t1681,
            t1682,
            t1683,
            t1684,
            t1685,
            t1686,
            t1687,
            t1689,
            t1690,
            t1691,
            t1692,
            t1693,
            t1694,
            t1695,
            t1696,
            t1697,
            t1698,
            t1699,
            t1700,
            t1701,
            t1702,
            t1703,
            t1704,
            t1705,
            t1706,
            t1707,
            t1708,
            t1709,
            t1710,
            t1711,
            t1712,
            t1713,
            t1714,
            t1715,
            t1716,
            t1717,
            t1718,
            t1719,
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
            t1732,
            t1734,
            t1735,
            t1736,
            t1737,
            t1738,
            t1739,
            t1740,
            t1741,
            t1742,
            t1743,
            t1744,
            t1745,
            t1746,
            t1747,
            t1748,
            t1749,
            t1750,
            t1751,
            t1752,
            t1753,
            t1754,
            t1755,
            t1756,
            t1757,
            t1758,
            t1759,
            t1760,
            t1761,
            t1762,
            t1763,
            t1764,
            t1765,
            t1766,
            t1767,
            t1768,
            t1769,
            t1770,
            t1771,
            t1772,
            t1773,
            t1774,
            t1775,
            t1776,
            t1777,
            t1778,
            t1779,
            t1780,
            t1781,
            t1782,
            t1783,
            t1784,
            t1785,
            t1787,
            t1789,
            t1790,
            t1791,
            t1793,
            t1794,
            t1795,
            t1797,
            t1798,
            t1799,
            t1800,
            t1801,
            t1802,
            t1803,
            t1804,
            t1805,
            t1806,
            t1807,
            t1808,
            t1809,
            t1810,
            t1811,
            t1812,
            t1813,
            t1814,
            t1815,
            t1816,
            t1817,
            t1818,
            t1819,
            t1820,
            t1821,
            t1822,
            t1823,
            t1824,
            t1825,
            t1826,
            t1827,
            t1828,
            t1829,
            t1830,
            t1831,
            t1833,
            t1834,
            t1835,
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
            t1848,
            t1849,
            t1850,
            t1851,
            t1852,
            t1853,
            t1854,
            t1855,
            t1856,
            t1857,
            t1858,
            t1859,
            t1860,
            t1861,
            t1862,
            t1863,
            t1864,
            t1865,
            t1866,
            t1867,
            t1868,
            t1869,
            t1870,
            t1871,
            t1872,
            t1873,
            t1874,
            t1875,
            t1876,
            t1877,
            t1878,
            t1879,
            t1880,
            t1881,
            t1882,
            t1883,
            t1884,
            t1885,
            t1886,
            t1888,
            t1889,
            t1890,
            t1892,
            t1893,
            t1895,
            t1896,
            t1897,
            t1898,
            t1899,
            t1900,
            t1901,
            t1902,
            t1903,
            t1904,
            t1905,
            t1906,
            t1907,
            t1908,
            t1909,
            t1910,
            t1911,
            t1912,
            t1913,
            t1914,
            t1915,
            t1916,
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
            t1929,
            t1930,
            t1931,
            t1932,
            t1933,
            t1934,
            t1935,
            t1936,
            t1937,
            t1939,
            t1940,
            t1941,
            t1942,
            t1943,
            t1944,
            t1945,
            t1946,
            t1947,
            t1948,
            t1949,
            t1950,
            t1951,
            t1952,
            t1953,
            t1954,
            t1955,
            t1956,
            t1957,
            t1958,
            t1959,
            t1960,
            t1961,
            t1962,
            t1963,
            t1964,
            t1965,
            t1966,
            t1967,
            t1968,
            t1969,
            t1970,
            t1971,
            t1972,
            t1973,
            t1974,
            t1975,
            t1976,
            t1977,
            t1978,
            t1979,
            t1980,
            t1981,
            t1982,
            t1983,
            t1984,
            t1985,
            t1986,
            t1987,
            t1988,
            t1989,
            t1991,
            t1992,
            t1993,
            t1994,
            t1995,
            t1996,
            t1997,
            t1998,
            t1999,
            t2000,
            t2001,
            t2002,
            t2003,
            t2004,
            t2005,
            t2006,
            t2007,
            t2008,
            t2009,
            t2010,
            t2011,
            t2012,
            t2013,
            t2014,
            t2015,
            t2016,
            t2017,
            t2018,
            t2019,
            t2020,
            t2021,
            t2022,
            t2023,
            t2024,
            t2025,
            t2026,
            t2027,
            t2028,
            t2029,
            t2030,
            t2031,
            t2032,
            t2033,
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
            t2045,
            t2046,
            t2047,
            t2048,
            t2049,
            t2050,
            t2051,
            t2052,
            t2053,
            t2054,
            t2055,
            t2056,
            t2057,
            t2058,
            t2059,
            t2060,
            t2061,
            t2062,
            t2063,
            t2064,
            t2065,
            t2066,
            t2067,
            t2068,
            t2069,
            t2070,
            t2071,
            t2072,
            t2073,
            t2074,
            t2075,
            t2076,
            t2077,
            t2078,
            t2079,
            t2080,
            t2081,
            t2082,
            t2083,
            t2084,
            t2085,
            t2087,
            t2088,
            t2089,
            t2090,
            t2091,
            t2092,
            t2093,
            t2094,
            t2095,
            t2096,
            t2097,
            t2098,
            t2099,
            t2100,
            t2101,
            t2102,
            t2103,
            t2104,
            t2105,
            t2106,
            t2107,
            t2108,
            t2109,
            t2110,
            t2111,
            t2112,
            t2113,
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
            t2126,
            t2127,
            t2128,
            t2129,
            t2130,
            t2131,
            t2132,
            t2133,
            t2134,
            t2135,
            t2136,
            t2137,
            t2138,
            t2139,
            t2140,
            t2141,
            t2142,
            t2143,
            t2144,
            t2145,
            t2147,
            t2148,
            t2149,
            t2151,
            t2152,
            t2153,
            t2154,
            t2155,
            t2156,
            t2157,
            t2159,
            t2160,
            t2161,
            t2163,
            t2164,
            t2165,
            t2166,
            t2167,
            t2168,
            t2169,
            t2171,
            t2172,
            t2173,
            t2175,
            t2176,
            t2177,
            t2178,
            t2179,
            t2180,
            t2181,
            t2183,
            t2184,
            t2185,
            t2187,
            t2188,
            t2189,
            t2190,
            t2191,
            t2192,
            t2193,
            t2194,
            t2195,
            t2196,
            t2197,
            t2198,
            t2199,
            t2201,
            t2202,
            t2203,
            t2205,
            t2206,
            t2207,
            t2208,
            t2209,
            t2210,
            t2211,
            t2213,
            t2214,
            t2215,
            t2217,
            t2218,
            t2219,
            t2220,
            t2221,
            t2222,
            t2223,
            t2225,
            t2226,
            t2227,
            t2229,
            t2230,
            t2231,
            t2232,
            t2233,
            t2234,
            t2235,
            t2237,
            t2238,
            t2239,
            t2241,
            t2242,
            t2243,
            t2244,
            t2245,
            t2246,
            t2247,
            t2248,
            t2250,
            t2251,
            t2253,
            t2254,
            t2255,
            t2256,
            t2257,
            t2258,
            t2259,
            t2261,
            t2262,
            t2263,
            t2264,
            t2265,
            t2266,
            t2267,
            t2269,
            t2270,
            t2272,
            t2273,
            t2274,
            t2275,
            t2276,
            t2277,
            t2278,
            t2279,
            t2280,
            t2281,
            t2283,
            t2285,
            t2286,
            t2288,
            t2289,
            t2290,
            t2291,
            t2292,
            t2293,
            t2294,
            t2295,
            t2296,
            t2297,
            t2298,
            t2299,
            t2301,
            t2302,
            t2303,
            t2304,
            t2305,
            t2306,
            t2307,
            t2308,
            t2309,
            t2310,
            t2311,
            t2313,
            t2315,
            t2317,
            t2319,
            t2320,
        )
