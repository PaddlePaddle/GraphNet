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
        t793: paddle.Tensor,
        t794: paddle.Tensor,
        t795: paddle.Tensor,
        t796: paddle.Tensor,
        t797: paddle.Tensor,
        t798: paddle.Tensor,
        t799: paddle.Tensor,
        t800: paddle.Tensor,
        t801: paddle.Tensor,
        t802: paddle.Tensor,
        t803: paddle.Tensor,
        t804: paddle.Tensor,
        t805: paddle.Tensor,
        t806: paddle.Tensor,
        t807: paddle.Tensor,
        t808: paddle.Tensor,
        t809: paddle.Tensor,
        t810: paddle.Tensor,
        t811: paddle.Tensor,
        t812: paddle.Tensor,
        t813: paddle.Tensor,
        t814: paddle.Tensor,
        t815: paddle.Tensor,
        t816: paddle.Tensor,
        t817: paddle.Tensor,
        t818: paddle.Tensor,
        t819: paddle.Tensor,
        t820: paddle.Tensor,
        t821: paddle.Tensor,
        t822: paddle.Tensor,
        t823: paddle.Tensor,
        t824: paddle.Tensor,
        t825: paddle.Tensor,
        t826: paddle.Tensor,
        t827: paddle.Tensor,
        t828: paddle.Tensor,
        t829: paddle.Tensor,
        t830: paddle.Tensor,
        t831: paddle.Tensor,
        t832: paddle.Tensor,
        t833: paddle.Tensor,
        t834: paddle.Tensor,
        t835: paddle.Tensor,
        t836: paddle.Tensor,
        t837: paddle.Tensor,
        t838: paddle.Tensor,
        t839: paddle.Tensor,
        t840: paddle.Tensor,
        t841: paddle.Tensor,
        t842: paddle.Tensor,
        t843: paddle.Tensor,
        t844: paddle.Tensor,
        t845: paddle.Tensor,
        t846: paddle.Tensor,
        t847: paddle.Tensor,
        t848: paddle.Tensor,
        t849: paddle.Tensor,
        t850: paddle.Tensor,
        t851: paddle.Tensor,
        t852: paddle.Tensor,
        t853: paddle.Tensor,
        t854: paddle.Tensor,
        t855: paddle.Tensor,
        t856: paddle.Tensor,
        t857: paddle.Tensor,
        t858: paddle.Tensor,
        t859: paddle.Tensor,
        t860: paddle.Tensor,
        t861: paddle.Tensor,
        t862: paddle.Tensor,
        t863: paddle.Tensor,
        t864: paddle.Tensor,
        t865: paddle.Tensor,
        t866: paddle.Tensor,
        t867: paddle.Tensor,
        t868: paddle.Tensor,
        t869: paddle.Tensor,
        t870: paddle.Tensor,
        t871: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x48x112x112xf32) <- (-1x3x224x224xf32, 48x3x3x3xf32)
        t872 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t873, t874, t875, t876, t877, t878 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t872,
                t1,
                t2,
                t3,
                t4,
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
        del t872, t4, t3, t2, t1

        # pd_op.relu: (-1x48x112x112xf32) <- (-1x48x112x112xf32)
        t879 = paddle._C_ops.relu(t873)
        del t873

        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x48x112x112xf32, 24x48x2x2xf32)
        t880 = paddle._C_ops.conv2d(t879, t5, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW")
        del t5

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        t881, t882, t883, t884, t885, t886 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t880,
                t6,
                t7,
                t8,
                t9,
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
        del t880, t9, t8, t7, t6

        # pd_op.relu: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        t887 = paddle._C_ops.relu(t881)
        del t881

        # pd_op.conv2d: (-1x48x112x112xf32) <- (-1x24x112x112xf32, 48x24x2x2xf32)
        t888 = paddle._C_ops.conv2d(
            t887, t10, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del t10, t887

        # pd_op.batch_norm_: (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t889, t890, t891, t892, t893, t894 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t888,
                t11,
                t12,
                t13,
                t14,
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
        del t888, t14, t13, t12, t11

        # pd_op.relu: (-1x48x112x112xf32) <- (-1x48x112x112xf32)
        t895 = paddle._C_ops.relu(t889)
        del t889

        # pd_op.full_int_array: (2xi64) <- ()
        t896 = [2, 2]

        # pd_op.pool2d: (-1x48x112x112xf32) <- (-1x48x112x112xf32, 2xi64)
        t897 = paddle._C_ops.pool2d(
            t879, t896, [1, 1], [0, 0], True, True, "NCHW", "max", False, False, "SAME"
        )
        del t896, t879

        # pd_op.full: (1xi32) <- ()
        t898 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x48x112x112xf32, -1x48x112x112xf32]) <- (-1x48x112x112xf32, -1x48x112x112xf32)
        t899 = [t897, t895]
        del t897, t895

        # pd_op.concat: (-1x96x112x112xf32) <- ([-1x48x112x112xf32, -1x48x112x112xf32], 1xi32)
        t900 = paddle._C_ops.concat(t899, t898)
        del t899

        # pd_op.conv2d: (-1x48x56x56xf32) <- (-1x96x112x112xf32, 48x96x3x3xf32)
        t901 = paddle._C_ops.conv2d(
            t900, t15, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t900, t15

        # pd_op.batch_norm_: (-1x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        t902, t903, t904, t905, t906, t907 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t901,
                t16,
                t17,
                t18,
                t19,
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
        del t901, t19, t18, t17, t16

        # pd_op.relu: (-1x48x56x56xf32) <- (-1x48x56x56xf32)
        t908 = paddle._C_ops.relu(t902)
        del t902

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x48x56x56xf32, 96x48x1x1xf32)
        t909 = paddle._C_ops.conv2d(
            t908, t20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20, t908

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t910, t911, t912, t913, t914, t915 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t909,
                t21,
                t22,
                t23,
                t24,
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
        del t909, t24, t23, t22, t21

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t916 = paddle._C_ops.relu(t910)
        del t910

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t917 = paddle._C_ops.conv2d(
            t916, t25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t918, t919, t920, t921, t922, t923 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t917,
                t26,
                t27,
                t28,
                t29,
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
        del t917, t29, t28, t27, t26

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t924 = paddle._C_ops.relu(t918)
        del t918

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t925 = paddle._C_ops.conv2d(
            t924, t30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t926, t927, t928, t929, t930, t931 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t925,
                t31,
                t32,
                t33,
                t34,
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
        del t925, t34, t33, t32, t31

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t932 = paddle._C_ops.relu(t926)
        del t926

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t933 = paddle._C_ops.conv2d(
            t932, t35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t934, t935, t936, t937, t938, t939 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t933,
                t36,
                t37,
                t38,
                t39,
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
        del t933, t39, t38, t37, t36

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t940 = paddle._C_ops.relu(t934)
        del t934

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t941 = paddle._C_ops.conv2d(
            t940, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t942, t943, t944, t945, t946, t947 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t941,
                t41,
                t42,
                t43,
                t44,
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
        del t941, t44, t43, t42, t41

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t948 = paddle._C_ops.relu(t942)
        del t942

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t949 = paddle._C_ops.conv2d(
            t948, t45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t950, t951, t952, t953, t954, t955 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t949,
                t46,
                t47,
                t48,
                t49,
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
        del t949, t49, t48, t47, t46

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t956 = paddle._C_ops.relu(t950)
        del t950

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t957 = paddle._C_ops.conv2d(
            t956, t50, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t958, t959, t960, t961, t962, t963 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t957,
                t51,
                t52,
                t53,
                t54,
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
        del t957, t54, t53, t52, t51

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t964 = paddle._C_ops.relu(t958)
        del t958

        # builtin.combine: ([-1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32]) <- (-1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32)
        t965 = [t916, t924, t932, t940, t948, t956, t964]
        del t964, t916, t924, t932, t940, t948, t956

        # pd_op.concat: (-1x672x56x56xf32) <- ([-1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32], 1xi32)
        t966 = paddle._C_ops.concat(t965, t898)
        del t965

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x672x56x56xf32, 96x672x1x1xf32)
        t967 = paddle._C_ops.conv2d(
            t966, t55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t966, t55

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t968, t969, t970, t971, t972, t973 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t967,
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
        del t967, t59, t58, t57, t56

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t974 = paddle._C_ops.relu(t968)
        del t968

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x96x56x56xf32, 192x96x1x1xf32)
        t975 = paddle._C_ops.conv2d(
            t974, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60, t974

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t976, t977, t978, t979, t980, t981 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t975,
                t61,
                t62,
                t63,
                t64,
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
        del t975, t64, t63, t62, t61

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t982 = paddle._C_ops.relu(t976)
        del t976

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x192x56x56xf32, 96x192x3x3xf32)
        t983 = paddle._C_ops.conv2d(
            t982, t65, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t984, t985, t986, t987, t988, t989 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t983,
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
        del t983, t69, t68, t67, t66

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t990 = paddle._C_ops.relu(t984)
        del t984

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t991 = paddle._C_ops.conv2d(
            t990, t70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t992, t993, t994, t995, t996, t997 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t991,
                t71,
                t72,
                t73,
                t74,
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
        del t991, t74, t73, t72, t71

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t998 = paddle._C_ops.relu(t992)
        del t992

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t999 = paddle._C_ops.conv2d(
            t998, t75, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1000, t1001, t1002, t1003, t1004, t1005 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t999,
                t76,
                t77,
                t78,
                t79,
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
        del t999, t79, t78, t77, t76

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t1006 = paddle._C_ops.relu(t1000)
        del t1000

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t1007 = paddle._C_ops.conv2d(
            t1006, t80, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1008, t1009, t1010, t1011, t1012, t1013 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1007,
                t81,
                t82,
                t83,
                t84,
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
        del t1007, t84, t83, t82, t81

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t1014 = paddle._C_ops.relu(t1008)
        del t1008

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t1015 = paddle._C_ops.conv2d(
            t1014, t85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1016, t1017, t1018, t1019, t1020, t1021 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1015,
                t86,
                t87,
                t88,
                t89,
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
        del t1015, t89, t88, t87, t86

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t1022 = paddle._C_ops.relu(t1016)
        del t1016

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        t1023 = paddle._C_ops.conv2d(
            t1022, t90, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1024, t1025, t1026, t1027, t1028, t1029 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1023,
                t91,
                t92,
                t93,
                t94,
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
        del t1023, t94, t93, t92, t91

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t1030 = paddle._C_ops.relu(t1024)
        del t1024

        # builtin.combine: ([-1x192x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32]) <- (-1x192x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32)
        t1031 = [t982, t990, t998, t1006, t1014, t1022, t1030]
        del t990, t998, t1006, t1014, t1022, t1030

        # pd_op.concat: (-1x768x56x56xf32) <- ([-1x192x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32], 1xi32)
        t1032 = paddle._C_ops.concat(t1031, t898)
        del t1031

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x768x56x56xf32, 96x768x1x1xf32)
        t1033 = paddle._C_ops.conv2d(
            t1032, t95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1032, t95

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        t1034, t1035, t1036, t1037, t1038, t1039 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1033,
                t96,
                t97,
                t98,
                t99,
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
        del t1033, t99, t98, t97, t96

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        t1040 = paddle._C_ops.relu(t1034)
        del t1034

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x96x56x56xf32, 192x96x1x1xf32)
        t1041 = paddle._C_ops.conv2d(
            t1040, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100, t1040

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1042, t1043, t1044, t1045, t1046, t1047 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1041,
                t101,
                t102,
                t103,
                t104,
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
        del t1041, t104, t103, t102, t101

        # pd_op.relu: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        t1048 = paddle._C_ops.relu(t1042)
        del t1042

        # pd_op.add: (-1x192x56x56xf32) <- (-1x192x56x56xf32, -1x192x56x56xf32)
        t1049 = paddle._C_ops.add(t1048, t982)
        del t982, t1048

        # pd_op.depthwise_conv2d: (-1x192x28x28xf32) <- (-1x192x56x56xf32, 192x1x3x3xf32)
        t1050 = paddle._C_ops.depthwise_conv2d(
            t1049, t105, [2, 2], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del t1049, t105

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1051, t1052, t1053, t1054, t1055, t1056 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1050,
                t106,
                t107,
                t108,
                t109,
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
        del t1050, t109, t108, t107, t106

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1057 = paddle._C_ops.conv2d(
            t1051, t110, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1058, t1059, t1060, t1061, t1062, t1063 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1057,
                t111,
                t112,
                t113,
                t114,
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
        del t1057, t114, t113, t112, t111

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1064 = paddle._C_ops.relu(t1058)
        del t1058

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1065 = paddle._C_ops.conv2d(
            t1064, t115, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1066, t1067, t1068, t1069, t1070, t1071 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1065,
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
        del t1065, t119, t118, t117, t116

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1072 = paddle._C_ops.relu(t1066)
        del t1066

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1073 = paddle._C_ops.conv2d(
            t1072, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1074, t1075, t1076, t1077, t1078, t1079 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1073,
                t121,
                t122,
                t123,
                t124,
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
        del t1073, t124, t123, t122, t121

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1080 = paddle._C_ops.relu(t1074)
        del t1074

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1081 = paddle._C_ops.conv2d(
            t1080, t125, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1082, t1083, t1084, t1085, t1086, t1087 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1081,
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
        del t1081, t129, t128, t127, t126

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1088 = paddle._C_ops.relu(t1082)
        del t1082

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1089 = paddle._C_ops.conv2d(
            t1088, t130, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1090, t1091, t1092, t1093, t1094, t1095 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1089,
                t131,
                t132,
                t133,
                t134,
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
        del t1089, t134, t133, t132, t131

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1096 = paddle._C_ops.relu(t1090)
        del t1090

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1097 = paddle._C_ops.conv2d(
            t1096, t135, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1098, t1099, t1100, t1101, t1102, t1103 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1097,
                t136,
                t137,
                t138,
                t139,
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
        del t1097, t139, t138, t137, t136

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1104 = paddle._C_ops.relu(t1098)
        del t1098

        # builtin.combine: ([-1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32]) <- (-1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32)
        t1105 = [t1051, t1064, t1072, t1080, t1088, t1096, t1104]
        del t1051, t1064, t1072, t1080, t1088, t1096, t1104

        # pd_op.concat: (-1x1344x28x28xf32) <- ([-1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32], 1xi32)
        t1106 = paddle._C_ops.concat(t1105, t898)
        del t1105

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x1344x28x28xf32, 256x1344x1x1xf32)
        t1107 = paddle._C_ops.conv2d(
            t1106, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1106, t140

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1108, t1109, t1110, t1111, t1112, t1113 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1107,
                t141,
                t142,
                t143,
                t144,
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
        del t1107, t144, t143, t142, t141

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t1114 = paddle._C_ops.relu(t1108)
        del t1108

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        t1115 = paddle._C_ops.conv2d(
            t1114, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145, t1114

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1116, t1117, t1118, t1119, t1120, t1121 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1115,
                t146,
                t147,
                t148,
                t149,
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
        del t1115, t149, t148, t147, t146

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1122 = paddle._C_ops.relu(t1116)
        del t1116

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x512x28x28xf32, 192x512x3x3xf32)
        t1123 = paddle._C_ops.conv2d(
            t1122, t150, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t150

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1124, t1125, t1126, t1127, t1128, t1129 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1123,
                t151,
                t152,
                t153,
                t154,
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
        del t1123, t154, t153, t152, t151

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1130 = paddle._C_ops.relu(t1124)
        del t1124

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1131 = paddle._C_ops.conv2d(
            t1130, t155, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1132, t1133, t1134, t1135, t1136, t1137 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1131,
                t156,
                t157,
                t158,
                t159,
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
        del t1131, t159, t158, t157, t156

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1138 = paddle._C_ops.relu(t1132)
        del t1132

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1139 = paddle._C_ops.conv2d(
            t1138, t160, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1140, t1141, t1142, t1143, t1144, t1145 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1139,
                t161,
                t162,
                t163,
                t164,
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
        del t1139, t164, t163, t162, t161

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1146 = paddle._C_ops.relu(t1140)
        del t1140

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1147 = paddle._C_ops.conv2d(
            t1146, t165, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1148, t1149, t1150, t1151, t1152, t1153 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1147,
                t166,
                t167,
                t168,
                t169,
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
        del t1147, t169, t168, t167, t166

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1154 = paddle._C_ops.relu(t1148)
        del t1148

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1155 = paddle._C_ops.conv2d(
            t1154, t170, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1156, t1157, t1158, t1159, t1160, t1161 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1155,
                t171,
                t172,
                t173,
                t174,
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
        del t1155, t174, t173, t172, t171

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1162 = paddle._C_ops.relu(t1156)
        del t1156

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1163 = paddle._C_ops.conv2d(
            t1162, t175, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1164, t1165, t1166, t1167, t1168, t1169 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1163,
                t176,
                t177,
                t178,
                t179,
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
        del t1163, t179, t178, t177, t176

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1170 = paddle._C_ops.relu(t1164)
        del t1164

        # builtin.combine: ([-1x512x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32]) <- (-1x512x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32)
        t1171 = [t1122, t1130, t1138, t1146, t1154, t1162, t1170]
        del t1130, t1138, t1146, t1154, t1162, t1170

        # pd_op.concat: (-1x1664x28x28xf32) <- ([-1x512x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32], 1xi32)
        t1172 = paddle._C_ops.concat(t1171, t898)
        del t1171

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x1664x28x28xf32, 256x1664x1x1xf32)
        t1173 = paddle._C_ops.conv2d(
            t1172, t180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1172, t180

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1174, t1175, t1176, t1177, t1178, t1179 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1173,
                t181,
                t182,
                t183,
                t184,
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
        del t1173, t184, t183, t182, t181

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t1180 = paddle._C_ops.relu(t1174)
        del t1174

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        t1181 = paddle._C_ops.conv2d(
            t1180, t185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185, t1180

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1182, t1183, t1184, t1185, t1186, t1187 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1181,
                t186,
                t187,
                t188,
                t189,
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
        del t1181, t189, t188, t187, t186

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1188 = paddle._C_ops.relu(t1182)
        del t1182

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1189 = paddle._C_ops.add(t1188, t1122)
        del t1122, t1188

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x512x28x28xf32, 192x512x3x3xf32)
        t1190 = paddle._C_ops.conv2d(
            t1189, t190, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1191, t1192, t1193, t1194, t1195, t1196 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1190,
                t191,
                t192,
                t193,
                t194,
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
        del t1190, t194, t193, t192, t191

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1197 = paddle._C_ops.relu(t1191)
        del t1191

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1198 = paddle._C_ops.conv2d(
            t1197, t195, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1199, t1200, t1201, t1202, t1203, t1204 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1198,
                t196,
                t197,
                t198,
                t199,
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
        del t1198, t199, t198, t197, t196

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1205 = paddle._C_ops.relu(t1199)
        del t1199

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1206 = paddle._C_ops.conv2d(
            t1205, t200, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1207, t1208, t1209, t1210, t1211, t1212 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1206,
                t201,
                t202,
                t203,
                t204,
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
        del t1206, t204, t203, t202, t201

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1213 = paddle._C_ops.relu(t1207)
        del t1207

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1214 = paddle._C_ops.conv2d(
            t1213, t205, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1215, t1216, t1217, t1218, t1219, t1220 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1214,
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
        del t1214, t209, t208, t207, t206

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1221 = paddle._C_ops.relu(t1215)
        del t1215

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1222 = paddle._C_ops.conv2d(
            t1221, t210, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1223, t1224, t1225, t1226, t1227, t1228 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1222,
                t211,
                t212,
                t213,
                t214,
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
        del t1222, t214, t213, t212, t211

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1229 = paddle._C_ops.relu(t1223)
        del t1223

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x192x3x3xf32)
        t1230 = paddle._C_ops.conv2d(
            t1229, t215, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        t1231, t1232, t1233, t1234, t1235, t1236 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1230,
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
        del t1230, t219, t218, t217, t216

        # pd_op.relu: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        t1237 = paddle._C_ops.relu(t1231)
        del t1231

        # builtin.combine: ([-1x512x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32]) <- (-1x512x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32)
        t1238 = [t1189, t1197, t1205, t1213, t1221, t1229, t1237]
        del t1197, t1205, t1213, t1221, t1229, t1237

        # pd_op.concat: (-1x1664x28x28xf32) <- ([-1x512x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32, -1x192x28x28xf32], 1xi32)
        t1239 = paddle._C_ops.concat(t1238, t898)
        del t1238

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x1664x28x28xf32, 256x1664x1x1xf32)
        t1240 = paddle._C_ops.conv2d(
            t1239, t220, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1239, t220

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1241, t1242, t1243, t1244, t1245, t1246 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1240,
                t221,
                t222,
                t223,
                t224,
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
        del t1240, t224, t223, t222, t221

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t1247 = paddle._C_ops.relu(t1241)
        del t1241

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        t1248 = paddle._C_ops.conv2d(
            t1247, t225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225, t1247

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1249, t1250, t1251, t1252, t1253, t1254 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1248,
                t226,
                t227,
                t228,
                t229,
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
        del t1248, t229, t228, t227, t226

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1255 = paddle._C_ops.relu(t1249)
        del t1249

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1256 = paddle._C_ops.add(t1255, t1189)
        del t1189, t1255

        # pd_op.depthwise_conv2d: (-1x512x14x14xf32) <- (-1x512x28x28xf32, 512x1x3x3xf32)
        t1257 = paddle._C_ops.depthwise_conv2d(
            t1256, t230, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del t1256, t230

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1258, t1259, t1260, t1261, t1262, t1263 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1257,
                t231,
                t232,
                t233,
                t234,
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
        del t1257, t234, t233, t232, t231

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x512x14x14xf32, 384x512x1x1xf32)
        t1264 = paddle._C_ops.conv2d(
            t1258, t235, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1265, t1266, t1267, t1268, t1269, t1270 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1264,
                t236,
                t237,
                t238,
                t239,
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
        del t1264, t239, t238, t237, t236

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1271 = paddle._C_ops.depthwise_conv2d(
            t1265, t240, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1265, t240

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1272, t1273, t1274, t1275, t1276, t1277 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1271,
                t241,
                t242,
                t243,
                t244,
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
        del t1271, t244, t243, t242, t241

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1278 = paddle._C_ops.relu(t1272)
        del t1272

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1279 = paddle._C_ops.conv2d(
            t1278, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1280, t1281, t1282, t1283, t1284, t1285 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1279,
                t246,
                t247,
                t248,
                t249,
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
        del t1279, t249, t248, t247, t246

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1286 = paddle._C_ops.depthwise_conv2d(
            t1280, t250, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1280, t250

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1287, t1288, t1289, t1290, t1291, t1292 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1286,
                t251,
                t252,
                t253,
                t254,
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
        del t1286, t254, t253, t252, t251

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1293 = paddle._C_ops.relu(t1287)
        del t1287

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1294 = paddle._C_ops.conv2d(
            t1293, t255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1295, t1296, t1297, t1298, t1299, t1300 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1294,
                t256,
                t257,
                t258,
                t259,
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
        del t1294, t259, t258, t257, t256

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1301 = paddle._C_ops.depthwise_conv2d(
            t1295, t260, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1295, t260

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1302, t1303, t1304, t1305, t1306, t1307 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1301,
                t261,
                t262,
                t263,
                t264,
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
        del t1301, t264, t263, t262, t261

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1308 = paddle._C_ops.relu(t1302)
        del t1302

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1309 = paddle._C_ops.conv2d(
            t1308, t265, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t265

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1310, t1311, t1312, t1313, t1314, t1315 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1309,
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
        del t1309, t269, t268, t267, t266

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1316 = paddle._C_ops.depthwise_conv2d(
            t1310, t270, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1310, t270

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1317, t1318, t1319, t1320, t1321, t1322 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1316,
                t271,
                t272,
                t273,
                t274,
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
        del t1316, t274, t273, t272, t271

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1323 = paddle._C_ops.relu(t1317)
        del t1317

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1324 = paddle._C_ops.conv2d(
            t1323, t275, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t275

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1325, t1326, t1327, t1328, t1329, t1330 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1324,
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
        del t1324, t279, t278, t277, t276

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1331 = paddle._C_ops.depthwise_conv2d(
            t1325, t280, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1325, t280

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1332, t1333, t1334, t1335, t1336, t1337 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1331,
                t281,
                t282,
                t283,
                t284,
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
        del t1331, t284, t283, t282, t281

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1338 = paddle._C_ops.relu(t1332)
        del t1332

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1339 = paddle._C_ops.conv2d(
            t1338, t285, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t285

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1340, t1341, t1342, t1343, t1344, t1345 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1339,
                t286,
                t287,
                t288,
                t289,
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
        del t1339, t289, t288, t287, t286

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1346 = paddle._C_ops.depthwise_conv2d(
            t1340, t290, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1340, t290

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1347, t1348, t1349, t1350, t1351, t1352 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1346,
                t291,
                t292,
                t293,
                t294,
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
        del t1346, t294, t293, t292, t291

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1353 = paddle._C_ops.relu(t1347)
        del t1347

        # builtin.combine: ([-1x512x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32)
        t1354 = [t1258, t1278, t1293, t1308, t1323, t1338, t1353]
        del t1258, t1278, t1293, t1308, t1323, t1338, t1353

        # pd_op.concat: (-1x2816x14x14xf32) <- ([-1x512x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32], 1xi32)
        t1355 = paddle._C_ops.concat(t1354, t898)
        del t1354

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x2816x14x14xf32, 512x2816x1x1xf32)
        t1356 = paddle._C_ops.conv2d(
            t1355, t295, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1355, t295

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1357, t1358, t1359, t1360, t1361, t1362 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1356,
                t296,
                t297,
                t298,
                t299,
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
        del t1356, t299, t298, t297, t296

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1363 = paddle._C_ops.relu(t1357)
        del t1357

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1364 = paddle._C_ops.conv2d(
            t1363, t300, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t300, t1363

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1365, t1366, t1367, t1368, t1369, t1370 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1364,
                t301,
                t302,
                t303,
                t304,
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
        del t1364, t304, t303, t302, t301

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1371 = paddle._C_ops.relu(t1365)
        del t1365

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x1024x14x14xf32, 384x1024x1x1xf32)
        t1372 = paddle._C_ops.conv2d(
            t1371, t305, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t305

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1373, t1374, t1375, t1376, t1377, t1378 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1372,
                t306,
                t307,
                t308,
                t309,
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
        del t1372, t309, t308, t307, t306

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1379 = paddle._C_ops.depthwise_conv2d(
            t1373, t310, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1373, t310

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1380, t1381, t1382, t1383, t1384, t1385 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1379,
                t311,
                t312,
                t313,
                t314,
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
        del t1379, t314, t313, t312, t311

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1386 = paddle._C_ops.relu(t1380)
        del t1380

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1387 = paddle._C_ops.conv2d(
            t1386, t315, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t315

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1388, t1389, t1390, t1391, t1392, t1393 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1387,
                t316,
                t317,
                t318,
                t319,
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
        del t1387, t319, t318, t317, t316

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1394 = paddle._C_ops.depthwise_conv2d(
            t1388, t320, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1388, t320

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1395, t1396, t1397, t1398, t1399, t1400 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1394,
                t321,
                t322,
                t323,
                t324,
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
        del t1394, t324, t323, t322, t321

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1401 = paddle._C_ops.relu(t1395)
        del t1395

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1402 = paddle._C_ops.conv2d(
            t1401, t325, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t325

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1403, t1404, t1405, t1406, t1407, t1408 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1402,
                t326,
                t327,
                t328,
                t329,
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
        del t1402, t329, t328, t327, t326

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1409 = paddle._C_ops.depthwise_conv2d(
            t1403, t330, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1403, t330

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1410, t1411, t1412, t1413, t1414, t1415 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1409,
                t331,
                t332,
                t333,
                t334,
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
        del t1409, t334, t333, t332, t331

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1416 = paddle._C_ops.relu(t1410)
        del t1410

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1417 = paddle._C_ops.conv2d(
            t1416, t335, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t335

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1418, t1419, t1420, t1421, t1422, t1423 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1417,
                t336,
                t337,
                t338,
                t339,
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
        del t1417, t339, t338, t337, t336

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1424 = paddle._C_ops.depthwise_conv2d(
            t1418, t340, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1418, t340

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1425, t1426, t1427, t1428, t1429, t1430 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1424,
                t341,
                t342,
                t343,
                t344,
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
        del t1424, t344, t343, t342, t341

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1431 = paddle._C_ops.relu(t1425)
        del t1425

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1432 = paddle._C_ops.conv2d(
            t1431, t345, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t345

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1433, t1434, t1435, t1436, t1437, t1438 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1432,
                t346,
                t347,
                t348,
                t349,
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
        del t1432, t349, t348, t347, t346

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1439 = paddle._C_ops.depthwise_conv2d(
            t1433, t350, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1433, t350

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1440, t1441, t1442, t1443, t1444, t1445 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1439,
                t351,
                t352,
                t353,
                t354,
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
        del t1439, t354, t353, t352, t351

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1446 = paddle._C_ops.relu(t1440)
        del t1440

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1447 = paddle._C_ops.conv2d(
            t1446, t355, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t355

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1448, t1449, t1450, t1451, t1452, t1453 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1447,
                t356,
                t357,
                t358,
                t359,
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
        del t1447, t359, t358, t357, t356

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1454 = paddle._C_ops.depthwise_conv2d(
            t1448, t360, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1448, t360

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1455, t1456, t1457, t1458, t1459, t1460 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1454,
                t361,
                t362,
                t363,
                t364,
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
        del t1454, t364, t363, t362, t361

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1461 = paddle._C_ops.relu(t1455)
        del t1455

        # builtin.combine: ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32]) <- (-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32)
        t1462 = [t1371, t1386, t1401, t1416, t1431, t1446, t1461]
        del t1386, t1401, t1416, t1431, t1446, t1461

        # pd_op.concat: (-1x3328x14x14xf32) <- ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32], 1xi32)
        t1463 = paddle._C_ops.concat(t1462, t898)
        del t1462

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x3328x14x14xf32, 512x3328x1x1xf32)
        t1464 = paddle._C_ops.conv2d(
            t1463, t365, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1463, t365

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1465, t1466, t1467, t1468, t1469, t1470 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1464,
                t366,
                t367,
                t368,
                t369,
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
        del t1464, t369, t368, t367, t366

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1471 = paddle._C_ops.relu(t1465)
        del t1465

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1472 = paddle._C_ops.conv2d(
            t1471, t370, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t370, t1471

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1473, t1474, t1475, t1476, t1477, t1478 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1472,
                t371,
                t372,
                t373,
                t374,
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
        del t1472, t374, t373, t372, t371

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1479 = paddle._C_ops.relu(t1473)
        del t1473

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1480 = paddle._C_ops.add(t1479, t1371)
        del t1371, t1479

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x1024x14x14xf32, 384x1024x1x1xf32)
        t1481 = paddle._C_ops.conv2d(
            t1480, t375, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t375

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1482, t1483, t1484, t1485, t1486, t1487 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1481,
                t376,
                t377,
                t378,
                t379,
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
        del t1481, t379, t378, t377, t376

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1488 = paddle._C_ops.depthwise_conv2d(
            t1482, t380, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1482, t380

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1489, t1490, t1491, t1492, t1493, t1494 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1488,
                t381,
                t382,
                t383,
                t384,
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
        del t1488, t384, t383, t382, t381

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1495 = paddle._C_ops.relu(t1489)
        del t1489

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1496 = paddle._C_ops.conv2d(
            t1495, t385, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t385

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1497, t1498, t1499, t1500, t1501, t1502 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1496,
                t386,
                t387,
                t388,
                t389,
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
        del t1496, t389, t388, t387, t386

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1503 = paddle._C_ops.depthwise_conv2d(
            t1497, t390, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1497, t390

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1504, t1505, t1506, t1507, t1508, t1509 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1503,
                t391,
                t392,
                t393,
                t394,
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
        del t1503, t394, t393, t392, t391

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1510 = paddle._C_ops.relu(t1504)
        del t1504

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1511 = paddle._C_ops.conv2d(
            t1510, t395, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t395

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1512, t1513, t1514, t1515, t1516, t1517 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1511,
                t396,
                t397,
                t398,
                t399,
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
        del t1511, t399, t398, t397, t396

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1518 = paddle._C_ops.depthwise_conv2d(
            t1512, t400, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1512, t400

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1519, t1520, t1521, t1522, t1523, t1524 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1518,
                t401,
                t402,
                t403,
                t404,
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
        del t1518, t404, t403, t402, t401

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1525 = paddle._C_ops.relu(t1519)
        del t1519

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1526 = paddle._C_ops.conv2d(
            t1525, t405, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t405

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1527, t1528, t1529, t1530, t1531, t1532 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1526,
                t406,
                t407,
                t408,
                t409,
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
        del t1526, t409, t408, t407, t406

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1533 = paddle._C_ops.depthwise_conv2d(
            t1527, t410, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1527, t410

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1534, t1535, t1536, t1537, t1538, t1539 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1533,
                t411,
                t412,
                t413,
                t414,
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
        del t1533, t414, t413, t412, t411

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1540 = paddle._C_ops.relu(t1534)
        del t1534

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1541 = paddle._C_ops.conv2d(
            t1540, t415, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t415

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1542, t1543, t1544, t1545, t1546, t1547 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1541,
                t416,
                t417,
                t418,
                t419,
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
        del t1541, t419, t418, t417, t416

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1548 = paddle._C_ops.depthwise_conv2d(
            t1542, t420, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1542, t420

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1549, t1550, t1551, t1552, t1553, t1554 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1548,
                t421,
                t422,
                t423,
                t424,
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
        del t1548, t424, t423, t422, t421

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1555 = paddle._C_ops.relu(t1549)
        del t1549

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1556 = paddle._C_ops.conv2d(
            t1555, t425, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t425

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1557, t1558, t1559, t1560, t1561, t1562 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1556,
                t426,
                t427,
                t428,
                t429,
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
        del t1556, t429, t428, t427, t426

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1563 = paddle._C_ops.depthwise_conv2d(
            t1557, t430, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1557, t430

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1564, t1565, t1566, t1567, t1568, t1569 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1563,
                t431,
                t432,
                t433,
                t434,
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
        del t1563, t434, t433, t432, t431

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1570 = paddle._C_ops.relu(t1564)
        del t1564

        # builtin.combine: ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32]) <- (-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32)
        t1571 = [t1480, t1495, t1510, t1525, t1540, t1555, t1570]
        del t1495, t1510, t1525, t1540, t1555, t1570

        # pd_op.concat: (-1x3328x14x14xf32) <- ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32], 1xi32)
        t1572 = paddle._C_ops.concat(t1571, t898)
        del t1571

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x3328x14x14xf32, 512x3328x1x1xf32)
        t1573 = paddle._C_ops.conv2d(
            t1572, t435, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1572, t435

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1574, t1575, t1576, t1577, t1578, t1579 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1573,
                t436,
                t437,
                t438,
                t439,
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
        del t1573, t439, t438, t437, t436

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1580 = paddle._C_ops.relu(t1574)
        del t1574

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1581 = paddle._C_ops.conv2d(
            t1580, t440, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t440, t1580

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1582, t1583, t1584, t1585, t1586, t1587 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1581,
                t441,
                t442,
                t443,
                t444,
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
        del t1581, t444, t443, t442, t441

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1588 = paddle._C_ops.relu(t1582)
        del t1582

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1589 = paddle._C_ops.add(t1588, t1480)
        del t1480, t1588

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x1024x14x14xf32, 384x1024x1x1xf32)
        t1590 = paddle._C_ops.conv2d(
            t1589, t445, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t445

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1591, t1592, t1593, t1594, t1595, t1596 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1590,
                t446,
                t447,
                t448,
                t449,
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
        del t1590, t449, t448, t447, t446

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1597 = paddle._C_ops.depthwise_conv2d(
            t1591, t450, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1591, t450

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1598, t1599, t1600, t1601, t1602, t1603 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1597,
                t451,
                t452,
                t453,
                t454,
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
        del t1597, t454, t453, t452, t451

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1604 = paddle._C_ops.relu(t1598)
        del t1598

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1605 = paddle._C_ops.conv2d(
            t1604, t455, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t455

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1606, t1607, t1608, t1609, t1610, t1611 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1605,
                t456,
                t457,
                t458,
                t459,
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
        del t1605, t459, t458, t457, t456

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1612 = paddle._C_ops.depthwise_conv2d(
            t1606, t460, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1606, t460

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1613, t1614, t1615, t1616, t1617, t1618 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1612,
                t461,
                t462,
                t463,
                t464,
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
        del t1612, t464, t463, t462, t461

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1619 = paddle._C_ops.relu(t1613)
        del t1613

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1620 = paddle._C_ops.conv2d(
            t1619, t465, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t465

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1621, t1622, t1623, t1624, t1625, t1626 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1620,
                t466,
                t467,
                t468,
                t469,
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
        del t1620, t469, t468, t467, t466

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1627 = paddle._C_ops.depthwise_conv2d(
            t1621, t470, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1621, t470

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1628, t1629, t1630, t1631, t1632, t1633 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1627,
                t471,
                t472,
                t473,
                t474,
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
        del t1627, t474, t473, t472, t471

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1634 = paddle._C_ops.relu(t1628)
        del t1628

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1635 = paddle._C_ops.conv2d(
            t1634, t475, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t475

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1636, t1637, t1638, t1639, t1640, t1641 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1635,
                t476,
                t477,
                t478,
                t479,
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
        del t1635, t479, t478, t477, t476

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1642 = paddle._C_ops.depthwise_conv2d(
            t1636, t480, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1636, t480

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1643, t1644, t1645, t1646, t1647, t1648 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1642,
                t481,
                t482,
                t483,
                t484,
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
        del t1642, t484, t483, t482, t481

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1649 = paddle._C_ops.relu(t1643)
        del t1643

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1650 = paddle._C_ops.conv2d(
            t1649, t485, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t485

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1651, t1652, t1653, t1654, t1655, t1656 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1650,
                t486,
                t487,
                t488,
                t489,
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
        del t1650, t489, t488, t487, t486

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1657 = paddle._C_ops.depthwise_conv2d(
            t1651, t490, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1651, t490

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1658, t1659, t1660, t1661, t1662, t1663 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1657,
                t491,
                t492,
                t493,
                t494,
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
        del t1657, t494, t493, t492, t491

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1664 = paddle._C_ops.relu(t1658)
        del t1658

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1665 = paddle._C_ops.conv2d(
            t1664, t495, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t495

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1666, t1667, t1668, t1669, t1670, t1671 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1665,
                t496,
                t497,
                t498,
                t499,
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
        del t1665, t499, t498, t497, t496

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1672 = paddle._C_ops.depthwise_conv2d(
            t1666, t500, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1666, t500

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1673, t1674, t1675, t1676, t1677, t1678 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1672,
                t501,
                t502,
                t503,
                t504,
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
        del t1672, t504, t503, t502, t501

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1679 = paddle._C_ops.relu(t1673)
        del t1673

        # builtin.combine: ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32]) <- (-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32)
        t1680 = [t1589, t1604, t1619, t1634, t1649, t1664, t1679]
        del t1604, t1619, t1634, t1649, t1664, t1679

        # pd_op.concat: (-1x3328x14x14xf32) <- ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32], 1xi32)
        t1681 = paddle._C_ops.concat(t1680, t898)
        del t1680

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x3328x14x14xf32, 512x3328x1x1xf32)
        t1682 = paddle._C_ops.conv2d(
            t1681, t505, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1681, t505

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1683, t1684, t1685, t1686, t1687, t1688 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1682,
                t506,
                t507,
                t508,
                t509,
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
        del t1682, t509, t508, t507, t506

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1689 = paddle._C_ops.relu(t1683)
        del t1683

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1690 = paddle._C_ops.conv2d(
            t1689, t510, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t510, t1689

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1691, t1692, t1693, t1694, t1695, t1696 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1690,
                t511,
                t512,
                t513,
                t514,
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
        del t1690, t514, t513, t512, t511

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1697 = paddle._C_ops.relu(t1691)
        del t1691

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1698 = paddle._C_ops.add(t1697, t1589)
        del t1589, t1697

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x1024x14x14xf32, 384x1024x1x1xf32)
        t1699 = paddle._C_ops.conv2d(
            t1698, t515, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t515

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1700, t1701, t1702, t1703, t1704, t1705 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1699,
                t516,
                t517,
                t518,
                t519,
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
        del t1699, t519, t518, t517, t516

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1706 = paddle._C_ops.depthwise_conv2d(
            t1700, t520, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1700, t520

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1707, t1708, t1709, t1710, t1711, t1712 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1706,
                t521,
                t522,
                t523,
                t524,
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
        del t1706, t524, t523, t522, t521

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1713 = paddle._C_ops.relu(t1707)
        del t1707

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1714 = paddle._C_ops.conv2d(
            t1713, t525, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t525

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1715, t1716, t1717, t1718, t1719, t1720 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1714,
                t526,
                t527,
                t528,
                t529,
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
        del t1714, t529, t528, t527, t526

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1721 = paddle._C_ops.depthwise_conv2d(
            t1715, t530, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1715, t530

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1722, t1723, t1724, t1725, t1726, t1727 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1721,
                t531,
                t532,
                t533,
                t534,
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
        del t1721, t534, t533, t532, t531

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1728 = paddle._C_ops.relu(t1722)
        del t1722

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1729 = paddle._C_ops.conv2d(
            t1728, t535, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t535

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1730, t1731, t1732, t1733, t1734, t1735 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1729,
                t536,
                t537,
                t538,
                t539,
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
        del t1729, t539, t538, t537, t536

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1736 = paddle._C_ops.depthwise_conv2d(
            t1730, t540, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1730, t540

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1737, t1738, t1739, t1740, t1741, t1742 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1736,
                t541,
                t542,
                t543,
                t544,
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
        del t1736, t544, t543, t542, t541

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1743 = paddle._C_ops.relu(t1737)
        del t1737

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1744 = paddle._C_ops.conv2d(
            t1743, t545, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t545

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1745, t1746, t1747, t1748, t1749, t1750 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1744,
                t546,
                t547,
                t548,
                t549,
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
        del t1744, t549, t548, t547, t546

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1751 = paddle._C_ops.depthwise_conv2d(
            t1745, t550, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1745, t550

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1752, t1753, t1754, t1755, t1756, t1757 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1751,
                t551,
                t552,
                t553,
                t554,
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
        del t1751, t554, t553, t552, t551

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1758 = paddle._C_ops.relu(t1752)
        del t1752

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1759 = paddle._C_ops.conv2d(
            t1758, t555, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t555

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1760, t1761, t1762, t1763, t1764, t1765 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1759,
                t556,
                t557,
                t558,
                t559,
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
        del t1759, t559, t558, t557, t556

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1766 = paddle._C_ops.depthwise_conv2d(
            t1760, t560, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1760, t560

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1767, t1768, t1769, t1770, t1771, t1772 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1766,
                t561,
                t562,
                t563,
                t564,
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
        del t1766, t564, t563, t562, t561

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1773 = paddle._C_ops.relu(t1767)
        del t1767

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1774 = paddle._C_ops.conv2d(
            t1773, t565, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t565

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1775, t1776, t1777, t1778, t1779, t1780 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1774,
                t566,
                t567,
                t568,
                t569,
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
        del t1774, t569, t568, t567, t566

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1781 = paddle._C_ops.depthwise_conv2d(
            t1775, t570, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1775, t570

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1782, t1783, t1784, t1785, t1786, t1787 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1781,
                t571,
                t572,
                t573,
                t574,
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
        del t1781, t574, t573, t572, t571

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1788 = paddle._C_ops.relu(t1782)
        del t1782

        # builtin.combine: ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32]) <- (-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32)
        t1789 = [t1698, t1713, t1728, t1743, t1758, t1773, t1788]
        del t1713, t1728, t1743, t1758, t1773, t1788

        # pd_op.concat: (-1x3328x14x14xf32) <- ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32], 1xi32)
        t1790 = paddle._C_ops.concat(t1789, t898)
        del t1789

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x3328x14x14xf32, 512x3328x1x1xf32)
        t1791 = paddle._C_ops.conv2d(
            t1790, t575, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1790, t575

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1792, t1793, t1794, t1795, t1796, t1797 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1791,
                t576,
                t577,
                t578,
                t579,
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
        del t1791, t579, t578, t577, t576

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1798 = paddle._C_ops.relu(t1792)
        del t1792

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1799 = paddle._C_ops.conv2d(
            t1798, t580, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t580, t1798

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1800, t1801, t1802, t1803, t1804, t1805 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1799,
                t581,
                t582,
                t583,
                t584,
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
        del t1799, t584, t583, t582, t581

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1806 = paddle._C_ops.relu(t1800)
        del t1800

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1807 = paddle._C_ops.add(t1806, t1698)
        del t1698, t1806

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x1024x14x14xf32, 384x1024x1x1xf32)
        t1808 = paddle._C_ops.conv2d(
            t1807, t585, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t585

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1809, t1810, t1811, t1812, t1813, t1814 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1808,
                t586,
                t587,
                t588,
                t589,
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
        del t1808, t589, t588, t587, t586

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1815 = paddle._C_ops.depthwise_conv2d(
            t1809, t590, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1809, t590

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1816, t1817, t1818, t1819, t1820, t1821 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1815,
                t591,
                t592,
                t593,
                t594,
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
        del t1815, t594, t593, t592, t591

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1822 = paddle._C_ops.relu(t1816)
        del t1816

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1823 = paddle._C_ops.conv2d(
            t1822, t595, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t595

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1824, t1825, t1826, t1827, t1828, t1829 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1823,
                t596,
                t597,
                t598,
                t599,
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
        del t1823, t599, t598, t597, t596

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1830 = paddle._C_ops.depthwise_conv2d(
            t1824, t600, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1824, t600

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1831, t1832, t1833, t1834, t1835, t1836 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1830,
                t601,
                t602,
                t603,
                t604,
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
        del t1830, t604, t603, t602, t601

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1837 = paddle._C_ops.relu(t1831)
        del t1831

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1838 = paddle._C_ops.conv2d(
            t1837, t605, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t605

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1839, t1840, t1841, t1842, t1843, t1844 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1838,
                t606,
                t607,
                t608,
                t609,
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
        del t1838, t609, t608, t607, t606

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1845 = paddle._C_ops.depthwise_conv2d(
            t1839, t610, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1839, t610

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1846, t1847, t1848, t1849, t1850, t1851 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1845,
                t611,
                t612,
                t613,
                t614,
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
        del t1845, t614, t613, t612, t611

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1852 = paddle._C_ops.relu(t1846)
        del t1846

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1853 = paddle._C_ops.conv2d(
            t1852, t615, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t615

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1854, t1855, t1856, t1857, t1858, t1859 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1853,
                t616,
                t617,
                t618,
                t619,
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
        del t1853, t619, t618, t617, t616

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1860 = paddle._C_ops.depthwise_conv2d(
            t1854, t620, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1854, t620

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1861, t1862, t1863, t1864, t1865, t1866 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1860,
                t621,
                t622,
                t623,
                t624,
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
        del t1860, t624, t623, t622, t621

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1867 = paddle._C_ops.relu(t1861)
        del t1861

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1868 = paddle._C_ops.conv2d(
            t1867, t625, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t625

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1869, t1870, t1871, t1872, t1873, t1874 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1868,
                t626,
                t627,
                t628,
                t629,
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
        del t1868, t629, t628, t627, t626

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1875 = paddle._C_ops.depthwise_conv2d(
            t1869, t630, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1869, t630

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1876, t1877, t1878, t1879, t1880, t1881 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1875,
                t631,
                t632,
                t633,
                t634,
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
        del t1875, t634, t633, t632, t631

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1882 = paddle._C_ops.relu(t1876)
        del t1876

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x384x1x1xf32)
        t1883 = paddle._C_ops.conv2d(
            t1882, t635, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t635

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1884, t1885, t1886, t1887, t1888, t1889 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1883,
                t636,
                t637,
                t638,
                t639,
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
        del t1883, t639, t638, t637, t636

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x5x5xf32)
        t1890 = paddle._C_ops.depthwise_conv2d(
            t1884, t640, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del t1884, t640

        # pd_op.batch_norm_: (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        t1891, t1892, t1893, t1894, t1895, t1896 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1890,
                t641,
                t642,
                t643,
                t644,
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
        del t1890, t644, t643, t642, t641

        # pd_op.relu: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        t1897 = paddle._C_ops.relu(t1891)
        del t1891

        # builtin.combine: ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32]) <- (-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32)
        t1898 = [t1807, t1822, t1837, t1852, t1867, t1882, t1897]
        del t1822, t1837, t1852, t1867, t1882, t1897

        # pd_op.concat: (-1x3328x14x14xf32) <- ([-1x1024x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32, -1x384x14x14xf32], 1xi32)
        t1899 = paddle._C_ops.concat(t1898, t898)
        del t1898

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x3328x14x14xf32, 512x3328x1x1xf32)
        t1900 = paddle._C_ops.conv2d(
            t1899, t645, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1899, t645

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1901, t1902, t1903, t1904, t1905, t1906 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1900,
                t646,
                t647,
                t648,
                t649,
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
        del t1900, t649, t648, t647, t646

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t1907 = paddle._C_ops.relu(t1901)
        del t1901

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1908 = paddle._C_ops.conv2d(
            t1907, t650, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t650, t1907

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1909, t1910, t1911, t1912, t1913, t1914 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1908,
                t651,
                t652,
                t653,
                t654,
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
        del t1908, t654, t653, t652, t651

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1915 = paddle._C_ops.relu(t1909)
        del t1909

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1916 = paddle._C_ops.add(t1915, t1807)
        del t1807, t1915

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf32) <- (-1x1024x14x14xf32, 1024x1x3x3xf32)
        t1917 = paddle._C_ops.depthwise_conv2d(
            t1916, t655, [2, 2], [1, 1], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del t1916, t655

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1918, t1919, t1920, t1921, t1922, t1923 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1917,
                t656,
                t657,
                t658,
                t659,
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
        del t1917, t659, t658, t657, t656

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x1024x7x7xf32, 768x1024x1x1xf32)
        t1924 = paddle._C_ops.conv2d(
            t1918, t660, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t660

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1925, t1926, t1927, t1928, t1929, t1930 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1924,
                t661,
                t662,
                t663,
                t664,
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
        del t1924, t664, t663, t662, t661

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t1931 = paddle._C_ops.depthwise_conv2d(
            t1925, t665, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t1925, t665

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1932, t1933, t1934, t1935, t1936, t1937 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1931,
                t666,
                t667,
                t668,
                t669,
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
        del t1931, t669, t668, t667, t666

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t1938 = paddle._C_ops.relu(t1932)
        del t1932

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t1939 = paddle._C_ops.conv2d(
            t1938, t670, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t670

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1940, t1941, t1942, t1943, t1944, t1945 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1939,
                t671,
                t672,
                t673,
                t674,
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
        del t1939, t674, t673, t672, t671

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t1946 = paddle._C_ops.depthwise_conv2d(
            t1940, t675, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t1940, t675

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1947, t1948, t1949, t1950, t1951, t1952 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1946,
                t676,
                t677,
                t678,
                t679,
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
        del t1946, t679, t678, t677, t676

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t1953 = paddle._C_ops.relu(t1947)
        del t1947

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t1954 = paddle._C_ops.conv2d(
            t1953, t680, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t680

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1955, t1956, t1957, t1958, t1959, t1960 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1954,
                t681,
                t682,
                t683,
                t684,
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
        del t1954, t684, t683, t682, t681

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t1961 = paddle._C_ops.depthwise_conv2d(
            t1955, t685, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t1955, t685

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1962, t1963, t1964, t1965, t1966, t1967 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1961,
                t686,
                t687,
                t688,
                t689,
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
        del t1961, t689, t688, t687, t686

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t1968 = paddle._C_ops.relu(t1962)
        del t1962

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t1969 = paddle._C_ops.conv2d(
            t1968, t690, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t690

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1970, t1971, t1972, t1973, t1974, t1975 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1969,
                t691,
                t692,
                t693,
                t694,
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
        del t1969, t694, t693, t692, t691

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t1976 = paddle._C_ops.depthwise_conv2d(
            t1970, t695, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t1970, t695

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1977, t1978, t1979, t1980, t1981, t1982 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1976,
                t696,
                t697,
                t698,
                t699,
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
        del t1976, t699, t698, t697, t696

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t1983 = paddle._C_ops.relu(t1977)
        del t1977

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t1984 = paddle._C_ops.conv2d(
            t1983, t700, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t700

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1985, t1986, t1987, t1988, t1989, t1990 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1984,
                t701,
                t702,
                t703,
                t704,
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
        del t1984, t704, t703, t702, t701

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t1991 = paddle._C_ops.depthwise_conv2d(
            t1985, t705, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t1985, t705

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t1992, t1993, t1994, t1995, t1996, t1997 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1991,
                t706,
                t707,
                t708,
                t709,
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
        del t1991, t709, t708, t707, t706

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t1998 = paddle._C_ops.relu(t1992)
        del t1992

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t1999 = paddle._C_ops.conv2d(
            t1998, t710, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t710

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2000, t2001, t2002, t2003, t2004, t2005 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1999,
                t711,
                t712,
                t713,
                t714,
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
        del t1999, t714, t713, t712, t711

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2006 = paddle._C_ops.depthwise_conv2d(
            t2000, t715, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2000, t715

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2007, t2008, t2009, t2010, t2011, t2012 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2006,
                t716,
                t717,
                t718,
                t719,
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
        del t2006, t719, t718, t717, t716

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2013 = paddle._C_ops.relu(t2007)
        del t2007

        # builtin.combine: ([-1x1024x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32]) <- (-1x1024x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32)
        t2014 = [t1918, t1938, t1953, t1968, t1983, t1998, t2013]
        del t1918, t1938, t1953, t1968, t1983, t1998, t2013

        # pd_op.concat: (-1x5632x7x7xf32) <- ([-1x1024x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32], 1xi32)
        t2015 = paddle._C_ops.concat(t2014, t898)
        del t2014

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x5632x7x7xf32, 1024x5632x1x1xf32)
        t2016 = paddle._C_ops.conv2d(
            t2015, t720, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t2015, t720

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2017, t2018, t2019, t2020, t2021, t2022 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2016,
                t721,
                t722,
                t723,
                t724,
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
        del t2016, t724, t723, t722, t721

        # pd_op.relu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t2023 = paddle._C_ops.relu(t2017)
        del t2017

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        t2024 = paddle._C_ops.conv2d(
            t2023, t725, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t725, t2023

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t2025, t2026, t2027, t2028, t2029, t2030 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2024,
                t726,
                t727,
                t728,
                t729,
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
        del t2024, t729, t728, t727, t726

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t2031 = paddle._C_ops.relu(t2025)
        del t2025

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x2048x7x7xf32, 768x2048x1x1xf32)
        t2032 = paddle._C_ops.conv2d(
            t2031, t730, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t730

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2033, t2034, t2035, t2036, t2037, t2038 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2032,
                t731,
                t732,
                t733,
                t734,
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
        del t2032, t734, t733, t732, t731

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2039 = paddle._C_ops.depthwise_conv2d(
            t2033, t735, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2033, t735

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2040, t2041, t2042, t2043, t2044, t2045 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2039,
                t736,
                t737,
                t738,
                t739,
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
        del t2039, t739, t738, t737, t736

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2046 = paddle._C_ops.relu(t2040)
        del t2040

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2047 = paddle._C_ops.conv2d(
            t2046, t740, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t740

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2048, t2049, t2050, t2051, t2052, t2053 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2047,
                t741,
                t742,
                t743,
                t744,
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
        del t2047, t744, t743, t742, t741

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2054 = paddle._C_ops.depthwise_conv2d(
            t2048, t745, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2048, t745

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2055, t2056, t2057, t2058, t2059, t2060 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2054,
                t746,
                t747,
                t748,
                t749,
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
        del t2054, t749, t748, t747, t746

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2061 = paddle._C_ops.relu(t2055)
        del t2055

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2062 = paddle._C_ops.conv2d(
            t2061, t750, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t750

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2063, t2064, t2065, t2066, t2067, t2068 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2062,
                t751,
                t752,
                t753,
                t754,
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
        del t2062, t754, t753, t752, t751

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2069 = paddle._C_ops.depthwise_conv2d(
            t2063, t755, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2063, t755

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2070, t2071, t2072, t2073, t2074, t2075 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2069,
                t756,
                t757,
                t758,
                t759,
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
        del t2069, t759, t758, t757, t756

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2076 = paddle._C_ops.relu(t2070)
        del t2070

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2077 = paddle._C_ops.conv2d(
            t2076, t760, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t760

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2078, t2079, t2080, t2081, t2082, t2083 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2077,
                t761,
                t762,
                t763,
                t764,
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
        del t2077, t764, t763, t762, t761

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2084 = paddle._C_ops.depthwise_conv2d(
            t2078, t765, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2078, t765

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2085, t2086, t2087, t2088, t2089, t2090 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2084,
                t766,
                t767,
                t768,
                t769,
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
        del t2084, t769, t768, t767, t766

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2091 = paddle._C_ops.relu(t2085)
        del t2085

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2092 = paddle._C_ops.conv2d(
            t2091, t770, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t770

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2093, t2094, t2095, t2096, t2097, t2098 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2092,
                t771,
                t772,
                t773,
                t774,
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
        del t2092, t772, t771, t774, t773

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2099 = paddle._C_ops.depthwise_conv2d(
            t2093, t775, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2093, t775

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2100, t2101, t2102, t2103, t2104, t2105 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2099,
                t776,
                t777,
                t778,
                t779,
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
        del t2099, t779, t778, t777, t776

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2106 = paddle._C_ops.relu(t2100)
        del t2100

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2107 = paddle._C_ops.conv2d(
            t2106, t780, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t780

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2108, t2109, t2110, t2111, t2112, t2113 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2107,
                t781,
                t782,
                t783,
                t784,
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
        del t2107, t784, t783, t782, t781

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2114 = paddle._C_ops.depthwise_conv2d(
            t2108, t785, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2108, t785

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2115, t2116, t2117, t2118, t2119, t2120 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2114,
                t786,
                t787,
                t788,
                t789,
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
        del t2114, t789, t788, t787, t786

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2121 = paddle._C_ops.relu(t2115)
        del t2115

        # builtin.combine: ([-1x2048x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32]) <- (-1x2048x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32)
        t2122 = [t2031, t2046, t2061, t2076, t2091, t2106, t2121]
        del t2046, t2061, t2076, t2091, t2106, t2121

        # pd_op.concat: (-1x6656x7x7xf32) <- ([-1x2048x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32], 1xi32)
        t2123 = paddle._C_ops.concat(t2122, t898)
        del t2122

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x6656x7x7xf32, 1024x6656x1x1xf32)
        t2124 = paddle._C_ops.conv2d(
            t2123, t790, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t2123, t790

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2125, t2126, t2127, t2128, t2129, t2130 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2124,
                t791,
                t792,
                t793,
                t794,
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
        del t2124, t794, t793, t792, t791

        # pd_op.relu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t2131 = paddle._C_ops.relu(t2125)
        del t2125

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        t2132 = paddle._C_ops.conv2d(
            t2131, t795, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t795, t2131

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t2133, t2134, t2135, t2136, t2137, t2138 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2132,
                t796,
                t797,
                t798,
                t799,
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
        del t2132, t799, t798, t797, t796

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t2139 = paddle._C_ops.relu(t2133)
        del t2133

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t2140 = paddle._C_ops.add(t2139, t2031)
        del t2031, t2139

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x2048x7x7xf32, 768x2048x1x1xf32)
        t2141 = paddle._C_ops.conv2d(
            t2140, t800, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t800

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2142, t2143, t2144, t2145, t2146, t2147 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2141,
                t801,
                t802,
                t803,
                t804,
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
        del t2141, t804, t803, t802, t801

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2148 = paddle._C_ops.depthwise_conv2d(
            t2142, t805, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2142, t805

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2149, t2150, t2151, t2152, t2153, t2154 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2148,
                t806,
                t807,
                t808,
                t809,
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
        del t2148, t809, t808, t807, t806

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2155 = paddle._C_ops.relu(t2149)
        del t2149

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2156 = paddle._C_ops.conv2d(
            t2155, t810, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t810

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2157, t2158, t2159, t2160, t2161, t2162 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2156,
                t811,
                t812,
                t813,
                t814,
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
        del t2156, t814, t813, t812, t811

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2163 = paddle._C_ops.depthwise_conv2d(
            t2157, t815, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2157, t815

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2164, t2165, t2166, t2167, t2168, t2169 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2163,
                t816,
                t817,
                t818,
                t819,
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
        del t2163, t819, t818, t817, t816

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2170 = paddle._C_ops.relu(t2164)
        del t2164

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2171 = paddle._C_ops.conv2d(
            t2170, t820, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t820

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2172, t2173, t2174, t2175, t2176, t2177 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2171,
                t821,
                t822,
                t823,
                t824,
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
        del t2171, t824, t823, t822, t821

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2178 = paddle._C_ops.depthwise_conv2d(
            t2172, t825, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2172, t825

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2179, t2180, t2181, t2182, t2183, t2184 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2178,
                t826,
                t827,
                t828,
                t829,
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
        del t2178, t829, t828, t827, t826

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2185 = paddle._C_ops.relu(t2179)
        del t2179

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2186 = paddle._C_ops.conv2d(
            t2185, t830, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t830

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2187, t2188, t2189, t2190, t2191, t2192 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2186,
                t831,
                t832,
                t833,
                t834,
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
        del t2186, t834, t833, t832, t831

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2193 = paddle._C_ops.depthwise_conv2d(
            t2187, t835, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2187, t835

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2194, t2195, t2196, t2197, t2198, t2199 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2193,
                t836,
                t837,
                t838,
                t839,
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
        del t2193, t839, t838, t837, t836

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2200 = paddle._C_ops.relu(t2194)
        del t2194

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2201 = paddle._C_ops.conv2d(
            t2200, t840, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t840

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2202, t2203, t2204, t2205, t2206, t2207 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2201,
                t841,
                t842,
                t843,
                t844,
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
        del t2201, t844, t843, t842, t841

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2208 = paddle._C_ops.depthwise_conv2d(
            t2202, t845, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2202, t845

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2209, t2210, t2211, t2212, t2213, t2214 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2208,
                t846,
                t847,
                t848,
                t849,
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
        del t2208, t849, t848, t847, t846

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2215 = paddle._C_ops.relu(t2209)
        del t2209

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x768x1x1xf32)
        t2216 = paddle._C_ops.conv2d(
            t2215, t850, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t850

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2217, t2218, t2219, t2220, t2221, t2222 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2216,
                t851,
                t852,
                t853,
                t854,
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
        del t2216, t854, t853, t852, t851

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x5x5xf32)
        t2223 = paddle._C_ops.depthwise_conv2d(
            t2217, t855, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del t2217, t855

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        t2224, t2225, t2226, t2227, t2228, t2229 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2223,
                t856,
                t857,
                t858,
                t859,
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
        del t2223, t859, t858, t857, t856

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        t2230 = paddle._C_ops.relu(t2224)
        del t2224

        # builtin.combine: ([-1x2048x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32]) <- (-1x2048x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32)
        t2231 = [t2140, t2155, t2170, t2185, t2200, t2215, t2230]
        del t2155, t2170, t2185, t2200, t2215, t2230

        # pd_op.concat: (-1x6656x7x7xf32) <- ([-1x2048x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32, -1x768x7x7xf32], 1xi32)
        t2232 = paddle._C_ops.concat(t2231, t898)
        del t2231, t898

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x6656x7x7xf32, 1024x6656x1x1xf32)
        t2233 = paddle._C_ops.conv2d(
            t2232, t860, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t2232, t860

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2234, t2235, t2236, t2237, t2238, t2239 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2233,
                t861,
                t862,
                t863,
                t864,
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
        del t2233, t862, t861, t864, t863

        # pd_op.relu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        t2240 = paddle._C_ops.relu(t2234)
        del t2234

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        t2241 = paddle._C_ops.conv2d(
            t2240, t865, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t865, t2240

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t2242, t2243, t2244, t2245, t2246, t2247 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2241,
                t866,
                t867,
                t868,
                t869,
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
        del t2241, t869, t868, t867, t866

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t2248 = paddle._C_ops.relu(t2242)
        del t2242

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t2249 = paddle._C_ops.add(t2248, t2140)
        del t2140, t2248

        # pd_op.full_int_array: (2xi64) <- ()
        t2250 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        t2251 = paddle._C_ops.pool2d(
            t2249,
            t2250,
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
        del t2249, t2250

        # pd_op.conv2d: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32, 2048x2048x1x1xf32)
        t2252 = paddle._C_ops.conv2d(
            t2251, t870, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t870, t2251

        # pd_op.relu: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32)
        t2253 = paddle._C_ops.relu(t2252)
        del t2252

        # pd_op.flatten: (-1x2048xf32) <- (-1x2048x1x1xf32)
        t2254 = paddle._C_ops.flatten(t2253, 1, 3)
        del t2253

        # pd_op.matmul: (-1x102xf32) <- (-1x2048xf32, 2048x102xf32)
        t2255 = paddle._C_ops.matmul(t2254, t871, False, False)
        del t2254, t871

        return t2255
