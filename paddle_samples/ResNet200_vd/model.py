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
        t872: paddle.Tensor,
        t873: paddle.Tensor,
        t874: paddle.Tensor,
        t875: paddle.Tensor,
        t876: paddle.Tensor,
        t877: paddle.Tensor,
        t878: paddle.Tensor,
        t879: paddle.Tensor,
        t880: paddle.Tensor,
        t881: paddle.Tensor,
        t882: paddle.Tensor,
        t883: paddle.Tensor,
        t884: paddle.Tensor,
        t885: paddle.Tensor,
        t886: paddle.Tensor,
        t887: paddle.Tensor,
        t888: paddle.Tensor,
        t889: paddle.Tensor,
        t890: paddle.Tensor,
        t891: paddle.Tensor,
        t892: paddle.Tensor,
        t893: paddle.Tensor,
        t894: paddle.Tensor,
        t895: paddle.Tensor,
        t896: paddle.Tensor,
        t897: paddle.Tensor,
        t898: paddle.Tensor,
        t899: paddle.Tensor,
        t900: paddle.Tensor,
        t901: paddle.Tensor,
        t902: paddle.Tensor,
        t903: paddle.Tensor,
        t904: paddle.Tensor,
        t905: paddle.Tensor,
        t906: paddle.Tensor,
        t907: paddle.Tensor,
        t908: paddle.Tensor,
        t909: paddle.Tensor,
        t910: paddle.Tensor,
        t911: paddle.Tensor,
        t912: paddle.Tensor,
        t913: paddle.Tensor,
        t914: paddle.Tensor,
        t915: paddle.Tensor,
        t916: paddle.Tensor,
        t917: paddle.Tensor,
        t918: paddle.Tensor,
        t919: paddle.Tensor,
        t920: paddle.Tensor,
        t921: paddle.Tensor,
        t922: paddle.Tensor,
        t923: paddle.Tensor,
        t924: paddle.Tensor,
        t925: paddle.Tensor,
        t926: paddle.Tensor,
        t927: paddle.Tensor,
        t928: paddle.Tensor,
        t929: paddle.Tensor,
        t930: paddle.Tensor,
        t931: paddle.Tensor,
        t932: paddle.Tensor,
        t933: paddle.Tensor,
        t934: paddle.Tensor,
        t935: paddle.Tensor,
        t936: paddle.Tensor,
        t937: paddle.Tensor,
        t938: paddle.Tensor,
        t939: paddle.Tensor,
        t940: paddle.Tensor,
        t941: paddle.Tensor,
        t942: paddle.Tensor,
        t943: paddle.Tensor,
        t944: paddle.Tensor,
        t945: paddle.Tensor,
        t946: paddle.Tensor,
        t947: paddle.Tensor,
        t948: paddle.Tensor,
        t949: paddle.Tensor,
        t950: paddle.Tensor,
        t951: paddle.Tensor,
        t952: paddle.Tensor,
        t953: paddle.Tensor,
        t954: paddle.Tensor,
        t955: paddle.Tensor,
        t956: paddle.Tensor,
        t957: paddle.Tensor,
        t958: paddle.Tensor,
        t959: paddle.Tensor,
        t960: paddle.Tensor,
        t961: paddle.Tensor,
        t962: paddle.Tensor,
        t963: paddle.Tensor,
        t964: paddle.Tensor,
        t965: paddle.Tensor,
        t966: paddle.Tensor,
        t967: paddle.Tensor,
        t968: paddle.Tensor,
        t969: paddle.Tensor,
        t970: paddle.Tensor,
        t971: paddle.Tensor,
        t972: paddle.Tensor,
        t973: paddle.Tensor,
        t974: paddle.Tensor,
        t975: paddle.Tensor,
        t976: paddle.Tensor,
        t977: paddle.Tensor,
        t978: paddle.Tensor,
        t979: paddle.Tensor,
        t980: paddle.Tensor,
        t981: paddle.Tensor,
        t982: paddle.Tensor,
        t983: paddle.Tensor,
        t984: paddle.Tensor,
        t985: paddle.Tensor,
        t986: paddle.Tensor,
        t987: paddle.Tensor,
        t988: paddle.Tensor,
        t989: paddle.Tensor,
        t990: paddle.Tensor,
        t991: paddle.Tensor,
        t992: paddle.Tensor,
        t993: paddle.Tensor,
        t994: paddle.Tensor,
        t995: paddle.Tensor,
        t996: paddle.Tensor,
        t997: paddle.Tensor,
        t998: paddle.Tensor,
        t999: paddle.Tensor,
        t1000: paddle.Tensor,
        t1001: paddle.Tensor,
        t1002: paddle.Tensor,
        t1003: paddle.Tensor,
        t1004: paddle.Tensor,
        t1005: paddle.Tensor,
        t1006: paddle.Tensor,
        t1007: paddle.Tensor,
        t1008: paddle.Tensor,
        t1009: paddle.Tensor,
        t1010: paddle.Tensor,
        t1011: paddle.Tensor,
        t1012: paddle.Tensor,
        t1013: paddle.Tensor,
        t1014: paddle.Tensor,
        t1015: paddle.Tensor,
        t1016: paddle.Tensor,
        t1017: paddle.Tensor,
        t1018: paddle.Tensor,
        t1019: paddle.Tensor,
        t1020: paddle.Tensor,
        t1021: paddle.Tensor,
        t1022: paddle.Tensor,
        t1023: paddle.Tensor,
        t1024: paddle.Tensor,
        t1025: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        t1026 = paddle._C_ops.conv2d(
            input0, t0, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t1027, t1028, t1029, t1030, t1031, t1032 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1026,
                t1,
                t2,
                t3,
                t4,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1026, t4, t3, t2, t1

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t1033 = paddle._C_ops.relu(t1027)
        del t1027

        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 32x32x3x3xf32)
        t1034 = paddle._C_ops.conv2d(
            t1033, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5, t1033

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        t1035, t1036, t1037, t1038, t1039, t1040 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1034,
                t6,
                t7,
                t8,
                t9,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1034, t9, t8, t7, t6

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        t1041 = paddle._C_ops.relu(t1035)
        del t1035

        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x32x112x112xf32, 64x32x3x3xf32)
        t1042 = paddle._C_ops.conv2d(
            t1041, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10, t1041

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1043, t1044, t1045, t1046, t1047, t1048 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1042,
                t11,
                t12,
                t13,
                t14,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1042, t14, t13, t12, t11

        # pd_op.relu: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        t1049 = paddle._C_ops.relu(t1043)
        del t1043

        # pd_op.full_int_array: (2xi64) <- ()
        t1050 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        t1051 = paddle._C_ops.pool2d(
            t1049,
            t1050,
            [2, 2],
            [1, 1],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )
        del t1050, t1049

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        t1052 = paddle._C_ops.conv2d(
            t1051, t15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1053, t1054, t1055, t1056, t1057, t1058 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1052,
                t16,
                t17,
                t18,
                t19,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1052, t19, t18, t17, t16

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t1059 = paddle._C_ops.relu(t1053)
        del t1053

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t1060 = paddle._C_ops.conv2d(
            t1059, t20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t20, t1059

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1061, t1062, t1063, t1064, t1065, t1066 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1060,
                t21,
                t22,
                t23,
                t24,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1060, t24, t23, t22, t21

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t1067 = paddle._C_ops.relu(t1061)
        del t1061

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t1068 = paddle._C_ops.conv2d(
            t1067, t25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t25, t1067

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1069, t1070, t1071, t1072, t1073, t1074 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1068,
                t26,
                t27,
                t28,
                t29,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1068, t26, t29, t28, t27

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t1075 = paddle._C_ops.conv2d(
            t1051, t30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t30, t1051

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1076, t1077, t1078, t1079, t1080, t1081 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1075,
                t31,
                t32,
                t33,
                t34,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1075, t34, t33, t32, t31

        # pd_op.add: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        t1082 = paddle._C_ops.add(t1069, t1076)
        del t1069, t1076

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        t1083 = paddle._C_ops.relu(t1082)
        del t1082

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        t1084 = paddle._C_ops.conv2d(
            t1083, t35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t35

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1085, t1086, t1087, t1088, t1089, t1090 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1084,
                t36,
                t37,
                t38,
                t39,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1084, t39, t38, t37, t36

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t1091 = paddle._C_ops.relu(t1085)
        del t1085

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t1092 = paddle._C_ops.conv2d(
            t1091, t40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t40, t1091

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1093, t1094, t1095, t1096, t1097, t1098 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1092,
                t41,
                t42,
                t43,
                t44,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1092, t44, t43, t42, t41

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t1099 = paddle._C_ops.relu(t1093)
        del t1093

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t1100 = paddle._C_ops.conv2d(
            t1099, t45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t45, t1099

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1101, t1102, t1103, t1104, t1105, t1106 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1100,
                t46,
                t47,
                t48,
                t49,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1100, t49, t48, t47, t46

        # pd_op.add: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        t1107 = paddle._C_ops.add(t1101, t1083)
        del t1101, t1083

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        t1108 = paddle._C_ops.relu(t1107)
        del t1107

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        t1109 = paddle._C_ops.conv2d(
            t1108, t50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t50

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1110, t1111, t1112, t1113, t1114, t1115 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1109,
                t51,
                t52,
                t53,
                t54,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1109, t54, t53, t52, t51

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t1116 = paddle._C_ops.relu(t1110)
        del t1110

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        t1117 = paddle._C_ops.conv2d(
            t1116, t55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t55, t1116

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t1118, t1119, t1120, t1121, t1122, t1123 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1117,
                t56,
                t57,
                t58,
                t59,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1117, t59, t58, t57, t56

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        t1124 = paddle._C_ops.relu(t1118)
        del t1118

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        t1125 = paddle._C_ops.conv2d(
            t1124, t60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t60, t1124

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1126, t1127, t1128, t1129, t1130, t1131 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1125,
                t61,
                t62,
                t63,
                t64,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1125, t64, t63, t62, t61

        # pd_op.add: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        t1132 = paddle._C_ops.add(t1126, t1108)
        del t1126, t1108

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        t1133 = paddle._C_ops.relu(t1132)
        del t1132

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x256x56x56xf32, 128x256x1x1xf32)
        t1134 = paddle._C_ops.conv2d(
            t1133, t65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t65

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1135, t1136, t1137, t1138, t1139, t1140 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1134,
                t66,
                t67,
                t68,
                t69,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1134, t69, t68, t67, t66

        # pd_op.relu: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        t1141 = paddle._C_ops.relu(t1135)
        del t1135

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x56x56xf32, 128x128x3x3xf32)
        t1142 = paddle._C_ops.conv2d(
            t1141, t70, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t70, t1141

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1143, t1144, t1145, t1146, t1147, t1148 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1142,
                t71,
                t72,
                t73,
                t74,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1142, t74, t73, t72, t71

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1149 = paddle._C_ops.relu(t1143)
        del t1143

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1150 = paddle._C_ops.conv2d(
            t1149, t75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t75, t1149

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1151, t1152, t1153, t1154, t1155, t1156 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1150,
                t76,
                t77,
                t78,
                t79,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1150, t79, t78, t77, t76

        # pd_op.full_int_array: (2xi64) <- ()
        t1157 = [2, 2]

        # pd_op.pool2d: (-1x256x28x28xf32) <- (-1x256x56x56xf32, 2xi64)
        t1158 = paddle._C_ops.pool2d(
            t1133,
            t1157,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "SAME",
        )
        del t1133

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        t1159 = paddle._C_ops.conv2d(
            t1158, t80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t80, t1158

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1160, t1161, t1162, t1163, t1164, t1165 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1159,
                t81,
                t82,
                t83,
                t84,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1159, t84, t83, t82, t81

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1166 = paddle._C_ops.add(t1151, t1160)
        del t1151, t1160

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1167 = paddle._C_ops.relu(t1166)
        del t1166

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1168 = paddle._C_ops.conv2d(
            t1167, t85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t85

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1169, t1170, t1171, t1172, t1173, t1174 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1168,
                t86,
                t87,
                t88,
                t89,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1168, t89, t88, t87, t86

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1175 = paddle._C_ops.relu(t1169)
        del t1169

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1176 = paddle._C_ops.conv2d(
            t1175, t90, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t90, t1175

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1177, t1178, t1179, t1180, t1181, t1182 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1176,
                t91,
                t92,
                t93,
                t94,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1176, t94, t93, t92, t91

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1183 = paddle._C_ops.relu(t1177)
        del t1177

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1184 = paddle._C_ops.conv2d(
            t1183, t95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t95, t1183

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1185, t1186, t1187, t1188, t1189, t1190 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1184,
                t96,
                t97,
                t98,
                t99,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1184, t99, t98, t97, t96

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1191 = paddle._C_ops.add(t1185, t1167)
        del t1185, t1167

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1192 = paddle._C_ops.relu(t1191)
        del t1191

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1193 = paddle._C_ops.conv2d(
            t1192, t100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t100

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1194, t1195, t1196, t1197, t1198, t1199 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1193,
                t101,
                t102,
                t103,
                t104,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1193, t104, t103, t102, t101

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1200 = paddle._C_ops.relu(t1194)
        del t1194

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1201 = paddle._C_ops.conv2d(
            t1200, t105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t105, t1200

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1202, t1203, t1204, t1205, t1206, t1207 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1201,
                t106,
                t107,
                t108,
                t109,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1201, t109, t108, t107, t106

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1208 = paddle._C_ops.relu(t1202)
        del t1202

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1209 = paddle._C_ops.conv2d(
            t1208, t110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t110, t1208

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1210, t1211, t1212, t1213, t1214, t1215 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1209,
                t111,
                t112,
                t113,
                t114,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1209, t114, t113, t112, t111

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1216 = paddle._C_ops.add(t1210, t1192)
        del t1210, t1192

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1217 = paddle._C_ops.relu(t1216)
        del t1216

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1218 = paddle._C_ops.conv2d(
            t1217, t115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t115

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1219, t1220, t1221, t1222, t1223, t1224 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1218,
                t116,
                t117,
                t118,
                t119,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1218, t119, t118, t117, t116

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1225 = paddle._C_ops.relu(t1219)
        del t1219

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1226 = paddle._C_ops.conv2d(
            t1225, t120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t120, t1225

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1227, t1228, t1229, t1230, t1231, t1232 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1226,
                t121,
                t122,
                t123,
                t124,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1226, t124, t123, t122, t121

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1233 = paddle._C_ops.relu(t1227)
        del t1227

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1234 = paddle._C_ops.conv2d(
            t1233, t125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t125, t1233

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1235, t1236, t1237, t1238, t1239, t1240 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1234,
                t126,
                t127,
                t128,
                t129,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1234, t129, t128, t127, t126

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1241 = paddle._C_ops.add(t1235, t1217)
        del t1235, t1217

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1242 = paddle._C_ops.relu(t1241)
        del t1241

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1243 = paddle._C_ops.conv2d(
            t1242, t130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t130

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1244, t1245, t1246, t1247, t1248, t1249 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1243,
                t131,
                t132,
                t133,
                t134,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1243, t134, t133, t132, t131

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1250 = paddle._C_ops.relu(t1244)
        del t1244

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1251 = paddle._C_ops.conv2d(
            t1250, t135, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t135, t1250

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1252, t1253, t1254, t1255, t1256, t1257 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1251,
                t136,
                t137,
                t138,
                t139,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1251, t139, t138, t137, t136

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1258 = paddle._C_ops.relu(t1252)
        del t1252

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1259 = paddle._C_ops.conv2d(
            t1258, t140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t140, t1258

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1260, t1261, t1262, t1263, t1264, t1265 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1259,
                t141,
                t142,
                t143,
                t144,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1259, t144, t143, t142, t141

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1266 = paddle._C_ops.add(t1260, t1242)
        del t1260, t1242

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1267 = paddle._C_ops.relu(t1266)
        del t1266

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1268 = paddle._C_ops.conv2d(
            t1267, t145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t145

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1269, t1270, t1271, t1272, t1273, t1274 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1268,
                t146,
                t147,
                t148,
                t149,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1268, t149, t148, t147, t146

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1275 = paddle._C_ops.relu(t1269)
        del t1269

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1276 = paddle._C_ops.conv2d(
            t1275, t150, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t150, t1275

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1277, t1278, t1279, t1280, t1281, t1282 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1276,
                t151,
                t152,
                t153,
                t154,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1276, t154, t153, t152, t151

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1283 = paddle._C_ops.relu(t1277)
        del t1277

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1284 = paddle._C_ops.conv2d(
            t1283, t155, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t155, t1283

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1285, t1286, t1287, t1288, t1289, t1290 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1284,
                t156,
                t157,
                t158,
                t159,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1284, t159, t158, t157, t156

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1291 = paddle._C_ops.add(t1285, t1267)
        del t1285, t1267

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1292 = paddle._C_ops.relu(t1291)
        del t1291

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1293 = paddle._C_ops.conv2d(
            t1292, t160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t160

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1294, t1295, t1296, t1297, t1298, t1299 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1293,
                t161,
                t162,
                t163,
                t164,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1293, t164, t163, t162, t161

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1300 = paddle._C_ops.relu(t1294)
        del t1294

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1301 = paddle._C_ops.conv2d(
            t1300, t165, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t165, t1300

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1302, t1303, t1304, t1305, t1306, t1307 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1301,
                t166,
                t167,
                t168,
                t169,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1301, t169, t168, t167, t166

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1308 = paddle._C_ops.relu(t1302)
        del t1302

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1309 = paddle._C_ops.conv2d(
            t1308, t170, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t170, t1308

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1310, t1311, t1312, t1313, t1314, t1315 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1309,
                t171,
                t172,
                t173,
                t174,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1309, t174, t173, t172, t171

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1316 = paddle._C_ops.add(t1310, t1292)
        del t1310, t1292

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1317 = paddle._C_ops.relu(t1316)
        del t1316

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1318 = paddle._C_ops.conv2d(
            t1317, t175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t175

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1319, t1320, t1321, t1322, t1323, t1324 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1318,
                t176,
                t177,
                t178,
                t179,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1318, t179, t178, t177, t176

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1325 = paddle._C_ops.relu(t1319)
        del t1319

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1326 = paddle._C_ops.conv2d(
            t1325, t180, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t180, t1325

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1327, t1328, t1329, t1330, t1331, t1332 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1326,
                t181,
                t182,
                t183,
                t184,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1326, t184, t183, t182, t181

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1333 = paddle._C_ops.relu(t1327)
        del t1327

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1334 = paddle._C_ops.conv2d(
            t1333, t185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t185, t1333

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1335, t1336, t1337, t1338, t1339, t1340 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1334,
                t186,
                t187,
                t188,
                t189,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1334, t189, t188, t187, t186

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1341 = paddle._C_ops.add(t1335, t1317)
        del t1335, t1317

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1342 = paddle._C_ops.relu(t1341)
        del t1341

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1343 = paddle._C_ops.conv2d(
            t1342, t190, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t190

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1344, t1345, t1346, t1347, t1348, t1349 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1343,
                t191,
                t192,
                t193,
                t194,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1343, t194, t193, t192, t191

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1350 = paddle._C_ops.relu(t1344)
        del t1344

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1351 = paddle._C_ops.conv2d(
            t1350, t195, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t195, t1350

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1352, t1353, t1354, t1355, t1356, t1357 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1351,
                t196,
                t197,
                t198,
                t199,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1351, t199, t198, t197, t196

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1358 = paddle._C_ops.relu(t1352)
        del t1352

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1359 = paddle._C_ops.conv2d(
            t1358, t200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t200, t1358

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1360, t1361, t1362, t1363, t1364, t1365 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1359,
                t201,
                t202,
                t203,
                t204,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1359, t204, t203, t202, t201

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1366 = paddle._C_ops.add(t1360, t1342)
        del t1360, t1342

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1367 = paddle._C_ops.relu(t1366)
        del t1366

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1368 = paddle._C_ops.conv2d(
            t1367, t205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t205

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1369, t1370, t1371, t1372, t1373, t1374 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1368,
                t206,
                t207,
                t208,
                t209,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1368, t209, t208, t207, t206

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1375 = paddle._C_ops.relu(t1369)
        del t1369

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1376 = paddle._C_ops.conv2d(
            t1375, t210, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t210, t1375

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1377, t1378, t1379, t1380, t1381, t1382 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1376,
                t211,
                t212,
                t213,
                t214,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1376, t214, t213, t212, t211

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1383 = paddle._C_ops.relu(t1377)
        del t1377

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1384 = paddle._C_ops.conv2d(
            t1383, t215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t215, t1383

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1385, t1386, t1387, t1388, t1389, t1390 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1384,
                t216,
                t217,
                t218,
                t219,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1384, t219, t218, t217, t216

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1391 = paddle._C_ops.add(t1385, t1367)
        del t1385, t1367

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1392 = paddle._C_ops.relu(t1391)
        del t1391

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1393 = paddle._C_ops.conv2d(
            t1392, t220, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t220

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1394, t1395, t1396, t1397, t1398, t1399 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1393,
                t221,
                t222,
                t223,
                t224,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1393, t224, t223, t222, t221

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1400 = paddle._C_ops.relu(t1394)
        del t1394

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1401 = paddle._C_ops.conv2d(
            t1400, t225, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t225, t1400

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1402, t1403, t1404, t1405, t1406, t1407 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1401,
                t226,
                t227,
                t228,
                t229,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1401, t229, t228, t227, t226

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1408 = paddle._C_ops.relu(t1402)
        del t1402

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1409 = paddle._C_ops.conv2d(
            t1408, t230, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t230, t1408

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1410, t1411, t1412, t1413, t1414, t1415 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1409,
                t231,
                t232,
                t233,
                t234,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1409, t234, t233, t232, t231

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1416 = paddle._C_ops.add(t1410, t1392)
        del t1410, t1392

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1417 = paddle._C_ops.relu(t1416)
        del t1416

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        t1418 = paddle._C_ops.conv2d(
            t1417, t235, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t235

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1419, t1420, t1421, t1422, t1423, t1424 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1418,
                t236,
                t237,
                t238,
                t239,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1418, t239, t238, t237, t236

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1425 = paddle._C_ops.relu(t1419)
        del t1419

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        t1426 = paddle._C_ops.conv2d(
            t1425, t240, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t240, t1425

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        t1427, t1428, t1429, t1430, t1431, t1432 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1426,
                t241,
                t242,
                t243,
                t244,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1426, t244, t243, t242, t241

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        t1433 = paddle._C_ops.relu(t1427)
        del t1427

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        t1434 = paddle._C_ops.conv2d(
            t1433, t245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t245, t1433

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t1435, t1436, t1437, t1438, t1439, t1440 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1434,
                t246,
                t247,
                t248,
                t249,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1434, t249, t248, t247, t246

        # pd_op.add: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        t1441 = paddle._C_ops.add(t1435, t1417)
        del t1435, t1417

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        t1442 = paddle._C_ops.relu(t1441)
        del t1441

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x512x28x28xf32, 256x512x1x1xf32)
        t1443 = paddle._C_ops.conv2d(
            t1442, t250, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t250

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1444, t1445, t1446, t1447, t1448, t1449 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1443,
                t251,
                t252,
                t253,
                t254,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1443, t254, t253, t252, t251

        # pd_op.relu: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        t1450 = paddle._C_ops.relu(t1444)
        del t1444

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x28x28xf32, 256x256x3x3xf32)
        t1451 = paddle._C_ops.conv2d(
            t1450, t255, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t255, t1450

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1452, t1453, t1454, t1455, t1456, t1457 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1451,
                t256,
                t257,
                t258,
                t259,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1451, t259, t258, t257, t256

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1458 = paddle._C_ops.relu(t1452)
        del t1452

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1459 = paddle._C_ops.conv2d(
            t1458, t260, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t260, t1458

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1460, t1461, t1462, t1463, t1464, t1465 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1459,
                t261,
                t262,
                t263,
                t264,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1459, t264, t263, t262, t261

        # pd_op.pool2d: (-1x512x14x14xf32) <- (-1x512x28x28xf32, 2xi64)
        t1466 = paddle._C_ops.pool2d(
            t1442,
            t1157,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "SAME",
        )
        del t1442

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        t1467 = paddle._C_ops.conv2d(
            t1466, t265, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t265, t1466

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1468, t1469, t1470, t1471, t1472, t1473 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1467,
                t266,
                t267,
                t268,
                t269,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1467, t269, t268, t267, t266

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1474 = paddle._C_ops.add(t1460, t1468)
        del t1460, t1468

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1475 = paddle._C_ops.relu(t1474)
        del t1474

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1476 = paddle._C_ops.conv2d(
            t1475, t270, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t270

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1477, t1478, t1479, t1480, t1481, t1482 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1476,
                t271,
                t272,
                t273,
                t274,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1476, t274, t273, t272, t271

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1483 = paddle._C_ops.relu(t1477)
        del t1477

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1484 = paddle._C_ops.conv2d(
            t1483, t275, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t275, t1483

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1485, t1486, t1487, t1488, t1489, t1490 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1484,
                t276,
                t277,
                t278,
                t279,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1484, t279, t278, t277, t276

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1491 = paddle._C_ops.relu(t1485)
        del t1485

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1492 = paddle._C_ops.conv2d(
            t1491, t280, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t280, t1491

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1493, t1494, t1495, t1496, t1497, t1498 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1492,
                t281,
                t282,
                t283,
                t284,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1492, t284, t283, t282, t281

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1499 = paddle._C_ops.add(t1493, t1475)
        del t1493, t1475

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1500 = paddle._C_ops.relu(t1499)
        del t1499

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1501 = paddle._C_ops.conv2d(
            t1500, t285, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t285

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1502, t1503, t1504, t1505, t1506, t1507 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1501,
                t286,
                t287,
                t288,
                t289,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1501, t289, t288, t287, t286

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1508 = paddle._C_ops.relu(t1502)
        del t1502

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1509 = paddle._C_ops.conv2d(
            t1508, t290, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t290, t1508

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1510, t1511, t1512, t1513, t1514, t1515 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1509,
                t291,
                t292,
                t293,
                t294,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1509, t294, t293, t292, t291

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1516 = paddle._C_ops.relu(t1510)
        del t1510

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1517 = paddle._C_ops.conv2d(
            t1516, t295, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t295, t1516

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1518, t1519, t1520, t1521, t1522, t1523 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1517,
                t296,
                t297,
                t298,
                t299,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1517, t299, t298, t297, t296

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1524 = paddle._C_ops.add(t1518, t1500)
        del t1518, t1500

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1525 = paddle._C_ops.relu(t1524)
        del t1524

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1526 = paddle._C_ops.conv2d(
            t1525, t300, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t300

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1527, t1528, t1529, t1530, t1531, t1532 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1526,
                t301,
                t302,
                t303,
                t304,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1526, t304, t303, t302, t301

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1533 = paddle._C_ops.relu(t1527)
        del t1527

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1534 = paddle._C_ops.conv2d(
            t1533, t305, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t305, t1533

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1535, t1536, t1537, t1538, t1539, t1540 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1534,
                t306,
                t307,
                t308,
                t309,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1534, t309, t308, t307, t306

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1541 = paddle._C_ops.relu(t1535)
        del t1535

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1542 = paddle._C_ops.conv2d(
            t1541, t310, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t310, t1541

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1543, t1544, t1545, t1546, t1547, t1548 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1542,
                t311,
                t312,
                t313,
                t314,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1542, t314, t313, t312, t311

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1549 = paddle._C_ops.add(t1543, t1525)
        del t1543, t1525

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1550 = paddle._C_ops.relu(t1549)
        del t1549

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1551 = paddle._C_ops.conv2d(
            t1550, t315, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t315

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1552, t1553, t1554, t1555, t1556, t1557 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1551,
                t316,
                t317,
                t318,
                t319,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1551, t319, t318, t317, t316

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1558 = paddle._C_ops.relu(t1552)
        del t1552

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1559 = paddle._C_ops.conv2d(
            t1558, t320, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t320, t1558

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1560, t1561, t1562, t1563, t1564, t1565 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1559,
                t321,
                t322,
                t323,
                t324,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1559, t324, t323, t322, t321

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1566 = paddle._C_ops.relu(t1560)
        del t1560

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1567 = paddle._C_ops.conv2d(
            t1566, t325, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t325, t1566

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1568, t1569, t1570, t1571, t1572, t1573 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1567,
                t326,
                t327,
                t328,
                t329,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1567, t329, t328, t327, t326

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1574 = paddle._C_ops.add(t1568, t1550)
        del t1568, t1550

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1575 = paddle._C_ops.relu(t1574)
        del t1574

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1576 = paddle._C_ops.conv2d(
            t1575, t330, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t330

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1577, t1578, t1579, t1580, t1581, t1582 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1576,
                t331,
                t332,
                t333,
                t334,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1576, t334, t333, t332, t331

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1583 = paddle._C_ops.relu(t1577)
        del t1577

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1584 = paddle._C_ops.conv2d(
            t1583, t335, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t335, t1583

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1585, t1586, t1587, t1588, t1589, t1590 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1584,
                t336,
                t337,
                t338,
                t339,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1584, t339, t338, t337, t336

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1591 = paddle._C_ops.relu(t1585)
        del t1585

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1592 = paddle._C_ops.conv2d(
            t1591, t340, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t340, t1591

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1593, t1594, t1595, t1596, t1597, t1598 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1592,
                t341,
                t342,
                t343,
                t344,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1592, t344, t343, t342, t341

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1599 = paddle._C_ops.add(t1593, t1575)
        del t1593, t1575

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1600 = paddle._C_ops.relu(t1599)
        del t1599

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1601 = paddle._C_ops.conv2d(
            t1600, t345, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t345

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1602, t1603, t1604, t1605, t1606, t1607 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1601,
                t346,
                t347,
                t348,
                t349,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1601, t349, t348, t347, t346

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1608 = paddle._C_ops.relu(t1602)
        del t1602

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1609 = paddle._C_ops.conv2d(
            t1608, t350, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t350, t1608

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1610, t1611, t1612, t1613, t1614, t1615 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1609,
                t351,
                t352,
                t353,
                t354,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1609, t354, t353, t352, t351

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1616 = paddle._C_ops.relu(t1610)
        del t1610

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1617 = paddle._C_ops.conv2d(
            t1616, t355, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t355, t1616

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1618, t1619, t1620, t1621, t1622, t1623 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1617,
                t356,
                t357,
                t358,
                t359,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1617, t359, t358, t357, t356

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1624 = paddle._C_ops.add(t1618, t1600)
        del t1618, t1600

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1625 = paddle._C_ops.relu(t1624)
        del t1624

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1626 = paddle._C_ops.conv2d(
            t1625, t360, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t360

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1627, t1628, t1629, t1630, t1631, t1632 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1626,
                t361,
                t362,
                t363,
                t364,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1626, t364, t363, t362, t361

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1633 = paddle._C_ops.relu(t1627)
        del t1627

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1634 = paddle._C_ops.conv2d(
            t1633, t365, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t365, t1633

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1635, t1636, t1637, t1638, t1639, t1640 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1634,
                t366,
                t367,
                t368,
                t369,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1634, t369, t368, t367, t366

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1641 = paddle._C_ops.relu(t1635)
        del t1635

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1642 = paddle._C_ops.conv2d(
            t1641, t370, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t370, t1641

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1643, t1644, t1645, t1646, t1647, t1648 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1642,
                t371,
                t372,
                t373,
                t374,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1642, t374, t373, t372, t371

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1649 = paddle._C_ops.add(t1643, t1625)
        del t1643, t1625

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1650 = paddle._C_ops.relu(t1649)
        del t1649

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1651 = paddle._C_ops.conv2d(
            t1650, t375, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t375

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1652, t1653, t1654, t1655, t1656, t1657 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1651,
                t376,
                t377,
                t378,
                t379,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1651, t379, t378, t377, t376

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1658 = paddle._C_ops.relu(t1652)
        del t1652

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1659 = paddle._C_ops.conv2d(
            t1658, t380, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t380, t1658

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1660, t1661, t1662, t1663, t1664, t1665 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1659,
                t381,
                t382,
                t383,
                t384,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1659, t384, t383, t382, t381

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1666 = paddle._C_ops.relu(t1660)
        del t1660

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1667 = paddle._C_ops.conv2d(
            t1666, t385, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t385, t1666

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1668, t1669, t1670, t1671, t1672, t1673 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1667,
                t386,
                t387,
                t388,
                t389,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1667, t389, t388, t387, t386

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1674 = paddle._C_ops.add(t1668, t1650)
        del t1668, t1650

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1675 = paddle._C_ops.relu(t1674)
        del t1674

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1676 = paddle._C_ops.conv2d(
            t1675, t390, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t390

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1677, t1678, t1679, t1680, t1681, t1682 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1676,
                t391,
                t392,
                t393,
                t394,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1676, t394, t393, t392, t391

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1683 = paddle._C_ops.relu(t1677)
        del t1677

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1684 = paddle._C_ops.conv2d(
            t1683, t395, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t395, t1683

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1685, t1686, t1687, t1688, t1689, t1690 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1684,
                t396,
                t397,
                t398,
                t399,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1684, t399, t398, t397, t396

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1691 = paddle._C_ops.relu(t1685)
        del t1685

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1692 = paddle._C_ops.conv2d(
            t1691, t400, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t400, t1691

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1693, t1694, t1695, t1696, t1697, t1698 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1692,
                t401,
                t402,
                t403,
                t404,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1692, t404, t403, t402, t401

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1699 = paddle._C_ops.add(t1693, t1675)
        del t1693, t1675

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1700 = paddle._C_ops.relu(t1699)
        del t1699

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1701 = paddle._C_ops.conv2d(
            t1700, t405, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t405

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1702, t1703, t1704, t1705, t1706, t1707 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1701,
                t406,
                t407,
                t408,
                t409,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1701, t409, t408, t407, t406

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1708 = paddle._C_ops.relu(t1702)
        del t1702

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1709 = paddle._C_ops.conv2d(
            t1708, t410, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t410, t1708

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1710, t1711, t1712, t1713, t1714, t1715 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1709,
                t411,
                t412,
                t413,
                t414,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1709, t414, t413, t412, t411

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1716 = paddle._C_ops.relu(t1710)
        del t1710

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1717 = paddle._C_ops.conv2d(
            t1716, t415, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t415, t1716

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1718, t1719, t1720, t1721, t1722, t1723 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1717,
                t416,
                t417,
                t418,
                t419,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1717, t419, t418, t417, t416

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1724 = paddle._C_ops.add(t1718, t1700)
        del t1718, t1700

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1725 = paddle._C_ops.relu(t1724)
        del t1724

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1726 = paddle._C_ops.conv2d(
            t1725, t420, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t420

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1727, t1728, t1729, t1730, t1731, t1732 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1726,
                t421,
                t422,
                t423,
                t424,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1726, t424, t423, t422, t421

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1733 = paddle._C_ops.relu(t1727)
        del t1727

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1734 = paddle._C_ops.conv2d(
            t1733, t425, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t425, t1733

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1735, t1736, t1737, t1738, t1739, t1740 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1734,
                t426,
                t427,
                t428,
                t429,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1734, t429, t428, t427, t426

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1741 = paddle._C_ops.relu(t1735)
        del t1735

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1742 = paddle._C_ops.conv2d(
            t1741, t430, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t430, t1741

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1743, t1744, t1745, t1746, t1747, t1748 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1742,
                t431,
                t432,
                t433,
                t434,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1742, t434, t433, t432, t431

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1749 = paddle._C_ops.add(t1743, t1725)
        del t1743, t1725

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1750 = paddle._C_ops.relu(t1749)
        del t1749

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1751 = paddle._C_ops.conv2d(
            t1750, t435, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t435

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1752, t1753, t1754, t1755, t1756, t1757 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1751,
                t436,
                t437,
                t438,
                t439,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1751, t439, t438, t437, t436

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1758 = paddle._C_ops.relu(t1752)
        del t1752

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1759 = paddle._C_ops.conv2d(
            t1758, t440, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t440, t1758

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1760, t1761, t1762, t1763, t1764, t1765 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1759,
                t441,
                t442,
                t443,
                t444,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1759, t444, t443, t442, t441

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1766 = paddle._C_ops.relu(t1760)
        del t1760

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1767 = paddle._C_ops.conv2d(
            t1766, t445, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t445, t1766

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1768, t1769, t1770, t1771, t1772, t1773 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1767,
                t446,
                t447,
                t448,
                t449,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1767, t449, t448, t447, t446

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1774 = paddle._C_ops.add(t1768, t1750)
        del t1768, t1750

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1775 = paddle._C_ops.relu(t1774)
        del t1774

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1776 = paddle._C_ops.conv2d(
            t1775, t450, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t450

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1777, t1778, t1779, t1780, t1781, t1782 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1776,
                t451,
                t452,
                t453,
                t454,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1776, t454, t453, t452, t451

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1783 = paddle._C_ops.relu(t1777)
        del t1777

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1784 = paddle._C_ops.conv2d(
            t1783, t455, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t455, t1783

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1785, t1786, t1787, t1788, t1789, t1790 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1784,
                t456,
                t457,
                t458,
                t459,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1784, t459, t458, t457, t456

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1791 = paddle._C_ops.relu(t1785)
        del t1785

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1792 = paddle._C_ops.conv2d(
            t1791, t460, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t460, t1791

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1793, t1794, t1795, t1796, t1797, t1798 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1792,
                t461,
                t462,
                t463,
                t464,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1792, t464, t463, t462, t461

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1799 = paddle._C_ops.add(t1793, t1775)
        del t1793, t1775

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1800 = paddle._C_ops.relu(t1799)
        del t1799

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1801 = paddle._C_ops.conv2d(
            t1800, t465, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t465

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1802, t1803, t1804, t1805, t1806, t1807 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1801,
                t466,
                t467,
                t468,
                t469,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1801, t469, t468, t467, t466

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1808 = paddle._C_ops.relu(t1802)
        del t1802

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1809 = paddle._C_ops.conv2d(
            t1808, t470, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t470, t1808

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1810, t1811, t1812, t1813, t1814, t1815 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1809,
                t471,
                t472,
                t473,
                t474,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1809, t474, t473, t472, t471

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1816 = paddle._C_ops.relu(t1810)
        del t1810

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1817 = paddle._C_ops.conv2d(
            t1816, t475, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t475, t1816

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1818, t1819, t1820, t1821, t1822, t1823 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1817,
                t476,
                t477,
                t478,
                t479,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1817, t479, t478, t477, t476

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1824 = paddle._C_ops.add(t1818, t1800)
        del t1818, t1800

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1825 = paddle._C_ops.relu(t1824)
        del t1824

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1826 = paddle._C_ops.conv2d(
            t1825, t480, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t480

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1827, t1828, t1829, t1830, t1831, t1832 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1826,
                t481,
                t482,
                t483,
                t484,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1826, t484, t483, t482, t481

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1833 = paddle._C_ops.relu(t1827)
        del t1827

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1834 = paddle._C_ops.conv2d(
            t1833, t485, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t485, t1833

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1835, t1836, t1837, t1838, t1839, t1840 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1834,
                t486,
                t487,
                t488,
                t489,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1834, t489, t488, t487, t486

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1841 = paddle._C_ops.relu(t1835)
        del t1835

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1842 = paddle._C_ops.conv2d(
            t1841, t490, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t490, t1841

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1843, t1844, t1845, t1846, t1847, t1848 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1842,
                t491,
                t492,
                t493,
                t494,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1842, t494, t493, t492, t491

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1849 = paddle._C_ops.add(t1843, t1825)
        del t1843, t1825

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1850 = paddle._C_ops.relu(t1849)
        del t1849

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1851 = paddle._C_ops.conv2d(
            t1850, t495, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t495

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1852, t1853, t1854, t1855, t1856, t1857 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1851,
                t496,
                t497,
                t498,
                t499,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1851, t499, t498, t497, t496

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1858 = paddle._C_ops.relu(t1852)
        del t1852

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1859 = paddle._C_ops.conv2d(
            t1858, t500, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t500, t1858

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1860, t1861, t1862, t1863, t1864, t1865 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1859,
                t501,
                t502,
                t503,
                t504,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1859, t504, t503, t502, t501

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1866 = paddle._C_ops.relu(t1860)
        del t1860

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1867 = paddle._C_ops.conv2d(
            t1866, t505, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t505, t1866

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1868, t1869, t1870, t1871, t1872, t1873 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1867,
                t506,
                t507,
                t508,
                t509,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1867, t509, t508, t507, t506

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1874 = paddle._C_ops.add(t1868, t1850)
        del t1868, t1850

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1875 = paddle._C_ops.relu(t1874)
        del t1874

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1876 = paddle._C_ops.conv2d(
            t1875, t510, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t510

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1877, t1878, t1879, t1880, t1881, t1882 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1876,
                t511,
                t512,
                t513,
                t514,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1876, t514, t513, t512, t511

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1883 = paddle._C_ops.relu(t1877)
        del t1877

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1884 = paddle._C_ops.conv2d(
            t1883, t515, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t515, t1883

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1885, t1886, t1887, t1888, t1889, t1890 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1884,
                t516,
                t517,
                t518,
                t519,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1884, t519, t518, t517, t516

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1891 = paddle._C_ops.relu(t1885)
        del t1885

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1892 = paddle._C_ops.conv2d(
            t1891, t520, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t520, t1891

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1893, t1894, t1895, t1896, t1897, t1898 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1892,
                t521,
                t522,
                t523,
                t524,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1892, t524, t523, t522, t521

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1899 = paddle._C_ops.add(t1893, t1875)
        del t1893, t1875

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1900 = paddle._C_ops.relu(t1899)
        del t1899

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1901 = paddle._C_ops.conv2d(
            t1900, t525, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t525

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1902, t1903, t1904, t1905, t1906, t1907 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1901,
                t526,
                t527,
                t528,
                t529,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1901, t529, t528, t527, t526

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1908 = paddle._C_ops.relu(t1902)
        del t1902

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1909 = paddle._C_ops.conv2d(
            t1908, t530, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t530, t1908

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1910, t1911, t1912, t1913, t1914, t1915 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1909,
                t531,
                t532,
                t533,
                t534,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1909, t534, t533, t532, t531

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1916 = paddle._C_ops.relu(t1910)
        del t1910

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1917 = paddle._C_ops.conv2d(
            t1916, t535, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t535, t1916

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1918, t1919, t1920, t1921, t1922, t1923 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1917,
                t536,
                t537,
                t538,
                t539,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1917, t539, t538, t537, t536

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1924 = paddle._C_ops.add(t1918, t1900)
        del t1918, t1900

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1925 = paddle._C_ops.relu(t1924)
        del t1924

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1926 = paddle._C_ops.conv2d(
            t1925, t540, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t540

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1927, t1928, t1929, t1930, t1931, t1932 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1926,
                t541,
                t542,
                t543,
                t544,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1926, t544, t543, t542, t541

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1933 = paddle._C_ops.relu(t1927)
        del t1927

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1934 = paddle._C_ops.conv2d(
            t1933, t545, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t545, t1933

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1935, t1936, t1937, t1938, t1939, t1940 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1934,
                t546,
                t547,
                t548,
                t549,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1934, t549, t548, t547, t546

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1941 = paddle._C_ops.relu(t1935)
        del t1935

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1942 = paddle._C_ops.conv2d(
            t1941, t550, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t550, t1941

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1943, t1944, t1945, t1946, t1947, t1948 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1942,
                t551,
                t552,
                t553,
                t554,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1942, t554, t553, t552, t551

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1949 = paddle._C_ops.add(t1943, t1925)
        del t1943, t1925

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1950 = paddle._C_ops.relu(t1949)
        del t1949

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1951 = paddle._C_ops.conv2d(
            t1950, t555, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t555

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1952, t1953, t1954, t1955, t1956, t1957 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1951,
                t556,
                t557,
                t558,
                t559,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1951, t559, t558, t557, t556

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1958 = paddle._C_ops.relu(t1952)
        del t1952

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1959 = paddle._C_ops.conv2d(
            t1958, t560, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t560, t1958

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1960, t1961, t1962, t1963, t1964, t1965 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1959,
                t561,
                t562,
                t563,
                t564,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1959, t564, t563, t562, t561

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1966 = paddle._C_ops.relu(t1960)
        del t1960

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1967 = paddle._C_ops.conv2d(
            t1966, t565, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t565, t1966

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1968, t1969, t1970, t1971, t1972, t1973 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1967,
                t566,
                t567,
                t568,
                t569,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1967, t569, t568, t567, t566

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1974 = paddle._C_ops.add(t1968, t1950)
        del t1968, t1950

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t1975 = paddle._C_ops.relu(t1974)
        del t1974

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t1976 = paddle._C_ops.conv2d(
            t1975, t570, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t570

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1977, t1978, t1979, t1980, t1981, t1982 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1976,
                t571,
                t572,
                t573,
                t574,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1976, t574, t573, t572, t571

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1983 = paddle._C_ops.relu(t1977)
        del t1977

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t1984 = paddle._C_ops.conv2d(
            t1983, t575, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t575, t1983

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t1985, t1986, t1987, t1988, t1989, t1990 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1984,
                t576,
                t577,
                t578,
                t579,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1984, t579, t578, t577, t576

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t1991 = paddle._C_ops.relu(t1985)
        del t1985

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t1992 = paddle._C_ops.conv2d(
            t1991, t580, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t580, t1991

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t1993, t1994, t1995, t1996, t1997, t1998 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t1992,
                t581,
                t582,
                t583,
                t584,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t1992, t584, t583, t582, t581

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t1999 = paddle._C_ops.add(t1993, t1975)
        del t1993, t1975

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2000 = paddle._C_ops.relu(t1999)
        del t1999

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2001 = paddle._C_ops.conv2d(
            t2000, t585, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t585

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2002, t2003, t2004, t2005, t2006, t2007 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2001,
                t586,
                t587,
                t588,
                t589,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2001, t589, t588, t587, t586

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2008 = paddle._C_ops.relu(t2002)
        del t2002

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2009 = paddle._C_ops.conv2d(
            t2008, t590, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t590, t2008

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2010, t2011, t2012, t2013, t2014, t2015 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2009,
                t591,
                t592,
                t593,
                t594,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2009, t594, t593, t592, t591

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2016 = paddle._C_ops.relu(t2010)
        del t2010

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2017 = paddle._C_ops.conv2d(
            t2016, t595, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t595, t2016

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2018, t2019, t2020, t2021, t2022, t2023 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2017,
                t596,
                t597,
                t598,
                t599,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2017, t599, t598, t597, t596

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2024 = paddle._C_ops.add(t2018, t2000)
        del t2018, t2000

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2025 = paddle._C_ops.relu(t2024)
        del t2024

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2026 = paddle._C_ops.conv2d(
            t2025, t600, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t600

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2027, t2028, t2029, t2030, t2031, t2032 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2026,
                t601,
                t602,
                t603,
                t604,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2026, t604, t603, t602, t601

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2033 = paddle._C_ops.relu(t2027)
        del t2027

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2034 = paddle._C_ops.conv2d(
            t2033, t605, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t605, t2033

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2035, t2036, t2037, t2038, t2039, t2040 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2034,
                t606,
                t607,
                t608,
                t609,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2034, t609, t608, t607, t606

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2041 = paddle._C_ops.relu(t2035)
        del t2035

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2042 = paddle._C_ops.conv2d(
            t2041, t610, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t610, t2041

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2043, t2044, t2045, t2046, t2047, t2048 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2042,
                t611,
                t612,
                t613,
                t614,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2042, t614, t613, t612, t611

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2049 = paddle._C_ops.add(t2043, t2025)
        del t2043, t2025

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2050 = paddle._C_ops.relu(t2049)
        del t2049

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2051 = paddle._C_ops.conv2d(
            t2050, t615, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t615

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2052, t2053, t2054, t2055, t2056, t2057 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2051,
                t616,
                t617,
                t618,
                t619,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2051, t619, t618, t617, t616

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2058 = paddle._C_ops.relu(t2052)
        del t2052

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2059 = paddle._C_ops.conv2d(
            t2058, t620, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t620, t2058

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2060, t2061, t2062, t2063, t2064, t2065 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2059,
                t621,
                t622,
                t623,
                t624,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2059, t624, t623, t622, t621

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2066 = paddle._C_ops.relu(t2060)
        del t2060

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2067 = paddle._C_ops.conv2d(
            t2066, t625, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t625, t2066

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2068, t2069, t2070, t2071, t2072, t2073 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2067,
                t626,
                t627,
                t628,
                t629,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2067, t629, t628, t627, t626

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2074 = paddle._C_ops.add(t2068, t2050)
        del t2068, t2050

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2075 = paddle._C_ops.relu(t2074)
        del t2074

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2076 = paddle._C_ops.conv2d(
            t2075, t630, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t630

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2077, t2078, t2079, t2080, t2081, t2082 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2076,
                t631,
                t632,
                t633,
                t634,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2076, t634, t633, t632, t631

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2083 = paddle._C_ops.relu(t2077)
        del t2077

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2084 = paddle._C_ops.conv2d(
            t2083, t635, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t635, t2083

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2085, t2086, t2087, t2088, t2089, t2090 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2084,
                t636,
                t637,
                t638,
                t639,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2084, t639, t638, t637, t636

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2091 = paddle._C_ops.relu(t2085)
        del t2085

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2092 = paddle._C_ops.conv2d(
            t2091, t640, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t640, t2091

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2093, t2094, t2095, t2096, t2097, t2098 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2092,
                t641,
                t642,
                t643,
                t644,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2092, t644, t643, t642, t641

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2099 = paddle._C_ops.add(t2093, t2075)
        del t2093, t2075

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2100 = paddle._C_ops.relu(t2099)
        del t2099

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2101 = paddle._C_ops.conv2d(
            t2100, t645, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t645

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2102, t2103, t2104, t2105, t2106, t2107 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2101,
                t646,
                t647,
                t648,
                t649,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2101, t649, t648, t647, t646

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2108 = paddle._C_ops.relu(t2102)
        del t2102

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2109 = paddle._C_ops.conv2d(
            t2108, t650, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t650, t2108

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2110, t2111, t2112, t2113, t2114, t2115 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2109,
                t651,
                t652,
                t653,
                t654,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2109, t654, t653, t652, t651

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2116 = paddle._C_ops.relu(t2110)
        del t2110

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2117 = paddle._C_ops.conv2d(
            t2116, t655, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t655, t2116

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2118, t2119, t2120, t2121, t2122, t2123 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2117,
                t656,
                t657,
                t658,
                t659,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2117, t659, t658, t657, t656

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2124 = paddle._C_ops.add(t2118, t2100)
        del t2118, t2100

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2125 = paddle._C_ops.relu(t2124)
        del t2124

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2126 = paddle._C_ops.conv2d(
            t2125, t660, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t660

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2127, t2128, t2129, t2130, t2131, t2132 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2126,
                t661,
                t662,
                t663,
                t664,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2126, t664, t663, t662, t661

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2133 = paddle._C_ops.relu(t2127)
        del t2127

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2134 = paddle._C_ops.conv2d(
            t2133, t665, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t665, t2133

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2135, t2136, t2137, t2138, t2139, t2140 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2134,
                t666,
                t667,
                t668,
                t669,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2134, t669, t668, t667, t666

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2141 = paddle._C_ops.relu(t2135)
        del t2135

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2142 = paddle._C_ops.conv2d(
            t2141, t670, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t670, t2141

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2143, t2144, t2145, t2146, t2147, t2148 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2142,
                t671,
                t672,
                t673,
                t674,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2142, t674, t673, t672, t671

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2149 = paddle._C_ops.add(t2143, t2125)
        del t2143, t2125

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2150 = paddle._C_ops.relu(t2149)
        del t2149

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2151 = paddle._C_ops.conv2d(
            t2150, t675, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t675

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2152, t2153, t2154, t2155, t2156, t2157 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2151,
                t676,
                t677,
                t678,
                t679,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2151, t679, t678, t677, t676

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2158 = paddle._C_ops.relu(t2152)
        del t2152

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2159 = paddle._C_ops.conv2d(
            t2158, t680, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t680, t2158

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2160, t2161, t2162, t2163, t2164, t2165 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2159,
                t681,
                t682,
                t683,
                t684,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2159, t684, t683, t682, t681

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2166 = paddle._C_ops.relu(t2160)
        del t2160

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2167 = paddle._C_ops.conv2d(
            t2166, t685, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t685, t2166

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2168, t2169, t2170, t2171, t2172, t2173 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2167,
                t686,
                t687,
                t688,
                t689,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2167, t689, t688, t687, t686

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2174 = paddle._C_ops.add(t2168, t2150)
        del t2168, t2150

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2175 = paddle._C_ops.relu(t2174)
        del t2174

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2176 = paddle._C_ops.conv2d(
            t2175, t690, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t690

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2177, t2178, t2179, t2180, t2181, t2182 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2176,
                t691,
                t692,
                t693,
                t694,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2176, t694, t693, t692, t691

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2183 = paddle._C_ops.relu(t2177)
        del t2177

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2184 = paddle._C_ops.conv2d(
            t2183, t695, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t695, t2183

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2185, t2186, t2187, t2188, t2189, t2190 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2184,
                t696,
                t697,
                t698,
                t699,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2184, t699, t698, t697, t696

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2191 = paddle._C_ops.relu(t2185)
        del t2185

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2192 = paddle._C_ops.conv2d(
            t2191, t700, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t700, t2191

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2193, t2194, t2195, t2196, t2197, t2198 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2192,
                t701,
                t702,
                t703,
                t704,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2192, t704, t703, t702, t701

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2199 = paddle._C_ops.add(t2193, t2175)
        del t2193, t2175

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2200 = paddle._C_ops.relu(t2199)
        del t2199

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2201 = paddle._C_ops.conv2d(
            t2200, t705, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t705

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2202, t2203, t2204, t2205, t2206, t2207 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2201,
                t706,
                t707,
                t708,
                t709,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2201, t709, t708, t707, t706

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2208 = paddle._C_ops.relu(t2202)
        del t2202

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2209 = paddle._C_ops.conv2d(
            t2208, t710, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t710, t2208

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2210, t2211, t2212, t2213, t2214, t2215 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2209,
                t711,
                t712,
                t713,
                t714,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2209, t714, t713, t712, t711

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2216 = paddle._C_ops.relu(t2210)
        del t2210

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2217 = paddle._C_ops.conv2d(
            t2216, t715, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t715, t2216

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2218, t2219, t2220, t2221, t2222, t2223 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2217,
                t716,
                t717,
                t718,
                t719,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2217, t719, t718, t717, t716

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2224 = paddle._C_ops.add(t2218, t2200)
        del t2218, t2200

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2225 = paddle._C_ops.relu(t2224)
        del t2224

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2226 = paddle._C_ops.conv2d(
            t2225, t720, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t720

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2227, t2228, t2229, t2230, t2231, t2232 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2226,
                t721,
                t722,
                t723,
                t724,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2226, t724, t723, t722, t721

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2233 = paddle._C_ops.relu(t2227)
        del t2227

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2234 = paddle._C_ops.conv2d(
            t2233, t725, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t725, t2233

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2235, t2236, t2237, t2238, t2239, t2240 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2234,
                t726,
                t727,
                t728,
                t729,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2234, t729, t728, t727, t726

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2241 = paddle._C_ops.relu(t2235)
        del t2235

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2242 = paddle._C_ops.conv2d(
            t2241, t730, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t730, t2241

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2243, t2244, t2245, t2246, t2247, t2248 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2242,
                t731,
                t732,
                t733,
                t734,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2242, t734, t733, t732, t731

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2249 = paddle._C_ops.add(t2243, t2225)
        del t2243, t2225

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2250 = paddle._C_ops.relu(t2249)
        del t2249

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2251 = paddle._C_ops.conv2d(
            t2250, t735, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t735

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2252, t2253, t2254, t2255, t2256, t2257 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2251,
                t736,
                t737,
                t738,
                t739,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2251, t739, t738, t737, t736

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2258 = paddle._C_ops.relu(t2252)
        del t2252

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2259 = paddle._C_ops.conv2d(
            t2258, t740, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t740, t2258

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2260, t2261, t2262, t2263, t2264, t2265 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2259,
                t741,
                t742,
                t743,
                t744,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2259, t744, t743, t742, t741

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2266 = paddle._C_ops.relu(t2260)
        del t2260

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2267 = paddle._C_ops.conv2d(
            t2266, t745, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t745, t2266

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2268, t2269, t2270, t2271, t2272, t2273 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2267,
                t746,
                t747,
                t748,
                t749,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2267, t749, t748, t747, t746

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2274 = paddle._C_ops.add(t2268, t2250)
        del t2268, t2250

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2275 = paddle._C_ops.relu(t2274)
        del t2274

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2276 = paddle._C_ops.conv2d(
            t2275, t750, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t750

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2277, t2278, t2279, t2280, t2281, t2282 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2276,
                t751,
                t752,
                t753,
                t754,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2276, t754, t753, t752, t751

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2283 = paddle._C_ops.relu(t2277)
        del t2277

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2284 = paddle._C_ops.conv2d(
            t2283, t755, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t755, t2283

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2285, t2286, t2287, t2288, t2289, t2290 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2284,
                t756,
                t757,
                t758,
                t759,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2284, t759, t758, t757, t756

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2291 = paddle._C_ops.relu(t2285)
        del t2285

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2292 = paddle._C_ops.conv2d(
            t2291, t760, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t760, t2291

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2293, t2294, t2295, t2296, t2297, t2298 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2292,
                t761,
                t762,
                t763,
                t764,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2292, t764, t763, t762, t761

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2299 = paddle._C_ops.add(t2293, t2275)
        del t2293, t2275

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2300 = paddle._C_ops.relu(t2299)
        del t2299

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2301 = paddle._C_ops.conv2d(
            t2300, t765, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t765

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2302, t2303, t2304, t2305, t2306, t2307 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2301,
                t766,
                t767,
                t768,
                t769,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2301, t769, t768, t767, t766

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2308 = paddle._C_ops.relu(t2302)
        del t2302

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2309 = paddle._C_ops.conv2d(
            t2308, t770, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t770, t2308

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2310, t2311, t2312, t2313, t2314, t2315 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2309,
                t771,
                t772,
                t773,
                t774,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2309, t774, t773, t772, t771

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2316 = paddle._C_ops.relu(t2310)
        del t2310

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2317 = paddle._C_ops.conv2d(
            t2316, t775, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t775, t2316

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2318, t2319, t2320, t2321, t2322, t2323 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2317,
                t776,
                t777,
                t778,
                t779,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2317, t779, t778, t777, t776

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2324 = paddle._C_ops.add(t2318, t2300)
        del t2318, t2300

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2325 = paddle._C_ops.relu(t2324)
        del t2324

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2326 = paddle._C_ops.conv2d(
            t2325, t780, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t780

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2327, t2328, t2329, t2330, t2331, t2332 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2326,
                t781,
                t782,
                t783,
                t784,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2326, t784, t783, t782, t781

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2333 = paddle._C_ops.relu(t2327)
        del t2327

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2334 = paddle._C_ops.conv2d(
            t2333, t785, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t785, t2333

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2335, t2336, t2337, t2338, t2339, t2340 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2334,
                t786,
                t787,
                t788,
                t789,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2334, t789, t788, t787, t786

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2341 = paddle._C_ops.relu(t2335)
        del t2335

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2342 = paddle._C_ops.conv2d(
            t2341, t790, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t790, t2341

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2343, t2344, t2345, t2346, t2347, t2348 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2342,
                t791,
                t792,
                t793,
                t794,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2342, t794, t793, t792, t791

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2349 = paddle._C_ops.add(t2343, t2325)
        del t2343, t2325

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2350 = paddle._C_ops.relu(t2349)
        del t2349

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2351 = paddle._C_ops.conv2d(
            t2350, t795, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t795

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2352, t2353, t2354, t2355, t2356, t2357 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2351,
                t796,
                t797,
                t798,
                t799,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2351, t799, t798, t797, t796

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2358 = paddle._C_ops.relu(t2352)
        del t2352

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2359 = paddle._C_ops.conv2d(
            t2358, t800, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t800, t2358

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2360, t2361, t2362, t2363, t2364, t2365 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2359,
                t801,
                t802,
                t803,
                t804,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2359, t804, t803, t802, t801

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2366 = paddle._C_ops.relu(t2360)
        del t2360

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2367 = paddle._C_ops.conv2d(
            t2366, t805, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t805, t2366

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2368, t2369, t2370, t2371, t2372, t2373 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2367,
                t806,
                t807,
                t808,
                t809,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2367, t809, t808, t807, t806

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2374 = paddle._C_ops.add(t2368, t2350)
        del t2368, t2350

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2375 = paddle._C_ops.relu(t2374)
        del t2374

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2376 = paddle._C_ops.conv2d(
            t2375, t810, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t810

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2377, t2378, t2379, t2380, t2381, t2382 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2376,
                t811,
                t812,
                t813,
                t814,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2376, t814, t813, t812, t811

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2383 = paddle._C_ops.relu(t2377)
        del t2377

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2384 = paddle._C_ops.conv2d(
            t2383, t815, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t815, t2383

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2385, t2386, t2387, t2388, t2389, t2390 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2384,
                t816,
                t817,
                t818,
                t819,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2384, t819, t818, t817, t816

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2391 = paddle._C_ops.relu(t2385)
        del t2385

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2392 = paddle._C_ops.conv2d(
            t2391, t820, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t820, t2391

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2393, t2394, t2395, t2396, t2397, t2398 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2392,
                t821,
                t822,
                t823,
                t824,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2392, t824, t823, t822, t821

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2399 = paddle._C_ops.add(t2393, t2375)
        del t2393, t2375

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2400 = paddle._C_ops.relu(t2399)
        del t2399

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2401 = paddle._C_ops.conv2d(
            t2400, t825, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t825

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2402, t2403, t2404, t2405, t2406, t2407 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2401,
                t826,
                t827,
                t828,
                t829,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2401, t829, t828, t827, t826

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2408 = paddle._C_ops.relu(t2402)
        del t2402

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2409 = paddle._C_ops.conv2d(
            t2408, t830, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t830, t2408

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2410, t2411, t2412, t2413, t2414, t2415 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2409,
                t831,
                t832,
                t833,
                t834,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2409, t834, t833, t832, t831

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2416 = paddle._C_ops.relu(t2410)
        del t2410

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2417 = paddle._C_ops.conv2d(
            t2416, t835, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t835, t2416

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2418, t2419, t2420, t2421, t2422, t2423 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2417,
                t836,
                t837,
                t838,
                t839,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2417, t839, t838, t837, t836

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2424 = paddle._C_ops.add(t2418, t2400)
        del t2418, t2400

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2425 = paddle._C_ops.relu(t2424)
        del t2424

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2426 = paddle._C_ops.conv2d(
            t2425, t840, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t840

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2427, t2428, t2429, t2430, t2431, t2432 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2426,
                t841,
                t842,
                t843,
                t844,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2426, t844, t843, t842, t841

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2433 = paddle._C_ops.relu(t2427)
        del t2427

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2434 = paddle._C_ops.conv2d(
            t2433, t845, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t845, t2433

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2435, t2436, t2437, t2438, t2439, t2440 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2434,
                t846,
                t847,
                t848,
                t849,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2434, t849, t848, t847, t846

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2441 = paddle._C_ops.relu(t2435)
        del t2435

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2442 = paddle._C_ops.conv2d(
            t2441, t850, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t850, t2441

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2443, t2444, t2445, t2446, t2447, t2448 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2442,
                t851,
                t852,
                t853,
                t854,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2442, t854, t853, t852, t851

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2449 = paddle._C_ops.add(t2443, t2425)
        del t2443, t2425

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2450 = paddle._C_ops.relu(t2449)
        del t2449

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2451 = paddle._C_ops.conv2d(
            t2450, t855, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t855

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2452, t2453, t2454, t2455, t2456, t2457 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2451,
                t856,
                t857,
                t858,
                t859,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2451, t859, t858, t857, t856

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2458 = paddle._C_ops.relu(t2452)
        del t2452

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2459 = paddle._C_ops.conv2d(
            t2458, t860, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t860, t2458

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2460, t2461, t2462, t2463, t2464, t2465 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2459,
                t861,
                t862,
                t863,
                t864,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2459, t864, t863, t862, t861

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2466 = paddle._C_ops.relu(t2460)
        del t2460

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2467 = paddle._C_ops.conv2d(
            t2466, t865, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t865, t2466

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2468, t2469, t2470, t2471, t2472, t2473 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2467,
                t866,
                t867,
                t868,
                t869,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2467, t869, t868, t867, t866

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2474 = paddle._C_ops.add(t2468, t2450)
        del t2468, t2450

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2475 = paddle._C_ops.relu(t2474)
        del t2474

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2476 = paddle._C_ops.conv2d(
            t2475, t870, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t870

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2477, t2478, t2479, t2480, t2481, t2482 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2476,
                t871,
                t872,
                t873,
                t874,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2476, t874, t873, t872, t871

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2483 = paddle._C_ops.relu(t2477)
        del t2477

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2484 = paddle._C_ops.conv2d(
            t2483, t875, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t875, t2483

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2485, t2486, t2487, t2488, t2489, t2490 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2484,
                t876,
                t877,
                t878,
                t879,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2484, t879, t878, t877, t876

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2491 = paddle._C_ops.relu(t2485)
        del t2485

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2492 = paddle._C_ops.conv2d(
            t2491, t880, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t880, t2491

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2493, t2494, t2495, t2496, t2497, t2498 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2492,
                t881,
                t882,
                t883,
                t884,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2492, t884, t883, t882, t881

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2499 = paddle._C_ops.add(t2493, t2475)
        del t2493, t2475

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2500 = paddle._C_ops.relu(t2499)
        del t2499

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2501 = paddle._C_ops.conv2d(
            t2500, t885, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t885

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2502, t2503, t2504, t2505, t2506, t2507 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2501,
                t886,
                t887,
                t888,
                t889,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2501, t889, t888, t887, t886

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2508 = paddle._C_ops.relu(t2502)
        del t2502

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2509 = paddle._C_ops.conv2d(
            t2508, t890, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t890, t2508

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2510, t2511, t2512, t2513, t2514, t2515 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2509,
                t891,
                t892,
                t893,
                t894,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2509, t894, t893, t892, t891

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2516 = paddle._C_ops.relu(t2510)
        del t2510

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2517 = paddle._C_ops.conv2d(
            t2516, t895, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t895, t2516

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2518, t2519, t2520, t2521, t2522, t2523 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2517,
                t896,
                t897,
                t898,
                t899,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2517, t899, t898, t897, t896

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2524 = paddle._C_ops.add(t2518, t2500)
        del t2518, t2500

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2525 = paddle._C_ops.relu(t2524)
        del t2524

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2526 = paddle._C_ops.conv2d(
            t2525, t900, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t900

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2527, t2528, t2529, t2530, t2531, t2532 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2526,
                t901,
                t902,
                t903,
                t904,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2526, t904, t903, t902, t901

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2533 = paddle._C_ops.relu(t2527)
        del t2527

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2534 = paddle._C_ops.conv2d(
            t2533, t905, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t905, t2533

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2535, t2536, t2537, t2538, t2539, t2540 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2534,
                t906,
                t907,
                t908,
                t909,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2534, t909, t908, t907, t906

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2541 = paddle._C_ops.relu(t2535)
        del t2535

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2542 = paddle._C_ops.conv2d(
            t2541, t910, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t910, t2541

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2543, t2544, t2545, t2546, t2547, t2548 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2542,
                t911,
                t912,
                t913,
                t914,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2542, t914, t913, t912, t911

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2549 = paddle._C_ops.add(t2543, t2525)
        del t2543, t2525

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2550 = paddle._C_ops.relu(t2549)
        del t2549

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2551 = paddle._C_ops.conv2d(
            t2550, t915, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t915

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2552, t2553, t2554, t2555, t2556, t2557 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2551,
                t916,
                t917,
                t918,
                t919,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2551, t919, t918, t917, t916

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2558 = paddle._C_ops.relu(t2552)
        del t2552

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2559 = paddle._C_ops.conv2d(
            t2558, t920, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t920, t2558

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2560, t2561, t2562, t2563, t2564, t2565 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2559,
                t921,
                t922,
                t923,
                t924,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2559, t924, t923, t922, t921

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2566 = paddle._C_ops.relu(t2560)
        del t2560

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2567 = paddle._C_ops.conv2d(
            t2566, t925, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t925, t2566

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2568, t2569, t2570, t2571, t2572, t2573 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2567,
                t926,
                t927,
                t928,
                t929,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2567, t926, t929, t928, t927

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2574 = paddle._C_ops.add(t2568, t2550)
        del t2568, t2550

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2575 = paddle._C_ops.relu(t2574)
        del t2574

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2576 = paddle._C_ops.conv2d(
            t2575, t930, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t930

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2577, t2578, t2579, t2580, t2581, t2582 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2576,
                t931,
                t932,
                t933,
                t934,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2576, t934, t933, t932, t931

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2583 = paddle._C_ops.relu(t2577)
        del t2577

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2584 = paddle._C_ops.conv2d(
            t2583, t935, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t935, t2583

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2585, t2586, t2587, t2588, t2589, t2590 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2584,
                t936,
                t937,
                t938,
                t939,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2584, t939, t938, t937, t936

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2591 = paddle._C_ops.relu(t2585)
        del t2585

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2592 = paddle._C_ops.conv2d(
            t2591, t940, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t940, t2591

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2593, t2594, t2595, t2596, t2597, t2598 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2592,
                t941,
                t942,
                t943,
                t944,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2592, t944, t943, t942, t941

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2599 = paddle._C_ops.add(t2593, t2575)
        del t2593, t2575

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2600 = paddle._C_ops.relu(t2599)
        del t2599

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2601 = paddle._C_ops.conv2d(
            t2600, t945, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t945

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2602, t2603, t2604, t2605, t2606, t2607 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2601,
                t946,
                t947,
                t948,
                t949,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2601, t949, t948, t947, t946

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2608 = paddle._C_ops.relu(t2602)
        del t2602

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2609 = paddle._C_ops.conv2d(
            t2608, t950, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t950, t2608

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2610, t2611, t2612, t2613, t2614, t2615 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2609,
                t951,
                t952,
                t953,
                t954,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2609, t954, t953, t952, t951

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2616 = paddle._C_ops.relu(t2610)
        del t2610

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2617 = paddle._C_ops.conv2d(
            t2616, t955, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t955, t2616

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2618, t2619, t2620, t2621, t2622, t2623 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2617,
                t956,
                t957,
                t958,
                t959,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2617, t959, t958, t957, t956

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2624 = paddle._C_ops.add(t2618, t2600)
        del t2618, t2600

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2625 = paddle._C_ops.relu(t2624)
        del t2624

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        t2626 = paddle._C_ops.conv2d(
            t2625, t960, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t960

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2627, t2628, t2629, t2630, t2631, t2632 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2626,
                t961,
                t962,
                t963,
                t964,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2626, t964, t963, t962, t961

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2633 = paddle._C_ops.relu(t2627)
        del t2627

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        t2634 = paddle._C_ops.conv2d(
            t2633, t965, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t965, t2633

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t2635, t2636, t2637, t2638, t2639, t2640 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2634,
                t966,
                t967,
                t968,
                t969,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2634, t969, t968, t967, t966

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        t2641 = paddle._C_ops.relu(t2635)
        del t2635

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        t2642 = paddle._C_ops.conv2d(
            t2641, t970, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t970, t2641

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        t2643, t2644, t2645, t2646, t2647, t2648 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2642,
                t971,
                t972,
                t973,
                t974,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2642, t974, t973, t972, t971

        # pd_op.add: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        t2649 = paddle._C_ops.add(t2643, t2625)
        del t2643, t2625

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        t2650 = paddle._C_ops.relu(t2649)
        del t2649

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        t2651 = paddle._C_ops.conv2d(
            t2650, t975, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t975

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t2652, t2653, t2654, t2655, t2656, t2657 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2651,
                t976,
                t977,
                t978,
                t979,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2651, t979, t978, t977, t976

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        t2658 = paddle._C_ops.relu(t2652)
        del t2652

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x14x14xf32, 512x512x3x3xf32)
        t2659 = paddle._C_ops.conv2d(
            t2658, t980, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t980, t2658

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t2660, t2661, t2662, t2663, t2664, t2665 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2659,
                t981,
                t982,
                t983,
                t984,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2659, t984, t983, t982, t981

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t2666 = paddle._C_ops.relu(t2660)
        del t2660

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        t2667 = paddle._C_ops.conv2d(
            t2666, t985, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t985, t2666

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t2668, t2669, t2670, t2671, t2672, t2673 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2667,
                t986,
                t987,
                t988,
                t989,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2667, t989, t988, t987, t986

        # pd_op.pool2d: (-1x1024x7x7xf32) <- (-1x1024x14x14xf32, 2xi64)
        t2674 = paddle._C_ops.pool2d(
            t2650,
            t1157,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "SAME",
        )
        del t1157, t2650

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        t2675 = paddle._C_ops.conv2d(
            t2674, t990, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t990, t2674

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t2676, t2677, t2678, t2679, t2680, t2681 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2675,
                t991,
                t992,
                t993,
                t994,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2675, t994, t993, t992, t991

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t2682 = paddle._C_ops.add(t2668, t2676)
        del t2668, t2676

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t2683 = paddle._C_ops.relu(t2682)
        del t2682

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x2048x7x7xf32, 512x2048x1x1xf32)
        t2684 = paddle._C_ops.conv2d(
            t2683, t995, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t995

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t2685, t2686, t2687, t2688, t2689, t2690 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2684,
                t996,
                t997,
                t998,
                t999,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2684, t999, t998, t997, t996

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t2691 = paddle._C_ops.relu(t2685)
        del t2685

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t2692 = paddle._C_ops.conv2d(
            t2691, t1000, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1000, t2691

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t2693, t2694, t2695, t2696, t2697, t2698 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2692,
                t1001,
                t1002,
                t1003,
                t1004,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2692, t1004, t1003, t1002, t1001

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t2699 = paddle._C_ops.relu(t2693)
        del t2693

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        t2700 = paddle._C_ops.conv2d(
            t2699, t1005, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1005, t2699

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t2701, t2702, t2703, t2704, t2705, t2706 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2700,
                t1006,
                t1007,
                t1008,
                t1009,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2700, t1009, t1008, t1007, t1006

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t2707 = paddle._C_ops.add(t2701, t2683)
        del t2701, t2683

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t2708 = paddle._C_ops.relu(t2707)
        del t2707

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x2048x7x7xf32, 512x2048x1x1xf32)
        t2709 = paddle._C_ops.conv2d(
            t2708, t1010, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1010

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t2710, t2711, t2712, t2713, t2714, t2715 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2709,
                t1011,
                t1012,
                t1013,
                t1014,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2709, t1014, t1013, t1012, t1011

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t2716 = paddle._C_ops.relu(t2710)
        del t2710

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        t2717 = paddle._C_ops.conv2d(
            t2716, t1015, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1015, t2716

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        t2718, t2719, t2720, t2721, t2722, t2723 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2717,
                t1016,
                t1017,
                t1018,
                t1019,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2717, t1016, t1019, t1018, t1017

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        t2724 = paddle._C_ops.relu(t2718)
        del t2718

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        t2725 = paddle._C_ops.conv2d(
            t2724, t1020, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t1020, t2724

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        t2726, t2727, t2728, t2729, t2730, t2731 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t2725,
                t1021,
                t1022,
                t1023,
                t1024,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t2725, t1024, t1023, t1022, t1021

        # pd_op.add: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        t2732 = paddle._C_ops.add(t2726, t2708)
        del t2726, t2708

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        t2733 = paddle._C_ops.relu(t2732)
        del t2732

        # pd_op.full_int_array: (2xi64) <- ()
        t2734 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        t2735 = paddle._C_ops.pool2d(
            t2733,
            t2734,
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
        del t2734, t2733

        # pd_op.flatten: (-1x2048xf32) <- (-1x2048x1x1xf32)
        t2736 = paddle._C_ops.flatten(t2735, 1, 3)
        del t2735

        # pd_op.matmul: (-1x102xf32) <- (-1x2048xf32, 2048x102xf32)
        t2737 = paddle._C_ops.matmul(t2736, t1025, False, False)
        del t2736, t1025

        return t2737
