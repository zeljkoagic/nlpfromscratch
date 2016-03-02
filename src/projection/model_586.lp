Maximize
  - 3.9725342508324 edge_0_1_0_0 - 5.0972622327052 edge_0_2_0_0
   - 3.1579666314352 edge_0_3_0_0 - 1.228439631468 edge_0_4_0_0
   - 3.2426953087797 edge_0_5_0_0 - 4.464259925605001 edge_0_9_0_0
   - 3.565749786055 edge_0_10_0_0 - 3.4856547129482 edge_0_12_0_0
   - 0.1669120305651643 edge_1_2_0_0 - 0.4628778141605142 edge_1_3_0_0
   - 0.0250334311007974 edge_1_4_0_0 - 0.536752360153256 edge_1_5_0_0
   - 0.5848622801449478 edge_1_9_0_0 - 0.4521750928432128 edge_1_10_0_0
   - 0.9479976649521501 edge_1_12_0_0 + 0.2354151550906381 edge_2_1_0_0
   + 1.24402078766351 edge_2_3_0_0 + 0.2647877009476446 edge_2_4_0_0
   - 0.1849725934456128 edge_2_5_0_0 + 0.024467672299803 edge_2_9_0_0
   - 0.2996120800915721 edge_2_10_0_0 - 0.497513055837203 edge_2_12_0_0
   + 0.910689219268778 edge_3_1_0_0 + 3.192486378851899 edge_3_2_0_0
   + 0.8283805268235905 edge_3_4_0_0 + 0.3423351154002479 edge_3_5_0_0
   + 0.8952816201855445 edge_3_9_0_0 + 0.8398942815779201 edge_3_10_0_0
   - 0.0341090469878094 edge_3_11_0_0 + 0.8159963928093251 edge_3_12_0_0
   + 2.360430687061682 edge_4_1_0_0 + 0.8954076996738325 edge_4_2_0_0
   + 2.633226813626719 edge_4_3_0_0 + 2.761998828525703 edge_4_5_0_0
   + 3.169587137961373 edge_4_9_0_0 + 2.938238942233653 edge_4_10_0_0
   + 0.0698405694638989 edge_4_11_0_0 + 1.231388017462584 edge_4_12_0_0
   - 0.453664079995729 edge_5_1_0_0 + 0.0591698142230462 edge_5_2_0_0
   + 0.4608313605835763 edge_5_3_0_0 + 0.621588133317294 edge_5_4_0_0
   + 0.6950753495929421 edge_5_9_0_0 + 0.4937315139848973 edge_5_10_0_0
   - 0.5981096217445843 edge_5_11_0_0 - 0.4295093013287681 edge_5_12_0_0
   - 1.351719771199344 edge_9_1_0_0 - 0.7520150495131356 edge_9_3_0_0
   - 0.6855189048671163 edge_9_4_0_0 - 0.7076135755620914 edge_9_5_0_0
   + 0.0317833355943438 edge_9_10_0_0 - 0.2795086210540267 edge_9_12_0_0
   - 0.2713375881796914 edge_10_1_0_0 + 0.1986250796661345 edge_10_3_0_0
   - 0.2993368132857264 edge_10_4_0_0 + 0.0462162348861727 edge_10_5_0_0
   + 0.2664354793267102 edge_10_9_0_0 + 0.505370486808729 edge_10_11_0_0
   + 2.070417163524763 edge_10_12_0_0 - 0.9065246755930328 edge_11_9_0_0
   - 0.3753526648529887 edge_11_10_0_0 - 0.9755599005744033 edge_12_1_0_0
   + 0.1973125289572184 edge_12_2_0_0 - 0.2926518264132852 edge_12_3_0_0
   - 0.5176079239316466 edge_12_4_0_0 - 0.8130672124538563 edge_12_5_0_0
   - 0.2752760018062929 edge_12_9_0_0 + 0.6735920771776429 edge_12_10_0_0
   + 3.290413357129555 edge_12_11_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_9_1_0_0 + edge_10_1_0_0 + edge_12_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_9_0_0 - flow_1_10_0_0
   - flow_1_12_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_5_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0 + flow_12_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_12_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_9_0_0
   - flow_2_10_0_0 - flow_2_12_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_12_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_9_3_0_0 + edge_10_3_0_0 + edge_12_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_9_0_0 - flow_3_10_0_0 - flow_3_11_0_0 - flow_3_12_0_0
   + flow_4_3_0_0 + flow_5_3_0_0 + flow_9_3_0_0 + flow_10_3_0_0
   + flow_12_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_9_4_0_0 + edge_10_4_0_0 + edge_12_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_9_0_0 - flow_4_10_0_0 - flow_4_11_0_0
   - flow_4_12_0_0 + flow_5_4_0_0 + flow_9_4_0_0 + flow_10_4_0_0
   + flow_12_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_9_5_0_0 + edge_10_5_0_0 + edge_12_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_9_0_0 - flow_5_10_0_0
   - flow_5_11_0_0 - flow_5_12_0_0 + flow_9_5_0_0 + flow_10_5_0_0
   + flow_12_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_5_9_0_0 + edge_10_9_0_0 + edge_11_9_0_0
   + edge_12_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_5_9_0_0 - flow_9_1_0_0
   - flow_9_3_0_0 - flow_9_4_0_0 - flow_9_5_0_0 - flow_9_10_0_0
   - flow_9_12_0_0 + flow_10_9_0_0 + flow_11_9_0_0 + flow_12_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_5_10_0_0 + edge_9_10_0_0
   + edge_11_10_0_0 + edge_12_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_5_10_0_0 + flow_9_10_0_0
   - flow_10_1_0_0 - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_5_0_0
   - flow_10_9_0_0 - flow_10_11_0_0 - flow_10_12_0_0 + flow_11_10_0_0
   + flow_12_10_0_0 = 1
 one_parent_11: edge_3_11_0_0 + edge_4_11_0_0 + edge_5_11_0_0
   + edge_10_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_3_11_0_0 + flow_4_11_0_0 + flow_5_11_0_0
   + flow_10_11_0_0 - flow_11_9_0_0 - flow_11_10_0_0 + flow_12_11_0_0 = 1
 one_parent_12: edge_0_12_0_0 + edge_1_12_0_0 + edge_2_12_0_0
   + edge_3_12_0_0 + edge_4_12_0_0 + edge_5_12_0_0 + edge_9_12_0_0
   + edge_10_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_1_12_0_0 + flow_2_12_0_0
   + flow_3_12_0_0 + flow_4_12_0_0 + flow_5_12_0_0 + flow_9_12_0_0
   + flow_10_12_0_0 - flow_12_1_0_0 - flow_12_2_0_0 - flow_12_3_0_0
   - flow_12_4_0_0 - flow_12_5_0_0 - flow_12_9_0_0 - flow_12_10_0_0
   - flow_12_11_0_0 = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_9_0_0 + flow_0_10_0_0 + flow_0_12_0_0 = 13
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_12_1_0_0: flow_12_1_0_0 - 1000 edge_12_1_0_0 <= 0
 inactive_no_flow_12_2_0_0: flow_12_2_0_0 - 1000 edge_12_2_0_0 <= 0
 inactive_no_flow_12_3_0_0: flow_12_3_0_0 - 1000 edge_12_3_0_0 <= 0
 inactive_no_flow_12_4_0_0: flow_12_4_0_0 - 1000 edge_12_4_0_0 <= 0
 inactive_no_flow_12_5_0_0: flow_12_5_0_0 - 1000 edge_12_5_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_10_0_0: flow_12_10_0_0 - 1000 edge_12_10_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_9_0_0 edge_0_10_0_0 edge_0_12_0_0 edge_1_2_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_5_0_0 edge_1_9_0_0 edge_1_10_0_0 edge_1_12_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_9_0_0
 edge_2_10_0_0 edge_2_12_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0
 edge_3_5_0_0 edge_3_9_0_0 edge_3_10_0_0 edge_3_11_0_0 edge_3_12_0_0
 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_9_0_0
 edge_4_10_0_0 edge_4_11_0_0 edge_4_12_0_0 edge_5_1_0_0 edge_5_2_0_0
 edge_5_3_0_0 edge_5_4_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_5_11_0_0
 edge_5_12_0_0 edge_9_1_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_5_0_0
 edge_9_10_0_0 edge_9_12_0_0 edge_10_1_0_0 edge_10_3_0_0 edge_10_4_0_0
 edge_10_5_0_0 edge_10_9_0_0 edge_10_11_0_0 edge_10_12_0_0 edge_11_9_0_0
 edge_11_10_0_0 edge_12_1_0_0 edge_12_2_0_0 edge_12_3_0_0 edge_12_4_0_0
 edge_12_5_0_0 edge_12_9_0_0 edge_12_10_0_0 edge_12_11_0_0
End
