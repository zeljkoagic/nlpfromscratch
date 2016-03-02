Maximize
  - 3.65093433809 edge_0_1_0_4 - 2.7060435391465 edge_0_2_0_9
   - 1.5818515282368 edge_0_3_0_10 - 0.707158016954 edge_0_4_0_3
   - 0.777146117423 edge_0_5_0_3 - 2.862149656542 edge_0_6_0_2
   - 2.7988990002837 edge_0_7_0_6 + 0.2323551076349348 edge_1_2_4_9
   - 0.4654079819076274 edge_1_3_4_10 - 0.2346494895487558 edge_1_4_4_3
   - 1.284352247828711 edge_1_6_4_2 - 0.4934614561957557 edge_1_7_4_6
   + 0.5675704116484467 edge_1_8_4_5 - 0.4526144456254675 edge_2_1_9_4
   + 0.0198011211043407 edge_2_3_9_10 + 0.9578344663249405 edge_2_4_9_3
   - 0.5256218443975782 edge_2_5_9_3 + 0.82377168781376 edge_2_6_9_2
   + 0.2319715360592 edge_2_7_9_6 + 0.85280860416416 edge_2_8_9_5
   - 0.2831708088153052 edge_3_1_10_4 + 0.2136302822376683 edge_3_2_10_9
   + 0.2814438402764221 edge_3_4_10_3 - 0.1276779430207616 edge_3_5_10_3
   - 0.1291761380798216 edge_3_6_10_2 + 0.2200372896272633 edge_3_7_10_6
   + 0.5471423118921098 edge_3_8_10_5 + 1.275068112977987 edge_4_1_3_4
   + 2.799306673351885 edge_4_2_3_9 + 2.371940821122132 edge_4_3_3_10
   + 0.8605645176522579 edge_4_5_3_3 + 0.8671646117024405 edge_4_5_3_7
   + 2.739936575075 edge_4_6_3_2 + 2.077559121560564 edge_4_7_3_6
   + 3.601156476543297 edge_4_8_3_5 + 0.1197067570321921 edge_5_1_3_4
   - 2.194752396374442 edge_5_1_7_4 + 0.081848370838468 edge_5_2_3_9
   - 0.6498822115670936 edge_5_2_7_9 + 0.2003667846000659 edge_5_3_3_10
   + 0.2479027306888322 edge_5_4_3_3 + 1.8512490027856 edge_5_6_3_2
   + 0.3577519692481829 edge_5_7_3_6 + 0.92782225468317 edge_5_7_7_6
   + 1.487555065267045 edge_5_8_3_5 - 0.3198576746178 edge_5_8_7_5
   - 1.096337565860976 edge_6_1_2_4 + 0.91468827047136 edge_6_2_2_9
   - 0.5089156094345195 edge_6_3_2_10 + 0.5233653457925 edge_6_4_2_3
   + 0.3898782259416879 edge_6_5_2_3 + 3.1374071908839 edge_6_5_2_7
   + 3.55292 edge_6_7_2_6 + 1.25785 edge_6_8_2_5
   - 0.7295392423361976 edge_7_1_6_4 + 0.68485676859712 edge_7_2_6_9
   + 0.1580727194024918 edge_7_3_6_10 + 1.494966669077901 edge_7_4_6_3
   + 0.865510130323504 edge_7_5_6_3 + 1.6056383269245 edge_7_5_6_7
   + 1.82317261612186 edge_7_6_6_2 + 1.600311681234659 edge_7_8_6_5
   - 0.9242970110012453 edge_8_6_5_2
Subject To
 one_parent_1: edge_0_1_0_4 + edge_2_1_9_4 + edge_3_1_10_4 + edge_4_1_3_4
   + edge_5_1_3_4 + edge_5_1_7_4 + edge_6_1_2_4 + edge_7_1_6_4 = 1
 one_pos_1: pos_1_4 = 1
 consume_one_flow_unit_1: flow_0_1_0_4 - flow_1_2_4_9 - flow_1_3_4_10
   - flow_1_4_4_3 - flow_1_6_4_2 - flow_1_7_4_6 - flow_1_8_4_5
   + flow_2_1_9_4 + flow_3_1_10_4 + flow_4_1_3_4 + flow_5_1_3_4
   + flow_5_1_7_4 + flow_6_1_2_4 + flow_7_1_6_4 = 1
 one_parent_2: edge_0_2_0_9 + edge_1_2_4_9 + edge_3_2_10_9 + edge_4_2_3_9
   + edge_5_2_3_9 + edge_5_2_7_9 + edge_6_2_2_9 + edge_7_2_6_9 = 1
 one_pos_2: pos_2_9 = 1
 consume_one_flow_unit_2: flow_0_2_0_9 + flow_1_2_4_9 - flow_2_1_9_4
   - flow_2_3_9_10 - flow_2_4_9_3 - flow_2_5_9_3 - flow_2_6_9_2
   - flow_2_7_9_6 - flow_2_8_9_5 + flow_3_2_10_9 + flow_4_2_3_9
   + flow_5_2_3_9 + flow_5_2_7_9 + flow_6_2_2_9 + flow_7_2_6_9 = 1
 one_parent_3: edge_0_3_0_10 + edge_1_3_4_10 + edge_2_3_9_10
   + edge_4_3_3_10 + edge_5_3_3_10 + edge_6_3_2_10 + edge_7_3_6_10 = 1
 one_pos_3: pos_3_10 = 1
 consume_one_flow_unit_3: flow_0_3_0_10 + flow_1_3_4_10 + flow_2_3_9_10
   - flow_3_1_10_4 - flow_3_2_10_9 - flow_3_4_10_3 - flow_3_5_10_3
   - flow_3_6_10_2 - flow_3_7_10_6 - flow_3_8_10_5 + flow_4_3_3_10
   + flow_5_3_3_10 + flow_6_3_2_10 + flow_7_3_6_10 = 1
 one_parent_4: edge_0_4_0_3 + edge_1_4_4_3 + edge_2_4_9_3 + edge_3_4_10_3
   + edge_5_4_3_3 + edge_6_4_2_3 + edge_7_4_6_3 = 1
 one_pos_4: pos_4_3 = 1
 consume_one_flow_unit_4: flow_0_4_0_3 + flow_1_4_4_3 + flow_2_4_9_3
   + flow_3_4_10_3 - flow_4_1_3_4 - flow_4_2_3_9 - flow_4_3_3_10
   - flow_4_5_3_3 - flow_4_5_3_7 - flow_4_6_3_2 - flow_4_7_3_6
   - flow_4_8_3_5 + flow_5_4_3_3 + flow_6_4_2_3 + flow_7_4_6_3 = 1
 one_parent_5: edge_0_5_0_3 + edge_2_5_9_3 + edge_3_5_10_3 + edge_4_5_3_3
   + edge_4_5_3_7 + edge_6_5_2_3 + edge_6_5_2_7 + edge_7_5_6_3
   + edge_7_5_6_7 = 1
 one_pos_5: pos_5_3 + pos_5_7 = 1
 consume_one_flow_unit_5: flow_0_5_0_3 + flow_2_5_9_3 + flow_3_5_10_3
   + flow_4_5_3_3 + flow_4_5_3_7 - flow_5_1_3_4 - flow_5_1_7_4
   - flow_5_2_3_9 - flow_5_2_7_9 - flow_5_3_3_10 - flow_5_4_3_3
   - flow_5_6_3_2 - flow_5_7_3_6 - flow_5_7_7_6 - flow_5_8_3_5
   - flow_5_8_7_5 + flow_6_5_2_3 + flow_6_5_2_7 + flow_7_5_6_3
   + flow_7_5_6_7 = 1
 one_parent_6: edge_0_6_0_2 + edge_1_6_4_2 + edge_2_6_9_2 + edge_3_6_10_2
   + edge_4_6_3_2 + edge_5_6_3_2 + edge_7_6_6_2 + edge_8_6_5_2 = 1
 one_pos_6: pos_6_2 = 1
 consume_one_flow_unit_6: flow_0_6_0_2 + flow_1_6_4_2 + flow_2_6_9_2
   + flow_3_6_10_2 + flow_4_6_3_2 + flow_5_6_3_2 - flow_6_1_2_4
   - flow_6_2_2_9 - flow_6_3_2_10 - flow_6_4_2_3 - flow_6_5_2_3
   - flow_6_5_2_7 - flow_6_7_2_6 - flow_6_8_2_5 + flow_7_6_6_2
   + flow_8_6_5_2 = 1
 one_parent_7: edge_0_7_0_6 + edge_1_7_4_6 + edge_2_7_9_6 + edge_3_7_10_6
   + edge_4_7_3_6 + edge_5_7_3_6 + edge_5_7_7_6 + edge_6_7_2_6 = 1
 one_pos_7: pos_7_6 = 1
 consume_one_flow_unit_7: flow_0_7_0_6 + flow_1_7_4_6 + flow_2_7_9_6
   + flow_3_7_10_6 + flow_4_7_3_6 + flow_5_7_3_6 + flow_5_7_7_6
   + flow_6_7_2_6 - flow_7_1_6_4 - flow_7_2_6_9 - flow_7_3_6_10
   - flow_7_4_6_3 - flow_7_5_6_3 - flow_7_5_6_7 - flow_7_6_6_2
   - flow_7_8_6_5 = 1
 one_parent_8: edge_1_8_4_5 + edge_2_8_9_5 + edge_3_8_10_5 + edge_4_8_3_5
   + edge_5_8_3_5 + edge_5_8_7_5 + edge_6_8_2_5 + edge_7_8_6_5 = 1
 one_pos_8: pos_8_5 = 1
 consume_one_flow_unit_8: flow_1_8_4_5 + flow_2_8_9_5 + flow_3_8_10_5
   + flow_4_8_3_5 + flow_5_8_3_5 + flow_5_8_7_5 + flow_6_8_2_5
   + flow_7_8_6_5 - flow_8_6_5_2 = 1
 root_flow: flow_0_1_0_4 + flow_0_2_0_9 + flow_0_3_0_10 + flow_0_4_0_3
   + flow_0_5_0_3 + flow_0_6_0_2 + flow_0_7_0_6 = 8
 inactive_no_flow_0_1_0_4: flow_0_1_0_4 - 1000 edge_0_1_0_4 <= 0
 inactive_no_flow_0_2_0_9: flow_0_2_0_9 - 1000 edge_0_2_0_9 <= 0
 inactive_no_flow_0_3_0_10: flow_0_3_0_10 - 1000 edge_0_3_0_10 <= 0
 inactive_no_flow_0_4_0_3: flow_0_4_0_3 - 1000 edge_0_4_0_3 <= 0
 inactive_no_flow_0_5_0_3: flow_0_5_0_3 - 1000 edge_0_5_0_3 <= 0
 inactive_no_flow_0_6_0_2: flow_0_6_0_2 - 1000 edge_0_6_0_2 <= 0
 inactive_no_flow_0_7_0_6: flow_0_7_0_6 - 1000 edge_0_7_0_6 <= 0
 inactive_no_flow_1_2_4_9: flow_1_2_4_9 - 1000 edge_1_2_4_9 <= 0
 inactive_no_flow_1_3_4_10: flow_1_3_4_10 - 1000 edge_1_3_4_10 <= 0
 inactive_no_flow_1_4_4_3: flow_1_4_4_3 - 1000 edge_1_4_4_3 <= 0
 inactive_no_flow_1_6_4_2: flow_1_6_4_2 - 1000 edge_1_6_4_2 <= 0
 inactive_no_flow_1_7_4_6: flow_1_7_4_6 - 1000 edge_1_7_4_6 <= 0
 inactive_no_flow_1_8_4_5: flow_1_8_4_5 - 1000 edge_1_8_4_5 <= 0
 inactive_no_flow_2_1_9_4: flow_2_1_9_4 - 1000 edge_2_1_9_4 <= 0
 inactive_no_flow_2_3_9_10: flow_2_3_9_10 - 1000 edge_2_3_9_10 <= 0
 inactive_no_flow_2_4_9_3: flow_2_4_9_3 - 1000 edge_2_4_9_3 <= 0
 inactive_no_flow_2_5_9_3: flow_2_5_9_3 - 1000 edge_2_5_9_3 <= 0
 inactive_no_flow_2_6_9_2: flow_2_6_9_2 - 1000 edge_2_6_9_2 <= 0
 inactive_no_flow_2_7_9_6: flow_2_7_9_6 - 1000 edge_2_7_9_6 <= 0
 inactive_no_flow_2_8_9_5: flow_2_8_9_5 - 1000 edge_2_8_9_5 <= 0
 inactive_no_flow_3_1_10_4: flow_3_1_10_4 - 1000 edge_3_1_10_4 <= 0
 inactive_no_flow_3_2_10_9: flow_3_2_10_9 - 1000 edge_3_2_10_9 <= 0
 inactive_no_flow_3_4_10_3: flow_3_4_10_3 - 1000 edge_3_4_10_3 <= 0
 inactive_no_flow_3_5_10_3: flow_3_5_10_3 - 1000 edge_3_5_10_3 <= 0
 inactive_no_flow_3_6_10_2: flow_3_6_10_2 - 1000 edge_3_6_10_2 <= 0
 inactive_no_flow_3_7_10_6: flow_3_7_10_6 - 1000 edge_3_7_10_6 <= 0
 inactive_no_flow_3_8_10_5: flow_3_8_10_5 - 1000 edge_3_8_10_5 <= 0
 inactive_no_flow_4_1_3_4: flow_4_1_3_4 - 1000 edge_4_1_3_4 <= 0
 inactive_no_flow_4_2_3_9: flow_4_2_3_9 - 1000 edge_4_2_3_9 <= 0
 inactive_no_flow_4_3_3_10: flow_4_3_3_10 - 1000 edge_4_3_3_10 <= 0
 inactive_no_flow_4_5_3_3: flow_4_5_3_3 - 1000 edge_4_5_3_3 <= 0
 inactive_no_flow_4_5_3_7: flow_4_5_3_7 - 1000 edge_4_5_3_7 <= 0
 inactive_no_flow_4_6_3_2: flow_4_6_3_2 - 1000 edge_4_6_3_2 <= 0
 inactive_no_flow_4_7_3_6: flow_4_7_3_6 - 1000 edge_4_7_3_6 <= 0
 inactive_no_flow_4_8_3_5: flow_4_8_3_5 - 1000 edge_4_8_3_5 <= 0
 inactive_no_flow_5_1_3_4: flow_5_1_3_4 - 1000 edge_5_1_3_4 <= 0
 inactive_no_flow_5_1_7_4: flow_5_1_7_4 - 1000 edge_5_1_7_4 <= 0
 inactive_no_flow_5_2_3_9: flow_5_2_3_9 - 1000 edge_5_2_3_9 <= 0
 inactive_no_flow_5_2_7_9: flow_5_2_7_9 - 1000 edge_5_2_7_9 <= 0
 inactive_no_flow_5_3_3_10: flow_5_3_3_10 - 1000 edge_5_3_3_10 <= 0
 inactive_no_flow_5_4_3_3: flow_5_4_3_3 - 1000 edge_5_4_3_3 <= 0
 inactive_no_flow_5_6_3_2: flow_5_6_3_2 - 1000 edge_5_6_3_2 <= 0
 inactive_no_flow_5_7_3_6: flow_5_7_3_6 - 1000 edge_5_7_3_6 <= 0
 inactive_no_flow_5_7_7_6: flow_5_7_7_6 - 1000 edge_5_7_7_6 <= 0
 inactive_no_flow_5_8_3_5: flow_5_8_3_5 - 1000 edge_5_8_3_5 <= 0
 inactive_no_flow_5_8_7_5: flow_5_8_7_5 - 1000 edge_5_8_7_5 <= 0
 inactive_no_flow_6_1_2_4: flow_6_1_2_4 - 1000 edge_6_1_2_4 <= 0
 inactive_no_flow_6_2_2_9: flow_6_2_2_9 - 1000 edge_6_2_2_9 <= 0
 inactive_no_flow_6_3_2_10: flow_6_3_2_10 - 1000 edge_6_3_2_10 <= 0
 inactive_no_flow_6_4_2_3: flow_6_4_2_3 - 1000 edge_6_4_2_3 <= 0
 inactive_no_flow_6_5_2_3: flow_6_5_2_3 - 1000 edge_6_5_2_3 <= 0
 inactive_no_flow_6_5_2_7: flow_6_5_2_7 - 1000 edge_6_5_2_7 <= 0
 inactive_no_flow_6_7_2_6: flow_6_7_2_6 - 1000 edge_6_7_2_6 <= 0
 inactive_no_flow_6_8_2_5: flow_6_8_2_5 - 1000 edge_6_8_2_5 <= 0
 inactive_no_flow_7_1_6_4: flow_7_1_6_4 - 1000 edge_7_1_6_4 <= 0
 inactive_no_flow_7_2_6_9: flow_7_2_6_9 - 1000 edge_7_2_6_9 <= 0
 inactive_no_flow_7_3_6_10: flow_7_3_6_10 - 1000 edge_7_3_6_10 <= 0
 inactive_no_flow_7_4_6_3: flow_7_4_6_3 - 1000 edge_7_4_6_3 <= 0
 inactive_no_flow_7_5_6_3: flow_7_5_6_3 - 1000 edge_7_5_6_3 <= 0
 inactive_no_flow_7_5_6_7: flow_7_5_6_7 - 1000 edge_7_5_6_7 <= 0
 inactive_no_flow_7_6_6_2: flow_7_6_6_2 - 1000 edge_7_6_6_2 <= 0
 inactive_no_flow_7_8_6_5: flow_7_8_6_5 - 1000 edge_7_8_6_5 <= 0
 inactive_no_flow_8_6_5_2: flow_8_6_5_2 - 1000 edge_8_6_5_2 <= 0
 arc_0_1_0_4_pos_must_match_v: - edge_0_1_0_4 + pos_1_4 >= 0
 arc_0_2_0_9_pos_must_match_v: - edge_0_2_0_9 + pos_2_9 >= 0
 arc_0_3_0_10_pos_must_match_v: - edge_0_3_0_10 + pos_3_10 >= 0
 arc_0_4_0_3_pos_must_match_v: - edge_0_4_0_3 + pos_4_3 >= 0
 arc_0_5_0_3_pos_must_match_v: - edge_0_5_0_3 + pos_5_3 >= 0
 arc_0_6_0_2_pos_must_match_v: - edge_0_6_0_2 + pos_6_2 >= 0
 arc_0_7_0_6_pos_must_match_v: - edge_0_7_0_6 + pos_7_6 >= 0
 arc_1_2_4_9_pos_must_match_u: - edge_1_2_4_9 + pos_1_4 >= 0
 arc_1_2_4_9_pos_must_match_v: - edge_1_2_4_9 + pos_2_9 >= 0
 arc_1_3_4_10_pos_must_match_u: - edge_1_3_4_10 + pos_1_4 >= 0
 arc_1_3_4_10_pos_must_match_v: - edge_1_3_4_10 + pos_3_10 >= 0
 arc_1_4_4_3_pos_must_match_u: - edge_1_4_4_3 + pos_1_4 >= 0
 arc_1_4_4_3_pos_must_match_v: - edge_1_4_4_3 + pos_4_3 >= 0
 arc_1_6_4_2_pos_must_match_u: - edge_1_6_4_2 + pos_1_4 >= 0
 arc_1_6_4_2_pos_must_match_v: - edge_1_6_4_2 + pos_6_2 >= 0
 arc_1_7_4_6_pos_must_match_u: - edge_1_7_4_6 + pos_1_4 >= 0
 arc_1_7_4_6_pos_must_match_v: - edge_1_7_4_6 + pos_7_6 >= 0
 arc_1_8_4_5_pos_must_match_u: - edge_1_8_4_5 + pos_1_4 >= 0
 arc_1_8_4_5_pos_must_match_v: - edge_1_8_4_5 + pos_8_5 >= 0
 arc_2_1_9_4_pos_must_match_u: - edge_2_1_9_4 + pos_2_9 >= 0
 arc_2_1_9_4_pos_must_match_v: - edge_2_1_9_4 + pos_1_4 >= 0
 arc_2_3_9_10_pos_must_match_u: - edge_2_3_9_10 + pos_2_9 >= 0
 arc_2_3_9_10_pos_must_match_v: - edge_2_3_9_10 + pos_3_10 >= 0
 arc_2_4_9_3_pos_must_match_u: - edge_2_4_9_3 + pos_2_9 >= 0
 arc_2_4_9_3_pos_must_match_v: - edge_2_4_9_3 + pos_4_3 >= 0
 arc_2_5_9_3_pos_must_match_u: - edge_2_5_9_3 + pos_2_9 >= 0
 arc_2_5_9_3_pos_must_match_v: - edge_2_5_9_3 + pos_5_3 >= 0
 arc_2_6_9_2_pos_must_match_u: - edge_2_6_9_2 + pos_2_9 >= 0
 arc_2_6_9_2_pos_must_match_v: - edge_2_6_9_2 + pos_6_2 >= 0
 arc_2_7_9_6_pos_must_match_u: - edge_2_7_9_6 + pos_2_9 >= 0
 arc_2_7_9_6_pos_must_match_v: - edge_2_7_9_6 + pos_7_6 >= 0
 arc_2_8_9_5_pos_must_match_u: - edge_2_8_9_5 + pos_2_9 >= 0
 arc_2_8_9_5_pos_must_match_v: - edge_2_8_9_5 + pos_8_5 >= 0
 arc_3_1_10_4_pos_must_match_u: - edge_3_1_10_4 + pos_3_10 >= 0
 arc_3_1_10_4_pos_must_match_v: - edge_3_1_10_4 + pos_1_4 >= 0
 arc_3_2_10_9_pos_must_match_u: - edge_3_2_10_9 + pos_3_10 >= 0
 arc_3_2_10_9_pos_must_match_v: - edge_3_2_10_9 + pos_2_9 >= 0
 arc_3_4_10_3_pos_must_match_u: - edge_3_4_10_3 + pos_3_10 >= 0
 arc_3_4_10_3_pos_must_match_v: - edge_3_4_10_3 + pos_4_3 >= 0
 arc_3_5_10_3_pos_must_match_u: - edge_3_5_10_3 + pos_3_10 >= 0
 arc_3_5_10_3_pos_must_match_v: - edge_3_5_10_3 + pos_5_3 >= 0
 arc_3_6_10_2_pos_must_match_u: - edge_3_6_10_2 + pos_3_10 >= 0
 arc_3_6_10_2_pos_must_match_v: - edge_3_6_10_2 + pos_6_2 >= 0
 arc_3_7_10_6_pos_must_match_u: - edge_3_7_10_6 + pos_3_10 >= 0
 arc_3_7_10_6_pos_must_match_v: - edge_3_7_10_6 + pos_7_6 >= 0
 arc_3_8_10_5_pos_must_match_u: - edge_3_8_10_5 + pos_3_10 >= 0
 arc_3_8_10_5_pos_must_match_v: - edge_3_8_10_5 + pos_8_5 >= 0
 arc_4_1_3_4_pos_must_match_u: - edge_4_1_3_4 + pos_4_3 >= 0
 arc_4_1_3_4_pos_must_match_v: - edge_4_1_3_4 + pos_1_4 >= 0
 arc_4_2_3_9_pos_must_match_u: - edge_4_2_3_9 + pos_4_3 >= 0
 arc_4_2_3_9_pos_must_match_v: - edge_4_2_3_9 + pos_2_9 >= 0
 arc_4_3_3_10_pos_must_match_u: - edge_4_3_3_10 + pos_4_3 >= 0
 arc_4_3_3_10_pos_must_match_v: - edge_4_3_3_10 + pos_3_10 >= 0
 arc_4_5_3_3_pos_must_match_u: - edge_4_5_3_3 + pos_4_3 >= 0
 arc_4_5_3_3_pos_must_match_v: - edge_4_5_3_3 + pos_5_3 >= 0
 arc_4_5_3_7_pos_must_match_u: - edge_4_5_3_7 + pos_4_3 >= 0
 arc_4_5_3_7_pos_must_match_v: - edge_4_5_3_7 + pos_5_7 >= 0
 arc_4_6_3_2_pos_must_match_u: - edge_4_6_3_2 + pos_4_3 >= 0
 arc_4_6_3_2_pos_must_match_v: - edge_4_6_3_2 + pos_6_2 >= 0
 arc_4_7_3_6_pos_must_match_u: - edge_4_7_3_6 + pos_4_3 >= 0
 arc_4_7_3_6_pos_must_match_v: - edge_4_7_3_6 + pos_7_6 >= 0
 arc_4_8_3_5_pos_must_match_u: - edge_4_8_3_5 + pos_4_3 >= 0
 arc_4_8_3_5_pos_must_match_v: - edge_4_8_3_5 + pos_8_5 >= 0
 arc_5_1_3_4_pos_must_match_u: - edge_5_1_3_4 + pos_5_3 >= 0
 arc_5_1_3_4_pos_must_match_v: - edge_5_1_3_4 + pos_1_4 >= 0
 arc_5_1_7_4_pos_must_match_u: - edge_5_1_7_4 + pos_5_7 >= 0
 arc_5_1_7_4_pos_must_match_v: - edge_5_1_7_4 + pos_1_4 >= 0
 arc_5_2_3_9_pos_must_match_u: - edge_5_2_3_9 + pos_5_3 >= 0
 arc_5_2_3_9_pos_must_match_v: - edge_5_2_3_9 + pos_2_9 >= 0
 arc_5_2_7_9_pos_must_match_u: - edge_5_2_7_9 + pos_5_7 >= 0
 arc_5_2_7_9_pos_must_match_v: - edge_5_2_7_9 + pos_2_9 >= 0
 arc_5_3_3_10_pos_must_match_u: - edge_5_3_3_10 + pos_5_3 >= 0
 arc_5_3_3_10_pos_must_match_v: - edge_5_3_3_10 + pos_3_10 >= 0
 arc_5_4_3_3_pos_must_match_u: - edge_5_4_3_3 + pos_5_3 >= 0
 arc_5_4_3_3_pos_must_match_v: - edge_5_4_3_3 + pos_4_3 >= 0
 arc_5_6_3_2_pos_must_match_u: - edge_5_6_3_2 + pos_5_3 >= 0
 arc_5_6_3_2_pos_must_match_v: - edge_5_6_3_2 + pos_6_2 >= 0
 arc_5_7_3_6_pos_must_match_u: - edge_5_7_3_6 + pos_5_3 >= 0
 arc_5_7_3_6_pos_must_match_v: - edge_5_7_3_6 + pos_7_6 >= 0
 arc_5_7_7_6_pos_must_match_u: - edge_5_7_7_6 + pos_5_7 >= 0
 arc_5_7_7_6_pos_must_match_v: - edge_5_7_7_6 + pos_7_6 >= 0
 arc_5_8_3_5_pos_must_match_u: - edge_5_8_3_5 + pos_5_3 >= 0
 arc_5_8_3_5_pos_must_match_v: - edge_5_8_3_5 + pos_8_5 >= 0
 arc_5_8_7_5_pos_must_match_u: - edge_5_8_7_5 + pos_5_7 >= 0
 arc_5_8_7_5_pos_must_match_v: - edge_5_8_7_5 + pos_8_5 >= 0
 arc_6_1_2_4_pos_must_match_u: - edge_6_1_2_4 + pos_6_2 >= 0
 arc_6_1_2_4_pos_must_match_v: - edge_6_1_2_4 + pos_1_4 >= 0
 arc_6_2_2_9_pos_must_match_u: - edge_6_2_2_9 + pos_6_2 >= 0
 arc_6_2_2_9_pos_must_match_v: - edge_6_2_2_9 + pos_2_9 >= 0
 arc_6_3_2_10_pos_must_match_u: - edge_6_3_2_10 + pos_6_2 >= 0
 arc_6_3_2_10_pos_must_match_v: - edge_6_3_2_10 + pos_3_10 >= 0
 arc_6_4_2_3_pos_must_match_u: - edge_6_4_2_3 + pos_6_2 >= 0
 arc_6_4_2_3_pos_must_match_v: - edge_6_4_2_3 + pos_4_3 >= 0
 arc_6_5_2_3_pos_must_match_u: - edge_6_5_2_3 + pos_6_2 >= 0
 arc_6_5_2_3_pos_must_match_v: - edge_6_5_2_3 + pos_5_3 >= 0
 arc_6_5_2_7_pos_must_match_u: - edge_6_5_2_7 + pos_6_2 >= 0
 arc_6_5_2_7_pos_must_match_v: - edge_6_5_2_7 + pos_5_7 >= 0
 arc_6_7_2_6_pos_must_match_u: - edge_6_7_2_6 + pos_6_2 >= 0
 arc_6_7_2_6_pos_must_match_v: - edge_6_7_2_6 + pos_7_6 >= 0
 arc_6_8_2_5_pos_must_match_u: - edge_6_8_2_5 + pos_6_2 >= 0
 arc_6_8_2_5_pos_must_match_v: - edge_6_8_2_5 + pos_8_5 >= 0
 arc_7_1_6_4_pos_must_match_u: - edge_7_1_6_4 + pos_7_6 >= 0
 arc_7_1_6_4_pos_must_match_v: - edge_7_1_6_4 + pos_1_4 >= 0
 arc_7_2_6_9_pos_must_match_u: - edge_7_2_6_9 + pos_7_6 >= 0
 arc_7_2_6_9_pos_must_match_v: - edge_7_2_6_9 + pos_2_9 >= 0
 arc_7_3_6_10_pos_must_match_u: - edge_7_3_6_10 + pos_7_6 >= 0
 arc_7_3_6_10_pos_must_match_v: - edge_7_3_6_10 + pos_3_10 >= 0
 arc_7_4_6_3_pos_must_match_u: - edge_7_4_6_3 + pos_7_6 >= 0
 arc_7_4_6_3_pos_must_match_v: - edge_7_4_6_3 + pos_4_3 >= 0
 arc_7_5_6_3_pos_must_match_u: - edge_7_5_6_3 + pos_7_6 >= 0
 arc_7_5_6_3_pos_must_match_v: - edge_7_5_6_3 + pos_5_3 >= 0
 arc_7_5_6_7_pos_must_match_u: - edge_7_5_6_7 + pos_7_6 >= 0
 arc_7_5_6_7_pos_must_match_v: - edge_7_5_6_7 + pos_5_7 >= 0
 arc_7_6_6_2_pos_must_match_u: - edge_7_6_6_2 + pos_7_6 >= 0
 arc_7_6_6_2_pos_must_match_v: - edge_7_6_6_2 + pos_6_2 >= 0
 arc_7_8_6_5_pos_must_match_u: - edge_7_8_6_5 + pos_7_6 >= 0
 arc_7_8_6_5_pos_must_match_v: - edge_7_8_6_5 + pos_8_5 >= 0
 arc_8_6_5_2_pos_must_match_u: - edge_8_6_5_2 + pos_8_5 >= 0
 arc_8_6_5_2_pos_must_match_v: - edge_8_6_5_2 + pos_6_2 >= 0
Bounds
Binaries
 edge_0_1_0_4 edge_0_2_0_9 edge_0_3_0_10 edge_0_4_0_3 edge_0_5_0_3
 edge_0_6_0_2 edge_0_7_0_6 edge_1_2_4_9 edge_1_3_4_10 edge_1_4_4_3
 edge_1_6_4_2 edge_1_7_4_6 edge_1_8_4_5 edge_2_1_9_4 edge_2_3_9_10
 edge_2_4_9_3 edge_2_5_9_3 edge_2_6_9_2 edge_2_7_9_6 edge_2_8_9_5
 edge_3_1_10_4 edge_3_2_10_9 edge_3_4_10_3 edge_3_5_10_3 edge_3_6_10_2
 edge_3_7_10_6 edge_3_8_10_5 edge_4_1_3_4 edge_4_2_3_9 edge_4_3_3_10
 edge_4_5_3_3 edge_4_5_3_7 edge_4_6_3_2 edge_4_7_3_6 edge_4_8_3_5
 edge_5_1_3_4 edge_5_1_7_4 edge_5_2_3_9 edge_5_2_7_9 edge_5_3_3_10
 edge_5_4_3_3 edge_5_6_3_2 edge_5_7_3_6 edge_5_7_7_6 edge_5_8_3_5
 edge_5_8_7_5 edge_6_1_2_4 edge_6_2_2_9 edge_6_3_2_10 edge_6_4_2_3
 edge_6_5_2_3 edge_6_5_2_7 edge_6_7_2_6 edge_6_8_2_5 edge_7_1_6_4
 edge_7_2_6_9 edge_7_3_6_10 edge_7_4_6_3 edge_7_5_6_3 edge_7_5_6_7
 edge_7_6_6_2 edge_7_8_6_5 edge_8_6_5_2 pos_1_4 pos_2_9 pos_3_10 pos_4_3
 pos_5_3 pos_5_7 pos_6_2 pos_7_6 pos_8_5
End
