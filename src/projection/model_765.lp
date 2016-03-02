Maximize
  - 3.6219394817284 edge_0_1_0_0 - 2.1810898691346 edge_0_2_0_0
   - 2.60842 edge_0_3_0_0 - 2.589 edge_0_4_0_0
   - 3.8790422461252 edge_0_6_0_0 - 3.4242722352432 edge_0_7_0_0
   - 2.97071553198 edge_0_8_0_0 - 3.2484043101314 edge_0_13_0_0
   - 4.008327595002 edge_0_14_0_0 - 0.1080784189963345 edge_1_2_0_0
   - 0.11501034794536 edge_1_3_0_0 - 0.73228225268156 edge_1_4_0_0
   - 0.5992344094030739 edge_1_6_0_0 - 0.9927048696129662 edge_1_7_0_0
   - 0.8736306032525922 edge_1_8_0_0 - 1.479712443705177 edge_1_13_0_0
   - 0.502847909996353 edge_1_14_0_0 + 0.0138240674968199 edge_2_1_0_0
   + 1.010399939376 edge_2_3_0_0 + 0.8494079490355201 edge_2_4_0_0
   + 0.7469893851366343 edge_2_6_0_0 - 0.1899201986659074 edge_2_7_0_0
   - 0.3033297211842156 edge_2_8_0_0 - 0.3767191578344192 edge_2_13_0_0
   + 0.5329676482411392 edge_2_14_0_0 - 0.099373945680688 edge_3_1_0_0
   + 0.60295396382276 edge_3_2_0_0 + 0.443918 edge_3_4_0_0
   - 0.043416801399664 edge_3_6_0_0 - 0.71978983073712 edge_3_7_0_0
   - 0.6200306704263 edge_3_8_0_0 - 0.6784677229850701 edge_3_13_0_0
   + 0.4581387251166 edge_3_14_0_0 + 0.8340503900369199 edge_4_1_0_0
   + 2.0571798765692 edge_4_2_0_0 + 3.2156 edge_4_3_0_0
   + 1.8446293448044 edge_4_6_0_0 + 0.9427712573857601 edge_4_7_0_0
   + 0.8141550608646 edge_4_8_0_0 + 1.0965647035669 edge_4_13_0_0
   + 2.244158653504 edge_4_14_0_0 - 1.007040905443932 edge_6_1_0_0
   - 0.919295312060198 edge_6_2_0_0 - 0.8878324615005999 edge_6_3_0_0
   - 0.6869881304748 edge_6_4_0_0 + 0.4384483977932541 edge_6_7_0_0
   + 0.2512613245485303 edge_6_8_0_0 - 0.5138500358986591 edge_6_13_0_0
   - 0.0450826949910308 edge_6_14_0_0 - 2.035381186575829 edge_7_1_0_0
   - 2.01066404234895 edge_7_2_0_0 - 1.707843415432 edge_7_4_0_0
   - 0.3618797861121733 edge_7_6_0_0 + 0.6897550988530287 edge_7_8_0_0
   + 0.6763714878931344 edge_7_13_0_0 - 0.110853410238634 edge_7_14_0_0
   - 0.7671965374077295 edge_8_1_0_0 - 0.7055816065410511 edge_8_2_0_0
   - 0.6490263503892 edge_8_3_0_0 - 0.3883309414215 edge_8_4_0_0
   - 0.0194527754367063 edge_8_6_0_0 + 1.054013671258367 edge_8_7_0_0
   + 1.653245728326064 edge_8_13_0_0 + 0.577605765050333 edge_8_14_0_0
   - 1.291883837088929 edge_13_1_0_0 - 0.9184555087308662 edge_13_2_0_0
   - 0.7889791892121101 edge_13_3_0_0 - 0.63674392451199 edge_13_4_0_0
   - 0.7846814352081222 edge_13_6_0_0 + 0.4553314895403631 edge_13_7_0_0
   + 0.4778897454823307 edge_13_8_0_0 + 0.5066002491471782 edge_13_14_0_0
   - 2.186932277066246 edge_14_1_0_0 - 2.356168444927885 edge_14_2_0_0
   - 2.65579840652 edge_14_4_0_0 - 1.763922643678579 edge_14_6_0_0
   - 1.769007185455852 edge_14_7_0_0 - 1.325834406130379 edge_14_8_0_0
   - 1.513341782513886 edge_14_13_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 + edge_13_1_0_0
   + edge_14_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_8_0_0
   - flow_1_13_0_0 - flow_1_14_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 + flow_8_1_0_0
   + flow_13_1_0_0 + flow_14_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 + edge_13_2_0_0
   + edge_14_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 - flow_2_13_0_0 - flow_2_14_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0
   + flow_13_2_0_0 + flow_14_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_6_3_0_0 + edge_8_3_0_0 + edge_13_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_13_0_0 - flow_3_14_0_0
   + flow_4_3_0_0 + flow_6_3_0_0 + flow_8_3_0_0 + flow_13_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0 + edge_13_4_0_0
   + edge_14_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_13_0_0
   - flow_4_14_0_0 + flow_6_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0
   + flow_13_4_0_0 + flow_14_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0 + edge_13_6_0_0
   + edge_14_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_7_0_0 - flow_6_8_0_0
   - flow_6_13_0_0 - flow_6_14_0_0 + flow_7_6_0_0 + flow_8_6_0_0
   + flow_13_6_0_0 + flow_14_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 + edge_13_7_0_0
   + edge_14_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_4_0_0 - flow_7_6_0_0 - flow_7_8_0_0
   - flow_7_13_0_0 - flow_7_14_0_0 + flow_8_7_0_0 + flow_13_7_0_0
   + flow_14_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 + edge_13_8_0_0
   + edge_14_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_4_0_0
   - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_13_0_0 - flow_8_14_0_0
   + flow_13_8_0_0 + flow_14_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: edge_0_13_0_0 + edge_1_13_0_0 + edge_2_13_0_0
   + edge_3_13_0_0 + edge_4_13_0_0 + edge_6_13_0_0 + edge_7_13_0_0
   + edge_8_13_0_0 + edge_14_13_0_0 = 1
 consume_one_flow_unit_13: flow_0_13_0_0 + flow_1_13_0_0 + flow_2_13_0_0
   + flow_3_13_0_0 + flow_4_13_0_0 + flow_6_13_0_0 + flow_7_13_0_0
   + flow_8_13_0_0 - flow_13_1_0_0 - flow_13_2_0_0 - flow_13_3_0_0
   - flow_13_4_0_0 - flow_13_6_0_0 - flow_13_7_0_0 - flow_13_8_0_0
   - flow_13_14_0_0 + flow_14_13_0_0 = 1
 one_parent_14: edge_0_14_0_0 + edge_1_14_0_0 + edge_2_14_0_0
   + edge_3_14_0_0 + edge_4_14_0_0 + edge_6_14_0_0 + edge_7_14_0_0
   + edge_8_14_0_0 + edge_13_14_0_0 = 1
 consume_one_flow_unit_14: flow_0_14_0_0 + flow_1_14_0_0 + flow_2_14_0_0
   + flow_3_14_0_0 + flow_4_14_0_0 + flow_6_14_0_0 + flow_7_14_0_0
   + flow_8_14_0_0 + flow_13_14_0_0 - flow_14_1_0_0 - flow_14_2_0_0
   - flow_14_4_0_0 - flow_14_6_0_0 - flow_14_7_0_0 - flow_14_8_0_0
   - flow_14_13_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_13_0_0
   + flow_0_14_0_0 = 14
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_13_0_0: flow_0_13_0_0 - 1000 edge_0_13_0_0 <= 0
 inactive_no_flow_0_14_0_0: flow_0_14_0_0 - 1000 edge_0_14_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_13_0_0: flow_1_13_0_0 - 1000 edge_1_13_0_0 <= 0
 inactive_no_flow_1_14_0_0: flow_1_14_0_0 - 1000 edge_1_14_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_2_14_0_0: flow_2_14_0_0 - 1000 edge_2_14_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_13_0_0: flow_3_13_0_0 - 1000 edge_3_13_0_0 <= 0
 inactive_no_flow_3_14_0_0: flow_3_14_0_0 - 1000 edge_3_14_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_13_0_0: flow_4_13_0_0 - 1000 edge_4_13_0_0 <= 0
 inactive_no_flow_4_14_0_0: flow_4_14_0_0 - 1000 edge_4_14_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_13_0_0: flow_6_13_0_0 - 1000 edge_6_13_0_0 <= 0
 inactive_no_flow_6_14_0_0: flow_6_14_0_0 - 1000 edge_6_14_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_13_0_0: flow_7_13_0_0 - 1000 edge_7_13_0_0 <= 0
 inactive_no_flow_7_14_0_0: flow_7_14_0_0 - 1000 edge_7_14_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_13_0_0: flow_8_13_0_0 - 1000 edge_8_13_0_0 <= 0
 inactive_no_flow_8_14_0_0: flow_8_14_0_0 - 1000 edge_8_14_0_0 <= 0
 inactive_no_flow_13_1_0_0: flow_13_1_0_0 - 1000 edge_13_1_0_0 <= 0
 inactive_no_flow_13_2_0_0: flow_13_2_0_0 - 1000 edge_13_2_0_0 <= 0
 inactive_no_flow_13_3_0_0: flow_13_3_0_0 - 1000 edge_13_3_0_0 <= 0
 inactive_no_flow_13_4_0_0: flow_13_4_0_0 - 1000 edge_13_4_0_0 <= 0
 inactive_no_flow_13_6_0_0: flow_13_6_0_0 - 1000 edge_13_6_0_0 <= 0
 inactive_no_flow_13_7_0_0: flow_13_7_0_0 - 1000 edge_13_7_0_0 <= 0
 inactive_no_flow_13_8_0_0: flow_13_8_0_0 - 1000 edge_13_8_0_0 <= 0
 inactive_no_flow_13_14_0_0: flow_13_14_0_0 - 1000 edge_13_14_0_0 <= 0
 inactive_no_flow_14_1_0_0: flow_14_1_0_0 - 1000 edge_14_1_0_0 <= 0
 inactive_no_flow_14_2_0_0: flow_14_2_0_0 - 1000 edge_14_2_0_0 <= 0
 inactive_no_flow_14_4_0_0: flow_14_4_0_0 - 1000 edge_14_4_0_0 <= 0
 inactive_no_flow_14_6_0_0: flow_14_6_0_0 - 1000 edge_14_6_0_0 <= 0
 inactive_no_flow_14_7_0_0: flow_14_7_0_0 - 1000 edge_14_7_0_0 <= 0
 inactive_no_flow_14_8_0_0: flow_14_8_0_0 - 1000 edge_14_8_0_0 <= 0
 inactive_no_flow_14_13_0_0: flow_14_13_0_0 - 1000 edge_14_13_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_8_0_0 edge_0_13_0_0 edge_0_14_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_4_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_8_0_0
 edge_1_13_0_0 edge_1_14_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_2_13_0_0 edge_2_14_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_6_0_0 edge_3_7_0_0
 edge_3_8_0_0 edge_3_13_0_0 edge_3_14_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_13_0_0
 edge_4_14_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0
 edge_6_7_0_0 edge_6_8_0_0 edge_6_13_0_0 edge_6_14_0_0 edge_7_1_0_0
 edge_7_2_0_0 edge_7_4_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_13_0_0
 edge_7_14_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_6_0_0 edge_8_7_0_0 edge_8_13_0_0 edge_8_14_0_0 edge_13_1_0_0
 edge_13_2_0_0 edge_13_3_0_0 edge_13_4_0_0 edge_13_6_0_0 edge_13_7_0_0
 edge_13_8_0_0 edge_13_14_0_0 edge_14_1_0_0 edge_14_2_0_0 edge_14_4_0_0
 edge_14_6_0_0 edge_14_7_0_0 edge_14_8_0_0 edge_14_13_0_0
End
