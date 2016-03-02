Maximize
  - 3.69810784377 edge_0_2_0_0 - 4.3694181038805 edge_0_3_0_0
   - 3.969038528585001 edge_0_4_0_0 - 3.342849907529 edge_0_7_0_0
   - 4.2848677461447 edge_0_10_0_0 - 0.1844559905468113 edge_2_3_0_0
   - 0.0310676435306575 edge_2_4_0_0 - 0.2351526190656062 edge_2_7_0_0
   - 1.041152907181537 edge_2_10_0_0 + 1.316924561541497 edge_3_2_0_0
   + 0.4898297579516379 edge_3_4_0_0 + 0.2436115798077144 edge_3_7_0_0
   - 0.4025411836096762 edge_3_8_0_0 - 0.0138423711290304 edge_3_10_0_0
   + 2.421226605258131 edge_4_2_0_0 + 3.037535012662115 edge_4_3_0_0
   + 0.7569949369929521 edge_4_6_0_0 + 2.743679438608712 edge_4_7_0_0
   + 0.7051928977098915 edge_4_8_0_0 + 1.877572531579466 edge_4_10_0_0
   - 0.8344056205573347 edge_6_2_0_0 - 0.393189778038205 edge_6_3_0_0
   - 0.1949482641967084 edge_6_4_0_0 - 0.7162044591126655 edge_6_8_0_0
   - 0.1487098076169333 edge_6_10_0_0 + 0.1763489343305058 edge_7_2_0_0
   + 0.977916726971448 edge_7_3_0_0 + 0.8854782898874299 edge_7_4_0_0
   + 2.546538400703084 edge_7_6_0_0 + 0.3561777473496555 edge_7_8_0_0
   + 1.962343033266367 edge_7_10_0_0 - 0.6991634829363089 edge_10_2_0_0
   - 0.2152324282318639 edge_10_3_0_0 - 0.2272346987413666 edge_10_4_0_0
   + 0.2313443756513372 edge_10_7_0_0 + 2.907096034767739 edge_10_8_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_7_0_0 - flow_2_10_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_6_2_0_0 + flow_7_2_0_0 + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_6_3_0_0
   + edge_7_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_10_0_0
   + flow_4_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_6_4_0_0
   + edge_7_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_6_0_0 - flow_4_7_0_0
   - flow_4_8_0_0 - flow_4_10_0_0 + flow_6_4_0_0 + flow_7_4_0_0
   + flow_10_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_4_6_0_0 + edge_7_6_0_0 = 1
 consume_one_flow_unit_6: flow_4_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_8_0_0 - flow_6_10_0_0 + flow_7_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_10_0_0 + flow_10_7_0_0 = 1
 one_parent_8: edge_3_8_0_0 + edge_4_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0
   + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_3_8_0_0 + flow_4_8_0_0 + flow_6_8_0_0
   + flow_7_8_0_0 + flow_10_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_4_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   + flow_4_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 - flow_10_2_0_0
   - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_7_0_0 - flow_10_8_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_7_0_0
   + flow_0_10_0_0 = 10
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_7_0_0 edge_0_10_0_0
 edge_2_3_0_0 edge_2_4_0_0 edge_2_7_0_0 edge_2_10_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_10_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_10_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_8_0_0 edge_6_10_0_0
 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_6_0_0 edge_7_8_0_0
 edge_7_10_0_0 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_7_0_0
 edge_10_8_0_0
End
