Maximize
  - 1.1930859517098 edge_0_2_0_0 - 2.1193284286895 edge_0_3_0_0
   - 1.3425252969632 edge_0_5_0_0 - 2.1407472853579 edge_0_6_0_0
   - 2.1753976095496 edge_0_8_0_0 - 3.109358969664 edge_0_9_0_0
   + 2.590421912071792 edge_2_3_0_0 + 2.482326797354414 edge_2_5_0_0
   + 2.23548917469161 edge_2_6_0_0 + 1.899584878407438 edge_2_8_0_0
   + 1.710589267109084 edge_2_9_0_0 - 0.1543210164826591 edge_3_2_0_0
   + 0.0792916788838334 edge_3_5_0_0 - 0.0203602050524797 edge_3_6_0_0
   - 0.1209814858813733 edge_3_8_0_0 - 0.0648290280188315 edge_3_9_0_0
   + 1.173157821376882 edge_5_2_0_0 + 0.9926762186289525 edge_5_3_0_0
   + 2.099663748537271 edge_5_6_0_0 + 1.581052357999702 edge_5_8_0_0
   + 1.177488035822732 edge_5_9_0_0 - 0.3957721857886771 edge_6_2_0_0
   - 0.2733430526978582 edge_6_3_0_0 - 0.1612620203498644 edge_6_5_0_0
   + 0.1575228358796391 edge_6_8_0_0 - 0.0489821056288567 edge_6_9_0_0
   + 0.353095904688244 edge_8_2_0_0 + 0.3484838667278798 edge_8_3_0_0
   + 1.175988116455404 edge_8_5_0_0 + 0.9987886316558457 edge_8_6_0_0
   + 1.36007289775129 edge_8_9_0_0 - 1.078649040008603 edge_9_2_0_0
   - 0.8616611777289622 edge_9_3_0_0 - 0.8561058309494755 edge_9_5_0_0
   - 0.6369374286375695 edge_9_6_0_0 - 0.4758535086966492 edge_9_8_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0
   + edge_8_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_5_0_0
   - flow_2_6_0_0 - flow_2_8_0_0 - flow_2_9_0_0 + flow_3_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_8_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_5_3_0_0 + edge_6_3_0_0
   + edge_8_3_0_0 + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_8_0_0 - flow_3_9_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_8_3_0_0 + flow_9_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0
   + edge_8_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_6_0_0 - flow_5_8_0_0
   - flow_5_9_0_0 + flow_6_5_0_0 + flow_8_5_0_0 + flow_9_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_5_6_0_0
   + edge_8_6_0_0 + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_5_0_0
   - flow_6_8_0_0 - flow_6_9_0_0 + flow_8_6_0_0 + flow_9_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 - flow_8_2_0_0 - flow_8_3_0_0
   - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_9_0_0 + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_5_9_0_0
   + edge_6_9_0_0 + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_5_9_0_0 + flow_6_9_0_0 + flow_8_9_0_0 - flow_9_2_0_0
   - flow_9_3_0_0 - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_8_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0 + flow_0_6_0_0
   + flow_0_8_0_0 + flow_0_9_0_0 = 9
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_6_0_0 edge_0_8_0_0
 edge_0_9_0_0 edge_2_3_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_8_0_0
 edge_2_9_0_0 edge_3_2_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_8_0_0
 edge_3_9_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_6_0_0 edge_5_8_0_0
 edge_5_9_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_5_0_0 edge_6_8_0_0
 edge_6_9_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_5_0_0 edge_8_6_0_0
 edge_8_9_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_5_0_0 edge_9_6_0_0
 edge_9_8_0_0
End
