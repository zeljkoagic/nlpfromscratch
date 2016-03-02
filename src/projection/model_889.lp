Maximize
  - 2.680808304746 edge_0_1_0_0 - 4.4422322107002 edge_0_2_0_0
   - 1.598279708065 edge_0_4_0_0 - 2.451994471805 edge_0_8_0_0
   - 3.0357478856904 edge_0_9_0_0 + 0.3500627356444549 edge_1_2_0_0
   - 0.1233768172939552 edge_1_4_0_0 - 0.6122106839625215 edge_1_8_0_0
   + 0.2750433234801171 edge_1_9_0_0 + 0.0347016664382153 edge_2_1_0_0
   - 0.8670344225905122 edge_2_4_0_0 - 0.5617685557632414 edge_2_8_0_0
   + 0.114697183513845 edge_2_9_0_0 + 1.042433579300735 edge_4_1_0_0
   + 0.7460165796136931 edge_4_2_0_0 + 3.336100459312813 edge_4_3_0_0
   + 0.7422477021636155 edge_4_5_0_0 + 1.775096021934803 edge_4_8_0_0
   + 1.637736661953329 edge_4_9_0_0 - 0.3953357366199757 edge_5_9_0_0
   + 0.2092494506934808 edge_8_1_0_0 + 0.0650077392698001 edge_8_2_0_0
   + 0.3246646018816396 edge_8_3_0_0 + 0.201107807270689 edge_8_4_0_0
   + 1.848535729803342 edge_8_5_0_0 + 0.2387383985324189 edge_8_9_0_0
   - 1.731603711796912 edge_9_1_0_0 - 1.285208917769959 edge_9_4_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_4_1_0_0 + edge_8_1_0_0
   + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_4_0_0
   - flow_1_8_0_0 - flow_1_9_0_0 + flow_2_1_0_0 + flow_4_1_0_0
   + flow_8_1_0_0 + flow_9_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_4_2_0_0 + edge_8_2_0_0
   = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_4_0_0 - flow_2_8_0_0 - flow_2_9_0_0 + flow_4_2_0_0
   + flow_8_2_0_0 = 1
 one_parent_3: edge_4_3_0_0 + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_4_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_8_4_0_0
   + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0
   - flow_4_8_0_0 - flow_4_9_0_0 + flow_8_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_4_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_4_5_0_0 - flow_5_9_0_0 + flow_8_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_4_8_0_0
   = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_4_8_0_0 - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_3_0_0
   - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_9_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_8_9_0_0 - flow_9_1_0_0
   - flow_9_4_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_4_0_0 + flow_0_8_0_0
   + flow_0_9_0_0 = 9
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_4_0_0 edge_0_8_0_0 edge_0_9_0_0
 edge_1_2_0_0 edge_1_4_0_0 edge_1_8_0_0 edge_1_9_0_0 edge_2_1_0_0
 edge_2_4_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_5_0_0 edge_4_8_0_0 edge_4_9_0_0 edge_5_9_0_0
 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0 edge_8_5_0_0
 edge_8_9_0_0 edge_9_1_0_0 edge_9_4_0_0
End
