Maximize
  - 2.405934752928 edge_0_1_0_0 - 3.3701993703552 edge_0_2_0_0
   - 1.0791075471792 edge_0_4_0_0 - 0.7386190134624 edge_0_5_0_0
   - 4.4688228051837 edge_0_9_0_0 + 0.5508491031821114 edge_1_2_0_0
   + 0.6222379312605718 edge_1_4_0_0 + 0.3057679459071546 edge_1_5_0_0
   + 0.665562142605589 edge_1_9_0_0 + 0.9331859516525712 edge_2_1_0_0
   + 1.283957834067828 edge_2_4_0_0 + 0.3094426416924186 edge_2_5_0_0
   + 0.109794894354346 edge_2_9_0_0 + 2.689579357453876 edge_4_1_0_0
   + 3.304059317425062 edge_4_2_0_0 + 2.56679156652568 edge_4_5_0_0
   + 2.584859965894155 edge_4_9_0_0 + 2.06585313058423 edge_5_1_0_0
   + 2.682076694198138 edge_5_2_0_0 + 2.700442376784737 edge_5_4_0_0
   + 2.422077554985004 edge_5_9_0_0 - 1.467001751584447 edge_9_1_0_0
   - 1.073014529077681 edge_9_2_0_0 - 0.7021701826969965 edge_9_4_0_0
   - 0.8691672797094955 edge_9_5_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_4_1_0_0 + edge_5_1_0_0
   + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_4_0_0
   - flow_1_5_0_0 - flow_1_9_0_0 + flow_2_1_0_0 + flow_4_1_0_0
   + flow_5_1_0_0 + flow_9_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_9_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_5_4_0_0
   + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_5_0_0 - flow_4_9_0_0
   + flow_5_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_4_5_0_0
   + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_4_0_0
   - flow_5_9_0_0 + flow_9_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0
   - flow_9_4_0_0 - flow_9_5_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_9_0_0 = 9
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_9_0_0
 edge_1_2_0_0 edge_1_4_0_0 edge_1_5_0_0 edge_1_9_0_0 edge_2_1_0_0
 edge_2_4_0_0 edge_2_5_0_0 edge_2_9_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_5_0_0 edge_4_9_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_4_0_0
 edge_5_9_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_4_0_0 edge_9_5_0_0
End
