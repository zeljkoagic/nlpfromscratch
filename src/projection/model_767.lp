Maximize
  - 3.9293965999184 edge_0_1_0_0 - 2.842419998625 edge_0_2_0_0
   - 1.2279769518264 edge_0_3_0_0 - 3.36442 edge_0_4_0_0
   - 3.8147935977297 edge_0_5_0_0 - 3.2811939548831 edge_0_7_0_0
   - 0.4443817549873633 edge_1_2_0_0 - 0.0247522674838372 edge_1_3_0_0
   - 0.81277230893872 edge_1_4_0_0 - 1.156617770710731 edge_1_5_0_0
   - 1.304396041275767 edge_1_7_0_0 - 0.4962551970908001 edge_2_1_0_0
   + 1.405003079769927 edge_2_3_0_0 + 0.97796406618561 edge_2_4_0_0
   + 0.0496613077141319 edge_2_5_0_0 - 0.6819211092208767 edge_2_6_0_0
   + 0.2794823737374311 edge_2_7_0_0 + 0.63905946903624 edge_2_8_0_0
   + 2.420015046724184 edge_3_1_0_0 + 2.721626872630195 edge_3_2_0_0
   + 2.827342761984 edge_3_4_0_0 + 1.240151009981031 edge_3_5_0_0
   + 0.9518995445868733 edge_3_6_0_0 + 2.225025481577185 edge_3_7_0_0
   + 4.57170307396978 edge_3_8_0_0 + 0.033586188161448 edge_4_1_0_0
   + 0.6062605017975 edge_4_2_0_0 + 0.5655935520768 edge_4_3_0_0
   + 1.027832959296 edge_4_5_0_0 + 0.6278090651227899 edge_4_6_0_0
   + 1.3529133842212 edge_4_7_0_0 + 1.8093588058224 edge_4_8_0_0
   - 0.6534704053639574 edge_5_1_0_0 - 0.335227281620021 edge_5_2_0_0
   - 0.2926352159180181 edge_5_3_0_0 + 0.11895902051904 edge_5_4_0_0
   + 0.8154652019416649 edge_5_6_0_0 + 0.7859432953324021 edge_5_7_0_0
   + 1.251647543958476 edge_5_8_0_0 - 0.6430363694228977 edge_6_1_0_0
   - 0.3491840619959801 edge_6_2_0_0 - 0.5026912190609339 edge_6_3_0_0
   - 0.60140220439424 edge_6_4_0_0 + 1.032026517089272 edge_6_5_0_0
   + 0.7306509472199566 edge_6_7_0_0 + 0.2357143018269381 edge_6_8_0_0
   - 0.4974976382557326 edge_7_1_0_0 + 0.4812867008894471 edge_7_2_0_0
   + 0.0588497290612269 edge_7_3_0_0 + 0.32703588446722 edge_7_4_0_0
   + 2.87849124710189 edge_7_5_0_0 + 3.804240009002726 edge_7_6_0_0
   + 1.458541905052207 edge_7_8_0_0 - 0.964217648565462 edge_8_7_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_7_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0
   + flow_7_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 - flow_2_8_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0
   + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_6_5_0_0 + edge_7_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   - flow_5_8_0_0 + flow_6_5_0_0 + flow_7_5_0_0 = 1
 one_parent_6: edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0
   + edge_7_6_0_0 = 1
 consume_one_flow_unit_6: flow_2_6_0_0 + flow_3_6_0_0 + flow_4_6_0_0
   + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0
   + flow_7_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 = 1
 consume_one_flow_unit_8: flow_2_8_0_0 + flow_3_8_0_0 + flow_4_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_7_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_7_0_0 = 8
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_7_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0 edge_1_5_0_0
 edge_1_7_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0
 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0
 edge_4_7_0_0 edge_4_8_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0
 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0
 edge_6_8_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_8_7_0_0
End
